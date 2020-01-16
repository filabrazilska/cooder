extern crate crossbeam_utils;
extern crate num_cpus;
use std::fs::File;
use std::io::BufWriter;
use std::ops::{Add,Mul,Neg,Not,Rem,Sub};
use std::path::PathBuf;
use std::vec::Vec;
use crossbeam_utils::thread;
use png;
use rand::prelude::*;
use structopt::StructOpt;

#[derive(Debug,Clone,Copy)]
struct Vec3f {
    x : f32,
    y : f32,
    z : f32,
}

impl Default for Vec3f {
    fn default() -> Vec3f {
        Vec3f {
            x : 0.0,
            y : 0.0,
            z : 0.0,
        }
    }
}

impl<'a> Add for &'a Vec3f {
    type Output = Vec3f;

    fn add(self, rhs : Self) -> Vec3f {
        Vec3f {
            x : self.x + rhs.x,
            y : self.y + rhs.y,
            z : self.z + rhs.z,
        }
    }
}

impl<'a> Add for &'a mut Vec3f {
    type Output = Vec3f;

    fn add(self, rhs : Self) -> Vec3f {
        Vec3f {
            x : self.x + rhs.x,
            y : self.y + rhs.y,
            z : self.z + rhs.z,
        }
    }
}

impl<'a> Sub for &'a Vec3f {
    type Output = Vec3f;

    fn sub(self, rhs : Self) -> Vec3f {
        Vec3f::new( self.x - rhs.x, self.y - rhs.y, self.z - rhs.z )
    }
}

impl<'a> Mul for &'a Vec3f {
    type Output = Vec3f;

    fn mul(self, rhs : Self) -> Vec3f {
        Vec3f {
            x : self.x * rhs.x,
            y : self.y * rhs.y,
            z : self.z * rhs.z,
        }
    }
}

// Normalization
impl<'a> Not for &'a Vec3f {
    type Output = Vec3f;

    fn not(self) -> Vec3f {
        self.scale(1 as f32 / (self%self).sqrt())
    }
}

// Dot product
impl<'a> Rem for &'a Vec3f {
    type Output = f32;

    fn rem(self, rhs : Self) -> f32 {
        self.x*rhs.x + self.y*rhs.y + self.z*rhs.z
    }
}

impl<'a> Neg for &'a Vec3f {
    type Output = Vec3f;

    fn neg(self) -> Vec3f {
        Vec3f::new(-self.x, -self.y, -self.z)
    }
}

impl Vec3f {
    fn new(x : f32, y : f32, z : f32) -> Vec3f {
        Vec3f {
            x : x,
            y : y,
            z : z,
        }
    }

    fn from_float(f : f32) -> Vec3f {
        Vec3f {
            x : f,
            y : f,
            z : f,
        }
    }

    fn scale(&self, f : f32) -> Vec3f {
        Vec3f {
            x : self.x * f,
            y : self.y * f,
            z : self.z * f,
        }
    }

    fn copy(from : &Vec3f) -> Vec3f {
        Vec3f {
            x : from.x,
            y : from.y,
            z : from.z,
        }
    }

    fn reset(&mut self, other : &Vec3f) -> () {
        self.x = other.x;
        self.y = other.y;
        self.z = other.z;
    }
}

#[derive(Debug,StructOpt)]
#[structopt(name="cooder", about="A toy ray tracer")]
struct Opt {
    // Output file
    #[structopt(parse(from_os_str), default_value="dump.png", short, long)]
    output: PathBuf,

    #[structopt(default_value="320", short, long)]
    width: usize,

    #[structopt(default_value="200", short, long)]
    height: usize,

    #[structopt(default_value="16", short, long)]
    samples: u16,
}

fn main() {
    let opt = Opt::from_args();
    let w = opt.width;
    let h = opt.height;
    let mut wf = w as f32;
    let hf = h as f32/2.;
    let samples = opt.samples;
    let position = Vec3f::new( -22.0, 5.0, 25.0 );
    let goal = !&(&Vec3f::new(-3.0, 4.0, 0.0) - &position);
    let left : Vec3f = (!&Vec3f::new(goal.z, 0.0, -goal.x)).scale(1./wf);
    wf = wf / 2.;

    let output_file = File::create(opt.output).unwrap();
    let ref mut output_buffer = BufWriter::new(output_file);

    let mut framebuffer : Vec<Vec3f> = Vec::new();
    for _ in 0..w*h {
        framebuffer.push(Vec3f::from_float(0.));
    }

    //Cross product to get up from goal x left
    let up = Vec3f::new(
        goal.y*left.z - goal.z*left.y,
        goal.z*left.x - goal.x*left.z,
        goal.x*left.y - goal.y*left.x,
    );

    let cpus = num_cpus::get();

    let mut chunk_len = framebuffer.len() / cpus;
    if chunk_len * cpus != framebuffer.len() {
        chunk_len += 1;
    }
    let fb_chunks = framebuffer.chunks_mut(chunk_len);

    thread::scope(|s| {
        let mut i = 0;
        for c in fb_chunks {
            s.spawn(move |_| {
                let mut rng = rand::thread_rng();

                // find starting y, x
                let mut y = (chunk_len*i)/w;
                let mut x = (chunk_len*i)%w;
                for pix in c.iter_mut() {

                    let mut color = Vec3f::from_float(0.0);
                    let xf = x as f32;
                    let yf = y as f32;
                    for _ in 0..samples {
                        color = &color +
                                &trace(&position, &!&(
                                    &goal +
                                    &( &(left.scale(xf-wf+random_val(&mut rng))) +
                                       &(up.scale(yf-hf+random_val(&mut rng))))
                                    ),
                                    &mut rng
                                );
                    }

                    //Reinhard tone-mapping
                    color = &color.scale(1.0 / samples as f32) + &(Vec3f::from_float(14.0/241.0));
                    let o = &color + &(Vec3f::from_float(1.0));
                    color = Vec3f::new(color.x / o.x, color.y / o.y, color.z / o.z).scale(255.);
                    *pix = color; // framebuffer[(x+y*(w)) as usize] = color;

                    if x == w-1 {
                        x = 0;
                        y += 1;
                    } else {
                        x += 1;
                    }
                }
            });
            i += 1;
        }
    }).unwrap();

    write_output_file(output_buffer, opt.width as u32, opt.height as u32, &framebuffer);
}

fn write_output_file(output_buffer : &mut BufWriter<std::fs::File>, width : u32, height : u32, framebuffer : &Vec<Vec3f>) {
    let mut encoder = png::Encoder::new(output_buffer, width, height);
    encoder.set_color(png::ColorType::RGB);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();

    let mut data : Vec<u8> = Vec::new();
    for v in framebuffer {
        data.push(v.x as u8);
        data.push(v.y as u8);
        data.push(v.z as u8);
    }
    data.reverse(); // correct image orientation
    writer.write_image_data(&data).unwrap(); // Save
}

// ================== CSG primitives ==================

/* not used
fn test_sphere(p : &Vec3f, c : &Vec3f, r : f32) -> f32 {
    let delta = c - p;
    let dist = (&delta%&delta).sqrt();
    r - dist
}

fn test_rect(p : &Vec3f, c1 : &Vec3f, c2 : &Vec3f) -> f32 {
    -box_test(p, c1, c2)
}

fn test_carved_rect(p : &Vec3f, c1 : &Vec3f, c2 : &Vec3f) -> f32 {
    box_test(p, c1, c2)
}
*/

// ================== Tracing ==================

#[derive(PartialEq, Eq)]
enum HitType {
    NONE,
    WALL,
    SUN,
}

fn trace(position : &Vec3f, direction : &Vec3f, rng : &mut ThreadRng) -> Vec3f {
    let mut sampled_position = Vec3f::copy(position);
    let mut current_position = Vec3f::copy(position);
    let mut current_direction = Vec3f::copy(direction);
    let mut normal = Vec3f::from_float(0.);
    let mut color = Vec3f::from_float(0.);
    let mut attenuation = 1.0 as f32;
    let light_direction = !(&Vec3f::new(0.6, 0.6, 1.0));

    for _ in 0..4 {
        match ray_march(&current_position, &current_direction, &mut sampled_position, &mut normal) {
            HitType::NONE => {
                break
            }
            HitType::WALL => {
                let incidence : f32 = &normal%&light_direction;
                let p : f32 = 6.283185 * random_val(rng);
                let c = random_val(rng);
                let s = (1. - c).sqrt();
                let g = match normal.z {
                    x if x < 0. => -1.,
                    _ => 1.,
                };
                let u = -1. / (g + normal.z);
                let v = normal.x * normal.y * u;
                current_direction =
                    &(&(Vec3f::new(v, g + normal.y*normal.y*u, -normal.y).scale(s*p.cos())) +
                    &(Vec3f::new(1. + g * normal.x * normal.x * u, g * v, -g * normal.x).scale(s*p.sin()))) +
                    &(normal.scale(c.sqrt()));
                current_position = &sampled_position + &(current_direction.scale(0.1));
                attenuation = attenuation * 0.2;
                let refl_position : Vec3f = &sampled_position + &(normal.scale(0.1)); // just to ensure we don't hit ourselves again
                if incidence > 0. && ray_march(&refl_position, &light_direction, &mut sampled_position, &mut normal) == HitType::SUN {
                    color = &color + &(Vec3f::new(500., 400., 100.).scale(attenuation*incidence));
                }
            }
            HitType::SUN => {
                color = &color + &Vec3f::new(50., 80., 100.).scale(attenuation);
                break;
            }
        }
    }
    color
}

fn ray_march(origin : &Vec3f, direction : &Vec3f, hit_position : &mut Vec3f, normal : &mut Vec3f) -> HitType {
    let mut hit_type = HitType::NONE;
    let mut hit_count = 0;
    let mut total_dist = 0. as f32;

    while total_dist < 100. {
        hit_position.reset(&(origin + &(direction.scale(total_dist))));
        let closest_dist = query_db(hit_position, &mut hit_type);
        total_dist = total_dist + closest_dist;
        hit_count = hit_count + 1;
        if closest_dist < 0.01 || hit_count > 99 {
            let mut throwaway_hc1 = HitType::NONE;
            let mut throwaway_hc2 = HitType::NONE;
            let mut throwaway_hc3 = HitType::NONE;
            let x_wiggle = Vec3f::new(hit_position.x + 0.01, hit_position.y, hit_position.z);
            let y_wiggle = Vec3f::new(hit_position.x, hit_position.y + 0.01, hit_position.z);
            let z_wiggle = Vec3f::new(hit_position.x, hit_position.y, hit_position.z + 0.01);
            *normal = !&Vec3f::new(
                query_db(&x_wiggle, &mut throwaway_hc1) - closest_dist,
                query_db(&y_wiggle, &mut throwaway_hc2) - closest_dist,
                query_db(&z_wiggle, &mut throwaway_hc3) - closest_dist,
            );
            return hit_type
        }
    }
    return HitType::NONE
}

fn query_db(position : &Vec3f, hit_type : &mut HitType) -> f32 {
    let mut f = Vec3f::copy(position); // flattened position (z=0)
    f.z = 0.;

    let mut dist = min(// min(A,B) = Union with Constructive solid geometry
        //-min carves an empty space
        -min(
            box_test(position, &Vec3f::new(-30., -0.5, -30.), &Vec3f::new(30., 18., 30.)), // lower room
            box_test(position, &Vec3f::new(-25.,  17., -25.), &Vec3f::new(25., 20., 25.))  // upper_room
        ),
        box_test( // ceiling planks spaced 8 units apart
            &Vec3f::new(position.x.abs().rem(8.), position.y, position.z),
            &Vec3f::new(1.5, 18.5, -25.),
            &Vec3f::new(6.5, 20. ,  25.)
        )
    );
    *hit_type = HitType::WALL;

    let sun = 19.9 - position.y; // everything above 19.9 is sun
    if sun < dist {
        dist = sun;
        *hit_type = HitType::SUN;
    }

    return dist
}

// ================== Utils ==================

fn random_val(rng : &mut ThreadRng) -> f32 {
    rng.gen()
}

fn min(a : f32, b : f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

// Rectangle CSG equation. Returns minimum signed distance from
// space carved by lowerLeft vertex and opposite rectangle
// vertex upperRight.
fn box_test(position : &Vec3f, lower_left : &Vec3f, upper_right : &Vec3f) -> f32 {
    let ll = &(position - lower_left);
    let ur = &(upper_right - position);
    let minx = ll.x.min(ur.x);
    let miny = ll.y.min(ur.y);
    let minz = ll.z.min(ur.z);
    let minxy = minx.min(miny);
    return -(minxy.min(minz))
}
