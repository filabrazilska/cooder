use std::ops::{Add,Mul,Neg,Not,Rem,Sub};

#[derive(Debug,Clone,Copy)]
pub struct Vec3f {
    pub x : f32,
    pub y : f32,
    pub z : f32,
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
        self.scale(1_f32 / (self%self).sqrt())
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
    pub fn new(x : f32, y : f32, z : f32) -> Vec3f {
        Vec3f { x, y, z, }
    }

    pub fn from_float(f : f32) -> Vec3f {
        Vec3f {
            x : f,
            y : f,
            z : f,
        }
    }

    pub fn scale(&self, f : f32) -> Vec3f {
        Vec3f {
            x : self.x * f,
            y : self.y * f,
            z : self.z * f,
        }
    }

    pub fn copy(from : &Vec3f) -> Vec3f {
        Vec3f {
            x : from.x,
            y : from.y,
            z : from.z,
        }
    }

    pub fn reset(&mut self, other : &Vec3f) {
        self.x = other.x;
        self.y = other.y;
        self.z = other.z;
    }
}
