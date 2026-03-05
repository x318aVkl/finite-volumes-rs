use mpi::traits::Equivalence;



/// Small vector used to represent velocity, f64 field gradients, points in 2D or 3D space...
#[derive(Clone, Copy)]
pub struct Vector<const N: usize> {
    data: [f64; N],
}


impl<const N: usize> std::fmt::Debug for Vector<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}


impl<const N: usize> Vector<N> {

    pub fn new() -> Vector<N> {
        Self::zero()
    }

    pub fn zero() -> Vector<N> {
        Vector {
            data: [0.0; N]
        }
    }

    pub fn one() -> Vector<N> {
        Vector {
            data: [1.0; N]
        }
    }

    pub fn len(&self) -> usize {
        N
    }

    pub fn dot(self, rhs: Self) -> f64 {
        let mut x: f64 = 0.0;
        for i in 0..N {
            x += self[i] * rhs[i];
        }
        x
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }


    pub fn format_spaces(&self) -> String {
        let mut out = String::new();
        for i in 0..N {
            out.push_str(format!("{}", self[i]).as_str());
            if i < (N - 1) {
                out.push(' ');
            }
        }
        out
    }

    pub fn x(&self) -> f64 {
        self.data[0]
    }

    pub fn y(&self) -> f64 {
        self.data[1]
    }

    pub fn z(&self) -> f64 {
        self.data[2]
    }

    pub fn average(&self) -> f64 {
        let mut out: f64 = 0.0;
        for i in 0..N {
            out += self.data[i];
        }
        out / (N as f64)
    }


    pub fn min(mut self, other: Self) -> Self {
        for i in 0..N {
            self.data[i] = self.data[i].min(other.data[i]);
        }
        self
    }

    pub fn max(mut self, other: Self) -> Self {
        for i in 0..N {
            self.data[i] = self.data[i].max(other.data[i]);
        }
        self
    }


    pub fn set_zero(mut self, i: usize) -> Self {
        self.data[i] = 0.0;
        self
    }

    pub fn abs(mut self) -> Self {
        for i in 0..N {
            self.data[i] = self.data[i].abs();
        }
        self
    }

    pub fn self_max(self) -> f64 {
        let mut max = self.data[0];
        for i in 1..N {
            max = max.max(self.data[i]);
        }
        max
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

}


impl<const N: usize> std::ops::Index<usize> for Vector<N> {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize> std::ops::IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}


impl<const N: usize> std::ops::AddAssign<Self> for Vector<N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..self.len() {
            self[i] += rhs[i];
        }
    }
}

impl<const N: usize> std::ops::SubAssign<Self> for Vector<N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..self.len() {
            self[i] -= rhs[i];
        }
    }
}


impl<const N: usize> std::ops::DivAssign<f64> for Vector<N> {
    fn div_assign(&mut self, rhs: f64) {
        for i in 0..self.len() {
            self[i] /= rhs;
        }
    }
}


impl<const N: usize> std::ops::MulAssign<f64> for Vector<N> {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..self.len() {
            self[i] *= rhs;
        }
    }
}


impl<const N: usize> std::ops::Add for Vector<N> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            self[i] += rhs[i];
        }
        self
    }
}


impl<const N: usize> std::ops::Sub for Vector<N> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..N {
            self[i] -= rhs[i];
        }
        self
    }
}

impl<const N: usize> std::ops::Mul<f64> for Vector<N> {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self::Output {
        for i in 0..N {
            self[i] *= rhs;
        }
        self
    }
}


impl<const N: usize> std::ops::Div<f64> for Vector<N> {
    type Output = Self;
    fn div(mut self, rhs: f64) -> Self::Output {
        for i in 0..N {
            self[i] /= rhs;
        }
        self
    }
}


impl<const N: usize> std::ops::Mul<Vector<N>> for f64 {
    type Output = Vector<N>;
    fn mul(self, mut rhs: Vector<N>) -> Self::Output {
        for i in 0..N {
            rhs[i] *= self;
        }
        rhs
    }
}



impl<const N: usize> std::ops::Mul<Vector<N>> for Vector<N> {
    type Output = f64;
    fn mul(self, rhs: Vector<N>) -> Self::Output {
        self.dot(rhs)
    }
}


impl<const N: usize> std::ops::Neg for Vector<N> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        for i in 0..N {
            self.data[i] = -self.data[i];
        }
        self
    }
}

/// Cross product in 2D or 3D space.
/// Only implemented for size 2 or 3 vectors
pub trait Cross {
    type Output;
    fn cross(self, other: Self) -> Self::Output;
}


impl Cross for Vector<2> {
    type Output = f64;
    fn cross(self, rhs: Self) -> Self::Output {
        self[0]*rhs[1] - self[1]*rhs[0]
    }
}

impl Cross for Vector<3> {
    type Output = Self;
    fn cross(self, rhs: Self) -> Self::Output {
        Self {data: [
            self[1]*rhs[2] - self[2]*rhs[1],
            self[2]*rhs[0] - self[0]*rhs[2],
            self[0]*rhs[1] - self[1]*rhs[0],
        ]}
    }
}

/// Returns the sized normal direction to the lower dimensional object defined by self and rhs
/// - In 2D, this is a line orthogonal to self and rhs. Assumes self and rhs are colinear
/// - In 3D, this is a line perpendicular to the plane defined as self and rhs
pub trait Normal {
    fn normal(self, rhs: Self) -> Self;
}


impl<const N: usize> Normal for Vector<N> {
    fn normal(self, rhs: Self) -> Self {
        if N == 1 {
            return Vector::one();
        } else if N == 2 {

            let mut n = Vector::new();

            n[0] = - self[1];
            n[1] = self[0];

            n

        } else if N == 3 {
            let mut n = Vector::zero();
            n[0] = self[1]*rhs[2] - self[2]*rhs[1];
            n[1] = self[2]*rhs[0] - self[0]*rhs[2];
            n[2] = self[0]*rhs[1] - self[1]*rhs[0];

            //n /= n.norm();
            //n *= self.norm() + rhs.norm();
            n *= 0.5;

            n
        } else {
            panic!("Vector dimension not valid for normal calculation");
        }
    }
}



impl<const N: usize> Into<Vector<N>> for [f64; N] {
    fn into(self) -> Vector<N> {
        Vector { data: self }
    }
}


impl<const N: usize> Into<Vector<N>> for &[f64] {
    fn into(self) -> Vector<N> {
        let mut out = Vector::new();

        for i in 0..N {
            out[i] = self[i];
        }

        out
    }
}

impl<const N: usize> Into<Vector<N>> for &Vec<f64> {
    fn into(self) -> Vector<N> {
        let mut out = Vector::new();

        for i in 0..N {
            out[i] = self[i];
        }

        out
    }
}



impl<const N: usize> Vector<N> {
    pub fn write_raw_str<T: std::io::Write>(&self, f: &mut T) -> std::io::Result<()> {
        write!(f, "{}", self[0])?;
        for i in 1..N {
            write!(f, " {}", self[i])?;
        }
        Ok(())
    }
}
impl<const N: usize> Vector<N> {
    pub fn from_raw_str(s: &str) -> Result<Self, std::num::ParseFloatError> {
        let mut out = Self::new();

        let ss = s.split(" ");

        for (i, s) in ss.enumerate() {
            if i >= N {break}
            out[i] = s.trim().parse()?;
        }

        Ok(out)
    }
}

impl<const N: usize> Default for Vector<N> {
    fn default() -> Self {
        Self::zero()
    }
}

unsafe impl<const N: usize> Equivalence for Vector<N> {
    type Out = mpi::datatype::UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        mpi::datatype::UserDatatype::contiguous(N as i32, &f64::equivalent_datatype())
    }
}
