/*! Contains distributed memory linear algebra functionality

- [DistributedVector] and [DistributedMatrix] structs representing parallel dense vectors and sparse matrices
- [solvers], including [solvers::conjugate_gradient()]
- [preconditioners], including [preconditioners::IncompleteLowerUpperPreconditioner]

*/

pub mod sparse_vector;

pub mod dvector;
pub mod dmatrix;
pub mod solvers;
pub mod preconditioners;
pub mod sparse_array;

pub use dvector::DistributedVector;
pub use dmatrix::DistributedMatrix;

pub use preconditioners::Preconditioner;

use crate::{Matrix, Vector};



pub trait Inverse {
    fn inverse(self) -> Self;
}

pub trait SquareRoot {
    fn square_root(self) -> Self;
}

pub trait AbsoluteValue {
    fn absolute_value(self) -> Self;
}
pub trait ApproximateCmp {
    fn cmp_approx(self, rhs: Self) -> bool;
}


pub trait Magnitude {
    type Output;
    fn magnitude(self) -> Self::Output;
}


impl Inverse for f64 {
    fn inverse(self) -> Self {
        1.0 / self
    }
}

impl SquareRoot for f64 {
    fn square_root(self) -> Self {
        self.sqrt()
    }
}

impl AbsoluteValue for f64 {
    fn absolute_value(self) -> Self {
        self.abs()
    }
}

impl Magnitude for f64 {
    type Output = f64;
    fn magnitude(self) -> Self::Output {
        self.abs()
    }
}

impl ApproximateCmp for f64 {
    fn cmp_approx(self, rhs: Self) -> bool {
        (self - rhs).abs() < 1e-14
    }
}


impl<const N: usize> Magnitude for Vector<N> {
    type Output = f64;
    fn magnitude(self) -> Self::Output {
        let mut out = 0.0;
        for i in 0..N {
            out += self[i].powi(2);
        }
        out.sqrt()
    }
}



impl<const N: usize> Inverse for Matrix<N, N> {
    fn inverse(self) -> Self {
        self.inv().expect("matrix is invertible")
    }
}

impl<const M: usize, const N: usize> AbsoluteValue for Matrix<M, N> {
    fn absolute_value(mut self) -> Self {
        for i in 0..M {
            for j in 0..N {
                self[[i, j]] = self[[i, j]].abs();
            }
        }
        self
    }
}

impl<const M: usize, const N: usize> Magnitude for Matrix<M, N> {
    type Output = f64;
    fn magnitude(self) -> Self::Output {
        let mut out = 0.0;
        for i in 0..M {
            for j in 0..N {
                out += self[[i, j]].powi(2);
            }
        }
        out.sqrt()
    }
}

impl<const M: usize, const N: usize> ApproximateCmp for Matrix<M, N> {
    fn cmp_approx(self, rhs: Self) -> bool {
        (self - rhs).magnitude() < 1e-13
    }
}