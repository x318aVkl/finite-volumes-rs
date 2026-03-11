use crate::{core::{Communicator, mesh::Geometry}, linalg::Inverse};

use super::{DistributedMatrix, DistributedVector};


pub mod incomplete_cholesky;
pub mod incomplete_lower_upper;


pub use incomplete_cholesky::IncompleteCholesky;
pub use incomplete_lower_upper::IncompleteLowerUpper;

pub trait Preconditioner<T> {

    fn precondition<G: Geometry<DIM>, const DIM: usize>(&self, solution: &mut DistributedVector<T>, rhs: &DistributedVector<T>, comm: &Communicator<G, DIM>);

}




pub struct IdentityPreconditioner {

}


impl<T> Preconditioner<T> for IdentityPreconditioner where T: Copy {
    // applies the identity matrix
    fn precondition<G: Geometry<DIM>, const DIM: usize>(&self, solution: &mut DistributedVector<T>, rhs: &DistributedVector<T>, _comm: &Communicator<G, DIM>) where T: Clone {
        for i in 0..rhs.len() {
            solution[i] = rhs[i];
        }
    }
}



pub struct DiagonalPreconditioner<T> {
    diagonal: Vec<T>,
}


impl<T> DiagonalPreconditioner<T> where T: Inverse + Clone + Copy + Default {
    fn compute(&mut self, matrix: &DistributedMatrix<T>) {
        self.diagonal.resize(matrix.nrows(), T::default());

        for i in 0..self.diagonal.len() {
            let aii = matrix[[i, i]];

            self.diagonal[i] = aii.inverse();
        }
    }
    pub fn from_matrix(matrix: &DistributedMatrix<T>) -> DiagonalPreconditioner<T> {
        let mut p = DiagonalPreconditioner{diagonal: vec![]};
        p.compute(matrix);
        p
    }
}


impl<Rhs, T> Preconditioner<Rhs> for DiagonalPreconditioner<T> where T: std::ops::Mul<Rhs, Output = Rhs> + Copy, Rhs: Copy {

    fn precondition<G: Geometry<DIM>, const DIM: usize>(&self, solution: &mut DistributedVector<Rhs>, rhs: &DistributedVector<Rhs>, _comm: &Communicator<G, DIM>) {
        for i in 0..self.diagonal.len() {
            solution[i] = self.diagonal[i] * rhs[i];
        }
    }
}


