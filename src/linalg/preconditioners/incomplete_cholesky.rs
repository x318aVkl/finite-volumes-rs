
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::ops::Mul;
use std::ops::Sub;
use std::ops::SubAssign;

use crate::core::Sparsity;
use crate::core::mesh::Geometry;
use crate::linalg::ApproximateCmp;
use crate::linalg::Inverse;
use crate::linalg::SquareRoot;

use super::DistributedMatrix;
use super::Preconditioner;


// Parallel incomplete cholesky preconditioner
pub struct IncompleteCholesky<T> {
    l: DistributedMatrix<T>,    // lower triangular factors
    sp_t: Sparsity<usize>, // sparsity of the columns of l
}



impl<T> IncompleteCholesky<T> {

    pub fn factors(&self) -> &DistributedMatrix<T> {
        &self.l
    }

    pub fn row_i_sparsity(i: usize, matrix: &DistributedMatrix<T>, level: usize) -> BTreeSet<usize> {
        let mut s = BTreeSet::new();

        let mut q = vec![i];
        let mut visited: HashMap<usize, usize> = HashMap::new();
        visited.insert(i, i);

        let mut length: HashMap<usize, usize> = HashMap::new();
        length.insert(i, 0);

        loop {

            let k = match q.pop() {
                Some(v) => v,
                None => break
            };
            let lk = match length.get(&k) {
                Some(v) => *v,
                None => {
                    panic!("k  {} not found in length", k)
                }
            };

            for kj in matrix.sparsity().major_start(k)..matrix.sparsity().major_end(k) {
                let j = matrix.sparsity().flat_index(kj);

                if j == k {
                    continue;
                }

                match visited.get(&j) {
                    Some(vj) => {
                        if *vj == i {continue;}
                    }
                    None => {}
                }
                visited.insert(j, i);


                if lk < level {
                    q.push(j);
                    length.insert(j, lk + 1);
                    s.insert(j);
                }

            }

        }

        s
    }

    fn compute(&mut self) where T: Default + Copy + Mul<T, Output = T> + SubAssign + Inverse + SquareRoot {

        for k in 0..self.l.nrows() {


            let lkk = self.l[[k, k]];
            
            let lkk = lkk.square_root();

            let lkkinv = lkk.inverse();

            self.l[[k, k]] = lkk;


            for k_i in self.sp_t.major_start(k)..self.sp_t.major_end(k) {
                let i = self.sp_t.flat_index(k_i);
                if i <= k {continue;}

                match self.l.get_mut(i, k) {
                    Some(lik) => {
                        *lik = lkkinv * *lik;
                    },
                    None => {}
                }
            }

            for k_j in self.sp_t.major_start(k)..self.sp_t.major_end(k) {
                for k_i in self.sp_t.major_start(k)..self.sp_t.major_end(k) {
                    let i = self.sp_t.flat_index(k_i);
                    let j = self.sp_t.flat_index(k_j);

                    if i < j {continue;}
                    if j <= k {continue;}

                    let lik = self.l.get(i, k);
                    let ljk = self.l.get(j, k);

                    match self.l.get_mut(i, j) {
                        Some(lij) => {
                            *lij -= lik * ljk;
                        },
                        None => {}
                    }
                }
            }
            

        }

    }


    fn build(matrix: &DistributedMatrix<T>, level: usize) -> IncompleteCholesky<T> where T: Sub<T, Output = T> + ApproximateCmp + Copy + Default {
        
        // ignore the parallel entries
        let matrix = matrix.square_block();
        
        assert!(matrix.is_symmetric());

        let mut sp = Sparsity::new();
        let mut sp_t = Sparsity::new();
        let mut values = vec![];

        for i in 0..matrix.nrows() {

            // get this rows sparsity
            let mut row_sp: BTreeSet<usize> = Self::row_i_sparsity(i, &matrix, level);

            for (j, aij) in matrix.iter_row(i) {

                // ignore the parallel entries
                if j <= i {
                    sp.push_to_major(j);
                    values.push(aij);
                }
                if j >= i {
                    sp_t.push_to_major(j);
                }

                row_sp.remove(&j);
            }
            // add the extra row_sp entries as zeros
            for j in row_sp {
                if j <= i {
                    sp.push_to_major(j);
                    values.push(T::default());
                }
                if j >= i {
                    sp_t.push_to_major(j);
                }
            }

            sp.close_major();
            sp_t.close_major();
        }

        IncompleteCholesky { l: DistributedMatrix::from_sparsity_and_values(sp, values), sp_t }
    }


    pub fn from_matrix(matrix: &DistributedMatrix<T>, level: usize) -> IncompleteCholesky<T> where T: Sub<T, Output = T> + ApproximateCmp + Copy + Default + Mul<T, Output = T> + SubAssign + Inverse + SquareRoot {
        let mut p = IncompleteCholesky::build(matrix, level);
        
        p.compute();

        p
    }

}



impl<T, Rhs> Preconditioner<Rhs> for IncompleteCholesky<T> where T: Default + Copy + Mul<Rhs, Output = Rhs> + Inverse, Rhs: SubAssign + Copy {
    fn precondition<G: Geometry<DIM>, const DIM: usize>(&self, solution: &mut crate::linalg::DistributedVector<Rhs>, rhs: &crate::linalg::DistributedVector<Rhs>, _comm: &crate::core::Communicator<G, DIM>) {
        
        // solve  l*lT * solution = rhs

        // first solve l * solution' = rhs
        for i in 0..self.l.nrows() {
            let mut si = rhs[i];

            for (j, lij) in self.l.iter_row(i) {
                if i != j {
                    si -= lij * solution[j];
                }
            }
            solution[i] = self.l[[i, i]].inverse() * si;
        }

        // solve lT * solution = solution'
        for i in (0..self.l.nrows()).rev() {
            let mut si = solution[i];

            for j in self.sp_t.major_range(i) {
                if i != *j {
                    si -= self.l.get(*j, i) * solution[*j];
                }
            }
            solution[i] = self.l[[i, i]].inverse() * si;
        }

    }
}

