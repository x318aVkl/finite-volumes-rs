
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Mul;
use std::ops::SubAssign;

use super::super::sparse_array::SparseArray;
use crate::core::mesh::Geometry;
use crate::linalg::Inverse;
use crate::linalg::Magnitude;

use super::DistributedMatrix;
use super::Preconditioner;


// Parallel incomplete cholesky preconditioner
pub struct IncompleteLowerUpper<T> {
    lu: DistributedMatrix<T>,    // lower triangular factors
}



impl<T> IncompleteLowerUpper<T> {

    pub fn factors(&self) -> &DistributedMatrix<T> {
        &self.lu
    }


    pub fn row_i_u_sparsity(i: usize, matrix: &DistributedMatrix<T>, level: usize) -> BTreeSet<usize> {
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


                if (j < i) && (lk < level) {
                    q.push(j);
                    length.insert(j, lk + 1);
                } else if j > i {
                    s.insert(j);
                }

            }

        }

        s
    }


    pub fn row_i_l_sparsity(i: usize, matrix: &DistributedMatrix<T>, level: usize) -> BTreeSet<usize> {
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

                //println!("{} {}", k, j);

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


                if (j > i) && (lk < level) {
                    q.push(j);
                    length.insert(j, lk + 1);
                } else if j < i {
                    s.insert(j);
                }

            }

        }

        s
    }



    pub fn row_i_l_sparsity_m(i: usize, matrix: &DistributedMatrix<T>, level: usize) -> BTreeSet<usize> {
        let mut s = BTreeSet::new();

        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(i);
        s.insert(i);

        let mut queue = vec![i];

        for _l in 0..(level + 1) {
            let k = match queue.pop() {
                Some(v) => v,
                None => break
            };

            for kk in matrix.sparsity().major_range(k) {
                let j = *kk;

                if !visited.contains(&j) {
                    queue.push(j);
                    visited.insert(j);
                    if j < i {s.insert(j);}
                }
            }
        }

        s
    }

    pub fn row_i_u_sparsity_m(i: usize, matrix: &DistributedMatrix<T>, level: usize) -> BTreeSet<usize> {
        let mut s = BTreeSet::new();

        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(i);
        s.insert(i);

        let mut queue = vec![i];

        for _l in 0..(level + 1) {
            let k = match queue.pop() {
                Some(v) => v,
                None => break
            };

            for kk in matrix.sparsity().major_range(k) {
                let j = *kk;

                if !visited.contains(&j) {
                    queue.push(j);
                    visited.insert(j);
                    if j > i {s.insert(j);}
                }
            }
        }

        s
    }


    pub fn build_and_compute_row_by_row(&mut self, matrix: &DistributedMatrix<T>, level: usize) where T: Copy + Default + Magnitude<Output = f64> + Inverse + Mul<T, Output=T> + SubAssign {

        let matrix = matrix.square_block();

        self.lu = DistributedMatrix::<T>::new();
        let n = matrix.nrows();

        for i in 0..n {
            let l_row_sp = Self::row_i_l_sparsity(i, &matrix, level);
            let u_row_sp = Self::row_i_u_sparsity(i, &matrix, level);
            let mut lu_sparsity: HashSet<usize> = HashSet::new();
            lu_sparsity.insert(i);

            let mut lu_row = SparseArray::new();


            for k in l_row_sp.iter() {
                //l_row.push(*k, matrix.get(i, *k));
                lu_row.push(*k, matrix.get(i, *k));
            }
            //u_row.push(i, matrix.get(i, i));
            lu_row.push(i, matrix.get(i, i));
            for k in u_row_sp.iter() {
                //u_row.push(*k, matrix.get(i, *k));
                lu_row.push(*k, matrix.get(i, *k));
            }

            for k in l_row_sp.iter() {
                let k = *k;
                if k >= i {continue;}

                //println!("{} {}", i, k);

                let lukk = self.lu[[k, k]];
                if lukk.magnitude() < 1e-16 {
                    panic!("error, lu[k, k] too small {}", lukk.magnitude());
                }
                let lukkinv = lukk.inverse();
                lu_row[k] = lukkinv * lu_row[k];


                let lik = lu_row[k];
                for (j, ukj) in self.lu.iter_row(k) {
                    if j <= k {continue;}
                    let term = lik * ukj;
                    if !lu_row.contains(j) {
                        continue;
                    }
                    lu_row[j] -= term; 
                    
                }

                
            }

            // sparsify row
            let mut retain = HashSet::new();
            retain.insert(i);
            lu_row.sparsify(lu_row[i].magnitude() * 1e-6, retain);

            self.lu.push_row(lu_row);
        }

    }


    pub fn from_matrix(matrix: &DistributedMatrix<T>, level: usize) -> IncompleteLowerUpper<T> where T: Default + Copy + Magnitude<Output = f64> + Inverse + Mul<T, Output = T> + SubAssign {
        // let mut p = IncompleteLowerUpperPreconditioner::build(matrix, level);

        // for _sweep in 0..5 {
        //     let res = p.fixed_point_iter(matrix);
        //     println!("{} {}", _sweep, res);
        //     if res < 1e-7 {
        //         break;
        //     }
        // }
        
        //p.compute();
        let mut p = IncompleteLowerUpper { lu: DistributedMatrix::new() };
        p.build_and_compute_row_by_row(matrix, level);

        p
    }
}



impl<T, Rhs> Preconditioner<Rhs> for IncompleteLowerUpper<T> 
where T: Copy + Mul<Rhs, Output = Rhs> + Inverse,
Rhs: SubAssign + Copy
{
    fn precondition<G: Geometry<DIM>, const DIM: usize>(&self, solution: &mut crate::linalg::DistributedVector<Rhs>, rhs: &crate::linalg::DistributedVector<Rhs>, _comm: &crate::core::Communicator<G, DIM>) {
        
        // solve  l * u * solution = rhs

        // first solve l * solution' = rhs
        for i in 0..self.lu.nrows() {
            let mut si = rhs[i];

            //for (j, aij) in self.lu.iter_row(i) {
            for k in self.lu.sparsity().major_start(i)..self.lu.sparsity().major_end(i) {
                let j = self.lu.sparsity().flat_index(k);
                if j < i {
                    let aij = self.lu.flat_index(k);
                    si -= aij * solution[j];
                }
            }
            solution[i] = si; // / self.lu[[i, i]];
            //println!("{} {} {} {:?}", solution[i], rhs[i], self.lu[[i, i]], self.lu.iter_row(i).collect::<Vec<_>>());

            //if i == 5 {panic!()}
        }

        // solve u * solution = solution'
        for i in (0..self.lu.nrows()).rev() {
            let mut si = solution[i];

            //for (j, aij) in self.lu.iter_row(i) {
            for k in self.lu.sparsity().major_start(i)..self.lu.sparsity().major_end(i) {
                let j = self.lu.sparsity().flat_index(k);
                if j > i {
                    let aij = self.lu.flat_index(k);
                    si -= aij * solution[j];
                }
            }
            solution[i] = self.lu[[i, i]].inverse() * si;
        }

    }
}




