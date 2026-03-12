
use std::ops::{Add, AddAssign, SubAssign};

use crate::{core::{Sparsity, mesh::CellIndex}, linalg::ApproximateCmp};


use super::dvector::DistributedVector;
pub use super::sparse_array::SparseArray;


// Distributed sparse matrix
// in compressed sparse row format
#[derive(Debug, Clone)]
pub struct DistributedMatrix<T> {
    sparsity: Sparsity<usize>,
    values: Vec<T>,

    nrows: usize,
    ncols: usize,
}



impl<T> DistributedMatrix<T> 
where T: Default + Clone + Copy {

    pub fn new() -> Self {
        DistributedMatrix { sparsity: Sparsity::new(), values: vec![], nrows: 0, ncols: 0 }
    }

    pub fn from_usize_sparsity(sparsity: Sparsity<usize>) -> DistributedMatrix<T> {
        let nrows = sparsity.major_len();
        let ncols = sparsity.max_minor() + 1;
        let size = sparsity.minor_len();
        DistributedMatrix { sparsity: sparsity.sorted(), values: vec![T::default(); size], nrows, ncols }
    }

    pub fn from_cut_sparsity<I>(sparsity: &Sparsity<I>, nrows: usize) -> DistributedMatrix<T> where usize: From<I>, I: Add<Output = I> + From<usize> + Clone + Copy + PartialOrd + Ord {
        let mut out_sp: Sparsity<usize> = Sparsity::<usize>::new();
        for i in 0..nrows {
            for k in sparsity.major_range(i) {
                let ku = usize::from(*k);
                out_sp.push_to_major(ku);
            }
            out_sp.close_major();
        }
        let ncols = out_sp.max_minor() + 1;
        let size = out_sp.minor_len();
        DistributedMatrix { sparsity: out_sp.sorted(), values: vec![T::default(); size], nrows, ncols }
    }


    pub fn from_sparsity_and_values(sparsity: Sparsity<usize>, values: Vec<T>) -> DistributedMatrix<T> {
        let nrows = sparsity.major_len();
        let ncols = sparsity.max_minor() + 1;
        let (sparsity, values) = sparsity.sorted_with(values);
        DistributedMatrix { sparsity: sparsity, values, nrows, ncols }
    }
}

impl<T> DistributedMatrix<T> {

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn sparsity(&self) -> &Sparsity<usize> {
        &self.sparsity
    }


    pub fn is_symmetric(&self) -> bool where T: std::ops::Sub<T, Output=T> + ApproximateCmp + Copy {
        for i in 0..self.nrows() {
            for (j, aij) in self.iter_row(i) {
                if j < self.nrows() {
                    if !aij.cmp_approx(self[[j, i]]) {
                        //println!("{} {}", self[[i, j]], self[[j, i]]);
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn contains(&self, i: usize, j: usize) -> bool {
        if i >= self.sparsity.major_len() {
            return false;
        }
        match self.sparsity.major_range(i).binary_search(&j) {
            Ok(_k) => {
                true
            },
            Err(_) => {
                false
            }
        }
    }


    pub fn get(&self, i: usize, j: usize) -> T where T: Default + Copy {
        if i >= self.sparsity.major_len() {
            return T::default();
        }
        match self.sparsity.major_range(i).binary_search(&j) {
            Ok(k) => {
                self.values[self.sparsity.major_start(i) + k]
            },
            Err(_) => {
                T::default()
            }
        }
    }


    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if i >= self.sparsity.major_len() {
            return None;
        }

        match self.sparsity.major_range(i).binary_search(&j) {
            Ok(k) => {
                Some(&mut self.values[self.sparsity.major_start(i) + k])
            },
            Err(_) => {
                None
            }
        }
    }


    pub fn imul<V>(&self, result: &mut DistributedVector<V>, rhs: &[V]) where V: Default + AddAssign + Copy, T: std::ops::Mul<V, Output = V> + Copy {

        for i in 0..self.nrows {
            result[i] = V::default();
            for k in self.sparsity.major_start(i)..self.sparsity.major_end(i) {
                let j = self.sparsity.flat_index(k);
                result[i] += self.values[k] * rhs[j];
            }
        }

    }


    pub fn calc_h<V>(&self, result: &mut DistributedVector<V>, rhs: &[V]) where V: Default + AddAssign + SubAssign + Copy, T: std::ops::Mul<V, Output = V> + Copy {

        for i in 0..self.nrows {
            result[i] = V::default();
            for k in self.sparsity.major_start(i)..self.sparsity.major_end(i) {
                let j = self.sparsity.flat_index(k);
                if i == j {
                    result[i] += self.values[k] * rhs[j];
                } else {
                    result[i] -= self.values[k] * rhs[j];
                }
            }
        }

    }

    pub fn diag(&self) -> DistributedVector<T> where T: Default + Clone {
        let mut diag = DistributedVector::from_size(self.nrows);

        for i in 0..self.nrows {
            diag[i] = self[[i, i]].clone();
        }

        diag
    }


    pub fn iter_row<'a>(&'a self, row: usize) -> impl Iterator<Item = (usize, T)> + 'a where T: Copy {
        (self.sparsity.major_start(row)..self.sparsity.major_end(row)).map(|k| (self.sparsity.flat_index(k), self.values[k]))
    }



    pub fn flat_index(&self, k: usize) -> T where T: Copy {
        self.values[k]
    }

    pub fn flat_index_mut(&mut self, k: usize) -> &mut T {
        &mut self.values[k]
    }


    pub fn push_row(&mut self, row: SparseArray<T>) {
        let row = row.to_sorted();

        for (i, v) in row {
            self.sparsity.push_to_major(i);
            self.values.push(v);
        }
        self.sparsity.close_major();
        self.nrows += 1;
    }

    pub fn insert(&mut self, row: usize, col: usize, value: T) {
        // Very slow, insert value into matrix
        // should be used only for debug purposes
        let pos = match self.sparsity.insert(row, col) {
            Err(_) => panic!("Entry {} {} already in sparse matrix", row, col),
            Ok(p) => p,
        };
        self.values.insert(pos, value);
    }



    pub fn square_block(&self) -> DistributedMatrix<T> where T: Copy + Default {
        let mut sp = Sparsity::new();
        let mut vals = vec![];

        for i in 0..self.nrows {
            for (j, v) in self.iter_row(i) {
                if j < self.nrows {
                    sp.push_to_major(j);
                    vals.push(v);
                }
            }
            sp.close_major();
        }

        DistributedMatrix::from_sparsity_and_values(sp, vals)
    }


    pub fn set_row(&mut self, row: usize, val: T) where T: Clone {
        for k in self.sparsity.major_start(row)..self.sparsity.major_end(row) {
            //let col = self.sparsity.flat_index(k);
            self.values[k] = val.clone();
        }
    }


    pub fn dirichlet<V>(&mut self, rhs: &mut DistributedVector<V>, row: usize, val: V) where T: std::ops::Mul<V, Output = V> + Copy + Default, V: SubAssign + Copy {
        let diag = self[[row, row]];

        for k in self.sparsity.major_start(row)..self.sparsity.major_end(row) {
            let col = self.sparsity.flat_index(k);
            let aji = self[[col, row]];
            rhs[col] -= aji * val;
        }

        for k in self.sparsity.major_start(row)..self.sparsity.major_end(row) {
            let col = self.sparsity.flat_index(k);
            self.values[k] = T::default();
            self[[col, row]] = T::default();
        }

        self[[row, row]] = diag;
        rhs[row] = diag * val;
    }

}


impl<T> std::ops::Index<[usize; 2]> for DistributedMatrix<T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let i = index[0];
        let j = index[1];
        match self.sparsity.major_range(i).binary_search(&j) {
            Ok(k) => {
                &self.values[self.sparsity.major_start(i) + k]
            },
            Err(_) => {
                panic!("Sparse matrix index ({}, {}) not found", index[0], index[1])
            }
        }
    }
}


impl<T> std::ops::IndexMut<[usize; 2]> for DistributedMatrix<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        match self.get_mut(index[0], index[1]) {
            Some(v) => v,
            None => panic!("Sparse matrix index ({}, {}) not found", index[0], index[1])
        }
    }
}



impl<T> std::ops::Index<[CellIndex; 2]> for DistributedMatrix<T> {
    type Output = T;
    fn index(&self, index: [CellIndex; 2]) -> &Self::Output {
        &self[[usize::from(index[0]), usize::from(index[1])]]
    }
}

impl<T> std::ops::IndexMut<[CellIndex; 2]> for DistributedMatrix<T> {
    fn index_mut(&mut self, index: [CellIndex; 2]) -> &mut Self::Output {
        &mut self[[usize::from(index[0]), usize::from(index[1])]]
    }
}
