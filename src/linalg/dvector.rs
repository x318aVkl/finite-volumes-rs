use std::ops::{AddAssign, MulAssign, SubAssign};

use crate::{core::mesh::CellIndex, linalg::Inverse};




// Distributed dense vector
#[derive(Debug, Clone)]
pub struct DistributedVector<T> {
    data: Vec<T>,
}




impl<T> DistributedVector<T> where T: Default + Clone {

    pub fn from_size(size: usize) -> DistributedVector<T> {
        DistributedVector { data: vec![T::default(); size] }
    }

    pub fn from_data(data: &[T]) -> DistributedVector<T> {
        DistributedVector { data: data.to_vec() }
    }

}

impl<T> DistributedVector<T> {

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

}

impl<T> DistributedVector<T> where T: Clone {
    pub fn set(&mut self, rhs: &[T]) {
        for i in 0..self.len() {
            self[i] = rhs[i].clone();
        }
    }

    pub fn set_smaller(&mut self, rhs: &[T]) {
        for i in 0..self.len().min(rhs.len()) {
            self[i] = rhs[i].clone();
        }
    }

}


impl<T> DistributedVector<T> {

    pub fn dot<V>(&self, rhs: &DistributedVector<V>) -> f64 where T: Copy + std::ops::Mul<V, Output = f64>, V: Default + Clone + Copy {
        assert_eq!(self.len(), rhs.len());

        let mut out = 0.0;
        for i in 0..self.len() {
            out += self[i] * rhs[i];
        }
        out
    }

    pub fn dot_smaller<V>(&self, rhs: &DistributedVector<V>) -> f64 where T: Copy + std::ops::Mul<V, Output = f64>, V: Default + Clone + Copy {
        let mut out = 0.0;
        for i in 0..self.len().min(rhs.len()) {
            out += self[i] * rhs[i];
        }
        out
    }


    pub fn inv(mut self) -> Self where T: Inverse + Copy {
        for i in 0..self.len() {
            self.data[i] = self.data[i].inverse();
        }
        self
    }

    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }


}

impl<T> DistributedVector<T> where T: Default + AddAssign + Copy {
    pub fn sum(&self) -> T {
        let mut out = T::default();
        for i in 0..self.len() {
            out += self[i];
        }
        out
    }
}


impl<T> std::ops::Index<usize> for DistributedVector<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}


impl<T> std::ops::IndexMut<usize> for DistributedVector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}



impl<T> std::ops::AddAssign<&DistributedVector<T>> for DistributedVector<T> where T: AddAssign + Copy {
    fn add_assign(&mut self, rhs: &DistributedVector<T>) {
        assert_eq!(self.len(), rhs.len());
        for i in 0..self.len() {
            self[i] += rhs[i];
        }
    }
}


impl<T> std::ops::SubAssign<&DistributedVector<T>> for DistributedVector<T> where T: SubAssign + Copy {
    fn sub_assign(&mut self, rhs: &DistributedVector<T>) {
        assert_eq!(self.len(), rhs.len());
        for i in 0..self.len() {
            self[i] -= rhs[i];
        }
    }
}

impl<T> std::ops::MulAssign<f64> for DistributedVector<T> where T: MulAssign<f64> {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..self.len() {
            self[i] *= rhs;
        }
    }
}


// impl std::ops::DivAssign<f64> for DistributedVector {
//     fn div_assign(&mut self, rhs: f64) {
//         for i in 0..self.len() {
//             self[i] /= rhs;
//         }
//     }
// }




impl<T> std::ops::Index<CellIndex> for DistributedVector<T> {
    type Output = T;
    fn index(&self, index: CellIndex) -> &Self::Output {
        &self[usize::from(index)]
    }
}

impl<T> std::ops::IndexMut<CellIndex> for DistributedVector<T> {
    fn index_mut(&mut self, index: CellIndex) -> &mut Self::Output {
        &mut self[usize::from(index)]
    }
}






