
use std::collections::HashMap;

use crate::linalg::Magnitude;

/// Array with sparse index-value pairs
#[derive(Clone, Debug)]
pub struct SparseArray<T> {
    data: HashMap<usize, T>,
}



impl<T> SparseArray<T> {
    pub fn new() -> SparseArray<T> {
        SparseArray { data: HashMap::new() }
    }

    pub fn push(&mut self, index: usize, value: T) {
        self.data.insert(index, value);
    }


    pub fn iter(&self) -> impl Iterator<Item = (&usize, &T)> {
        self.data.iter()
    }

    pub fn to_sorted(self) -> Vec<(usize, T)> {
        let mut m = self.data.into_iter().collect::<Vec<_>>();
        m.sort_by(|a, b| a.0.cmp(&b.0));
        m
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn contains(&self, i: usize) -> bool {
        self.data.contains_key(&i)
    }

}

impl<T> SparseArray<T> where T: Magnitude<Output = f64> + Copy {
    pub fn sparsify(&mut self, tol: f64) {
        self.data.retain(|_, v| v.magnitude() > tol);
    }
}


impl<T> std::ops::Index<usize> for SparseArray<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        match self.data.get(&index) {
            Some(v) => v,
            None => panic!("index {} not found in sparse array", index)
        }
    }
}

impl<T> std::ops::IndexMut<usize> for SparseArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self.data.get_mut(&index) {
            Some(v) => v,
            None => panic!("index {} not found in sparse array", index)
        }
    }
}



impl<T: Clone, const N: usize> Into<SparseArray<T>> for [(usize, T); N] {
    fn into(self) -> SparseArray<T> {
        let mut sparr = SparseArray::new();
        for i in 0..self.len() {
            sparr.push(self[i].0, self[i].1.clone());
        }
        sparr
    }
}

