



#[derive(Clone, Debug)]
pub struct SparseVector {
    indices: Vec<usize>,
    values: Vec<f64>,
    default_value: f64,
}

impl SparseVector {

    pub fn new() -> SparseVector {
        SparseVector { indices: vec![], values: vec![], default_value: 0.0, }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn insert(&mut self, i: usize, x: f64) -> Option<&f64> {

        let result = self.indices.binary_search_by(|probe| {
            probe.cmp(&i)
        });

        match result {
            Ok(id) => {Some(&self.values[id])},
            Err(id) => {
                self.indices.insert(id, i);
                self.values.insert(id, x);
                None
            }
        }

    }

    pub fn insert_or_add(&mut self, i: usize, x: f64) {

        let result = self.indices.binary_search_by(|probe| {
            probe.cmp(&i)
        });

        match result {
            Ok(id) => {self.values[id] += x;},
            Err(id) => {
                self.indices.insert(id, i);
                self.values.insert(id, x);
            }
        }

    }

    pub fn find(&self, i: usize) -> Option<&f64> {
        let result = self.indices.binary_search_by(|probe| {
            probe.cmp(&i)
        });

        match result {
            Ok(id) => {Some(&self.values[id])},
            Err(_) => None,
        }
    }


    pub fn get(&self, i: usize) -> &f64 {
        match self.find(i) {
            Some(v) => v,
            None => &self.default_value,
        }
    }

    pub fn get_mut(&mut self, i: usize) -> &mut f64 {
        let result = self.indices.binary_search_by(|probe| {
            probe.cmp(&i)
        });

        match result {
            Ok(id) => {&mut self.values[id]},
            Err(id) => {
                self.indices.insert(id, i);
                self.values.insert(id, self.default_value);
                &mut self.values[id]
            },
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&usize, &f64)> {
        self.indices.iter().zip(self.values.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&usize, &mut f64)> {
        self.indices.iter().zip(self.values.iter_mut())
    }


    pub fn dot(&self, rhs: &Self) -> f64 {
        let mut out = 0.0;

        for (i, ai) in self.iter() {
            let bi = rhs.get(*i);
            out += ai * bi;
        }

        out
    }

}


impl std::ops::Index<usize> for SparseVector {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}


impl std::ops::IndexMut<usize> for SparseVector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
    }
}


impl std::ops::Add for SparseVector {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        for (id, value) in rhs.iter() {
            self.insert_or_add(*id, *value);
        }
        self
    }
}


impl std::ops::Sub for SparseVector {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        for (id, value) in rhs.iter() {
            self.insert_or_add(*id, -(*value));
        }
        self
    }
}


impl std::ops::Mul<f64> for SparseVector {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self::Output {
        for (_id, value) in self.iter_mut() {
            *value *= rhs;
        }
        self
    }
}

impl std::ops::Div<f64> for SparseVector {
    type Output = Self;
    fn div(mut self, rhs: f64) -> Self::Output {
        for (_id, value) in self.iter_mut() {
            *value /= rhs;
        }
        self
    }
}



impl std::ops::MulAssign<f64> for SparseVector {
    fn mul_assign(&mut self, rhs: f64) {
        for (_id, value) in self.iter_mut() {
            *value *= rhs;
        }
    }
}

impl std::ops::DivAssign<f64> for SparseVector {
    fn div_assign(&mut self, rhs: f64) {
        for (_id, value) in self.iter_mut() {
            *value /= rhs;
        }
    }
}


impl std::ops::AddAssign for SparseVector {
    fn add_assign(&mut self, rhs: Self) {
        for (id, value) in rhs.iter() {
            self.insert_or_add(*id, *value);
        }
    }
}


impl std::ops::SubAssign for SparseVector {
    fn sub_assign(&mut self, rhs: Self) {
        for (id, value) in rhs.iter() {
            self.insert_or_add(*id, -(*value));
        }
    }
}




pub struct SparseVectorView<'a> {
    indices: &'a [usize],
    values: &'a [f64],
    default_value: f64,
}



pub struct SparseVectorViewMut<'a> {
    indices: &'a [usize],
    values: &'a mut [f64],
    default_value: f64,
}




impl<'a> SparseVectorView<'a> {

    pub fn len(&self) -> usize {
        self.indices.len()
    }


    pub fn get(&self, i: usize) -> &f64 {

        let result = self.indices.binary_search(&i);

        match result {
            Ok(id) => &self.values[id],
            Err(_) => &self.default_value,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&usize, &f64)> {
        self.indices.iter().zip(self.values.iter())
    }

}



impl<'a> SparseVectorViewMut<'a> {

    pub fn len(&self) -> usize {
        self.indices.len()
    }


    pub fn get(&self, i: usize) -> &f64 {

        let result = self.indices.binary_search(&i);

        match result {
            Ok(id) => &self.values[id],
            Err(_) => &self.default_value,
        }
    }

    pub fn get_mut(&mut self, i: usize) -> &mut f64 {

        let result = self.indices.binary_search(&i);

        match result {
            Ok(id) => &mut self.values[id],
            Err(_) => panic!("index {} not found in sparse vector view", i),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&usize, &f64)> {
        self.indices.iter().zip(self.values.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&usize, &mut f64)> {
        self.indices.iter().zip(self.values.iter_mut())
    }

}


impl<'a> std::ops::Index<usize> for SparseVectorView<'a> {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}











