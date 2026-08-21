use std::marker::PhantomData;

use crate::{Mesh, core::mesh::{Geometry, MeshGet}};
use crate::core::traits::FloatBuffered;



#[derive(Clone, Debug)]
pub struct EvaluatedData {
    value_buffer: Vec<f64>,
    value_starts: Vec<usize>,
}


/// Evaluator based on a generic geometry-wise data representation
/// evaluates custom expressions at geometry positions
pub struct Evaluator<'a, G: Geometry<DIM>, const DIM: usize> {
    data: EvaluatedData,
    evaluators: Vec<Box<dyn Fn(G::IndexType, &'a Mesh<DIM>, &mut [f64]) + 'a>>,
    gphantom: PhantomData<G>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvaluatorIndex<T>(usize, PhantomData<T>);

impl<T> From<EvaluatorIndex<T>> for usize {
    fn from(value: EvaluatorIndex<T>) -> Self {
        value.0
    }
}
impl<T> From<usize> for EvaluatorIndex<T> {
    fn from(value: usize) -> Self {
        Self(value, PhantomData)
    }
}




impl EvaluatedData {
    pub fn new() -> Self {
        Self { value_buffer: vec![], value_starts: vec![0] }
    }
    pub fn len(&self) -> usize {
        self.value_starts.len() - 1
    }
    fn append_data<T: FloatBuffered>(&mut self) {
        // add the buffer
        let value_size = T::f64_buffer_size();
        for _ in 0..value_size {
            self.value_buffer.push(0.0);
        }
        self.value_starts.push(self.value_buffer.len());
    }
    pub fn get<T: FloatBuffered>(&self, index: EvaluatorIndex<T>) -> T {
        let i = usize::from(index);
        assert!(i < self.len());
        T::build_from_f64_buffer(&self.value_buffer[self.value_starts[i]..self.value_starts[i+1]])
    }
    fn buffer(&mut self, index: usize) -> &mut [f64] {
        &mut self.value_buffer[self.value_starts[index]..self.value_starts[index + 1]]
    }
}



impl<'a, G: Geometry<DIM>, const DIM: usize> Evaluator<'a, G, DIM> where Mesh<DIM>: MeshGet<'a, G::IndexType, Output = G::ElementType<'a>>, G::IndexType: Clone {

    pub fn new() -> Self {
        Self { data: EvaluatedData::new(), evaluators: vec![], gphantom: PhantomData }
    }

    pub fn register_fn<T>(&mut self, f: impl Fn(G::ElementType<'a>) -> T + 'a) -> EvaluatorIndex<T> where T: FloatBuffered {

        // add the function evaluator
        self.evaluators.push(Box::new(move |element, mesh, buffer| {
            let val = f(mesh.get(element));
            val.put_in_f64_buffer(buffer);
        }));

        // add to the data
        self.data.append_data::<T>();

        EvaluatorIndex(self.evaluators.len() - 1, PhantomData)
    }


    pub fn update(&mut self, index: G::IndexType, mesh: &'a Mesh<DIM>) {
        // recompute the nodal values at that node
        for i in 0..self.evaluators.len() {
            self.evaluators[i](
                index.clone(),
                mesh,
                self.data.buffer(i),
            );
        }
    }

    pub fn get<T: FloatBuffered>(&self, index: EvaluatorIndex<T>) -> T {
        self.data.get(index)
    }

    pub fn data(&self) -> &EvaluatedData {
        &self.data
    }

}




