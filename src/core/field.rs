use std::marker::PhantomData;

use mpi::traits::Equivalence;

use crate::core::{communicator::Communicator, Mesh, mesh::{Geometry, MeshGet}};




#[derive(Clone)]
pub struct Field<T, G: Geometry<DIM>, const DIM: usize> {
    data: Vec<T>,
    local_len: usize,
    comm: Communicator<G, DIM>,
    _phantom: PhantomData<G>,
}



impl<'a, I, T: Default + Clone, G: Geometry<DIM, IndexType = I>, const DIM: usize> Field<T, G, DIM>
where T: Equivalence, Mesh<DIM>: MeshGet<'a, I>, I: From<usize>, usize: From<I> {
    pub fn from_mesh(mesh: &Mesh<DIM>) -> Self {
        Field {
            data: vec![T::default(); G::global_size_from_mesh(mesh)],
            local_len: G::size_from_mesh(mesh),
            comm: Communicator::from_mesh(&mesh),
            _phantom: PhantomData,
        }
    }

    pub fn update(&mut self) {
        self.comm.collect(&mut self.data);
    }

    pub fn len(&self) -> usize {
        self.local_len
    }

    pub fn total_len(&self) -> usize {
        self.data.len()
    }

    pub fn raw_data(&self) -> &[T] {
        &self.data
    }
}


impl<T, G: Geometry<DIM>, const DIM: usize> std::ops::Index<G::IndexType> for Field<T, G, DIM> where usize: From<<G as Geometry<DIM>>::IndexType> {
    type Output = T;
    fn index(&self, index: G::IndexType) -> &Self::Output {
        &self.data[usize::from(index)]
    }
}


impl<T, G: Geometry<DIM>, const DIM: usize> std::ops::IndexMut<G::IndexType> for Field<T, G, DIM> where usize: From<<G as Geometry<DIM>>::IndexType> {
    fn index_mut(&mut self, index: G::IndexType) -> &mut Self::Output {
        &mut self.data[usize::from(index)]
    }
}




