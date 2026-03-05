use std::marker::PhantomData;

use mpi::traits::Equivalence;

use crate::{communicator::Communicator, Mesh, mesh::{Geometry, GlobalRelation, MeshGet}};




pub struct Field<'a, T, G: Geometry<DIM>, const DIM: usize> {
    data: Vec<T>,
    comm: Communicator<'a, G, DIM>,
    _phantom: PhantomData<G>,
}



impl<'a, E, I, T: Default + Clone, G: Geometry<DIM, IndexType = I, ElementType<'a> = E>, const DIM: usize> Field<'a, T, G, DIM>
where T: Equivalence, E: GlobalRelation, Mesh<DIM>: MeshGet<'a, I>, I: From<usize>, usize: From<I> {
    pub fn from_mesh(mesh: &'a Mesh<DIM>) -> Self {
        Field {
            data: vec![T::default(); G::global_size_from_mesh(mesh)],
            comm: Communicator::from_mesh(&mesh),
            _phantom: PhantomData,
        }
    }

    pub fn update(&mut self) {
        self.comm.collect(&mut self.data);
    }
}


impl<'a, T, G: Geometry<DIM>, const DIM: usize> std::ops::Index<G::IndexType> for Field<'a, T, G, DIM> where usize: From<<G as Geometry<DIM>>::IndexType> {
    type Output = T;
    fn index(&self, index: G::IndexType) -> &Self::Output {
        &self.data[usize::from(index)]
    }
}


impl<'a, T, G: Geometry<DIM>, const DIM: usize> std::ops::IndexMut<G::IndexType> for Field<'a, T, G, DIM> where usize: From<<G as Geometry<DIM>>::IndexType> {
    fn index_mut(&mut self, index: G::IndexType) -> &mut Self::Output {
        &mut self.data[usize::from(index)]
    }
}




