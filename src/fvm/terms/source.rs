use std::marker::PhantomData;

use crate::{Field, fvm::terms::{Term, TermWrapper}, prelude::{Zero, geometry}};




pub struct LinearSource<'a, T, V, const DIM: usize> {
    linear_coefficients: &'a Field<T, geometry::Cell, DIM>,
    pdv: PhantomData<V>,
}


impl<'a, T, V, const DIM: usize> LinearSource<'a, T, V, DIM> {
    pub fn new(linear_coefficients: &'a Field<T, geometry::Cell, DIM>) -> Self {
        Self { linear_coefficients, pdv: PhantomData }
    }
}



pub fn linear_source<'a, T, V, const DIM: usize>(linear_coefficients: &'a Field<T, geometry::Cell, DIM>) -> TermWrapper<LinearSource<'a, T, V, DIM>, DIM>
where T: Copy + Zero,
V: Zero
{
    LinearSource::new(linear_coefficients).wrap()
}


impl<'b, T, V, const DIM: usize> Term<DIM> for LinearSource<'b, T, V, DIM> where T: Copy + Zero, V: Zero {
    type Lhs = T;
    type Rhs = V;

    fn cell_terms<'a>(&self, cell: &'a crate::prelude::CellRef<'a, DIM>, _mesh: &'a crate::Mesh<DIM>) -> (Self::Lhs, Self::Rhs) {
        (
            self.linear_coefficients[cell.id()],
            Self::Rhs::zero(),
        )
    }

}
