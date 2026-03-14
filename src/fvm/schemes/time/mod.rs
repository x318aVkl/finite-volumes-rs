use crate::{Mesh, prelude::CellRef};




pub mod euler;

pub use euler::Euler;

pub trait TimeScheme<const DIM: usize> {
    type Lhs;
    type Rhs;

    fn terms<'a>(&self, cell: &CellRef<'a, DIM>, mesh: &Mesh<DIM>) -> (Self::Lhs, Self::Rhs);
}


impl<'b, Lhs, Rhs, const DIM: usize> TimeScheme<DIM> for Box<dyn TimeScheme<DIM, Lhs=Lhs, Rhs=Rhs> + 'b> {
    type Lhs = Lhs;
    type Rhs = Rhs;

    fn terms<'a>(&'a self, cell: &CellRef<'a, DIM>, mesh: &Mesh<DIM>) -> (Self::Lhs, Self::Rhs) {
        self.as_ref().terms(cell, mesh)
    }
}

