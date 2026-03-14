use crate::{Mesh, prelude::CellRef};




pub mod euler;

pub use euler::Euler;

pub trait TimeScheme<const DIM: usize> {
    type Lhs;
    type Rhs;

    fn terms<'a>(&self, cell: &CellRef<'a, DIM>, mesh: &Mesh<DIM>) -> (Self::Lhs, Self::Rhs);
}

