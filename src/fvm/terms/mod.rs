use crate::{Mesh, prelude::{CellRef, FaceRef, Zero}};




pub mod laplacian;
pub mod convection;
pub mod time;
pub mod source;


pub use laplacian::laplacian;
pub use convection::convection;
pub use time::time;
pub use source::linear_source;

pub mod operations;




pub trait Term<const DIM: usize>: Sized {
    type Lhs: Zero;
    type Rhs: Zero;

    fn cell_terms<'a>(&self, _cell: &'a CellRef<'a, DIM>, _mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Rhs) {
        (Self::Lhs::zero(), Self::Rhs::zero())
    }

    fn face_terms<'a>(&self, _face: &'a FaceRef<'a, DIM>, _mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        (Self::Lhs::zero(), Self::Lhs::zero(), Self::Rhs::zero())
    }


    fn wrap(self) -> TermWrapper<Self, DIM> {
        TermWrapper { term: self }
    }
}




pub struct TermWrapper<T: Term<DIM>, const DIM: usize> {
    term: T,
}

impl<T: Term<DIM>, const DIM: usize> TermWrapper<T, DIM> {
    pub fn new(term: T) -> Self {
        Self { term }
    }
}


impl<T: Term<DIM>, const DIM: usize> Term<DIM> for TermWrapper<T, DIM> {
    type Lhs = <T as Term<DIM>>::Lhs;
    type Rhs = <T as Term<DIM>>::Rhs;

    fn cell_terms<'a>(&self, cell: &'a CellRef<'a, DIM>, mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Rhs) {
        self.term.cell_terms(cell, mesh)
    }

    fn face_terms<'a>(&self, face: &'a FaceRef<'a, DIM>, mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        self.term.face_terms(face, mesh)
    }
}







