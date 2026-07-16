use std::{marker::PhantomData, ops::{Mul, Neg}};

use crate::{Field, Mesh, fvm::terms::{Term, TermWrapper}, prelude::{FaceRef, Zero, geometry}};





pub struct Divergence<'a, D, Lhs, Rhs, const DIM: usize> {
    flux: &'a Field<D, geometry::Face, DIM>,
    pdv: PhantomData<Lhs>,
    rdv: PhantomData<Rhs>,
}


impl<'a, D, Lhs, Rhs, const DIM: usize> Divergence<'a, D, Rhs, Lhs, DIM> {
    pub fn new(flux: &'a Field<D, geometry::Face, DIM>) -> Self {
        Self { flux, pdv: PhantomData, rdv: PhantomData, }
    }
}

pub fn divergence<'a, D, Lhs, Rhs, const DIM: usize>(flux: &'a Field<D, geometry::Face, DIM>)
-> TermWrapper<Divergence<'a, D, Lhs, Rhs, DIM>, DIM> 
where D: Copy + Zero + Mul<f64, Output = D> + Neg<Output = D>, Lhs: Zero,
{
    TermWrapper { term: Divergence::new(flux) }
}


impl<'b, D, Lhs, Rhs, const DIM: usize> Term<DIM> for Divergence<'b, D, Lhs, Rhs, DIM> where D: Copy + Zero + Mul<f64, Output = D> + Neg<Output = D>, Lhs: Zero {
    type Lhs = Lhs;
    type Rhs = D;

    fn face_terms<'a>(&self, face: &'a FaceRef<'a, DIM>, _mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        //let area = face.area();
        let flux = self.flux[face.id()];

        (
            Lhs::zero(),
            Lhs::zero(),
            - flux
        )
    }
}


