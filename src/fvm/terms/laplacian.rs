use std::ops::{Mul, Neg};

use crate::{Field, Mesh, fvm::{schemes::facengrad::FaceNormalGradientScheme, terms::{Term, TermWrapper}}, prelude::{FaceRef, Zero, geometry}};





pub struct Laplacian<'a, D, Lhs, Rhs, const DIM: usize> {
    scheme: Box<dyn FaceNormalGradientScheme<DIM, Lhs = Lhs, Rhs = Rhs> + 'a>,
    diffusivity: &'a Field<D, geometry::Face, DIM>,
}


impl<'a, D, Lhs, Rhs, const DIM: usize> Laplacian<'a, D, Lhs, Rhs, DIM> {
    pub fn new(scheme: impl FaceNormalGradientScheme<DIM, Lhs = Lhs, Rhs = Rhs> + 'a, diffusivity: &'a Field<D, geometry::Face, DIM>) -> Self {
        Self { scheme: Box::new(scheme), diffusivity }
    }
}

pub fn laplacian<'a, D, Lhs, Rhs, const DIM: usize>(scheme: impl FaceNormalGradientScheme<DIM, Lhs = Lhs, Rhs = Rhs> + 'a, diffusivity: &'a Field<D, geometry::Face, DIM>)
-> TermWrapper<Laplacian<'a, D, Lhs, Rhs, DIM>, DIM> 
where D: Mul<Lhs, Output = Lhs> + Mul<Rhs, Output = Rhs> + Copy,
Lhs: Zero,
Rhs: Zero + Neg<Output = Rhs>
{
    TermWrapper { term: Laplacian::new(scheme, diffusivity) }
}


impl<'b, D, Lhs, Rhs, const DIM: usize> Term<DIM> for Laplacian<'b, D, Lhs, Rhs, DIM> where D: Mul<Lhs, Output = Lhs> + Mul<Rhs, Output = Rhs> + Copy, Lhs: Zero, Rhs: Zero + Neg<Output = Rhs> {
    type Lhs = Lhs;
    type Rhs = Rhs;

    fn face_terms<'a>(&self, face: &'a FaceRef<'a, DIM>, mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        let (lhs0, lhs1, rhs) = self.scheme.terms(face, mesh);

        let diff = self.diffusivity[face.id()];

        (
            diff * lhs0,
            diff * lhs1,
            -(diff * rhs)
        )
    }
}



