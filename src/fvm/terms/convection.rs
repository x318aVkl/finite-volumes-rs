use std::ops::{Mul, Neg};

use crate::{Field, Mesh, fvm::{schemes::faceinterp::FaceInterpolationScheme, terms::TermWrapper}, prelude::{FaceRef, Zero, geometry}};

use super::Term;



pub struct Convection<'a, Lhs, Rhs, const DIM: usize> {
    scheme: Box<dyn FaceInterpolationScheme<DIM, Lhs = Lhs, Rhs = Rhs> + 'a>,
    flux: &'a Field<f64, geometry::Face, DIM>,
}

impl<'a, Lhs, Rhs, const DIM: usize> Convection<'a, Lhs, Rhs, DIM> {
    pub fn new(scheme: impl FaceInterpolationScheme<DIM, Lhs = Lhs, Rhs = Rhs> + 'a, flux: &'a Field<f64, geometry::Face, DIM>) -> Self {
        Self {
            scheme: Box::new(scheme),
            flux,
        }
    }
}


pub fn convection<'a, Lhs, Rhs, const DIM: usize>(scheme: impl FaceInterpolationScheme<DIM, Lhs = Lhs, Rhs = Rhs> + 'a, flux: &'a Field<f64, geometry::Face, DIM>) -> TermWrapper<Convection<'a, Lhs, Rhs, DIM>, DIM> 
where Lhs: Mul<f64, Output = Lhs> + Zero + Copy, Rhs: Mul<f64, Output = Rhs> + Neg<Output = Rhs> + Zero + Copy
{
    TermWrapper { term: Convection::new(scheme, flux) }
}



impl<'b, Lhs, Rhs, const DIM: usize> Term<DIM> for Convection<'b, Lhs, Rhs, DIM> where Lhs: Mul<f64, Output = Lhs> + Zero, Rhs: Mul<f64, Output = Rhs> + Neg<Output = Rhs> + Zero + Copy {
    type Lhs = Lhs;
    type Rhs = Rhs;

    fn face_terms<'a>(&self, face: &'a FaceRef<'a, DIM>, mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        let flux = self.flux[face.id()];
        let (l0, l1, r) = self.scheme.terms(face, mesh);
        (
            l0 * flux,
            l1 * flux,
            - r * flux
        )
    }
}



