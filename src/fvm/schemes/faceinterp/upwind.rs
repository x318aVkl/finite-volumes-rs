
use std::marker::PhantomData;

use crate::{Field, core::traits::{Unit, Zero}, prelude::geometry};

use super::FaceInterpolationScheme;



pub struct Upwind<'a, T, Lhs, const DIM: usize> {
    flux: &'a Field<f64, geometry::Face, DIM>,
    pht: PhantomData<T>,
    phl: PhantomData<Lhs>,
}

impl<'a, T, Lhs, const DIM: usize> Upwind<'a, T, Lhs, DIM> {
    pub fn new(flux: &'a Field<f64, geometry::Face, DIM>) -> Self {
        Self { flux, pht: PhantomData, phl: PhantomData }
    }
}

impl<'b, Lhs, const DIM: usize, T> FaceInterpolationScheme<DIM> for Upwind<'b, T, Lhs, DIM> where Lhs: Unit + Zero, T: Zero {
    type Lhs = Lhs;
    type Rhs = T;

    fn terms<'a>(&self, face: &'a crate::prelude::FaceRef<'a, DIM>, _mesh: &'a crate::Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        let flux = self.flux[face.id()];
        if flux > 0.0 {
            (Lhs::unit(), Lhs::zero(), T::zero())
        } else {
            (Lhs::zero(), Lhs::unit(), T::zero())
        }
    }
}
