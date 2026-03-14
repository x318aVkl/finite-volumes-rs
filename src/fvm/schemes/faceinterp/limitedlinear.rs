
use std::{marker::PhantomData, ops::Mul};

use crate::{Field, prelude::{FaceNeighbor, Unit, Zero, geometry}};

use super::FaceInterpolationScheme;


pub struct LimitedLinear<'a, V, Lhs, const DIM: usize> {
    flux: &'a Field<f64, geometry::Face, DIM>,
    limiters: &'a Field<f64, geometry::Face, DIM>,
    pdv: PhantomData<V>,
    pdl: PhantomData<Lhs>,
}

impl<'a, V, Lhs, const DIM: usize> LimitedLinear<'a, V, Lhs, DIM> {
    pub fn new(
        flux: &'a Field<f64, geometry::Face, DIM>,
        limiters: &'a Field<f64, geometry::Face, DIM>,
    ) -> Self {
        Self { flux, limiters, pdv: PhantomData, pdl: PhantomData }
    }
}


impl<'b, Lhs, const DIM: usize, V> FaceInterpolationScheme<DIM> for LimitedLinear<'b, V, Lhs, DIM> where Lhs: Unit + Zero + Mul<f64, Output = Lhs>, V: Zero {
    type Lhs = Lhs;
    type Rhs = V;

    fn terms<'a>(&self, face: &'a crate::prelude::FaceRef<'a, DIM>, _mesh: &'a crate::Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        
        let flux = self.flux[face.id()];

        let u = if flux > 0.0 {0} else {1};
        let d = if u == 0 {1} else {0};

        let mut t = [0.0, 0.0];

        let lim = self.limiters[face.id()];

        t[u] = 0.5 * lim + (1.0 - lim);
        t[d] = 0.5 * lim;

        match face.neighbor() {
            FaceNeighbor::Cell(_) => {
                (Lhs::unit() * t[0], Lhs::unit() * t[1], V::zero())
            },
            FaceNeighbor::Boundary(_) => {

                // Boundary treatment will be consistent
                if flux > 0.0 {
                    // outlet
                    (Lhs::unit(), Lhs::zero(), V::zero())
                } else {
                    // inlet
                    (Lhs::zero(), Lhs::unit(), V::zero())
                }
            },
        }
    }
}

