
use std::{marker::PhantomData, ops::Mul};

use crate::{core::traits::{Unit, Zero}, prelude::FaceNeighbor};

use super::FaceInterpolationScheme;



pub struct Linear<T, Lhs, const DIM: usize> {
    pht: PhantomData<T>,
    phl: PhantomData<Lhs>,
}

impl<'a, T, Lhs, const DIM: usize> Linear<T, Lhs, DIM> {
    pub fn new() -> Self {
        Self { pht: PhantomData, phl: PhantomData }
    }
}

impl<Lhs, const DIM: usize, T> FaceInterpolationScheme<DIM> for Linear<T, Lhs, DIM> where Lhs: Unit + Zero + Mul<f64, Output = Lhs>, T: Zero {
    type Lhs = Lhs;
    type Rhs = T;

    fn terms<'a>(&self, face: &'a crate::prelude::FaceRef<'a, DIM>, mesh: &'a crate::Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        
        let fc = face.center();

        let cc = mesh.cell(face.owner()).center();

        let nc = match face.neighbor() {
            FaceNeighbor::Cell(c) => {
                mesh.cell(c).center()
            },
            FaceNeighbor::Boundary(_) => {
                face.center()
            }
        };

        let w = face.normal().dot(nc - fc) / (face.normal().dot(nc - cc)).max(1e-14);
        
        (Lhs::unit()*w, Lhs::unit()*(1.0 - w), T::zero())
    }
}
