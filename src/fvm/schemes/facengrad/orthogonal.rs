
use std::{marker::PhantomData, ops::Mul};

use crate::{Mesh, prelude::{FaceNeighbor, FaceRef, Unit, Zero}};

use super::FaceNormalGradientScheme;



pub struct Orthogonal<T, Lhs> {
    pht: PhantomData<T>,
    phl: PhantomData<Lhs>,
}

impl<T, Lhs> Orthogonal<T, Lhs> {
    pub fn new() -> Self {
        Self { pht: PhantomData, phl: PhantomData }
    }
}


impl<T, Lhs, const DIM: usize> FaceNormalGradientScheme<DIM> for Orthogonal<T, Lhs> where Lhs: Unit + Mul<f64, Output = Lhs>, T: Zero {
    type Lhs = Lhs;
    type Rhs = T;

    fn terms<'a>(&self, face: &'a FaceRef<'a, DIM>, mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        let celli = mesh.cell(face.owner());

        match face.neighbor() {
            FaceNeighbor::Cell(j) => {
                let cellj = mesh.cell(j);
                let dx = (cellj.center() - celli.center()).norm();
                let t = 1.0 / dx;

                (Lhs::unit() * (-t), Lhs::unit() * t, T::zero())
            },
            FaceNeighbor::Boundary(_) => {
                let dx = (face.center() - celli.center()).norm();
                let t = 1.0 / dx;

                (Lhs::unit() * (-t), Lhs::unit() * t, T::zero())
            },
        }
    }
}

