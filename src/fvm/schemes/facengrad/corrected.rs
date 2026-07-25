use std::{marker::PhantomData, ops::{Add, Mul, Neg}};

use crate::{Field, Mesh, Vector, prelude::{FaceNeighbor, FaceRef, Unit, geometry}};


use super::FaceNormalGradientScheme;



pub struct Corrected<'a, V, G, Lhs, const DIM: usize> {
    gradients: &'a Field<G, geometry::Cell, DIM>,
    /// coefficient that multiplies the non-orthogonal correction, 0: no correction, 1: full correction
    corrector_reduction: f64,
    phv: PhantomData<V>,
    pdl: PhantomData<Lhs>,
}


impl<'a, V, G, Lhs, const DIM: usize> Corrected<'a, V, G, Lhs, DIM> {
    pub fn new(gradients: &'a Field<G, geometry::Cell, DIM>, corrector_reduction: f64) -> Self {
        Self {
            gradients,
            corrector_reduction,
            phv: PhantomData,
            pdl: PhantomData
        }
    }
}



impl<'b, G, V, Lhs, const DIM: usize> FaceNormalGradientScheme<DIM> for Corrected<'b, V, G, Lhs, DIM> where G: Copy + Mul<Vector<DIM>, Output = V> + Mul<f64, Output = G> + Add<G, Output = G>, V: Neg<Output=V> + Mul<f64, Output = V>, Lhs: Unit + Mul<f64, Output = Lhs> {
    type Lhs = Lhs;
    type Rhs = V;

    fn terms<'a>(&self, face: &'a FaceRef<'a, DIM>, mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        let i = face.owner();
        let celli = mesh.cell(i);

        match face.neighbor() {
            FaceNeighbor::Cell(j) => {
                let cellj = mesh.cell(j);
                let d = cellj.center() - celli.center();
                let n = face.normal();
                let c_corr = 1.0 / n.dot(d);

                let g0 = self.gradients[i];
                let g1 = self.gradients[j];

                let w = (face.center() - celli.center()).dot(n) / d.dot(n);
            
                let gf = g0 * (1.0 - w) + g1 * w;

                let explicit = gf * (n - c_corr * d);

                (Lhs::unit() * ( -c_corr ), Lhs::unit() * c_corr, explicit * self.corrector_reduction )
            },
            FaceNeighbor::Boundary(_) => {
                let d = face.center() - celli.center();
                let n = face.normal();
                let c_corr = 1.0 / n.dot(d);

                let explicit = self.gradients[i] * (n - c_corr * d);

                (Lhs::unit() * ( -c_corr ), Lhs::unit() * c_corr,  explicit * self.corrector_reduction )
            },
        }
    }
}

