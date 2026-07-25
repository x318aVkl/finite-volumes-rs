use std::ops::{Add, AddAssign, Mul};

use crate::{Field, Mesh, Vector, fvm::schemes::{faceinterp::FaceInterpolationScheme, facengrad::FaceNormalGradientScheme, limiters::LimiterScheme}, prelude::{FaceNeighbor, FaceRef, geometry}};



pub trait LimiterFrom<const DIM: usize> {
    
    fn to_limiter_r(
        fngrad: Self,
        gradient_dot_delta: Self,
        delta: Vector<DIM>,
    ) -> f64;
}

impl<const DIM: usize> LimiterFrom<DIM> for f64 {
    fn to_limiter_r(
        fngrad: Self,
        gradient_dot_delta: Self,
        delta: Vector<DIM>,
    ) -> f64 {
        if fngrad.abs() > 1e-12 {
            (
                2.0 * gradient_dot_delta / (delta.norm() * fngrad) - 1.0
            ).max(0.0)
        } else {
            1.0
        }
    }
}
impl<const N: usize, const DIM: usize> LimiterFrom<DIM> for Vector<N> {
    fn to_limiter_r(
        fngrad: Self,
        gradient_dot_delta: Self,
        delta: Vector<DIM>,
    ) -> f64 {
        if fngrad.norm() > 1e-12 {
            (
                2.0 * fngrad.dot(gradient_dot_delta) / (delta.norm() * fngrad.dot(fngrad)) - 1.0
            ).max(0.0)
        } else {
            1.0
        }
    }
}

pub fn compute_limiters<'a, V, G, FngLhs, FngRhs, FigLhs, BndLhs, const DIM: usize>(
    limiters: &mut Field<f64, geometry::Face, DIM>,
    values: &Field<V, geometry::Cell, DIM>,
    gradients: &Field<G, geometry::Cell, DIM>,
    limiter_scheme: impl LimiterScheme,
    fgradinterp_scheme: impl FaceInterpolationScheme<DIM, Lhs = FigLhs, Rhs = G>,
    fngrad_scheme: impl FaceNormalGradientScheme<DIM, Lhs = FngLhs, Rhs = FngRhs>,
    boundary_condition: impl Fn(&FaceRef<'a, DIM>) -> (BndLhs, V),
    mesh: &'a Mesh<DIM>,
) 
where
V: LimiterFrom<DIM> + Default + Copy + AddAssign + AddAssign<FngRhs> + Add<V, Output = V>,
FngLhs: Mul<V, Output = V>,
FigLhs: Default + Mul<G, Output = G>,
G: Copy + Add<G, Output=G> + Mul<f64, Output = G> + Mul<Vector<DIM>, Output = V> + Default + AddAssign,
BndLhs: Mul<V, Output = V>,
{


    for face in mesh.iter_faces() {
        let i = face.owner();
        let celli = mesh.cell(i);


        match face.neighbor() {
            FaceNeighbor::Cell(j) => {
                let cellj = mesh.cell(j);
                let delta = cellj.center() - celli.center();

                let (gi, gj, grhs) = fngrad_scheme.terms(&face, &mesh);

                let mut fngrad = V::default();
                fngrad += gi * values[i];
                fngrad += gj * values[j];

                fngrad += grhs;

                let (interp_i, interp_j, interp_rhs) = fgradinterp_scheme.terms(&face, &mesh);
                let mut grad_face = interp_rhs;

                grad_face += interp_i * gradients[i];
                grad_face += interp_j * gradients[j];

                let grad_face_delta = grad_face * delta;

                let r = V::to_limiter_r(fngrad, grad_face_delta, delta);

                let lim = limiter_scheme.get_limiter(r);

                limiters[face.id()] = lim;
            },
            FaceNeighbor::Boundary(_) => {
                let (bci, bce) = boundary_condition(&face);

                let fval = bci * values[i] + bce;

                let delta = face.center() - celli.center();

                let (gi, gj, grhs) = fngrad_scheme.terms(&face, &mesh);

                let mut fngrad = V::default();
                fngrad += gi * values[i];
                fngrad += gj * fval;

                fngrad += grhs;

                let grad_face = gradients[i];
                let grad_face_delta = grad_face * delta;

                let r = V::to_limiter_r(fngrad, grad_face_delta, delta);

                let lim = limiter_scheme.get_limiter(r);

                limiters[face.id()] = lim;
            },
        }
    }
    limiters.update();

}

