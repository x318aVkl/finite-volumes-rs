use std::{ops::{Add, AddAssign, DivAssign, Mul, Sub}, process::Output};

use mpi::traits::Equivalence;

use crate::{Field, Mesh, Vector, linalg::Magnitude, prelude::{CellIndex, geometry}};


pub trait ElemwiseMax {
    fn elemwise_max(self, rhs: Self) -> Self;
}
pub trait ElemwiseFloatMax {
    fn elemwise_fmax(self, rhs: f64) -> Self;
}
pub trait ElemwiseFloatMin {
    fn elemwise_fmin(self, rhs: f64) -> Self;
}
pub trait ElemwiseMin {
    fn elemwise_min(self, rhs: Self) -> Self;
}
pub trait ElemwiseAbs {
    fn elemwise_abs(self) -> Self;
}
pub trait ElemwiseDiv {
    fn elemwise_div(self, rhs: Self) -> Self;
}
pub trait ElemwiseOwnMax {
    fn elemwise_own_max(self) -> f64;
}
pub trait ElemwiseOwnMin {
    fn elemwise_own_min(self) -> f64;
}
pub fn compute_hessian_criteria<G, V, const DIM: usize>(
    criteria: &mut Field<f64, geometry::Cell, DIM>,
    gradients: &Field<G, geometry::Cell, DIM>,
    field: &Field<V, geometry::Cell, DIM>,
    mesh: &Mesh<DIM>,
)
where G: Mul<f64, Output = G> + Magnitude<Output = f64> + Copy + Default + AddAssign + Add<G, Output = G> + DivAssign<f64>,
V: Clone + Copy + ElemwiseMax + ElemwiseMin + ElemwiseOwnMax + ElemwiseOwnMin + Sub<V, Output = V> + Equivalence + Default
{
    let mut average_hessiandx2_mag: f64 = 0.0;
    let mut maxv = field[CellIndex::from(0)];
    let mut minv = field[CellIndex::from(0)];
    for cell in mesh.iter_cells() {
        let c0 = cell.id();

        let mut dgdx = G::default();
        let mut dgdy = G::default();
        let mut dgdz = G::default();

        let grad_cell = gradients[c0];

        for f in cell.faces() {
            let face = mesh.face(*f);
            let n = face.outer_normal(cell.center());
            match face.other_cell(cell.id()) {
                Some(ocell) => {
                    let w = face.linear_factor();
                    let gface = grad_cell * w + gradients[ocell] * (1.0 - w);
                    dgdx += gface * (face.area() * n.x());
                    dgdy += gface * (face.area() * n.y());
                    if DIM == 3 {
                        dgdz += gface * (face.area() * n.z());
                    }
                },
                None => {
                    dgdx += grad_cell * (face.area() * n.x());
                    dgdy += grad_cell * (face.area() * n.y());
                    if DIM == 3 {
                        dgdz += grad_cell * (face.area() * n.z());
                    }
                }
            }
        }
        dgdx /= cell.volume();
        dgdy /= cell.volume();
        dgdz /= cell.volume();

        let mut hmag = 0.0;
        hmag += dgdx.magnitude().powi(2);
        hmag += dgdy.magnitude().powi(2);
        hmag += dgdz.magnitude().powi(2);

        hmag = hmag.sqrt();

        let h = hmag;
        let dx = cell.volume().powf(1.0/(DIM as f64));

        let hdx2 = h * dx * dx;

        let hdx2mag = hdx2;

        criteria[c0] = hdx2mag;
        average_hessiandx2_mag = average_hessiandx2_mag.max(hdx2mag);

        let vi = field[c0];
        maxv = maxv.elemwise_max(vi);
        minv = minv.elemwise_min(vi);
    }

    average_hessiandx2_mag = mesh.comm().reduce_max(average_hessiandx2_mag);
    let maxv = mesh.comm().reduce_max(maxv.elemwise_own_max());
    let minv = mesh.comm().reduce_min(minv.elemwise_own_min());

    let delta = (maxv - minv).max(1e-4);
    //println!("delta = {}, hdx2mag = {}", delta, average_hessiandx2_mag);

    let normalize_factor = average_hessiandx2_mag.max(delta);

    for cell in mesh.iter_cells() {
        let c0 = cell.id();
        criteria[c0] /= normalize_factor;
        //criteria[c0] = criteria[c0];
    }

    criteria.update();
}




impl ElemwiseMax for f64 {
    fn elemwise_max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
}

impl ElemwiseFloatMax for f64 {
    fn elemwise_fmax(self, rhs: f64) -> Self {
        self.max(rhs)
    }
}
impl ElemwiseMin for f64 {
    fn elemwise_min(self, rhs: Self) -> Self {
        self.min(rhs)
    }
}
impl ElemwiseFloatMin for f64 {
    fn elemwise_fmin(self, rhs: f64) -> Self {
        self.min(rhs)
    }
}
impl ElemwiseAbs for f64 {
    fn elemwise_abs(self) -> Self {
        self.abs()
    }
}
impl ElemwiseDiv for f64 {
    fn elemwise_div(self, rhs: Self) -> Self {
        self / rhs
    }
}
impl ElemwiseOwnMax for f64 {
    fn elemwise_own_max(self) -> f64 {
        self
    }
}
impl ElemwiseOwnMin for f64 {
    fn elemwise_own_min(self) -> f64 {
        self
    }
}





impl<const N: usize> ElemwiseMax for Vector<N> {
    fn elemwise_max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
}

impl<const N: usize> ElemwiseFloatMax for Vector<N> {
    fn elemwise_fmax(mut self, rhs: f64) -> Self {
        for i in 0..N {
            self[i] = self[i].max(rhs);
        }
        self
    }
}
impl<const N: usize> ElemwiseMin for Vector<N> {
    fn elemwise_min(self, rhs: Self) -> Self {
        self.min(rhs)
    }
}
impl<const N: usize> ElemwiseFloatMin for Vector<N> {
    fn elemwise_fmin(mut self, rhs: f64) -> Self {
        for i in 0..N {
            self[i] = self[i].min(rhs);
        }
        self
    }
}
impl<const N: usize> ElemwiseAbs for Vector<N> {
    fn elemwise_abs(self) -> Self {
        self.abs()
    }
}
impl<const N: usize> ElemwiseDiv for Vector<N> {
    fn elemwise_div(mut self, rhs: Self) -> Self {
        for i in 0..N {
            self[i] = self[i] / rhs[i];
        }
        self
    }
}
impl<const N: usize> ElemwiseOwnMax for Vector<N> {
    fn elemwise_own_max(self) -> f64 {
        self.self_max()
    }
}
impl<const N: usize> ElemwiseOwnMin for Vector<N> {
    fn elemwise_own_min(self) -> f64 {
        self.self_min()
    }
}