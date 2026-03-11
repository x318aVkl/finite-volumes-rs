use std::ops::{AddAssign, Mul};

use mpi::traits::Equivalence;

use crate::{Mesh, Vector, core::{field::Field, mesh::{NodeIndex, geometry::Cell}}};


pub trait GlobalBounds {
    fn global_min() -> Self;
    fn global_max() -> Self;
}

pub trait Elementwise {
    type Element;

    fn elemwise_min(self, rhs: Self) -> Self;
    fn elemwise_max(self, rhs: Self) -> Self;

    fn elemwise_max_single(self, rhs: Self::Element) -> Self;
    fn elemwise_min_single(self, rhs: Self::Element) -> Self;

    fn elemwise_mul(self, rhs: Self) -> Self;
    fn elemwise_div(self, rhs: Self) -> Self;
    fn elemwise_add(self, rhs: Self) -> Self;
    fn elemwise_sub(self, rhs: Self) -> Self;
    fn elemwise_abs(self) -> Self;
    
    fn elemwise_powf(self, pow: f64) -> Self;

    fn elemwise_map(self, map: impl Fn(Self::Element) -> Self::Element) -> Self;
    fn elemwise_map_zip(self, rhs: Self, map: impl Fn(Self::Element, Self::Element) -> Self::Element) -> Self;
}

// returns one
pub trait UnitValue {
    fn unit_value() -> Self;
}



pub fn compute_limiters<T, const DIM: usize>(
    field: &Field<T, Node, DIM>,
    mesh: &Mesh<DIM>,
    limiters: &mut Field<T, Node, DIM>,
) 
where T: Default + Equivalence + Clone + Elementwise<Element = f64> + GlobalBounds + UnitValue + Copy + Default
 + Mul<f64, Output = T>
 + AddAssign,
{

    let mut phimax: Vec<T> = vec![T::global_min(); mesh.n_total_nodes()];
    let mut phimin: Vec<T> = vec![T::global_max(); mesh.n_total_nodes()];
    let mut nodedv = vec![0.0; mesh.n_total_nodes()];
    let mut quadrature = CellQuadrature::from_mesh(&mesh);

    let mut gphimax = T::global_min();
    let mut gphimin = T::global_max();
    for cell in mesh.iter_all_cells() {
        quadrature.update(&cell);

        for i in cell.nodes() {
            for j in cell.nodes() {
                let phij = field[*j];
                let  i = usize::from(*i);
                phimax[i] = phimax[i].elemwise_max(phij);
                phimin[i] = phimin[i].elemwise_min(phij);
            }
            gphimax = gphimax.elemwise_max(field[*i]);
            gphimin = gphimin.elemwise_min(field[*i]);
        }


        for subcell in quadrature.subcells() {
            let i = subcell.owner();

            // let mut phic = T::default();
            // for j in 0..quadrature.ndofs() {
            //     let n = NodeIndex::from(quadrature.dofs()[j]);
            //     phic += field[n] * subcell.value(j);
            // }
            let i = usize::from(quadrature.dofs()[i]);

            // phimax[i] = phimax[i].elemwise_max(phic);
            // phimin[i] = phimin[i].elemwise_min(phic);

            nodedv[i] += subcell.volume();

            //gphimax = gphimax.elemwise_max(phic);
            //gphimin = gphimin.elemwise_min(phic);
        }
    }

    let gphidelta = gphimax.elemwise_sub(gphimin).elemwise_abs();
    for i in 0..limiters.len() {
        let phii = field[NodeIndex::from(i)];

        let deltamax = phimax[i].elemwise_sub(phii).elemwise_abs();
        let deltamin = phii.elemwise_sub(phimin[i]).elemwise_abs();

        let dumax = deltamax.elemwise_max(deltamin);

        let kdx = (gphidelta * nodedv[i]).elemwise_powf(1.0 / (DIM as f64)) * 1.0;
        let kdx3 = kdx.elemwise_powf(3.0);
        let dumax2 = dumax.elemwise_powf(2.0);

        let s = dumax2.elemwise_map_zip(kdx3, |dumax2, kdx3| {
            if dumax2 > 2.0*kdx3 {
                0.0
            } else if dumax2 > kdx3 {
                let t = (dumax2 - kdx3) / kdx3;
                1.0 - 3.0 * (t*t - 2.0/3.0 * t*t*t)
            } else {
                1.0
            }
        });

        let r = ( deltamax.elemwise_div(deltamin.elemwise_abs().elemwise_max_single(1e-14)) )
            .elemwise_min( deltamin.elemwise_div(deltamax.elemwise_abs().elemwise_max_single(1e-14)) )
            * 1.5;

        let l = r.elemwise_map(|y| {
            if y < 1e-200 {
                0.0
            } else if y < 2.0 {
                y - y.powi(3) / 6.75
            } else {
                1.0
            }
        });

        limiters[NodeIndex::from(i)] = s.elemwise_add(T::unit_value().elemwise_sub(s).elemwise_mul(l));
    }
    limiters.update();

}




// Implement traits for f64 and vector
impl GlobalBounds for f64 {
    fn global_max() -> Self {
        f64::MAX
    }
    fn global_min() -> Self {
        f64::MIN
    }
}

impl UnitValue for f64 {
    fn unit_value() -> Self {
        1.0
    }
}

impl Elementwise for f64 {
    type Element = f64;
    fn elemwise_min(self, rhs: Self) -> Self {
        self.min(rhs)
    }
    fn elemwise_max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    fn elemwise_max_single(self, rhs: f64) -> Self {
        self.max(rhs)
    }
    fn elemwise_min_single(self, rhs: f64) -> Self {
        self.min(rhs)
    }

    fn elemwise_mul(self, rhs: Self) -> Self {
        self * rhs
    }
    fn elemwise_div(self, rhs: Self) -> Self {
        self / rhs
    }
    fn elemwise_add(self, rhs: Self) -> Self {
        self + rhs
    }
    fn elemwise_sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn elemwise_abs(self) -> Self {
        self.abs()
    }
    
    fn elemwise_powf(self, pow: f64) -> Self {
        self.powf(pow)
    }

    fn elemwise_map(self, map: impl Fn(Self::Element) -> Self::Element) -> Self {
        map(self)
    }
    fn elemwise_map_zip(self, rhs: Self, map: impl Fn(Self::Element, Self::Element) -> Self::Element) -> Self {
        map(self, rhs)
    }
}


impl<const DIM: usize> GlobalBounds for Vector<DIM> {
    fn global_max() -> Self {
        [f64::MAX; DIM].into()
    }
    fn global_min() -> Self {
        [f64::MIN; DIM].into()
    }
}

impl<const DIM: usize> UnitValue for Vector<DIM> {
    fn unit_value() -> Self {
        [1.0; DIM].into()
    }
}

impl<const DIM: usize> Elementwise for Vector<DIM> {
    type Element = f64;
    fn elemwise_min(self, rhs: Self) -> Self {
        self.min(rhs)
    }
    fn elemwise_max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    fn elemwise_max_single(self, rhs: f64) -> Self {
        self.max([rhs; DIM].into())
    }
    fn elemwise_min_single(self, rhs: f64) -> Self {
        self.min([rhs; DIM].into())
    }

    fn elemwise_mul(self, rhs: Self) -> Self {
        let mut out = Vector::new();
        for i in 0..DIM {
            out[i] = self[i] * rhs[i];
        }
        out
    }
    fn elemwise_div(self, rhs: Self) -> Self {
        let mut out = Vector::new();
        for i in 0..DIM {
            out[i] = self[i] / rhs[i];
        }
        out
    }
    fn elemwise_add(self, rhs: Self) -> Self {
        self + rhs
    }
    fn elemwise_sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn elemwise_abs(self) -> Self {
        self.abs()
    }
    
    fn elemwise_powf(self, pow: f64) -> Self {
        let mut out = Vector::new();
        for i in 0..DIM {
            out[i] = self[i].powf(pow);
        }
        out
    }

    fn elemwise_map(self, map: impl Fn(Self::Element) -> Self::Element) -> Self {
        let mut out = Vector::new();
        for i in 0..DIM {
            out[i] = map(self[i]);
        }
        out
    }
    fn elemwise_map_zip(self, rhs: Self, map: impl Fn(Self::Element, Self::Element) -> Self::Element) -> Self {
        let mut out = Vector::new();
        for i in 0..DIM {
            out[i] = map(self[i], rhs[i])
        }
        out
    }
}