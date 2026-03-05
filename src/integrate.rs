use std::ops::{AddAssign, Div, Mul};

use crate::{Mesh, Vector, mesh::{CellIndex, FaceIndex}};




struct MeshIntegral<'a, const DIM: usize> {
    mesh: &'a Mesh<DIM>,
}


pub struct SurfaceIntegralData<'a, const DIM: usize> {
    boundary: Option<u16>,
    face: FaceIndex,
    cell: CellIndex,
    mesh: &'a Mesh<DIM>,
}

impl<'a, const DIM: usize> SurfaceIntegralData<'a, DIM> {
    pub fn ds(&self) -> f64 {
        match self.boundary {
            Some(_v) => {
                0.0
            },
            None => {
                1.0
            }
        }
    }
    pub fn dsb(&self, boundary: u16) -> f64 {
        match self.boundary {
            Some(v) => {
                if v == boundary {
                    1.0
                } else {
                    0.0
                }
            },
            None => {
                0.0
            }
        }
    }
    pub fn normal(&self) -> Vector<DIM> {
        self.mesh.face(self.face).outer_normal(self.mesh.cell(self.cell).center())
    }
}


impl<'a, const DIM: usize> MeshIntegral<'a, DIM> {


    fn integrate_cell<T: AddAssign + Mul<f64, Output = T> + Div<f64, Output = T> + Default + Copy, F>(&self, cell: CellIndex, node_data: &[T], volume_function: fn(T) -> T, flux_function: fn(T, SurfaceIntegralData<DIM>) -> F, mut initial: T) -> T
    where F: Mul<Vector<DIM>, Output = T>
    {

        let cell = self.mesh.cell(cell);

        for n in cell.nodes() {
            // approximate centroid value of subelement as node value
            let val = node_data[usize::from(*n)];
            
            initial += volume_function(val) * cell.volume() / (cell.nodes().len() as f64);
        }

        for f in cell.faces() {
            let face = self.mesh.face(*f);

            for n in face.nodes() {
                // approximate centroid value of subelement as node value
                let val = node_data[usize::from(*n)];

                let data = SurfaceIntegralData {
                    boundary: face.boundary(),
                    face: *f,
                    cell: cell.id(),
                    mesh: self.mesh,
                };

                let flux = flux_function(val, data);
                
                initial += flux * face.outer_normal(cell.center()) * face.area() / (face.nodes().len() as f64);
            }
        }

        initial
    }

}



pub fn integrate<T: AddAssign + Mul<f64, Output = T> + Div<f64, Output = T> + Default + Copy, F, const DIM: usize>(mesh: &Mesh<DIM>, node_function: fn(Vector<DIM>) -> T, volume_function: fn(T) -> T, flux_function: fn(T, SurfaceIntegralData<DIM>) -> F, initial: T) -> T
where F: Mul<Vector<DIM>, Output = T> {

    let mut integral = initial.clone();

    let integrator = MeshIntegral {mesh};

    let mut ndata = vec![initial; mesh.n_nodes()];
    for i in 0..ndata.len() {
        ndata[i] = node_function(mesh.node(i.into()).position());
    }

    for c in 0..mesh.n_cells() {
        integral += integrator.integrate_cell(c.into(), &ndata, volume_function, flux_function, initial);
    }

    integral
}
