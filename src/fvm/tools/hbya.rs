/*

    hbya

    tools for fluid flow problems with pressure-based solution process

*/

use crate::{Field, Mesh, Vector, linalg::{DistributedMatrix, DistributedVector}, prelude::{CellIndex, FaceNeighbor, FaceRef, geometry}};





pub fn compute_hbya_ainv<const DIM: usize>(
    hbya: &mut Field<Vector<DIM>, geometry::Cell, DIM>,
    ainv: &mut Field<f64, geometry::Cell, DIM>,
    velocity: &Field<Vector<DIM>, geometry::Cell, DIM>,
    mlhs: &DistributedMatrix<f64>,
    mrhs: &DistributedVector<Vector<DIM>>,
    mesh: &Mesh<DIM>,
) {

    // compute ainv
    for i in 0..mlhs.nrows() {
        let ci = CellIndex::from(i);

        let cell = mesh.cell(ci);

        ainv[ci] = cell.volume() / mlhs[[i, i]];
    }
    ainv.update();

    // compute hbya
    for i in 0..mrhs.len() {
        let ci = CellIndex::from(i);

        let mut hi = mrhs[i];

        for (j, aij) in mlhs.iter_row(i) {
            if i != j {
                hi -= velocity[CellIndex::from(j)] * aij;
            }
        }

        hbya[ci] = hi / mlhs[[i, i]];
    }
    hbya.update();

}





pub fn intepolate_hbya_ainv_faces<'a, const DIM: usize>(
    hbyan_face: &mut Field<f64, geometry::Face, DIM>,
    ainv_face: &mut Field<f64, geometry::Face, DIM>,
    hbya: &Field<Vector<DIM>, geometry::Cell, DIM>,
    ainv: &Field<f64, geometry::Cell, DIM>,
    velocity_bc: impl Fn(&FaceRef<'a, DIM>) -> (f64, Vector<DIM>),
    pressure_bc: impl Fn(&FaceRef<'a, DIM>) -> (f64, f64),
    mesh: &'a Mesh<DIM>,
) {
    for face in mesh.iter_faces() {
        
        let i = face.owner();
        
        match face.neighbor() {
            FaceNeighbor::Cell(j) => {
                hbyan_face[face.id()] = (hbya[i] + hbya[j]).dot(face.normal()) * 0.5;
                ainv_face[face.id()] = (ainv[i] + ainv[j]) * 0.5;
            },
            FaceNeighbor::Boundary(_) => {
                //get velocity and pressure bcs
                let (blhsu, bvu) = velocity_bc(&face);
                let (blhsp, bvp) = pressure_bc(&face);

                let p_wall = ((blhsp - 1.0).abs() < 1e-10) && (bvp.abs() < 1e-10);
                let u_wall = (blhsu.abs() < 1e-10) && (bvu.dot(face.normal()).abs() < 1e-10);

                if p_wall && u_wall {
                    // zero since wall
                    hbyan_face[face.id()] = 0.0;
                } else {
                    hbyan_face[face.id()] = hbya[i].dot(face.normal());
                }

                ainv_face[face.id()] = ainv[i];
            },
        }
    }
    hbyan_face.update();
    ainv_face.update();
}





pub fn correct_phi<'a, const DIM: usize>(
    phi: &mut Field<f64, geometry::Face, DIM>,
    hbyan_face: &Field<f64, geometry::Face, DIM>,
    ainv_face: &Field<f64, geometry::Face, DIM>,
    pressure: &Field<f64, geometry::Cell, DIM>,
    pressure_bc: impl Fn(&FaceRef<'a, DIM>) -> (f64, f64),
    mesh: &'a Mesh<DIM>,
) {
    for face in mesh.iter_faces() {
        let i = face.owner();
        let celli = mesh.cell(i);

        let ainv = ainv_face[face.id()];
        let hbyan = hbyan_face[face.id()];

        match face.neighbor() {
            FaceNeighbor::Cell(j) => {

                // diffusion, central scheme
                let cellj = mesh.cell(j);

                let delta = cellj.center() - celli.center();
                let dx = delta.norm();
                let pgrad_n = (pressure[j] - pressure[i]) / dx;

                phi[face.id()] = hbyan - ainv * pgrad_n;
            },
            FaceNeighbor::Boundary(_) => {
                let (blhsp, bvp) = pressure_bc(&face);

                let p_face = blhsp * pressure[i] + bvp;

                let delta = face.center() - celli.center();
                let dx = delta.norm();

                let pgrad_n = (p_face - pressure[i]) / dx;

                phi[face.id()] = hbyan - ainv * pgrad_n;
            },
        }
    }

    // call update to collect phi from other ranks
    phi.update();

}



pub fn correct_velocity<const DIM: usize>(
    velocity: &mut Field<Vector<DIM>, geometry::Cell, DIM>,
    hbya: &Field<Vector<DIM>, geometry::Cell, DIM>,
    ainv: &Field<f64, geometry::Cell, DIM>,
    pressure_gradient: &Field<Vector<DIM>, geometry::Cell, DIM>,
    mesh: &Mesh<DIM>,
) {

    for cell in mesh.iter_cells() {
        let i = cell.id();

        let pgrad = pressure_gradient[i];

        // update velocity
        velocity[i] = hbya[i] - ainv[i] * pgrad;
    }
    velocity.update();

}





