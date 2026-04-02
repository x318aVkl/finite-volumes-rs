


use finite_volumes::prelude::*;



pub fn assemble_momentum_equation<'a, const DIM: usize>(
    mesh: &'a Mesh<DIM>,
    velocity: &Field<Vector<DIM>, geometry::Cell, DIM>,
    velocity_gradient: &Field<Matrix<DIM, DIM>, geometry::Cell, DIM>,
    phi: &Field<f64, geometry::Face, DIM>,
    viscosity: f64,
    dt: f64,
    velocity_bc: impl Fn(&FaceRef<'a, DIM>) -> (f64, Vector<DIM>),
) -> Result<(DistributedMatrix<f64>, DistributedVector<Vector<DIM>>), finite_volumes::error::Error> {
    

    let mut lhs = DistributedMatrix::from_cut_sparsity(mesh.cell_to_cell_sparsity(), mesh.n_cells());
    let mut rhs = DistributedVector::from_size(mesh.n_cells());

    // time term
    for cell in mesh.iter_cells() {
        let i = cell.id();

        lhs[[i, i]] += 1.0 / dt * cell.volume();

        rhs[i] += velocity[i] / dt * cell.volume();
    }


    // diffusion-convection
    for face in mesh.iter_faces() {
        let i = face.owner();
        let celli = mesh.cell(i);

        let phi = phi[face.id()];

        match face.neighbor() {
            FaceNeighbor::Cell(j) => {

                // diffusion, central scheme
                let cellj = mesh.cell(j);

                let delta = cellj.center() - celli.center();
                let dx = delta.norm();
                let t = - viscosity * 1.0 / dx * face.area();

                if celli.owned() {
                    lhs[[i, i]] -= t;
                    lhs[[i, j]] += t;
                }

                if cellj.owned() {
                    lhs[[j, i]] += t;
                    lhs[[j, j]] -= t; 
                }

                // convection, limited upwind scheme
                let u = if phi > 0.0 {i} else {j};
                let d = if u == i {j} else {i};
                let grad_n = (velocity[cellj.id()] - velocity[celli.id()]) / delta.norm();

                let r = if grad_n.norm() > 1e-8 {
                    (2.0*grad_n.dot(velocity_gradient[u].dot(delta)) / (delta.norm() * grad_n.dot(grad_n)) - 1.0).max(0.0)
                } else {
                    1.0
                };
                let lim = (1.0*r).min(1.0);

                let t = - phi * face.area();

                if celli.owned() {
                    lhs[[i, u]] -= t * (0.5 * lim + (1.0 - lim));
                    lhs[[i, d]] -= t * 0.5 * lim;
                }
                if cellj.owned() {
                    lhs[[j, u]] += t * (0.5 * lim + (1.0 - lim));
                    lhs[[j, d]] += t * 0.5 * lim;
                }
            },
            FaceNeighbor::Boundary(_) => {

                let (blhs, bv) = velocity_bc(&face); 

                // face value = blhs*cell_value + bv

                // diffusion: t * (face_value - cell_value)
                // t * (blhs*cell_Value + bv - cell_value)
                // tt * ((blhs - 1.0) * cell_value + bv)

                // diffusion
                let delta = face.center() - celli.center();
                let dx = delta.norm();
                let t = - viscosity * 1.0 / dx * face.area();
                lhs[[i, i]] -= t * (1.0 - blhs);
                rhs[i] -= bv * t;

                // convection
                let t = - phi * face.area();
                lhs[[i, i]] -= t * blhs;
                rhs[i] += t * bv;
            },
        }
    }

    Ok((lhs, rhs))
}



pub fn compute_velocity_gradients<'a, const DIM: usize>(
    velocity_gradient: &mut Field<Matrix<DIM, DIM>, geometry::Cell, DIM>,
    velocity: &Field<Vector<DIM>, geometry::Cell, DIM>,
    mesh: &'a Mesh<DIM>,
    velocity_bc: impl Fn(&FaceRef<'a, DIM>) -> (f64, Vector<DIM>),
) {
    // compute and update gradients
    for cell in mesh.iter_cells() {

        let mut grad = velocity[cell.id()].outer(cell.own_grad());

        for (f, g) in cell.iter_grad() {
            let face = mesh.face(f);
            match face.neighbor() {
                FaceNeighbor::Cell(_c) => {
                    let c = face.other_cell(cell.id()).unwrap();
                    grad += velocity[c].outer(g);
                },
                FaceNeighbor::Boundary(_) => {
                    let (blhs, bv) = velocity_bc(&face); 
                    
                    grad += (blhs * velocity[cell.id()] + bv).outer(g);

                },
            }
        }

        velocity_gradient[cell.id()] = grad;
    }
    velocity_gradient.update();
}


pub fn estimate_phi<'a, const DIM: usize>(
    phi: &mut Field<f64, geometry::Face, DIM>,
    velocity: &Field<Vector<DIM>, geometry::Cell, DIM>,
    mesh: &'a Mesh<DIM>,
    velocity_bc: impl Fn(&FaceRef<'a, DIM>) -> (f64, Vector<DIM>)
) {
    for face in mesh.iter_faces() {
        let cell0 = face.owner();
        match face.neighbor() {
            FaceNeighbor::Cell(cell1) => {
                let w = face.linear_factor();
                let u_face = velocity[cell0] * w + velocity[cell1] * (1.0 - w);
                phi[face.id()] = u_face.dot(face.normal());
            },
            FaceNeighbor::Boundary(_) => {
                let (blhs, bv) = velocity_bc(&face); 

                let u_face = bv + blhs * velocity[cell0];
                
                phi[face.id()] = u_face.dot(face.normal());
            },
        }
    }
}
