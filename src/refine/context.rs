use crate::{Field, Mesh, prelude::{CellRef, geometry}, refine::mesh::RefinementMesh};



#[derive(Clone, Copy, Debug)]
pub(super) enum RefCommand {
    Refine,
    Coarsen,
}


pub struct RefinementContext<const DIM: usize> {

    base_mesh: Mesh<DIM>,
    current_mesh: Option<Mesh<DIM>>,

    refinement_criteria: Field<f64, geometry::Cell, DIM>,
    level: f64,

    refinement_order: Vec<Option<(usize, RefCommand)>>,

    refined_mesh: RefinementMesh<DIM>,

}


impl<const DIM: usize> RefinementContext<DIM> {
    pub fn from_mesh(mesh: Mesh<DIM>) -> Self {

        let rmesh = RefinementMesh::from_mesh(&mesh);
        let refcrit = Field::from_mesh(&mesh);

        Self {
            base_mesh: mesh,
            current_mesh: None,
            refinement_criteria: refcrit,
            level: 0.9,
            refinement_order: vec![],
            refined_mesh: rmesh,
        } 
    }

    pub fn set_criteria(&mut self, f: impl Fn(CellRef<DIM>) -> f64) -> &mut Self {
        let meshref = if self.current_mesh.is_none() {
            &self.base_mesh
        } else {
            self.current_mesh.as_ref().unwrap()
        };
        for cell in meshref.iter_cells() {
            self.refinement_criteria[cell.id()] = f(cell);
        }
        self
    }
    pub fn set_level(&mut self, level: f64) -> &mut Self {
        self.level = level;
        self
    }

    pub fn refine(&mut self) -> Mesh<DIM> {

        // compute the refinement order
        self.refined_mesh.compute_refinement_order(&mut self.refinement_order, self.refinement_criteria.raw_data(), self.level);

        // refine the refined mesh
        self.refined_mesh.refine(&self.refinement_order);
        let mesh = self.refined_mesh.build_mesh();
        self.current_mesh = Some(mesh.clone());

        // update the refinement criteria field
        self.refinement_criteria = Field::from_mesh(&mesh);

        mesh
    }

    pub fn mesh(&self) -> &Mesh<DIM> {
        if self.current_mesh.is_some() {
            self.current_mesh.as_ref().unwrap()
        } else {
            &self.base_mesh
        }
    }

}

