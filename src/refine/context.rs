use crate::{Field, Mesh, prelude::{CellRef, geometry}, refine::mesh::RefinementMesh};




pub struct RefinementContext<const DIM: usize> {

    base_mesh: Mesh<DIM>,

    refinement_criteria: Field<f64, geometry::Cell, DIM>,
    level: f64,

    refined_mesh: RefinementMesh<DIM>,

}


impl<const DIM: usize> RefinementContext<DIM> {
    pub fn from_mesh(mesh: Mesh<DIM>) -> Self {

        let rmesh = RefinementMesh::from_mesh(&mesh);
        let refcrit = Field::from_mesh(&mesh);

        Self {
            base_mesh: mesh,
            refinement_criteria: refcrit,
            level: 0.9,
            refined_mesh: rmesh,
        } 
    }

    pub fn build(self) -> Mesh<DIM> {
        self.refined_mesh.build_mesh()
    } 

    pub fn criteria(mut self, f: impl Fn(CellRef<DIM>) -> f64) -> Self {
        for cell in self.base_mesh.iter_cells() {
            self.refinement_criteria[cell.id()] = f(cell);
        }
        self
    }
    pub fn level(mut self, level: f64) -> Self {
        self.level = level;
        self
    }

    pub fn refine(mut self) -> Self {

        self.refined_mesh.refine(self.refinement_criteria.raw_data(), self.level);

        self
    }

}

