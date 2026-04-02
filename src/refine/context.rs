use std::collections::HashMap;

use mpi::traits::Equivalence;

use crate::{Field, Mesh, prelude::{CellIndex, CellRef, geometry}, refine::mesh::RefinementMesh};



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

    max_refinement: usize,

    refinement_order: Vec<Option<(usize, RefCommand)>>,

    refined_mesh: RefinementMesh<DIM>,

    previous_local_to_leaf: Vec<Option<usize>>,

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
            max_refinement: 1000,
            refinement_order: vec![],
            refined_mesh: rmesh,
            previous_local_to_leaf: vec![],
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

    pub fn set_max_refinement(&mut self, refinement: usize) -> &mut Self {
        self.max_refinement = refinement;
        self
    }

    pub fn refine(&mut self) -> Mesh<DIM> {

        // save the leaf to local map
        self.previous_local_to_leaf = self.refined_mesh.get_local_to_leaf().clone();

        // compute the refinement order
        self.refined_mesh.compute_refinement_order(&mut self.refinement_order, self.refinement_criteria.raw_data(), self.level, self.max_refinement);

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


    pub fn map_field<T>(&self, field: Field<T, geometry::Cell, DIM>) -> Field<T, geometry::Cell, DIM> 
    where T: Default + Clone + Copy + Equivalence
    {

        let mut new_field = Field::from_mesh(match self.current_mesh.as_ref() {
            Some(m) => m,
            None => panic!("Trying to map field, but mesh has not been refined yet")
        });

        for new_leaf_id in 0..new_field.len() {
            let new_local_id = *self.refined_mesh.get_leaf_to_local_map().get(&new_leaf_id).unwrap();

            if new_local_id >= self.previous_local_to_leaf.len() {
                // this cell was not in the previous mesh
                // use its parent
                let parent_local_id = self.refined_mesh.get_cell_parent_id(new_local_id).unwrap();
                let parent_old_leaf_id = self.previous_local_to_leaf[parent_local_id].expect("parent cell was in the previous mesh");
                new_field[CellIndex::from(new_leaf_id)] = field[CellIndex::from(parent_old_leaf_id)];
            } else {
                match self.previous_local_to_leaf[new_local_id] {
                    Some(previous_leaf_id) => {
                        // this cell was in the previous mesh
                        // just copy its value
                        new_field[CellIndex::from(new_leaf_id)] = field[CellIndex::from(previous_leaf_id)];
                    },
                    None => {
                        // cell new id was not in the mesh
                        // use the parent cell, assumes it is in the mesh
                        let parent_local_id = self.refined_mesh.get_cell_parent_id(new_local_id).unwrap();
                        let parent_old_leaf_id = self.previous_local_to_leaf[parent_local_id].expect("parent cell was in the previous mesh");
                        new_field[CellIndex::from(new_leaf_id)] = field[CellIndex::from(parent_old_leaf_id)];
                    }
                }
            }
        }

        new_field.update();

        new_field
    }

}

