use std::ops::AddAssign;

use mpi::traits::Equivalence;

use crate::{Field, Mesh, Vector, core::mesh::geometry, fvm::tools::gradients::GradientFrom, refine::context::{RefinementMesh, transfer_field_adapt, transfer_field_partition}};


#[derive(Clone, Copy, Debug)]
enum AMRStage {
    Startup,
    Refinement,
    Partition,
}

pub struct AMRHandler<F, const DIM: usize> {
    old_ctx: RefinementMesh<DIM>,
    new_ctx: RefinementMesh<DIM>,
    old_mesh: Mesh<DIM>,
    new_mesh: Option<Mesh<DIM>>,
    criteria: Field<f64, geometry::Cell, DIM>,

    transfer_operator: Option<F>,
    
    refine_tolerance: f64,
    coarsen_tolerance: f64,
    max_level: u8,
    min_level: u8,
    target_number_of_cells: usize,

    stage: AMRStage,

}



pub struct AMRTransferOperator<'a, const DIM: usize> {
    old_ctx: &'a RefinementMesh<DIM>,
    new_ctx: &'a RefinementMesh<DIM>,
    old_mesh: &'a Mesh<DIM>,
    new_mesh: &'a Mesh<DIM>,
    stage: AMRStage,
}



impl<'a, F, const DIM: usize> AMRHandler<F, DIM> where F: FnMut(&AMRTransferOperator<'a, DIM>) -> Result<(), crate::error::Error> {

    pub fn new(
        refinement: RefinementMesh<DIM>,
        mesh: Mesh<DIM>,
        criteria: Field<f64, geometry::Cell, DIM>,
    ) -> Self {
        let ncells = mesh.n_cells();
        Self {
            old_ctx: refinement.clone(),
            new_ctx: refinement,
            old_mesh: mesh,
            new_mesh: None,
            criteria,
            transfer_operator: None,
            refine_tolerance: 0.9,
            coarsen_tolerance: 0.1,
            max_level: 7,
            min_level: 2,
            target_number_of_cells: ncells,
            stage: AMRStage::Startup,
        }
    }

    pub fn with_transfer(mut self, transfer: F) -> Self {
        self.transfer_operator = Some(transfer);
        self
    }

    pub fn with_tolerances(
        mut self,
        refine_tolerance: f64,
        coarsen_tolerance: f64,
    ) -> Self {
        self.refine_tolerance = refine_tolerance;
        self.coarsen_tolerance = coarsen_tolerance;
        self
    }

    pub fn with_levels(
        mut self,
        max_level: u8,
        min_level: u8,
    ) -> Self {
        self.max_level = max_level;
        self.min_level = min_level;
        self
    }

    pub fn apply(mut self) -> Result<(RefinementMesh<DIM>, Mesh<DIM>), crate::error::Error> 
    where F: 
    {

        self.stage = AMRStage::Refinement;

        self.new_ctx.refine(|cell| {
            let crit = self.criteria[cell.local_id.into()];
            let target_level = (((self.max_level - self.min_level) as f64) * crit).round() as u8 + self.min_level;
            //(crit > self.refine_tolerance) & (cell.level < self.max_level)
            cell.level < target_level
        });
        self.new_ctx.balance();
        self.new_mesh = Some(self.new_ctx.build_mesh()?);

        self.criteria = transfer_field_adapt::<f64, Vector<DIM>, DIM>(
            &self.old_ctx, 
            &self.new_ctx, 
            &self.old_mesh, 
            self.new_mesh.as_ref().unwrap(), 
            self.criteria, 
            None
        )?;

        // apply the user defined field mapping
        if let Some(f) = self.transfer_operator.as_mut() {
            unsafe {    
                let top = AMRTransferOperator {
                    old_ctx: &*(&self.old_ctx as *const RefinementMesh<DIM>),
                    new_ctx: &*(&self.new_ctx as *const RefinementMesh<DIM>),
                    old_mesh: &*(&self.old_mesh as *const Mesh<DIM>),
                    new_mesh: &*(self.new_mesh.as_ref().unwrap() as *const Mesh<DIM>),
                    stage: self.stage,
                };        
                let fp: *mut F = f as *const F as *mut F;
                let fm = &mut *fp;
                fm(&top)?
            }
        }

        // now also coarsen
        self.old_ctx = self.new_ctx.clone();
        self.new_ctx.coarsen(|cells| {
            let mut crit = 0.;
            for cell in cells.iter() {
                let crit_i = self.criteria[cell.local_id.into()];
                crit += crit_i;
            }
            crit /= cells.len() as f64;
            let target_level = (((self.max_level - self.min_level) as f64) * crit).round() as u8 + self.min_level;
            //(crit < self.coarsen_tolerance) & (cells[0].level > self.min_level)
            cells[0].level > target_level
        });
        self.new_ctx.balance();
        self.old_mesh = self.new_mesh.unwrap();
        self.new_mesh = Some(self.new_ctx.build_mesh()?);

        // do not transfer criteria this time since its not needed anymore
        // self.criteria = transfer_field_adapt::<f64, Vector<DIM>, DIM>(
        //     &self.old_ctx, 
        //     &self.new_ctx, 
        //     &self.old_mesh, 
        //     self.new_mesh.as_ref().unwrap(), 
        //     self.criteria, 
        //     None
        // )?;

        // apply the user defined field mapping
        if let Some(f) = self.transfer_operator.as_mut() {
            unsafe {    
                let top = AMRTransferOperator {
                    old_ctx: &*(&self.old_ctx as *const RefinementMesh<DIM>),
                    new_ctx: &*(&self.new_ctx as *const RefinementMesh<DIM>),
                    old_mesh: &*(&self.old_mesh as *const Mesh<DIM>),
                    new_mesh: &*(self.new_mesh.as_ref().unwrap() as *const Mesh<DIM>),
                    stage: self.stage,
                };        
                let fp: *mut F = f as *const F as *mut F;
                let fm = &mut *fp;
                fm(&top)?
            }
        }

        // finally, partition the mesh
        self.stage = AMRStage::Partition;

        self.old_ctx = self.new_ctx.clone();
        self.new_ctx.partition();

        self.old_mesh = self.new_mesh.unwrap();
        self.new_mesh = Some(self.new_ctx.build_mesh()?);

        // apply the user defined field mapping
        if let Some(f) = self.transfer_operator.as_mut() {
            unsafe {    
                let top = AMRTransferOperator {
                    old_ctx: &*(&self.old_ctx as *const RefinementMesh<DIM>),
                    new_ctx: &*(&self.new_ctx as *const RefinementMesh<DIM>),
                    old_mesh: &*(&self.old_mesh as *const Mesh<DIM>),
                    new_mesh: &*(self.new_mesh.as_ref().unwrap() as *const Mesh<DIM>),
                    stage: self.stage,
                };        
                let fp: *mut F = f as *const F as *mut F;
                let fm = &mut *fp;
                fm(&top)?
            }
        }

        Ok((self.new_ctx, self.new_mesh.unwrap()))
    }



}



impl<'a, const DIM: usize> AMRTransferOperator<'a, DIM> {

    pub fn transfer_field<T>(
        &self,
        field: Field<T, geometry::Cell, DIM>,
    ) -> Result<Field<T, geometry::Cell, DIM>, crate::error::Error> 
    where T: Copy + Default + Equivalence + AddAssign + std::ops::Add<T, Output = T> + std::ops::Div<f64, Output=T> + GradientFrom<DIM>,
    <T as GradientFrom<DIM>>::GradientType: Default + std::ops::Mul<Vector<DIM>, Output = T> + Copy
    {
        match self.stage {
            AMRStage::Startup => Ok(field),
            AMRStage::Refinement => {
                transfer_field_adapt::<T, <T as GradientFrom<DIM>>::GradientType, DIM>(
                    self.old_ctx, 
                    self.new_ctx, 
                    self.old_mesh, 
                    self.new_mesh, 
                    field,
                    None
                )
            },
            AMRStage::Partition => {
                transfer_field_partition(
                    &self.old_ctx, 
                    &self.new_ctx, 
                    self.new_mesh, 
                    field
                )
            }
        }
    }


    pub fn transfer_field_linear<T>(
        &self,
        field: Field<T, geometry::Cell, DIM>,
        gradients: &Field<<T as GradientFrom<DIM>>::GradientType, geometry::Cell, DIM>,
    ) -> Result<Field<T, geometry::Cell, DIM>, crate::error::Error> 
    where T: Copy + Default + Equivalence + AddAssign + std::ops::Add<T, Output = T> + std::ops::Div<f64, Output=T> + GradientFrom<DIM>,
    <T as GradientFrom<DIM>>::GradientType: Default + std::ops::Mul<Vector<DIM>, Output = T> + Copy
    {
        match self.stage {
            AMRStage::Startup => Ok(field),
            AMRStage::Refinement => {
                transfer_field_adapt::<T, <T as GradientFrom<DIM>>::GradientType, DIM>(
                    self.old_ctx, 
                    self.new_ctx, 
                    self.old_mesh, 
                    self.new_mesh, 
                    field,
                    Some(gradients)
                )
            },
            AMRStage::Partition => {
                transfer_field_partition(
                    self.old_ctx, 
                    self.new_ctx, 
                    self.new_mesh, 
                    field
                )
            }
        }
    }

}


