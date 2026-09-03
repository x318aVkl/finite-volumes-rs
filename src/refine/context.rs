use std::{io::Read, ops::AddAssign, sync::OnceLock};

use mpi::{topology::SimpleCommunicator, traits::{Communicator, Equivalence}};

use crate::{Field, Mesh, Vector, core::mesh::{CellIndex, geometry}};



pub struct RefinementContext<const DIM: usize> {
    pub(super) grid: p4est::grid::Grid<()>,
    pub(super) mpi_comm: SimpleCommunicator,
}


impl<const DIM: usize> Clone for RefinementContext<DIM> {
    fn clone(&self) -> Self {
        Self {
            grid: self.grid.clone(),
            mpi_comm: self.mpi_comm.duplicate(),
        }
    }
}


static INITIALIZED: OnceLock<bool> = OnceLock::new();

pub fn initialize(world: &SimpleCommunicator) {
    INITIALIZED.get_or_init(|| {
        p4est::env::initialize(world);
        true
    });

}


impl<const DIM: usize> RefinementContext<DIM> {
    pub fn read<T: Read>(source: T, mpi_comm: SimpleCommunicator) -> Result<Self, crate::error::Error> {
        assert!(DIM == p4est::consts::DIM);

        // initialize the library if not initialized already
        initialize(&mpi_comm);

        let grid = p4est::grid::Grid::from_su2(source, mpi_comm.duplicate())?;
        Ok(Self {
            grid,
            mpi_comm,
        })
    }

    pub fn partition(&mut self) {
        self.grid.partition();
    }

    pub fn refine_uniform(&mut self) {
        self.grid.refine_uniform();
    }

    pub fn refine<'a, F>(&'a mut self, f: F) where F: Fn(p4est::grid::cell::Cell<'a, ()>) -> bool {
        self.grid.refine(f)
    }

    pub fn coarsen<'a, F>(&'a mut self, f: F) where F: Fn([p4est::grid::cell::Cell<'_, ()>; 4]) -> bool {
        self.grid.coarsen(f);
    }

    pub fn balance(&mut self) {
        self.grid.balance();
    }
}

pub fn transfer_adapt<T, FC, FR, const DIM: usize>(
    old: &RefinementContext<DIM>, 
    old_data: &[T], 
    new: &RefinementContext<DIM>,
    new_data: &mut [T],
    coarsening_function: FC,
    refining_function: FR,
) -> Result<(), crate::error::Error> where T: Clone + Default,
FC: FnMut([(p4est::grid::cell::Cell<'_, ()>, &T); 4], (p4est::grid::cell::Cell<'_, ()>, &mut T)),
FR: FnMut((p4est::grid::cell::Cell<'_, ()>, &T), [(p4est::grid::cell::Cell<'_, ()>, &mut T); 4])
{
    match p4est::grid::transfer::transfer_data_custom_adapt(&old.grid, old_data, &new.grid, new_data, coarsening_function, refining_function) {
        Ok(()) => Ok(()),
        Err(e) => Err(crate::error::Error::RefinementError(e)),
    }
}


pub fn transfer_partition<T, const DIM: usize>(
    old: &RefinementContext<DIM>, 
    old_data: &[T], 
    new: &RefinementContext<DIM>,
    new_data: &mut [T],
) -> Result<(), crate::error::Error> where T: Clone + Default,
{
    match p4est::grid::transfer::transfer_data_custom_partition(&old.grid, old_data, &new.grid, new_data) {
        Ok(()) => Ok(()),
        Err(e) => Err(crate::error::Error::RefinementError(e)),
    }
}


pub fn transfer_field_partition<T, const DIM: usize>(
    old_ctx: &RefinementContext<DIM>, 
    new_ctx: &RefinementContext<DIM>,
    new_mesh: &Mesh<DIM>,
    old_field: Field<T, geometry::Cell, DIM>, 
) -> Result<Field<T, geometry::Cell, DIM>, crate::error::Error> where T: Clone + Default + Equivalence,
{   
    let mut result = Field::from(new_mesh);
    match p4est::grid::transfer::transfer_data_custom_partition(&old_ctx.grid, old_field.raw_data(), &new_ctx.grid, result.raw_data_mut()) {
        Ok(()) => Ok(result),
        Err(e) => Err(crate::error::Error::RefinementError(e)),
    }
}


pub fn transfer_field_adapt<T, G, const DIM: usize>(
    old_ctx: &RefinementContext<DIM>, 
    new_ctx: &RefinementContext<DIM>,
    old_mesh: &Mesh<DIM>,
    new_mesh: &Mesh<DIM>,
    old_field: Field<T, geometry::Cell, DIM>,
    old_gradients: Option<&Field<G, geometry::Cell, DIM>>,
) -> Result<Field<T, geometry::Cell, DIM>, crate::error::Error> 
where T: Clone + Copy + Default + Equivalence + AddAssign + std::ops::Add<T, Output=T> + std::ops::Div<f64, Output = T>,
G: Default + std::ops::Mul<Vector<DIM>, Output = T> + Copy
{
    let mut new_field = Field::from(new_mesh);

    transfer_adapt(
        old_ctx,
        old_field.raw_data(),
        new_ctx,
        new_field.raw_data_mut(),
        |old, new| {
            let mut sum = T::default();
            for (_cell, value) in old.iter() {
                sum += (*value).clone();
            }
            *new.1 = sum / (old.len() as f64);
        },
        |old, new| {
            let old_val = old.1.clone();
            let old_grad = if let Some(g) = old_gradients {
                g[CellIndex::from(old.0.local_id)]
            } else {G::default()};
            for (cell, value) in new {
                let delta = new_mesh.cell(CellIndex::from(cell.local_id)).center() - old_mesh.cell(CellIndex::from(old.0.local_id)).center();
                let delta = old_grad * delta;
                *value = old_val + delta;
            }
        }
    )?;

    Ok(new_field)
}

