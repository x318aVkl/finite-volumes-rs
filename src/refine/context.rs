use std::{io::Read, sync::OnceLock};

use mpi::{topology::SimpleCommunicator, traits::Communicator};


pub struct RefinementContext<const DIM: usize> {
    pub(super) grid: p4est::grid::Grid<()>,
    pub(super) mpi_comm: SimpleCommunicator,
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


