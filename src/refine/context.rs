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

}