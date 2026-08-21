use std::io::Read;

use mpi::topology::SimpleCommunicator;




pub struct RefinementContext {
    grid: p4est::grid::Grid<()>,
}


impl RefinementContext {
    pub fn read<T: Read>(source: T, mpi_comm: SimpleCommunicator) -> Result<Self, std::io::Error> {
        let grid = p4est::grid::Grid::from_su2(source, mpi_comm)?;
        Ok(Self {
            grid,
        })
    }
}


