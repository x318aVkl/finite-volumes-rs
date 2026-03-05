

use fvfem::{Matrix, Mesh, Vector, field::Field, mesh::geometry};
use mpi::traits::Communicator as MpiCommunicator;


fn test_read_write_decompose() -> Result<(), fvfem::error::Error> {

    println!("Reading mesh in su2 format");
    let mesh: Mesh<2> = Mesh::read_su2(
        std::io::BufReader::new(
            std::fs::File::open("data/mesh.su2").unwrap()
            )
            , 
            None
        ).unwrap();
    println!("  Read mesh with {} cells", mesh.n_cells());
    
    println!("Writing mesh");
    mesh.write(std::io::BufWriter::new(std::fs::File::create("data/mesh.msh").unwrap())).unwrap();
    
    println!("Writing mesh partitions");
    for (rank, part) in mesh.decompose(4)?.enumerate() {
        let part = part?;

        println!("  Writing part {} with {} cells", rank, part.n_cells());

        part.write(std::io::BufWriter::new(
            std::fs::File::create(format!("data/mesh_{}.msh", rank)).unwrap()
        )).unwrap();
    }

    println!("Reading mesh in own format");
    let mesh: Mesh<2> = Mesh::read(
        std::io::BufReader::new(
            std::fs::File::open("data/mesh.msh").unwrap()
            )
            , 
            None
        ).unwrap();
    println!("  Read mesh with {} cells", mesh.n_cells());

    Ok(())
}



fn test_field_comm() -> Result<(), fvfem::error::Error> {
    let universe = mpi::initialize().ok_or(fvfem::error::Error::MpiInitializeFailed)?;
    let world = universe.world();
    let rank = world.rank() as usize;

    let mesh: Mesh<2> = Mesh::read(std::io::BufReader::new(std::fs::File::open(format!("data/mesh_{}.msh", rank).as_str()).unwrap()), Some(world)).unwrap();

    // Create a field with a scalar f64 value in every cell
    let mut field: Field<Matrix<2, 2>, geometry::Cell, _> = Field::from_mesh(&mesh);

    for n in mesh.iter_cells() {
        field[n.id()] = n.center().outer(n.center());
    }

    field.update();

    let mut part_volume = 0.0;
    for cell in mesh.iter_cells() {
        part_volume += cell.volume();
    }
    println!("rank {} part volume = {:.8}", rank, part_volume);

    mesh.comm().barrier();
    let total_volume = mesh.comm().reduce_add(part_volume);

    if rank == 0 {println!("total volume = {:.8}", total_volume);}

    Ok(())
}

fn main() -> Result<(), fvfem::error::Error> {

    //test_read_write_decompose()?;

    test_field_comm()?;

    Ok(())
}
