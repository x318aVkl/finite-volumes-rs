

use finite_volumes::prelude::*;


use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {

    // Mesh file to decompose
    #[arg(short, long)]
    file: String,

    // Number of partitions
    #[arg(short, long)]
    nparts: usize,
}


fn run<const DIM: usize>(filepath: &str, nparts: usize) -> Result<(), finite_volumes::error::Error> {
    
    println!("Decomposing mesh in file {}", filepath);
    println!("  - Reading mesh");
    let mesh: Mesh<DIM> = if finite_volumes::core::mesh::io::check_file_extension(filepath, ".su2") {
        Mesh::read_su2(
        std::io::BufReader::new(
            std::fs::File::open(filepath).unwrap()
            )
            , 
            None
        ).unwrap()
    } else if finite_volumes::core::mesh::io::check_file_extension(filepath, ".msh") {
        Mesh::read(
        std::io::BufReader::new(
            std::fs::File::open(filepath).unwrap()
            )
            , 
            None
        ).unwrap()
    } else {
        panic!("Invalid file extension for file {}", filepath);
    };
    println!("    Read mesh with {} cells", mesh.n_cells());

    let chars = filepath.chars().collect::<Vec<char>>();

    let mut end_id = chars.len() - 1;
    while chars[end_id] != '.' {
        end_id -= 1;
        if end_id == 0 {break;}
    }
    let file_prefix = &filepath[0..end_id];
    
    if !finite_volumes::core::mesh::io::check_file_extension(filepath, ".msh") {
        println!("  - Writing mesh in own format to {}.msh", file_prefix);
        mesh.write(std::io::BufWriter::new(std::fs::File::create(format!("{}.msh", file_prefix).as_str()).unwrap())).unwrap();
    }
    
    println!("  - Writing mesh partitions");
    for (rank, part) in mesh.decompose(nparts)?.enumerate() {
        let part = part?;

        println!("    - Writing part {} with {} cells to file {}_{}.msh", rank, part.n_cells(), file_prefix, rank);

        part.write(std::io::BufWriter::new(
            std::fs::File::create(format!("{}_{}.msh", file_prefix, rank)).unwrap()
        )).unwrap();
    }

    println!("  Done");

    Ok(())
}


fn main() -> Result<(), finite_volumes::error::Error> {
    let args = Args::parse();

    let dim = finite_volumes::core::mesh::io::get_mesh_dimension(&args.file)?;

    match dim {
        1 => run::<1>(&args.file, args.nparts)?,
        2 => run::<2>(&args.file, args.nparts)?,
        3 => run::<3>(&args.file, args.nparts)?,
        _ => panic!("Invalid mesh dimension: {}", dim),
    }

    Ok(())
}
