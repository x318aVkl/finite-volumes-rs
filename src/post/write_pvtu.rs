use mpi::traits::Communicator;

use crate::core::field::Field;
use crate::core::mesh::{CellIndex, NodeIndex, geometry};
use crate::{Mesh, Vector};

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};



fn file_without_directory<'a>(filepath: &'a str) -> &'a str {
    let mut fpl = filepath.len() - 1;
    let chars = filepath.chars().collect::<Vec<_>>();
    while fpl > 0 {
        if (chars[fpl] == '/') || (chars[fpl] == '\\') {fpl += 1; break;}
        fpl -= 1;
    }
    &filepath[fpl..]
}


pub trait PvtuGetCellWise {
    type Output;
    fn get_cell_value(&self, cell: CellIndex) -> Self::Output;
}

pub struct PvtuWriter<'a, const DIM: usize> {
    scalar_data: Vec<(String, Box<dyn PvtuGetCellWise<Output = f64> + 'a>)>,
    vector_data: Vec<(String, Box<dyn PvtuGetCellWise<Output = Vector<DIM>> + 'a>)>,
    mesh: &'a Mesh<DIM>,
}

impl<'a, const DIM: usize> PvtuWriter<'a, DIM> {
    pub fn new(mesh: &'a Mesh<DIM>) -> Self {
        PvtuWriter { scalar_data: vec![], vector_data: vec![], mesh }
    }

    pub fn with_scalar(mut self, name: &str, data: impl PvtuGetCellWise<Output = f64> + 'a) -> Self {
        self.scalar_data.push((name.to_string(), Box::new(data)));
        self
    }

    pub fn with_vector(mut self, name: &str, data: impl PvtuGetCellWise<Output = Vector<DIM>> + 'a) -> Self {
        self.vector_data.push((name.to_string(), Box::new(data)));
        self
    }

    pub fn write(self, filepath: &'a str) -> Result<(), Box<dyn std::error::Error>> {
        self.write_pvtu(filepath)
    }
}


impl<'a, const DIM: usize> PvtuWriter<'a, DIM> {

    fn write_pvtu_parent_file (
        &'a self,
        fileprefix: &str,
        world_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {


        let filepath = format!("{}.pvtu", fileprefix);

        let file = File::create(filepath)?;
        let mut writer = BufWriter::new(file);

        
        writer.write("<VTKFile type=\"PUnstructuredGrid\">\n".as_bytes())?;


        writer.write("  <PUnstructuredGrid GhostLevel=\"0\">\n".as_bytes())?;

        if (self.scalar_data.len() > 0) || (self.vector_data.len() > 0) {
            writer.write("    <PCellData>\n".as_bytes())?;

            for i in 0..self.scalar_data.len() {
                write!(writer, "      <PDataArray Name=\"{}\" type=\"Float32\"/>\n", self.scalar_data[i].0)?;
            }

            for i in 0..self.vector_data.len() {
                write!(writer, "      <PDataArray Name=\"{}\" type=\"Float32\" NumberOfComponents=\"3\"/>\n", self.vector_data[i].0)?;
            }

            writer.write("    </PCellData>\n".as_bytes())?;
        }

        writer.write("    <PPoints>\n".as_bytes())?;
        writer.write("      <PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>\n".as_bytes())?;
        writer.write("    </PPoints>\n".as_bytes())?;

        let prefix_without_dir = file_without_directory(fileprefix);

        for i in 0..world_size {
            writer.write(format!("    <Piece Source=\"{}_{}.vtu\"/>\n", prefix_without_dir, i).as_bytes())?;
        }

        writer.write("  </PUnstructuredGrid>\n".as_bytes())?;

        writer.write("</VTKFile>\n".as_bytes())?;

        Ok(())
    }

}


struct VtuMeshData<const DIM: usize> {
    cell_faces: Vec<usize>,
    cell_faces_starts: Vec<usize>,
    cell_nodes: Vec<usize>,
    cell_nodes_starts: Vec<usize>,
    nodes: Vec<Vector<DIM>>,
}


impl<const DIM: usize> VtuMeshData<DIM> {
    fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn n_cells(&self) -> usize {
        self.cell_faces_starts.len() - 1
    }
}


fn collect_vtu_mesh_data<'a, const DIM: usize>(
    mesh: &'a Mesh<DIM>,
) -> VtuMeshData<DIM> {

    let mut cell_faces = vec![];
    let mut cell_faces_starts = vec![0];
    let mut cell_nodes= vec![];
    let mut cell_nodes_starts= vec![0];
    let mut nodes = vec![];

    let mut added_nodes: HashMap<NodeIndex, usize> = HashMap::new();

    for cell in mesh.iter_cells() {
        if !cell.owned() {continue;}

        // add the cell id
        //cell_local_ids.push(usize::from(cell.id()));

        cell_faces.push(cell.faces().len());

        let mut cni = HashSet::new();

        // collect the cell faces and their nodes
        for f in cell.faces() {
            let face = mesh.face(*f);

            cell_faces.push(face.nodes().len());

            for n in face.nodes() {
                let n = *n;

                let nl = if added_nodes.contains_key(&n) {
                    *added_nodes.get(&n).unwrap()
                } else {
                    added_nodes.insert(n, nodes.len());
                    nodes.push(mesh.node(n).center());
                    nodes.len() - 1
                };

                cell_faces.push(nl);
                cni.insert(nl);
            }
        }

        for i in cni {
            cell_nodes.push(i);
        }
        cell_nodes_starts.push(cell_nodes.len());

        cell_faces_starts.push(cell_faces.len());
    }


    VtuMeshData { cell_faces, cell_faces_starts, cell_nodes, cell_nodes_starts, nodes }
}




fn write_polyhedra<const DIM: usize, W: Write>(
    mesh_data: &VtuMeshData<DIM>,
    writer: &mut BufWriter<W>
) -> Result<(), Box<dyn std::error::Error>> {

    writer.write("        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n".as_bytes())?;
    for c in 0..mesh_data.n_cells() {
        let cs = mesh_data.cell_nodes_starts[c];
        let ce = mesh_data.cell_nodes_starts[c+1];

        for i in cs..ce {
            write!(writer, "{} ", mesh_data.cell_nodes[i])?;
        }
        write!(writer, "\n")?;
    }
    writer.write("        </DataArray>\n".as_bytes())?;

    writer.write("        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n".as_bytes())?;
    for c in 0..mesh_data.n_cells() {
        write!(writer, "{} ", mesh_data.cell_nodes_starts[c+1])?;
    }
    write!(writer, "\n")?;
    writer.write("        </DataArray>\n".as_bytes())?;

    writer.write("        <DataArray type=\"Int64\" Name=\"types\" format=\"ascii\">\n".as_bytes())?;
    for _n in 0..mesh_data.n_cells() {
        write!(writer, "42 ")?;
    }
    write!(writer, "\n")?;
    writer.write("        </DataArray>\n".as_bytes())?;
    writer.write("        <DataArray type=\"Int64\" Name=\"faces\" format=\"ascii\">\n".as_bytes())?;
    for c in 0..mesh_data.n_cells() {
        let cs = mesh_data.cell_faces_starts[c];
        let ce = mesh_data.cell_faces_starts[c+1];

        for i in cs..ce {
            write!(writer, "{} ", mesh_data.cell_faces[i])?;
        }
        write!(writer, "\n")?;
    }
    writer.write("        </DataArray>\n".as_bytes())?;
    writer.write("        <DataArray type=\"Int64\" Name=\"faceoffsets\" format=\"ascii\">\n".as_bytes())?;
    for c in 0..mesh_data.n_cells() {
        //let cs = mesh_data.cell_faces_starts[c];
        let ce = mesh_data.cell_faces_starts[c+1];

        write!(writer, "{} ", ce)?;
    }
    write!(writer, "\n")?;
    writer.write("        </DataArray>\n".as_bytes())?;

    Ok(())
}

fn write_polygons<const DIM: usize, W: Write>(
    mesh_data: &VtuMeshData<DIM>,
    writer: &mut BufWriter<W>
) -> Result<(), Box<dyn std::error::Error>> {

    writer.write("        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n".as_bytes())?;
    for c in 0..mesh_data.n_cells() {

        let mut cn: Vec<usize> = vec![];
        let cs = mesh_data.cell_faces_starts[c];
        //let ce = mesh_data.cell_faces_starts[c+1];

        let cfs = mesh_data.cell_faces[cs];
        for fi in 0..cfs {
            let fo = if fi == (cfs - 1) {0} else {fi + 1};

            let i = cs + 1 + fi * 3;
            let o = cs + 1 + fo * 3;
            let ni0 = mesh_data.cell_faces[i+1];
            let ni1 = mesh_data.cell_faces[i+2];

            let no0 = mesh_data.cell_faces[o+1];
            let no1 = mesh_data.cell_faces[o+2];
            
            if (ni0 == no0) || (ni0 == no1) {
                cn.push(ni1);
            } else {
                cn.push(ni0);
            }
        }

        for ni in cn {
            write!(writer, "{} ", ni)?;
        }
        write!(writer, "\n")?;
    }
    writer.write("        </DataArray>\n".as_bytes())?;

    writer.write("        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n".as_bytes())?;
    for c in 0..mesh_data.n_cells() {
        write!(writer, "{} ", mesh_data.cell_nodes_starts[c+1])?;
    }
    write!(writer, "\n")?;
    writer.write("        </DataArray>\n".as_bytes())?;

    writer.write("        <DataArray type=\"Int64\" Name=\"types\" format=\"ascii\">\n".as_bytes())?;
    for _n in 0..mesh_data.n_cells() {
        write!(writer, "7 ")?;
    }
    write!(writer, "\n")?;
    writer.write("        </DataArray>\n".as_bytes())?;


    Ok(())
}



impl<'a, const DIM: usize> PvtuWriter<'a, DIM> {

    fn write_vtu_file(
        &'a self,
        fileprefix: &str,
        world_rank: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {

        let filepath = format!("{}_{}.vtu", fileprefix, world_rank);

        let file = File::create(filepath)?;
        let mut writer = BufWriter::new(file);


        writer.write("<VTKFile type=\"UnstructuredGrid\">\n".as_bytes())?;


        writer.write("  <UnstructuredGrid>\n".as_bytes())?;


        let mesh_data= collect_vtu_mesh_data(self.mesh);


        writer.write(format!("    <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">\n", mesh_data.n_nodes(), mesh_data.n_cells()).as_bytes())?;
        
        writer.write("      <Points>\n".as_bytes())?;
        writer.write("        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n".as_bytes())?;
        for n in mesh_data.nodes.iter() {
            for i in 0..n.len() {
                write!(writer, "{} ", n[i])?;
            }
            for _ in n.len()..3 {
                write!(writer, "0 ")?;
            }
        }
        write!(writer, "\n")?;
        writer.write("        </DataArray>\n".as_bytes())?;
        writer.write("      </Points>\n".as_bytes())?;

        writer.write("      <Cells>\n".as_bytes())?;

        if DIM == 2 {
            write_polygons(&mesh_data, &mut writer)?;
        } else if DIM == 3{
            write_polyhedra(&mesh_data, &mut writer)?;
        } else {
            panic!("invalid dimension {} for write_vtu_file", DIM);
        }
        
        writer.write("      </Cells>\n".as_bytes())?;


        if (self.scalar_data.len() > 0) || (self.vector_data.len() > 0) {
            writer.write("    <CellData>\n".as_bytes())?;

            for i in 0..self.scalar_data.len() {
                write!(writer, "      <DataArray Name=\"{}\" type=\"Float32\">\n", self.scalar_data[i].0)?;

                for c in 0..mesh_data.n_cells() {
                    write!(writer, "{} ", self.scalar_data[i].1.get_cell_value(CellIndex::from(c)))?;
                }
                writer.write("\n".as_bytes())?;

                write!(writer, "      </DataArray>\n")?;
            }

            for i in 0..self.vector_data.len() {
                write!(writer, "      <DataArray Name=\"{}\" type=\"Float32\" NumberOfComponents=\"3\">\n", self.vector_data[i].0)?;

                for c in 0..mesh_data.n_cells() {
                    let vi = self.vector_data[i].1.get_cell_value(CellIndex::from(c));
                    for j in 0..vi.len() {
                        write!(writer, "{} ", vi[j])?;
                    }
                    for _j in vi.len()..3 {
                        write!(writer, "0 ")?;
                    }
                }
                writer.write("\n".as_bytes())?;

                write!(writer, "      </DataArray>\n")?;
            }

            writer.write("    </CellData>\n".as_bytes())?;
        }

        writer.write("    </Piece>\n".as_bytes())?;

        writer.write("  </UnstructuredGrid>\n".as_bytes())?;

        writer.write("</VTKFile>\n".as_bytes())?;
        

        Ok(())
    }

}


fn file_without_extension<'a>(filepath: &'a str) -> &'a str {
    let mut fpl = filepath.len() - 1;
    let chars = filepath.chars().collect::<Vec<_>>();
    while fpl > 0 {
        if chars[fpl] == '.' {break;}
        fpl -= 1;
    }
    &filepath[0..fpl]
}



impl<'a, const DIM: usize> PvtuWriter<'a, DIM> {

    fn write_pvtu(
        &'a self,
        filepath: &'a str,
    ) -> Result<(), Box<dyn std::error::Error>> {

        let (world_rank, world_size) = match self.mesh.communicator() {
            Some(v) => (v.rank() as usize, v.size() as usize),
            None => (0, 1),
        };


        let fileprefix =  file_without_extension(filepath);

        if world_rank == 0 {
            self.write_pvtu_parent_file(fileprefix, world_size)?;
        }

        self.write_vtu_file(fileprefix, world_rank)?;


        Ok(())
    }

}





impl<T, const DIM: usize> PvtuGetCellWise for &Field<T, geometry::Cell, DIM> where T: Clone {
    type Output = T;
    fn get_cell_value(&self, cell: CellIndex) -> Self::Output {
        self[cell].clone()
    }
}