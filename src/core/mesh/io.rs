use std::{collections::{HashMap, HashSet}, io::{BufRead, Read, SeekFrom, Write}};

use mpi::topology::SimpleCommunicator;

use crate::{Mesh, core::Sparsity, Vector, core::mesh::{FaceIndex, NodeIndex, Ownership}};



impl<const DIM: usize> Mesh<DIM> {
    pub fn write<T: Write>(&self, writer: T) -> Result<(), std::io::Error> {
        let mut w = writer;

        write!(w, "NDIME={}\n", DIM)?;

        write!(w, "NPATCH={}\n", self.patch_name_ids.len())?;
        for local_id in 0..self.patch_name_ids.len() {
            let (name, bid) = &self.patch_name_ids[local_id];
            let (fstart, len) = self.patch_fstart_len[local_id];
            write!(w, "{} {} {} {}\n", name, bid.0, fstart, len)?;
        }

        write!(w, "NNODES={}\n", self.n_total_nodes())?;
        for n in 0..self.n_total_nodes() {
            let n = self.nodes[n];

            n.write_raw_str(&mut w)?;

            write!(w, "\n")?;
        }

        write!(w, "NFACES={}\n", self.n_total_faces())?;
        for face in self.iter_all_faces() {
            write!(w, "{}", face.n_nodes())?;
            for n in face.nodes() {
                write!(w, " {}", usize::from(*n))?;
            }
            match face.boundary() {
                Some(v) => write!(w, " {}", v.0),
                None => write!(w, " -")
            }?;
            
            write!(w, " {}", face.global_id())?;
            match face.ownership() {
                Ownership::Owned => write!(w, " -"),
                Ownership::Remote(r) => write!(w, " {}", r)
            }?;

            write!(w, "\n")?;
        }

        write!(w, "NCELLS={}\n", self.n_total_cells())?;
        for cell in self.iter_all_cells() {
            write!(w, "{}", cell.n_faces())?;
            for f in cell.faces() {
                write!(w, " {}", usize::from(*f))?;
            }
            write!(w, " {}", cell.global_id())?;
            match cell.ownership() {
                Ownership::Owned => write!(w, " -"),
                Ownership::Remote(r) => write!(w, " {}", r)
            }?;
            write!(w, "\n")?;
        }

        Ok(())
    }

    pub fn read<T: Read>(source: T, mpi_comm: Option<SimpleCommunicator>) -> Result<Self, Box<dyn std::error::Error>> {
        let reader = std::io::BufReader::new(source);

        let mut mesh = Mesh::new(mpi_comm);

        let mut section = "none".to_string();

        let mut line_id: usize = 0;
        let mut nodes_to_read = 0;
        let mut faces_to_read = 0;
        let mut cells_to_read = 0;
        let mut patch_to_read = 0;
        for line in reader.lines().map_while(Result::ok) {
            line_id += 1;
            let ls = line.trim();

            if ls.len() == 0 {continue}

            if ls.chars().nth(0) == Some('N') {
                let mut ls = ls.split("=");
                section = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.to_string();
                
                let val: usize = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.parse()?;

                if section == "NDIME" {
                    if val != DIM {
                        return Err(Box::new(crate::error::Error::WrongMeshFileDimension(val)));
                    }
                } else if section == "NNODES" {
                    nodes_to_read = val;
                } else if section == "NFACES" {
                    faces_to_read = val;
                } else if section == "NCELLS" {
                    cells_to_read = val;
                } else if section == "NPATCH" {
                    patch_to_read = val;
                }
                
                continue;
            }

            if section == "none" {
                continue;
            }

            if section == "NNODES" {

                if nodes_to_read == 0 {continue}

                // read a node
                let node: Vector<DIM> = Vector::from_raw_str(ls)?;
                mesh.add_node(node);

                nodes_to_read -= 1;
            } else if section == "NFACES" {
                if faces_to_read == 0 {continue}

                let mut ls = ls.split(" ");
                let size: usize = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.parse()?;

                let mut nodes = [NodeIndex::from(0); 64];
                if size > 64 {
                    return Err(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }));
                }
                for i in 0..size {
                    let node: usize = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.parse()?;
                    nodes[i] = NodeIndex::from(node);
                }
                let tag = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?;
                let boundary = if tag == "-" {
                    None
                } else {
                    Some(tag.parse()?)
                };

                let gid: u32 = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.parse()?;
                
                let tag = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?;
                let ownership = if tag == "-" {
                    Ownership::Owned
                } else {
                    Ownership::Remote(tag.parse()?)
                };

                mesh.add_face(&nodes[0..size], boundary, ownership, Some(gid));

                faces_to_read -= 1;
            } else if section == "NCELLS" {
                if cells_to_read == 0 {continue}

                let mut ls = ls.split(" ");
                let size: usize = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.parse()?;

                let mut faces = [FaceIndex::from(0); 64];
                if size > 64 {
                    return Err(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }));
                }
                for i in 0..size {
                    let face: usize = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.parse()?;
                    faces[i] = FaceIndex::from(face);
                }

                let gid: u32 = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.parse()?;
                
                let tag = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?;
                let ownership = if tag == "-" {
                    Ownership::Owned
                } else {
                    Ownership::Remote(tag.parse()?)
                };

                mesh.add_cell(&faces[0..size], ownership, Some(gid));

                cells_to_read -= 1;
            } else if section == "NPATCH" {
                if patch_to_read == 0 {continue}

                let mut ls = ls.split(" ");

                let name = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?;

                let bid: u16 = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.parse()?;

                let fstart: usize = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.parse()?;

                let len: usize = ls.nth(0).ok_or(Box::new(crate::error::Error::MeshReadError { line: line_id - 1 }))?.parse()?;

                mesh.add_patch(bid, name, Some((FaceIndex::from(fstart), len)))?;

                patch_to_read -= 1;
            }

        }

        // if patches are empty, add a default patch
        if mesh.patch_fstart_len.len() == 0 {
            mesh.add_patch(0, "default", None)?;
        }

        mesh.compute()?;

        Ok(mesh)
    }
}













impl<const DIM: usize> Mesh<DIM> {


    fn read_su2_nodes<R: std::io::Read>(mesh: &mut Mesh<DIM>, source: &mut R) {
        let reader = std::io::BufReader::new(source);

        let mut node: usize = 0;
        let mut n_nodes: Option<usize> = None;
        for line in reader.lines() {

            let line = line.expect("unable to read line");
            
            match n_nodes {
                Some(v) => {
                    if node == v {
                        return;
                    }

                    let mut n: Vector<DIM> = Vector::new();
                    for i in 0..DIM {
                        n[i] = line.split(" ").nth(i).expect("found value in node").trim().parse().unwrap();
                    }

                    mesh.add_node(n);

                    node += 1;
                }, 
                None => {}
            }

            if line.contains("NPOIN=") {
                n_nodes = Some(line.split("=").nth(1).expect("found number of nodes").trim().parse().unwrap());
            }

        }

    }

    fn read_su2_elements<R: std::io::BufRead>(reader: &mut R) -> Result<Vec<super::vtk::VtkElement>, Box<dyn std::error::Error>> {

        let mut elements: Vec<super::vtk::VtkElement> = Vec::new();

        let mut elem: usize = 0;
        let mut nelem: Option<usize> = None;
        for line in reader.lines() {

            let line = line.expect("unable to read line");
            
            match nelem {
                Some(v) => {
                    if elem == v {
                        break;
                    }

                    let elem_kind: u8 = line.split(" ").nth(0).expect("found element kind").trim().parse()?;

                    let mut elem_nodes: Vec<usize> = line.split(" ").map(|v| v.parse::<usize>().unwrap()).collect();
                    elem_nodes.remove(0);

                    elements.push(super::vtk::VtkElement::from_kind_and_nodes(elem_kind, &elem_nodes)?);

                    elem += 1;
                }, 
                None => {}
            }

            if line.contains("NELEM=") {
                nelem = Some(line.split("=").nth(1).expect("found number of elements").trim().parse()?);
            }

        }

        Ok(elements)
    }


    fn read_su2_boundaries<R: std::io::BufRead>(reader: &mut R, mesh: &mut Mesh<DIM>) -> Result<HashMap<Vec<usize>, u16>, Box<dyn std::error::Error>> {

        let mut face_boundaries: HashMap<Vec<usize>, u16> = HashMap::new();


        let mut current_mark: Option<u16> = None;
        let mut current_nelem: Option<usize> = None;
        let mut current_elem = 0;
        let mut read_mark_elems = false;

        for line in reader.lines() {

            let line = line.expect("unable to read line");
            
            if read_mark_elems {
                match current_nelem {
                    None => {},
                    Some(current_nelem) => {
                        if current_elem < current_nelem {
                            //println!("{} {}", current_elem, current_nelem);

                            let mut elem_nodes: Vec<usize> = line.trim().split(" ").map(|v| v.parse::<usize>().unwrap()).collect();
                            elem_nodes.remove(0);
                            elem_nodes.sort();
        
                            face_boundaries.insert(elem_nodes, current_mark.unwrap());
        
                            current_elem += 1;
                        } else {
                            read_mark_elems = false;
                        }
                    }
                }
            }

            if line.contains("MARKER_TAG=") {
                
                let mark_name = line.split("=").nth(1).expect("found marker name").trim();
                match current_mark {
                    None => {current_mark = Some(0);}
                    Some(v) => {current_mark = Some(v + 1);}
                }

                //println!("mark name = {}", mark_name);

                let mark_id = current_mark.unwrap();

                mesh.add_patch(mark_id, mark_name, None)?;

                current_elem = 0;
            }

            if line.contains("MARKER_ELEMS=") {
                current_nelem = Some(line.split("=").nth(1).expect("found number of elements in marker").trim().parse()?);
                read_mark_elems = true;
            }
            
        }

        Ok(face_boundaries)
    }


    fn compute_vtk_faces(elements: &Vec<super::vtk::VtkElement>, boundaries: Option<&HashMap<Vec<usize>, u16>>) -> (Sparsity<NodeIndex>, Vec<Option<u16>>, Sparsity<FaceIndex>) {
        let mut face_nodes = Sparsity::<NodeIndex>::new();
        let mut elem_faces = Sparsity::<FaceIndex>::new();

        let mut face_hash: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut face_boundaries: Vec<Option<u16>> = vec![];

        // add all the internal faces, and then the boundary faces
        let mut bnd_face_nodes = Sparsity::<NodeIndex>::new();

        let mut face_unique_id: usize = 0;
        let mut face_unique_to_final_id = vec![];
        let mut unique_boundaries: HashSet<u16> = HashSet::new();
        for e in elements {

            let (face_nodes_i, face_starts) = e.faces();

            for i in 0..(face_starts.len() - 1) {
                let fi = &face_nodes_i[face_starts[i]..face_starts[i+1]];

                let mut f_hash = fi.to_vec();
                f_hash.sort();

                let fid = match face_hash.get(&f_hash) {
                    Some(x) => *x,
                    None => {
                        // add the face
                        let id = face_unique_id;
                        face_unique_id += 1;
                        let bnd = match boundaries {
                            Some(boundaries) => {
                                match boundaries.get(&f_hash) {Some(v) => Some(*v), None => None}
                            }, None => None,
                        };
                        face_boundaries.push(bnd);

                        face_hash.insert(f_hash, id);

                        match bnd {
                            Some(bndid) => {
                                unique_boundaries.insert(bndid);
                                face_unique_to_final_id.push(bnd_face_nodes.major_len());
                                for ni in fi {
                                    bnd_face_nodes.push_to_major(NodeIndex::from(*ni));
                                }
                                bnd_face_nodes.close_major();
                            },
                            None => {
                                face_unique_to_final_id.push(face_nodes.major_len());
                                for ni in fi {
                                    face_nodes.push_to_major(NodeIndex::from(*ni));
                                }
                                face_nodes.close_major();
                            }
                        }

                        id
                    }
                };

                elem_faces.push_to_major(FaceIndex::from(fid));
            }

            elem_faces.close_major();
        }

        // Add to the boundary face ids the number of no boundary faces, to put them at the end
        let n_nobnd_faces = face_nodes.major_len();
        // for i in 0..face_unique_to_final_id.len() {
        //     if face_boundaries[i].is_some() {
        //         face_unique_to_final_id[i] += n_nobnd_faces;
        //     }
        // }
        // now reorder the face unique to final id based on the boundary id, so that all boundaries are sequential
        // also rebuilds the boundary face nodes
        let mut new_bnd_face_nodes = Sparsity::new();
        let mut runningfaceid = n_nobnd_faces;
        let mut unique_boundaries: Vec<_> = unique_boundaries.into_iter().collect();
        unique_boundaries.sort();
        for bnd in unique_boundaries {
            for i in 0..face_unique_to_final_id.len() {
                if let Some(bndi) = face_boundaries[i] {
                    if bndi == bnd {
                        let old_bnd_id = face_unique_to_final_id[i];
                        face_unique_to_final_id[i] = runningfaceid;
                        runningfaceid += 1;
                        
                        for j in bnd_face_nodes.major_range(old_bnd_id) {
                            new_bnd_face_nodes.push_to_major(*j);
                        }
                        new_bnd_face_nodes.close_major();
                    }
                }
            }
        }
        let bnd_face_nodes = new_bnd_face_nodes;


        // add the nodes of the boundary faces to the face_nodes
        for i in 0..bnd_face_nodes.major_len() {
            for j in bnd_face_nodes.major_range(i) {
                face_nodes.push_to_major(*j);
            }
            face_nodes.close_major();
        }
        // update the elem faces to have new final ids
        for i in 0..elem_faces.major_len() {
            for k in elem_faces.major_start(i)..elem_faces.major_end(i) {
                let f = elem_faces.flat_index(k);
                *elem_faces.flat_index_mut(k) = FaceIndex::from(face_unique_to_final_id[usize::from(f)]);
            }
        }
        // rebuild the face boundaries
        let old_face_boundaries = face_boundaries;
        let mut face_boundaries = vec![None; old_face_boundaries.len()];
        for i in 0..old_face_boundaries.len() {
            let bnd = old_face_boundaries[i];
            let f = face_unique_to_final_id[i];
            face_boundaries[f] = bnd;
        }

        (face_nodes, face_boundaries, elem_faces)
    }


    pub fn read_su2<R: std::io::BufRead + std::io::Seek>(mut reader: R, mpi_comm: Option<SimpleCommunicator>) -> Result<Mesh<DIM>, Box<dyn std::error::Error>> {

        let mut mesh = Mesh::new(mpi_comm);

        {

            reader.seek(SeekFrom::Start(0))?;

            Mesh::read_su2_nodes(&mut mesh, &mut reader);
        }

        let elements = {

            reader.seek(SeekFrom::Start(0))?;

            Mesh::<DIM>::read_su2_elements(&mut reader)?
        };

        let boundaries = {

            reader.seek(SeekFrom::Start(0))?;

            Mesh::<DIM>::read_su2_boundaries(&mut reader, &mut mesh)?
        };

        {
            let (face_nodes, face_boundaries, elem_faces) = Mesh::<DIM>::compute_vtk_faces(&elements, Some(&boundaries));
            for i in 0..face_nodes.major_len() {
                mesh.add_face(face_nodes.major_range(i), face_boundaries[i], Ownership::Owned, None);
            }
            for i in 0..elem_faces.major_len() {
                mesh.add_cell(elem_faces.major_range(i), Ownership::Owned, None);
            }
        }

        mesh.compute()?;

        Ok(mesh)
    }
}





/// Returns the dimension of a mesh file
/// - Usefull for solvers to determine the problems dimension before calling a generic function
pub fn get_mesh_dimension(filepath: &str) -> Result<usize, crate::error::Error> {
    let file = std::fs::File::open(filepath)?;
    let reader = std::io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;

        if line.contains("NDIME=") {
            let dim = match line.trim().split("=").nth(1).unwrap().trim().parse::<usize>() {Ok(v) => Ok(v), Err(_) => Err(crate::error::Error::ParseError(line.to_string()))}?;
            return Ok(dim);
        }
    }

    Err(crate::error::Error::MeshDimensionReadError)
}



pub fn check_file_extension(filepath: &str, ext: &str) -> bool {
    let fpl = filepath.len();
    if fpl < ext.len() {
        return false;
    }
    &filepath[(fpl - ext.len())..fpl] == ext
}



#[cfg(test)]
mod test {
    use super::*;


    fn check_square_mesh<const DIM: usize>(mesh: &Mesh<DIM>) {
        assert_eq!(mesh.n_nodes(), 121);
        assert_eq!(mesh.n_faces(), 220);
        assert_eq!(mesh.n_cells(), 100);

        let mut total_volume = 0.0;
        for cell in mesh.iter_cells() {
            total_volume += cell.volume();
        }

        assert!((total_volume - 1.0).abs() < f64::EPSILON*10.0);

        let mut total_sf = Vector::new();
        for face in mesh.iter_faces() {
            if face.boundary().is_some() {
                total_sf += face.normal() * face.area();
            }
        }

        assert!(total_sf.norm() < f64::EPSILON*10.0);
    }

    #[test]
    fn read_write_square_mesh() {

        let mesh = super::super::examples::square().unwrap();

        check_square_mesh(&mesh);

        let buffer = vec![];
        let mut writer = std::io::BufWriter::new(buffer);
        mesh.write(&mut writer).unwrap();

        let bytes = writer.into_inner().unwrap();
        let result_string = String::from_utf8(bytes).unwrap();

        let mesh: Mesh<2> = Mesh::read(std::io::BufReader::new(std::io::Cursor::new(result_string.as_str())), None).unwrap();

        check_square_mesh(&mesh);
        
    }

}




