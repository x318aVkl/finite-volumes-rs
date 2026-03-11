


/// Vtk file format element types
#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub enum VtkElement {

    Line([usize; 2]) = 3,
    Triangle([usize; 3]) = 5,
    Quad([usize; 4]) = 9,

    Tetrahedron([usize; 4]) = 10,
    Hexahedron([usize; 8]) = 12,
}

/// Vtk error type
#[derive(Debug, Clone, Copy)]
pub enum VtkError {
    InvalidKind(u8)
}

impl std::fmt::Display for VtkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKind(kind) => {
                write!(f, "VtkError::InvalidKind({})", kind)
            }
        }
    }
}

impl std::error::Error for VtkError {
}


impl VtkElement {

    pub fn from_kind_and_nodes(kind: u8, nodes: &[usize]) -> Result<VtkElement, VtkError> {
        if kind == 3 {
            Ok(VtkElement::Line([nodes[0], nodes[1]]))
        } else if kind == 5 {
            Ok(VtkElement::Triangle([nodes[0], nodes[1], nodes[2]]))
        } else if kind == 9 {
            Ok(VtkElement::Quad([nodes[0], nodes[1], nodes[2], nodes[3]]))
        } else if kind == 10 {
            Ok(VtkElement::Tetrahedron([nodes[0], nodes[1], nodes[2], nodes[3]]))
        } else if kind == 12 {
            Ok(VtkElement::Hexahedron([nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7]]))
        } else {
            Err(VtkError::InvalidKind(kind))
        }
    }


    pub fn faces(self) -> (Vec<usize>, Vec<usize>) {

        let nodes;
        let starts;

        match self {
            VtkElement::Line(n) => {
                nodes = vec![n[0], n[1]];
                starts = vec![0, 1, 2];
            },
            VtkElement::Triangle(n) => {
                nodes = vec![
                    n[0], n[1], 
                    n[1], n[2], 
                    n[2], n[0],
                ];
                starts = vec![
                    0, 2, 4, 6,
                ]
            },
            VtkElement::Quad(n) => {
                nodes = vec![
                    n[0], n[1], 
                    n[1], n[2], 
                    n[2], n[3],
                    n[3], n[0],
                ];
                starts = vec![
                    0, 2, 4, 6, 8,
                ]
            },
            VtkElement::Tetrahedron(n) => {
                nodes = vec![
                    n[0], n[2], n[1],
                    n[0], n[1], n[3],
                    n[1], n[2], n[3],
                    n[0], n[3], n[2],
                ];
                starts = vec![
                    0,
                    3,
                    6,
                    9,
                    12,
                ]
            },
            VtkElement::Hexahedron(n) => {
                nodes = vec![
                    n[0], n[3], n[2], n[1],
                    n[4], n[5], n[6], n[7],
                    n[0], n[1], n[5], n[4],
                    n[1], n[2], n[6], n[5],
                    n[2], n[3], n[7], n[6],
                    n[3], n[0], n[4], n[7],
                ];
                starts = vec![
                    0,
                    4,
                    8,
                    12,
                    16,
                    20,
                    24,
                ]
            }
        }


        (nodes, starts)
    }

}

