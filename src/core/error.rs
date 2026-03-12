

#[derive(Debug)]
pub enum Error {
    SingularMatrix,
    InvalidDimension(usize),
    WrongMeshFileDimension(usize),
    MeshReadError{line: usize},
    MeshDimensionReadError,

    ParseError(String),

    MetisError(metis::Error),
    MetisNewGraphError(metis::NewGraphError),

    MpiInitializeFailed,

    StdIoError(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for Error {}


impl From<metis::Error> for Error {
    fn from(value: metis::Error) -> Self {
        Self::MetisError(value)
    }
}
impl From<metis::NewGraphError> for Error {
    fn from(value: metis::NewGraphError) -> Self {
        Self::MetisNewGraphError(value)
    }
}
impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::StdIoError(value)
    }
}