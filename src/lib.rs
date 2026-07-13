pub mod core;
pub mod linalg;
pub mod fvm;
pub mod refine;
pub mod post;

pub use core::{
    mesh::Mesh,
    vector::Vector,
    matrix::Matrix,
    field::Field,
};


pub mod error {
    pub use crate::core::error::Error;
}

pub mod prelude {

    pub use super::Vector;
    pub use super::Matrix;
    pub use super::Mesh;
    pub use super::Field;
    pub use super::core::evaluator::Evaluator;

    pub use super::core::{
        communicator::Communicator,
        mesh::{
            CellIndex,
            FaceIndex,
            FaceNeighbor,
            FaceRef,
            CellRef,
        },
        traits::*,
    };
    pub use super::post::{
        PvtuWriter,
    };
    pub use super::linalg::{
        DistributedMatrix,
        DistributedVector,
        preconditioners::{
            self,
            IncompleteCholesky,
            IncompleteLowerUpper,
        },
        solvers::{
            self,
            LinearSolverOptions,
        },
    };

    pub use super::fvm::{
        terms::Term,
        schemes::dynamic::{
            DynamicSchemeSet,
            SchemeType,
        },
        bcs::BoundaryCondition,
        bcs::StandardBoundaryCondition,
    };

    pub use super::core::mesh::geometry;

    pub use mpi::traits::Communicator as MpiCommunicatorTrait;
    pub use mpi::topology::SimpleCommunicator as MpiCommunicator;

}

