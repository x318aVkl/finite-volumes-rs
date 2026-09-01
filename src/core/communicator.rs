use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::ops::Range;

use crate::Mesh;
use crate::core::mesh::Geometry;
use crate::core::mesh::GlobalRelation;
use crate::core::mesh::MeshGet;
use crate::core::mesh::Ownership;


use mpi::topology::Communicator as MpiCommunicator;
use mpi::topology::SimpleCommunicator;
use mpi::traits::CommunicatorCollectives;
use mpi::traits::Destination;
use mpi::traits::Equivalence;
use mpi::traits::Source;



pub struct SingleDataCommunicator<'a> {
    mpi_comm: Option<&'a SimpleCommunicator>,
}


#[derive(Clone)]
struct OneRankCommunicator {
    
    other_rank: usize,

    send_ids: Vec<usize>,

    recv_range: Range<usize>,

}



pub struct Communicator<G: Geometry<DIM>, const DIM: usize> {

    comms: Vec<OneRankCommunicator>,

    mpi_comm: Option<SimpleCommunicator>,

    gp: PhantomData<G>,

}

impl<G: Geometry<DIM>, const DIM: usize> Communicator<G, DIM> {
    pub fn rank(&self) -> usize {
        match &self.mpi_comm {
            Some(m) => m.rank() as usize,
            None => 0,
        }
    }
    pub fn size(&self) -> usize {
        match &self.mpi_comm {
            Some(m) => m.size() as usize,
            None => 1,
        }
    }
}

impl<G: Geometry<DIM>, const DIM: usize> Clone for Communicator<G, DIM> {
    fn clone(&self) -> Self {
        Self {
            comms: self.comms.clone(),
            mpi_comm: match self.mpi_comm.as_ref() {Some(m) => Some(m.duplicate()), None => None},
            gp: self.gp
        }
    }
}



impl<'a, I, G: Geometry<DIM, IndexType = I>, const DIM: usize> From<&Mesh<DIM>> for Communicator<G, DIM>
where Mesh<DIM>: MeshGet<'a, I>, I: From<usize>, usize: From<I> {
    fn from(value: &Mesh<DIM>) -> Self {
        Self::from_mesh(value)
    }
}


impl<'a, I, G: Geometry<DIM, IndexType = I>, const DIM: usize> Communicator<G, DIM>
where Mesh<DIM>: MeshGet<'a, I>, I: From<usize>, usize: From<I> {

    fn from_mesh(mesh: &Mesh<DIM>) -> Self {

        let comm = match mesh.communicator() {
            Some(v) => v.duplicate(),
            None => return Self {comms: vec![], mpi_comm: mesh.communicator_clone(), gp: PhantomData }
        };

        let mut send_local_ids: HashMap<usize, Vec<usize>> = HashMap::new();
        for n in G::size_from_mesh(&mesh)..G::global_size_from_mesh(&mesh) {
            match G::get_from_mesh(&mesh, I::from(n)).ownership() {
                Ownership::Remote(r) => {
                    if send_local_ids.contains_key(&r) {
                        send_local_ids.get_mut(&r).unwrap().push(n);
                    } else {
                        send_local_ids.insert(r, vec![n]);
                    }
                },
                _ => {}
            }
        }

        let mut send_global_ids = send_local_ids.clone();
        for (r, arr) in send_global_ids.iter_mut() {
            if *r >= (comm.size() as usize) {
                panic!("Error in communicator::from_mesh(), mesh requested rank {} but mpi world size {} is smaller or equal", r, comm.size());
            }
            arr.iter_mut().map(|i| *i = G::get_from_mesh(&mesh, I::from(*i)).global_id() as usize).count();
        }

        // communicate to get the recv global ids
        let mut recv_global_ids: HashMap<usize, Vec<usize>> = HashMap::new();

        //let sizes: HashMap<usize, usize> = send_global_ids.iter().map(|(rank, global_ids)| (*rank, global_ids.len())).collect();

        let mut sizes_we_need: HashMap<usize, usize> = HashMap::new();
        for (r, buff) in send_global_ids.iter() {
            sizes_we_need.insert(*r, buff.len());
        }

        let world_size = comm.size() as usize;

        let default = 0;

        mpi::request::scope(|scope| {

            // send the sizes
            let requests: Vec<_> = (0..world_size).map(|rank| {
                let size = sizes_we_need.get(&rank).unwrap_or(&default);
                comm.process_at_rank(rank as i32).immediate_send_with_tag::<_, usize>(scope, size, 0)
            }).collect();

            // let requests: Vec<_> = sizes_we_need.iter().map(|(rank, size)| {
            //     comm.process_at_rank(*rank as i32).immediate_send_with_tag::<_, usize>(scope, size, 0)
            // }).collect();

            // recieve the sizes and build the buffers
            for rank in 0..world_size {
                //let rank = *rank;
                let mut slen = 0;
                comm.process_at_rank(rank as i32).receive_into_with_tag::<usize>(&mut slen, 0);
                if slen > 0 {
                    recv_global_ids.insert(rank, vec![0; slen]);
                }
            }
            // sizes_we_need.iter().map(|(rank, _)| {
            //     let rank = *rank;
            //     let mut slen = 0;
            //     comm.process_at_rank(rank as i32).receive_into_with_tag::<usize>(&mut slen, 0);
            //     if slen > 0 {
            //         recv_global_ids.insert(rank, vec![0; slen]);
            //     }
            // }).count();

            // wait for the requests
            for req in requests {
                req.wait();
            }

            // send the global ids
            let requests: Vec<_> = send_global_ids.iter().map(|(rank, global_ids)| {
                comm.process_at_rank(*rank as i32).immediate_send_with_tag::<_, [usize]>(scope, &global_ids, 0)
            }).collect();

            // recieve them
            let recv_ranks = recv_global_ids.iter().map(|(rank, _)| {*rank}).collect::<Vec<_>>();
            recv_ranks.iter().map(|rank| {
                let mut buffer = recv_global_ids.get_mut(rank).expect("recv global ids contains rank");
                comm.process_at_rank(*rank as i32).receive_into_with_tag::<[usize]>(&mut buffer, 0);
            }).count();

            // wait for the requests
            for req in requests {
                req.wait();
            }

        });

        // build a global to local node map id
        let mut global_to_local = HashMap::<usize, usize>::new();
        for n in 0..G::global_size_from_mesh(&mesh) {
            let gn = G::get_from_mesh(&mesh, I::from(n)).global_id() as usize;

            global_to_local.insert(gn, n);
        }

        let mut recv_local_ids = recv_global_ids.clone();
        for (_rank, ids) in recv_local_ids.iter_mut() {
            for i in ids {
                *i = *global_to_local.get(i).expect("global to local contains node");
            }
        }

        let mut all_required_ranks = BTreeSet::<usize>::new();
        for (rank, _) in send_local_ids.iter() {
            all_required_ranks.insert(*rank);
        }
        for (rank, _) in recv_local_ids.iter() {
            all_required_ranks.insert(*rank);
        }

        let empty = Vec::new();
        let comms: Vec<OneRankCommunicator> = all_required_ranks.into_iter().map(|rank| {
            let recv_local_ids = recv_local_ids.get(&rank).unwrap_or(&empty);
            let send_local_ids = send_local_ids.get(&rank).unwrap_or(&empty);
            let recv_start = send_local_ids.iter().min().cloned().unwrap_or(0);
            let recv_end = send_local_ids.iter().max().cloned().unwrap_or(usize::MAX);
            let recv_end = if recv_end == usize::MAX {recv_start} else {recv_end};
            OneRankCommunicator { other_rank: rank, send_ids: recv_local_ids.clone(), recv_range: recv_start..(recv_end+1) }
        }).collect();
        
        Self {
            comms,
            mpi_comm: Some(comm),
            gp: PhantomData,
        }
    }


    pub(crate) fn collect<T: Equivalence + Default + Clone>(&self, data: &mut [T]) {

        let mpi_comm = match &self.mpi_comm {
            Some(v) => v,
            None => return,
        };

        // create the send buffers
        let mut send_buffers: Vec<Vec<T>> = vec![];

        for comm in &self.comms {
            send_buffers.push(vec![T::default(); comm.send_ids.len()]);
        }

        // fill the send buffers
        for (k, comm) in self.comms.iter().enumerate() {
            for (i, idx) in comm.send_ids.iter().enumerate() {
                send_buffers[k][i] = data[*idx].clone();
            }
        }

        mpi::request::scope(|scope| {

            // send the data
            let requests = self.comms.iter().enumerate().map(|(k, comm)| {
                mpi_comm.process_at_rank(comm.other_rank as i32).immediate_send_with_tag::<_, [T]>(scope, &send_buffers[k], 0)
            }).collect::<Vec<_>>();

            // receive the data
            self.comms.iter().enumerate().map(|(_k, comm)| {
                mpi_comm.process_at_rank(comm.other_rank as i32).receive_into_with_tag::<[T]>(&mut data[comm.recv_range.clone()], 0);
            }).count();

            // wait for the requests
            for req in requests {
                req.wait();
            }
        });

        // all done!
    }


    pub fn single(&'a self) -> SingleDataCommunicator<'a> {
        SingleDataCommunicator { mpi_comm: self.mpi_comm.as_ref() }
    }

}


impl<'a> SingleDataCommunicator<'a> {

    pub fn from_mpi_comm(mpi_comm: Option<&'a SimpleCommunicator>) -> Self {
        Self { mpi_comm }
    }

    pub fn reduce_add<T: Equivalence + Default + Clone>(&self, value: T) -> T {
        let mpi_comm = match &self.mpi_comm {
            Some(v) => v,
            None => return value,
        };

        let mut recv = T::default();
        mpi_comm.all_reduce_into(&value, &mut recv, mpi::collective::SystemOperation::sum());

        recv
    }

    pub fn reduce_max<T: Equivalence + Default + Clone>(&self, value: T) -> T {
        let mpi_comm = match &self.mpi_comm {
            Some(v) => v,
            None => return value,
        };

        let mut recv = T::default();
        mpi_comm.all_reduce_into(&value, &mut recv, mpi::collective::SystemOperation::max());

        recv
    }

    pub fn reduce_min<T: Equivalence + Default + Clone>(&self, value: T) -> T {
        let mpi_comm = match &self.mpi_comm {
            Some(v) => v,
            None => return value,
        };

        let mut recv = T::default();
        mpi_comm.all_reduce_into(&value, &mut recv, mpi::collective::SystemOperation::min());

        recv
    }

    pub fn barrier(&self) {
        match &self.mpi_comm {
            Some(v) => v.barrier(),
            None => {}
        }
    }
}

