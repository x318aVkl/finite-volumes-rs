use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Range;

use crate::Mesh;
use crate::mesh::Geometry;
use crate::mesh::GlobalRelation;
use crate::mesh::MeshGet;
use crate::mesh::Ownership;


use mpi::topology::Communicator as MpiCommunicator;
use mpi::topology::SimpleCommunicator;
use mpi::traits::CommunicatorCollectives;
use mpi::traits::Destination;
use mpi::traits::Equivalence;
use mpi::traits::Source;



pub struct SingleDataCommunicator<'a> {
    mpi_comm: Option<&'a SimpleCommunicator>,
}


struct SendCommunicator {
    
    other_rank: usize,

    send_ids: Vec<usize>,

}

struct RecvCommunicator {
    
    other_rank: usize,
    recv_range: Range<usize>,

}


pub(crate) struct Communicator<'a, G: Geometry<DIM>, const DIM: usize> {


    send_comms: Vec<SendCommunicator>,

    recv_comms: Vec<RecvCommunicator>,

    mpi_comm: Option<&'a SimpleCommunicator>,

    gp: PhantomData<G>,

}


impl<'a, I, E, G: Geometry<DIM, IndexType = I, ElementType<'a> = E>, const DIM: usize> Communicator<'a, G, DIM>
where E: GlobalRelation, Mesh<DIM>: MeshGet<'a, I>, I: From<usize>, usize: From<I> {

    pub fn from_mesh(mesh: &'a Mesh<DIM>) -> Self {

        let comm = match mesh.communicator() {
            Some(v) => v,
            None => return Self {send_comms: vec![], recv_comms: vec![], mpi_comm: mesh.communicator(), gp: PhantomData }
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

        let mut sizes_we_need = vec![0; comm.size() as usize];
        for (r, buff) in send_global_ids.iter() {
            sizes_we_need[*r] = buff.len();
        }

        mpi::request::scope(|scope| {

            // send the sizes
            //let requests: Vec<_> = send_global_ids.iter().map(|(rank, _global_ids)| {
            //    comm.process_at_rank(*rank as i32).immediate_send_with_tag::<_, usize>(scope, sizes.get(rank).expect("contains rank"), 0)
            //}).collect();
            let requests: Vec<_> = (0..(comm.size() as usize)).map(|rank| {
                comm.process_at_rank(rank as i32).immediate_send_with_tag::<_, usize>(scope, &sizes_we_need[rank], 0)
            }).collect();

            // recieve the sizes and build the buffers
            let mut recv_ranks = vec![];
            (0..(comm.size() as usize)).map(|rank| {
                let mut slen = 0;
                comm.process_at_rank(rank as i32).receive_into_with_tag::<usize>(&mut slen, 0);
                if slen > 0 {
                    recv_global_ids.insert(rank, vec![0; slen]);
                    recv_ranks.push(rank);
                }
            }).count();
            // send_local_ids.iter().map(|(rank, _local_ids)| {
            //     let mut slen = 0;
            //     comm.process_at_rank(*rank as i32).receive_into_with_tag::<usize>(&mut slen, 0);
            //     recv_global_ids.insert(*rank, vec![0; slen]);
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
            

        let send_comms: Vec<SendCommunicator> = recv_local_ids.into_iter().map(|(rank, recv_local_ids)| {
            SendCommunicator { other_rank: rank, send_ids: recv_local_ids.clone(), }
        }).collect();

        let recv_comms: Vec<RecvCommunicator> = send_local_ids.into_iter().map(|(rank, send_local_ids)| {
            let recv_start = *send_local_ids.iter().min().unwrap();
            let recv_end = *send_local_ids.iter().max().unwrap();
            RecvCommunicator { other_rank: rank, recv_range: recv_start..(recv_end+1) }
        }).collect();
        
        Self {
            send_comms: send_comms,
            recv_comms: recv_comms,
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

        for comm in &self.send_comms {
            send_buffers.push(vec![T::default(); comm.send_ids.len()]);
        }

        // fill the send buffers
        for (k, comm) in self.send_comms.iter().enumerate() {
            for (i, idx) in comm.send_ids.iter().enumerate() {
                send_buffers[k][i] = data[*idx].clone();
            }
        }

        mpi::request::scope(|scope| {

            // send the data
            let requests = self.send_comms.iter().enumerate().map(|(k, comm)| {
                mpi_comm.process_at_rank(comm.other_rank as i32).immediate_send_with_tag::<_, [T]>(scope, &send_buffers[k], 0)
            }).collect::<Vec<_>>();

            // receive the data
            self.recv_comms.iter().enumerate().map(|(_k, comm)| {
                mpi_comm.process_at_rank(comm.other_rank as i32).receive_into_with_tag::<[T]>(&mut data[comm.recv_range.clone()], 0);
            }).count();

            // wait for the requests
            for req in requests {
                req.wait();
            }
        });

        // all done!
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

    pub fn barrier(&self) {
        match &self.mpi_comm {
            Some(v) => v.barrier(),
            None => {}
        }
    }
}

