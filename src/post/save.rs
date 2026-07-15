use std::io::BufRead;

use mpi::traits::Equivalence;

use crate::{Field, core::{mesh::{CellIndex, geometry}, traits::FloatBuffered}};

use super::write_pvtu::file_without_extension;


pub fn read_field_from_pvtu<T, const DIM: usize>(
    field: &mut Field<T, geometry::Cell, DIM>,
    field_name: &str,
    pvtu_file: &str,
) -> Result<(), crate::error::Error> 
where T: Clone + Copy + Default + Equivalence + FloatBuffered
{
    let rank = field.communicator().rank();

    let filenoext = file_without_extension(pvtu_file);

    let filepath = format!("{}_{}.vtu", filenoext, rank);


    let file = std::fs::File::open(filepath)?;
    let reader = std::io::BufReader::new(file);
    
    let mut readingcelldata = false;
    let mut readingdata = false;
    let mut data_read = 0;
    let data_to_read = field.len() * T::f64_buffer_size();
    for line in reader.lines() {
        let line = line?;
        if line.contains("<CellData>") {
            readingcelldata = true;
        }
        if line.contains("</CellData>") {
            readingcelldata = false;
        }
        if !readingcelldata {continue;}

        if readingdata {
            let ls = line.split(" ");
            let bfsize = T::f64_buffer_size();
            for v in ls {
                let v = v.trim();
                if v.len() == 0 {
                    continue;
                }
                let v: f64 = v.parse().unwrap();
                let cindex = data_read / bfsize;
                let iidx = data_read % bfsize;
                field[CellIndex::from(cindex)].put_single_in_f64_buffer(iidx, v);
                data_read += 1;
            }
            if data_read >= data_to_read {
                readingdata = false;
            }
        }

        if line.contains("DataArray Name=") {
            let name = line.split("\"").nth(1).unwrap();
            if name == field_name {
                readingdata = true;
            }
        }
        if line.contains("/DataArray") {
            readingdata = false;
        }
    }

    field.update();

    Ok(())
}

