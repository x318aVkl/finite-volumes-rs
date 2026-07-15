use crate::Mesh;




pub fn square() -> Result<Mesh<2>, Box<dyn std::error::Error>> {
    let raw_data = include_str!("examples/square.su2");
    Mesh::read_su2(
        std::io::Cursor::new(raw_data)
        , None
    )
}



