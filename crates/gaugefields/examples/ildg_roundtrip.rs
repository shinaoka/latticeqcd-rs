use gaugefields::{cold_su3, read_ildg, write_ildg, LatticeShape4};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!(
        "gaugefields-ildg-roundtrip-{}.ildg",
        std::process::id()
    ));
    let links = cold_su3(LatticeShape4::new([2, 2, 2, 2])?)?;
    write_ildg(&path, &links)?;
    let loaded = read_ildg(&path)?;
    assert_eq!(loaded.lattice(), links.lattice());
    assert_eq!(loaded.host_view()?.link(0, 0)?.trace().re, 3.0);
    std::fs::remove_file(path)?;
    println!("ILDG round trip preserved a 2x2x2x2 cold SU(3) configuration");
    Ok(())
}
