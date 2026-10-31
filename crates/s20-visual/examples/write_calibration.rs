//! A calibration with solved camera changes baked in, from a camera_refit report's camera object
//! (`{"intrinsic", "raw_mount"}`): `write_calibration <calibration.yaml> <camera.json> <out.yaml>`.

use anyhow::Result;
use s20_visual::calibrate::write;
use s20_visual::camera::Fisheye;
use s20_visual::refine::Cameras;
use std::path::Path;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let cal = s20_io::calib::load(Path::new(&a[1]))?;
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let change = Cameras {
        intrinsic: serde_json::from_value(v["intrinsic"].clone())?,
        mount: serde_json::from_value(v["raw_mount"].clone())?,
    };
    let (text, err) = write(
        &std::fs::read_to_string(&a[1])?,
        &[Fisheye::from(&cal.left), Fisheye::from(&cal.right)],
        &change,
    )?;
    std::fs::write(&a[3], text)?;
    println!("polynomial refit {:.5} / {:.5} px", err[0], err[1]);
    Ok(())
}
