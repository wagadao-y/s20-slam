//! Anchored observations of one image pair, for comparison with the Python scripts:
//! `anchor_pair <data dir> <ImgPose.txt> <cloud.laz> <calibration.yaml> <image a> <image b> <out.json>`.

use anyhow::Result;
use s20_visual::anchors::{Cloud, zgrid};
use s20_visual::camera::Fisheye;
use s20_visual::pairs::{View, features, observe, observe_guided, read_image_poses};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (data, imgpose, cloud, calib) = (
        Path::new(&a[1]),
        Path::new(&a[2]),
        Path::new(&a[3]),
        Path::new(&a[4]),
    );
    let (na, nb, out) = (&a[5], &a[6], Path::new(&a[7]));
    let t0 = Instant::now();
    let cloud = Cloud::load(cloud, 12)?;
    let t1 = Instant::now();
    let cal = s20_io::calib::load(calib)?;
    let cams = [Fisheye::from(&cal.left), Fisheye::from(&cal.right)];
    let cam = |n: &str| &cams[s20_visual::run::side(n)];
    let poses = read_image_poses(imgpose)?;
    let ia = poses.names.iter().position(|n| n == na).unwrap();
    let ib = poses.names.iter().position(|n| n == nb).unwrap();
    let fa = features(
        &s20_io::dataset::read_image(&data.join("images").join(na))?,
        cam(na),
        1800,
        8000,
    )?;
    let fb = features(
        &s20_io::dataset::read_image(&data.join("images").join(nb))?,
        cam(nb),
        1800,
        8000,
    )?;
    let t2 = Instant::now();
    let ga = zgrid(&cloud, &poses.poses[ia], cam(na));
    let gb = zgrid(&cloud, &poses.poses[ib], cam(nb));
    let t3 = Instant::now();
    let va = View {
        index: ia,
        pose: poses.poses[ia],
        cam: cam(na),
        grid: &ga,
        features: &fa,
    };
    let vb = View {
        index: ib,
        pose: poses.poses[ib],
        cam: cam(nb),
        grid: &gb,
        features: &fb,
    };
    let focal = (cal.left.a11 + cal.right.a11) / 2.;
    let obs = if std::env::var("GUIDED").is_ok() {
        observe_guided(&va, &vb, &cloud, focal, 40., 80., 0.8, false)
    } else {
        observe(&va, &vb, &cloud, focal, 40.)
    };
    let t4 = Instant::now();
    println!(
        "cloud {} points {:.1}s | features {:.2}s | z-buffers {:.2}s | observe {:.2}s | observations {}",
        cloud.xyz.len(),
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        (t3 - t2).as_secs_f64(),
        (t4 - t3).as_secs_f64(),
        obs.len()
    );
    let json = serde_json::json!({
        "image_a": obs.iter().map(|o| &poses.names[o.a]).collect::<Vec<_>>(),
        "image_b": obs.iter().map(|o| &poses.names[o.b]).collect::<Vec<_>>(),
        "uv_a": obs.iter().map(|o| o.uv_a).collect::<Vec<_>>(),
        "uv_b": obs.iter().map(|o| o.uv_b).collect::<Vec<_>>(),
        "world": obs.iter().map(|o| [o.world.x, o.world.y, o.world.z]).collect::<Vec<_>>(),
        "depth": obs.iter().map(|o| o.depth).collect::<Vec<_>>(),
    });
    std::fs::write(out, serde_json::to_vec(&json)?)?;
    Ok(())
}
