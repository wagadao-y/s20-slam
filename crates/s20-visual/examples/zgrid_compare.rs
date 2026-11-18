//! The refinement's z-buffers on the GPU against the CPU: the share of cells with the same point,
//! and the time of each.
//!
//! cargo run --release -p s20-visual --example zgrid_compare -- <ImgPose.txt> <calibration.yaml> <cloud .laz> [images]

use anyhow::Result;
use s20_visual::anchors::{Cloud, Pose, zgrid};
use s20_visual::camera::Fisheye;
use s20_visual::pairs::read_image_poses;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let poses = read_image_poses(Path::new(&a[1]))?;
    let cal = s20_io::calib::load(Path::new(&a[2]))?;
    let cloud = Cloud::load(Path::new(&a[3]), 12)?;
    let count: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(32);
    let cams = [Fisheye::from(&cal.left), Fisheye::from(&cal.right)];
    let step = (poses.names.len() / count).max(1);
    let views: Vec<(Pose, &Fisheye)> = (0..poses.names.len())
        .step_by(step)
        .take(count)
        .map(|i| {
            let side = usize::from(poses.names[i].contains("right"));
            (poses.poses[i], &cams[side])
        })
        .collect();
    let t = Instant::now();
    let cpu: Vec<_> = views.iter().map(|(p, c)| zgrid(&cloud, p, c)).collect();
    let cpu_s = t.elapsed().as_secs_f64();
    // twice: the first sends the cloud and builds the pipelines
    let _ = s20_visual::zgrid_gpu::zgrids(&cloud, &views);
    let t = Instant::now();
    let (gpu, on) = s20_visual::zgrid_gpu::zgrids(&cloud, &views);
    let gpu_s = t.elapsed().as_secs_f64();
    let (mut same, mut filled, mut total) = (0usize, 0usize, 0usize);
    for (c, g) in cpu.iter().zip(&gpu) {
        let (ci, gi) = (c.cells(), g.cells());
        for (x, y) in ci.iter().zip(gi) {
            total += 1;
            filled += usize::from(*x >= 0);
            same += usize::from(x == y);
        }
    }
    println!(
        "{} images: CPU {cpu_s:.2} s, {on} {gpu_s:.2} s; cells the same {:.4} % ({} of {}, {} with a point)",
        views.len(),
        100. * same as f64 / total as f64,
        same,
        total,
        filled
    );
    Ok(())
}
