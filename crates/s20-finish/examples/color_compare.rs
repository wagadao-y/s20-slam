//! Two colored clouds of the same points compared: per point the largest difference of its
//! RGB (in 8-bit steps), as mean, quantiles and largest, and the share of points that differ.
//!
//! cargo run --release -p s20-finish --example color_compare -- <a.laz> <b.laz> [<map.laz>]
//!
//! With `map.laz`: the points colored by their difference (gray none, to red at 40 steps and
//! more), to see where the colors differ.

use anyhow::{Result, ensure};
use std::path::Path;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (pa, info) = s20_io::las::read(Path::new(&a[1]))?;
    let (pb, _) = s20_io::las::read(Path::new(&a[2]))?;
    ensure!(pa.len() == pb.len(), "{} and {} points", pa.len(), pb.len());
    let mut d: Vec<f32> = pa
        .iter()
        .zip(&pb)
        .map(|(x, y)| {
            ensure!(x.xyz == y.xyz, "the points differ");
            Ok((0..3)
                .map(|c| (x.rgb[c] as f32 - y.rgb[c] as f32).abs() / 257.)
                .fold(0f32, f32::max))
        })
        .collect::<Result<_>>()?;
    if let Some(map) = a.get(3) {
        let heat: Vec<s20_io::las::LasPoint> = pa
            .iter()
            .zip(&d)
            .map(|(x, &v)| {
                let t = (v / 40.).min(1.);
                let rgb =
                    [0.5 + 0.5 * t, 0.5 * (1. - t), 0.5 * (1. - t)].map(|c| (c * 65535.) as u16);
                s20_io::las::LasPoint { rgb, ..*x }
            })
            .collect();
        // (on the first cloud's grid: its points stay where they are; with a scale per axis, the
        // coarsest of them, which holds every coordinate the finer ones did)
        let scale = info.scale.iter().copied().fold(0., f64::max);
        s20_io::las::write(Path::new(map), &heat, scale, info.offset)?;
    }
    let n = d.len();
    let mean = d.iter().map(|&x| x as f64).sum::<f64>() / n.max(1) as f64;
    let changed = d.iter().filter(|&&x| x >= 0.5).count();
    d.sort_by(f32::total_cmp);
    let q = |f: f64| d[((n - 1) as f64 * f).round() as usize];
    println!(
        "{n} points: RGB difference (8-bit steps) mean {mean:.3}, median {:.2}, 90% {:.2}, 99% {:.2}, 99.9% {:.2}, largest {:.1}; {:.2} % of the points differ by half a step or more",
        q(0.5),
        q(0.9),
        q(0.99),
        q(0.999),
        d[n - 1],
        100. * changed as f64 / n as f64
    );
    Ok(())
}
