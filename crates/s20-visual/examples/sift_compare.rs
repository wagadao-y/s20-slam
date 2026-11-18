//! SIFT on the GPU against the CPU, image by image: the keypoints of each found by the other (the
//! same place within 0.05 px, size within 0.1 %, angle within 0.5 degrees), the descriptors'
//! differences of those, and the time of each.
//!
//! cargo run --release -p s20-visual --example sift_compare -- <data dir> <image>...

use anyhow::Result;
use s20_visual::{image, sift};
use std::time::Instant;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let data = std::path::Path::new(&a[1]);
    let Some(name) = sift::gpu_name() else {
        anyhow::bail!("no GPU for SIFT");
    };
    eprintln!("GPU: {name}");
    for file in &a[2..] {
        let jpeg = s20_io::dataset::read_image(&data.join("images").join(file))?;
        let full = image::decode(&jpeg)?;
        let s = 1800. / full.w.max(full.h) as f64;
        let small = image::resize_area(
            &full,
            (full.w as f64 * s).round() as usize,
            (full.h as f64 * s).round() as usize,
        );
        let t = Instant::now();
        let (kc, dc) = sift::detect_with(&small, 8000, sift::CONTRAST, true);
        let cpu = t.elapsed().as_secs_f64();
        // twice: the first builds the buffers
        let _ = sift::detect_auto(&small, 8000, sift::CONTRAST, true);
        let t = Instant::now();
        let (kg, dg) = sift::detect_auto(&small, 8000, sift::CONTRAST, true);
        let gpu = t.elapsed().as_secs_f64();
        let same = |a: &sift::Keypoint, b: &sift::Keypoint| {
            (a.x - b.x).abs() < 0.05
                && (a.y - b.y).abs() < 0.05
                && (a.size - b.size).abs() < 1e-3 * a.size
                && {
                    let d = (a.angle - b.angle).abs();
                    d.min(360. - d) < 0.5
                }
        };
        // the GPU's keypoints by place, to find the CPU's among them
        let mut by: std::collections::HashMap<(i32, i32), Vec<usize>> = Default::default();
        for (j, k) in kg.iter().enumerate() {
            by.entry((k.x.floor() as i32, k.y.floor() as i32))
                .or_default()
                .push(j);
        }
        let (mut found, mut max_d, mut sum_d, mut n_d) = (0usize, 0u32, 0u64, 0usize);
        for (i, k) in kc.iter().enumerate() {
            let mut hit = None;
            'near: for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(v) = by.get(&(k.x.floor() as i32 + dx, k.y.floor() as i32 + dy))
                        && let Some(&j) = v.iter().find(|&&j| same(k, &kg[j]))
                    {
                        hit = Some(j);
                        break 'near;
                    }
                }
            }
            if let Some(j) = hit {
                found += 1;
                let d: u32 = dc[i]
                    .iter()
                    .zip(&dg[j])
                    .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
                    .sum();
                max_d = max_d.max(d);
                sum_d += d as u64;
                n_d += 1;
            }
        }
        println!(
            "{file}: CPU {} keypoints ({cpu:.3} s), GPU {} ({gpu:.3} s); CPU's found on the GPU {:.1} %; descriptors L1 mean {:.2}, max {max_d}",
            kc.len(),
            kg.len(),
            100. * found as f64 / kc.len().max(1) as f64,
            sum_d as f64 / n_d.max(1) as f64
        );
    }
    Ok(())
}
