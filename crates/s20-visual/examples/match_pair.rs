//! Match two images as the pose refinement does, for comparison with OpenCV:
//! `match_pair <data dir> <image a> <image b> <calibration.yaml> <out.json> [max size] [features]`.

use anyhow::{Context, Result};
use s20_visual::{camera::Fisheye, image, matching, sift};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (data, ia, ib, calib, out) = (
        Path::new(&a[1]),
        &a[2],
        &a[3],
        Path::new(&a[4]),
        Path::new(&a[5]),
    );
    let max_size: usize = a.get(6).map(|s| s.parse()).transpose()?.unwrap_or(1800);
    let features: usize = a.get(7).map(|s| s.parse()).transpose()?.unwrap_or(8000);
    let cal = s20_io::calib::load(calib)?;
    let cam = |name: &str| -> Fisheye {
        if name.starts_with("left") {
            (&cal.left).into()
        } else {
            (&cal.right).into()
        }
    };
    let t0 = Instant::now();
    // the keypoints' pixels, rays and descriptors
    type Features = (Vec<[f64; 2]>, Vec<[f64; 3]>, Vec<[u8; 128]>);
    let extract = |name: &str| -> Result<Features> {
        let jpeg = s20_io::dataset::read_image(&data.join("images").join(name))
            .with_context(|| name.to_string())?;
        let full = image::decode(&jpeg)?;
        let s = (max_size as f64 / full.w.max(full.h) as f64).min(1.);
        let (w, h) = (
            ((full.w as f64) * s).round() as usize,
            ((full.h as f64) * s).round() as usize,
        );
        let small = image::resize_area(&full, w, h);
        let (kps, desc) = sift::detect(&small, features);
        let c = cam(name);
        let (sx, sy) = (w as f64 / full.w as f64, h as f64 / full.h as f64);
        let mut uv = Vec::new();
        let mut rays = Vec::new();
        let mut d = Vec::new();
        for (k, ds) in kps.iter().zip(desc) {
            let p = [(k.x as f64 + 0.5) / sx - 0.5, (k.y as f64 + 0.5) / sy - 0.5];
            let r = c.ray(p);
            if r[2] > 80f64.to_radians().cos() {
                uv.push(p);
                rays.push(r);
                d.push(ds);
            }
        }
        Ok((uv, rays, d))
    };
    let (ua, ra, da) = extract(ia)?;
    let t1 = Instant::now();
    let (ub, rb, db) = extract(ib)?;
    let t2 = Instant::now();
    let pairs = matching::mutual_ratio(&da, &db, 0.8);
    let t3 = Instant::now();
    let na: Vec<[f64; 2]> = pairs
        .iter()
        .map(|&(i, _)| [ra[i][0] / ra[i][2], ra[i][1] / ra[i][2]])
        .collect();
    let nb: Vec<[f64; 2]> = pairs
        .iter()
        .map(|&(_, j)| [rb[j][0] / rb[j][2], rb[j][1] / rb[j][2]])
        .collect();
    let focal = (cal.left.a11 + cal.right.a11) / 2.;
    let mask = matching::ransac_fundamental(&na, &nb, 3. / focal, 0.999, 10000);
    let t4 = Instant::now();
    let kept: Vec<(usize, usize)> = pairs
        .iter()
        .zip(&mask)
        .filter(|(_, m)| **m)
        .map(|(p, _)| *p)
        .collect();
    println!(
        "features {} {} | mutual ratio matches {} | RANSAC inliers {} | time: image a {:.2}s, image b {:.2}s, match {:.2}s, ransac {:.2}s",
        ua.len(),
        ub.len(),
        pairs.len(),
        kept.len(),
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        (t3 - t2).as_secs_f64(),
        (t4 - t3).as_secs_f64()
    );
    let json = serde_json::json!({
        "uv_a": kept.iter().map(|&(i, _)| ua[i]).collect::<Vec<_>>(),
        "uv_b": kept.iter().map(|&(_, j)| ub[j]).collect::<Vec<_>>(),
        "features_a": ua.len(), "features_b": ub.len(), "mutual": pairs.len(),
    });
    std::fs::write(out, serde_json::to_vec(&json)?)?;
    Ok(())
}
