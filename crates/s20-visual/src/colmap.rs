//! A recording as a COLMAP model for 3D Gaussian Splatting: the images as they are, each camera as
//! COLMAP's `OPENCV_FISHEYE` (Kannala-Brandt) fitted to the S20 polynomial, the image poses, and
//! the coloured cloud thinned to one point per voxel as the seed points.
//!
//! Kannala-Brandt has odd powers of theta only and no skew, so the fit leaves a fraction of a
//! pixel (the report gives it). Pixels past `max_angle` lie outside the calibration's fit and are
//! masked (`masks/<image>.png`, white where kept, as Spirula Studio reads them).

use crate::anchors::Pose;
use crate::camera::Fisheye;
use crate::pairs::read_image_poses;
use crate::run::side;
use anyhow::{Context, Result, ensure};
use nalgebra::{SMatrix, SVector, UnitQuaternion};
use rayon::prelude::*;
use s20_io::job::Job;
use std::fmt::Write as _;
use std::path::PathBuf;

/// `OPENCV_FISHEYE`: pixel = f * theta (1 + k1 theta^2 + ... + k4 theta^8) + c, in S20 pixels
/// (centres on integers; COLMAP's are on halves).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kb {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k: [f64; 4],
}

impl Kb {
    fn distorted(&self, th: f64) -> f64 {
        let t2 = th * th;
        th * (1. + t2 * (self.k[0] + t2 * (self.k[1] + t2 * (self.k[2] + t2 * self.k[3]))))
    }

    pub fn project(&self, x: [f64; 3]) -> [f64; 2] {
        let rho = x[0].hypot(x[1]);
        let d = self.distorted(rho.atan2(x[2]));
        let (c, s) = if rho > 1e-15 {
            (x[0] / rho, x[1] / rho)
        } else {
            (1., 0.)
        };
        [self.fx * d * c + self.cx, self.fy * d * s + self.cy]
    }

    /// The `cameras.txt` line (COLMAP's pixel centres).
    pub fn colmap_line(&self, id: usize, width: u32, height: u32) -> String {
        let p = [
            self.fx,
            self.fy,
            self.cx + 0.5,
            self.cy + 0.5,
            self.k[0],
            self.k[1],
            self.k[2],
            self.k[3],
        ];
        let p: Vec<String> = p.iter().map(|v| format!("{v:.12e}")).collect();
        format!("{id} OPENCV_FISHEYE {width} {height} {}", p.join(" "))
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct FitError {
    pub max_px: f64,
    pub rms_px: f64,
    /// share of the image's pixels past the fitted angle (masked)
    pub masked: f64,
}

/// A pixel's direction and the pixel.
type Sample = ([f64; 3], [f64; 2]);

/// The directions of a grid of the image's pixels within `max_angle` (radians), with their pixels,
/// and the share of the pixels past it.
fn samples(
    cam: &Fisheye,
    width: u32,
    height: u32,
    max_angle: f64,
    step: usize,
) -> (Vec<Sample>, f64) {
    let edge = cam.radius(max_angle);
    let (mut out, mut total) = (Vec::new(), 0usize);
    for v in (0..height as usize).step_by(step) {
        for u in (0..width as usize).step_by(step) {
            total += 1;
            let uv = [u as f64, v as f64];
            let y = (uv[1] - cam.v0) / cam.a22;
            let x = (uv[0] - cam.u0 - cam.a12 * y) / cam.a11;
            if x.hypot(y) <= edge {
                out.push((cam.ray(uv), uv));
            }
        }
    }
    let masked = 1. - out.len() as f64 / total as f64;
    (out, masked)
}

/// Kannala-Brandt fitted by least squares to the S20 camera's pixels within `max_angle` (radians).
pub fn fit(cam: &Fisheye, width: u32, height: u32, max_angle: f64) -> Result<(Kb, FitError)> {
    let (pts, masked) = samples(cam, width, height, max_angle, 8);
    ensure!(pts.len() > 100, "no pixels within the angle");
    let mut kb = Kb {
        fx: cam.a11,
        fy: cam.a22,
        cx: cam.u0,
        cy: cam.v0,
        k: [0.; 4],
    };
    for _ in 0..10 {
        let mut h = SMatrix::<f64, 8, 8>::zeros();
        let mut g = SVector::<f64, 8>::zeros();
        for (x, uv) in &pts {
            let rho = x[0].hypot(x[1]);
            let th = rho.atan2(x[2]);
            let (c, s) = if rho > 1e-15 {
                (x[0] / rho, x[1] / rho)
            } else {
                (1., 0.)
            };
            let d = kb.distorted(th);
            let p = kb.project(*x);
            let powers = [th.powi(3), th.powi(5), th.powi(7), th.powi(9)];
            let mut ju = SVector::<f64, 8>::zeros();
            let mut jv = SVector::<f64, 8>::zeros();
            ju[0] = d * c;
            ju[2] = 1.;
            jv[1] = d * s;
            jv[3] = 1.;
            for i in 0..4 {
                ju[4 + i] = kb.fx * powers[i] * c;
                jv[4 + i] = kb.fy * powers[i] * s;
            }
            h += ju * ju.transpose() + jv * jv.transpose();
            g += ju * (p[0] - uv[0]) + jv * (p[1] - uv[1]);
        }
        let step = h.lu().solve(&-g).context("singular fit")?;
        kb.fx += step[0];
        kb.fy += step[1];
        kb.cx += step[2];
        kb.cy += step[3];
        for i in 0..4 {
            kb.k[i] += step[4 + i];
        }
        if step.amax() < 1e-12 {
            break;
        }
    }
    let n = 400;
    ensure!(
        (1..=n).all(|i| kb.distorted(max_angle * i as f64 / n as f64)
            > kb.distorted(max_angle * (i - 1) as f64 / n as f64)),
        "the fitted fisheye folds within the angle"
    );
    let (fine, _) = samples(cam, width, height, max_angle, 4);
    let e: Vec<f64> = fine
        .iter()
        .map(|(x, uv)| {
            let p = kb.project(*x);
            (p[0] - uv[0]).hypot(p[1] - uv[1])
        })
        .collect();
    let rms = (e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64).sqrt();
    Ok((
        kb,
        FitError {
            max_px: e.iter().cloned().fold(0., f64::max),
            rms_px: rms,
            masked,
        },
    ))
}

/// The `images.txt` line of a camera -> world pose: world -> camera as COLMAP's qw qx qy qz tx ty tz.
pub fn image_line(id: usize, camera: usize, name: &str, pose: &Pose) -> String {
    let rcw = pose.r.transpose();
    let q = UnitQuaternion::from_matrix(&rcw);
    let t = -(rcw * pose.p);
    format!(
        "{id} {:.15} {:.15} {:.15} {:.15} {:.9} {:.9} {:.9} {camera} {name}",
        q.w, q.i, q.j, q.k, t.x, t.y, t.z
    )
}

/// A point's position and colour.
type Point = ([f64; 3], [u8; 3]);

/// One point per voxel (the mean position and colour) from a uniform draw of the cloud, the voxel
/// found by bisection so that about `target` remain; and the voxel edge (m).
pub fn thin(pts: &[Point], target: usize) -> (Vec<Point>, f64) {
    if pts.len() <= target {
        return (pts.to_vec(), 0.);
    }
    let lo_corner = pts.iter().fold([f64::MAX; 3], |a, (p, _)| {
        std::array::from_fn(|d| a[d].min(p[d]))
    });
    let key = |p: &[f64; 3], edge: f64| -> u64 {
        let c: [u64; 3] =
            std::array::from_fn(|d| (((p[d] - lo_corner[d]) / edge) as u64).min((1 << 21) - 1));
        c[0] << 42 | c[1] << 21 | c[2]
    };
    let count = |edge: f64| -> usize {
        let mut k: Vec<u64> = pts.par_iter().map(|(p, _)| key(p, edge)).collect();
        k.par_sort_unstable();
        k.dedup();
        k.len()
    };
    let (mut lo, mut hi) = (1e-4f64, 10f64);
    for _ in 0..40 {
        let mid = (lo * hi).sqrt();
        if count(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi / lo < 1.01 {
            break;
        }
    }
    let mut keyed: Vec<(u64, usize)> = pts
        .par_iter()
        .enumerate()
        .map(|(i, (p, _))| (key(p, hi), i))
        .collect();
    keyed.par_sort_unstable();
    let out = keyed
        .chunk_by(|a, b| a.0 == b.0)
        .map(|run| {
            let n = run.len() as f64;
            let mut p = [0.; 3];
            let mut c = [0.; 3];
            for &(_, i) in run {
                for d in 0..3 {
                    p[d] += pts[i].0[d] / n;
                    c[d] += pts[i].1[d] as f64 / n;
                }
            }
            (p, c.map(|v| v.round() as u8))
        })
        .collect();
    (out, hi)
}

pub struct Input {
    pub data: PathBuf,
    pub imgpose: PathBuf,
    pub calib: PathBuf,
    pub cloud: PathBuf,
    pub out: PathBuf,
    /// seed points
    pub points: usize,
    /// degrees; pixels past it are masked
    pub max_angle: f64,
}

pub fn export(input: &Input, job: &Job) -> Result<serde_json::Value> {
    let cal = s20_io::calib::load(&input.calib)?;
    let sizes = [
        (cal.left.width, cal.left.height),
        (cal.right.width, cal.right.height),
    ];
    let cams = [Fisheye::from(&cal.left), Fisheye::from(&cal.right)];
    let max_angle = input.max_angle.to_radians();
    let sparse = input.out.join("sparse").join("0");
    std::fs::create_dir_all(&sparse)?;

    // cameras
    let mut fits = Vec::new();
    let mut text = String::from(
        "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[] (1 left, 2 right; fitted to the S20 fisheye)\n",
    );
    for s in 0..2 {
        let (kb, err) = fit(&cams[s], sizes[s].0, sizes[s].1, max_angle)?;
        writeln!(text, "{}", kb.colmap_line(s + 1, sizes[s].0, sizes[s].1))?;
        job.log(&format!(
            "camera {}: fit {:.3} px worst, {:.3} px rms, {:.1}% masked",
            s + 1,
            err.max_px,
            err.rms_px,
            100. * err.masked
        ));
        fits.push((kb, err));
    }
    std::fs::write(sparse.join("cameras.txt"), text)?;

    // masks: the same for every image of a camera
    let masks: Vec<Vec<u8>> = (0..2)
        .map(|s| -> Result<Vec<u8>> {
            let (w, h) = sizes[s];
            let edge = cams[s].radius(max_angle);
            let c = &cams[s];
            let img = image::GrayImage::from_fn(w, h, |u, v| {
                let y = (v as f64 - c.v0) / c.a22;
                let x = (u as f64 - c.u0 - c.a12 * y) / c.a11;
                image::Luma([if x.hypot(y) <= edge { 255 } else { 0 }])
            });
            let mut png = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
            Ok(png)
        })
        .collect::<Result<_>>()?;
    let any_mask = fits.iter().any(|(_, e)| e.masked > 0.);

    // images and their poses
    let poses = read_image_poses(&input.imgpose)?;
    let mut text = String::from(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (world to camera), then the 2D points (none)\n",
    );
    for (i, name) in poses.names.iter().enumerate() {
        writeln!(
            text,
            "{}\n",
            image_line(i + 1, side(name) + 1, name, &poses.poses[i])
        )?;
    }
    std::fs::write(sparse.join("images.txt"), text)?;
    let progress = job.part(0., 0.6);
    let done = std::sync::atomic::AtomicU64::new(0);
    poses.names.par_iter().try_for_each(|name| -> Result<()> {
        job.check()?;
        let to = input.out.join("images").join(name);
        if !to.exists() {
            let jpeg = s20_io::dataset::read_image(&input.data.join("images").join(name))
                .with_context(|| name.clone())?;
            std::fs::create_dir_all(to.parent().unwrap())?;
            std::fs::write(&to, jpeg)?;
        }
        if any_mask {
            let m = input.out.join("masks").join(format!("{name}.png"));
            std::fs::create_dir_all(m.parent().unwrap())?;
            std::fs::write(m, &masks[side(name)])?;
        }
        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        progress.report("images", n, poses.names.len() as u64);
        Ok(())
    })?;

    // seed points: a uniform draw of up to 40M points, thinned
    let mut rd = s20_io::las::LasReader::open(&input.cloud)?;
    let total = rd.len();
    let draw = (8 * input.points).clamp(8_000_000, 40_000_000);
    let every = total.div_ceil(draw).max(1);
    let mut pts = Vec::with_capacity(total / every + 1);
    let mut seen = 0usize;
    let reading = job.part(0.6, 0.9);
    loop {
        let chunk = rd.read(1 << 20)?;
        if chunk.is_empty() {
            break;
        }
        for q in &chunk {
            if seen.is_multiple_of(every) {
                pts.push((q.xyz, q.rgb.map(|c| (c >> 8) as u8)));
            }
            seen += 1;
        }
        reading.report("cloud", seen as u64, total as u64);
        job.check()?;
    }
    let (thinned, voxel) = thin(&pts, input.points);
    let mut text = String::with_capacity(thinned.len() * 64);
    text.push_str("# POINT3D_ID X Y Z R G B ERROR TRACK[] (LiDAR, no tracks)\n");
    for (i, (p, c)) in thinned.iter().enumerate() {
        writeln!(
            text,
            "{} {:.4} {:.4} {:.4} {} {} {} 0",
            i + 1,
            p[0],
            p[1],
            p[2],
            c[0],
            c[1],
            c[2]
        )?;
    }
    std::fs::write(sparse.join("points3D.txt"), text)?;
    std::fs::write(input.out.join("gauge.txt"), "oriented 1\nmetric 1\n")?;
    job.log(&format!(
        "{} images, {} seed points (voxel {:.1} cm, from {} drawn of {})",
        poses.names.len(),
        thinned.len(),
        voxel * 100.,
        pts.len(),
        total
    ));

    let report = serde_json::json!({
        "images": poses.names.len(),
        "seed_points": thinned.len(),
        "voxel_m": voxel,
        "cloud_points": total,
        "max_angle_deg": input.max_angle,
        "cameras": fits.iter().map(|(kb, e)| serde_json::json!({"fx": kb.fx, "fy": kb.fy, "cx": kb.cx + 0.5, "cy": kb.cy + 0.5, "k": kb.k, "fit": e})).collect::<Vec<_>>(),
        "calibration": input.calib, "imgpose": input.imgpose, "cloud": input.cloud,
    });
    std::fs::write(
        input.out.join("export.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anchors::V3;

    fn s20() -> Fisheye {
        Fisheye {
            k: [-1.33e-2, 6.54e-2, -9.10e-2, 9.59e-2, -5.04e-2, 1.14e-2],
            a11: 1485.,
            a12: 0.08,
            a22: 1485.6,
            u0: 1742.4,
            v0: 2321.5,
        }
    }

    #[test]
    fn the_fitted_fisheye_puts_rays_on_the_s20_pixels() {
        let (kb, e) = fit(&s20(), 3504, 4672, 85f64.to_radians()).unwrap();
        assert!(e.max_px < 1. && e.rms_px < 0.5, "{e:?}");
        assert!(e.masked > 0. && e.masked < 0.2, "{e:?}");
        // an exact Kannala-Brandt camera is found again
        let exact = Fisheye {
            k: [0., 0.02, 0., 0.004, 0., 0.],
            a11: 1400.,
            a12: 0.,
            a22: 1410.,
            u0: 1750.,
            v0: 2300.,
        };
        let (kb2, e2) = fit(&exact, 3504, 4672, 85f64.to_radians()).unwrap();
        assert!(e2.max_px < 1e-6, "{e2:?}");
        assert!(
            (kb2.k[0] - 0.02).abs() < 1e-8 && (kb2.k[1] - 0.004).abs() < 1e-8,
            "{kb2:?}"
        );
        assert!(
            kb.colmap_line(1, 3504, 4672)
                .starts_with("1 OPENCV_FISHEYE 3504 4672 ")
        );
    }

    #[test]
    fn an_image_line_maps_world_points_into_the_camera() {
        let r = *nalgebra::Rotation3::from_euler_angles(0.3, -1.2, 2.0).matrix();
        let pose = Pose {
            r,
            p: V3::new(4., -2., 1.5),
        };
        let line = image_line(7, 2, "right/right_00001.jpg", &pose);
        let v: Vec<f64> = line
            .split_whitespace()
            .skip(1)
            .take(7)
            .map(|s| s.parse().unwrap())
            .collect();
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(v[0], v[1], v[2], v[3]));
        let world = V3::new(1., 2., 3.);
        let colmap = q * world + V3::new(v[4], v[5], v[6]);
        let ours: V3 = r.transpose() * (world - pose.p);
        assert!((colmap - ours).norm() < 1e-8, "{colmap} {ours}");
        assert!(line.ends_with(" 2 right/right_00001.jpg"));
    }

    #[test]
    fn thinning_keeps_one_point_per_voxel_near_the_target() {
        let pts: Vec<([f64; 3], [u8; 3])> = (0..200_000)
            .map(|i| {
                (
                    [
                        (i % 100) as f64 * 0.01,
                        (i / 100 % 100) as f64 * 0.01,
                        (i / 10_000) as f64 * 0.01,
                    ],
                    [10, 20, 30],
                )
            })
            .collect();
        let (out, edge) = thin(&pts, 20_000);
        assert!(
            out.len() <= 20_000 && out.len() > 10_000,
            "{} {edge}",
            out.len()
        );
        assert!(out.iter().all(|(_, c)| *c == [10, 20, 30]));
    }
}
