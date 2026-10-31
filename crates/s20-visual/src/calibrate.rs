//! A calibration file with the cameras' changes baked in:
//! each fisheye's polynomial refitted so that it maps the changed rays to the same pixels, and the
//! lidar-to-camera extrinsics turned by the mounts. Export then gives image poses with the change.

use crate::camera::Fisheye;
use crate::refine::{Cameras, Intrinsic};
use anyhow::{Context, Result, bail, ensure};
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

/// The ray a pixel sees with the camera changed by `k` (exact: the pixel moved, then the angle).
pub fn changed_ray(cam: &Fisheye, uv: [f64; 2], k: &Intrinsic) -> [f64; 3] {
    let r = cam.ray([uv[0] + k[2], uv[1] + k[3]]);
    let rho = r[0].hypot(r[1]);
    let theta = rho.atan2(r[2]);
    let angle = theta + k[0] * theta + k[1] * theta.powi(3);
    let (x, y) = if rho > 1e-12 {
        (r[0] / rho, r[1] / rho)
    } else {
        (1., 0.)
    };
    [x * angle.sin(), y * angle.sin(), angle.cos()]
}

/// The polynomial fisheye that maps `changed_ray` back to the pixel, and its worst error in pixels
/// over the field (to 79 degrees).
pub fn native(cam: &Fisheye, k: &Intrinsic) -> Result<(Fisheye, f64)> {
    let (a, b) = (k[0], k[1]);
    let n = 2000;
    let theta0: Vec<f64> = (0..n)
        .map(|i| 85f64.to_radians() * i as f64 / (n - 1) as f64)
        .collect();
    let theta: Vec<f64> = theta0.iter().map(|t| t + a * t + b * t.powi(3)).collect();
    ensure!(
        theta.windows(2).all(|w| w[1] > w[0]),
        "the angular change is not monotone"
    );
    let design = DMatrix::from_fn(n, 6, |i, j| theta[i].powi(j as i32 + 2));
    let target = DVector::from_fn(n, |i, _| {
        let t = theta0[i];
        let radius = t + (0..6).map(|p| cam.k[p] * t.powi(p as i32 + 2)).sum::<f64>();
        (1. + a) * radius - theta[i]
    });
    let coef = design
        .svd(true, true)
        .solve(&target, 1e-15)
        .map_err(anyhow::Error::msg)?;
    let new = Fisheye {
        k: std::array::from_fn(|p| coef[p]),
        a11: cam.a11 / (1. + a),
        a12: cam.a12 / (1. + a),
        a22: cam.a22 / (1. + a),
        u0: cam.u0 - k[2],
        v0: cam.v0 - k[3],
    };
    let mut worst = 0f64;
    for i in 0..200 {
        let t = (0.001 + (79f64.to_radians() - 0.001) * i as f64 / 199.).min(79f64.to_radians());
        for j in 0..30 {
            let phi = std::f64::consts::TAU * j as f64 / 30.;
            let uv = cam.project([t.sin() * phi.cos(), t.sin() * phi.sin(), t.cos()]);
            let back = new.project(changed_ray(cam, uv, k));
            worst = worst.max((back[0] - uv[0]).hypot(back[1] - uv[1]));
        }
    }
    Ok((new, worst))
}

/// Replace the number after `key:` inside `block`.
fn set_number(block: &str, key: &str, value: f64) -> Result<String> {
    let mut found = None;
    let mut from = 0;
    while let Some(i) = block[from..].find(key) {
        let at = from + i;
        let before = block[..at].chars().last();
        let after = &block[at + key.len()..];
        if before.is_none_or(|c| !c.is_alphanumeric()) && after.starts_with(':') {
            ensure!(found.is_none(), "{key} appears twice");
            found = Some(at + key.len() + 1);
        }
        from = at + key.len();
    }
    let start = found.with_context(|| format!("no {key}"))?;
    let rest = &block[start..];
    let lead = rest.len() - rest.trim_start_matches([' ', '\t']).len();
    let len = rest[lead..]
        .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
        .unwrap_or(rest.len() - lead);
    ensure!(len > 0, "{key} has no number");
    Ok(format!(
        "{}{value:.17e}{}",
        &block[..start + lead],
        &rest[lead + len..]
    ))
}

/// `text` with the changed cameras: `cams` are the lines' current fisheyes (left, right).
pub fn write(text: &str, cams: &[Fisheye; 2], change: &Cameras) -> Result<(String, [f64; 2])> {
    let mut out = text.to_string();
    let mut worst = [0.; 2];
    for (side, (name, next)) in [
        ("fisheye_left:", "fisheye_right:"),
        ("fisheye_right:", "fisheye_middle:"),
    ]
    .into_iter()
    .enumerate()
    {
        let (new, err) = native(&cams[side], &change.intrinsic[side])?;
        ensure!(
            err < 0.1,
            "{name} the refitted polynomial misses by {err:.3} px"
        );
        worst[side] = err;
        let start = out.find(name).with_context(|| format!("no {name}"))?;
        let end = start
            + out[start..]
                .find(next)
                .with_context(|| format!("no {next}"))?;
        let mut block = out[start..end].to_string();
        for (p, v) in new.k.iter().enumerate() {
            block = set_number(&block, &format!("k{}", p + 2), *v)?;
        }
        for (key, v) in [
            ("A11", new.a11),
            ("A12", new.a12),
            ("A22", new.a22),
            ("u0", new.u0),
            ("v0", new.v0),
        ] {
            block = set_number(&block, key, v)?;
        }
        out.replace_range(start..end, &block);
        // lidar -> camera: (R, t) becomes (W^T R, W^T (t - offset)) for the mount (w, offset)
        let key = ["lidar_leftcamera:", "lidar_rightcamera:"][side];
        let at = out.find(key).with_context(|| format!("no {key}"))?;
        let data = at
            + out[at..]
                .find("data:")
                .with_context(|| format!("{key} has no data"))?;
        let open = data
            + out[data..]
                .find('[')
                .with_context(|| format!("{key} has no ["))?;
        let close = open + out[open..].find(']').context("no ]")?;
        let v: Vec<f64> = out[open + 1..close]
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<_, _>>()?;
        if v.len() != 16 {
            bail!("{key} has {} values", v.len());
        }
        let m = change.mount[side];
        let w = Rotation3::new(Vector3::new(m[0], m[1], m[2]));
        let r = Matrix3::from_fn(|i, j| v[i * 4 + j]);
        let t = Vector3::new(v[3], v[7], v[11]);
        let nr = w.matrix().transpose() * r;
        let nt = w.matrix().transpose() * (t - Vector3::new(m[3], m[4], m[5]));
        let mut nv = v.clone();
        for i in 0..3 {
            for j in 0..3 {
                nv[i * 4 + j] = nr[(i, j)];
            }
            nv[i * 4 + 3] = nt[i];
        }
        let data = format!(
            "[ {} ]",
            nv.iter()
                .map(|x| format!("{x:.17e}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        out.replace_range(open..=close, &data);
    }
    Ok((out, worst))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_refitted_polynomial_maps_changed_rays_to_their_pixels() {
        let cam = Fisheye {
            k: [-0.0133, 0.0651, -0.0908, 0.0957, -0.0502, 0.0114],
            a11: 1485.,
            a12: 0.08,
            a22: 1486.,
            u0: 1744.,
            v0: 2319.,
        };
        let (_, err) = native(&cam, &[-0.0022, 0.0005, -3.2, 0.1]).unwrap();
        assert!(err < 0.01, "{err}");
    }

    #[test]
    fn writes_the_changed_cameras_and_reads_back() {
        let text = r#"intrinsic:
   fisheye_left:
      image_width: 3504
      image_height: 4672
      projection_parameters:
         k2: 0.
         k3: 0.
         k4: 0.
         k5: 0.
         k6: 0.
         k7: 0.
         A11: 1.0e+03
         A12: 0.
         A22: 1.0e+03
         u0: 1752.
         v0: 2336.
   fisheye_right:
      image_width: 3504
      image_height: 4672
      projection_parameters:
         k2: 0.
         k3: 0.
         k4: 0.
         k5: 0.
         k6: 0.
         k7: 0.
         A11: 1.0e+03
         A12: 0.
         A22: 1.0e+03
         u0: 1752.
         v0: 2336.
   fisheye_middle:
extrinsic:
   lidar_leftcamera: !!opencv-matrix
      data: [ 1., 0., 0., 0.1, 0., 1., 0., 0., 0., 0., 1., 0.,
          0., 0., 0., 1. ]
   lidar_rightcamera: !!opencv-matrix
      data: [ 1., 0., 0., -0.1, 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1. ]
   lidar_middlecamera: !!opencv-matrix
      data: [ 1. ]
LIDAR_IMU_T: [0., 0., 0.]
RTK_offset: [0., 0., 0.]
"#;
        let cam = Fisheye {
            k: [0.; 6],
            a11: 1000.,
            a12: 0.,
            a22: 1000.,
            u0: 1752.,
            v0: 2336.,
        };
        let change = Cameras {
            intrinsic: [[0.; 4], [0.001, 0., 2., -1.]],
            mount: [[0.; 6], [0.01, 0., 0., 0.002, 0., 0.]],
        };
        let (out, _) = write(text, &[cam.clone(), cam.clone()], &change).unwrap();
        let dir = std::env::temp_dir().join(format!("s20-calibrate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("calibration.yaml");
        std::fs::write(&path, &out).unwrap();
        let cal = s20_io::calib::load(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(cal.left.u0, 1752.);
        assert!((cal.right.u0 - 1750.).abs() < 1e-9 && (cal.right.v0 - 2337.).abs() < 1e-9);
        assert!((cal.right.a11 - 1000. / 1.001).abs() < 1e-9);
        // the right extrinsic is turned by W^T and shifted
        let r = &cal.t_right_lidar;
        assert!(
            (r[5] - 0.01f64.cos()).abs() < 1e-12 && (r[6] - 0.01f64.sin()).abs() < 1e-12,
            "{r:?}"
        );
        assert!((r[3] - (-0.1 - 0.002)).abs() < 1e-12);
        assert_eq!(cal.t_left_lidar[3], 0.1);
    }
}
