//! Trajectory outputs in the maker's formats and the image poses for colouring
//! (ports of `export_traj_raw.py` and `imgpose_raw.py`).
//!
//! - `trajectory.txt`: poses at the scan ends (10 Hz), `x y z roll pitch yaw qx qy qz qw timestamp`
//!   (trajectory output frame, Euler 'xyz' in radians, output clock = sensor clock - maker offset,
//!   each session's own)
//! - `trajectory_100hz_tum.txt` (not the maker's: beside the trajectory it is made from, as
//!   `runs/`): the dense trajectory, TUM rows in the output clock
//! - `trajectory.las`: the 10 Hz poses as points with roll/pitch/yaw extra bytes
//! - `ImgPose.txt`: camera poses at the image stamps (each session's sensor clock), Euler in degrees
//! - `images/` (optional), as the maker's: the photos of `left/` and `right/` with their times
//!   (`leftImgTime.txt`, `rightImgTime.txt`), their `ImgPose.txt`, the poses as omega, phi, kappa
//!   (`xyzopk.txt`) and the cameras (`Left.opt`, `Right.opt`); not the maker's masks

use anyhow::{Context, Result};
use nalgebra::Matrix4;
use s20_geom::so3::{self, M3, V3};
use s20_geom::traj::Traj;
use s20_io::calib::Calibration;
use s20_io::dataset::{LidarData, session_at, sessions};
use s20_io::las::{ExtraKind, LasPoint};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// Euler angles of R = Rz(yaw) Ry(pitch) Rx(roll) (scipy `as_euler('xyz')`).
pub fn euler_xyz(r: &M3) -> [f64; 3] {
    let pitch = (-r[(2, 0)]).clamp(-1.0, 1.0).asin();
    let roll = r[(2, 1)].atan2(r[(2, 2)]);
    let yaw = r[(1, 0)].atan2(r[(0, 0)]);
    [roll, pitch, yaw]
}

/// `out_off`: per session, subtracted from a time to give the output clock
/// ([`s20_io::dataset::output_offsets`]).
pub fn trajectory(
    data: &LidarData,
    tr: &Traj,
    origin: [f64; 3],
    out_off: &[f64],
    out: &Path,
    tum_file: &Path,
) -> Result<usize> {
    let ranges = data.session_ranges();
    fs::create_dir_all(out)?;
    let o = V3::from(origin);
    let mut txt = String::from("# x y z roll pitch yaw qx qy qz qw timestamp\n");
    let mut pts = Vec::new();
    let mut eul = Vec::new();
    for i in 0..data.len() {
        let ts = data.scan_start(i) + 0.1;
        if ts < tr.start() || ts > tr.end() {
            continue;
        }
        let (r, p) = tr.at(ts);
        let p = p + o;
        let e = euler_xyz(&r);
        let q = so3::to_quat(&r);
        let tm = ts - out_off[data.scans[i].session];
        writeln!(
            txt,
            "{:.15} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15}",
            p.x, p.y, p.z, e[0], e[1], e[2], q[0], q[1], q[2], q[3], tm
        )?;
        pts.push(LasPoint {
            xyz: [p.x, p.y, p.z],
            gps_time: tm,
            ..Default::default()
        });
        eul.push(e);
    }
    fs::write(out.join("trajectory.txt"), txt)?;
    let mut tum = String::new();
    for k in 0..tr.len() {
        let q = tr.q[k].coords;
        let p = tr.p[k] + o;
        writeln!(
            tum,
            "{:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
            tr.t[k] - out_off[session_at(&ranges, tr.t[k])],
            p.x,
            p.y,
            p.z,
            q.x,
            q.y,
            q.z,
            q.w
        )?;
    }
    fs::write(tum_file, tum)?;
    let mut lo = [f64::INFINITY; 3];
    for p in &pts {
        for (l, v) in lo.iter_mut().zip(p.xyz) {
            *l = l.min(v);
        }
    }
    let off = lo.map(|v| if v.is_finite() { v.round() } else { 0.0 });
    let ex = [
        ("roll", ExtraKind::F64),
        ("pitch", ExtraKind::F64),
        ("yaw", ExtraKind::F64),
    ];
    s20_io::las::write_extra(&out.join("trajectory.las"), &pts, 1e-3, off, &ex, |i, b| {
        for c in 0..3 {
            b[8 * c..8 * c + 8].copy_from_slice(&eul[i][c].to_le_bytes());
        }
    })?;
    Ok(pts.len())
}

fn mat4(m: &[f64; 16]) -> Matrix4<f64> {
    Matrix4::from_row_slice(m)
}

/// A camera image placed on the trajectory.
pub struct CameraPose {
    /// its data folder (absolute) and name there (`left/left_00001.jpg`)
    pub dir: PathBuf,
    pub name: String,
    /// sensor-clock time
    pub t: f64,
    /// camera to world (output frame)
    pub r: M3,
    pub p: V3,
}

impl CameraPose {
    pub fn side(&self) -> &str {
        self.name.split('/').next().unwrap_or("")
    }
}

/// The images of every session within the trajectory, in their order, placed with the camera
/// extrinsics.
pub fn camera_poses(
    data: &Path,
    tr: &Traj,
    cal: &Calibration,
    origin: [f64; 3],
    lidar_imu_t: [f64; 3],
) -> Result<Vec<CameraPose>> {
    let mut ext = Vec::new();
    for (side, m) in [("left", &cal.t_left_lidar), ("right", &cal.t_right_lidar)] {
        let inv = mat4(m).try_inverse().context("singular camera extrinsic")?; // cam -> lidar
        let r: M3 = inv.fixed_view::<3, 3>(0, 0).into_owned();
        let t = V3::new(inv[(0, 3)], inv[(1, 3)], inv[(2, 3)]) + V3::from(lidar_imu_t); // cam -> imu
        ext.push((side, r, t));
    }
    let o = V3::from(origin);
    let mut out = Vec::new();
    for sd in sessions(data)? {
        let dir = std::path::absolute(&sd.dir)?;
        for im in s20_io::dataset::image_index(&dir)? {
            let tt = im.t + sd.shift;
            if tt < tr.start() || tt > tr.end() {
                continue;
            }
            let side = im.name.split('/').next().unwrap_or("");
            let Some((_, rbc, tbc)) = ext.iter().find(|e| e.0 == side) else {
                continue;
            };
            let (rb, pb) = tr.at(tt);
            out.push(CameraPose {
                dir: dir.clone(),
                name: im.name,
                t: im.t,
                r: rb * rbc,
                p: rb * tbc + pb + o,
            });
        }
    }
    Ok(out)
}

const IMG_POSE_HEADER: &str = "index x y z roll pitch yaw qx qy qz qw timestamp\n";

/// A row of `ImgPose.txt` for the image named `name`.
fn pose_row(s: &mut String, name: &str, c: &CameraPose) -> Result<()> {
    let e = euler_xyz(&c.r).map(f64::to_degrees);
    let q = so3::to_quat(&c.r);
    writeln!(
        s,
        "{name} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.9}",
        c.p.x, c.p.y, c.p.z, e[0], e[1], e[2], q[0], q[1], q[2], q[3], c.t
    )?;
    Ok(())
}

/// `ImgPose.txt` of `poses`; the names are absolute when the images are of several sessions (as
/// colouring reads a relative name under its own image folder).
pub fn image_poses(poses: &[CameraPose], out: &Path) -> Result<usize> {
    let multi = poses.iter().any(|c| c.dir != poses[0].dir);
    let mut s = String::from(IMG_POSE_HEADER);
    for c in poses {
        let name = if multi {
            c.dir
                .join("images")
                .join(&c.name)
                .to_string_lossy()
                .replace('\\', "/")
        } else {
            c.name.clone()
        };
        pose_row(&mut s, &name, c)?;
    }
    if let Some(p) = out.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(out, s)?;
    Ok(poses.len())
}

/// Omega, phi, kappa (degrees) of a camera, as photogrammetry: R = Rx(omega) Ry(phi) Rz(kappa)
/// of the camera looking down its -z with y up (the image's y and z turned over).
pub fn omega_phi_kappa(r: &M3) -> [f64; 3] {
    // the extrinsics are orthonormal only to some 1e-7
    let m = so3::normalize(r) * M3::from_diagonal(&V3::new(1.0, -1.0, -1.0));
    let phi = m[(0, 2)].clamp(-1.0, 1.0).asin();
    let omega = (-m[(1, 2)]).atan2(m[(2, 2)]);
    let kappa = (-m[(0, 1)]).atan2(m[(0, 0)]);
    [omega, phi, kappa].map(f64::to_degrees)
}

/// The sensor's longer side (mm) as the maker writes it in `Left.opt` / `Right.opt`.
const SENSOR_SIZE_MM: f64 = 11.21;

/// A number with six significant digits, without trailing zeros (as C's `%g`).
fn g6(v: f64) -> String {
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    let digits = (5 - v.abs().log10().floor() as i32).max(0) as usize;
    let s = format!("{v:.digits$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// A camera's optical properties as the maker's `.opt` file (CRLF, no newline at the end).
fn optical_properties(c: &s20_io::calib::FisheyeIntrinsics) -> String {
    let focal = c.a11 * SENSOR_SIZE_MM / c.width.max(c.height) as f64;
    format!(
        r#"<?xml version="1.0" encoding="utf-8"?>
<OpticalProperties version="1.0">
    <Id>0</Id>
    <Name>x</Name>
    <Description />
    <Directory />
    <ImageDimensions>
        <Width>{}</Width>
        <Height>{}</Height>
    </ImageDimensions>
    <CameraModelType>Fisheye</CameraModelType>
    <CameraModelBand>Visible</CameraModelBand>
    <SensorSize>{SENSOR_SIZE_MM}</SensorSize>
    <FocalLength>{}</FocalLength>
    <FisheyeFocalMatrix>
        <M_00>{}</M_00>
        <M_01>{}</M_01>
        <M_10>0</M_10>
        <M_11>{}</M_11>
    </FisheyeFocalMatrix>
    <Exif />
</OpticalProperties>"#,
        c.width,
        c.height,
        g6(focal),
        g6(c.a11),
        g6(c.a12),
        g6(c.a22)
    )
    .replace('\n', "\r\n")
}

/// The names `image_set` gives the photos of `poses` (`<side>/<file>`, in their order): their own
/// for one session, numbered on per camera across several.
pub fn set_names(poses: &[CameraPose]) -> Vec<String> {
    let multi = poses.iter().any(|c| c.dir != poses[0].dir);
    let mut count = std::collections::HashMap::new();
    poses
        .iter()
        .map(|c| {
            let side = c.side();
            let k = count.entry(side.to_string()).or_insert(0usize);
            *k += 1;
            let file = if multi {
                format!("{side}_{:05}.jpg", k)
            } else {
                c.name.rsplit('/').next().unwrap_or(&c.name).to_string()
            };
            format!("{side}/{file}")
        })
        .collect()
}

/// `ImgPose.txt` of `poses` named as `image_set` names them (the left camera's first).
pub fn set_poses(poses: &[CameraPose], out: &Path) -> Result<()> {
    let names = set_names(poses);
    let mut s = String::from(IMG_POSE_HEADER);
    for side in ["left", "right"] {
        for (c, name) in poses.iter().zip(&names).filter(|(c, _)| c.side() == side) {
            pose_row(&mut s, name, c)?;
        }
    }
    fs::write(out, s).with_context(|| out.display().to_string())
}

/// The maker's `images/` folder into `out`: per camera the photos (their names kept for one
/// session, numbered on across several) with their times, the poses of all (`ImgPose.txt`, the
/// left camera's first) and as omega, phi, kappa (`xyzopk.txt`), and the cameras. Returns the
/// images written.
pub fn image_set(
    poses: &[CameraPose],
    cal: &Calibration,
    out: &Path,
    job: &s20_io::job::Job,
) -> Result<usize> {
    let names = set_names(poses);
    let mut img_pose = String::from(IMG_POSE_HEADER);
    let mut opk = String::from("#Index,X,Y,Z,Omega,Phi,Kappa\n");
    let mut done = 0;
    for side in ["left", "right"] {
        let dir = out.join(side);
        fs::create_dir_all(&dir)?;
        let mut times = String::new();
        let mine = poses.iter().zip(&names).filter(|(c, _)| c.side() == side);
        for (k, (c, name)) in mine.enumerate() {
            job.check()?;
            job.report("export images", done as u64, poses.len() as u64);
            let file = name.rsplit('/').next().unwrap_or(name).to_string();
            // the number in the file name
            let number: usize = file
                .trim_start_matches(&format!("{side}_"))
                .trim_end_matches(".jpg")
                .parse()
                .unwrap_or(k + 1);
            let jpeg = s20_io::dataset::read_image(&c.dir.join("images").join(&c.name))?;
            fs::write(dir.join(&file), jpeg)?;
            writeln!(times, "{number} {:.8}", c.t)?;
            pose_row(&mut img_pose, &format!("{side}/{file}"), c)?;
            let a = omega_phi_kappa(&c.r);
            writeln!(
                opk,
                "{file},{:.12},{:.12},{:.12},{},{},{}",
                c.p.x, c.p.y, c.p.z, a[0], a[1], a[2]
            )?;
            done += 1;
        }
        fs::write(dir.join(format!("{side}ImgTime.txt")), times)?;
    }
    fs::write(out.join("ImgPose.txt"), img_pose)?;
    fs::write(out.join("xyzopk.txt"), opk)?;
    fs::write(out.join("Left.opt"), optical_properties(&cal.left))?;
    fs::write(out.join("Right.opt"), optical_properties(&cal.right))?;
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_names_keep_one_sessions_names_and_number_on_across_several() {
        let pose = |dir: &str, name: &str| CameraPose {
            dir: PathBuf::from(dir),
            name: name.to_string(),
            t: 0.,
            r: M3::identity(),
            p: V3::zeros(),
        };
        let one = [
            pose("a", "left/left_00007.jpg"),
            pose("a", "right/right_00007.jpg"),
        ];
        assert_eq!(
            set_names(&one),
            ["left/left_00007.jpg", "right/right_00007.jpg"]
        );
        let two = [
            pose("a", "left/left_00007.jpg"),
            pose("a", "right/right_00007.jpg"),
            pose("b", "left/left_00001.jpg"),
        ];
        assert_eq!(
            set_names(&two),
            [
                "left/left_00001.jpg",
                "right/right_00001.jpg",
                "left/left_00002.jpg"
            ]
        );
    }

    #[test]
    fn omega_phi_kappa_as_the_maker() {
        // left_00001 of a maker's output: its ImgPose quaternion and xyzopk angles
        let r = so3::from_quat([-0.684158, 0.185882, -0.184669, 0.680642]);
        let want = [89.65141489651188, -30.37939213726911, -0.19647934517865842];
        let got = omega_phi_kappa(&r);
        for d in 0..3 {
            assert!((got[d] - want[d]).abs() < 1e-3, "{got:?} {want:?}");
        }
    }

    #[test]
    fn images_of_several_sessions_are_numbered_on_per_camera() {
        let root = std::env::temp_dir().join(format!("s20_image_set_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut poses = Vec::new();
        for (s, t0) in [("a", 100.0), ("b", 500.0)] {
            let dir = root.join(s);
            for (k, side) in ["left", "right", "left"].into_iter().enumerate() {
                let name = format!("{side}/{side}_{:05}.jpg", k + 1);
                let f = dir.join("images").join(&name);
                fs::create_dir_all(f.parent().unwrap()).unwrap();
                fs::write(&f, format!("{s}{k}")).unwrap();
                poses.push(CameraPose {
                    dir: dir.clone(),
                    name,
                    t: t0 + k as f64,
                    r: M3::identity(),
                    p: V3::new(k as f64, 0.0, 0.0),
                });
            }
        }
        let cam = intrinsics();
        let cal = Calibration {
            lidar_imu_t: [0.0; 3],
            rtk_offset: [0.0; 3],
            left: cam.clone(),
            right: cam,
            t_left_lidar: [0.0; 16],
            t_right_lidar: [0.0; 16],
        };
        let out = root.join("images");
        let n = image_set(&poses, &cal, &out, &s20_io::job::Job::stderr()).unwrap();
        assert_eq!(n, 6);
        let read = |p: &str| fs::read_to_string(out.join(p)).unwrap();
        assert_eq!(read("left/left_00003.jpg"), "b0");
        assert_eq!(read("left/left_00004.jpg"), "b2");
        assert_eq!(read("right/right_00002.jpg"), "b1");
        assert_eq!(
            read("left/leftImgTime.txt").lines().nth(2).unwrap(),
            "3 500.00000000"
        );
        let names: Vec<String> = read("ImgPose.txt")
            .lines()
            .skip(1)
            .map(|l| l.split(' ').next().unwrap().to_string())
            .collect();
        assert_eq!(
            names[..4],
            [
                "left/left_00001.jpg",
                "left/left_00002.jpg",
                "left/left_00003.jpg",
                "left/left_00004.jpg"
            ]
        );
        assert!(
            read("xyzopk.txt")
                .lines()
                .nth(1)
                .unwrap()
                .starts_with("left_00001.jpg,0.000000000000,")
        );
        fs::remove_dir_all(&root).unwrap();
    }

    /// The left camera of a calibration.yaml.
    fn intrinsics() -> s20_io::calib::FisheyeIntrinsics {
        s20_io::calib::FisheyeIntrinsics {
            k: [0.0; 6],
            a11: 1483.996941726002,
            a12: 0.080_565_488_147_538_79,
            a22: 1484.6569866369118,
            u0: 1741.25,
            v0: 2317.12,
            width: 3504,
            height: 4672,
        }
    }

    #[test]
    fn optical_properties_as_the_maker() {
        let s = optical_properties(&intrinsics());
        for line in [
            "<Width>3504</Width>",
            "<Height>4672</Height>",
            "<SensorSize>11.21</SensorSize>",
            "<FocalLength>3.5607</FocalLength>",
            "<M_00>1484</M_00>",
            "<M_01>0.0805655</M_01>",
            "<M_10>0</M_10>",
            "<M_11>1484.66</M_11>",
        ] {
            assert!(s.contains(line), "{line} in {s}");
        }
    }
}
