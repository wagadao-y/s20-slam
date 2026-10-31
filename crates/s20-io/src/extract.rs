//! Indexing of an S20 raw folder into a data directory. Points and images stay in the bag (read
//! through a memory map); only their positions are written, with the small IMU / GNSS tables:
//! - source.txt      : the bag (absolute path, size), see [`crate::dataset::Source`]
//! - lidar_index.npy : [header_stamp, timebase_ns, byte offset of the points in the bag, point count]
//!   per scan (packed Livox points, 19 bytes: u32 offset_ns, f32 x y z, u8 refl, tag, line)
//! - imu.npy         : [t, wx, wy, wz, ax, ay, az]   (acceleration in g)
//! - gnss.npy        : rtk_agent PVTSLN rows (see msgs::pvtsln)
//! - temp.npy        : [t, degrees C] LiDAR core temperature from the info bag (`info/info_*.bag`), when present
//! - device_pose.txt : the trajectory the device estimated while recording (`frame_pose.txt`), as
//!   world_T_imu on the sensor clock (see [`extract_device_pose`]), when present
//! - images.txt      : `<side>/<side>_NNNNN.jpg t offset length` per fisheye JPEG in the bag

use crate::bag::Bag;
use crate::dataset::Source;
use crate::{msgs, npy};
use anyhow::{Context, Result};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

pub struct ExtractStats {
    pub scans: usize,
    pub points: usize,
    pub imu: usize,
    pub gnss: usize,
    pub images: usize,
}

/// Find `all_*.bag` in an S20 raw folder.
pub fn find_bag(raw: &Path) -> Result<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(raw)
        .with_context(|| format!("read {}", raw.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("all_") && n.ends_with(".bag"))
        })
        .collect();
    v.sort();
    v.into_iter()
        .next()
        .with_context(|| format!("no all_*.bag in {}", raw.display()))
}

/// The sensor clock minus the system clock (s): the median of (IMU header stamp - record time) in the
/// main bag. The record times, the info bag and `frame_pose.txt` are on the system clock.
fn clock_offset(bag: &Bag) -> Result<f64> {
    let im = bag.topic_messages("/livox/imu");
    let mut d: Vec<f64> = im
        .iter()
        .step_by(50)
        .map(|m| Ok(msgs::imu(bag.data(m))?[0] - m.time_ns as f64 * 1e-9))
        .collect::<Result<_>>()?;
    anyhow::ensure!(!d.is_empty(), "no imu messages for the clock offset");
    d.sort_by(f64::total_cmp);
    Ok(d[d.len() / 2])
}

/// LiDAR core temperature from `<raw>/info/info_*.bag` into `<out>/temp.npy` as `[t, degrees C]` on the
/// sensor clock (its record times moved by [`clock_offset`]). Returns the row count, or `None` without
/// an info bag.
pub fn extract_temperature(raw: &Path, bag: &Bag, out: &Path) -> Result<Option<usize>> {
    let Ok(rd) = fs::read_dir(raw.join("info")) else {
        return Ok(None);
    };
    let mut infos: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("info_") && n.ends_with(".bag"))
        })
        .collect();
    infos.sort();
    let Some(info) = infos.first() else {
        return Ok(None);
    };
    let offset = clock_offset(bag)?;
    let ib = Bag::open(info)?;
    let mut rows = Vec::new();
    for m in ib.topic_messages("/livox/base_info") {
        rows.push(m.time_ns as f64 * 1e-9 + offset);
        rows.push(msgs::lidar_core_temp(ib.data(&m))?);
    }
    npy::write_f64(&out.join("temp.npy"), &rows, 2)?;
    Ok(Some(rows.len() / 2))
}

/// The device's own trajectory in a data directory.
pub const DEVICE_POSE: &str = "device_pose.txt";

/// The tilt of the LiDAR (and its IMU) about the device's y axis: `frame_pose.txt` holds the device
/// body, world_T_imu = world_T_body * Ry(25°) (the S20's LiDAR is mounted tilted 25° to the ground).
/// Fitted against the optimised trajectory of a garden recording: 24.5° about y, under 0.2° about
/// the other axes, and the body's origin within a few cm of the IMU's (taken as the same).
const DEVICE_TILT_DEG: f64 = 25.0;

/// The trajectory the device estimated while recording, `<raw>/frame_pose.txt` (`t x y z qx qy qz
/// qw` of the body, t on the system clock), into `<out>/device_pose.txt` as world_T_imu on the sensor
/// clock (`offset` added, see [`clock_offset`]), as it is: an estimate that diverged stays so. Only
/// the poses over `span` (the scans, sensor clock) are kept, with one either side. Returns the row
/// count, or `None` without the file.
pub fn extract_device_pose(
    raw: &Path,
    offset: f64,
    span: [f64; 2],
    out: &Path,
) -> Result<Option<usize>> {
    let path = raw.join("frame_pose.txt");
    if !path.exists() {
        let _ = fs::remove_file(out.join(DEVICE_POSE));
        return Ok(None);
    }
    let (s, c) = (DEVICE_TILT_DEG.to_radians() * 0.5).sin_cos();
    let rows: Vec<[f64; 8]> = crate::dataset::read_tum(&path)?
        .into_iter()
        .map(|[t, px, py, pz, x, y, z, w]| {
            // q * (0, sin, 0, cos): the rotation about y applied first
            let q = [x * c - z * s, w * s + y * c, x * s + z * c, w * c - y * s];
            let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
            [
                t + offset,
                px,
                py,
                pz,
                q[0] / n,
                q[1] / n,
                q[2] / n,
                q[3] / n,
            ]
        })
        .collect();
    let first = rows.partition_point(|r| r[0] < span[0]).saturating_sub(1);
    let last = (rows.partition_point(|r| r[0] <= span[1]) + 1).min(rows.len());
    let rows = rows.get(first..last).unwrap_or_default();
    crate::dataset::write_tum(
        &out.join(DEVICE_POSE),
        rows,
        "t x y z qx qy qz qw (world_T_imu, sensor clock; the device's estimate, frame_pose.txt)",
    )?;
    Ok(Some(rows.len()))
}

pub fn extract(raw: &Path, out: &Path, job: &crate::job::Job) -> Result<ExtractStats> {
    let log = |s: &str| job.log(s);
    let bag_path = std::path::absolute(find_bag(raw)?)?;
    log(&format!("bag {}", bag_path.display()));
    let bag = Bag::open(&bag_path)?;
    log(&format!(
        "{} connections, {} messages",
        bag.connections.len(),
        bag.messages.len()
    ));
    fs::create_dir_all(out)?;
    job.check()?;

    // --- lidar
    let lid = bag.topic_messages("/livox/lidar");
    let mut scans = Vec::with_capacity(lid.len() * 4);
    let mut total = 0usize;
    for m in &lid {
        let s = msgs::lidar(bag.data(m))?;
        let n = s.points.len() / msgs::POINT_SIZE;
        scans.extend_from_slice(&[
            s.stamp,
            s.timebase_ns as f64,
            bag.offset_of(s.points) as f64,
            n as f64,
        ]);
        total += n;
    }
    npy::write_f64(&out.join("lidar_index.npy"), &scans, 4)?;
    log(&format!("lidar: {} scans, {} points", lid.len(), total));

    // --- imu
    let im = bag.topic_messages("/livox/imu");
    let mut imu = Vec::with_capacity(im.len() * 7);
    for m in &im {
        imu.extend_from_slice(&msgs::imu(bag.data(m))?);
    }
    npy::write_f64(&out.join("imu.npy"), &imu, 7)?;

    // --- gnss (receiver solution synchronised to the device clock)
    let gm = bag.topic_messages("/rtk_agent/pvtsln_sync");
    let mut gnss = Vec::with_capacity(gm.len() * 14);
    for m in &gm {
        gnss.extend_from_slice(&msgs::pvtsln(bag.data(m))?);
    }
    npy::write_f64(&out.join("gnss.npy"), &gnss, 14)?;
    log(&format!("imu: {}  gnss: {}", im.len(), gm.len()));
    match extract_temperature(raw, &bag, out)? {
        Some(n) => log(&format!("temperature: {n} rows")),
        None => log("temperature: no info bag"),
    }
    if let (Some(a), Some(b)) = (scans.first(), scans.chunks(4).last()) {
        // the last scan ends about 0.1 s after its stamp
        let span = [*a, b[0] + 0.1];
        let offset = clock_offset(&bag)?;
        match extract_device_pose(raw, offset, span, out)? {
            Some(n) => log(&format!("device trajectory: {n} poses")),
            None => log("device trajectory: no frame_pose.txt"),
        }
    }

    // --- high resolution fisheye images, numbered per side in record time order
    let left = bag.topic_conns("/camera_agent/img_left/compressed");
    let right = bag.topic_conns("/camera_agent/img_right/compressed");
    let mut ims: Vec<_> = bag
        .messages
        .iter()
        .filter(|m| left.contains(&m.conn) || right.contains(&m.conn))
        .copied()
        .collect();
    ims.sort_by_key(|m| m.time_ns);
    let mut index = String::new();
    let (mut nl, mut nr) = (0, 0);
    for m in &ims {
        let (side, k) = if left.contains(&m.conn) {
            nl += 1;
            ("left", nl)
        } else {
            nr += 1;
            ("right", nr)
        };
        let (t, jpg) = msgs::compressed_image(bag.data(m))?;
        writeln!(
            index,
            "{side}/{side}_{k:05}.jpg {t:.9} {} {}",
            bag.offset_of(jpg),
            jpg.len()
        )?;
    }
    fs::write(out.join("images.txt"), index)?;
    log(&format!("images: {}", ims.len()));

    // written last: its presence means the indices above are complete
    let size = fs::metadata(&bag_path)?.len();
    Source {
        bag: bag_path,
        size,
    }
    .write(out)?;
    for old in ["lidar_points.bin", "lidar_scans.npy", "images"] {
        if out.join(old).exists() {
            log(&format!(
                "note: {} is from an older version (points and images are now read from the bag); it can be deleted",
                out.join(old).display()
            ));
        }
    }
    Ok(ExtractStats {
        scans: lid.len(),
        points: total,
        imu: im.len(),
        gnss: gm.len(),
        images: ims.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_pose_is_tilted_moved_to_the_sensor_clock_and_cut_to_the_scans() {
        let d = std::env::temp_dir().join(format!("s20_device_pose_{}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        let h = 0.3f64; // a yaw of 2h
        fs::write(
            d.join("frame_pose.txt"),
            format!(
                "# timestamp tx ty tz qx qy qz qw
                 100.0 0 0 0 0 0 0 1
                 101.0 1 2 3 0 0 0 1
                 102.0 1 2 3 0 0 {} {}
                 103.0 4 5 6 0 0 0 1
                 104.0 4 5 6 0 0 0 1
",
                h.sin(),
                h.cos()
            ),
        )
        .unwrap();
        // scans 1.5..2.5 s after the first pose, on a clock 1000 s ahead
        let n = extract_device_pose(&d, 1000.0, [1101.5, 1102.5], &d)
            .unwrap()
            .unwrap();
        let rows = crate::dataset::read_tum(&d.join(DEVICE_POSE)).unwrap();
        fs::remove_dir_all(&d).unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            rows.iter().map(|r| r[0]).collect::<Vec<_>>(),
            [1101.0, 1102.0, 1103.0]
        );
        assert_eq!(rows[0][1..4], [1.0, 2.0, 3.0]);
        // no body rotation: the tilt alone, about y
        let (s, c) = (DEVICE_TILT_DEG.to_radians() * 0.5).sin_cos();
        let e = (rows[0][4..8].iter().zip([0.0, s, 0.0, c]))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(e < 1e-9, "{:?}", &rows[0][4..8]);
        // a yaw, then the tilt: the yaw stays about the vertical, the tilt about the turned y
        let e = (rows[1][4..8]
            .iter()
            .zip([-h.sin() * s, h.cos() * s, h.sin() * c, h.cos() * c]))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
        assert!(e < 1e-9, "{:?}", &rows[1][4..8]);
    }
}
