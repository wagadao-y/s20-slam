//! Reader for the S20 `info/calibration.yaml` (OpenCV YAML). Only the values used by the pipeline.

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FisheyeIntrinsics {
    /// polynomial r(theta) = theta + k2 theta^2 + ... + k7 theta^7
    pub k: [f64; 6],
    pub a11: f64,
    pub a12: f64,
    pub a22: f64,
    pub u0: f64,
    pub v0: f64,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct Calibration {
    /// lidar origin in the IMU frame: p_imu = p_lidar + lidar_imu_t (rotation is identity on the S20)
    pub lidar_imu_t: [f64; 3],
    /// RTK antenna offset relative to the lidar
    pub rtk_offset: [f64; 3],
    pub left: FisheyeIntrinsics,
    pub right: FisheyeIntrinsics,
    /// 4x4 row-major T_cam_lidar
    pub t_left_lidar: [f64; 16],
    pub t_right_lidar: [f64; 16],
}

fn numbers(s: &str) -> Vec<f64> {
    s.replace(['[', ']', ','], " ")
        .split_whitespace()
        .filter_map(|x| x.parse().ok())
        .collect()
}

/// Value text of `key:` (including following lines until the closing bracket for lists).
fn value_after<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let i = text.find(&format!("{key}:"))?;
    let rest = &text[i + key.len() + 1..];
    if let Some(open) = rest.find('[') {
        // list value only if the bracket comes before the next newline-separated key
        let line_end = rest.find('\n').unwrap_or(rest.len());
        let nl2 = rest[line_end..].trim_start();
        if open < line_end
            || nl2.starts_with("rows")
            || nl2.starts_with("data")
            || open < rest.find(':').unwrap_or(usize::MAX)
        {
            let close = rest[open..].find(']')? + open;
            return Some(&rest[open..=close]);
        }
    }
    Some(rest.lines().next().unwrap_or(""))
}

fn scalar(block: &str, key: &str) -> Result<f64> {
    let v = value_after(block, key).with_context(|| format!("calibration: no {key}"))?;
    numbers(v)
        .first()
        .copied()
        .with_context(|| format!("calibration: bad {key}"))
}

fn block<'a>(text: &'a str, start: &str, end: Option<&str>) -> Result<&'a str> {
    let i = text
        .find(start)
        .with_context(|| format!("calibration: no {start}"))?;
    let rest = &text[i..];
    let j = end
        .and_then(|e| rest[1..].find(e).map(|j| j + 1))
        .unwrap_or(rest.len());
    Ok(&rest[..j])
}

fn fisheye(b: &str) -> Result<FisheyeIntrinsics> {
    let mut k = [0.0; 6];
    for (i, kk) in k.iter_mut().enumerate() {
        *kk = scalar(b, &format!("k{}", i + 2))?;
    }
    Ok(FisheyeIntrinsics {
        k,
        a11: scalar(b, "A11")?,
        a12: scalar(b, "A12")?,
        a22: scalar(b, "A22")?,
        u0: scalar(b, "u0")?,
        v0: scalar(b, "v0")?,
        width: scalar(b, "image_width")? as u32,
        height: scalar(b, "image_height")? as u32,
    })
}

fn matrix16(text: &str, key: &str) -> Result<[f64; 16]> {
    let b = block(text, key, Some("lidar_"))?;
    let d = b.find("data:").context("matrix without data")?;
    let v = numbers(&b[d + 5..b[d..].find(']').map(|x| x + d + 1).unwrap_or(b.len())]);
    if v.len() < 16 {
        bail!("calibration: {key} has {} values", v.len());
    }
    Ok(v[..16].try_into().unwrap())
}

pub fn load(path: &Path) -> Result<Calibration> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let v3 = |k: &str| -> Result<[f64; 3]> {
        let v = numbers(value_after(&text, k).with_context(|| format!("calibration: no {k}"))?);
        if v.len() < 3 {
            bail!("calibration: bad {k}");
        }
        Ok([v[0], v[1], v[2]])
    };
    Ok(Calibration {
        lidar_imu_t: v3("LIDAR_IMU_T")?,
        rtk_offset: v3("RTK_offset")?,
        left: fisheye(block(&text, "fisheye_left:", Some("fisheye_right:"))?)?,
        right: fisheye(block(&text, "fisheye_right:", Some("fisheye_middle:"))?)?,
        t_left_lidar: matrix16(&text, "lidar_leftcamera:")?,
        t_right_lidar: matrix16(&text, "lidar_rightcamera:")?,
    })
}
