//! The observations of a dataset placed with a trajectory: each point with the pose at its own
//! time, world = R(t)·(x + lever) + p(t), the sensor at R(t)·lever + p(t). The trajectories are
//! in the dataset's local frame (its projection origin subtracted), so f32 keeps millimetres.
use crate::{FinishParams, Observation};
use rayon::prelude::*;
use s20_geom::so3::V3;
use s20_geom::traj::Traj;
use s20_io::dataset::LidarData;

/// Points nearer than this (m) are not read (the device and the operator's hands).
pub(crate) const NEAR: f64 = 0.3;

/// Which scans are left out (see [`FinishParams::trim_still`]): per session, those before the
/// sensor first moves `dist` from its first position and after it last is `dist` from its last.
pub fn set_down_scans(data: &LidarData, tr: &Traj, dist: f64) -> Vec<bool> {
    let n = data.len();
    let mut out = vec![false; n];
    if dist <= 0.0 {
        return out;
    }
    let pos: Vec<Option<V3>> = (0..n)
        .map(|i| {
            let t = data.scan_start(i);
            (t >= tr.start() && t <= tr.end()).then(|| tr.at(t).1)
        })
        .collect();
    let mut a = 0;
    while a < n {
        let s = data.scans[a].session;
        let b = (a..n).find(|&i| data.scans[i].session != s).unwrap_or(n);
        let known: Vec<usize> = (a..b).filter(|&i| pos[i].is_some()).collect();
        if let (Some(&f), Some(&l)) = (known.first(), known.last()) {
            let (p0, p1) = (pos[f].unwrap(), pos[l].unwrap());
            let away = |i: usize, p: V3| pos[i].is_some_and(|q| (q - p).norm() >= dist);
            let first = known.iter().copied().find(|&i| away(i, p0));
            let last = known.iter().rev().copied().find(|&i| away(i, p1));
            for (i, o) in out.iter_mut().enumerate().take(b).skip(a) {
                *o = first.is_none_or(|f| i < f) || last.is_none_or(|l| i > l);
            }
        }
        a = b;
    }
    out
}

/// The observations of scan `i` (empty when the scan lies outside the
/// trajectory). Points tagged by the sensor (noise, rain) and outside NEAR..`max_range` are skipped.
pub fn scan(data: &LidarData, tr: &Traj, lever: V3, max_range: f64, i: usize) -> Vec<Observation> {
    let sc = data.scan(i);
    let (Some(start), Some(end)) = (
        sc.t.iter().copied().reduce(f64::min),
        sc.t.iter().copied().reduce(f64::max),
    ) else {
        return Vec::new();
    };
    if start < tr.start() || end > tr.end() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(sc.t.len());
    for k in 0..sc.t.len() {
        let x = V3::from(sc.xyz[k]);
        let range = x.norm();
        if !range.is_finite() || range < NEAR || range > max_range || sc.tag[k] & 0x3f != 0 {
            continue;
        }
        let (r, p) = tr.at(sc.t[k]);
        let w = r * (x + lever) + p;
        let o = r * lever + p;
        out.push(Observation {
            time: sc.t[k],
            xyz: [w.x as f32, w.y as f32, w.z as f32],
            origin: [o.x as f32, o.y as f32, o.z as f32],
            range: range as f32,
            scan: i as u32,
            intensity: sc.refl[k] as u16,
        });
    }
    out
}

/// All observations of the scans kept (see [`set_down_scans`]), in scan order.
pub fn observations(
    data: &LidarData,
    tr: &Traj,
    lever: [f64; 3],
    p: &FinishParams,
) -> Vec<Observation> {
    let left_out = set_down_scans(data, tr, p.trim_still);
    let lever = V3::from(lever);
    let parts: Vec<Vec<Observation>> = (0..data.len())
        .into_par_iter()
        .map(|i| {
            if left_out[i] {
                Vec::new()
            } else {
                scan(data, tr, lever, p.max_range, i)
            }
        })
        .collect();
    let mut out = Vec::with_capacity(parts.iter().map(|v| v.len()).sum());
    for v in parts {
        out.extend(v);
    }
    out
}
