//! Per-knot quality of an optimised trajectory, to show where it is doubtful.
//!
//! - plane residual: RMS distance of the knot's points to the planes of the final stage, and the
//!   fraction of its points that lie on planes (few means a degenerate or cluttered view)
//! - IMU: chi-square per degree of freedom of the preintegration factor to the next knot
//! - RTK: residual of the fixes in the knot's interval, horizontal and vertical
//! - correction: how much the motion over one second changed from the odometry (rotation and
//!   translation per second); large values mark odometry jumps that the optimisation repaired

use crate::imu::imu_residual;
use crate::planes::{self, NONE};
use crate::{Params, Problem};
use rayon::prelude::*;
use s20_geom::so3::{self, V3};

#[derive(Clone, Copy, Debug)]
pub struct KnotQuality {
    pub t: f64,
    pub plane_rms_mm: f64,
    pub plane_frac: f64,
    pub imu_chi2: f64,
    pub rtk_h_mm: f64,
    pub rtk_v_mm: f64,
    pub corr_rot_dps: f64,
    pub corr_trans_cmps: f64,
}

pub const HEADER: &str =
    "t plane_rms_mm plane_frac imu_chi2 rtk_h_mm rtk_v_mm corr_rot_dps corr_trans_cmps";

impl KnotQuality {
    pub fn row(&self) -> String {
        let f = |v: f64| {
            if v.is_finite() {
                format!("{v:.4}")
            } else {
                "nan".into()
            }
        };
        format!(
            "{:.6} {} {} {} {} {} {} {}",
            self.t,
            f(self.plane_rms_mm),
            f(self.plane_frac),
            f(self.imu_chi2),
            f(self.rtk_h_mm),
            f(self.rtk_v_mm),
            f(self.corr_rot_dps),
            f(self.corr_trans_cmps)
        )
    }
}

/// Knots over which the correction is measured (about one second).
const CORR_WINDOW: usize = 10;

impl Problem {
    pub fn quality(&self, prm: &Params) -> Vec<KnotQuality> {
        let k = self.kt.len();
        let pre = self.preintegrate(&prm.noise);
        // planes of the final stage on the current map
        let stage = prm.stages.last().expect("at least one stage");
        let (w, sv, _) = self.world(&pre, false);
        let pl = planes::extract(&w, &sv, &stage.planes);
        drop(sv);
        let off = &self.pts.off;
        let plane: Vec<(f64, f64)> = (0..k)
            .into_par_iter()
            .map(|j| {
                let (a, b) = (off[j], off[j + 1]);
                if b == a {
                    return (f64::NAN, f64::NAN);
                }
                let (mut sq, mut n) = (0.0, 0usize);
                for (&id, wi) in pl.pid[a..b].iter().zip(&w[a..b]) {
                    if id == NONE {
                        continue;
                    }
                    let p = &pl.planes[id as usize];
                    let q = V3::new(wi[0] as f64, wi[1] as f64, wi[2] as f64);
                    let r = p.n.dot(&(q - p.c));
                    sq += r * r;
                    n += 1;
                }
                let rms = if n > 0 {
                    (sq / n as f64).sqrt() * 1e3
                } else {
                    f64::NAN
                };
                (rms, n as f64 / (b - a) as f64)
            })
            .collect();
        drop(w);
        let imu: Vec<f64> = (0..k)
            .into_par_iter()
            .map(|j| {
                let Some(Some(pi)) = pre.get(j) else {
                    return f64::NAN;
                };
                let (r, _, _) = imu_residual(pi, &self.states[j], &self.states[j + 1]);
                match pi.cov.try_inverse() {
                    Some(info) => r.dot(&(info * r)) / 9.0,
                    None => f64::NAN,
                }
            })
            .collect();
        // RTK residuals per knot (mean of the fixes in its interval)
        let mut rtk = vec![(0.0, 0.0, 0usize); k];
        for t in self.rtk_terms(&self.states, &pre, self.rtk_prm.c, false) {
            let e = &mut rtk[t.knot];
            e.0 += t.e.xy().norm();
            e.1 += t.e.z.abs();
            e.2 += 1;
        }
        // correction of the relative motion over a window, against the odometry
        let corr: Vec<(f64, f64)> = (0..k)
            .into_par_iter()
            .map(|j| {
                let e = (j + CORR_WINDOW).min(k - 1);
                if e == j || self.session[e] != self.session[j] {
                    return (f64::NAN, f64::NAN);
                }
                let dt = self.kt[e] - self.kt[j];
                let (a0, b0) = (&self.init[j], &self.init[e]);
                let (a1, b1) = (&self.states[j], &self.states[e]);
                let r0 = a0.r.transpose() * b0.r;
                let r1 = a1.r.transpose() * b1.r;
                let p0 = a0.r.transpose() * (b0.p - a0.p);
                let p1 = a1.r.transpose() * (b1.p - a1.p);
                let rot = so3::log(&(r0.transpose() * r1)).norm().to_degrees() / dt;
                let tr = (p1 - p0).norm() * 100.0 / dt;
                (rot, tr)
            })
            .collect();
        (0..k)
            .map(|j| {
                let (h, v) = if rtk[j].2 > 0 {
                    let n = rtk[j].2 as f64;
                    (rtk[j].0 / n * 1e3, rtk[j].1 / n * 1e3)
                } else {
                    (f64::NAN, f64::NAN)
                };
                KnotQuality {
                    t: self.kt[j],
                    plane_rms_mm: plane[j].0,
                    plane_frac: plane[j].1,
                    imu_chi2: imu[j],
                    rtk_h_mm: h,
                    rtk_v_mm: v,
                    corr_rot_dps: corr[j].0,
                    corr_trans_cmps: corr[j].1,
                }
            })
            .collect()
    }
}
