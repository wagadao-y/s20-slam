//! Unified continuous-time trajectory optimisation.
//!
//! The trajectory is a chain of knot states x_k = (R, p, v, bg, ba) at the LIO scan ends (~0.1 s). Between
//! knots the motion is the IMU integration from the previous knot, so every LiDAR point gets its own pose
//! without interpolating corrections. One cost holds everything:
//! - IMU preintegration factors between consecutive knots and bias random walks,
//! - point-to-plane residuals on adaptive voxel planes of the whole map (BALM-style alternation: planes are
//!   re-extracted from the current map before every Gauss-Newton step), robust Cauchy weights,
//! - a gauge prior on the first knot.
//!
//! With planes fixed during a step, the normal equations are block tridiagonal and are solved directly.

pub mod coupling;
pub mod gnss_check;
pub mod imu;
pub mod links;
pub mod planes;
pub mod quality;
pub mod relpose;
pub mod tridiag;

#[cfg(test)]
mod pose_graph_regression;

use anyhow::{Result, bail};
use coupling::{Coupling, PlaneCoupling};
use imu::{ImuData, M15, Noise, Preint, State, V15, imu_residual};
use nalgebra::{SMatrix, SVector};
use planes::{NONE, PlaneParams};
use rayon::prelude::*;
use s20_geom::so3::{self, M3, V3, skew};
use s20_io::dataset::LidarData;
use std::ops::AddAssign;
use tridiag::BlockTridiag;

/// One session's LIO rows: poses `t x y z qx qy qz qw` and states `t v bg ba`.
pub type SessionRows = (Vec<[f64; 8]>, Vec<Vec<f64>>);
/// Linearised IMU factor between knots j and j+1: H_aa, H_ab, H_bb, g_a, g_b, chi2.
type ImuTerm = (M15, M15, M15, V15, V15, f64);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Stage {
    pub iters: usize,
    pub planes: PlaneParams,
    pub c_cauchy: f64,
    /// plane centroid coupling in this stage (with Params::couple)
    pub couple: bool,
    /// GNSS Huber threshold in this stage (None: RtkParams::c); coarse stages start from a loose fit
    pub rtk_c: Option<f64>,
    /// pose-graph stage: relative pose factors of the odometry keep the shape (no planes)
    pub odometry: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Params {
    pub every: usize,
    pub rmin: f64,
    pub rmax: f64,
    pub lidar_imu_t: [f64; 3],
    pub imu_shift: f64,
    pub noise: Noise,
    /// point noise (m) and extra scale of the LiDAR information (correlated points)
    pub sigma_pt: f64,
    pub lidar_w: f64,
    /// no IMU factor across larger knot gaps (s)
    pub max_gap: f64,
    pub stages: Vec<Stage>,
    pub lambda: f64,
    /// plane centroid coupling between knots (solved by PCG); false = fixed planes (block tridiagonal)
    pub couple: bool,
    pub cg_tol: f64,
    /// CG tolerance of the plane stages when there are loops (the pose-graph stage, which spreads
    /// the loop corrections, solves them out fully)
    pub loop_cg_tol: f64,
    pub cg_max: usize,
    pub lm_tries: usize,
    /// iterations of the pose-graph stages: run first with GNSS (before the loops are registered) and
    /// again when there are loops
    pub loop_pre_iters: usize,
    /// extra iterations of the last stage while the largest step exceeds conv_mm / conv_deg
    pub extra_iters: usize,
    pub conv_mm: f64,
    pub conv_deg: f64,
    /// registration of the loops the user marked
    pub loop_reg: LoopParams,
}

/// A stage on its own (a stage added to the list starts as the finest one).
impl Default for Stage {
    fn default() -> Self {
        Stage {
            iters: 10,
            planes: PlaneParams::default(),
            c_cauchy: 0.03,
            couple: true,
            rtk_c: None,
            odometry: false,
        }
    }
}

/// Registration of a loop: the local maps around the two times are aligned, and the measured
/// relative pose becomes a factor when the fit is good and unambiguous.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LoopParams {
    /// the local map holds the knots within this many seconds of the loop time
    pub half_s: f64,
    /// and the points within this distance of its knot (m)
    pub radius: f64,
    /// accepted when at least this fraction of the points match at the finest distance
    pub min_inliers: f64,
    /// and the guesses that reach the best fit agree within this distance (m)
    pub max_spread: f64,
    /// standard deviations of the loop factor (deg, m)
    pub sig_r_deg: f64,
    pub sig_t: f64,
    /// measure the loops again on the trajectory after each plane stage but the last: the first
    /// measurement places the local maps with the odometry, whose distortion biases it
    pub remeasure: bool,
}

impl Default for LoopParams {
    fn default() -> Self {
        LoopParams {
            half_s: 1.5,
            radius: 30.0,
            min_inliers: 0.4,
            max_spread: 0.1,
            sig_r_deg: 0.2,
            sig_t: 0.02,
            remeasure: true,
        }
    }
}

impl Default for Params {
    fn default() -> Self {
        let pp = |levels: &[f64], max_std: f64| PlaneParams {
            levels: levels.to_vec(),
            max_std,
            ..Default::default()
        };
        Params {
            // indoors and outdoors: every 4 gave the same clouds (haze, gaps, thickness against the
            // maker's) indoors and in the park 2026-10-20, in a half to a third of the time
            every: 8,
            rmin: 0.5,
            rmax: 60.0,
            lidar_imu_t: s20_lio::LIDAR_IMU_T,
            imu_shift: 0.005,
            noise: Noise::default(),
            sigma_pt: 0.01,
            lidar_w: 1.0,
            max_gap: 0.5,
            stages: vec![
                Stage {
                    iters: 6,
                    planes: pp(&[2.0, 1.0, 0.5], 0.08),
                    c_cauchy: 0.15,
                    couple: true,
                    rtk_c: Some(1.0),
                    odometry: false,
                },
                Stage {
                    iters: 6,
                    planes: pp(&[1.0, 0.5, 0.25], 0.05),
                    c_cauchy: 0.08,
                    couple: true,
                    rtk_c: Some(0.2),
                    odometry: false,
                },
                Stage {
                    iters: 10,
                    planes: pp(&[1.0, 0.5, 0.25, 0.125], 0.025),
                    c_cauchy: 0.03,
                    couple: true,
                    rtk_c: None,
                    odometry: false,
                },
            ],
            lambda: 1e-6,
            couple: true,
            cg_tol: 1e-4,
            loop_cg_tol: 1e-4,
            cg_max: 600,
            lm_tries: 6,
            loop_pre_iters: 8,
            extra_iters: 10,
            conv_mm: 5.0,
            conv_deg: 0.02,
            loop_reg: LoopParams::default(),
        }
    }
}

/// LiDAR points attached to knots (sorted by knot).
pub struct Points {
    /// IMU-frame position
    pub x: Vec<[f32; 3]>,
    /// time after the knot
    pub tau: Vec<f32>,
    /// points of knot k: off[k]..off[k+1]
    pub off: Vec<usize>,
}

impl Points {
    pub fn len(&self) -> usize {
        self.x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

pub struct Problem {
    pub kt: Vec<f64>,
    pub states: Vec<State>,
    /// the states as given (odometry), to measure the corrections
    pub init: Vec<State>,
    /// interval k -> k+1 has an IMU factor
    pub linked: Vec<bool>,
    pub pts: Points,
    pub imu: ImuData,
    /// session of each knot (sessions are separate IMU chains, coupled through the map and RTK)
    pub session: Vec<usize>,
    /// gauge priors: (knot, state) at the first knot of every session
    priors: Vec<(usize, State)>,
    /// per session: no gauge prior (held by its links to other sessions)
    free: Vec<bool>,
    /// current Levenberg-Marquardt damping
    lambda: f64,
    /// loops: two knots at the same place, with the measured relative pose
    pub loops: Vec<relpose::RelPose>,
    /// RTK antenna observations (empty without RTK)
    pub rtk: Vec<RtkObs>,
    pub rtk_prm: RtkParams,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RtkParams {
    pub sig_h: f64,
    pub sig_v: f64,
    /// Huber threshold (m): beyond it a fix pulls with a constant force. Unlike a redescending kernel, a
    /// correct fix far from a drifted trajectory is not discarded; false fixes are screened beforehand.
    pub c: f64,
    /// antenna position in the IMU frame
    pub lever: [f64; 3],
    /// false-fix screening against the odometry
    pub check: gnss_check::CheckParams,
}

impl Default for RtkParams {
    fn default() -> Self {
        let l = s20_io::gnss::RTK_OFFSET;
        let t = s20_lio::LIDAR_IMU_T;
        RtkParams {
            sig_h: 0.01,
            sig_v: 0.02,
            c: 0.05,
            lever: [l[0] + t[0], l[1] + t[1], l[2] + t[2]],
            check: gnss_check::CheckParams::default(),
        }
    }
}

/// Result of attaching GNSS fixes.
#[derive(Clone, Debug)]
pub struct GnssSetup {
    /// fixes inside the trajectory kept as factors
    pub used: usize,
    /// rms horizontal residual after the initial alignment (m)
    pub rms: f64,
    /// segments removed by the false-fix check
    pub rejected: Vec<gnss_check::Rejected>,
    /// sessions with too few fixes to be placed by them (left where they are)
    pub unplaced: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct RtkObs {
    pub t: f64,
    pub knot: usize,
    pub tau: f64,
    pub p: V3,
    /// ENU standard deviation; zero retains legacy RTK weighting.
    pub sd: V3,
}

pub fn load_points(data: &LidarData, kt: &[f64], linked: &[bool], prm: &Params) -> Points {
    let lever = V3::from(prm.lidar_imu_t);
    let k = kt.len();
    let (t0, t1) = (kt[0], kt[k - 1]);
    let per_scan: Vec<Vec<(u32, f32, [f32; 3])>> = (0..data.len())
        .into_par_iter()
        .filter(|&i| {
            let s = data.scan_start(i);
            s < t1 && s + 0.2 > t0
        })
        .map(|i| {
            let sc = data.scan(i);
            let mut v = Vec::new();
            for n in (0..sc.t.len()).step_by(prm.every.max(1)) {
                if sc.tag[n] & 0x3F != 0 {
                    continue;
                }
                let q = V3::from(sc.xyz[n]);
                let r = q.norm();
                if r <= prm.rmin || r >= prm.rmax {
                    continue;
                }
                let t = sc.t[n];
                let j = kt.partition_point(|&s| s <= t);
                if j == 0 || j >= k {
                    continue;
                }
                let j = j - 1;
                if !linked[j] {
                    continue;
                }
                let x = q + lever;
                v.push((
                    j as u32,
                    (t - kt[j]) as f32,
                    [x.x as f32, x.y as f32, x.z as f32],
                ));
            }
            v
        })
        .collect();
    let mut cnt = vec![0usize; k + 1];
    for v in &per_scan {
        for p in v {
            cnt[p.0 as usize + 1] += 1;
        }
    }
    for i in 0..k {
        cnt[i + 1] += cnt[i];
    }
    let n = cnt[k];
    let mut x = vec![[0f32; 3]; n];
    let mut tau = vec![0f32; n];
    let mut fill = cnt.clone();
    for v in &per_scan {
        for p in v {
            let d = &mut fill[p.0 as usize];
            x[*d] = p.2;
            tau[*d] = p.1;
            *d += 1;
        }
    }
    Points { x, tau, off: cnt }
}

#[derive(Debug, Default, Clone)]
pub struct IterStats {
    pub stage: usize,
    pub it: usize,
    pub planes: usize,
    pub used: f64,
    pub rms_mm: f64,
    pub imu_chi2: f64,
    pub step_rot_deg: f64,
    pub step_mm: f64,
    /// time (from the first knot) of the largest rotation step
    pub step_at: f64,
    pub sec: f64,
    pub cg_iters: usize,
    pub cost0: f64,
    pub cost1: f64,
    pub lambda: f64,
    pub tries: usize,
    pub rtk_h_mm: f64,
    pub rtk_v_mm: f64,
}

struct RtkTerm {
    knot: usize,
    h: SMatrix<f64, 9, 9>,
    g: SVector<f64, 9>,
    cost: f64,
    e: V3,
}

impl Problem {
    /// Knots from LIO rows: poses `t x y z qx qy qz qw` and states `t v bg ba`, one (poses, states) pair per
    /// session in time order.
    pub fn new(
        data: &LidarData,
        imu_rows: &[[f64; 7]],
        sessions: &[SessionRows],
        prm: &Params,
    ) -> Result<Self> {
        let mut kt = Vec::new();
        let mut states = Vec::new();
        let mut session = Vec::new();
        for (si, (poses, st)) in sessions.iter().enumerate() {
            if poses.len() != st.len() || poses.len() < 2 {
                bail!(
                    "session {si}: pose ({}) and state ({}) rows differ",
                    poses.len(),
                    st.len()
                );
            }
            for (q, s) in poses.iter().zip(st) {
                if (q[0] - s[0]).abs() > 1e-6 || s.len() < 10 {
                    bail!("state rows must match the pose rows (t v bg ba)");
                }
                if let Some(&l) = kt.last()
                    && q[0] <= l
                {
                    continue;
                }
                kt.push(q[0]);
                session.push(si);
                states.push(State {
                    r: so3::from_quat([q[4], q[5], q[6], q[7]]),
                    p: V3::new(q[1], q[2], q[3]),
                    v: V3::new(s[1], s[2], s[3]),
                    bg: V3::new(s[4], s[5], s[6]),
                    ba: V3::new(s[7], s[8], s[9]),
                });
            }
        }
        let linked: Vec<bool> = (0..kt.len() - 1)
            .map(|k| kt[k + 1] - kt[k] <= prm.max_gap && session[k] == session[k + 1])
            .collect();
        let pts = load_points(data, &kt, &linked, prm);
        let imu = ImuData::from_rows(imu_rows, prm.imu_shift);
        let priors: Vec<(usize, State)> = (0..kt.len())
            .filter(|&k| k == 0 || session[k] != session[k - 1])
            .map(|k| (k, states[k]))
            .collect();
        let init = states.clone();
        Ok(Problem {
            kt,
            states,
            init,
            linked,
            pts,
            imu,
            free: vec![false; priors.len()],
            session,
            priors,
            lambda: prm.lambda,
            rtk: Vec::new(),
            rtk_prm: RtkParams::default(),
            loops: Vec::new(),
        })
    }

    pub fn n_sessions(&self) -> usize {
        self.session.last().map_or(0, |&s| s + 1)
    }

    pub fn preintegrate(&self, nz: &Noise) -> Vec<Option<Preint>> {
        (0..self.kt.len() - 1)
            .into_par_iter()
            .map(|k| {
                self.linked[k].then(|| {
                    Preint::new(
                        &self.imu,
                        self.kt[k],
                        self.kt[k + 1],
                        &self.states[k].bg,
                        &self.states[k].ba,
                        nz,
                    )
                })
            })
            .collect()
    }

    fn preintegrate_states(&self, states: &[State], nz: &Noise) -> Vec<Option<Preint>> {
        (0..self.kt.len() - 1)
            .into_par_iter()
            .map(|k| {
                self.linked[k].then(|| {
                    Preint::new(
                        &self.imu,
                        self.kt[k],
                        self.kt[k + 1],
                        &states[k].bg,
                        &states[k].ba,
                        nz,
                    )
                })
            })
            .collect()
    }

    #[inline]
    fn point(&self, s: &State, pi: &Preint, i: usize) -> V3 {
        let x = V3::new(
            self.pts.x[i][0] as f64,
            self.pts.x[i][1] as f64,
            self.pts.x[i][2] as f64,
        );
        let (r, p) = s.pose_at(pi, self.pts.tau[i] as f64);
        r * x + p
    }

    /// Total cost at `states` for a fixed plane assignment: robust point-to-plane terms (plane normal fixed,
    /// centroid = mean of the plane points weighted by `wts`), IMU factors, bias random walks, gauge prior.
    #[allow(clippy::needless_range_loop)] // `i` indexes the points and their weights together
    #[allow(clippy::too_many_arguments)]
    /// `known`: `point` of every point at `states`, when already computed.
    fn cost(
        &self,
        states: &[State],
        known: Option<&[V3]>,
        prm: &Params,
        pl: &planes::Planes,
        wts: &[f32],
        cc: f64,
        rc: f64,
        rels: &[relpose::RelPose],
        pose_graph: bool,
    ) -> f64 {
        let k = self.kt.len();
        let nz = prm.noise;
        let pre = if pose_graph && pl.planes.is_empty() && self.rtk.is_empty() {
            Vec::new()
        } else {
            self.preintegrate_states(states, &nz)
        };
        let lidar = if pl.planes.is_empty() {
            0.0
        } else {
            let np = pl.planes.len();
            let off = &self.pts.off;
            // weighted centroids: sums over fixed chunks of knots, added in order (the same sums
            // whatever the thread scheduling); the plane points of each chunk are kept for the
            // residuals below (computed once, unless `known`)
            const CHUNK: usize = 512;
            let (parts, qs): (Vec<Vec<[f64; 4]>>, Vec<Vec<V3>>) = (0..k.div_ceil(CHUNK))
                .into_par_iter()
                .map(|c| {
                    let mut a = vec![[0f64; 4]; np];
                    let mut qc = Vec::new();
                    for j in c * CHUNK..((c + 1) * CHUNK).min(k) {
                        if let Some(Some(pi)) = pre.get(j) {
                            for i in off[j]..off[j + 1] {
                                let id = pl.pid[i];
                                if id == NONE {
                                    continue;
                                }
                                let q = match known {
                                    Some(kq) => kq[i],
                                    None => self.point(&states[j], pi, i),
                                };
                                let w = wts[i] as f64;
                                let e = &mut a[id as usize];
                                e[0] += w * q.x;
                                e[1] += w * q.y;
                                e[2] += w * q.z;
                                e[3] += w;
                                if known.is_none() {
                                    qc.push(q);
                                }
                            }
                        }
                    }
                    (a, qc)
                })
                .unzip();
            let sums: Vec<[f64; 4]> = (0..np)
                .into_par_iter()
                .map(|p| {
                    let mut s = [0f64; 4];
                    for a in &parts {
                        for d in 0..4 {
                            s[d] += a[p][d];
                        }
                    }
                    s
                })
                .collect();
            let cen: Vec<V3> = sums
                .iter()
                .map(|s| {
                    if s[3] > 0.0 {
                        V3::new(s[0], s[1], s[2]) / s[3]
                    } else {
                        V3::zeros()
                    }
                })
                .collect();
            let wpt = prm.lidar_w / (prm.sigma_pt * prm.sigma_pt);
            // per knot, summed in knot order
            let (pre, cen) = (&pre, &cen);
            let lidar: f64 = qs
                .par_iter()
                .enumerate()
                .flat_map_iter(|(c, qc)| {
                    let mut qc = qc.iter();
                    (c * CHUNK..((c + 1) * CHUNK).min(k)).map(move |j| {
                        let Some(Some(_)) = pre.get(j) else {
                            return 0.0;
                        };
                        let mut c = 0.0;
                        for i in off[j]..off[j + 1] {
                            let id = pl.pid[i];
                            if id == NONE {
                                continue;
                            }
                            let q = match known {
                                Some(kq) => &kq[i],
                                None => qc.next().expect("a point per plane point"),
                            };
                            let r = pl.planes[id as usize].n.dot(&(q - cen[id as usize]));
                            c += 0.5 * cc * cc * (r / cc * (r / cc)).ln_1p();
                        }
                        c * wpt
                    })
                })
                .collect::<Vec<f64>>()
                .iter()
                .sum();
            lidar
        };
        let imu: f64 = (0..k - 1)
            .into_par_iter()
            .map(|j| {
                if pose_graph {
                    return 0.0;
                }
                let Some(pi) = pre[j].as_ref() else {
                    return 0.0;
                };
                let (a, b) = (&states[j], &states[j + 1]);
                let (r, _, _) = imu_residual(pi, a, b);
                let Some(info) = pi.cov.try_inverse() else {
                    return 0.0;
                };
                let mut c = 0.5 * r.dot(&(info * r));
                c += 0.5 * (b.bg - a.bg).norm_squared() / (nz.nbg * nz.nbg * pi.dt);
                c += 0.5 * (b.ba - a.ba).norm_squared() / (nz.nba * nz.nba * pi.dt);
                c
            })
            .collect::<Vec<f64>>()
            .iter()
            .sum();
        let rtk: f64 = self
            .rtk_terms(states, &pre, rc, pose_graph)
            .iter()
            .map(|t| t.cost)
            .sum();
        let rel: f64 = rels.iter().map(|r| r.term(states).cost).sum();
        lidar + imu + rtk + rel + self.prior_cost(states)
    }

    fn prior_weights(&self, s: &State) -> (M3, f64, f64, f64) {
        // world-frame rotation weights: yaw tight, tilt loose (gravity observes it). With RTK the position and
        // yaw are observed, the prior only keeps the system regular.
        let (sy, sp) = if self.rtk.is_empty() {
            (1e-5f64, 1e-4f64)
        } else {
            (10f64.to_radians(), 10.0)
        };
        let wy = 1.0 / sy.powi(2);
        let wt = 1.0 / (1f64.to_radians()).powi(2);
        let wmat = s.r.transpose() * M3::from_diagonal(&V3::new(wt, wt, wy)) * s.r;
        (
            wmat,
            1.0 / sp.powi(2),
            1.0 / (0.1f64.to_radians()).powi(2),
            1.0 / 0.05f64.powi(2),
        )
    }

    /// Antenna position of state `s` at `tau` after its knot, and the body offset a = dR lever + dp.
    fn antenna(&self, s: &State, pi: &Preint, tau: f64) -> (V3, V3) {
        let (dr, dp) = pi.at(tau);
        let a = dr * V3::from(self.rtk_prm.lever) + dp;
        (
            s.r * a + s.p + s.v * tau + 0.5 * imu::gravity() * tau * tau,
            a,
        )
    }

    /// Robust GNSS terms: residual e = antenna - fix, normalised by (sig_h, sig_h, sig_v), Huber on |u|.
    /// `pose_graph`: the velocities stay fixed. Without IMU factors a knot's velocity would otherwise
    /// absorb the fix between knots (antenna = ... + v tau) instead of moving the pose.
    fn rtk_terms(
        &self,
        states: &[State],
        pre: &[Option<Preint>],
        c: f64,
        pose_graph: bool,
    ) -> Vec<RtkTerm> {
        let rp = &self.rtk_prm;
        let kh = c / rp.sig_h;
        self.rtk
            .par_iter()
            .filter_map(|o| {
                let sig = V3::new(
                    o.sd.x.max(rp.sig_h),
                    o.sd.y.max(rp.sig_h),
                    o.sd.z.max(rp.sig_v),
                );
                let pi = pre.get(o.knot)?.as_ref()?;
                let s = &states[o.knot];
                let (q, a) = self.antenna(s, pi, o.tau);
                let e = q - o.p;
                let u = e.component_div(&sig);
                let s2 = u.norm_squared();
                let s1 = s2.sqrt();
                let (w, cost) = if s1 <= kh {
                    (1.0, 0.5 * s2)
                } else {
                    (kh / s1, kh * s1 - 0.5 * kh * kh)
                };
                let mut j = SMatrix::<f64, 3, 9>::zeros();
                j.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-s.r * skew(&a)));
                j.fixed_view_mut::<3, 3>(0, 3).copy_from(&M3::identity());
                if !pose_graph {
                    j.fixed_view_mut::<3, 3>(0, 6)
                        .copy_from(&(M3::identity() * o.tau));
                }
                let wm = M3::from_diagonal(&sig.map(|x| w / (x * x)));
                Some(RtkTerm {
                    knot: o.knot,
                    h: j.transpose() * wm * j,
                    g: j.transpose() * wm * e,
                    cost,
                    e,
                })
            })
            .collect()
    }

    /// Attach RTK fixes: align the (gravity-aligned) trajectory to them by yaw and translation (robust),
    /// screen false fixes against the odometry, then keep the rest as factors.
    pub fn set_rtk(
        &mut self,
        fixes: &[s20_io::gnss::Fix],
        prm: &Params,
        rp: RtkParams,
    ) -> Result<GnssSetup> {
        self.set_gnss(fixes, prm, rp, false)
    }

    /// PPK uses per-epoch uncertainty, with the configured RTK sigmas as lower bounds.
    pub fn set_gnss(
        &mut self,
        fixes: &[s20_io::gnss::Fix],
        prm: &Params,
        rp: RtkParams,
        uncertainty: bool,
    ) -> Result<GnssSetup> {
        self.rtk_prm = rp;
        let k = self.kt.len();
        let mut obs = Vec::new();
        for f in fixes {
            let j = self.kt.partition_point(|&s| s <= f.t);
            if j == 0 || j >= k || !self.linked[j - 1] {
                continue;
            }
            obs.push(RtkObs {
                t: f.t,
                knot: j - 1,
                tau: f.t - self.kt[j - 1],
                p: V3::from(f.p),
                sd: if uncertainty {
                    V3::from(f.sd)
                } else {
                    V3::zeros()
                },
            });
        }
        if obs.len() < 10 {
            bail!("only {} RTK fixes inside the trajectory", obs.len());
        }
        let pre = self.preintegrate(&prm.noise);
        let mut rms_all = Vec::new();
        // a session with too few fixes is placed by its links to the others
        let unplaced: Vec<usize> = (0..self.n_sessions())
            .filter(|&si| obs.iter().filter(|o| self.session[o.knot] == si).count() < 10)
            .collect();
        obs.retain(|o| !unplaced.contains(&self.session[o.knot]));
        for si in 0..self.n_sessions() {
            if unplaced.contains(&si) {
                continue;
            }
            let so: Vec<&RtkObs> = obs.iter().filter(|o| self.session[o.knot] == si).collect();
            let ant: Vec<V3> = so
                .iter()
                .map(|o| {
                    self.antenna(&self.states[o.knot], pre[o.knot].as_ref().unwrap(), o.tau)
                        .0
                })
                .collect();
            // robust 2D Procrustes (yaw) + translation
            let mut use_ = vec![true; so.len()];
            let (mut yaw, mut tr) = (0.0, V3::zeros());
            let mut rms = 0.0;
            for _ in 0..5 {
                let n = use_.iter().filter(|u| **u).count().max(1) as f64;
                let ma: V3 = ant
                    .iter()
                    .zip(&use_)
                    .filter(|(_, u)| **u)
                    .map(|(a, _)| a)
                    .sum::<V3>()
                    / n;
                let mb: V3 = so
                    .iter()
                    .zip(&use_)
                    .filter(|(_, u)| **u)
                    .map(|(o, _)| o.p)
                    .sum::<V3>()
                    / n;
                let (mut sxx, mut sxy) = (0.0, 0.0);
                for ((a, o), u) in ant.iter().zip(&so).zip(&use_) {
                    if *u {
                        let (da, db) = (a - ma, o.p - mb);
                        sxx += da.x * db.x + da.y * db.y;
                        sxy += da.x * db.y - da.y * db.x;
                    }
                }
                yaw = sxy.atan2(sxx);
                let rz = so3::exp(&V3::new(0.0, 0.0, yaw));
                tr = mb - rz * ma;
                let res: Vec<f64> = ant
                    .iter()
                    .zip(&so)
                    .map(|(a, o)| (rz * a + tr - o.p).norm())
                    .collect();
                let mut sr: Vec<f64> = res.clone();
                sr.sort_by(f64::total_cmp);
                let thr = (3.0 * sr[sr.len() / 2]).max(0.3);
                use_ = res.iter().map(|r| *r < thr).collect();
                let used: Vec<f64> = res
                    .iter()
                    .zip(&use_)
                    .filter(|(_, u)| **u)
                    .map(|(r, _)| r * r)
                    .collect();
                rms = (used.iter().sum::<f64>() / used.len().max(1) as f64).sqrt();
            }
            let rz = so3::exp(&V3::new(0.0, 0.0, yaw));
            for (k, s) in self.states.iter_mut().enumerate() {
                if self.session[k] == si {
                    s.r = rz * s.r;
                    s.p = rz * s.p + tr;
                    s.v = rz * s.v;
                }
            }
            rms_all.push(rms);
        }
        for pr in self.priors.iter_mut() {
            pr.1 = self.states[pr.0];
        }
        let rms = rms_all.iter().cloned().fold(0.0, f64::max);
        // false-fix screening per session, on the aligned odometry
        let lio_bad: Vec<f64> = (0..k.saturating_sub(1))
            .filter(|&j| self.linked[j] && self.session[j] == self.session[j + 1])
            .filter(|&j| {
                let (a, b) = (&self.init[j], &self.init[j + 1]);
                let dt = self.kt[j + 1] - self.kt[j];
                (b.p - a.p - 0.5 * (a.v + b.v) * dt).norm() > rp.check.lio_tol
            })
            .map(|j| self.kt[j + 1])
            .collect();
        obs.sort_by(|a, b| a.t.total_cmp(&b.t));
        let mut keep = vec![true; obs.len()];
        let mut rejected = Vec::new();
        for si in 0..self.n_sessions() {
            let idx: Vec<usize> = (0..obs.len())
                .filter(|&i| self.session[obs[i].knot] == si)
                .collect();
            let t: Vec<f64> = idx.iter().map(|&i| obs[i].t).collect();
            let o: Vec<V3> = idx
                .iter()
                .map(|&i| {
                    let ob = &obs[i];
                    let pi = pre[ob.knot].as_ref().unwrap();
                    ob.p - self.antenna(&self.states[ob.knot], pi, ob.tau).0
                })
                .collect();
            let (kp, rj) = gnss_check::screen(&t, &o, &lio_bad, &rp.check);
            for (&i, k) in idx.iter().zip(kp) {
                keep[i] = k;
            }
            rejected.extend(rj);
        }
        let mut it = keep.iter();
        obs.retain(|_| *it.next().unwrap());
        if obs.len() < 10 {
            bail!(
                "only {} RTK fixes left after the false-fix check",
                obs.len()
            );
        }
        let used = obs.len();
        self.rtk = obs;
        Ok(GnssSetup {
            used,
            rms,
            rejected,
            unplaced,
        })
    }

    fn prior_cost(&self, states: &[State]) -> f64 {
        self.priors
            .iter()
            .filter(|(k, _)| !self.free[self.session[*k]])
            .map(|(k, p0)| {
                let s = &states[*k];
                let (wmat, wp, wbg, wba) = self.prior_weights(s);
                let rr = so3::log(&(p0.r.transpose() * s.r));
                0.5 * (rr.dot(&(wmat * rr))
                    + wp * (s.p - p0.p).norm_squared()
                    + wbg * (s.bg - p0.bg).norm_squared()
                    + wba * (s.ba - p0.ba).norm_squared())
            })
            .sum()
    }

    /// World points (single precision) and point -> sensor vectors evaluated with `pre`, and with
    /// `keep` the world points in double precision too (`point` of every point: the same values).
    fn world(&self, pre: &[Option<Preint>], keep: bool) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<V3>) {
        let n = self.pts.len();
        let mut w = vec![[0f32; 3]; n];
        let mut sv = vec![[0f32; 3]; n];
        let mut qs = if keep {
            vec![V3::zeros(); n]
        } else {
            Vec::new()
        };
        let off = &self.pts.off;
        // split the output buffers per knot
        let mut wchunks: Vec<&mut [[f32; 3]]> = Vec::with_capacity(self.kt.len());
        let mut schunks: Vec<&mut [[f32; 3]]> = Vec::with_capacity(self.kt.len());
        let mut qchunks: Vec<&mut [V3]> = Vec::with_capacity(self.kt.len());
        let (mut wr, mut sr, mut qr) = (&mut w[..], &mut sv[..], &mut qs[..]);
        for k in 0..self.kt.len() {
            let len = off[k + 1] - off[k];
            let (a, b) = wr.split_at_mut(len);
            let (c, d) = sr.split_at_mut(len);
            let (e, f) = qr.split_at_mut(if keep { len } else { 0 });
            wchunks.push(a);
            schunks.push(c);
            qchunks.push(e);
            wr = b;
            sr = d;
            qr = f;
        }
        wchunks
            .into_par_iter()
            .zip(schunks)
            .zip(qchunks)
            .enumerate()
            .for_each(|(k, ((wc, sc), qc))| {
                let Some(Some(pi)) = pre.get(k) else { return };
                let s = &self.states[k];
                for (m, i) in (off[k]..off[k + 1]).enumerate() {
                    let x = V3::new(
                        self.pts.x[i][0] as f64,
                        self.pts.x[i][1] as f64,
                        self.pts.x[i][2] as f64,
                    );
                    let (r, p) = s.pose_at(pi, self.pts.tau[i] as f64);
                    let rx = r * x;
                    let q = rx + p;
                    wc[m] = [q.x as f32, q.y as f32, q.z as f32];
                    sc[m] = [-rx.x as f32, -rx.y as f32, -rx.z as f32];
                    if let Some(qm) = qc.get_mut(m) {
                        *qm = q;
                    }
                }
            });
        (w, sv, qs)
    }

    /// One Gauss-Newton step with planes re-extracted from the current map.
    pub fn step(&mut self, prm: &Params, stage: &Stage) -> Result<IterStats> {
        let t0 = std::time::Instant::now();
        let k = self.kt.len();
        // A pure pose-graph step needs neither deskewed points nor IMU integration.
        // RTK still needs preintegration to evaluate the antenna between knots.
        let pre = if stage.odometry && stage.planes.levels.is_empty() && self.rtk.is_empty() {
            Vec::new()
        } else {
            self.preintegrate(&prm.noise)
        };
        // the world points in double precision are kept for the cost at the current states
        let (pl, qs) = if stage.planes.levels.is_empty() {
            let pl = planes::Planes {
                pid: Vec::new(),
                planes: Vec::new(),
            };
            (pl, None)
        } else {
            let (w, sv, qs) = self.world(&pre, true);
            (planes::extract(&w, &sv, &stage.planes), Some(qs))
        };
        let mut h = BlockTridiag::new(k);
        // --- LiDAR plane residuals (each point touches only its knot)
        let wpt = prm.lidar_w / (prm.sigma_pt * prm.sigma_pt);
        let cc = stage.c_cauchy;
        let rc = stage.rtk_c.unwrap_or(self.rtk_prm.c).max(self.rtk_prm.c);
        let couple = prm.couple && stage.couple;
        let (mut sq, mut used) = (0.0, 0usize);
        let mut cps: Vec<Coupling> = Vec::new();
        let mut wts: Vec<f32> = Vec::new();
        if !pl.planes.is_empty() {
            let off = &self.pts.off;
            type Lid = (
                SMatrix<f64, 9, 9>,
                SVector<f64, 9>,
                f64,
                usize,
                Vec<Coupling>,
                Vec<f32>,
            );
            let lid: Vec<Lid> = (0..k)
                .into_par_iter()
                .map(|j| {
                    let mut hh = SMatrix::<f64, 9, 9>::zeros();
                    let mut gg = SVector::<f64, 9>::zeros();
                    let mut sq = 0.0;
                    let mut used = 0;
                    let mut cp: Vec<Coupling> = Vec::new();
                    let mut wk = vec![0f32; off[j + 1] - off[j]];
                    let Some(Some(pi)) = pre.get(j) else {
                        return (hh, gg, sq, used, cp, wk);
                    };
                    let s = &self.states[j];
                    for i in off[j]..off[j + 1] {
                        let id = pl.pid[i];
                        if id == NONE {
                            continue;
                        }
                        let plane = &pl.planes[id as usize];
                        let x = V3::new(
                            self.pts.x[i][0] as f64,
                            self.pts.x[i][1] as f64,
                            self.pts.x[i][2] as f64,
                        );
                        let tau = self.pts.tau[i] as f64;
                        let (dr, dp) = pi.at(tau);
                        let a = dr * x + dp;
                        let q = s.r * a + s.p + s.v * tau + 0.5 * imu::gravity() * tau * tau;
                        let r = plane.n.dot(&(q - plane.c));
                        let wt = wpt / (1.0 + (r / cc) * (r / cc));
                        let m = s.r.transpose() * plane.n;
                        let jt = -m.cross(&a);
                        let mut jv = SVector::<f64, 9>::zeros();
                        jv.fixed_rows_mut::<3>(0).copy_from(&jt);
                        jv.fixed_rows_mut::<3>(3).copy_from(&plane.n);
                        jv.fixed_rows_mut::<3>(6).copy_from(&(plane.n * tau));
                        hh += wt * jv * jv.transpose();
                        gg += wt * r * jv;
                        sq += r * r;
                        used += 1;
                        wk[i - off[j]] = (wt / wpt) as f32;
                        if couple {
                            cp.push(Coupling {
                                plane: id,
                                knot: j as u32,
                                a: wt * jt,
                                w: wt,
                                wtau: wt * tau,
                                wr: wt * r,
                            });
                        }
                    }
                    if couple {
                        cp = coupling::merge(cp);
                    }
                    (hh, gg, sq, used, cp, wk)
                })
                .collect();
            wts.reserve(self.pts.len());
            for (j, (hh, gg, s, u, cp, wk)) in lid.into_iter().enumerate() {
                wts.extend(wk);
                h.d[j].fixed_view_mut::<9, 9>(0, 0).add_assign(&hh);
                h.g[j].fixed_rows_mut::<9>(0).add_assign(&gg);
                sq += s;
                used += u;
                cps.extend(cp);
            }
        }
        let coupling =
            couple.then(|| PlaneCoupling::new(cps, pl.planes.iter().map(|p| p.n).collect()));
        if let Some(c) = &coupling {
            c.correct_gradient(&mut h.g);
        }
        // --- RTK antenna positions (single knot each)
        let rtk_terms = self.rtk_terms(&self.states, &pre, rc, stage.odometry);
        let (mut rtk_h2, mut rtk_v2) = (0.0, 0.0);
        for t in &rtk_terms {
            h.d[t.knot].fixed_view_mut::<9, 9>(0, 0).add_assign(&t.h);
            h.g[t.knot].fixed_rows_mut::<9>(0).add_assign(&t.g);
            rtk_h2 += t.e.x * t.e.x + t.e.y * t.e.y;
            rtk_v2 += t.e.z * t.e.z;
        }
        let nr = rtk_terms.len().max(1) as f64;
        // --- IMU factors and bias random walks
        let nz = prm.noise;
        let imu_terms: Vec<Option<ImuTerm>> = (0..k - 1)
            .into_par_iter()
            .map(|j| {
                // the pose-graph stage is pure: IMU factors would hold large corrections back
                if stage.odometry {
                    return None;
                }
                let pi = pre[j].as_ref()?;
                let (a, b) = (&self.states[j], &self.states[j + 1]);
                let (r, ja, jb) = imu_residual(pi, a, b);
                let info = pi.cov.try_inverse()?;
                let info = 0.5 * (info + info.transpose());
                let mut haa = ja.transpose() * info * ja;
                let mut hab = ja.transpose() * info * jb;
                let mut hbb = jb.transpose() * info * jb;
                let mut ga = ja.transpose() * info * r;
                let mut gb = jb.transpose() * info * r;
                let chi2 = r.dot(&(info * r));
                // bias random walk
                for (o, sg) in [(9, nz.nbg), (12, nz.nba)] {
                    let wb = 1.0 / (sg * sg * pi.dt);
                    for i in 0..3 {
                        let d = (if o == 9 { b.bg - a.bg } else { b.ba - a.ba })[i];
                        let c = o + i;
                        haa[(c, c)] += wb;
                        hbb[(c, c)] += wb;
                        hab[(c, c)] -= wb;
                        ga[c] -= wb * d;
                        gb[c] += wb * d;
                    }
                }
                Some((haa, hab, hbb, ga, gb, chi2))
            })
            .collect();
        let (mut chi2, mut nf) = (0.0, 0usize);
        for (j, t) in imu_terms.into_iter().enumerate() {
            let Some((haa, hab, hbb, ga, gb, c)) = t else {
                continue;
            };
            h.d[j] += haa;
            h.d[j + 1] += hbb;
            h.u[j] += hab;
            h.g[j] += ga;
            h.g[j + 1] += gb;
            chi2 += c;
            nf += 1;
        }
        // --- gauge priors on the first knot of each session (position, yaw strongly; roll/pitch and biases weakly)
        for (k0, p0) in &self.priors {
            let k0 = *k0;
            if self.free[self.session[k0]] {
                continue;
            }
            let s = &self.states[k0];
            let rr = so3::log(&(p0.r.transpose() * s.r));
            let (wmat, wp, wbg, wba) = self.prior_weights(s);
            h.d[k0].fixed_view_mut::<3, 3>(0, 0).add_assign(&wmat);
            h.g[k0].fixed_rows_mut::<3>(0).add_assign(&(wmat * rr));
            for i in 0..3 {
                h.d[k0][(3 + i, 3 + i)] += wp;
                h.g[k0][3 + i] += wp * (s.p[i] - p0.p[i]);
                h.d[k0][(9 + i, 9 + i)] += wbg;
                h.g[k0][9 + i] += wbg * (s.bg[i] - p0.bg[i]);
                h.d[k0][(12 + i, 12 + i)] += wba;
                h.g[k0][12 + i] += wba * (s.ba[i] - p0.ba[i]);
            }
        }
        // knots without any constraint (isolated) get a weak prior to stay solvable
        for j in 0..k {
            for i in 0..15 {
                h.d[j][(i, i)] += 1e-6;
            }
        }
        // --- relative pose factors: loops, and the odometry in the pose-graph stage
        let mut rels = self.loops.clone();
        if stage.odometry {
            rels.extend(self.odometry_factors());
        }
        let mut far: Vec<(usize, usize, M15)> = Vec::new();
        for r in &rels {
            let t = r.term(&self.states);
            h.d[t.a] += relpose::embed(&t.haa);
            h.d[t.b] += relpose::embed(&t.hbb);
            h.g[t.a].fixed_rows_mut::<6>(0).add_assign(&t.ga);
            h.g[t.b].fixed_rows_mut::<6>(0).add_assign(&t.gb);
            if t.b == t.a + 1 {
                h.u[t.a] += relpose::embed(&t.hab);
            } else if t.a == t.b + 1 {
                h.u[t.b] += relpose::embed(&t.hab.transpose());
            } else {
                far.push((t.a, t.b, relpose::embed(&t.hab)));
            }
        }
        // --- Levenberg-Marquardt: accept the step only if the cost (same plane assignment) decreases
        let c0 = self.cost(
            &self.states,
            qs.as_deref(),
            prm,
            &pl,
            &wts,
            cc,
            rc,
            &rels,
            stage.odometry,
        );
        drop(qs);
        let mut lambda = self.lambda.max(prm.lambda);
        let mut cg_iters = 0;
        let mut accepted = None;
        let mut tries = 0;
        let mut c1 = c0;
        while tries < prm.lm_tries {
            tries += 1;
            // a loop is a small part of the residual next to the whole map: the pose-graph stage
            // solves it out fully, so that its correction reaches the whole trajectory
            let cg_tol = if far.is_empty() {
                prm.cg_tol
            } else if stage.odometry {
                prm.cg_tol.min(1e-10)
            } else {
                prm.cg_tol.min(prm.loop_cg_tol)
            };
            let (dx, it) = match (&coupling, far.is_empty()) {
                (None, true) => (h.solve(lambda)?, 0),
                (Some(c), _) => c.solve(&h, lambda, cg_tol, prm.cg_max, &far)?,
                (None, false) => PlaneCoupling::new(Vec::new(), Vec::new())
                    .solve(&h, lambda, cg_tol, prm.cg_max, &far)?,
            };
            cg_iters += it;
            let cand: Vec<State> = self
                .states
                .iter()
                .zip(&dx)
                .map(|(s, d)| {
                    let mut n = s.plus(d);
                    n.r = so3::normalize(&n.r);
                    n
                })
                .collect();
            c1 = self.cost(&cand, None, prm, &pl, &wts, cc, rc, &rels, stage.odometry);
            if c1.is_finite() && c1 <= c0 {
                self.states = cand;
                accepted = Some(dx);
                self.lambda = (lambda / 10.0).max(prm.lambda);
                break;
            }
            lambda *= 10.0;
        }
        if accepted.is_none() {
            self.lambda = lambda;
        }
        let dx = accepted.unwrap_or_else(|| vec![V15::zeros(); k]);
        let (mut mr, mut mt, mut at) = (0.0f64, 0.0f64, 0usize);
        for (j, d) in dx.iter().enumerate() {
            let a = V3::new(d[0], d[1], d[2]).norm();
            if a > mr {
                mr = a;
                at = j;
            }
            mt = mt.max(V3::new(d[3], d[4], d[5]).norm());
        }
        Ok(IterStats {
            planes: pl.planes.len(),
            used: used as f64 / self.pts.len().max(1) as f64,
            rms_mm: (sq / used.max(1) as f64).sqrt() * 1e3,
            imu_chi2: chi2 / nf.max(1) as f64,
            rtk_h_mm: (rtk_h2 / nr).sqrt() * 1e3,
            rtk_v_mm: (rtk_v2 / nr).sqrt() * 1e3,
            step_rot_deg: mr.to_degrees(),
            step_at: self.kt[at] - self.kt[0],
            step_mm: mt * 1e3,
            sec: t0.elapsed().as_secs_f64(),
            cg_iters,
            cost0: c0,
            cost1: c1,
            lambda,
            tries,
            ..Default::default()
        })
    }

    /// The pose-graph stage: odometry relative poses keep the shape, GNSS and loops move it.
    fn pose_graph_stage(prm: &Params) -> Stage {
        Stage {
            iters: prm.loop_pre_iters,
            planes: relpose::no_planes(),
            c_cauchy: 0.15,
            couple: false,
            rtk_c: Some(1.0),
            odometry: true,
        }
    }

    /// Pulls the odometry onto the GNSS fixes with a pose-graph stage, before the loops are registered:
    /// a drifted part would otherwise be measured and held by the loops away from correct fixes.
    pub fn gnss_pose_graph(
        &mut self,
        prm: &Params,
        job: &s20_io::job::Job,
    ) -> Result<Vec<IterStats>> {
        if self.rtk.is_empty() || prm.loop_pre_iters == 0 {
            return Ok(Vec::new());
        }
        let prm = &Params {
            stages: vec![Self::pose_graph_stage(prm)],
            ..prm.clone()
        };
        self.run_stages(prm, "gnss ", "opt gnss", job)
    }

    pub fn run(&mut self, prm: &Params, job: &s20_io::job::Job) -> Result<Vec<IterStats>> {
        // with loops, a pose-graph stage first spreads the loop corrections along the trajectory
        let mut stages = prm.stages.clone();
        if !self.loops.is_empty() && prm.loop_pre_iters > 0 {
            stages.insert(0, Self::pose_graph_stage(prm));
        }
        let prm = &Params {
            stages,
            ..prm.clone()
        };
        self.run_stages(prm, "", "opt", job)
    }

    /// Runs the stages of `prm`, logging each iteration with `tag` in front and reporting the
    /// iterations done as progress under `report`.
    fn run_stages(
        &mut self,
        prm: &Params,
        tag: &str,
        report: &str,
        job: &s20_io::job::Job,
    ) -> Result<Vec<IterStats>> {
        let log = |s: &str| job.log(s);
        let total_iters: usize = prm.stages.iter().map(|s| s.iters).sum();
        let mut hist: Vec<IterStats> = Vec::new();
        let last = prm.stages.len().saturating_sub(1);
        for (si, st) in prm.stages.iter().enumerate() {
            // the last stage continues until the steps are small (at most `extra_iters` more)
            let max_it = if si == last {
                st.iters + prm.extra_iters
            } else {
                st.iters
            };
            let before = st.odometry.then(|| self.states.clone());
            for it in 0..max_it {
                if it >= st.iters
                    && let Some(p) = hist.last()
                    && p.step_mm < prm.conv_mm
                    && p.step_rot_deg < prm.conv_deg
                {
                    break;
                }
                job.check()?;
                job.report(report, hist.len() as u64, total_iters as u64);
                let mut s = self.step(prm, st)?;
                s.stage = si;
                s.it = it;
                let rtk = if self.rtk.is_empty() {
                    String::new()
                } else {
                    format!(" rtk h {:.1} v {:.1}mm", s.rtk_h_mm, s.rtk_v_mm)
                };
                log(&format!(
                    "{tag}{}{rtk}",
                    format_args!(
                        "stage {si} it {it}: planes {} used {:.3} rms {:.2}mm imu chi2 {:.2} step {:.4}deg (at {:.1}s) {:.2}mm  cost {:.6e} -> {:.6e} (lambda {:.0e}, tries {})  cg {}  {:.1}s",
                        s.planes,
                        s.used,
                        s.rms_mm,
                        s.imu_chi2,
                        s.step_rot_deg,
                        s.step_at,
                        s.step_mm,
                        s.cost0,
                        s.cost1,
                        s.lambda,
                        s.tries,
                        s.cg_iters,
                        s.sec
                    )
                ));
                hist.push(s);
            }
            // after the pose graph: keep each knot's velocity in its body frame (the IMU factors
            // of the next stages then start from a consistent motion)
            if let Some(b) = before {
                for (s, o) in self.states.iter_mut().zip(&b) {
                    s.v = s.r * o.r.transpose() * o.v;
                }
            }
            if prm.loop_reg.remeasure && !st.odometry && si != last && !self.loops.is_empty() {
                self.remeasure_loops(&prm.loop_reg, prm.rmax, &log);
            }
        }
        Ok(hist)
    }

    /// Dense trajectory sampled every `dt` (IMU-integrated between knots), rows `t x y z qx qy qz qw`.
    pub fn dense(&self, prm: &Params, dt: f64) -> Vec<[f64; 8]> {
        let pre = self.preintegrate(&prm.noise);
        let mut out = Vec::new();
        for k in 0..self.kt.len() {
            let s = &self.states[k];
            let push = |out: &mut Vec<[f64; 8]>, t: f64, r: &M3, p: &V3| {
                let q = so3::to_quat(r);
                out.push([t, p.x, p.y, p.z, q[0], q[1], q[2], q[3]]);
            };
            push(&mut out, self.kt[k], &s.r, &s.p);
            if let Some(Some(pi)) = pre.get(k) {
                let mut t = (self.kt[k] / dt).floor() * dt + dt;
                while t < self.kt[k + 1] - 1e-6 {
                    let (r, p) = s.pose_at(pi, t - self.kt[k]);
                    push(&mut out, t, &r, &p);
                    t += dt;
                }
            }
        }
        out
    }

    /// Knot states rows `t x y z qx qy qz qw vx vy vz bgx bgy bgz bax bay baz`.
    pub fn knots(&self) -> Vec<[f64; 17]> {
        self.kt
            .iter()
            .zip(&self.states)
            .map(|(&t, s)| {
                let q = so3::to_quat(&s.r);
                [
                    t, s.p.x, s.p.y, s.p.z, q[0], q[1], q[2], q[3], s.v.x, s.v.y, s.v.z, s.bg.x,
                    s.bg.y, s.bg.z, s.ba.x, s.ba.y, s.ba.z,
                ]
            })
            .collect()
    }
}
