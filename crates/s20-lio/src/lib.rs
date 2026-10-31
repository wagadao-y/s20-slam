//! Tightly coupled LiDAR-inertial odometry (iterated error-state Kalman filter), port of `lio_tc.py`.
//!
//! State (body = IMU): R_wb, p_w, v_w, b_g, b_a. Error state (15): dtheta (right, body), dp, dv (world), dbg, dba.
//! - IMU propagation at every sample, covariance with gyro/acc noise and bias random walks.
//! - Deskew: per-point poses from the IMU-propagated states inside the scan.
//! - Update: VGICP of the deskewed scan against the local Gaussian voxel map. At each iterate the registration
//!   is linearised (H, b in the right perturbation [w, v] of world_T_imu) and fused with the IMU prior in
//!   information form. Directions without geometric constraint follow the IMU prediction.
//! - Initialisation: gravity direction and gyro bias from the initial static period, or from a previous run.
//! - Heading hold: in intervals the user marks (round hills, featureless slopes, where the registration drags
//!   the heading), the update leaves the rotation about the vertical and the gyro bias as the IMU predicts them:
//!   over minutes the gyro with the bias estimated before holds the heading better than such a registration.
//! - Gravity: the registration only aligns a scan to the map, so a slowly tilting map tilts the pose with it
//!   (the gravity in the IMU propagation is far too weak against it). Each update also observes the vertical
//!   in the body frame from the accelerometer over the last `grav_win` seconds: the specific force rotated to
//!   the scan-end frame with the gyro, minus the velocity change, integrates to gravity. `tilt_lidar` scales
//!   down the LiDAR information on the tilt (rotation about horizontal world axes). The direction alone cannot
//!   tell a horizontal accelerometer bias from a tilt, so the bias starts from `ba0` and barely walks.

use anyhow::{Result, bail};
use nalgebra::{SMatrix, SVector};
use rayon::prelude::*;
use s20_geom::cloud::preprocess;
use s20_geom::so3::{self, M3, V3, skew};
use s20_geom::vgicp::{Opts, linearize_opts};
use s20_geom::voxelmap::GaussianVoxelMap;
use s20_io::dataset::LidarData;

pub const GRAV: f64 = 9.80665;
pub const LIDAR_IMU_T: [f64; 3] = [-0.011, -0.02329, 0.04412];
/// Accelerometer bias of the S20 used so far (m/s^2, body), as the PPK unified optimisations of the
/// park recordings (three of them) estimated it.
pub const S20_BA0: [f64; 3] = [-0.13, 0.0, -0.10];

type M15 = SMatrix<f64, 15, 15>;
type V15 = SVector<f64, 15>;
type M6 = SMatrix<f64, 6, 6>;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Params {
    /// scan range of a partial run (command line only; not part of the saved settings)
    #[serde(skip)]
    pub scans: Option<(usize, usize)>,
    pub imu_shift: f64,
    pub static_sec: f64,
    pub rmin: f64,
    pub rmax: f64,
    pub down: f64,
    pub map_vox: f64,
    pub max_corr: f64,
    pub lru: usize,
    pub iters: usize,
    pub min_inlier: f64,
    pub lidar_scale: f64,
    pub ng: f64,
    pub na: f64,
    pub nbg: f64,
    pub nba: f64,
    pub num_neighbors: usize,
    pub lidar_imu_t: [f64; 3],
    /// registration: also search the 6 face-neighbour voxels
    pub search7: bool,
    /// registration: Huber threshold on the Mahalanobis residual (0: off)
    pub huber: f64,
    /// correct the tilt with the gravity observation (with `grav_sd`, `grav_win`, `tilt_lidar`, `grav_nba`)
    pub grav: bool,
    /// gravity observation: standard deviation of the vertical direction per scan (deg)
    pub grav_sd: f64,
    /// gravity observation: accelerometer window (s)
    pub grav_win: f64,
    /// with the gravity observation: factor on the LiDAR information on the tilt (1: unchanged)
    pub tilt_lidar: f64,
    /// with the gravity observation: accelerometer bias random walk in place of `nba` (the bias must not
    /// take over the tilt the observation corrects)
    pub grav_nba: f64,
    /// initial accelerometer bias (body, m/s^2); also removed from the static start's gravity direction.
    /// Default: [`S20_BA0`]
    pub ba0: [f64; 3],
    /// consistency gate (0: off): an update whose correction of the IMU prediction exceeds this NIS
    /// (15 dof) is searched again from the prediction with correspondences up to `gate_corr` (m), and that
    /// search is taken when it stays within the gate. Against wrong correspondences that drag the pose
    /// away while the IMU shows only a turn (2026-10-07 10-31-51, 470 s: 3 m and 0.9 m down)
    pub gate_nis: f64,
    pub gate_corr: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            scans: None,
            imu_shift: 0.005,
            static_sec: 2.0,
            rmin: 1.0,
            rmax: 100.0,
            down: 0.2,
            map_vox: 0.5,
            max_corr: 0.5,
            lru: 100000,
            iters: 6,
            min_inlier: 0.3,
            lidar_scale: 30.0,
            // Keep short-term rotations closer to the gyro prediction. The old
            // 0.005 setting let scan matching make unnecessarily large corrections.
            // Validated on the indoor and both park recordings; see the LIO experiments.
            ng: 0.0002,
            na: 0.05,
            nbg: 1e-4,
            nba: 1e-3,
            num_neighbors: 10,
            lidar_imu_t: LIDAR_IMU_T,
            search7: false,
            huber: 0.0,
            // indoors and outdoors: validated on the 97 min park recording 2026-10-15 (tilt 3.2 -> 0.17
            // deg mean), on the park recordings with PPK references and indoors (tilt 0.48 ->
            // 0.19 deg, the optimised trajectory within 6 mm); see docs/processing.md
            grav: true,
            grav_sd: 0.2,
            grav_win: 3.0,
            tilt_lidar: 0.01,
            grav_nba: 1e-5,
            ba0: S20_BA0,
            // validated on the 10-07 recordings, 10-03 and the 97 min 10-15 park and indoors;
            // see docs/processing.md
            gate_nis: 200.0,
            gate_corr: 0.25,
        }
    }
}

impl Params {
    /// Outdoor preset: the defaults.
    pub fn outdoor() -> Self {
        Self::default()
    }

    /// Indoor preset (fine map, near points; proven on stairs).
    pub fn indoor() -> Self {
        Params {
            down: 0.1,
            map_vox: 0.2,
            max_corr: 0.5,
            rmin: 0.5,
            lidar_scale: 100.0,
            ..Default::default()
        }
    }
}

/// Initial state taken from a previous run (pose at the first scan end, velocity and biases).
#[derive(Clone, Debug)]
pub struct InitState {
    pub r: M3,
    pub p: V3,
    pub v: V3,
    pub bg: V3,
    pub ba: V3,
}

/// One output row per processed scan.
#[derive(Clone, Copy, Debug)]
pub struct ScanPose {
    pub scan: usize,
    pub t: f64,
    pub r: M3,
    pub p: V3,
    pub v: V3,
    pub bg: V3,
    pub ba: V3,
    pub inlier: f64,
    pub ok: bool,
    pub diag: Diag,
}

/// Per-scan diagnostics of the update (written to `_scans.txt`, not used by the filter).
#[derive(Clone, Copy, Debug, Default)]
pub struct Diag {
    /// registration cost per inlier at the last iterate (0.5 r^T M r)
    pub err: f64,
    /// correction of the IMU prediction in its own covariance: e^T P^-1 e (15 dof)
    pub nis: f64,
    /// correction of the IMU prediction: position (m), rotation (deg)
    pub dpos: f64,
    pub drot: f64,
    /// world vertical component of the rotation correction (deg, signed)
    pub dyaw: f64,
    /// smallest eigenvalue of the LiDAR information, rotation (1/rad^2) and position (1/m^2) blocks
    pub eig_rot: f64,
    pub eig_pos: f64,
    pub iters: usize,
    /// the pose motion the LiDAR constrains least
    pub weak: WeakMotion,
    /// LiDAR information on the rotation about the world vertical with the position eliminated (1/rad^2),
    /// and the number of registered points
    pub yaw_info: f64,
    pub npts: usize,
    /// in a heading hold interval
    pub held: bool,
    /// angle between the vertical of the pose and the one observed from the accelerometer (deg; 0: no observation)
    pub grav: f64,
    /// consistency gate: 0 passed (or off), 1 taken from the second search near the prediction,
    /// 2 the second search did not fit either and the first was kept
    pub gate: u8,
}

pub struct Output {
    pub poses: Vec<ScanPose>,
    pub n_skip: usize,
    /// scans in heading hold intervals
    pub n_held: usize,
    /// consistency gate: scans taken from the second search
    pub n_retried: usize,
}

/// One iterated update: the state (R, p, v, bg, ba), whether the registration held, its information.
struct Solution {
    st: (M3, V3, V3, V3, V3),
    ok: bool,
    hf: M15,
    inl: f64,
    diag: Diag,
}

struct Imu {
    t: Vec<f64>,
    gyr: Vec<V3>,
    acc: Vec<V3>,
}

/// Recorded states inside one propagation (for deskewing).
#[derive(Default)]
struct Rec {
    t: Vec<f64>,
    r: Vec<M3>,
    p: Vec<V3>,
    w: Vec<V3>,
}

#[allow(clippy::too_many_arguments)]
fn propagate(
    prm: &Params,
    t0: f64,
    st: (&M3, &V3, &V3),
    bg: &V3,
    ba: &V3,
    pcov: &M15,
    ts: &[f64],
    gyr: &[V3],
    acc: &[V3],
    rec: &mut Rec,
) -> (M3, V3, V3, M15) {
    let g = V3::new(0.0, 0.0, -GRAV);
    let (mut r, mut p, mut v) = (*st.0, *st.1, *st.2);
    let mut pc = *pcov;
    let mut tc = t0;
    rec.t.clear();
    rec.r.clear();
    rec.p.clear();
    rec.w.clear();
    rec.t.push(tc);
    rec.r.push(r);
    rec.p.push(p);
    rec.w.push(gyr[0] - bg);
    let i3 = M3::identity();
    let nba = if prm.grav { prm.grav_nba } else { prm.nba };
    for k in 0..ts.len() {
        let dt = ts[k] - tc;
        if dt <= 0.0 {
            continue;
        }
        let w = gyr[k] - bg;
        let a = acc[k] - ba;
        let ra = r * a;
        let mut f = M15::identity();
        f.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&so3::exp(&(-w * dt)));
        f.fixed_view_mut::<3, 3>(0, 9).copy_from(&(-i3 * dt));
        f.fixed_view_mut::<3, 3>(3, 6).copy_from(&(i3 * dt));
        f.fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-r * skew(&a) * dt));
        f.fixed_view_mut::<3, 3>(6, 12).copy_from(&(-r * dt));
        pc = f * pc * f.transpose();
        for i in 0..3 {
            pc[(i, i)] += prm.ng * prm.ng * dt;
            pc[(6 + i, 6 + i)] += prm.na * prm.na * dt;
            pc[(9 + i, 9 + i)] += prm.nbg * prm.nbg * dt;
            pc[(12 + i, 12 + i)] += nba * nba * dt;
        }
        let acc_w = ra + g;
        p += v * dt + 0.5 * acc_w * dt * dt;
        v += acc_w * dt;
        r *= so3::exp(&(w * dt));
        tc = ts[k];
        rec.t.push(tc);
        rec.r.push(r);
        rec.p.push(p);
        rec.w.push(w);
    }
    (r, p, v, pc)
}

/// Points (IMU frame, time t) -> predicted scan-end frame.
fn deskew_to_end(t: &[f64], x: &[V3], rec: &Rec, re: &M3, pe: &V3) -> Vec<V3> {
    let n = rec.t.len();
    let rt = re.transpose();
    t.par_iter()
        .zip(x.par_iter())
        .map(|(&ti, xi)| {
            let k =
                (rec.t.partition_point(|&s| s < ti) as isize - 1).clamp(0, n as isize - 1) as usize;
            let dt = ti - rec.t[k];
            let rk = rec.r[k] * so3::exp(&(rec.w[k] * dt));
            let pk = if k < n - 1 {
                let a = dt / (rec.t[k + 1] - rec.t[k]).max(1e-9);
                rec.p[k] * (1.0 - a) + rec.p[k + 1] * a
            } else {
                rec.p[k]
            };
            rt * (rk * xi + pk - pe)
        })
        .collect()
}

/// Accelerometer integrals over the window (t0, t1], rotated to the body frame at t1 with the gyro:
/// `a` = int Q f dt, `b` = int Q dt, with Q = R(t1)^T R(t). With the velocities v0, v1 and bias ba,
/// R(t1)^T z g dt_w = a - b ba - R(t1)^T (v1 - v0).
struct GravWin {
    a: V3,
    b: M3,
    dt: f64,
    v0: V3,
}

/// The window from the last output state at least `win` before `t1` to `t1` (sample assignment as in
/// [`propagate`]); None at the start.
fn grav_window(imu: &Imu, out: &[ScanPose], t1: f64, win: f64, bg: &V3) -> Option<GravWin> {
    let j = out.partition_point(|s| s.t <= t1 - win).checked_sub(1)?;
    let (t0, v0) = (out[j].t, out[j].v);
    let k0 = imu.t.partition_point(|&s| s <= t0);
    let k1 = imu.t.partition_point(|&s| s <= t1);
    let kl = k1.min(imu.t.len() - 1);
    let (mut q, mut a, mut b) = (M3::identity(), V3::zeros(), M3::zeros());
    let mut tc = t0;
    for k in (k0..k1).chain(std::iter::once(kl)) {
        let tk = if k == kl && k >= k1 {
            t1
        } else {
            imu.t[k].min(t1)
        };
        let dt = tk - tc;
        if dt <= 0.0 {
            continue;
        }
        let w = imu.gyr[k] - bg;
        let qm = q * so3::exp(&(w * (0.5 * dt)));
        a += qm * imu.acc[k] * dt;
        b += qm * dt;
        q *= so3::exp(&(w * dt));
        tc = tk;
    }
    let dt = tc - t0;
    if dt < 0.5 * win {
        return None;
    }
    // relative to the start -> relative to the end
    let qt = q.transpose();
    Some(GravWin {
        a: qt * a,
        b: qt * b,
        dt,
        v0,
    })
}

/// Gravity observation at the state (r, v, ba): residual r = R^T z - m/|m| with m = (a - b ba - R^T (v - v0)) / dt,
/// as normal-equation terms on the 15-dof error state (weight 1/sd^2), and the angle (deg).
fn grav_term(g: &GravWin, r: &M3, v: &V3, ba: &V3, sd: f64) -> (M15, V15, f64) {
    let rt = r.transpose();
    let m = (g.a - g.b * ba - rt * (v - g.v0)) / g.dt;
    let n = m.norm();
    let mh = m / n;
    let u = rt * V3::z();
    let res = u - mh;
    let pr = (M3::identity() - mh * mh.transpose()) / (n * g.dt);
    let mut j = SMatrix::<f64, 3, 15>::zeros();
    j.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew(&u));
    j.fixed_view_mut::<3, 3>(0, 6).copy_from(&(pr * rt));
    j.fixed_view_mut::<3, 3>(0, 12).copy_from(&(pr * g.b));
    let w = 1.0 / (sd * sd);
    (
        j.transpose() * j * w,
        j.transpose() * res * w,
        u.dot(&mh).clamp(-1.0, 1.0).acos().to_degrees(),
    )
}

/// Information on the rotation about the world vertical in the pose information `h` ([dtheta (body), dp]),
/// with the position eliminated (Schur complement): how well the scan alone fixes the heading.
fn yaw_information(h: &M6, r: &M3) -> f64 {
    let hrr = h.fixed_view::<3, 3>(0, 0);
    let hrt = h.fixed_view::<3, 3>(0, 3);
    let Some(htt_inv) = h.fixed_view::<3, 3>(3, 3).into_owned().try_inverse() else {
        return 0.0;
    };
    let s = hrr - hrt * htt_inv * hrt.transpose();
    let u = r.transpose() * V3::z();
    u.dot(&(s * u))
}

/// The pose motion the LiDAR information `h` ([dtheta (body), dp (world)]) constrains least, with rotations
/// weighted as their motion at 10 m. A rotation about a vertical axis away from the sensor (walking around
/// a round hill) shows up as a vertical rotation with an axis position.
#[derive(Clone, Copy, Debug, Default)]
pub struct WeakMotion {
    /// smallest eigenvalue (1/m^2 at 10 m)
    pub eig: f64,
    /// share of rotation in it (0: translation .. 1: rotation)
    pub rot: f64,
    /// |vertical component| of its rotation axis (0..1)
    pub vert: f64,
    /// where the rotation axis crosses the horizontal plane (world x, y; meaningful when `rot`, `vert` are high)
    pub axis: [f64; 2],
}

fn weakest_motion(h: &M6, r: &M3, p: &V3) -> WeakMotion {
    const L: f64 = 10.0;
    let mut dm = M6::identity();
    for k in 0..3 {
        dm[(k, k)] = 1.0 / L;
    }
    let hs = dm * h * dm;
    let eig = (0.5 * (hs + hs.transpose())).symmetric_eigen();
    let k = eig.eigenvalues.imin();
    let u = eig.eigenvectors.column(k);
    let rot = u.fixed_rows::<3>(0).norm();
    let w = r * (u.fixed_rows::<3>(0) / L);
    let dp: V3 = u.fixed_rows::<3>(3).into_owned();
    let n2 = w.norm_squared();
    let c = if n2 > 0.0 { p - dp.cross(&w) / n2 } else { *p };
    WeakMotion {
        eig: eig.eigenvalues[k],
        rot,
        vert: if n2 > 0.0 { w.z.abs() / n2.sqrt() } else { 0.0 },
        axis: [c.x, c.y],
    }
}

fn diag15(v: [f64; 5]) -> M15 {
    let mut m = M15::zeros();
    for (b, &x) in v.iter().enumerate() {
        for i in 0..3 {
            m[(3 * b + i, 3 * b + i)] = x;
        }
    }
    m
}

fn std3(v: &[V3]) -> V3 {
    let n = v.len() as f64;
    let m: V3 = v.iter().sum::<V3>() / n;
    let s: V3 = v
        .iter()
        .map(|x| (x - m).component_mul(&(x - m)))
        .sum::<V3>()
        / n;
    s.map(f64::sqrt)
}

/// `hold`: heading hold intervals (sensor clock, s); see the module comment.
pub fn run(
    data: &LidarData,
    imu_rows: &[[f64; 7]],
    prm: &Params,
    init: Option<InitState>,
    hold: &[(f64, f64)],
    job: &s20_io::job::Job,
) -> Result<Output> {
    let log = |s: &str| job.log(s);
    let imu = Imu {
        t: imu_rows.iter().map(|r| r[0] - prm.imu_shift).collect(),
        gyr: imu_rows.iter().map(|r| V3::new(r[1], r[2], r[3])).collect(),
        acc: imu_rows
            .iter()
            .map(|r| V3::new(r[4], r[5], r[6]) * GRAV)
            .collect(),
    };
    if imu.t.is_empty() || data.is_empty() {
        bail!("no imu or lidar data");
    }
    let (i0, i1) = prm.scans.unwrap_or((0, data.len()));
    let ids: Vec<usize> = (i0..i1.min(data.len())).collect();
    let lever = V3::from(prm.lidar_imu_t);

    // --- initialisation from the static start
    let first = data.scan(ids[0]);
    let t_first = first.t.iter().copied().fold(f64::INFINITY, f64::min);
    let m: Vec<usize> = (0..imu.t.len())
        .filter(|&k| imu.t[k] >= t_first && imu.t[k] < t_first + prm.static_sec)
        .collect();
    if m.is_empty() {
        bail!("no imu samples in the static period");
    }
    let gm: V3 = m.iter().map(|&k| imu.gyr[k]).sum::<V3>() / m.len() as f64;
    let am: V3 = m.iter().map(|&k| imu.acc[k]).sum::<V3>() / m.len() as f64;
    let gstd = std3(&m.iter().map(|&k| imu.gyr[k]).collect::<Vec<_>>());
    let mut stat = gstd.norm() < 1f64.to_radians();
    let mut ba = V3::from(prm.ba0);
    let mut r = so3::align_vectors(&(am - ba), &V3::z());
    let mut p = V3::zeros();
    let mut v = V3::zeros();
    let mut bg = if stat { gm } else { V3::zeros() };
    let d2r = |x: f64| x.to_radians();
    let mut pcov = diag15([
        d2r(1.0).powi(2),
        1e-6,
        if stat { 0.01 } else { 1.0 },
        (if stat { d2r(0.05) } else { d2r(1.0) }).powi(2),
        0.05f64.powi(2),
    ]);
    if let Some(s) = init {
        r = s.r;
        p = s.p;
        v = s.v;
        bg = s.bg;
        ba = s.ba;
        pcov = diag15([
            d2r(0.1).powi(2),
            1e-6,
            0.01,
            d2r(0.05).powi(2),
            0.02f64.powi(2),
        ]);
        stat = false;
    }
    log(&format!(
        "init: static={stat} gyro bias [{:.3} {:.3} {:.3}] deg/s  |acc| {:.3}",
        bg.x.to_degrees(),
        bg.y.to_degrees(),
        bg.z.to_degrees(),
        am.norm()
    ));

    let mut vmap = GaussianVoxelMap::new(prm.map_vox);
    vmap.set_lru(prm.lru, 10);
    let mut rec = Rec::default();
    let mut t_state: Option<f64> = None;
    let mut out = Vec::with_capacity(ids.len());
    let mut n_skip = 0;
    let mut n_held = 0;
    let mut n_retried = 0;
    let t_start = std::time::Instant::now();
    let nk = imu.t.len();
    for (n_, &i) in ids.iter().enumerate() {
        if n_ % 50 == 0 {
            job.check()?;
            job.report("lio", n_ as u64, ids.len() as u64);
        }
        let sc = data.scan(i);
        let mut t = Vec::with_capacity(sc.t.len());
        let mut x = Vec::with_capacity(sc.t.len());
        for k in 0..sc.t.len() {
            let q = V3::from(sc.xyz[k]);
            let rr = q.norm();
            if rr > prm.rmin && rr < prm.rmax && (sc.tag[k] & 0x3F) == 0 {
                t.push(sc.t[k]);
                x.push(q + lever);
            }
        }
        if t.len() < 200 {
            continue;
        }
        let te = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let Some(ts0) = t_state else {
            // start state at the end of the first scan (assumed static over it)
            t_state = Some(te);
            let pc = preprocess(&x, prm.down, prm.num_neighbors);
            vmap.insert(&pc, &r, &p);
            out.push(ScanPose {
                scan: i,
                t: te,
                r,
                p,
                v,
                bg,
                ba,
                inlier: 1.0,
                ok: true,
                diag: Diag::default(),
            });
            continue;
        };
        // --- IMU propagation t_state -> te
        let k0 = imu.t.partition_point(|&s| s <= ts0);
        let k1 = imu.t.partition_point(|&s| s <= te);
        let kl = k1.min(nk - 1);
        let mut ts: Vec<f64> = imu.t[k0..k1].to_vec();
        ts.push(te);
        let mut gy: Vec<V3> = imu.gyr[k0..k1].to_vec();
        gy.push(imu.gyr[kl]);
        let mut ac: Vec<V3> = imu.acc[k0..k1].to_vec();
        ac.push(imu.acc[kl]);
        let (rp, pp, vp, pcp) = propagate(
            prm,
            ts0,
            (&r, &p, &v),
            &bg,
            &ba,
            &pcov,
            &ts,
            &gy,
            &ac,
            &mut rec,
        );
        // --- deskew into the predicted end frame
        let xe = deskew_to_end(&t, &x, &rec, &rp, &pp);
        let pc = preprocess(&xe, prm.down, prm.num_neighbors);
        // --- iterated EKF update
        let Some(pinv) = pcp.try_inverse() else {
            bail!("singular covariance at scan {i}")
        };
        let state_err = |ri: &M3, pi: &V3, vi: &V3, bgi: &V3, bai: &V3| {
            let mut e = V15::zeros();
            e.fixed_rows_mut::<3>(0)
                .copy_from(&so3::log(&(rp.transpose() * ri)));
            e.fixed_rows_mut::<3>(3).copy_from(&(pi - pp));
            e.fixed_rows_mut::<3>(6).copy_from(&(vi - vp));
            e.fixed_rows_mut::<3>(9).copy_from(&(bgi - bg));
            e.fixed_rows_mut::<3>(12).copy_from(&(bai - ba));
            e
        };
        let held = hold.iter().any(|&(a, b)| te >= a && te < b);
        n_held += held as usize;
        let grav = if prm.grav && prm.grav_sd > 0.0 {
            grav_window(&imu, &out, te, prm.grav_win, &bg)
        } else {
            None
        };
        let grav_sd = prm.grav_sd.to_radians();
        // the iterated update from the prediction, with correspondences up to `max_corr`
        let solve = |max_corr: f64| -> Result<Solution> {
            let (mut ri, mut pi, mut vi, mut bgi, mut bai) = (rp, pp, vp, bg, ba);
            let mut ok = true;
            let mut hf = M15::zeros();
            let mut inl = 0.0;
            let mut diag = Diag {
                held,
                ..Default::default()
            };
            let mut e = V15::zeros();
            for _ in 0..prm.iters {
                let lin = linearize_opts(
                    &vmap,
                    &pc,
                    &ri,
                    &pi,
                    max_corr,
                    &Opts {
                        search7: prm.search7,
                        huber: prm.huber,
                    },
                );
                inl = lin.inliers as f64 / pc.len().max(1) as f64;
                if inl < prm.min_inlier {
                    ok = false;
                    break;
                }
                let h6 = lin.h * prm.lidar_scale;
                diag.iters += 1;
                diag.err = lin.e / lin.inliers.max(1) as f64;
                diag.eig_rot = h6.fixed_view::<3, 3>(0, 0).symmetric_eigenvalues().min();
                diag.eig_pos = h6.fixed_view::<3, 3>(3, 3).symmetric_eigenvalues().min();
                let b6 = lin.b * prm.lidar_scale;
                // [w; v] = J [dtheta; dp] with J = diag(I, R^T)
                let mut j = M6::identity();
                j.fixed_view_mut::<3, 3>(3, 3).copy_from(&ri.transpose());
                let mut hs = j.transpose() * h6 * j;
                let mut bs = j.transpose() * b6;
                diag.yaw_info = yaw_information(&hs, &ri);
                diag.weak = weakest_motion(&hs, &ri, &pi);
                diag.npts = pc.len();
                if prm.grav && prm.tilt_lidar < 1.0 {
                    // scale the information on the rotation about the horizontal world axes (S H S), keeping the
                    // pose the registration alone prefers (x* = -H^-1 b): b' = -S H S x*
                    let u = ri.transpose() * V3::z();
                    let uu = u * u.transpose();
                    let mut s = M6::identity();
                    s.fixed_view_mut::<3, 3>(0, 0)
                        .copy_from(&((M3::identity() - uu) * prm.tilt_lidar.max(0.0).sqrt() + uu));
                    let reg = M6::identity() * (1e-9 * hs.trace());
                    if let Some(xs) = (hs + reg).cholesky().map(|c| c.solve(&(-bs))) {
                        hs = s * hs * s;
                        bs = -(hs * xs);
                    }
                }
                hf = M15::zeros();
                hf.fixed_view_mut::<6, 6>(0, 0).copy_from(&hs);
                let mut bf = V15::zeros();
                bf.fixed_rows_mut::<6>(0).copy_from(&bs);
                if let Some(g) = &grav {
                    let (hg, bgr, ang) = grav_term(g, &ri, &vi, &bai, grav_sd);
                    hf += hg;
                    bf += bgr;
                    diag.grav = ang;
                }
                e = state_err(&ri, &pi, &vi, &bgi, &bai);
                let a = pinv + hf;
                let rhs = -(pinv * e + bf);
                let Some(mut d) = a.lu().solve(&rhs) else {
                    bail!("singular update at scan {i}")
                };
                let mut dth = V3::new(d[0], d[1], d[2]);
                if held {
                    // keep the predicted heading (rotation about the world vertical) and gyro bias
                    let u = ri.transpose() * V3::z();
                    dth -= u * u.dot(&dth);
                    for k in 9..12 {
                        d[k] = 0.0;
                    }
                }
                let dp = V3::new(d[3], d[4], d[5]);
                ri *= so3::exp(&dth);
                pi += dp;
                vi += V3::new(d[6], d[7], d[8]);
                bgi += V3::new(d[9], d[10], d[11]);
                bai += V3::new(d[12], d[13], d[14]);
                if dth.norm() < 0.01f64.to_radians() && dp.norm() < 1e-3 {
                    break;
                }
            }
            if ok {
                e = state_err(&ri, &pi, &vi, &bgi, &bai);
            }
            diag.nis = e.dot(&(pinv * e));
            diag.dpos = (pi - pp).norm();
            diag.drot = e.fixed_rows::<3>(0).norm().to_degrees();
            diag.dyaw = (rp * e.fixed_rows::<3>(0)).z.to_degrees();
            Ok(Solution {
                st: (ri, pi, vi, bgi, bai),
                ok,
                hf,
                inl,
                diag,
            })
        };
        let mut s = solve(prm.max_corr)?;
        if prm.gate_nis > 0.0 && s.ok && s.diag.nis > prm.gate_nis {
            // the registration contradicts the IMU prediction: search again near the prediction. When
            // nothing near it fits either, the prediction is the one off (fast motion, stairs): keep the first
            let s2 = solve(prm.gate_corr)?;
            if s2.ok && s2.diag.nis <= prm.gate_nis {
                s = s2;
                s.diag.gate = 1;
            } else {
                s.diag.gate = 2;
            }
        }
        let (ok, inl, diag) = (s.ok, s.inl, s.diag);
        n_retried += (diag.gate == 1) as usize;
        if ok {
            (r, p, v, bg, ba) = s.st;
            let Some(pn) = (pinv + s.hf).try_inverse() else {
                bail!("singular posterior at scan {i}")
            };
            pcov = 0.5 * (pn + pn.transpose());
            vmap.insert(&pc, &r, &p);
        } else {
            n_skip += 1;
            (r, p, v, pcov) = (rp, pp, vp, pcp);
        }
        r = so3::normalize(&r);
        t_state = Some(te);
        out.push(ScanPose {
            scan: i,
            t: te,
            r,
            p,
            v,
            bg,
            ba,
            inlier: inl,
            ok,
            diag,
        });
        if n_ % 1000 == 0 {
            log(&format!(
                "{n_}/{} pos [{:.2} {:.2} {:.2}] |v| {:.2} bg [{:.3} {:.3} {:.3}] ba [{:.3} {:.3} {:.3}] sd(yaw) {:.3} deg  skipped {n_skip}  map {} voxels  {:.0}s",
                ids.len(),
                p.x,
                p.y,
                p.z,
                v.norm(),
                bg.x.to_degrees(),
                bg.y.to_degrees(),
                bg.z.to_degrees(),
                ba.x,
                ba.y,
                ba.z,
                pcov[(2, 2)].sqrt().to_degrees(),
                vmap.len(),
                t_start.elapsed().as_secs_f64()
            ));
        }
    }
    Ok(Output {
        poses: out,
        n_skip,
        n_held,
        n_retried,
    })
}

/// Initial state from a previous run: pose rows `t x y z qx qy qz qw` and state rows `t v bg ba`,
/// taken at the end of scan `scan`.
pub fn init_from(
    data: &LidarData,
    scan: usize,
    poses: &[[f64; 8]],
    states: &[Vec<f64>],
) -> Result<InitState> {
    let sc = data.scan(scan);
    let te = sc.t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let near = |t: f64| (t - te).abs();
    let k = (0..poses.len())
        .min_by(|&a, &b| near(poses[a][0]).total_cmp(&near(poses[b][0])))
        .ok_or_else(|| anyhow::anyhow!("empty trajectory"))?;
    let ks = (0..states.len())
        .min_by(|&a, &b| near(states[a][0]).total_cmp(&near(states[b][0])))
        .ok_or_else(|| anyhow::anyhow!("empty state file"))?;
    let s = &states[ks];
    if s.len() < 10 {
        bail!("state rows need 10 columns");
    }
    let q = &poses[k];
    Ok(InitState {
        r: so3::from_quat([q[4], q[5], q[6], q[7]]),
        p: V3::new(q[1], q[2], q[3]),
        v: V3::new(s[1], s[2], s[3]),
        bg: V3::new(s[4], s[5], s[6]),
        ba: V3::new(s[7], s[8], s[9]),
    })
}

/// Write `<out>.txt`, `<out>_scans.txt`, `<out>_state.txt` (same formats as the Python version).
pub fn save(out: &std::path::Path, res: &Output) -> Result<()> {
    use std::fmt::Write;
    let stem = out.with_extension("");
    let stem = stem.to_string_lossy();
    let mut a = String::from("# t x y z qx qy qz qw (world_T_imu, sensor clock)\n");
    let mut b = String::from(
        "# scan t_end x y z qx qy qz qw inlier_ratio err ok nis dpos drot_deg eig_rot eig_pos iters dyaw_deg weak_eig weak_rot weak_vert axis_x axis_y yaw_info npts held grav_deg gate (tightly coupled LIO)\n",
    );
    let mut c = String::from("# t vx vy vz bgx bgy bgz bax bay baz\n");
    for s in &res.poses {
        let q = so3::to_quat(&s.r);
        writeln!(
            a,
            "{:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
            s.t, s.p.x, s.p.y, s.p.z, q[0], q[1], q[2], q[3]
        )?;
        writeln!(
            b,
            "{} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.6} {} {:.4} {:.5} {:.4} {:.4e} {:.4e} {} {:.5} {:.4e} {:.3} {:.3} {:.3} {:.3} {:.4e} {} {} {:.4} {}",
            s.scan,
            s.t,
            s.p.x,
            s.p.y,
            s.p.z,
            q[0],
            q[1],
            q[2],
            q[3],
            s.inlier,
            s.diag.err,
            s.ok as u8,
            s.diag.nis,
            s.diag.dpos,
            s.diag.drot,
            s.diag.eig_rot,
            s.diag.eig_pos,
            s.diag.iters,
            s.diag.dyaw,
            s.diag.weak.eig,
            s.diag.weak.rot,
            s.diag.weak.vert,
            s.diag.weak.axis[0],
            s.diag.weak.axis[1],
            s.diag.yaw_info,
            s.diag.npts,
            s.diag.held as u8,
            s.diag.grav,
            s.diag.gate
        )?;
        writeln!(
            c,
            "{:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
            s.t, s.v.x, s.v.y, s.v.z, s.bg.x, s.bg.y, s.bg.z, s.ba.x, s.ba.y, s.ba.z
        )?;
    }
    std::fs::write(out, a)?;
    std::fs::write(format!("{stem}_scans.txt"), b)?;
    std::fs::write(format!("{stem}_state.txt"), c)?;
    Ok(())
}
