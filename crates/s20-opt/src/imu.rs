//! IMU preintegration between knots (on-manifold, Forster et al.) with first-order bias correction,
//! and the within-interval motion used to place LiDAR points.
//!
//! Sample assignment matches the LIO: the segment (t_{k-1}, t_k] uses sample k; the last partial segment up to
//! the interval end uses the next sample.
//!
//! Knot state blocks are ordered [theta (right, body), p (world), v (world), bg, ba] (15).

use nalgebra::{SMatrix, SVector};
use s20_geom::so3::{self, M3, V3, skew};

pub type M9 = SMatrix<f64, 9, 9>;
pub type M15 = SMatrix<f64, 15, 15>;
pub type V15 = SVector<f64, 15>;

pub const GRAV: f64 = 9.80665;

pub fn gravity() -> V3 {
    V3::new(0.0, 0.0, -GRAV)
}

pub struct ImuData {
    pub t: Vec<f64>,
    pub gyr: Vec<V3>,
    /// m/s^2
    pub acc: Vec<V3>,
}

impl ImuData {
    /// Rows [t, w, a(g)] from the data folder, time shifted by -shift.
    pub fn from_rows(rows: &[[f64; 7]], shift: f64) -> Self {
        ImuData {
            t: rows.iter().map(|r| r[0] - shift).collect(),
            gyr: rows.iter().map(|r| V3::new(r[1], r[2], r[3])).collect(),
            acc: rows
                .iter()
                .map(|r| V3::new(r[4], r[5], r[6]) * GRAV)
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Noise {
    pub ng: f64,
    pub na: f64,
    pub nbg: f64,
    pub nba: f64,
}

impl Default for Noise {
    fn default() -> Self {
        Noise {
            ng: 2e-4,
            na: 0.005,
            nbg: 1e-4,
            nba: 1e-3,
        }
    }
}

/// One integration segment (for evaluating the motion at any time inside the interval).
#[derive(Clone, Copy)]
struct Seg {
    /// segment end (relative to the interval start)
    tau1: f64,
    /// bias-corrected rates of the segment
    w: V3,
    a: V3,
    /// state at the segment start
    dr: M3,
    dv: V3,
    dp: V3,
}

#[derive(Clone)]
pub struct Preint {
    pub dt: f64,
    pub bg: V3,
    pub ba: V3,
    pub dr: M3,
    pub dv: V3,
    pub dp: V3,
    pub j_rg: M3,
    pub j_vg: M3,
    pub j_va: M3,
    pub j_pg: M3,
    pub j_pa: M3,
    /// covariance of [theta, v, p]
    pub cov: M9,
    segs: Vec<Seg>,
}

pub fn jr(phi: &V3) -> M3 {
    let th2 = phi.norm_squared();
    let k = skew(phi);
    if th2 < 1e-12 {
        return M3::identity() - 0.5 * k;
    }
    let th = th2.sqrt();
    M3::identity() - ((1.0 - th.cos()) / th2) * k + ((th - th.sin()) / (th2 * th)) * (k * k)
}

pub fn jr_inv(phi: &V3) -> M3 {
    let th2 = phi.norm_squared();
    let k = skew(phi);
    if th2 < 1e-12 {
        return M3::identity() + 0.5 * k;
    }
    let th = th2.sqrt();
    M3::identity() + 0.5 * k + (1.0 / th2 - (1.0 + th.cos()) / (2.0 * th * th.sin())) * (k * k)
}

impl Preint {
    pub fn new(imu: &ImuData, t0: f64, t1: f64, bg: &V3, ba: &V3, nz: &Noise) -> Self {
        let mut p = Preint {
            dt: t1 - t0,
            bg: *bg,
            ba: *ba,
            dr: M3::identity(),
            dv: V3::zeros(),
            dp: V3::zeros(),
            j_rg: M3::zeros(),
            j_vg: M3::zeros(),
            j_va: M3::zeros(),
            j_pg: M3::zeros(),
            j_pa: M3::zeros(),
            cov: M9::zeros(),
            segs: Vec::new(),
        };
        let n = imu.t.len();
        let k0 = imu.t.partition_point(|&s| s <= t0);
        let k1 = imu.t.partition_point(|&s| s <= t1);
        let mut tc = t0;
        let mut step = |p: &mut Preint, ts: f64, k: usize| {
            let dt = ts - tc;
            if dt <= 0.0 {
                return;
            }
            p.integrate(imu.gyr[k] - bg, imu.acc[k] - ba, dt, ts - t0, nz);
            tc = ts;
        };
        for k in k0..k1 {
            step(&mut p, imu.t[k], k);
        }
        step(&mut p, t1, k1.min(n - 1));
        p
    }

    fn integrate(&mut self, w: V3, a: V3, dt: f64, tau1: f64, nz: &Noise) {
        self.segs.push(Seg {
            tau1,
            w,
            a,
            dr: self.dr,
            dv: self.dv,
            dp: self.dp,
        });
        let dt2 = dt * dt;
        let ax = skew(&a);
        let er = so3::exp(&(w * dt));
        // covariance [theta, v, p]
        let mut am = M9::identity();
        am.fixed_view_mut::<3, 3>(0, 0).copy_from(&er.transpose());
        am.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(-self.dr * ax * dt));
        am.fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-0.5 * self.dr * ax * dt2));
        am.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(M3::identity() * dt));
        self.cov = am * self.cov * am.transpose();
        let (g2, a2) = (nz.ng * nz.ng, nz.na * nz.na);
        for i in 0..3 {
            self.cov[(i, i)] += g2 * dt;
            self.cov[(3 + i, 3 + i)] += a2 * dt;
            self.cov[(6 + i, 6 + i)] += 0.25 * a2 * dt2 * dt;
            self.cov[(3 + i, 6 + i)] += 0.5 * a2 * dt2;
            self.cov[(6 + i, 3 + i)] += 0.5 * a2 * dt2;
        }
        // bias Jacobians (use the values before this step)
        self.j_pa += self.j_va * dt - 0.5 * self.dr * dt2;
        self.j_pg += self.j_vg * dt - 0.5 * self.dr * ax * self.j_rg * dt2;
        self.j_va -= self.dr * dt;
        self.j_vg -= self.dr * ax * self.j_rg * dt;
        self.j_rg = er.transpose() * self.j_rg - jr(&(w * dt)) * dt;
        // state
        self.dp += self.dv * dt + 0.5 * self.dr * a * dt2;
        self.dv += self.dr * a * dt;
        self.dr *= er;
    }

    /// Body-frame motion at time tau after the interval start: (dR(tau), dp(tau), dv(tau)), with the
    /// preintegration biases.
    pub fn at(&self, tau: f64) -> (M3, V3) {
        if self.segs.is_empty() {
            return (M3::identity(), V3::zeros());
        }
        let s = self
            .segs
            .partition_point(|s| s.tau1 < tau)
            .min(self.segs.len() - 1);
        let sg = &self.segs[s];
        let t0 = if s == 0 { 0.0 } else { self.segs[s - 1].tau1 };
        let d = (tau - t0).max(0.0);
        let dr = sg.dr * so3::exp(&(sg.w * d));
        let dp = sg.dp + sg.dv * d + 0.5 * sg.dr * sg.a * d * d;
        (dr, dp)
    }
}

/// Knot state.
#[derive(Clone, Copy, Debug)]
pub struct State {
    pub r: M3,
    pub p: V3,
    pub v: V3,
    pub bg: V3,
    pub ba: V3,
}

impl State {
    pub fn plus(&self, d: &V15) -> State {
        State {
            r: self.r * so3::exp(&V3::new(d[0], d[1], d[2])),
            p: self.p + V3::new(d[3], d[4], d[5]),
            v: self.v + V3::new(d[6], d[7], d[8]),
            bg: self.bg + V3::new(d[9], d[10], d[11]),
            ba: self.ba + V3::new(d[12], d[13], d[14]),
        }
    }

    /// Pose (R, p) at tau after this knot, using the interval preintegration (its biases).
    pub fn pose_at(&self, pi: &Preint, tau: f64) -> (M3, V3) {
        let (dr, dp) = pi.at(tau);
        (
            self.r * dr,
            self.p + self.v * tau + 0.5 * gravity() * tau * tau + self.r * dp,
        )
    }
}

/// Residual [r_R, r_v, p_p] (9) and Jacobians w.r.t. the two knot states.
pub fn imu_residual(
    pi: &Preint,
    a: &State,
    b: &State,
) -> (SVector<f64, 9>, SMatrix<f64, 9, 15>, SMatrix<f64, 9, 15>) {
    let g = gravity();
    let dt = pi.dt;
    let dbg = a.bg - pi.bg;
    let dba = a.ba - pi.ba;
    let rg = pi.j_rg * dbg;
    let drc = pi.dr * so3::exp(&rg);
    let rt = a.r.transpose();
    let er = drc.transpose() * rt * b.r;
    let r_r = so3::log(&er);
    let uv = rt * (b.v - a.v - g * dt);
    let up = rt * (b.p - a.p - a.v * dt - 0.5 * g * dt * dt);
    let r_v = uv - (pi.dv + pi.j_vg * dbg + pi.j_va * dba);
    let r_p = up - (pi.dp + pi.j_pg * dbg + pi.j_pa * dba);
    let mut r = SVector::<f64, 9>::zeros();
    r.fixed_rows_mut::<3>(0).copy_from(&r_r);
    r.fixed_rows_mut::<3>(3).copy_from(&r_v);
    r.fixed_rows_mut::<3>(6).copy_from(&r_p);
    let jri = jr_inv(&r_r);
    let mut ja = SMatrix::<f64, 9, 15>::zeros();
    let mut jb = SMatrix::<f64, 9, 15>::zeros();
    // rotation
    ja.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-jri * b.r.transpose() * a.r));
    ja.fixed_view_mut::<3, 3>(0, 9)
        .copy_from(&(-jri * er.transpose() * jr(&rg) * pi.j_rg));
    jb.fixed_view_mut::<3, 3>(0, 0).copy_from(&jri);
    // velocity
    ja.fixed_view_mut::<3, 3>(3, 0).copy_from(&skew(&uv));
    ja.fixed_view_mut::<3, 3>(3, 6).copy_from(&(-rt));
    ja.fixed_view_mut::<3, 3>(3, 9).copy_from(&(-pi.j_vg));
    ja.fixed_view_mut::<3, 3>(3, 12).copy_from(&(-pi.j_va));
    jb.fixed_view_mut::<3, 3>(3, 6).copy_from(&rt);
    // position
    ja.fixed_view_mut::<3, 3>(6, 0).copy_from(&skew(&up));
    ja.fixed_view_mut::<3, 3>(6, 3).copy_from(&(-rt));
    ja.fixed_view_mut::<3, 3>(6, 6).copy_from(&(-rt * dt));
    ja.fixed_view_mut::<3, 3>(6, 9).copy_from(&(-pi.j_pg));
    ja.fixed_view_mut::<3, 3>(6, 12).copy_from(&(-pi.j_pa));
    jb.fixed_view_mut::<3, 3>(6, 3).copy_from(&rt);
    (r, ja, jb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth() -> (ImuData, Noise) {
        let n = 400;
        let mut imu = ImuData {
            t: vec![],
            gyr: vec![],
            acc: vec![],
        };
        for k in 0..n {
            let t = k as f64 * 0.005;
            imu.t.push(t);
            imu.gyr
                .push(V3::new(0.3 * (3.0 * t).sin(), -0.2, 0.5 * (2.0 * t).cos()));
            imu.acc
                .push(V3::new(0.5 * t.cos(), 0.2, GRAV + 0.3 * (5.0 * t).sin()));
        }
        (
            imu,
            Noise {
                ng: 0.005,
                na: 0.05,
                nbg: 1e-4,
                nba: 1e-3,
            },
        )
    }

    fn st(seed: f64) -> State {
        State {
            r: so3::exp(&V3::new(0.1 * seed, -0.2, 0.3)),
            p: V3::new(1.0, 2.0 * seed, -0.5),
            v: V3::new(0.3, -0.1, 0.2 * seed),
            bg: V3::new(0.01, -0.02, 0.005 * seed),
            ba: V3::new(0.05, 0.02 * seed, -0.03),
        }
    }

    #[test]
    fn jacobians_match_numeric() {
        let (imu, nz) = synth();
        let a = st(1.0);
        let b = st(-0.7);
        let pi = Preint::new(
            &imu,
            0.1012,
            0.2034,
            &V3::new(0.012, -0.018, 0.004),
            &V3::new(0.04, 0.03, -0.02),
            &nz,
        );
        let (r0, ja, jb) = imu_residual(&pi, &a, &b);
        let h = 1e-6;
        for c in 0..15 {
            let mut d = V15::zeros();
            d[c] = h;
            let (ra, _, _) = imu_residual(&pi, &a.plus(&d), &b);
            let (rb, _, _) = imu_residual(&pi, &a, &b.plus(&d));
            let na = (ra - r0) / h;
            let nb = (rb - r0) / h;
            assert!(
                (na - ja.column(c)).norm() < 1e-4 * (1.0 + na.norm()),
                "a col {c}: {na} vs {}",
                ja.column(c)
            );
            assert!(
                (nb - jb.column(c)).norm() < 1e-4 * (1.0 + nb.norm()),
                "b col {c}: {nb} vs {}",
                jb.column(c)
            );
        }
    }

    #[test]
    fn preint_consistent_with_states() {
        // integrate a state forward with the same samples: residual must vanish
        let (imu, nz) = synth();
        let a = st(0.5);
        let pi = Preint::new(&imu, 0.1012, 0.2034, &a.bg, &a.ba, &nz);
        let (r, p) = a.pose_at(&pi, pi.dt);
        let v = a.v + gravity() * pi.dt + a.r * pi.dv;
        let b = State {
            r,
            p,
            v,
            bg: a.bg,
            ba: a.ba,
        };
        let (res, _, _) = imu_residual(&pi, &a, &b);
        assert!(res.norm() < 1e-9, "{res}");
        // bias Jacobians: preintegrate with perturbed biases and compare with the first-order correction
        let dbg = V3::new(1e-3, -2e-3, 5e-4);
        let dba = V3::new(1e-2, 5e-3, -1e-2);
        let p2 = Preint::new(&imu, 0.1012, 0.2034, &(a.bg + dbg), &(a.ba + dba), &nz);
        let dr1 = pi.dr * so3::exp(&(pi.j_rg * dbg));
        assert!(so3::log(&(dr1.transpose() * p2.dr)).norm() < 1e-7);
        assert!((pi.dv + pi.j_vg * dbg + pi.j_va * dba - p2.dv).norm() < 1e-6);
        assert!((pi.dp + pi.j_pg * dbg + pi.j_pa * dba - p2.dp).norm() < 1e-7);
    }
}
