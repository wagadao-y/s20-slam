//! Relative pose factors between two knots: loops (two times at the same place, measured by
//! registering the local maps around them) and, in the pose-graph stage, the odometry between
//! consecutive knots.
//!
//! Residual r = [Log(R_m^T R_a^T R_b); R_a^T (p_b - p_a) - t_m] in the frame of knot a, with the
//! right perturbation of the rotations and world positions of the knot states.

use crate::imu::{M15, State, jr_inv};
use crate::{LoopParams, Problem, planes};
use anyhow::{Result, bail};
use nalgebra::{SMatrix, SVector};
use rayon::prelude::*;
use s20_geom::cloud::preprocess;
use s20_geom::so3::{self, M3, V3, skew};
use s20_geom::vgicp::align;
use s20_geom::voxelmap::GaussianVoxelMap;

type M6 = SMatrix<f64, 6, 6>;
type V6 = SVector<f64, 6>;

/// Pose of knot `b` in the frame of knot `a`, with standard deviations (rad, m).
#[derive(Clone, Copy, Debug)]
pub struct RelPose {
    pub a: usize,
    pub b: usize,
    pub r: M3,
    pub t: V3,
    pub sig_r: f64,
    pub sig_t: f64,
    /// measured by registering the points around these places (in the bodies of knots a and b)
    /// rather than the local maps of the two times (see [`crate::links`])
    pub region: Option<(V3, V3)>,
}

/// A linearised relative pose factor on the [theta, p] blocks of two knots.
pub struct RelTerm {
    pub a: usize,
    pub b: usize,
    pub haa: M6,
    pub hab: M6,
    pub hbb: M6,
    pub ga: V6,
    pub gb: V6,
    pub cost: f64,
}

impl RelPose {
    /// Residual and Jacobians w.r.t. [theta_a, p_a] and [theta_b, p_b].
    fn linearize(&self, sa: &State, sb: &State) -> (V6, M6, M6) {
        let rt = sa.r.transpose();
        let er = self.r.transpose() * rt * sb.r;
        let r_r = so3::log(&er);
        let d = rt * (sb.p - sa.p);
        let mut r = V6::zeros();
        r.fixed_rows_mut::<3>(0).copy_from(&r_r);
        r.fixed_rows_mut::<3>(3).copy_from(&(d - self.t));
        let jri = jr_inv(&r_r);
        let mut ja = M6::zeros();
        let mut jb = M6::zeros();
        ja.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-jri * sb.r.transpose() * sa.r));
        jb.fixed_view_mut::<3, 3>(0, 0).copy_from(&jri);
        ja.fixed_view_mut::<3, 3>(3, 0).copy_from(&skew(&d));
        ja.fixed_view_mut::<3, 3>(3, 3).copy_from(&(-rt));
        jb.fixed_view_mut::<3, 3>(3, 3).copy_from(&rt);
        (r, ja, jb)
    }

    /// How far the states are from the measured relative pose: (rotation deg, translation m).
    pub fn error(&self, states: &[State]) -> (f64, f64) {
        let (r, _, _) = self.linearize(&states[self.a], &states[self.b]);
        (
            r.fixed_rows::<3>(0).norm().to_degrees(),
            r.fixed_rows::<3>(3).norm(),
        )
    }

    pub fn term(&self, states: &[State]) -> RelTerm {
        let (r, ja, jb) = self.linearize(&states[self.a], &states[self.b]);
        let wr = 1.0 / (self.sig_r * self.sig_r);
        let wt = 1.0 / (self.sig_t * self.sig_t);
        let w = M6::from_diagonal(&V6::new(wr, wr, wr, wt, wt, wt));
        RelTerm {
            a: self.a,
            b: self.b,
            haa: ja.transpose() * w * ja,
            hab: ja.transpose() * w * jb,
            hbb: jb.transpose() * w * jb,
            ga: ja.transpose() * w * r,
            gb: jb.transpose() * w * r,
            cost: 0.5 * r.dot(&(w * r)),
        }
    }
}

/// A 6x6 block on [theta, p] placed in a 15x15 knot block.
pub fn embed(m: &M6) -> M15 {
    let mut o = M15::zeros();
    o.fixed_view_mut::<6, 6>(0, 0).copy_from(m);
    o
}

/// Result of registering the local maps of a loop.
#[derive(Clone, Debug)]
pub struct LoopFit {
    pub rel: RelPose,
    /// fraction of the source points with a correspondence at the finest distance
    pub inliers: f64,
    /// how far the measured pose is from the current trajectory (m, deg)
    pub gap_m: f64,
    pub gap_deg: f64,
    /// spread of the converged guesses that agree with the best (m); large means ambiguous
    pub spread_m: f64,
    pub accepted: bool,
}

impl Problem {
    /// The knot whose interval holds time `t`.
    pub fn knot_at(&self, t: f64) -> Option<usize> {
        let i = self.kt.partition_point(|&s| s <= t);
        (i > 0 && i <= self.kt.len()).then(|| i - 1)
    }

    /// Points around knot `c` (knots within `half` seconds, same session, points within `radius` m),
    /// in the frame of knot c.
    fn local_map(&self, c: usize, half: f64, radius: f64) -> Vec<V3> {
        let pre = self.preintegrate(&crate::imu::Noise {
            ng: 2e-4,
            na: 5e-3,
            nbg: 1e-4,
            nba: 1e-3,
        });
        let (rc, pc) = (self.states[c].r, self.states[c].p);
        let rct = rc.transpose();
        let ks: Vec<usize> = (0..self.kt.len())
            .filter(|&k| {
                self.session[k] == self.session[c] && (self.kt[k] - self.kt[c]).abs() <= half
            })
            .collect();
        ks.par_iter()
            .flat_map_iter(|&k| {
                let pi = pre.get(k).and_then(|p| p.as_ref());
                let s = self.states[k];
                let off = &self.pts.off;
                (off[k]..off[k + 1]).filter_map(move |i| {
                    let pi = pi?;
                    let x = V3::new(
                        self.pts.x[i][0] as f64,
                        self.pts.x[i][1] as f64,
                        self.pts.x[i][2] as f64,
                    );
                    let (r, p) = s.pose_at(pi, self.pts.tau[i] as f64);
                    let w = r * x + p;
                    // far points add little and slow the alignment
                    let q = rct * (w - pc);
                    (q.norm() < radius).then_some(q)
                })
            })
            .collect()
    }

    /// Registers the local maps around times `ta` and `tb` (said to be the same place): the pose of
    /// b's knot in the frame of a's knot. Several initial guesses are tried (the current trajectory,
    /// and the same position with yaw offsets), coarse to fine; the best is accepted when enough
    /// points match and the guesses that reach it agree.
    pub fn register_loop(&self, ta: f64, tb: f64, lp: &LoopParams) -> Result<LoopFit> {
        let (Some(a), Some(b)) = (self.knot_at(ta), self.knot_at(tb)) else {
            bail!("loop times outside the trajectory");
        };
        if a == b {
            bail!("the two loop times fall on the same knot");
        }
        let ma = self.local_map(a, lp.half_s, lp.radius);
        let mb = self.local_map(b, lp.half_s, lp.radius);
        if ma.len() < 1000 || mb.len() < 1000 {
            bail!("too few points around the loop times");
        }
        let ca = preprocess(&ma, 0.1, 10);
        let cb = preprocess(&mb, 0.15, 10);
        let mut map = GaussianVoxelMap::new(0.3);
        map.insert(&ca, &M3::identity(), &V3::zeros());
        let (sa, sb) = (&self.states[a], &self.states[b]);
        let cur_r = sa.r.transpose() * sb.r;
        let cur_t = sa.r.transpose() * (sb.p - sa.p);
        let mut guesses = vec![(cur_r, cur_t)];
        for deg in [0.0f64, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 180.0] {
            let yaw = so3::exp(&(sa.r.transpose() * V3::z() * deg.to_radians()));
            guesses.push((yaw * cur_r, V3::zeros()));
        }
        let fits: Vec<(M3, V3, f64)> = guesses
            .par_iter()
            .map(|(r0, t0)| {
                let (mut r, mut t) = (*r0, *t0);
                for corr in [2.0, 1.0, 0.5, 0.25] {
                    let (r1, t1, _) = align(&map, &cb, &r, &t, corr, 20);
                    r = r1;
                    t = t1;
                }
                let lin = s20_geom::vgicp::linearize(&map, &cb, &r, &t, 0.15);
                (r, t, lin.inliers as f64 / cb.len().max(1) as f64)
            })
            .collect();
        let best = fits.iter().max_by(|x, y| x.2.total_cmp(&y.2)).unwrap();
        // the guesses that reached (nearly) the best fit: how much they disagree
        let agree: Vec<&(M3, V3, f64)> = fits.iter().filter(|f| f.2 > best.2 - 0.05).collect();
        let spread = agree
            .iter()
            .map(|f| (f.1 - best.1).norm())
            .fold(0.0, f64::max);
        let gap_m = (best.1 - cur_t).norm();
        let gap_deg = so3::log(&(cur_r.transpose() * best.0)).norm().to_degrees();
        let accepted = best.2 >= lp.min_inliers && spread < lp.max_spread;
        Ok(LoopFit {
            rel: RelPose {
                a,
                b,
                r: best.0,
                t: best.1,
                sig_r: lp.sig_r_deg.to_radians(),
                sig_t: lp.sig_t,
                region: None,
            },
            inliers: best.2,
            gap_m,
            gap_deg,
            spread_m: spread,
            accepted,
        })
    }

    /// The time of knot `k` as seconds from its session's first knot, with the session when there
    /// are several.
    pub fn knot_label(&self, k: usize) -> String {
        let s = self.session[k];
        let first = self.session.partition_point(|&x| x < s);
        if self.n_sessions() > 1 {
            format!("session {s} {:.1} s", self.kt[k] - self.kt[first])
        } else {
            format!("{:.1} s", self.kt[k] - self.kt[first])
        }
    }

    /// Registers every loop again on the current states and takes the new measurement when it is
    /// accepted (else the previous one stays), logging how far each one moved. `range`: of the
    /// LiDAR, for the loops measured on the regions around picked points.
    pub fn remeasure_loops(&mut self, lp: &LoopParams, range: f64, log: &dyn Fn(&str)) {
        for k in 0..self.loops.len() {
            let l = self.loops[k];
            let what = format!("loop {} = {}", self.knot_label(l.a), self.knot_label(l.b));
            let fit = match l.region {
                None => self.register_loop(self.kt[l.a], self.kt[l.b], lp),
                Some((xa, xb)) => {
                    let (sa, sb) = (&self.states[l.a], &self.states[l.b]);
                    let (ca, cb) = (sa.r * xa + sa.p, sb.r * xb + sb.p);
                    self.register_regions(
                        self.session[l.a],
                        &ca,
                        self.session[l.b],
                        &cb,
                        &[(M3::identity(), V3::zeros())],
                        lp,
                        range,
                    )
                    .map(|f| {
                        let mut rel = self.link_factor(l.a, l.b, &f.r, &f.t, lp);
                        rel.region = l.region;
                        crate::relpose::LoopFit {
                            rel,
                            inliers: f.inliers,
                            gap_m: 0.0,
                            gap_deg: 0.0,
                            spread_m: f.spread_m,
                            accepted: f.accepted,
                        }
                    })
                }
            };
            let across = self.session[l.a] != self.session[l.b];
            match fit {
                Ok(f)
                    if f.accepted
                        || (l.region.is_none() && crate::links::loop_accepted(&f, across, lp)) =>
                {
                    log(&format!(
                        "  {what} measured again: {:.1} mm / {:.3} deg from the previous measurement ({:.0}% matched)",
                        (f.rel.t - l.t).norm() * 1e3,
                        so3::log(&(l.r.transpose() * f.rel.r)).norm().to_degrees(),
                        f.inliers * 100.0
                    ));
                    self.loops[k] = f.rel;
                }
                Ok(f) => log(&format!(
                    "  {what} measured again: not accepted ({:.0}% matched), the previous measurement stays",
                    f.inliers * 100.0
                )),
                Err(e) => log(&format!(
                    "  {what} measured again: {e:#}; the previous measurement stays"
                )),
            }
        }
    }

    /// Relative pose factors of the odometry (the states as given) between consecutive knots of a
    /// session: they keep the trajectory's shape in the pose-graph stage.
    pub fn odometry_factors(&self) -> Vec<RelPose> {
        (0..self.kt.len().saturating_sub(1))
            .filter(|&k| self.session[k] == self.session[k + 1])
            .map(|k| {
                let (a, b) = (&self.init[k], &self.init[k + 1]);
                RelPose {
                    a: k,
                    b: k + 1,
                    r: a.r.transpose() * b.r,
                    t: a.r.transpose() * (b.p - a.p),
                    sig_r: 0.05f64.to_radians(),
                    sig_t: 0.005,
                    region: None,
                }
            })
            .collect()
    }
}

/// The planes of a stage without LiDAR terms (pose-graph stage).
pub fn no_planes() -> planes::PlaneParams {
    planes::PlaneParams {
        levels: Vec::new(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::V15;

    fn st(seed: f64) -> State {
        State {
            r: so3::exp(&V3::new(0.1 * seed, -0.2, 0.3 + seed)),
            p: V3::new(1.0, 2.0 * seed, -0.5),
            v: V3::zeros(),
            bg: V3::zeros(),
            ba: V3::zeros(),
        }
    }

    #[test]
    fn jacobians_match_numeric() {
        let (a, b) = (st(0.4), st(-0.9));
        let rel = RelPose {
            a: 0,
            b: 1,
            r: so3::exp(&V3::new(0.05, 0.4, -0.2)),
            t: V3::new(0.3, -1.0, 0.2),
            sig_r: 1.0,
            sig_t: 1.0,
            region: None,
        };
        let (r0, ja, jb) = rel.linearize(&a, &b);
        let h = 1e-6;
        for c in 0..6 {
            let mut d = V15::zeros();
            d[c] = h;
            let (ra, _, _) = rel.linearize(&a.plus(&d), &b);
            let (rb, _, _) = rel.linearize(&a, &b.plus(&d));
            assert!(((ra - r0) / h - ja.column(c)).norm() < 1e-4, "a col {c}");
            assert!(((rb - r0) / h - jb.column(c)).norm() < 1e-4, "b col {c}");
        }
    }
}
