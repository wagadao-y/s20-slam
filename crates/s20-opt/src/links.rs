//! Links between sessions without GNSS: placing a session relative to another and the relative
//! pose factors that keep them together.
//!
//! A link is either two times at the same place (a loop between sessions, measured by
//! [`Problem::register_loop`]) or points picked at the same features in both sessions. Points give
//! the initial placement (x, y, z and yaw: the sessions agree on gravity) and the region around them
//! is registered with the points of both sessions seen from anywhere along their trajectories, so
//! sessions whose paths never meet (another floor, seen through a stairwell or a window) are linked
//! by what they both see.

use crate::imu::State;
use crate::relpose::RelPose;
use crate::{LoopParams, Problem};
use anyhow::{Result, bail};
use nalgebra::SymmetricEigen;
use rayon::prelude::*;
use s20_geom::cloud::preprocess;
use s20_geom::so3::{self, M3, V3};
use s20_geom::vgicp::{align, linearize};
use s20_geom::voxelmap::GaussianVoxelMap;

/// The rotation about the vertical closest to `r` (its yaw).
pub fn yaw_only(r: &M3) -> M3 {
    // the yaw of the rotated x axis, or of the y axis when x is near vertical
    let x = r * V3::x();
    let yaw = if x.x.hypot(x.y) > 0.3 {
        x.y.atan2(x.x)
    } else {
        let y = r * V3::y();
        y.y.atan2(y.x) - std::f64::consts::FRAC_PI_2
    };
    so3::exp(&V3::new(0.0, 0.0, yaw))
}

/// The yaw and translation taking the points `b` onto `a` (least squares; two pairs at least).
pub fn place_points(a: &[V3], b: &[V3]) -> (M3, V3) {
    let n = a.len().max(1) as f64;
    let ma: V3 = a.iter().sum::<V3>() / n;
    let mb: V3 = b.iter().sum::<V3>() / n;
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for (pa, pb) in a.iter().zip(b) {
        let (da, db) = (pa - ma, pb - mb);
        sxx += db.x * da.x + db.y * da.y;
        sxy += db.x * da.y - db.y * da.x;
    }
    let rz = so3::exp(&V3::new(0.0, 0.0, sxy.atan2(sxx)));
    (rz, ma - rz * mb)
}

/// The result of registering two regions.
#[derive(Clone, Debug)]
pub struct RegionFit {
    /// the transform taking session b's world onto session a's
    pub r: M3,
    pub t: V3,
    /// fraction of b's points with a correspondence at the finest distance
    pub inliers: f64,
    /// spread of the converged guesses that agree with the best (m)
    pub spread_m: f64,
    /// the direction the points hardly fix (world), with the ratio of its weight to the strongest
    pub weak: Option<(V3, f64)>,
    pub accepted: bool,
}

/// The points within this distance of the picked places are registered (m).
pub const REGION_RADIUS: f64 = 12.0;

/// A region fit is accepted when at least this fraction of the points match at the finest
/// distance. Lower than for a loop: the regions gather points from the whole of each session
/// (blurred by the odometry's drift before the optimisation) and each session sees parts the
/// other does not.
pub const REGION_MIN_INLIERS: f64 = 0.25;

/// A loop between two sessions is accepted when at least this fraction of the points match (and
/// the guesses agree): lower than within a session, as things move between the recordings and
/// the two passes see the place from other sides.
pub const LINK_MIN_INLIERS: f64 = 0.3;

/// Whether a loop fit is accepted, between sessions or not.
pub fn loop_accepted(f: &crate::relpose::LoopFit, across: bool, lp: &LoopParams) -> bool {
    let min = if across {
        LINK_MIN_INLIERS
    } else {
        lp.min_inliers
    };
    f.inliers >= min && f.spread_m < lp.max_spread
}

/// A direction is weak when its weight is below this fraction of the strongest.
const WEAK: f64 = 0.02;

impl Problem {
    /// Moves session `s` rigidly: x -> r x + t (states, odometry and gauge prior).
    pub fn move_session(&mut self, s: usize, r: &M3, t: &V3) {
        let mv = |st: &mut State| {
            st.r = r * st.r;
            st.p = r * st.p + t;
            st.v = r * st.v;
        };
        for k in 0..self.kt.len() {
            if self.session[k] == s {
                mv(&mut self.states[k]);
                mv(&mut self.init[k]);
            }
        }
        for (k, p) in self.priors.iter_mut() {
            if self.session[*k] == s {
                mv(p);
            }
        }
    }

    /// Removes the gauge prior of session `s`: it is held by its links to the other sessions.
    pub fn free_session(&mut self, s: usize) {
        self.free[s] = true;
    }

    /// The knot of session `s` nearest to the world point `c` (current states).
    pub fn nearest_knot(&self, s: usize, c: &V3) -> Option<usize> {
        (0..self.kt.len())
            .filter(|&k| self.session[k] == s)
            .min_by(|&a, &b| {
                (self.states[a].p - c)
                    .norm_squared()
                    .total_cmp(&(self.states[b].p - c).norm_squared())
            })
    }

    /// The points of session `s` within `radius` of the world point `c`, seen from anywhere along
    /// the session (current states, world frame).
    fn region_map(&self, s: usize, c: &V3, radius: f64, range: f64) -> Vec<V3> {
        let pre = self.preintegrate(&crate::imu::Noise {
            ng: 2e-4,
            na: 5e-3,
            nbg: 1e-4,
            nba: 1e-3,
        });
        let ks: Vec<usize> = (0..self.kt.len())
            .filter(|&k| self.session[k] == s && (self.states[k].p - c).norm() < radius + range)
            .collect();
        ks.par_iter()
            .flat_map_iter(|&k| {
                let pi = pre.get(k).and_then(|p| p.as_ref());
                let st = self.states[k];
                let off = &self.pts.off;
                (off[k]..off[k + 1]).filter_map(move |i| {
                    let pi = pi?;
                    let x = V3::new(
                        self.pts.x[i][0] as f64,
                        self.pts.x[i][1] as f64,
                        self.pts.x[i][2] as f64,
                    );
                    let (r, p) = st.pose_at(pi, self.pts.tau[i] as f64);
                    let w = r * x + p;
                    ((w - c).norm() < radius).then_some(w)
                })
            })
            .collect()
    }

    /// Registers the points of session `sb` around `cb` onto those of session `sa` around `ca`
    /// (world points of the current states) from the guesses (transforms b -> a), coarse to fine.
    /// The best fit is accepted when enough points match and the guesses that reach it agree.
    #[allow(clippy::too_many_arguments)]
    pub fn register_regions(
        &self,
        sa: usize,
        ca: &V3,
        sb: usize,
        cb: &V3,
        guesses: &[(M3, V3)],
        lp: &LoopParams,
        range: f64,
    ) -> Result<RegionFit> {
        let ma = self.region_map(sa, ca, REGION_RADIUS, range);
        let mb = self.region_map(sb, cb, REGION_RADIUS, range);
        if ma.len() < 1000 || mb.len() < 1000 {
            bail!(
                "too few points around the picked places ({} and {})",
                ma.len(),
                mb.len()
            );
        }
        // about the centre of a: small numbers for the alignment
        let ma: Vec<V3> = ma.iter().map(|p| p - ca).collect();
        let mb: Vec<V3> = mb.iter().map(|p| p - cb).collect();
        let ca_ = preprocess(&ma, 0.1, 10);
        let cb_ = preprocess(&mb, 0.15, 10);
        let mut map = GaussianVoxelMap::new(0.3);
        map.insert(&ca_, &M3::identity(), &V3::zeros());
        // x_a = r x_b + t  <=>  (x_a - ca) = r (x_b - cb) + (r cb + t - ca)
        let fits: Vec<(M3, V3, f64, nalgebra::Matrix6<f64>)> = guesses
            .par_iter()
            .map(|(r0, t0)| {
                let (mut r, mut t) = (*r0, r0 * cb + t0 - ca);
                for corr in [2.0, 1.0, 0.5, 0.25] {
                    let (r1, t1, _) = align(&map, &cb_, &r, &t, corr, 20);
                    r = r1;
                    t = t1;
                }
                let lin = linearize(&map, &cb_, &r, &t, 0.15);
                (r, t, lin.inliers as f64 / cb_.len().max(1) as f64, lin.h)
            })
            .collect();
        let best = fits.iter().max_by(|x, y| x.2.total_cmp(&y.2)).unwrap();
        let agree: Vec<_> = fits.iter().filter(|f| f.2 > best.2 - 0.05).collect();
        let spread = agree
            .iter()
            .map(|f| (f.1 - best.1).norm())
            .fold(0.0, f64::max);
        // the translation the points hardly fix (the perturbation is in b's frame)
        let e = SymmetricEigen::new(best.3.fixed_view::<3, 3>(3, 3).into_owned());
        let (mut lo, mut hi) = (0, 0);
        for i in 0..3 {
            if e.eigenvalues[i] < e.eigenvalues[lo] {
                lo = i;
            }
            if e.eigenvalues[i] > e.eigenvalues[hi] {
                hi = i;
            }
        }
        let ratio = e.eigenvalues[lo].max(0.0) / e.eigenvalues[hi].max(1e-12);
        let weak = (ratio < WEAK).then(|| (best.0 * e.eigenvectors.column(lo).into_owned(), ratio));
        let (r, t) = (best.0, best.1 - best.0 * cb + ca);
        Ok(RegionFit {
            r,
            t,
            inliers: best.2,
            spread_m: spread,
            weak,
            accepted: best.2 >= REGION_MIN_INLIERS && spread < lp.max_spread,
        })
    }

    /// The relative pose factor between the knots `a` and `b` of two sessions held together by
    /// the transform (r, t) taking b's world onto a's.
    pub fn link_factor(&self, a: usize, b: usize, r: &M3, t: &V3, lp: &LoopParams) -> RelPose {
        let (sa, sb) = (&self.states[a], &self.states[b]);
        // b's pose moved into a's world, seen from a
        let rb = r * sb.r;
        let pb = r * sb.p + t;
        RelPose {
            a,
            b,
            r: sa.r.transpose() * rb,
            t: sa.r.transpose() * (pb - sa.p),
            sig_r: lp.sig_r_deg.to_radians(),
            sig_t: lp.sig_t,
            region: None,
            info: None,
            robust: None,
        }
    }

    /// The transform taking b's world onto a's that a relative pose of the knots `a` and `b`
    /// states: T_a rel T_b^-1.
    pub fn link_transform(&self, rel: &RelPose) -> (M3, V3) {
        let (sa, sb) = (&self.states[rel.a], &self.states[rel.b]);
        let ra = sa.r * rel.r;
        let pa = sa.r * rel.t + sa.p;
        let r = ra * sb.r.transpose();
        (r, pa - r * sb.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaw_and_points_place_a_session() {
        let rz = so3::exp(&V3::new(0.0, 0.0, 2.5));
        let tilt = so3::exp(&V3::new(0.01, -0.02, 0.0));
        assert!((yaw_only(&(rz * tilt)) - rz).norm() < 0.03);
        let b = [
            V3::new(1.0, 2.0, 0.5),
            V3::new(-3.0, 4.0, 1.5),
            V3::new(0.5, -1.0, 2.0),
        ];
        let t = V3::new(10.0, -5.0, 3.0);
        let a: Vec<V3> = b.iter().map(|p| rz * p + t).collect();
        let (r, tt) = place_points(&a, &b);
        assert!((r - rz).norm() < 1e-9 && (tt - t).norm() < 1e-9);
    }
}
