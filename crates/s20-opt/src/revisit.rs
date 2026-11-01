//! Revisits: places the trajectory passes again, later in the same session or in another one,
//! found and registered automatically, so that every "same place" holds the trajectory the same
//! way, whether the user marked it or not.
//!
//! The trajectory is cut into submaps of `len_s` seconds of one session (the points around the
//! centre knot, placed by the current states: within a submap the odometry is good to a few
//! centimetres). Two submaps whose centres are within `max_dist` and that are not neighbours in
//! time are registered (VGICP from the current relative pose and a few degrees either way); of the
//! submaps of one pass near a submap only the nearest is taken. An accepted registration becomes a
//! relative pose factor between the two centre knots, weighted by the registration's information
//! (strong where walls and corners fix the pose, weak along a hedge or a road) scaled so that its
//! strongest direction has `sig_t` / `sig_r_deg`, and robust (Cauchy, `robust` standard
//! deviations): where the revisits and the loops disagree, the disagreement goes where the points
//! fix the least, and a wrong registration pulls little. A registration along a path (the points
//! hardly fix the position along it) is trusted less as a whole: in a wood the passes share few
//! points and its rotation and position across the path scatter by decimetres.

use crate::Problem;
use crate::imu::Preint;
use crate::relpose::RelPose;
use nalgebra::{Matrix3, SMatrix, SymmetricEigen};
use rayon::prelude::*;
use s20_geom::cloud::{CovCloud, preprocess};
use s20_geom::so3::{self, M3, V3};
use s20_geom::vgicp::{align, linearize};
use s20_geom::voxelmap::GaussianVoxelMap;
use std::collections::HashMap;

type M6 = SMatrix<f64, 6, 6>;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RevisitParams {
    /// find and use the revisits (off: only the loops marked, as before)
    pub enabled: bool,
    /// a submap holds the knots within half this many seconds of its centre (s)
    pub len_s: f64,
    /// and the points within this distance of its centre knot (m)
    pub radius: f64,
    /// two submaps of one session are a revisit when their centres are this far apart in time (s)
    pub min_gap_s: f64,
    /// submaps whose centres are within this distance are registered (m)
    pub max_dist: f64,
    /// fraction of the points matched at the finest distance for a registration to be used
    /// (passes that see a wall from far apart share few points but fix each other well)
    pub min_inliers: f64,
    /// a registration is along a path when its weakest translation direction has less than this
    /// share of the strongest's information
    pub min_weak: f64,
    /// and its standard deviations are this many times `sig_t` / `sig_r_deg`
    pub path_sd: f64,
    /// the guesses that reach the best fit agree within this distance (m)
    pub max_spread: f64,
    /// standard deviations of the strongest direction of a registration (m, deg)
    pub sig_t: f64,
    pub sig_r_deg: f64,
    /// Cauchy scale on the whitened residual of the revisits and the loops (standard deviations)
    pub robust: f64,
}

impl Default for RevisitParams {
    fn default() -> Self {
        RevisitParams {
            enabled: false,
            len_s: 15.0,
            radius: 30.0,
            min_gap_s: 60.0,
            max_dist: 20.0,
            min_inliers: 0.2,
            min_weak: 0.05,
            path_sd: 3.0,
            max_spread: 0.1,
            sig_t: 0.01,
            sig_r_deg: 0.05,
            robust: 3.0,
        }
    }
}

impl RevisitParams {
    /// Whether a registration whose weakest direction has this share of the strongest is along
    /// a path.
    pub fn along_path(&self, weak: f64) -> bool {
        weak < self.min_weak
    }
}

/// A registered pair of submaps.
#[derive(Clone, Debug)]
pub struct Revisit {
    /// the factor between the centre knots (weighted, not yet robust)
    pub rel: RelPose,
    pub inliers: f64,
    pub spread_m: f64,
    pub accepted: bool,
    /// how far the registration is from the current trajectory (m, deg): how doubled the two
    /// passes' points are
    pub gap_m: f64,
    pub gap_deg: f64,
    /// the weakest direction's information over the strongest's (translation)
    pub weak: f64,
}

/// A submap: the points around its centre knot in that knot's frame, as the registration takes
/// them (target voxels, source points).
struct Submap {
    target: GaussianVoxelMap,
    source: CovCloud,
}

impl Problem {
    /// The centre knots of the submaps: every `len_s` seconds of each session.
    pub fn submap_centres(&self, rp: &RevisitParams) -> Vec<usize> {
        let mut out = Vec::new();
        let mut next = f64::NEG_INFINITY;
        for k in 0..self.kt.len() {
            if k == 0 || self.session[k] != self.session[k - 1] {
                next = self.kt[k] + 0.5 * rp.len_s;
            }
            if self.kt[k] >= next {
                out.push(k);
                next += rp.len_s;
            }
        }
        out
    }

    /// The pairs of centre knots to register: within `max_dist`, of other sessions or at least
    /// `min_gap_s` apart; of the submaps of one pass near a submap, the nearest.
    pub fn revisit_pairs(&self, rp: &RevisitParams) -> Vec<(usize, usize)> {
        let c = self.submap_centres(rp);
        let mut out = Vec::new();
        for i in 0..c.len() {
            let (a, pa) = (c[i], self.states[c[i]].p);
            // (index of the last submap of the pass, nearest submap, its distance)
            let mut pass: Option<(usize, usize, f64)> = None;
            for (j, &b) in c.iter().enumerate().skip(i + 1) {
                let same = self.session[a] == self.session[b];
                let d = (self.states[b].p - pa).norm();
                if d > rp.max_dist || (same && self.kt[b] - self.kt[a] < rp.min_gap_s) {
                    continue;
                }
                match &mut pass {
                    Some((last, best, bd))
                        if j == *last + 1 && self.session[c[*last]] == self.session[b] =>
                    {
                        *last = j;
                        if d < *bd {
                            (*best, *bd) = (b, d);
                        }
                    }
                    _ => {
                        if let Some((_, best, _)) = pass {
                            out.push((a, best));
                        }
                        pass = Some((j, b, d));
                    }
                }
            }
            if let Some((_, best, _)) = pass {
                out.push((a, best));
            }
        }
        out
    }

    fn submap(&self, pre: &[Option<Preint>], c: usize, rp: &RevisitParams) -> Submap {
        let pts = self.local_map_with(pre, c, 0.5 * rp.len_s, rp.radius);
        let target = preprocess(&pts, 0.1, 10);
        let mut map = GaussianVoxelMap::new(0.3);
        map.insert(&target, &M3::identity(), &V3::zeros());
        Submap {
            target: map,
            source: preprocess(&pts, 0.15, 10),
        }
    }

    /// Registers the submaps of the pairs of centre knots, on the current states.
    pub fn register_revisits(&self, rp: &RevisitParams, pairs: &[(usize, usize)]) -> Vec<Revisit> {
        let pre = self.map_preint();
        let mut knots: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        knots.sort_unstable();
        knots.dedup();
        let maps: HashMap<usize, Submap> = knots
            .par_iter()
            .map(|&c| (c, self.submap(&pre, c, rp)))
            .collect();
        pairs
            .par_iter()
            .map(|&(a, b)| self.register_pair(rp, a, &maps[&a], b, &maps[&b]))
            .collect()
    }

    /// Finds and registers the revisits again on the current states (between the stages, as the
    /// loops are measured again) and puts them in place of the previous ones: a submap measured
    /// on a distorted odometry is measured again on a better one, and passes that were metres
    /// apart before the loops pulled them together are registered from close by (along a path
    /// the registration reaches the right place only from there).
    pub fn remeasure_revisits(&mut self, rp: &RevisitParams, log: &dyn Fn(&str)) {
        let before = self.loops.iter().filter(|l| l.info.is_some()).count();
        let pairs = self.revisit_pairs(rp);
        let found = self.register_revisits(rp, &pairs);
        self.loops.retain(|l| l.info.is_none());
        let mut gap: Vec<f64> = Vec::new();
        let mut along = 0;
        for f in found.iter().filter(|f| f.accepted) {
            gap.push(f.gap_m * 1e3);
            along += usize::from(rp.along_path(f.weak));
            self.loops.push(RelPose {
                robust: Some(rp.robust),
                ..f.rel
            });
        }
        gap.sort_by(f64::total_cmp);
        log(&format!(
            "  revisits found again: {} of {} pairs accepted ({} along a path; {} before), gap to the trajectory median {:.1} mm, max {:.1} mm",
            gap.len(),
            pairs.len(),
            along,
            before,
            gap.get(gap.len() / 2).copied().unwrap_or(0.0),
            gap.last().copied().unwrap_or(0.0)
        ));
    }

    fn register_pair(
        &self,
        rp: &RevisitParams,
        a: usize,
        ma: &Submap,
        b: usize,
        mb: &Submap,
    ) -> Revisit {
        let (sa, sb) = (&self.states[a], &self.states[b]);
        let cur_r = sa.r.transpose() * sb.r;
        let cur_t = sa.r.transpose() * (sb.p - sa.p);
        // the current relative pose, and turned a few degrees about a's vertical
        let up = sa.r.transpose() * V3::z();
        let fits: Vec<(M3, V3, f64, M6)> = [0.0f64, 5.0, -5.0]
            .par_iter()
            .map(|deg| {
                let yaw = so3::exp(&(up * deg.to_radians()));
                let (mut r, mut t) = (yaw * cur_r, yaw * cur_t);
                for corr in [2.0, 1.0, 0.5, 0.25] {
                    let (r1, t1, _) = align(&ma.target, &mb.source, &r, &t, corr, 20);
                    r = r1;
                    t = t1;
                }
                let lin = linearize(&ma.target, &mb.source, &r, &t, 0.15);
                (
                    r,
                    t,
                    lin.inliers as f64 / mb.source.points.len().max(1) as f64,
                    lin.h,
                )
            })
            .collect();
        let best = fits.iter().max_by(|x, y| x.2.total_cmp(&y.2)).unwrap();
        let spread = fits
            .iter()
            .filter(|f| f.2 > best.2 - 0.05)
            .map(|f| (f.1 - best.1).norm())
            .fold(0.0, f64::max);
        let (info, weak) = residual_info(&best.0, &best.3, rp);
        Revisit {
            rel: RelPose {
                a,
                b,
                r: best.0,
                t: best.1,
                sig_r: rp.sig_r_deg.to_radians(),
                sig_t: rp.sig_t,
                region: None,
                info: Some(info),
                robust: None,
            },
            inliers: best.2,
            spread_m: spread,
            accepted: best.2 >= rp.min_inliers && spread < rp.max_spread,
            gap_m: (best.1 - cur_t).norm(),
            gap_deg: so3::log(&(cur_r.transpose() * best.0)).norm().to_degrees(),
            weak,
        }
    }
}

/// The information of the relative pose factor's residual [Log(R_m^T R); R_a^T dp - t_m] from the
/// registration's Hessian H on the right perturbation [omega, v] of the transform b -> a (v in b's
/// frame: the residual's translation is R_m v), scaled so that the strongest rotation and
/// translation directions are no stronger than `sig_r_deg` and `sig_t`. Also the weakest
/// translation direction's share of the strongest.
fn residual_info(r_m: &M3, h: &M6, rp: &RevisitParams) -> (M6, f64) {
    let mut d = M6::identity();
    d.fixed_view_mut::<3, 3>(3, 3).copy_from(r_m);
    let info = d * h * d.transpose();
    let info = 0.5 * (info + info.transpose());
    let eig = |m: Matrix3<f64>| {
        let e = SymmetricEigen::new(m).eigenvalues;
        (e.min().max(0.0), e.max().max(1e-12))
    };
    let (_, rmax) = eig(info.fixed_view::<3, 3>(0, 0).into_owned());
    let (tmin, tmax) = eig(info.fixed_view::<3, 3>(3, 3).into_owned());
    let weak = tmin / tmax;
    let k = if rp.along_path(weak) { rp.path_sd } else { 1.0 };
    let (sig_t, sig_r) = (k * rp.sig_t, k * rp.sig_r_deg.to_radians());
    let s = (1.0 / (sig_t * sig_t) / tmax).min(1.0 / (sig_r * sig_r) / rmax);
    (info * s, weak)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_information_turns_with_the_measurement_and_keeps_a_weak_direction_weak() {
        let rp = RevisitParams::default();
        // translation fixed along b's x and y, hardly along z; rotation well fixed
        let h = M6::from_diagonal(&nalgebra::Vector6::new(1e4, 1e4, 1e4, 100.0, 100.0, 1.0));
        let r_m = so3::exp(&V3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));
        let (info, weak) = residual_info(&r_m, &h, &rp);
        assert!((weak - 0.01).abs() < 1e-9);
        // b's z is a's -y after the quarter turn about x: the weak direction there
        let t = info.fixed_view::<3, 3>(3, 3);
        assert!(t[(1, 1)] < 0.02 * t[(0, 0)]);
        assert!((t[(2, 2)] - t[(0, 0)]).abs() < 1e-6 * t[(0, 0)]);
        // along a path: the strongest translation direction at path_sd sig_t, the rotation no
        // stronger than path_sd sig_r
        let (sig_t, sig_r) = (
            rp.path_sd * rp.sig_t,
            rp.path_sd * rp.sig_r_deg.to_radians(),
        );
        assert!((t[(0, 0)] - 1.0 / (sig_t * sig_t)).abs() < 1e-6 * t[(0, 0)]);
        assert!(info[(0, 0)] <= 1.0 / (sig_r * sig_r) * (1.0 + 1e-9));
    }

    #[test]
    fn a_registration_along_a_path_is_trusted_less() {
        let rp = RevisitParams::default();
        let r_m = M3::identity();
        let firm = M6::from_diagonal(&nalgebra::Vector6::new(1e4, 1e4, 1e4, 100.0, 100.0, 50.0));
        let path = M6::from_diagonal(&nalgebra::Vector6::new(1e4, 1e4, 1e4, 100.0, 100.0, 1.0));
        let (fi, fw) = residual_info(&r_m, &firm, &rp);
        let (pi, pw) = residual_info(&r_m, &path, &rp);
        assert!(!rp.along_path(fw) && rp.along_path(pw));
        let k = rp.path_sd * rp.path_sd;
        assert!((fi[(3, 3)] / pi[(3, 3)] - k).abs() < 1e-6 * k);
    }
}
