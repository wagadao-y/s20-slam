//! The point cloud from the deskewed observations: representative sampling, free space and
//! isolation, plane projection from all observations ([`surface`]), then post thinning, and
//! passing objects over the scans read again ([`transient`]). Every output point is an
//! observation, moved along its plane's normal at most; no centroids.
//! Ported from the author's earlier prototype (slam-experiment), where the settings were chosen.
pub mod color;
pub mod freespace;
pub mod gpu;
pub mod gpu_free;
pub mod gpu_transient;
pub mod load;
pub mod memory;
pub mod noreturn;
pub mod surface;
pub mod tiles;
pub mod transient;

use anyhow::{Result, ensure};
use rayon::prelude::*;
use s20_io::job::Job;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One measured point: world position, the sensor position it was measured from, the range,
/// the reflectivity, the scan (of the dataset, over its sessions) and the time (sensor clock).
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub time: f64,
    pub xyz: [f32; 3],
    pub origin: [f32; 3],
    pub range: f32,
    pub scan: u32,
    pub intensity: u16,
}

/// The finished state of a representative observation.
#[derive(Clone, Copy, Debug)]
pub struct PointResult {
    pub xyz: [f32; 3],
    pub normal: [f32; 3],
    pub shift: f32,
    /// Absolute distance from the original query to the fitted support plane (NaN: no reliable fit).
    pub residual: f32,
    pub sigma: f32,
    pub neighbors: u32,
    /// [`STATUS_NAMES`] index: 3 projected, 5 sparse, 6 nonplanar, 9 projected at a smaller
    /// radius, 16 off its plane and kept; 11 isolated, 12 free space, 13 post thinning and 17 a
    /// passing object are left out. The other codes are unused (kept for the numbering of
    /// slam-experiment).
    pub status: u32,
}
/// Status of points left out by the post-finishing uniform sampling (`post_leaf`).
pub const THINNED: u32 = 13;
/// Status of sparse and nonplanar points a passing object left ([`transient`]).
pub const TRANSIENT: u32 = 17;
pub const STATUS_NAMES: [&str; 18] = [
    "未補正",
    "SOR除去",
    "面外れ点除去",
    "面へ投影",
    "境界保護",
    "支持不足",
    "非平面",
    "補正上限",
    "境界の面へ投影",
    "小半径で投影",
    "もや除去",
    "裏付け不足",
    "空き（通り抜け）",
    "後段の間引き",
    "近くの面へ寄せる",
    "別向きの面へ投影",
    "外れ点を残す",
    "通過物",
];
/// Whether a status leaves the point out of the cloud.
pub fn removed(status: u32) -> bool {
    matches!(status, 1 | 2 | 10..=13 | TRANSIENT)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FinishParams {
    /// Leave out the scans of each session before the sensor first moves this far (m) from its
    /// first position and after it is last this far from its final one: the device set down
    /// with the operator standing behind it. 0: keep all.
    pub trim_still: f64,
    /// Observations farther than this (m) are not read.
    pub max_range: f64,
    /// Observations nearer than this are not output (still used as support unless
    /// `support_min_range` also excludes them). 0: keep all.
    pub min_range: f64,
    /// Observations nearer than this are not used for planes.
    pub support_min_range: f64,
    /// Cell size of the per-cell sums (moments) the planes are fitted on.
    pub moment_cell: f64,
    /// Minimum number of observations behind a plane.
    pub min_support: usize,
    /// Retries at half radius when the full radius gives no plane or rejects (a retry below
    /// 2.5 moment cells is not made: with 8 cm and 1 cm cells, one retry at 4 cm).
    pub fallback_scales: usize,
    /// Skip the consensus where one plane over all cells already holds 97% of the weight (same
    /// result, several times faster on walls and floors).
    pub fast_plane: bool,
    /// The consensus tries the local planes (within `consensus_seed_radius`) around this many
    /// seed cells, the first nearest the query and the rest by farthest-point sampling, so the
    /// result does not depend on the order of the neighbour list.
    pub consensus_seeds: usize,
    pub consensus_seed_radius: f64,
    /// The planes are fitted once per cell of this size (m) and side (octant of the viewing ray),
    /// at the mean of its queries, and shared by them; each query still picks its own plane and
    /// is tested against it. A query seen from the other side of a shared plane than the cell's
    /// mean ray is fitted alone; a query the shared fit rejects or leaves without a plane goes to
    /// the smaller-radius retries, fitted for the query alone. 0: every query alone. (2 cm to
    /// 4 cm, 2026-10-29: the plane classification 220 -> 101 s on the shrine 2026-09-23, 106 ->
    /// 46 s indoors; the surfaces as thick, 0.3-0.6% of the points over 1 cm away, on
    /// rough surfaces, none on smooth ones; no difference seen.)
    pub anchor_grid: f64,
    /// A query with fewer raw observations than this within max(`support_radius`,
    /// `support_range` × its range) is not output (0: off).
    pub support_min_obs: usize,
    pub support_radius: f64,
    pub support_range: f64,
    /// A point whose voxel more than `free_ratio` × the scans that measured it saw through (and
    /// at least `free_min_pass`) is free space, not output (0: off).
    pub free_ratio: f64,
    pub free_min_pass: u32,
    pub free_voxel: f64,
    /// Rays stop this far before their own point.
    pub free_margin: f64,
    /// Observations in free space do not support planes either.
    pub free_support: bool,
    /// A ray only passes a measured voxel when it comes within this distance (m) of the centroid
    /// of the voxel's measurements (0: whenever it enters the voxel). A thin object (a handrail)
    /// fills a corner of its voxels; rays grazing it on their way to the wall behind otherwise
    /// count as passing through it.
    pub free_pass_radius: f64,
    /// With a pass radius: a ray missing the centroid still passes when it goes on at least this
    /// far (m) beyond the voxel (0: no exception). Rays grazing a thin object end on the wall just
    /// behind it; rays through a mirror image or a moved object go on.
    pub free_pass_far: f64,
    /// Thin-object protection: a voxel measured by at least `free_protect_direction_scans`
    /// distinct scans whose viewing directions spread (1 - length of the mean unit direction)
    /// more than this is never free (0: off). A pull-up bar is seen from all around the room by
    /// many scans; a mirror image only through the mirror, haze in foliage by few scans.
    pub free_protect_direction_spread: f64,
    pub free_protect_direction_scans: u32,
    /// A second free-space count on voxels of this size (m) with `free_fine_ratio` and no
    /// protection; a point free in either count is left out (0: off). On 1 cm voxels a thin
    /// object (a light cord) and the mixed returns around it fall into separate voxels: the
    /// latter are seen through, the object is not.
    pub free_fine_voxel: f64,
    pub free_fine_ratio: f64,
    /// Passing objects ([`transient`]): a sparse or nonplanar point at most `transient_max_hits`
    /// scans measured, all within `transient_hit_span_s` seconds, which at least
    /// `transient_min_pass` and more than `transient_ratio` x the hits saw through, is not output.
    /// A scan sees through the point when one of its rays within this angle (degrees) of the
    /// sensor-to-point direction ends `transient_far` (m) beyond it and none ends short of it
    /// (at least `transient_clean_share` of the scans with a ray beyond must see through so; on
    /// ground seen at a grazing angle some rays always land short), and sees it when one ends
    /// within `transient_hit` (m) of its distance; only scans within `transient_max_range` (m)
    /// count. 0: off.
    pub transient_angle_deg: f64,
    pub transient_far: f64,
    pub transient_hit: f64,
    pub transient_max_hits: u32,
    pub transient_clean_share: f64,
    pub transient_hit_span_s: f64,
    pub transient_ratio: f64,
    pub transient_min_pass: u32,
    pub transient_max_range: f64,
    /// A query further than this x max(plane σ, [`range_sigma`]) from its reliable plane is off
    /// it: it is tried again at a smaller radius, and otherwise stays in the output moved as far
    /// as the taper allows ("外れ点を残す"). A query within this distance is measurement error
    /// and is projected: the taper ends here when it is beyond `mls_taper_end` (the returns from
    /// afar sit up to several cm off the surface the close ones measured).
    pub reject_sigma: f64,
    /// The projection weakens linearly from `mls_strength` at this distance from the plane to
    /// none at `mls_taper_end` (0: off), as PointClouds Studio 2.7.0.
    pub mls_taper_start: f64,
    pub mls_taper_end: f64,
    /// Representative observed point nearest each cell centre, never a centroid. 0: all.
    pub sample_leaf: f64,
    /// After finishing, keep one point per cell of this size (the one nearest the cell centre,
    /// on the finished positions); 0: off.
    pub post_leaf: f64,
    /// Radius (m) of the disc of cell moments a query's plane is fitted on.
    pub mls_radius: f64,
    /// Share of the distance to the plane a projected point moves (along the plane normal).
    pub mls_strength: f64,
    /// Consensus inlier distance, used in support selection, not the deletion threshold.
    pub plane_band: f64,
    /// A plane must explain this share of the neighbourhood's weight.
    pub min_plane_fraction: f64,
    /// Planarity tests: smallest / middle eigenvalue and plane σ (m).
    pub max_plane_ratio: f64,
    pub max_surface_sigma: f64,
    /// The cloud is made in squares of this size (m) in xy, so that the memory follows the
    /// observations of a square and not of the data (0: all at once; see [`tiles`]). The result
    /// is the same.
    pub tile_size: f64,
    /// A square whose observations (with its margin) are more than this is split in four.
    pub tile_obs: usize,
}
/// The expected error (m) of an observation measured at this range: 1.6 mm, growing by 2.8 mm
/// per metre (the MID-360's error grows with the range). With `reject_sigma` 2.5: 4 mm or 0.7%
/// of the range.
pub fn range_sigma(range: f64) -> f64 {
    SIGMA_MIN.max(SIGMA_PER_M * range)
}
const SIGMA_MIN: f64 = 0.0016;
const SIGMA_PER_M: f64 = 0.0028;

impl Default for FinishParams {
    fn default() -> Self {
        Self {
            trim_still: 0.3,
            max_range: 40.,
            min_range: 0.75,
            support_min_range: 0.,
            moment_cell: 0.01,
            min_support: 60,
            fallback_scales: 1,
            fast_plane: true,
            consensus_seeds: 4,
            consensus_seed_radius: 0.04,
            anchor_grid: 0.04,
            // Fully isolated observations: no other within 1.5% of the range (at least 15 mm).
            support_min_obs: 2,
            support_radius: 0.015,
            support_range: 0.015,
            free_ratio: 2.,
            free_min_pass: 3,
            free_voxel: 0.02,
            free_margin: 0.06,
            free_support: true,
            free_pass_radius: 0.01,
            free_pass_far: 0.2,
            free_protect_direction_spread: 0.02,
            free_protect_direction_scans: 10,
            free_fine_voxel: 0.01,
            free_fine_ratio: 4.,
            transient_angle_deg: 0.8,
            transient_far: 0.3,
            transient_hit: 0.1,
            transient_max_hits: 20,
            transient_clean_share: 0.5,
            transient_hit_span_s: 2.,
            transient_ratio: 8.,
            transient_min_pass: 10,
            transient_max_range: 40.,
            reject_sigma: 2.5,
            mls_taper_start: 0.02,
            mls_taper_end: 0.03,
            // One point per 5 mm cell, before and after smoothing.
            sample_leaf: 0.005,
            post_leaf: 0.005,
            mls_radius: 0.08,
            mls_strength: 0.95,
            plane_band: 0.016,
            min_plane_fraction: 0.6,
            max_plane_ratio: 0.15,
            max_surface_sigma: 0.015,
            tile_size: 50.,
            tile_obs: 20_000_000,
        }
    }
}
impl FinishParams {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.free_fine_voxel.is_finite()
                && self.free_fine_voxel >= 0.
                && self.free_fine_ratio.is_finite()
                && self.free_fine_ratio >= 0.,
            "free_fine_voxel / free_fine_ratio must be nonnegative"
        );
        ensure!(
            (0. ..=5.).contains(&self.transient_angle_deg)
                && [
                    self.transient_far,
                    self.transient_hit,
                    self.transient_hit_span_s,
                    self.transient_ratio,
                    self.transient_max_range
                ]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.)
                && (0. ..=1.).contains(&self.transient_clean_share)
                && (self.transient_angle_deg == 0. || self.transient_max_range > 0.),
            "transient: angle 0 (off) to 5 degrees, the other settings nonnegative"
        );
        ensure!(
            self.free_protect_direction_spread.is_finite()
                && (0. ..=1.).contains(&self.free_protect_direction_spread)
                && (self.free_protect_direction_spread == 0.
                    || self.free_protect_direction_scans >= 4),
            "direction protection requires spread in 0..1 and at least four scans"
        );
        ensure!(
            self.consensus_seeds <= 64
                && self.consensus_seed_radius.is_finite()
                && self.consensus_seed_radius > 0.,
            "consensus seeds: at most 64, seed radius positive"
        );
        ensure!(
            self.anchor_grid.is_finite()
                && self.anchor_grid >= 0.
                && self.anchor_grid <= 0.5 * self.mls_radius,
            "anchor grid: 0 (off) up to half the MLS radius"
        );
        ensure!(
            self.sample_leaf.is_finite() && self.sample_leaf >= 0.,
            "sample leaf must be nonnegative"
        );
        for (n, x) in [
            ("moment_cell", self.moment_cell),
            ("reject_sigma", self.reject_sigma),
            ("mls_radius", self.mls_radius),
            ("plane_band", self.plane_band),
            ("max_plane_ratio", self.max_plane_ratio),
            ("max_surface_sigma", self.max_surface_sigma),
            ("max_range", self.max_range),
        ] {
            ensure!(x.is_finite() && x > 0., "{n} must be positive");
        }
        ensure!(
            [self.min_range, self.support_min_range, self.trim_still]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.),
            "min_range / support_min_range / trim_still must be nonnegative"
        );
        ensure!(self.min_support >= 6, "min_support: at least 6");
        ensure!(
            self.tile_size.is_finite() && self.tile_size >= 0. && self.tile_obs > 0,
            "tile_size >= 0, tile_obs > 0"
        );
        if self.tile_size > 0. {
            for (n, leaf) in [
                ("sample_leaf", self.sample_leaf),
                ("post_leaf", self.post_leaf),
            ] {
                ensure!(
                    leaf == 0. || ((BASE / leaf).round() * leaf - BASE).abs() < 1e-9,
                    "{n} must divide {BASE} m (the grid of the tiles)"
                );
            }
        }
        ensure!(
            self.min_plane_fraction > 0.5 && self.min_plane_fraction <= 1.,
            "plane fraction: (0.5, 1]"
        );
        ensure!(
            self.mls_strength.is_finite() && (0. ..=1.).contains(&self.mls_strength),
            "MLS strength: 0..1"
        );
        ensure!(
            self.mls_taper_end == 0.
                || (self.mls_taper_start.is_finite()
                    && self.mls_taper_start >= 0.
                    && self.mls_taper_end.is_finite()
                    && self.mls_taper_end > self.mls_taper_start),
            "MLS taper: 0 <= start < end (end 0: off)"
        );
        Ok(())
    }
}

pub(crate) fn timed_stage(
    timings: &mut BTreeMap<String, f64>,
    clock: &mut std::time::Instant,
    name: &str,
    job: &Job,
) {
    let seconds = clock.elapsed().as_secs_f64();
    timings.insert(name.into(), seconds);
    job.log(&format!("stage {name}: {seconds:.3}s"));
    *clock = std::time::Instant::now();
}

/// The finished cloud: the representative observations and their results, in the same order,
/// and which of them the passing-object test is to look at ([`transient::candidates`]; it runs
/// over the scans read again, [`transient::find`]).
pub struct Finished {
    pub input: Vec<Observation>,
    pub points: Vec<PointResult>,
    pub candidates: Vec<bool>,
    pub timings: BTreeMap<String, f64>,
}
impl Finished {
    /// The observations and results of the points in the cloud.
    pub fn kept(&self) -> impl Iterator<Item = (&Observation, &PointResult)> {
        self.input
            .iter()
            .zip(&self.points)
            .filter(|(_, r)| !removed(r.status))
    }
}

/// The grid (m) the parts of a tiled cloud are aligned to: the cells of the sampling, the
/// moments, the voxels and the shared fits lie in one part each.
pub const BASE: f64 = 0.02;

/// A rectangle in xy of [`BASE`] cells, `lo` inclusive, `hi` exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub lo: [i64; 2],
    pub hi: [i64; 2],
}
impl Rect {
    pub fn contains(&self, c: [i64; 2]) -> bool {
        (0..2).all(|d| c[d] >= self.lo[d] && c[d] < self.hi[d])
    }
    /// Grown by `m` cells on every side.
    pub fn grow(&self, m: i64) -> Rect {
        Rect {
            lo: self.lo.map(|x| x - m),
            hi: self.hi.map(|x| x + m),
        }
    }
}
/// The [`BASE`] cell in xy of the cell `k` of a grid of size `leaf` (which divides it), or of
/// position `x` when `leaf` is 0.
fn base_of(k: [i64; 2], leaf: f64, x: [f32; 3]) -> [i64; 2] {
    if leaf > 0. {
        let r = (BASE / leaf).round() as i64;
        k.map(|v| v.div_euclid(r))
    } else {
        [0, 1].map(|d| (x[d] as f64 / BASE).floor() as i64)
    }
}

/// A part of the data (see [`tiles`]): the representatives whose sampling cell lies in
/// `queries` are finished, and the points whose final post-thinning cell lies in `out` are
/// kept. `queries` covers `out` grown by the farthest a point moves, and the observations
/// given cover `queries` grown by what a query depends on.
#[derive(Clone, Copy, Debug)]
pub struct Part {
    pub queries: Rect,
    pub out: Rect,
}

/// The representative observation nearest each cell centre of size `leaf` (all when 0), among
/// the `active` ones.
fn sample_ids(
    raw: &[Observation],
    leaf: f64,
    active: Option<&[bool]>,
    area: Option<Rect>,
) -> Vec<usize> {
    if leaf <= 0. {
        return (0..raw.len())
            .filter(|&i| active.is_none_or(|a| a[i]))
            .filter(|&i| area.is_none_or(|r| r.contains(base_of([0; 2], 0., raw[i].xyz))))
            .collect();
    }
    let inv = 1. / leaf;
    let key = |p: &Observation| p.xyz.map(|v| (v as f64 * inv).floor() as i32);
    let mut ids: Vec<([i32; 3], usize)> = raw
        .par_iter()
        .enumerate()
        .filter(|(i, _)| active.is_none_or(|a| a[*i]))
        .map(|(i, p)| (key(p), i))
        .collect();
    ids.par_sort_unstable();
    let mut out = Vec::new();
    let mut a = 0;
    while a < ids.len() {
        let k = ids[a].0;
        let mut b = a + 1;
        while b < ids.len() && ids[b].0 == k {
            b += 1;
        }
        let center = k.map(|v| (v as f64 + 0.5) * leaf);
        let dist = |j: usize| {
            raw[ids[j].1]
                .xyz
                .iter()
                .zip(center)
                .map(|(&p, c)| (p as f64 - c).powi(2))
                .sum::<f64>()
        };
        if area.is_none_or(|r| r.contains(base_of([k[0] as i64, k[1] as i64], leaf, [0.; 3]))) {
            let best = (a..b)
                .min_by(|&x, &y| dist(x).total_cmp(&dist(y)).then(ids[x].1.cmp(&ids[y].1)))
                .unwrap();
            out.push(ids[best].1);
        }
        a = b;
    }
    out
}

/// The cloud from all observations `raw` (in scan order, see [`load`]).
pub fn finish(raw: &[Observation], p: &FinishParams, job: &Job) -> Result<Finished> {
    finish_part(raw, &[], None, p, job)
}

/// The cloud of `part` (all when None) from the observations `raw` (in scan order) and the
/// `rays` of observations elsewhere that may cross them (in scan order, passes only).
pub fn finish_part(
    raw: &[Observation],
    rays: &[Observation],
    part: Option<&Part>,
    p: &FinishParams,
    job: &Job,
) -> Result<Finished> {
    p.validate()?;
    job.check()?;
    let start = std::time::Instant::now();
    let mut clock = start;
    let mut timings = BTreeMap::new();
    job.report("sampling", 0, 1);
    let near: Option<Vec<bool>> =
        (p.min_range > 0.).then(|| raw.iter().map(|o| o.range as f64 >= p.min_range).collect());
    let input: Vec<Observation> =
        sample_ids(raw, p.sample_leaf, near.as_deref(), part.map(|t| t.queries))
            .into_iter()
            .map(|i| raw[i])
            .collect();
    let n = input.len();
    ensure!(n > 0 || part.is_some(), "empty input cloud");
    job.log(&format!(
        "representative sampling: {} -> {} points (leaf {:.1} mm)",
        raw.len(),
        n,
        p.sample_leaf * 1000.
    ));
    let mut out: Vec<_> = input
        .iter()
        .map(|o| PointResult {
            xyz: o.xyz,
            normal: [0.; 3],
            shift: 0.,
            residual: f32::NAN,
            sigma: f32::NAN,
            neighbors: 0,
            status: 0,
        })
        .collect();
    let mut active = vec![true; n];
    timed_stage(&mut timings, &mut clock, "sampling", job);
    if n > 0 {
        timings.extend(surface::finish(
            raw,
            rays,
            &input,
            &mut out,
            &mut active,
            p,
            job,
        )?);
    }
    job.check()?;
    clock = std::time::Instant::now();
    if p.post_leaf > 0. {
        // Second uniform sampling on the finished (projected) positions: the points of one
        // surface that the first grid kept in several cells across its noise layer collapse onto
        // it, and only the one nearest each cell centre stays.
        let inv = 1. / p.post_leaf;
        let mut cells: Vec<([i64; 3], f64, usize)> = out
            .iter()
            .enumerate()
            .filter(|(i, o)| active[*i] && !removed(o.status))
            .map(|(i, o)| {
                let c = o.xyz.map(|x| (x as f64 * inv).floor() as i64);
                let d: f64 = (0..3)
                    .map(|k| (o.xyz[k] as f64 - (c[k] as f64 + 0.5) * p.post_leaf).powi(2))
                    .sum();
                (c, d, i)
            })
            .collect();
        cells.par_sort_unstable_by(|a, b| {
            a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2))
        });
        let mut thinned = 0;
        for w in 1..cells.len() {
            if cells[w].0 == cells[w - 1].0 {
                out[cells[w].2].status = THINNED;
                thinned += 1;
            }
        }
        job.log(&format!(
            "post thinning {:.1} mm: {thinned} points left out",
            p.post_leaf * 1000.
        ));
    }
    timed_stage(&mut timings, &mut clock, "post_thinning", job);
    let (input, out) = match part {
        // the points whose final cell is the part's own
        Some(t) => {
            let inv = if p.post_leaf > 0. {
                1. / p.post_leaf
            } else {
                0.
            };
            input
                .into_iter()
                .zip(out)
                .filter(|(_, o)| {
                    let k = [0, 1].map(|d| (o.xyz[d] as f64 * inv).floor() as i64);
                    t.out.contains(base_of(k, p.post_leaf, o.xyz))
                })
                .unzip()
        }
        None => (input, out),
    };
    let n = input.len();
    let kept = out.iter().filter(|o| !removed(o.status)).count();
    let projected = out
        .iter()
        .filter(|o| matches!(o.status, 3 | 9 | 16))
        .count();
    let candidates = if p.transient_angle_deg > 0. {
        transient::candidates(raw, &out)
    } else {
        vec![false; n]
    };
    timed_stage(&mut timings, &mut clock, "transient_candidates", job);
    job.log(&format!(
        "finish: {n} -> {kept}, projected {projected}, {} to test for passing objects; {:.2}s",
        candidates.iter().filter(|&&c| c).count(),
        start.elapsed().as_secs_f64()
    ));
    Ok(Finished {
        input,
        points: out,
        candidates,
        timings,
    })
}
