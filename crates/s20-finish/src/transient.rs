//! Passing objects (a car driving by, a person walking past) from the viewing directions. Ported
//! from the author's earlier prototype (slam-experiment), checked there on six recordings.
//!
//! The voxel free space ([`crate::freespace`]) counts the rays through a point's own 2 cm voxel; at
//! 20 m the MID-360's rays are several cm apart, so a car's trail gets one or two passes there, no
//! more than the bushes beside it. Here a scan sees through a point when one of its rays within
//! `angle` of the sensor-to-point direction ends `far` beyond the point and none ends short of it,
//! and sees the point when one ends within `hit` of its distance: the tolerance grows with the
//! range like the spacing of the rays. A ray of the cone ending short of the point means a surface
//! the cone grazes: on ground seen at a few degrees the rays just above a point go on for metres
//! and those just below land before it, every scan; around a car the rays go on. A place few
//! scans measured (a car is in 1-3 scans), all within a moment, and many more saw through was there
//! only briefly; bushes, also seen through, are measured by dozens of scans, and the tips of the
//! grass by a few scans minutes apart, and stay.
//!
//! The rays are every return of the scan (also beyond `max_range`, which the cloud does not read)
//! and the shots without a return ([`crate::noreturn`]) at least [`OPEN_MIN_ELEVATION_DEG`] above
//! the horizon, each passing a point when a returned shot bounding its gap reached beyond it: a
//! trail with only the sky behind it is seen through too. Below the horizon a shot without a
//! return is dark or wet ground, water or a surface seen at a glancing angle.
//!
//! The test runs after the cloud, over the scans read again from the bag (see [`find`]), on the
//! kept sparse and nonplanar points whose 20 cm voxel few scans measured ([`candidates`]). The
//! points it takes leave the cloud as status [`crate::TRANSIENT`]. On the GPU when there is one
//! (`gpu_transient.rs`; S20_NO_GPU set: on the CPU), the scans' rays sorted by cell on the CPU.
use crate::{FinishParams, Observation, PointResult, load, noreturn};
use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet};
use s20_geom::so3::V3;
use s20_geom::traj::Traj;
use s20_io::dataset::LidarData;
use s20_io::job::Job;
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

/// Shots without a return below this elevation (deg) are not counted.
pub const OPEN_MIN_ELEVATION_DEG: f64 = 2.;
/// A shot without a return is placed as the point this far (m) along its ray.
const FAR: f64 = 100.;
/// Only points in 20 cm voxels measured by at most this many scans are tested: a passing car
/// fills a voxel for a few scans, the bushes and walls for dozens. It saves testing the 40
/// million points of a park against every scan; the cone test decides.
const PRE_VOXEL: f64 = 0.2;
const PRE_SCANS: u32 = 16;

/// The settings of the test (see [`FinishParams::transient_angle_deg`]).
pub struct Options {
    /// Half angle (rad) of the cone of rays a scan looks along.
    pub angle: f64,
    pub far: f64,
    pub hit: f64,
    pub max_hits: u32,
    /// The scans seeing through cleanly (no ray short) are at least this share of those with a
    /// ray beyond.
    pub clean_share: f64,
    /// The scans that measured the point lie within this many seconds.
    pub hit_span: f64,
    pub ratio: f64,
    pub min_pass: u32,
    pub max_range: f64,
}
impl Options {
    pub fn of(p: &FinishParams) -> Self {
        Self {
            angle: p.transient_angle_deg.to_radians(),
            far: p.transient_far,
            hit: p.transient_hit,
            max_hits: p.transient_max_hits,
            clean_share: p.transient_clean_share,
            hit_span: p.transient_hit_span_s,
            ratio: p.transient_ratio,
            min_pass: p.transient_min_pass,
            max_range: p.transient_max_range,
        }
    }
}

fn key(p: [f32; 3]) -> u64 {
    let c = p.map(|x| ((x as f64 / PRE_VOXEL).floor() as i64 + (1 << 20)) as u64 & 0x1f_ffff);
    (c[0] << 42) | (c[1] << 21) | c[2]
}

/// The points of `out` to test: kept, sparse or nonplanar, in a 20 cm voxel at most `PRE_SCANS`
/// scans of `raw` (in scan order) measured. In a square of a tiled cloud the voxels of its
/// points are whole in its observations, so the choice is the same as all at once.
pub fn candidates(raw: &[Observation], out: &[PointResult]) -> Vec<bool> {
    let kind =
        |r: &PointResult| matches!(r.status, crate::surface::SPARSE | crate::surface::NONPLANAR);
    let wanted: FxHashSet<u64> = out.iter().filter(|r| kind(r)).map(|r| key(r.xyz)).collect();
    if wanted.is_empty() {
        return vec![false; out.len()];
    }
    let mut bounds = vec![0];
    for i in 1..raw.len() {
        if raw[i].scan != raw[i - 1].scan {
            bounds.push(i);
        }
    }
    bounds.push(raw.len());
    let mut keys: Vec<u64> = bounds
        .par_windows(2)
        .flat_map_iter(|w| {
            let mut k: Vec<u64> = raw[w[0]..w[1]]
                .iter()
                .map(|o| key(o.xyz))
                .filter(|k| wanted.contains(k))
                .collect();
            k.sort_unstable();
            k.dedup();
            k
        })
        .collect();
    keys.par_sort_unstable();
    let mut scans: HashMap<u64, u32> =
        HashMap::with_capacity_and_hasher(wanted.len(), Default::default());
    for k in &keys {
        *scans.entry(*k).or_insert(0) += 1;
    }
    out.iter()
        .map(|r| kind(r) && scans.get(&key(r.xyz)).is_some_and(|&n| n <= PRE_SCANS))
        .collect()
}

/// The rays of one scan by direction: a latitude-longitude grid of cells no smaller than the
/// cone, so a cone spans the neighbouring rows and a few columns. A ray's length is negative for
/// a shot without a return: minus the reach of the returned shots around it.
struct Image {
    rows: usize,
    cols: usize,
    start: Vec<u32>,
    rays: Vec<([f32; 3], f32)>,
    keyed: Vec<(u32, [f32; 3], f32)>,
}
impl Image {
    fn new(angle: f64) -> Self {
        let rows = ((PI / angle).floor() as usize).max(1);
        let cols = ((TAU / angle).floor() as usize).max(1);
        Self {
            rows,
            cols,
            start: vec![0; rows * cols + 1],
            rays: Vec::new(),
            keyed: Vec::new(),
        }
    }
    fn row(&self, z: f64) -> usize {
        (((z.clamp(-1., 1.).asin() + FRAC_PI_2) / PI * self.rows as f64) as usize)
            .min(self.rows - 1)
    }
    fn col(&self, x: f64, y: f64) -> usize {
        (((y.atan2(x) + PI) / TAU * self.cols as f64) as usize).min(self.cols - 1)
    }
    /// The rays from `origin` to the returns `hits` (their ranges) and along the shots without a
    /// return `open` (a point along the ray, the reach).
    fn fill(&mut self, origin: [f64; 3], hits: &[[f32; 3]], open: &[([f32; 3], f32)]) {
        let up = OPEN_MIN_ELEVATION_DEG.to_radians().sin();
        let (rows, cols) = (self.rows, self.cols);
        let cell = |u: [f64; 3]| {
            let row = (((u[2].clamp(-1., 1.).asin() + FRAC_PI_2) / PI * rows as f64) as usize)
                .min(rows - 1);
            let col = (((u[1].atan2(u[0]) + PI) / TAU * cols as f64) as usize).min(cols - 1);
            (row * cols + col) as u32
        };
        let ray = |p: [f32; 3]| {
            let d = [0, 1, 2].map(|k| p[k] as f64 - origin[k]);
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            (len > 1e-6).then(|| (d.map(|x| x / len), len))
        };
        self.keyed.clear();
        for &p in hits {
            if let Some((u, len)) = ray(p) {
                self.keyed.push((cell(u), u.map(|x| x as f32), len as f32));
            }
        }
        for &(p, reach) in open {
            if let Some((u, _)) = ray(p)
                && u[2] >= up
            {
                self.keyed.push((cell(u), u.map(|x| x as f32), -reach));
            }
        }
        self.keyed.sort_unstable_by_key(|x| x.0);
        self.start.fill(0);
        for x in &self.keyed {
            self.start[x.0 as usize + 1] += 1;
        }
        for i in 1..self.start.len() {
            self.start[i] += self.start[i - 1];
        }
        self.rays.clear();
        self.rays.extend(self.keyed.iter().map(|x| (x.1, x.2)));
    }
    /// (a ray of the cone around `u` ends beyond `r + far`, one ends short of `r - hit`, one ends
    /// within `hit` of `r`).
    fn look(&self, u: [f64; 3], r: f64, o: &Options) -> (bool, bool, bool) {
        let row = self.row(u[2]);
        let (r0, r1) = (row.saturating_sub(1), (row + 1).min(self.rows - 1));
        // Columns narrow towards the poles: widen by the cosine of the band's edge nearest one.
        let lat = |i: usize| i as f64 / self.rows as f64 * PI - FRAC_PI_2;
        let c = lat(r0).abs().max(lat(r1 + 1).abs()).cos();
        let width = TAU / self.cols as f64;
        let dc = if c < 1e-3 {
            self.cols
        } else {
            (o.angle / (width * c)).ceil() as usize
        };
        let col = self.col(u[0], u[1]);
        let all = 2 * dc + 1 >= self.cols;
        let cos = o.angle.cos() as f32;
        let q = u.map(|x| x as f32);
        let (mut beyond, mut short, mut hit) = (false, false, false);
        for i in r0..=r1 {
            for k in 0..if all { self.cols } else { 2 * dc + 1 } {
                let j = if all {
                    k
                } else {
                    (col + self.cols + k - dc) % self.cols
                };
                let cell = i * self.cols + j;
                for &(v, len) in
                    &self.rays[self.start[cell] as usize..self.start[cell + 1] as usize]
                {
                    if v[0] * q[0] + v[1] * q[1] + v[2] * q[2] < cos {
                        continue;
                    }
                    let len = len as f64;
                    if len < 0. {
                        // a shot without a return: nothing up to its gap's reach
                        beyond |= -len > r + o.far;
                        continue;
                    }
                    beyond |= len > r + o.far;
                    short |= len <= r - o.hit;
                    hit |= (len - r).abs() < o.hit;
                    if beyond && short && hit {
                        return (true, true, true);
                    }
                }
            }
        }
        (beyond, short, hit)
    }
}

/// The rays of scan `i` placed with `tr` (see [`load`]): every return the sensor did not tag,
/// nearer than `load::NEAR` left out, and the shots without a return with their reach. None when
/// the scan lies outside the trajectory.
#[allow(clippy::type_complexity)]
fn scan_rays(
    data: &LidarData,
    tr: &Traj,
    lever: V3,
    i: usize,
) -> Option<([f64; 3], Vec<[f32; 3]>, Vec<([f32; 3], f32)>)> {
    let sc = data.scan(i);
    let (start, end) = (
        sc.t.iter().copied().reduce(f64::min)?,
        sc.t.iter().copied().reduce(f64::max)?,
    );
    if start < tr.start() || end > tr.end() {
        return None;
    }
    let (r0, p0) = tr.at(start);
    let o = r0 * lever + p0;
    let mut hits = Vec::with_capacity(sc.t.len());
    for k in 0..sc.t.len() {
        let x = V3::from(sc.xyz[k]);
        let range = x.norm();
        if !range.is_finite() || range < load::NEAR || sc.tag[k] & 0x3f != 0 {
            continue;
        }
        let (r, p) = tr.at(sc.t[k]);
        let w = r * (x + lever) + p;
        hits.push([w.x as f32, w.y as f32, w.z as f32]);
    }
    let open = noreturn::directions(&sc.xyz, &sc.line)
        .into_iter()
        .map(|(k, u, reach)| {
            let (r, p) = tr.at(sc.t[k]);
            let w = r * lever + p + r * V3::from(u) * FAR;
            ([w.x as f32, w.y as f32, w.z as f32], reach as f32)
        })
        .collect();
    Some(([o.x, o.y, o.z], hits, open))
}

/// Whether a point with these counts (scans with a ray beyond it, seeing cleanly through, measuring
/// it, and the seconds between the first and last of those) held a passing object.
fn passing(o: &Options, beyond: u32, clean: u32, hits: u32, span: f64) -> bool {
    hits <= o.max_hits
        && span <= o.hit_span
        && clean >= o.min_pass
        && clean as f64 > o.ratio * hits as f64
        && clean as f64 >= o.clean_share * beyond as f64
}

/// Which of `points` held a passing object: per scan within `max_range` (the scans `trim_still`
/// leaves out excepted), does it see through the point, does it see the point; transient when at
/// most `max_hits` scans saw it, within `hit_span` seconds, and at least `min_pass`, more than
/// `ratio` × the hits and `clean_share` of those with a ray beyond saw cleanly through it.
pub fn find(
    data: &LidarData,
    tr: &Traj,
    lever: V3,
    p: &FinishParams,
    points: &[[f32; 3]],
    job: &Job,
) -> Vec<bool> {
    let o = Options::of(p);
    if points.is_empty() || o.angle <= 0. {
        return vec![false; points.len()];
    }
    if let Some(g) = crate::gpu::shared().filter(|_| std::env::var_os("S20_NO_GPU").is_none()) {
        match find_gpu(g, data, tr, lever, p, points, job) {
            Ok(t) => return t,
            Err(e) => job.log(&format!("transient on the GPU failed, on the CPU: {e:#}")),
        }
    }
    find_cpu(data, tr, lever, p, points, job)
}

/// [`find`] on the GPU: batches of scans, each with its rays sorted by cell and the ranges of the
/// points near it (in the order of their cells); the counts stay on the GPU until the end.
fn find_gpu(
    g: &crate::gpu::Gpu,
    data: &LidarData,
    tr: &Traj,
    lever: V3,
    p: &FinishParams,
    points: &[[f32; 3]],
    job: &Job,
) -> anyhow::Result<Vec<bool>> {
    use crate::gpu_transient::{Batch, Params, Scan};
    /// A batch holds at most this many scans and pairs.
    const SCANS: usize = 64;
    const PAIRS: u64 = 1 << 25;
    let o = Options::of(p);
    let n = points.len();
    let left_out = load::set_down_scans(data, tr, p.trim_still);
    let g_cell = o.max_range / 4.;
    let cell = |p: [f64; 3]| p.map(|x| (x / g_cell).floor() as i64);
    let mut order: Vec<u32> = (0..n as u32).collect();
    let keys: Vec<[i64; 3]> = points.iter().map(|x| cell(x.map(|v| v as f64))).collect();
    order.par_sort_unstable_by_key(|&s| (keys[s as usize], s));
    let mut cells: HashMap<[i64; 3], (u32, u32)> = HashMap::default();
    for (k, &s) in order.iter().enumerate() {
        let e = cells.entry(keys[s as usize]).or_insert((k as u32, 0));
        e.1 += 1;
    }
    drop(keys);
    let reach = (o.max_range / g_cell).ceil() as i64;
    let near = |origin: [f64; 3]| -> Vec<(u32, u32)> {
        let c = cell(origin);
        let mut v = Vec::new();
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    if let Some(&r) = cells.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                        v.push(r);
                    }
                }
            }
        }
        v
    };
    let t0 = data.scan_start(0);
    let times: Vec<f32> = (0..data.len())
        .map(|i| (data.scan_start(i) - t0) as f32)
        .collect();
    let probe = Image::new(o.angle);
    let params = Params {
        cos_angle: o.angle.cos() as f32,
        angle: o.angle as f32,
        far: o.far as f32,
        hit: o.hit as f32,
        max_range: o.max_range as f32,
        hit_span: o.hit_span as f32,
        rows: probe.rows as u32,
        cols: probe.cols as u32,
        max_hits: o.max_hits,
        ranges: 0,
        pairs: 0,
        width: 0,
    };
    drop(probe);
    let total = data.len() as u64;
    job.report("transient scans", 0, total);
    let mut next_scan = 0usize;
    // fills `b` with the next scans (none only at the end); returns whether scans are left
    let mut fill = |b: &mut Batch| {
        b.scans.clear();
        b.starts.clear();
        b.rays.clear();
        b.ranges.clear();
        b.pairs = 0;
        // the scans of the batch: up to SCANS, and PAIRS pairs by the ranges near their start
        let mut chosen: Vec<(usize, Vec<(u32, u32)>)> = Vec::new();
        let mut pairs = 0u64;
        while next_scan < data.len() && chosen.len() < SCANS {
            let w = next_scan;
            next_scan += 1;
            if left_out[w] {
                continue;
            }
            let t = data.scan_start(w);
            if t < tr.start() || t > tr.end() {
                continue;
            }
            let (r, pp) = tr.at(t);
            let o3 = r * lever + pp;
            let ranges = near([o3.x, o3.y, o3.z]);
            let k: u64 = ranges.iter().map(|r| r.1 as u64).sum();
            if !chosen.is_empty() && pairs + k > PAIRS {
                next_scan -= 1;
                break;
            }
            pairs += k;
            chosen.push((w, ranges));
        }
        // their rays, sorted by cell, in parallel: the scan's origin, where each cell's rays start
        // and the rays (direction and length)
        type Rays = ([f64; 3], Vec<u32>, Vec<[f32; 4]>);
        let images: Vec<Option<Rays>> = chosen
            .par_iter()
            .map_init(
                || Image::new(o.angle),
                |image, (w, _)| {
                    let (origin, hits, open) = scan_rays(data, tr, lever, *w)?;
                    image.fill(origin, &hits, &open);
                    Some((
                        origin,
                        image.start.clone(),
                        image
                            .rays
                            .iter()
                            .map(|(v, l)| [v[0], v[1], v[2], *l])
                            .collect(),
                    ))
                },
            )
            .collect();
        for ((w, ranges), img) in chosen.into_iter().zip(images) {
            let Some((origin, starts, rays)) = img else {
                continue;
            };
            let k = b.scans.len() as u32;
            b.scans.push(Scan {
                origin: [origin[0] as f32, origin[1] as f32, origin[2] as f32, 0.],
                index: w as u32,
                rays: b.rays.len() as u32,
                starts: b.starts.len() as u32,
                pad: 0,
            });
            b.starts.extend(starts);
            b.rays.extend(rays);
            for (first, count) in ranges {
                b.ranges.push([first, count, k, b.pairs]);
                b.pairs += count;
            }
        }
        job.report("transient scans", next_scan as u64, total);
        next_scan < data.len()
    };
    let next = |b: &mut Batch| -> bool {
        loop {
            let left = fill(b);
            if !b.scans.is_empty() || !left {
                return left;
            }
        }
    };
    let counts =
        crate::gpu_transient::counts(g, points, &order, &times, params, next, || job.check())?;
    job.report("transient scans", total, total);
    Ok((0..n)
        .map(|s| {
            let c = &counts[6 * s..6 * s + 6];
            let span = if c[2] == 0 {
                0.
            } else {
                times[c[4] as usize] as f64 - times[c[3] as usize] as f64
            };
            passing(&o, c[0], c[1], c[2], span)
        })
        .collect())
}

/// [`find`] on the CPU.
fn find_cpu(
    data: &LidarData,
    tr: &Traj,
    lever: V3,
    p: &FinishParams,
    points: &[[f32; 3]],
    job: &Job,
) -> Vec<bool> {
    let o = Options::of(p);
    let n = points.len();
    let left_out = load::set_down_scans(data, tr, p.trim_still);
    // The points in cells of a quarter of the range, to find those near each scan.
    let g = o.max_range / 4.;
    let cell = |p: [f64; 3]| p.map(|x| (x / g).floor() as i64);
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::default();
    for (s, x) in points.iter().enumerate() {
        grid.entry(cell(x.map(|v| v as f64)))
            .or_default()
            .push(s as u32);
    }
    let atomic = |v: u32| (0..n).map(|_| AtomicU32::new(v)).collect::<Vec<_>>();
    let (beyond, clean, hit) = (atomic(0), atomic(0), atomic(0));
    // The first and last scan that measured each point.
    let (first, last) = (atomic(u32::MAX), atomic(0));
    // Points no later scan can make transient: more than `max_hits` hits, or hits further apart
    // than `hit_span` (both only grow). Most points are static and settled within their first few
    // dozen hits; the scans after skip them (the result is the same).
    let settled: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
    let time = |w: u32| data.scan_start(w as usize);
    let total = data.len() as u64;
    job.report("transient scans", 0, total);
    let completed = AtomicUsize::new(0);
    let reach = (o.max_range / g).ceil() as i64;
    (0..data.len()).into_par_iter().for_each_init(
        || Image::new(o.angle),
        |image, w| {
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            job.report("transient scans", done as u64, total);
            if left_out[w] {
                return;
            }
            let Some((origin, hits, open)) = scan_rays(data, tr, lever, w) else {
                return;
            };
            image.fill(origin, &hits, &open);
            let c = cell(origin);
            for dx in -reach..=reach {
                for dy in -reach..=reach {
                    for dz in -reach..=reach {
                        let Some(ids) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                            continue;
                        };
                        for &s in ids {
                            let s = s as usize;
                            if settled[s].load(Ordering::Relaxed) {
                                continue;
                            }
                            let d = [0, 1, 2].map(|k| points[s][k] as f64 - origin[k]);
                            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                            if r > o.max_range || r < 1e-3 {
                                continue;
                            }
                            let (b, short, h) = image.look(d.map(|x| x / r), r, &o);
                            if b {
                                beyond[s].fetch_add(1, Ordering::Relaxed);
                                if !short {
                                    clean[s].fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            if h {
                                let k = hit[s].fetch_add(1, Ordering::Relaxed) + 1;
                                let a = first[s]
                                    .fetch_min(w as u32, Ordering::Relaxed)
                                    .min(w as u32);
                                let z =
                                    last[s].fetch_max(w as u32, Ordering::Relaxed).max(w as u32);
                                if k > o.max_hits || time(z) - time(a) > o.hit_span {
                                    settled[s].store(true, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
            }
        },
    );
    (0..n)
        .map(|s| {
            let (b, c, h) = (
                beyond[s].load(Ordering::Relaxed),
                clean[s].load(Ordering::Relaxed),
                hit[s].load(Ordering::Relaxed),
            );
            let span = if h == 0 {
                0.
            } else {
                time(last[s].load(Ordering::Relaxed)) - time(first[s].load(Ordering::Relaxed))
            };
            passing(&o, b, c, h, span)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt() -> Options {
        Options {
            angle: 0.8f64.to_radians(),
            far: 0.3,
            hit: 0.1,
            max_hits: 5,
            clean_share: 0.5,
            hit_span: 2.,
            ratio: 8.,
            min_pass: 10,
            max_range: 40.,
        }
    }
    fn unit(p: [f64; 3]) -> [f64; 3] {
        let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        p.map(|x| x / l)
    }

    #[test]
    fn the_cone_sees_through_beyond_and_marks_short_and_hit() {
        let mut image = Image::new(opt().angle);
        // a car at 10 m: a ray to the wall at 20 m beside it, one to the ground at 8 m, one to the car
        image.fill(
            [0.; 3],
            &[[20., 0.1, 0.05], [8., 0.05, 0.], [10., 0.02, 0.]],
            &[],
        );
        assert_eq!(
            image.look(unit([1., 0., 0.]), 10., &opt()),
            (true, true, true)
        );
        image.fill([0.; 3], &[[20., 0.1, 0.05]], &[]);
        assert_eq!(
            image.look(unit([1., 0., 0.]), 10., &opt()),
            (true, false, false)
        );
    }

    #[test]
    fn shots_without_a_return_pass_up_to_their_reach_and_only_upwards() {
        let mut image = Image::new(opt().angle);
        // up 11 deg: passes a point at 10 m with a reach of 30 m, not with 8 m
        image.fill([0.; 3], &[], &[([100., 0.5, 20.], 30.)]);
        assert_eq!(
            image.look(unit([10., 0., 2.]), 10., &opt()),
            (true, false, false)
        );
        image.fill([0.; 3], &[], &[([100., 0.5, 20.], 8.)]);
        assert_eq!(
            image.look(unit([10., 0., 2.]), 10., &opt()),
            (false, false, false)
        );
        // below the horizon: not counted
        image.fill([0.; 3], &[], &[([100., 0.5, -20.], 30.)]);
        assert_eq!(
            image.look(unit([10., 0., -2.]), 10., &opt()),
            (false, false, false)
        );
    }

    #[test]
    fn the_cone_reaches_across_the_longitude_seam_and_near_the_pole() {
        let mut image = Image::new(opt().angle);
        image.fill(
            [0.; 3],
            &[
                [-20., 0.05, 0.],
                [-20., -0.05, 0.],
                [0.05, 0.02, 20.],
                [-0.05, -0.02, 20.],
            ],
            &[],
        );
        assert_eq!(
            image.look(unit([-1., 0.001, 0.]), 10., &opt()),
            (true, false, false)
        );
        assert_eq!(
            image.look(unit([-1., -0.001, 0.]), 10., &opt()),
            (true, false, false)
        );
        assert_eq!(
            image.look(unit([0.001, 0., 1.]), 10., &opt()),
            (true, false, false)
        );
    }

    #[test]
    fn candidates_are_sparse_points_few_scans_measured() {
        let obs = |x: [f32; 3], scan: u32| Observation {
            time: scan as f64,
            xyz: x,
            origin: [0.; 3],
            range: 1.,
            scan,
            intensity: 0,
        };
        let res = |x: [f32; 3], status: u32| PointResult {
            xyz: x,
            normal: [0.; 3],
            shift: 0.,
            residual: f32::NAN,
            sigma: f32::NAN,
            neighbors: 0,
            status,
        };
        // a point seen by 3 scans and one by 30, both sparse; a projected one seen by 3
        let (a, b, c) = (
            [1.05f32, 0.05, 0.05],
            [3.05f32, 0.05, 0.05],
            [5.05f32, 0.05, 0.05],
        );
        let mut raw = Vec::new();
        for s in 0..30u32 {
            if s < 3 {
                raw.push(obs(a, s));
                raw.push(obs(c, s));
            }
            raw.push(obs(b, s));
        }
        let out = [
            res(a, crate::surface::SPARSE),
            res(b, crate::surface::SPARSE),
            res(c, crate::surface::PROJECTED),
        ];
        assert_eq!(candidates(&raw, &out), vec![true, false, false]);
    }
}
