//! Plane finishing from all raw observations.
//!
//! The local MLS in `geometry` fits each query to at most a few hundred representative points,
//! so the fitted plane itself wobbles by about σ/√N from point to point. Here every valid
//! observation contributes through per-cell moments (count, sum, second moment), so a plane over
//! an 8 cm disc uses thousands of measurements at the cost of a few hundred cells.
//!
//! A query far from its plane, relative to its measured range, is kept at its measured position,
//! and the projection weakens with the distance (`mls_taper_*`), out to the off-plane limit
//! (`reject_sigma`) when that is farther.
use crate::timed_stage;
use crate::{FinishParams, Observation, PointResult};
use anyhow::{Result, ensure};
use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rayon::prelude::*;
use s20_io::job::Job;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::BTreeMap, time::Instant};

/// Where the plane classification spends its time, summed over the worker threads (CPU
/// seconds, not wall time). Each thread accumulates its own copy; `collect` merges them.
/// Reported under `profile` in report.json; it never changes a result.
#[derive(Default)]
struct Profile {
    /// Per final status: queries and CPU seconds (search, fits and fallbacks included).
    status: BTreeMap<u32, (f64, f64)>,
    /// `classify_at` calls by radius tier: 0 the first radius, 1 the fallbacks.
    calls: [f64; 2],
    call_seconds: [f64; 2],
    search_seconds: f64,
    /// Cells returned by the neighbourhood searches, and a histogram of them (first radius).
    cells: f64,
    cell_hist: [f64; 8],
    fast_hits: f64,
    consensus_calls: f64,
    consensus_seconds: f64,
    support_seconds: f64,
    /// Shared fits: groups, their CPU seconds, and queries fitted alone instead because they
    /// face a shared plane from the other side.
    anchor_groups: f64,
    anchor_seconds: f64,
    anchor_alone: f64,
    /// Queries fitted alone by reason (other side, or the shared fit's status), and the CPU
    /// seconds of their classification.
    alone_by: BTreeMap<&'static str, (f64, f64)>,
}
thread_local! {
    static PROFILE: RefCell<Profile> = RefCell::new(Profile::default());
    /// Radius tier of the current `classify_at` call (see `Profile::calls`).
    static TIER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
fn profile(f: impl FnOnce(&mut Profile)) {
    PROFILE.with(|p| f(&mut p.borrow_mut()));
}
const CELL_BINS: [usize; 7] = [25, 50, 100, 200, 400, 800, 1600];
impl Profile {
    fn take_all() -> Vec<Profile> {
        rayon::broadcast(|_| PROFILE.with(|p| p.take()))
    }
    fn collect(job: &Job) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        let mut add = |k: String, x: f64| *m.entry(k).or_insert(0.) += x;
        for p in Self::take_all() {
            for (s, (n, t)) in &p.status {
                let name = crate::STATUS_NAMES.get(*s as usize).copied().unwrap_or("?");
                add(format!("status.{s:02}.{name}.n"), *n);
                add(format!("status.{s:02}.{name}.cpu_s"), *t);
            }
            for (k, tier) in ["first", "fallback"].iter().enumerate() {
                add(format!("classify_at.{tier}.n"), p.calls[k]);
                add(format!("classify_at.{tier}.cpu_s"), p.call_seconds[k]);
            }
            add("search.cpu_s".into(), p.search_seconds);
            add("search.cells".into(), p.cells);
            for (k, h) in p.cell_hist.iter().enumerate() {
                let label = match (k.checked_sub(1).map(|j| CELL_BINS[j]), CELL_BINS.get(k)) {
                    (None, Some(hi)) => format!("<{hi}"),
                    (Some(lo), Some(hi)) => format!("{lo}-{hi}"),
                    (Some(lo), None) => format!(">={lo}"),
                    (None, None) => unreachable!(),
                };
                add(format!("first_cells.{k}.{label}"), *h);
            }
            add("fast_path.hits".into(), p.fast_hits);
            add("consensus.n".into(), p.consensus_calls);
            add("consensus.cpu_s".into(), p.consensus_seconds);
            add("support_check.cpu_s".into(), p.support_seconds);
            add("anchor.groups".into(), p.anchor_groups);
            add("anchor.fit_cpu_s".into(), p.anchor_seconds);
            add("anchor.alone".into(), p.anchor_alone);
            for (why, (n, s)) in &p.alone_by {
                add(format!("anchor.alone.{why}.n"), *n);
                add(format!("anchor.alone.{why}.cpu_s"), *s);
            }
        }
        for (k, x) in &m {
            job.log(&format!("profile {k}: {x:.3}"));
        }
        m
    }
}

type V = Vector3<f64>;
fn v(p: [f32; 3]) -> V {
    V::new(p[0] as f64, p[1] as f64, p[2] as f64)
}

/// Sufficient statistics of the observations in one cell, relative to the cell centre.
#[derive(Clone)]
struct Cell {
    centre: V,
    n: f64,
    sum: V,
    // Lower triangle, column-major: 00, 10, 20, 11, 21, 22. Keeping a cell at
    // 128 bytes reduces support-cloud memory traffic without rounding its moments.
    second: [f64; 6],
    /// Mean unit viewing direction (sensor to point) of the observations.
    ray: V,
}
impl Cell {
    fn centroid(&self) -> V {
        self.centre + self.sum / self.n
    }
}
pub struct Moments {
    cells: Vec<Cell>,
    grid: CellGrid,
}

/// Cell centroids bucketed on a coarse grid for radius searches. The cells already sit on a
/// regular 1 cm grid, so a hash of coarse buckets, each a contiguous run of centroids, finds
/// a neighbourhood with a few lookups and a linear scan, where a KD-tree over tens of millions
/// of cells spends most of the plane classification in pointer chasing.
struct CellGrid {
    inv: f64,
    buckets: rustc_hash::FxHashMap<[i32; 3], (u32, u32)>,
    /// Cell ids and their centroids, ordered by bucket.
    ids: Vec<u32>,
    pts: Vec<[f64; 3]>,
}
/// Bucket edge (m): small against the 8 cm plane radius, so the scanned buckets hold few
/// cells outside the sphere, and large enough that a search touches few buckets.
const GRID_BUCKET: f64 = 0.04;
/// Debug: `S20_TRACE=<file>` and `S20_TRACE_BOX=minx,miny,minz,maxx,maxy,maxz` write one JSON line
/// per plane fit centred in the box: the PCA over all cells, the fast path, each consensus's
/// candidates and scores, and the planes fitted.
struct Trace {
    file: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
    lo: [f64; 3],
    hi: [f64; 3],
    /// Optional exact f32 query positions for targeted diagnostics.
    queries: Option<std::collections::HashSet<[u32; 3]>>,
}
fn trace() -> Option<&'static Trace> {
    static T: std::sync::OnceLock<Option<Trace>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let path = std::env::var("S20_TRACE").ok()?;
        let b: Vec<f64> = std::env::var("S20_TRACE_BOX")
            .ok()?
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        if b.len() != 6 {
            return None;
        }
        Some(Trace {
            file: std::sync::Mutex::new(std::io::BufWriter::new(std::fs::File::create(path).ok()?)),
            lo: [b[0], b[1], b[2]],
            hi: [b[3], b[4], b[5]],
            queries: std::env::var("S20_TRACE_POINTS").ok().and_then(|p| {
                let points: Vec<[f32; 3]> = serde_json::from_slice(&std::fs::read(p).ok()?).ok()?;
                Some(points.into_iter().map(|p| p.map(f32::to_bits)).collect())
            }),
        })
    })
    .as_ref()
}
fn trace_selected(q: &V) -> bool {
    trace().is_some_and(|t| {
        (0..3).all(|i| q[i] >= t.lo[i] && q[i] <= t.hi[i])
            && t.queries.as_ref().is_none_or(|points| {
                points.contains(&[q.x as f32, q.y as f32, q.z as f32].map(f32::to_bits))
            })
    })
}
fn trace_record(q: &V, value: serde_json::Value) {
    if trace_selected(q)
        && let Some(t) = trace()
    {
        use std::io::Write;
        let mut file = t.file.lock().unwrap();
        let _ = writeln!(file, "{value}");
        let _ = file.flush();
    }
}
fn trace_plane(pl: &Plane) -> serde_json::Value {
    serde_json::json!({"mu": pl.mu.as_slice(), "normal": pl.n.as_slice(),
        "sigma": pl.sigma, "ratio": pl.ratio, "tangent": pl.tangent, "count": pl.count})
}
thread_local! {
    /// The trace entries of the fit in progress on this thread (None: not traced).
    static TRACE_LOG: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}
fn trace_push(s: impl FnOnce() -> String) {
    TRACE_LOG.with(|t| {
        if let Some(v) = t.borrow_mut().as_mut() {
            v.push(s());
        }
    });
}
fn js(v: &V) -> String {
    format!("[{:.5},{:.5},{:.5}]", v.x, v.y, v.z)
}
fn js_plane(pl: &Plane) -> String {
    format!(
        "{{\"n\":{},\"mu\":{},\"sigma\":{:.5},\"ratio\":{:.4},\"count\":{}}}",
        js(&pl.n),
        js(&pl.mu),
        pl.sigma,
        pl.ratio,
        pl.count
    )
}
/// Stability check only: `S20_REVERSE_NEIGHBOURS=1` lists the neighbours in descending cell
/// order. A method whose result changes with it depends on an arbitrary order.
fn reverse_neighbours() -> bool {
    static R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *R.get_or_init(|| std::env::var("S20_REVERSE_NEIGHBOURS").is_ok_and(|v| v == "1"))
}
impl CellGrid {
    fn key(&self, p: [f64; 3]) -> [i32; 3] {
        p.map(|x| (x * self.inv).floor() as i32)
    }
    fn new(coords: &[[f64; 3]], size: f64) -> Self {
        let inv = 1. / size;
        let key = |p: &[f64; 3]| p.map(|x| (x * inv).floor() as i32);
        let mut keyed: Vec<([i32; 3], u32)> = coords
            .par_iter()
            .enumerate()
            .map(|(i, p)| (key(p), i as u32))
            .collect();
        keyed.par_sort_unstable();
        let mut buckets = rustc_hash::FxHashMap::default();
        let mut start = 0;
        for i in 1..=keyed.len() {
            if i == keyed.len() || keyed[i].0 != keyed[start].0 {
                buckets.insert(keyed[start].0, (start as u32, i as u32));
                start = i;
            }
        }
        let ids: Vec<u32> = keyed.iter().map(|k| k.1).collect();
        let pts = ids.iter().map(|&i| coords[i as usize]).collect();
        Self {
            inv,
            buckets,
            ids,
            pts,
        }
    }
    /// The sum of `weight` over the ids within `r` (inclusive, as `within`), stopping as soon as
    /// it reaches `limit`: buckets are visited in shells around the query's, so a supported
    /// query stops near itself even when `r` spans hundreds of buckets. Below `limit` the sum is
    /// complete.
    fn sum_until(&self, q: [f64; 3], r: f64, weight: impl Fn(u32) -> f64, limit: f64) -> f64 {
        let r2 = r * r;
        let c = self.key(q);
        let lo = self.key(q.map(|x| x - r - 1e-9));
        let hi = self.key(q.map(|x| x + r + 1e-9));
        let kmax = (0..3)
            .map(|i| (c[i] - lo[i]).max(hi[i] - c[i]))
            .max()
            .unwrap_or(0);
        let mut sum = 0.;
        let mut visit = |x: i32, y: i32, z: i32| -> bool {
            if x < lo[0] || x > hi[0] || y < lo[1] || y > hi[1] || z < lo[2] || z > hi[2] {
                return false;
            }
            let Some(&(s, e)) = self.buckets.get(&[x, y, z]) else {
                return false;
            };
            for j in s as usize..e as usize {
                let p = self.pts[j];
                let d = (q[0] - p[0]) * (q[0] - p[0])
                    + (q[1] - p[1]) * (q[1] - p[1])
                    + (q[2] - p[2]) * (q[2] - p[2]);
                if d <= r2 {
                    sum += weight(self.ids[j]);
                }
            }
            sum >= limit
        };
        for k in 0..=kmax {
            for dx in -k..=k {
                for dy in -k..=k {
                    let side = dx.abs() == k || dy.abs() == k;
                    let mut dz = -k;
                    while dz <= k {
                        if visit(c[0] + dx, c[1] + dy, c[2] + dz) {
                            return sum;
                        }
                        // Inside the shell's x/y range only its top and bottom are on the shell.
                        dz += if side || k == 0 { 1 } else { 2 * k };
                    }
                }
            }
        }
        sum
    }
    /// Ids within `r` (inclusive, squared distance summed axis by axis like kiddo's
    /// `SquaredEuclidean`), unsorted.
    fn within(&self, q: [f64; 3], r: f64, out: &mut Vec<u32>) {
        let r2 = r * r;
        // A hair of margin so rounding in q ± r never drops a bucket that holds a match.
        let lo = self.key(q.map(|x| x - r - 1e-9));
        let hi = self.key(q.map(|x| x + r + 1e-9));
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    let Some(&(s, e)) = self.buckets.get(&[x, y, z]) else {
                        continue;
                    };
                    for j in s as usize..e as usize {
                        let p = self.pts[j];
                        let d = (q[0] - p[0]) * (q[0] - p[0])
                            + (q[1] - p[1]) * (q[1] - p[1])
                            + (q[2] - p[2]) * (q[2] - p[2]);
                        if d <= r2 {
                            out.push(self.ids[j]);
                        }
                    }
                }
            }
        }
    }
}
impl Moments {
    pub fn new(raw: &[Observation], cell: f64, min_range: f64, job: &Job) -> Result<Self> {
        ensure!(cell > 0., "moment cell must be positive");
        job.report("moment grid", 0, 1);
        let inv = 1. / cell;
        let mut keyed: Vec<([i32; 3], usize)> = raw
            .par_iter()
            .enumerate()
            .filter(|(_, o)| o.range as f64 >= min_range)
            .map(|(i, o)| (o.xyz.map(|x| (x as f64 * inv).floor() as i32), i))
            .collect();
        keyed.par_sort_unstable();
        let mut bounds = vec![0];
        for i in 1..keyed.len() {
            if keyed[i].0 != keyed[i - 1].0 {
                bounds.push(i);
            }
        }
        bounds.push(keyed.len());
        let cells: Vec<Cell> = bounds
            .par_windows(2)
            .map(|w| {
                let k = keyed[w[0]].0;
                let centre = V::new(
                    (k[0] as f64 + 0.5) * cell,
                    (k[1] as f64 + 0.5) * cell,
                    (k[2] as f64 + 0.5) * cell,
                );
                let mut c = Cell {
                    centre,
                    n: 0.,
                    sum: V::zeros(),
                    second: [0.; 6],
                    ray: V::zeros(),
                };
                for &(_, i) in &keyed[w[0]..w[1]] {
                    let d = v(raw[i].xyz) - centre;
                    c.n += 1.;
                    c.sum += d;
                    for (s, (i, j)) in
                        c.second
                            .iter_mut()
                            .zip([(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)])
                    {
                        *s += d[i] * d[j];
                    }
                    c.ray += (v(raw[i].xyz) - v(raw[i].origin)).normalize();
                }
                c.ray /= c.n;
                c
            })
            .collect();
        let coords: Vec<[f64; 3]> = cells.iter().map(|c| c.centroid().into()).collect();
        job.log(&format!(
            "moment grid: {} observations in {} cells of {:.1} mm ({} bytes/cell)",
            keyed.len(),
            cells.len(),
            cell * 1000.,
            std::mem::size_of::<Cell>()
        ));
        Ok(Self {
            grid: CellGrid::new(&coords, GRID_BUCKET),
            cells,
        })
    }
    /// The cells whose centroid lies within `r` of `q`, in cell order: the fits that follow sum
    /// and pick candidates by position in this list, so the order must not depend on the search.
    fn within(&self, q: &V, r: f64) -> Vec<u32> {
        let mut ids = Vec::new();
        self.grid.within((*q).into(), r, &mut ids);
        ids.sort_unstable();
        if reverse_neighbours() {
            ids.reverse();
        }
        ids
    }
}

#[derive(Clone)]
struct Plane {
    mu: V,
    n: V,
    /// Thickness of all member observations (raw measurement noise plus scan offsets).
    sigma: f64,
    ratio: f64,
    tangent: f64,
    /// Number of observations behind the fit.
    count: f64,
}
/// Weighted PCA over whole cells: exact second moments of every observation, not of centroids.
fn plane(cells: &[&Cell], w: &[f64]) -> Option<Plane> {
    let mut sw = 0.;
    let mut s = V::zeros();
    let mut count = 0.;
    for (c, &w) in cells.iter().zip(w) {
        if w <= 0. {
            continue;
        }
        sw += w * c.n;
        s += w * (c.centre * c.n + c.sum);
        count += c.n;
    }
    if sw < 1e-9 {
        return None;
    }
    let mu = s / sw;
    let mut cov = Matrix3::zeros();
    for (c, &w) in cells.iter().zip(w) {
        if w <= 0. {
            continue;
        }
        let a = c.centre - mu;
        // SymmetricEigen reads the lower triangle. Preserve the arithmetic order of each
        // entry, but avoid computing the unused upper triangle on every robust refit.
        for (&second, (i, j)) in
            c.second
                .iter()
                .zip([(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)])
        {
            cov[(i, j)] += w * (second + c.sum[i] * a[j] + a[i] * c.sum[j] + (c.n * a[i]) * a[j]);
        }
    }
    cov /= sw;
    let eig = SymmetricEigen::new(cov);
    let mut ix = [0, 1, 2];
    ix.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let small = eig.eigenvalues[ix[0]].max(0.);
    let mid = eig.eigenvalues[ix[1]].max(0.);
    Some(Plane {
        mu,
        n: eig.eigenvectors.column(ix[0]).into_owned(),
        sigma: small.sqrt(),
        ratio: small / mid.max(1e-12),
        tangent: mid,
        count,
    })
}
/// A cell seen from the other side of a candidate plane than the query belongs to the opposite
/// face of a thin object (door, board, towel): it never supports the query's plane.
fn facing(n: &V, ray: &V, rq: &V) -> bool {
    let (a, b) = (n.dot(ray), n.dot(rq));
    a * b >= 0. || a.abs() < 0.15 || b.abs() < 0.15
}

/// Robust weighted refit on the member cells: a cell whose centroid leaves the band loses weight.
fn refine(cells: &[&Cell], cent: &[V], spatial: &[f64], mask: &[bool], band: f64) -> Option<Plane> {
    let w: Vec<f64> = mask
        .iter()
        .zip(spatial)
        .map(|(&m, &s)| if m { s } else { 0. })
        .collect();
    let mut pl = plane(cells, &w)?;
    for _ in 0..3 {
        let scale = band * 0.5;
        let w: Vec<f64> = cent
            .iter()
            .zip(mask)
            .zip(spatial)
            .map(|((p, &m), &s)| {
                if !m {
                    return 0.;
                }
                let e = pl.n.dot(&(p - pl.mu)).abs();
                s * (scale / e.max(scale)).powi(2)
            })
            .collect();
        pl = plane(cells, &w)?;
    }
    Some(pl)
}
/// Candidate planes (normal, point) for the consensus, besides the PCA over all the cells.
/// `seeds` 0: 32 cell triples picked by position in the list (the result depends on the list
/// order). Otherwise the local plane within `seed_radius` of each seed cell: the first seed is
/// the cell nearest the query (its own surface), the next ones are farthest-point sampled
/// (the other faces of a corner or a thin object), ties to the lower cell id.
fn candidates(cent: &[V], cells: &[&Cell], q: &V, seeds: usize, seed_radius: f64) -> Vec<(V, V)> {
    let len = cent.len();
    if seeds == 0 {
        return (1..33usize)
            .filter_map(|c| {
                let a = (c * 7 + 1) % len;
                let b = (c * 17 + len / 3) % len;
                let d = (c * 29 + 2 * len / 3) % len;
                let cross = (cent[b] - cent[a]).cross(&(cent[d] - cent[a]));
                (cross.norm_squared() >= 1e-14).then(|| (cross.normalize(), cent[a]))
            })
            .collect();
    }
    let argmax = |d: &[f64]| (0..len).fold(0, |best, i| if d[i] > d[best] { i } else { best });
    let mut dist: Vec<f64> = cent.iter().map(|c| -(c - q).norm_squared()).collect();
    let mut seed = argmax(&dist); // nearest the query
    dist.iter_mut().for_each(|d| *d = f64::INFINITY);
    let r2 = seed_radius * seed_radius;
    let mut out = Vec::with_capacity(seeds);
    for _ in 0..seeds.min(len) {
        let s = cent[seed];
        let near: Vec<f64> = cent
            .iter()
            .map(|c| if (c - s).norm_squared() <= r2 { 1. } else { 0. })
            .collect();
        // Refit on the cells seen from the seed's side of the first fit: the other face of a
        // thin object within the radius would otherwise pull the plane between the faces.
        let seed_ray = cells[seed].ray;
        if near.iter().filter(|&&x| x > 0.).count() >= 3
            && let Some(first) = plane(cells, &near)
            && let w = near
                .iter()
                .zip(cells)
                .map(|(&x, c)| {
                    if x > 0. && facing(&first.n, &c.ray, &seed_ray) {
                        1.
                    } else {
                        0.
                    }
                })
                .collect::<Vec<f64>>()
            && w.iter().filter(|&&x| x > 0.).count() >= 3
            && let Some(pl) = plane(cells, &w)
            && pl.tangent > 1e-7
        {
            out.push((pl.n, pl.mu));
        }
        for (d, c) in dist.iter_mut().zip(cent) {
            *d = d.min((c - s).norm_squared());
        }
        seed = argmax(&dist);
        if dist[seed] <= 0. {
            break;
        }
    }
    out
}

/// Deterministic consensus over cell centroids, scored by spatially weighted observation count.
fn consensus(
    cells: &[&Cell],
    spatial: &[f64],
    band: f64,
    rq: &V,
    q: &V,
    p: &FinishParams,
) -> Option<(Plane, Vec<bool>)> {
    if cells.len() < 3 {
        return None;
    }
    let initial = plane(cells, spatial)?;
    let cent: Vec<V> = cells.iter().map(|c| c.centroid()).collect();
    let len = cells.len();
    let member = |n: &V, mu: &V, i: usize| {
        n.dot(&(cent[i] - mu)).abs() <= band && facing(n, &cells[i].ray, rq)
    };
    let mut best = (f64::NEG_INFINITY, initial.n, initial.mu);
    let tries = std::iter::once((initial.n, initial.mu)).chain(candidates(
        &cent,
        cells,
        q,
        p.consensus_seeds,
        p.consensus_seed_radius,
    ));
    for (n, mu) in tries {
        let score: f64 = (0..len)
            .filter(|&i| member(&n, &mu, i))
            .map(|i| cells[i].n * spatial[i])
            .sum();
        trace_push(|| {
            format!(
                "{{\"cand\":{},\"mu\":{},\"score\":{:.3}}}",
                js(&n),
                js(&mu),
                score
            )
        });
        if score > best.0 {
            best = (score, n, mu);
        }
    }
    let (_, n, mu) = best;
    let mask: Vec<bool> = (0..len).map(|i| member(&n, &mu, i)).collect();
    let pl = refine(cells, &cent, spatial, &mask, band)?;
    let mask: Vec<bool> = (0..len).map(|i| member(&pl.n, &pl.mu, i)).collect();
    trace_push(|| {
        format!(
            "{{\"consensus\":{},\"cells\":{},\"best\":{}}}",
            js_plane(&pl),
            len,
            js(&n)
        )
    });
    Some((pl, mask))
}

pub struct Outcome {
    pub status: u32,
    pub plane: Option<(V, V, f64)>,
    pub count: u32,
    /// The plane came from the fast path: one plane over the neighbourhood.
    pub clean: bool,
}
/// Status codes shared with `geometry::STATUS_NAMES` (codes of removed stages stay unused so
/// saved results keep their meaning).
pub const PROJECTED: u32 = 3;
pub const SURFACE_REJECT: u32 = 2;
pub const SPARSE: u32 = 5;
pub const NONPLANAR: u32 = 6;
pub const SMALL_PROJECTED: u32 = 9;
pub const UNSUPPORTED: u32 = 11;
pub const FREE: u32 = 12;
pub const OUTLIER_KEPT: u32 = 16;

/// Large radius first (robust membership against walls and floors); a query that is off every
/// large plane or has none is tried again at half and quarter radius, where curved objects
/// (towels, pipes, fixtures) are locally planar. Only a reliable small plane overrides.
/// `first`: the outcome at the full radius when it is already known (a shared fit).
fn classify_with(
    m: &Moments,
    q: V,
    rq: &V,
    range: f64,
    p: &FinishParams,
    first: Option<Outcome>,
) -> Outcome {
    let first = first.unwrap_or_else(|| {
        TIER.set(0);
        classify_at(m, q, rq, range, p, p.mls_radius)
    });
    if first.status != SURFACE_REJECT && first.status != NONPLANAR {
        return first;
    }
    let mut r = p.mls_radius;
    for _ in 0..p.fallback_scales {
        r *= 0.5;
        if r < 2.5 * p.moment_cell {
            break;
        }
        TIER.set(1);
        let o = classify_at(m, q, rq, range, p, r);
        if o.status == PROJECTED {
            return Outcome {
                status: SMALL_PROJECTED,
                ..o
            };
        }
    }
    first
}

fn classify_at(m: &Moments, q: V, rq: &V, range: f64, p: &FinishParams, r: f64) -> Outcome {
    let t = Instant::now();
    let tier = TIER.get();
    let o = decide(&fit_at(m, q, rq, p, r, tier), q, range, p);
    let s = t.elapsed().as_secs_f64();
    profile(|f| {
        f.calls[tier] += 1.;
        f.call_seconds[tier] += s;
    });
    o
}

/// The plane of a neighbourhood, before a query is tested against it (see [`decide`]).
#[derive(Clone)]
enum Fit {
    /// No plane: `SPARSE` or `NONPLANAR`.
    None { status: u32, count: u32 },
    /// The plane; `clean` when it came from the fast path.
    Plane {
        plane: Plane,
        count: u32,
        clean: bool,
    },
}
impl Fit {
    fn planes(&self) -> &[Plane] {
        match self {
            Fit::None { .. } => &[],
            Fit::Plane { plane, .. } => std::slice::from_ref(plane),
        }
    }
}

/// The query's status and plane: everything else depends on the neighbourhood only, so a fit
/// can be shared by the queries near its centre seen from the same side.
fn decide(fit: &Fit, q: V, range: f64, p: &FinishParams) -> Outcome {
    match fit {
        Fit::None { status, count } => Outcome {
            status: *status,
            plane: None,
            count: *count,
            clean: false,
        },
        Fit::Plane {
            plane: pl,
            count,
            clean,
        } => {
            let reliable = pl.ratio <= p.max_plane_ratio
                && pl.sigma <= p.max_surface_sigma
                && pl.tangent > 1e-7
                && pl.count >= p.min_support as f64;
            if !reliable {
                return Outcome {
                    status: NONPLANAR,
                    plane: None,
                    count: *count,
                    clean: false,
                };
            }
            let limit = off_plane(pl.sigma, range, p);
            Outcome {
                status: if pl.n.dot(&(q - pl.mu)).abs() > limit {
                    SURFACE_REJECT
                } else {
                    PROJECTED
                },
                plane: Some((pl.mu, pl.n, pl.sigma)),
                count: *count,
                clean: *clean,
            }
        }
    }
}

/// The planes within `r` of `q` (the query, or the centre of the queries sharing the fit),
/// seen from the side of `rq`.
fn fit_at(m: &Moments, q: V, rq: &V, p: &FinishParams, r: f64, tier: usize) -> Fit {
    let traced = trace_selected(&q);
    if !traced {
        return fit_at_inner(m, q, rq, p, r, tier);
    }
    TRACE_LOG.with(|t| *t.borrow_mut() = Some(Vec::new()));
    let fit = fit_at_inner(m, q, rq, p, r, tier);
    let entries = TRACE_LOG
        .with(|t| t.borrow_mut().take())
        .unwrap_or_default();
    let planes: Vec<String> = fit.planes().iter().map(js_plane).collect();
    let kind = match &fit {
        Fit::None { status, .. } => format!("none{status}"),
        Fit::Plane { .. } => "plane".into(),
    };
    if let Some(t) = trace() {
        use std::io::Write;
        let _ = writeln!(
            t.file.lock().unwrap(),
            "{{\"q\":{},\"rq\":{},\"r\":{},\"tier\":{},\"fit\":\"{}\",\"planes\":[{}],\"log\":[{}]}}",
            js(&q),
            js(rq),
            r,
            tier,
            kind,
            planes.join(","),
            entries.join(",")
        );
        let _ = t.file.lock().unwrap().flush();
    }
    fit
}

fn fit_at_inner(m: &Moments, q: V, rq: &V, p: &FinishParams, r: f64, tier: usize) -> Fit {
    let t = Instant::now();
    let hits = m.within(&q, r);
    let cells: Vec<&Cell> = hits.iter().map(|&i| &m.cells[i as usize]).collect();
    let s = t.elapsed().as_secs_f64();
    profile(|f| {
        f.search_seconds += s;
        f.cells += cells.len() as f64;
        if tier == 0 {
            f.cell_hist[CELL_BINS.iter().take_while(|&&b| cells.len() >= b).count()] += 1.;
        }
    });
    let total: f64 = cells.iter().map(|c| c.n).sum();
    let count = total as u32;
    if total < p.min_support as f64 || cells.len() < 3 {
        return Fit::None {
            status: SPARSE,
            count,
        };
    }
    let spatial: Vec<f64> = cells
        .iter()
        .map(|c| (-(c.centroid() - q).norm_squared() / (0.5 * r * r)).exp())
        .collect();
    let initial = p.fast_plane.then(|| plane(&cells, &spatial)).flatten();
    let weight = |mask: &[bool]| -> f64 {
        cells
            .iter()
            .zip(mask)
            .zip(&spatial)
            .filter(|((_, m), _)| **m)
            .map(|((c, _), s)| c.n * s)
            .sum()
    };
    let all = weight(&vec![true; cells.len()]);
    // Fast path for the common case, a plain wall or floor: one plane over all the cells already
    // holds nearly all the weight within the band, seen from the query's side. The consensus
    // could not do better, and the rest is too light for a second plane (15% below).
    let clean = |pl: &Plane, cent: &[V]| -> Vec<bool> {
        (0..cells.len())
            .map(|i| {
                pl.n.dot(&(cent[i] - pl.mu)).abs() <= p.plane_band
                    && facing(&pl.n, &cells[i].ray, rq)
            })
            .collect()
    };
    let fast = p
        .fast_plane
        .then(|| -> Option<(Plane, Vec<bool>)> {
            let pca = initial.as_ref()?;
            if pca.ratio > 0.5 * p.max_plane_ratio {
                return None;
            }
            let cent: Vec<V> = cells.iter().map(|c| c.centroid()).collect();
            let mask = clean(pca, &cent);
            if weight(&mask) < 0.97 * all {
                return None;
            }
            let pl = refine(&cells, &cent, &spatial, &mask, p.plane_band)?;
            let mask = clean(&pl, &cent);
            (weight(&mask) >= 0.97 * all).then_some((pl, mask))
        })
        .flatten();
    if fast.is_some() {
        profile(|f| f.fast_hits += 1.);
    }
    trace_push(|| {
        format!(
            "{{\"cells\":{},\"initial\":{},\"fast\":{}}}",
            cells.len(),
            initial.as_ref().map_or("null".into(), js_plane),
            fast.as_ref().map_or("null".into(), |f| js_plane(&f.0))
        )
    });
    let clean = fast.is_some();
    let Some((p1, m1)) = fast.or_else(|| {
        let t = Instant::now();
        let c = consensus(&cells, &spatial, p.plane_band, rq, &q, p);
        let s = t.elapsed().as_secs_f64();
        profile(|f| {
            f.consensus_calls += 1.;
            f.consensus_seconds += s;
        });
        c
    }) else {
        return Fit::None {
            status: NONPLANAR,
            count,
        };
    };
    if weight(&m1) < p.min_plane_fraction * all {
        return Fit::None {
            status: NONPLANAR,
            count,
        };
    }
    Fit::Plane {
        plane: p1,
        count,
        clean,
    }
}

/// `rays`: observations elsewhere whose rays may cross `raw`'s voxels (free space only).
pub fn finish(
    raw: &[Observation],
    rays: &[Observation],
    input: &[Observation],
    out: &mut [PointResult],
    active: &mut [bool],
    p: &FinishParams,
    job: &Job,
) -> Result<BTreeMap<String, f64>> {
    let prepared = prepare(raw, rays, p, job)?;
    let mut timings = prepared.timings.clone();
    timings.extend(prepared.finish(input, out, active, p, job)?);
    Ok(timings)
}

/// Seen through by other scans: in the free-space count, or in the finer one.
fn is_free(
    free: &Option<crate::freespace::FreeSpace>,
    fine: &Option<crate::freespace::FreeSpace>,
    o: &Observation,
    p: &FinishParams,
) -> bool {
    let test = |f: &crate::freespace::FreeSpace, ratio: f64| {
        let (hit, pass) = f.at(o.xyz);
        pass >= p.free_min_pass && pass as f64 > ratio * hit as f64
    };
    free.as_ref().is_some_and(|f| test(f, p.free_ratio))
        || fine.as_ref().is_some_and(|f| test(f, p.free_fine_ratio))
}

/// Immutable evidence shared by independent query/parameter comparisons.
pub struct Prepared {
    moments: Moments,
    free: Option<crate::freespace::FreeSpace>,
    /// The finer count (`free_fine_voxel`).
    fine: Option<crate::freespace::FreeSpace>,
    params: FinishParams,
    pub timings: BTreeMap<String, f64>,
    pub raw_count: usize,
    pub support_count: usize,
    pub cell_count: usize,
}

pub fn prepare(
    raw: &[Observation],
    rays: &[Observation],
    p: &FinishParams,
    job: &Job,
) -> Result<Prepared> {
    let mut timings = BTreeMap::new();
    let mut clock = Instant::now();
    // Free space first: points in voxels other scans saw through are neither output nor support.
    let free = (p.free_ratio > 0.).then(|| {
        crate::freespace::FreeSpace::new(raw, rays, &crate::freespace::Options::of(p), job)
    });
    if let (Some(f), Ok(dir)) = (&free, std::env::var("S20_FREE_DUMP")) {
        f.dump(
            raw,
            &crate::freespace::Options::of(p),
            p.free_ratio,
            p.free_min_pass,
            std::path::Path::new(&dir),
        )?;
        job.log(&format!("free space dumped to {dir}"));
    }
    timed_stage(&mut timings, &mut clock, "free_space", job);
    // A second count on finer voxels: a thin object and the mixed returns around it fall into
    // separate voxels, so the latter are left out while the object stays.
    let fine = (p.free_fine_voxel > 0. && p.free_fine_ratio > 0.).then(|| {
        crate::freespace::FreeSpace::new(raw, rays, &crate::freespace::Options::fine(p), job)
    });
    let is_free = |o: &Observation| is_free(&free, &fine, o, p);
    timed_stage(&mut timings, &mut clock, "free_space_fine", job);
    let kept: Vec<Observation>;
    let support = if free.is_some() && p.free_support {
        kept = raw.par_iter().filter(|o| !is_free(o)).copied().collect();
        job.log(&format!(
            "free space: {} of {} observations left out of the planes",
            raw.len() - kept.len(),
            raw.len()
        ));
        &kept[..]
    } else {
        raw
    };
    timed_stage(&mut timings, &mut clock, "support_filter", job);
    let m = Moments::new(support, p.moment_cell, p.support_min_range, job)?;
    timed_stage(&mut timings, &mut clock, "moment_grid_and_tree", job);
    Ok(Prepared {
        raw_count: raw.len(),
        support_count: support.len(),
        cell_count: m.cells.len(),
        moments: m,
        free,
        fine,
        params: p.clone(),
        timings,
    })
}

impl Prepared {
    pub fn finish(
        &self,
        input: &[Observation],
        out: &mut [PointResult],
        active: &mut [bool],
        p: &FinishParams,
        job: &Job,
    ) -> Result<BTreeMap<String, f64>> {
        let mut allowed = self.params.clone();
        allowed.consensus_seeds = p.consensus_seeds;
        allowed.consensus_seed_radius = p.consensus_seed_radius;
        allowed.anchor_grid = p.anchor_grid;
        allowed.fallback_scales = p.fallback_scales;
        allowed.post_leaf = p.post_leaf;
        ensure!(
            allowed == *p,
            "shared evidence permits only query fitting and post-leaf changes"
        );
        ensure!(
            input.len() == out.len() && out.len() == active.len(),
            "query array counts differ"
        );
        let mut timings = BTreeMap::new();
        let mut clock = Instant::now();
        let is_free = |o: &Observation| is_free(&self.free, &self.fine, o, p);
        let n = input.len();
        // Leftovers of an earlier run on the same threads.
        Profile::take_all();
        let pending: Vec<bool> = out
            .par_iter_mut()
            .enumerate()
            .map(|(i, o)| active[i] && self.precheck(&input[i], o, p, &is_free))
            .collect();
        let shared = (p.anchor_grid > 0.)
            .then(|| self.shared_fits(input, &pending, p, job))
            .transpose()?;
        let done = AtomicUsize::new(0);
        out.par_iter_mut()
            .enumerate()
            .try_for_each(|(i, o)| -> Result<()> {
                let k = done.fetch_add(1, Ordering::Relaxed);
                if k.is_multiple_of(4096) {
                    job.check()?;
                    job.report("moment planes", k as u64, n as u64);
                }
                if !pending[i] {
                    return Ok(());
                }
                let obs = &input[i];
                let q = v(obs.xyz);
                let rq = (q - v(obs.origin)).normalize();
                let t = Instant::now();
                let mut why: Option<&'static str> = None;
                let first = shared.as_ref().and_then(|(group, fits)| {
                    let (fit, rep) = &fits[group[i] as usize];
                    // Seen from the other side of a shared plane: its own fit. Off the shared
                    // plane or without one: the smaller-radius retries (see `anchor_grid`).
                    let side = fit.planes().iter().all(|pl| facing(&pl.n, &rq, rep));
                    if !side {
                        why = Some("side");
                        profile(|f| f.anchor_alone += 1.);
                    }
                    side.then(|| decide(fit, q, obs.range as f64, p))
                });
                let t_alone = Instant::now();
                let c = classify_with(&self.moments, q, &rq, obs.range as f64, p, first);
                if trace_selected(&q) {
                    let shared_planes = shared.as_ref().map(|(group, fits)| {
                        fits[group[i] as usize]
                            .0
                            .planes()
                            .iter()
                            .map(trace_plane)
                            .collect::<Vec<_>>()
                    });
                    trace_record(
                        &q,
                        serde_json::json!({"kind": "query_outcome", "q": q.as_slice(),
                        "scan": obs.scan, "time": obs.time, "shared_fallback_reason": why,
                        "shared_planes": shared_planes, "status": c.status, "count": c.count,
                        "plane": c.plane.map(|(mu, normal, sigma)| serde_json::json!({
                            "mu": mu.as_slice(), "normal": normal.as_slice(), "sigma": sigma}))}),
                    );
                }
                if let Some(w) = why {
                    let s = t_alone.elapsed().as_secs_f64();
                    profile(|f| {
                        let e = f.alone_by.entry(w).or_default();
                        e.0 += 1.;
                        e.1 += s;
                    });
                }
                apply(obs, q, o, &c, p);
                let s = t.elapsed().as_secs_f64();
                profile(|f| {
                    let e = f.status.entry(o.status).or_default();
                    e.0 += 1.;
                    e.1 += s;
                });
                Ok(())
            })?;
        job.report("moment planes", n as u64, n as u64);
        timed_stage(&mut timings, &mut clock, "plane_classification", job);
        for (k, x) in Profile::collect(job) {
            timings.insert(format!("profile.{k}"), x);
        }
        clock = Instant::now();
        for (a, o) in active.iter_mut().zip(out.iter()) {
            *a = *a && !matches!(o.status, SURFACE_REJECT | UNSUPPORTED | FREE);
        }
        timed_stage(&mut timings, &mut clock, "active_update", job);
        Ok(timings)
    }
    /// The checks before the planes: free space and support. False when they settle the
    /// query's status.
    fn precheck(
        &self,
        obs: &Observation,
        o: &mut PointResult,
        p: &FinishParams,
        is_free: &impl Fn(&Observation) -> bool,
    ) -> bool {
        let m = &self.moments;
        let q = v(obs.xyz);
        if is_free(obs) {
            o.status = FREE;
            return false;
        }
        // Support: a place measured only a few times (a stray return, a mixed pixel, a
        // glimpse through a gap) is not output, whatever plane it might fit.
        if p.support_min_obs > 0 {
            let t = Instant::now();
            let r = p.support_radius.max(p.support_range * obs.range as f64);
            let n = m.grid.sum_until(
                q.into(),
                r,
                |i| m.cells[i as usize].n,
                p.support_min_obs as f64,
            );
            let s = t.elapsed().as_secs_f64();
            profile(|f| f.support_seconds += s);
            if n < p.support_min_obs as f64 {
                o.status = UNSUPPORTED;
                o.neighbors = n as u32;
                return false;
            }
        }
        true
    }

    /// The full-radius fits shared by the pending queries of each `anchor_grid` cell and
    /// viewing octant, at the mean of their positions and rays: (group of each query, and
    /// per group the fit and its mean ray).
    fn shared_fits(
        &self,
        input: &[Observation],
        pending: &[bool],
        p: &FinishParams,
        job: &Job,
    ) -> Result<SharedFits> {
        let inv = 1. / p.anchor_grid;
        let mut keyed: Vec<(([i64; 3], u8), u32)> = (0..input.len())
            .into_par_iter()
            .filter(|&i| pending[i])
            .map(|i| {
                let o = &input[i];
                let ray = (v(o.xyz) - v(o.origin)).normalize();
                let octant = u8::from(ray.x >= 0.)
                    | (u8::from(ray.y >= 0.) << 1)
                    | (u8::from(ray.z >= 0.) << 2);
                (
                    (o.xyz.map(|x| (x as f64 * inv).floor() as i64), octant),
                    i as u32,
                )
            })
            .collect();
        keyed.par_sort_unstable();
        let mut bounds = vec![0];
        for i in 1..keyed.len() {
            if keyed[i].0 != keyed[i - 1].0 {
                bounds.push(i);
            }
        }
        bounds.push(keyed.len());
        let groups = bounds.len() - 1;
        let done = AtomicUsize::new(0);
        let fits: Vec<(Fit, V)> = bounds
            .par_windows(2)
            .map(|w| -> Result<(Fit, V)> {
                let k = done.fetch_add(1, Ordering::Relaxed);
                if k.is_multiple_of(4096) {
                    job.check()?;
                    job.report("shared planes", k as u64, groups as u64);
                }
                let t = Instant::now();
                let (mut centre, mut ray) = (V::zeros(), V::zeros());
                for &(_, i) in &keyed[w[0]..w[1]] {
                    let o = &input[i as usize];
                    centre += v(o.xyz);
                    ray += (v(o.xyz) - v(o.origin)).normalize();
                }
                centre /= (w[1] - w[0]) as f64;
                let ray = ray.normalize();
                let fit = fit_at(&self.moments, centre, &ray, p, p.mls_radius, 0);
                let s = t.elapsed().as_secs_f64();
                profile(|f| {
                    f.anchor_groups += 1.;
                    f.anchor_seconds += s;
                });
                Ok((fit, ray))
            })
            .collect::<Result<_>>()?;
        job.report("shared planes", groups as u64, groups as u64);
        let mut group = vec![u32::MAX; input.len()];
        for (g, w) in bounds.windows(2).enumerate() {
            for &(_, i) in &keyed[w[0]..w[1]] {
                group[i as usize] = g as u32;
            }
        }
        job.log(&format!(
            "shared planes: {} queries in {} groups of {:.0} mm",
            keyed.len(),
            groups,
            p.anchor_grid * 1000.
        ));
        Ok((group, fits))
    }
}

/// The group of each query (`u32::MAX`: none), and per group the shared fit and its mean ray.
type SharedFits = (Vec<u32>, Vec<(Fit, V)>);

/// Beyond this distance from its plane a query is off it (`reject_sigma`).
fn off_plane(sigma: f64, range: f64, p: &FinishParams) -> f64 {
    p.reject_sigma * sigma.max(crate::range_sigma(range))
}

/// Share of `mls_strength` applied at this distance from the plane (`mls_taper_start`/`_end`).
/// The taper ends no nearer than `noise` (the start scaled with it).
fn taper(distance: f64, noise: f64, p: &FinishParams) -> f64 {
    if p.mls_taper_end <= 0. {
        return 1.;
    }
    let end = p.mls_taper_end.max(noise);
    let start = p.mls_taper_start * end / p.mls_taper_end;
    if distance <= start {
        1.
    } else {
        ((end - distance) / (end - start)).clamp(0., 1.)
    }
}

/// The query's classification written out, and its projection onto its plane.
fn apply(obs: &Observation, q: V, o: &mut PointResult, c: &Outcome, p: &FinishParams) {
    o.neighbors = c.count;
    o.status = if c.status == SURFACE_REJECT {
        OUTLIER_KEPT
    } else {
        c.status
    };
    if let Some((mu, nrm, sigma)) = c.plane {
        let signed = nrm.dot(&(q - mu));
        // Normals face the sensor that measured the point.
        let toward = v(obs.origin) - q;
        let nrm = if nrm.dot(&toward) < 0. { -nrm } else { nrm };
        o.normal = nrm.map(|x| x as f32).into();
        o.residual = signed.abs() as f32;
        o.sigma = sigma as f32;
        if c.status == PROJECTED || c.status == SMALL_PROJECTED || o.status == OUTLIER_KEPT {
            // Within the off-plane limit: the error of a return, growing with its range.
            let strength =
                p.mls_strength * taper(signed.abs(), off_plane(sigma, obs.range as f64, p), p);
            let shift = signed * strength;
            let moved = q - nrm * nrm.dot(&(q - mu)) * strength;
            o.xyz = moved.map(|x| x as f32).into();
            o.shift = shift.abs() as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CellGrid;
    use crate::Observation;
    use crate::{FinishParams, finish};
    use s20_io::job::Job;

    #[test]
    fn taper_weakens_projection_linearly_and_end_zero_turns_it_off() {
        let mut p = FinishParams {
            mls_taper_end: 0.,
            ..FinishParams::default()
        };
        assert_eq!(super::taper(0.5, 0., &p), 1.);
        p.mls_taper_start = 0.01;
        p.mls_taper_end = 0.03;
        assert_eq!(super::taper(0.005, 0., &p), 1.);
        assert!((super::taper(0.02, 0., &p) - 0.5).abs() < 1e-12);
        assert_eq!(super::taper(0.04, 0., &p), 0.);
        // Out to 60 mm (a return from afar): from 20 mm to 60 mm.
        assert!((super::taper(0.04, 0.06, &p) - 0.5).abs() < 1e-12);
    }

    /// Deterministic pseudo-random noise in [-1, 1].
    fn noise(i: u64) -> f32 {
        let x = i
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 33) as f64 / (1u64 << 31) as f64 * 2. - 1.) as f32
    }
    fn obs(p: [f32; 3], origin: [f32; 3]) -> Observation {
        let d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
        Observation {
            time: 0.,
            xyz: p,
            origin,
            range: (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt(),
            intensity: 100,
            scan: 0,
        }
    }
    /// Planes only: no trims, free space, isolation test, taper or post thinning; full
    /// projection.
    fn params() -> FinishParams {
        FinishParams {
            trim_still: 0.,
            min_range: 0.,
            free_ratio: 0.,
            free_protect_direction_spread: 0.,
            support_min_obs: 0,
            tile_size: 0.,
            mls_taper_start: 0.,
            mls_taper_end: 0.,
            sample_leaf: 0.008,
            post_leaf: 0.,
            mls_strength: 1.,
            plane_band: 0.012,
            ..Default::default()
        }
    }
    /// Range noise of 5 mm (uniform ±8.7 mm) along the viewing direction, dense like raw data.
    fn noisy(points: &[[f32; 3]], origin: [f32; 3], seed: u64) -> Vec<Observation> {
        points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
                let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let e = noise(seed + i as u64) * 0.0087;
                obs(
                    [
                        p[0] + d[0] / l * e,
                        p[1] + d[1] / l * e,
                        p[2] + d[2] / l * e,
                    ],
                    origin,
                )
            })
            .collect()
    }

    #[test]
    fn dense_plane_is_thin_after_projection() {
        for anchor_grid in ANCHORS {
            dense_plane(anchor_grid);
        }
    }
    /// Every query fitted alone, and planes shared on a 2 cm grid.
    const ANCHORS: [f64; 2] = [0., 0.02];
    fn dense_plane(anchor_grid: f64) {
        let mut pts = vec![];
        for i in 0..300 {
            for j in 0..300 {
                pts.push([i as f32 * 0.002 - 0.3, j as f32 * 0.002 - 0.3, 0.]);
            }
        }
        let raw = noisy(&pts, [0.1, 0.2, 1.5], 1);
        let p = FinishParams {
            anchor_grid,
            ..params()
        };
        let r = finish(&raw, &p, &Job::stderr()).unwrap();
        let mut z: Vec<f32> = r
            .points
            .iter()
            .zip(r.input.iter())
            .filter(|(p, o)| p.status == 3 && o.xyz[0].abs() < 0.2 && o.xyz[1].abs() < 0.2)
            .map(|(p, _)| p.xyz[2].abs())
            .collect();
        assert!(z.len() > 1000);
        z.sort_by(f32::total_cmp);
        let q90 = z[z.len() * 9 / 10];
        // Raw noise is 5 mm RMS; a plane from thousands of observations is far thinner.
        assert!(q90 < 0.0006, "anchor {anchor_grid}: q90 {q90}");
    }

    /// Without a second plane at an edge, the corner rounds a little: one face's 8 cm plane
    /// rejects the other face's points near the line, and the 4 / 2 cm retries lean between the
    /// faces (2026-10-21: 3.9% of the points within 10 cm beyond 2 mm, at most 6.4 mm).
    #[test]
    fn corner_rounds_by_a_few_millimetres_at_most() {
        for anchor_grid in ANCHORS {
            corner(anchor_grid);
        }
    }
    fn corner(anchor_grid: f64) {
        // Floor z=0 (x>0) and wall x=0 (z>0), seen from inside the corner.
        let mut pts = vec![];
        for i in 0..150 {
            for j in 0..300 {
                let a = i as f32 * 0.002;
                let y = j as f32 * 0.002 - 0.3;
                pts.push([a, y, 0.]);
                pts.push([0., y, a]);
            }
        }
        let raw = noisy(&pts, [0.8, 0., 0.8], 7);
        let p = FinishParams {
            anchor_grid,
            ..params()
        };
        let r = finish(&raw, &p, &Job::stderr()).unwrap();
        // Distance of the projected points near the corner to the union of the two true planes.
        let d: Vec<f32> = r
            .points
            .iter()
            .zip(r.input.iter())
            .filter(|(p, o)| {
                o.xyz[1].abs() <= 0.15
                    && o.xyz[0] <= 0.1
                    && o.xyz[2] <= 0.1
                    && (p.status == 3 || p.status == 9)
            })
            .map(|(p, _)| p.xyz[0].abs().min(p.xyz[2].abs()))
            .collect();
        assert!(d.len() > 300);
        let off = d.iter().filter(|&&x| x > 0.002).count() as f32 / d.len() as f32;
        let worst = d.iter().copied().fold(0f32, f32::max);
        assert!(
            off < 0.06 && worst < 0.008,
            "anchor {anchor_grid}: {:.1}% beyond 2 mm, worst {worst}",
            off * 100.
        );
    }

    #[test]
    fn thin_board_faces_are_not_merged() {
        for anchor_grid in ANCHORS {
            thin_board(anchor_grid);
        }
    }
    fn thin_board(anchor_grid: f64) {
        // Two parallel faces 25 mm apart, each seen from its own side.
        let mut a = vec![];
        let mut b = vec![];
        for i in 0..200 {
            for j in 0..200 {
                let (x, y) = (i as f32 * 0.002 - 0.2, j as f32 * 0.002 - 0.2);
                a.push([x, y, 0.]);
                b.push([x, y, 0.025]);
            }
        }
        let mut raw = noisy(&a, [0., 0., -1.], 3);
        raw.extend(noisy(&b, [0., 0., 1.], 5));
        let r = finish(
            &raw,
            &FinishParams {
                anchor_grid,
                ..params()
            },
            &Job::stderr(),
        )
        .unwrap();
        let moved: Vec<f32> = r
            .points
            .iter()
            .filter(|p| p.status == 3 || p.status == 9)
            .map(|p| p.xyz[2].abs().min((p.xyz[2] - 0.025).abs()))
            .collect();
        assert!(moved.len() > 1000);
        let bad = moved.iter().filter(|&&d| d > 0.002).count();
        eprintln!(
            "thin board, anchor {anchor_grid}: {bad} of {} between faces",
            moved.len()
        );
        assert!(
            bad * 100 < moved.len(),
            "{bad} of {} between faces",
            moved.len()
        );
    }

    /// The bucket grid returns exactly the brute-force set (inclusive radius), also for radii
    /// larger than a bucket and queries on bucket borders.
    #[test]
    fn cell_grid_matches_brute_force() {
        let pts: Vec<[f64; 3]> = (0..4000u64)
            .map(|i| {
                [0, 1, 2].map(|k| 0.3 * noise(3 * i + k) as f64 + if k == 2 { 0.04 } else { 0. })
            })
            .collect();
        let grid = CellGrid::new(&pts, 0.04);
        let mut queries: Vec<[f64; 3]> = pts.iter().step_by(37).copied().collect();
        queries.extend([[0.04, -0.08, 0.0], [0., 0., 0.], [-0.12, 0.16, 0.04]]);
        for q in queries {
            for r in [0.015, 0.02, 0.04, 0.08, 0.13, 0.31] {
                let mut got = Vec::new();
                grid.within(q, r, &mut got);
                got.sort_unstable();
                let want: Vec<u32> = (0..pts.len() as u32)
                    .filter(|&i| {
                        let p = pts[i as usize];
                        (q[0] - p[0]) * (q[0] - p[0])
                            + (q[1] - p[1]) * (q[1] - p[1])
                            + (q[2] - p[2]) * (q[2] - p[2])
                            <= r * r
                    })
                    .collect();
                assert_eq!(got, want, "q {q:?} r {r}");
            }
        }
    }

    /// `sum_until` stops only at or above the limit, and below it equals the full sum.
    #[test]
    fn cell_grid_sum_until_matches_brute_force() {
        let pts: Vec<[f64; 3]> = (0..3000u64)
            .map(|i| [0, 1, 2].map(|k| 0.4 * noise(5 * i + k) as f64))
            .collect();
        let grid = CellGrid::new(&pts, 0.04);
        for q in pts
            .iter()
            .step_by(53)
            .copied()
            .chain([[0.5, 0.5, 0.5], [0.02, -0.07, 0.11]])
        {
            for r in [0.01, 0.03, 0.09, 0.3] {
                let full = (0..pts.len())
                    .filter(|&i| {
                        let p = pts[i];
                        (q[0] - p[0]) * (q[0] - p[0])
                            + (q[1] - p[1]) * (q[1] - p[1])
                            + (q[2] - p[2]) * (q[2] - p[2])
                            <= r * r
                    })
                    .count() as f64;
                for limit in [1., 2., 5., 1e9] {
                    let s = grid.sum_until(q, r, |_| 1., limit);
                    if full >= limit {
                        assert!(
                            s >= limit && s <= full,
                            "q {q:?} r {r} limit {limit}: {s} of {full}"
                        );
                    } else {
                        assert_eq!(s, full, "q {q:?} r {r} limit {limit}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_close_floater_stays_and_far_returns_off_a_wall_are_projected() {
        let mut pts = vec![];
        for i in 0..200 {
            for j in 0..200 {
                pts.push([i as f32 * 0.002 - 0.2, j as f32 * 0.002 - 0.2, 0.]);
            }
        }
        let mut raw = noisy(&pts, [0., 0., 1.], 11);
        let idx = raw.len();
        // 4 cm in front of the wall, measured from 1 m: off the plane, left where it is.
        raw.push(obs([0.01, 0.013, 0.04], [0., 0., 1.]));
        // 4 cm behind it, measured from 15 m: within the error of a return from afar.
        raw.push(obs([-0.03, 0.021, -0.04], [0., 0., 15.]));
        // 4 cm in front of it, measured from 15 m: the same.
        raw.push(obs([0.03, -0.021, 0.04], [0., 0., 15.]));
        let p = FinishParams {
            sample_leaf: 0.,
            mls_taper_start: 0.01,
            mls_taper_end: 0.03,
            ..params()
        };
        let r = finish(&raw, &p, &Job::stderr()).unwrap();
        assert_eq!(r.points[idx].status, super::OUTLIER_KEPT);
        assert!(
            (r.points[idx].xyz[2] - 0.04).abs() < 1e-6,
            "{:?}",
            r.points[idx]
        );
        assert_eq!(r.points[idx + 1].status, super::PROJECTED);
        assert!(
            r.points[idx + 1].xyz[2].abs() < 0.005,
            "{:?}",
            r.points[idx + 1]
        );
        assert!(
            r.points[idx + 2].xyz[2].abs() < 0.005,
            "{:?}",
            r.points[idx + 2]
        );
    }
}
