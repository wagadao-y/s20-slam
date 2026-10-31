//! Free-space evidence from the laser rays: a voxel that many scans saw through, compared with
//! the scans that measured a point in it, holds something that was not there for most of the
//! recording (a person, a mirror image, a stray return), not a surface.
//!
//! Each scan's rays are walked from the sensor to `margin` before the measured point in steps of
//! half a voxel; a scan counts once per voxel, both for passing and for hitting. Only voxels that
//! hold measurements are tracked.
use crate::Observation;
use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet};
use s20_io::job::Job;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The settings of one free-space count (see `FinishParams`).
pub struct Options {
    pub voxel: f64,
    pub margin: f64,
    pub pass_radius: f64,
    pub pass_far: f64,
    pub protect_direction_spread: f64,
    pub protect_direction_scans: u32,
}
impl Options {
    pub fn of(p: &crate::FinishParams) -> Self {
        Self {
            voxel: p.free_voxel,
            margin: p.free_margin,
            pass_radius: p.free_pass_radius,
            pass_far: p.free_pass_far,
            protect_direction_spread: p.free_protect_direction_spread,
            protect_direction_scans: p.free_protect_direction_scans,
        }
    }
    /// The second, finer count (`free_fine_voxel`): the same rays, no thin-object protection.
    pub fn fine(p: &crate::FinishParams) -> Self {
        Self {
            voxel: p.free_fine_voxel,
            protect_direction_spread: 0.,
            ..Self::of(p)
        }
    }
}

pub struct FreeSpace {
    voxel: f64,
    /// voxel -> (scans that measured a point in it, scans whose rays passed through it)
    counts: HashMap<u64, (u32, u32)>,
    /// Voxels kept by the direction-spread protection: reported as never passed.
    protected: FxHashSet<u64>,
}
/// Measured voxels are grouped in blocks of 2^BLOCK_BITS voxels per axis (16 cm for 2 cm
/// voxels). Most steps of a ray are in the air of a room, in blocks without any measured
/// voxel: one look in the small block set, kept in cache, skips them without touching the
/// large voxel map. The voxels counted are the same.
const BLOCK_BITS: u32 = 3;
fn block(c: [i64; 3]) -> [i64; 3] {
    c.map(|x| x >> BLOCK_BITS)
}

fn cell(p: [f64; 3], inv: f64) -> [i64; 3] {
    p.map(|x| (x * inv).floor() as i64)
}
fn pack(c: [i64; 3]) -> u64 {
    let k = c.map(|x| (x + (1 << 20)) as u64 & 0x1f_ffff);
    (k[0] << 42) | (k[1] << 21) | k[2]
}
fn key(p: [f64; 3], inv: f64) -> u64 {
    pack(cell(p, inv))
}

/// One viewing direction per (voxel, scan), so repeated returns in one frame cannot
/// turn a transient into persistent support: from the sensor position of the scan's first
/// observation in the voxel (it moves during a scan) to the voxel centre.
fn direction_protection(
    raw: &[Observation],
    bounds: &[usize],
    voxel: f64,
    spread: f64,
    min_scans: u32,
) -> FxHashSet<u64> {
    let mut stats: HashMap<u64, ([f64; 3], u32)> = HashMap::default();
    for w in bounds.windows(2) {
        let mut keys: Vec<(u64, usize)> = (w[0]..w[1])
            .map(|i| (key(raw[i].xyz.map(|x| x as f64), 1. / voxel), i))
            .collect();
        keys.sort_unstable();
        keys.dedup_by_key(|k| k.0);
        for (k, i) in keys {
            let origin = raw[i].origin.map(|x| x as f64);
            let c = [42, 21, 0]
                .map(|s| ((((k >> s) & 0x1f_ffff) as i64 - (1 << 20)) as f64 + 0.5) * voxel);
            let d = [c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]];
            let len = d.iter().map(|x| x * x).sum::<f64>().sqrt();
            if len <= 1e-9 {
                continue;
            }
            let e = stats.entry(k).or_insert(([0.; 3], 0));
            for (a, b) in e.0.iter_mut().zip(d) {
                *a += b / len;
            }
            e.1 += 1;
        }
    }
    stats
        .into_iter()
        .filter_map(|(k, (sum, n))| {
            let mean_length = sum.iter().map(|x| x * x).sum::<f64>().sqrt() / n as f64;
            (n >= min_scans && 1. - mean_length > spread).then_some(k)
        })
        .collect()
}

/// Where each scan's observations start in `obs` (in scan order), with one more at the end.
fn scan_bounds(obs: &[Observation]) -> Vec<usize> {
    let mut bounds = vec![0];
    for i in 1..obs.len() {
        if obs[i].scan != obs[i - 1].scan {
            bounds.push(i);
        }
    }
    bounds.push(obs.len());
    bounds
}

impl FreeSpace {
    /// The counts of the voxels of `raw` (in scan order). `rays`: the observations of other
    /// parts whose rays may cross these voxels (in scan order), counted as passes only.
    pub fn new(raw: &[Observation], rays: &[Observation], o: &Options, job: &Job) -> Self {
        let (voxel, margin) = (o.voxel, o.margin);
        job.report("free space", 0, 1);
        let inv = 1. / voxel;
        let p = |o: &Observation| o.xyz.map(|x| x as f64);
        // Scans in order; the observations of one scan are contiguous.
        let bounds = scan_bounds(raw);
        let mut hits: Vec<u64> = bounds
            .par_windows(2)
            .flat_map_iter(|w| {
                let mut k: Vec<u64> = raw[w[0]..w[1]].iter().map(|o| key(p(o), inv)).collect();
                k.sort_unstable();
                k.dedup();
                k
            })
            .collect();
        let mut protected = FxHashSet::default();
        if o.protect_direction_spread > 0. {
            let directional = direction_protection(
                raw,
                &bounds,
                voxel,
                o.protect_direction_spread,
                o.protect_direction_scans,
            );
            job.log(&format!(
                "free space: {} voxels protected by direction spread > {} and {}+ scans",
                directional.len(),
                o.protect_direction_spread,
                o.protect_direction_scans
            ));
            protected.extend(directional);
        }
        hits.par_sort_unstable();
        let mut counts: HashMap<u64, (u32, u32)> =
            HashMap::with_capacity_and_hasher(hits.len() / 4, Default::default());
        for k in &hits {
            counts.entry(*k).or_insert((0, 0)).0 += 1;
        }
        hits.dedup();
        let keys = hits;
        let unpack = |k: u64| [42, 21, 0].map(|s| ((k >> s) & 0x1f_ffff) as i64 - (1 << 20));
        let blocks: FxHashSet<u64> = counts.keys().map(|&k| pack(block(unpack(k)))).collect();
        // Centroid of each measured voxel's observations, for `pass_radius`.
        let centroids: HashMap<u64, [f64; 3]> = if o.pass_radius > 0. {
            // summed in the order of the observations, so that a part of the data gives the
            // same centroids as the whole
            let mut keyed: Vec<(u64, u32, [f32; 3])> = raw
                .par_iter()
                .enumerate()
                .map(|(i, o)| (key(p(o), inv), i as u32, o.xyz))
                .collect();
            keyed.par_sort_unstable_by_key(|x| (x.0, x.1));
            let mut out = HashMap::with_capacity_and_hasher(counts.len(), Default::default());
            let mut i = 0;
            while i < keyed.len() {
                let (k, mut s, mut j) = (keyed[i].0, [0.; 3], i);
                while j < keyed.len() && keyed[j].0 == k {
                    for (a, b) in s.iter_mut().zip(keyed[j].2) {
                        *a += b as f64;
                    }
                    j += 1;
                }
                let n = (j - i) as f64;
                out.insert(k, s.map(|x| x / n));
                i = j;
            }
            out
        } else {
            HashMap::default()
        };
        let r2 = o.pass_radius * o.pass_radius;
        let far = o.pass_far;
        // Per scan: its observations here and its rays from elsewhere, one vote per voxel.
        let ray_bounds = scan_bounds(rays);
        let mut scans: Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> = Vec::new();
        let (mut a, mut b) = (0, 0);
        let (na, nb) = (
            bounds.len() - 1,
            if rays.is_empty() {
                0
            } else {
                ray_bounds.len() - 1
            },
        );
        while a < na || b < nb {
            let sa = (a < na).then(|| raw[bounds[a]].scan);
            let sb = (b < nb).then(|| rays[ray_bounds[b]].scan);
            let (own, other) = match (sa, sb) {
                (Some(x), Some(y)) if x == y => ((a, true), (b, true)),
                (Some(x), Some(y)) if x < y => ((a, true), (b, false)),
                (Some(_), None) => ((a, true), (b, false)),
                _ => ((a, false), (b, true)),
            };
            let r1 = if own.1 {
                bounds[a]..bounds[a + 1]
            } else {
                0..0
            };
            let r2 = if other.1 {
                ray_bounds[b]..ray_bounds[b + 1]
            } else {
                0..0
            };
            a += own.1 as usize;
            b += other.1 as usize;
            scans.push((r1, r2));
        }
        job.report("free space rays", 0, scans.len() as u64);
        // On the GPU when there is one (S20_NO_GPU set: on the CPU)
        let on_gpu = crate::gpu::shared()
            .filter(|_| std::env::var_os("S20_NO_GPU").is_none())
            .and_then(|g| {
                let mut bl: Vec<u64> = blocks.iter().copied().collect();
                bl.sort_unstable();
                let cent: Option<Vec<[f32; 3]>> = (o.pass_radius > 0.).then(|| {
                    keys.iter()
                        .map(|k| centroids.get(k).map_or([0.; 3], |m| m.map(|x| x as f32)))
                        .collect()
                });
                let walk = crate::gpu_free::Walk {
                    voxel,
                    margin,
                    pass_far: far,
                    pass_radius: o.pass_radius,
                };
                let rays_of = |j: usize, out: &mut Vec<[f32; 4]>| {
                    let (here, other) = &scans[j];
                    for o in raw[here.clone()].iter().chain(&rays[other.clone()]) {
                        out.push([o.origin[0], o.origin[1], o.origin[2], 0.]);
                        out.push([o.xyz[0], o.xyz[1], o.xyz[2], 0.]);
                    }
                };
                let progress = |done: usize| {
                    job.report("free space rays", done as u64, scans.len() as u64);
                    job.check()
                };
                match crate::gpu_free::passes(
                    g,
                    &keys,
                    &bl,
                    cent.as_deref(),
                    &walk,
                    scans.len(),
                    rays_of,
                    progress,
                ) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        job.log(&format!("free space: on the CPU (GPU: {e:#})"));
                        None
                    }
                }
            });
        if let Some(per_voxel) = on_gpu {
            let mut total = 0u64;
            for (k, n) in keys.iter().zip(per_voxel) {
                counts.get_mut(k).unwrap().1 = n;
                total += n as u64;
            }
            job.log(&format!(
                "free space: {} measured voxels of {:.0} mm, {} scan passes (GPU)",
                counts.len(),
                voxel * 1000.,
                total
            ));
            return Self {
                voxel,
                counts,
                protected,
            };
        }
        drop(keys);
        let completed = AtomicUsize::new(0);
        let step = 0.5 * voxel;
        let mut passes: Vec<u64> = scans
            .par_iter()
            .flat_map_iter(|(here, other)| {
                let mut k = Vec::new();
                for o in raw[here.clone()].iter().chain(&rays[other.clone()]) {
                    let a = o.origin.map(|x| x as f64);
                    let b = p(o);
                    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let stop = len - margin;
                    let u = d.map(|x| x / len);
                    let mut s = step;
                    // The last voxel and the last empty block seen on this ray: half-voxel
                    // steps visit each voxel about twice, and an empty block many times.
                    let mut last = None;
                    let mut empty = None;
                    while s < stop {
                        let f = s / len;
                        let c = cell([a[0] + d[0] * f, a[1] + d[1] * f, a[2] + d[2] * f], inv);
                        let left = len - s;
                        s += step;
                        if last == Some(c) {
                            continue;
                        }
                        last = Some(c);
                        let bc = block(c);
                        if empty == Some(bc) {
                            continue;
                        }
                        if !blocks.contains(&pack(bc)) {
                            empty = Some(bc);
                            continue;
                        }
                        let kk = pack(c);
                        if counts.contains_key(&kk)
                            && (far > 0. && left >= far
                                || centroids.get(&kk).is_none_or(|m| {
                                    let v = [m[0] - a[0], m[1] - a[1], m[2] - a[2]];
                                    let t = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
                                    v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - t * t <= r2
                                }))
                        {
                            k.push(kk);
                        }
                    }
                }
                k.sort_unstable();
                k.dedup();
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                job.report("free space rays", done as u64, scans.len() as u64);
                k
            })
            .collect();
        job.report("free space merge", 0, 1);
        passes.par_sort_unstable();
        for k in &passes {
            if let Some(c) = counts.get_mut(k) {
                c.1 += 1;
            }
        }
        job.log(&format!(
            "free space: {} measured voxels of {:.0} mm, {} scan passes",
            counts.len(),
            voxel * 1000.,
            passes.len()
        ));
        Self {
            voxel,
            counts,
            protected,
        }
    }
    /// Debug (`S20_FREE_DUMP=<dir>`): for the voxels this count finds free (`pass >= min_pass`
    /// and `pass > ratio * hit`), write per (voxel, scan) the passes (`passes.bin`: key u64,
    /// scan u32, the nearest distance of the scan's rays to the voxel's centroid, the mean
    /// direction of those rays, and the least distance left from the voxel to a ray's end, f32)
    /// and the hits (`hits.bin`: key u64, scan u32, sensor position and mean range, f32).
    pub fn dump(
        &self,
        raw: &[Observation],
        o: &Options,
        ratio: f64,
        min_pass: u32,
        dir: &std::path::Path,
    ) -> std::io::Result<()> {
        use std::io::Write;
        let inv = 1. / self.voxel;
        let p = |o: &Observation| o.xyz.map(|x| x as f64);
        // `S20_FREE_DUMP_BOX=minx,miny,minz,maxx,maxy,maxz` limits the dump to a box.
        let bx: Option<Vec<f64>> = std::env::var("S20_FREE_DUMP_BOX")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
        let unpack = |k: u64| [42, 21, 0].map(|s| ((k >> s) & 0x1f_ffff) as i64 - (1 << 20));
        let free: FxHashSet<u64> = self
            .counts
            .iter()
            .filter(|(_, (hit, pass))| *pass >= min_pass && *pass as f64 > ratio * *hit as f64)
            .map(|(k, _)| *k)
            .filter(|k| {
                bx.as_ref().is_none_or(|b| {
                    let c = unpack(*k).map(|x| (x as f64 + 0.5) * self.voxel);
                    (0..3).all(|i| c[i] >= b[i] && c[i] <= b[i + 3])
                })
            })
            .collect();
        let mut cent: HashMap<u64, ([f64; 3], f64)> = HashMap::default();
        for o in raw {
            let k = key(p(o), inv);
            if free.contains(&k) {
                let e = cent.entry(k).or_insert(([0.; 3], 0.));
                for (a, b) in e.0.iter_mut().zip(p(o)) {
                    *a += b;
                }
                e.1 += 1.;
            }
        }
        let cent: HashMap<u64, [f64; 3]> = cent
            .into_iter()
            .map(|(k, (s, n))| (k, s.map(|x| x / n)))
            .collect();
        let mut bounds = vec![0];
        for i in 1..raw.len() {
            if raw[i].scan != raw[i - 1].scan {
                bounds.push(i);
            }
        }
        bounds.push(raw.len());
        let step = 0.5 * self.voxel;
        type Pass = (f64, [f64; 3], f64, f64); // min distance, direction sum, rays, min left
        // per scan: its number, the voxels its rays passed, and the voxels it hit
        type Scan = (u32, Vec<(u64, Pass)>, Vec<(u64, ([f64; 4], f64))>);
        let per_scan: Vec<Scan> = bounds
            .par_windows(2)
            .map(|w| {
                let mut passes: HashMap<u64, Pass> = HashMap::default();
                let mut hits: HashMap<u64, ([f64; 4], f64)> = HashMap::default();
                for ob in &raw[w[0]..w[1]] {
                    let a = ob.origin.map(|x| x as f64);
                    let b = p(ob);
                    let kb = key(b, inv);
                    if free.contains(&kb) {
                        let e = hits.entry(kb).or_insert(([0.; 4], 0.));
                        for (x, y) in e.0.iter_mut().zip([a[0], a[1], a[2], ob.range as f64]) {
                            *x += y;
                        }
                        e.1 += 1.;
                    }
                    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let u = d.map(|x| x / len);
                    let mut s = step;
                    let mut last = None;
                    while s < len - o.margin {
                        let f = s / len;
                        let kk = key([a[0] + d[0] * f, a[1] + d[1] * f, a[2] + d[2] * f], inv);
                        let left = len - s;
                        s += step;
                        if last == Some(kk) {
                            continue;
                        }
                        last = Some(kk);
                        let Some(m) = cent.get(&kk) else { continue };
                        let v = [m[0] - a[0], m[1] - a[1], m[2] - a[2]];
                        let t = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
                        let dist = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - t * t)
                            .max(0.)
                            .sqrt();
                        let e =
                            passes
                                .entry(kk)
                                .or_insert((f64::INFINITY, [0.; 3], 0., f64::INFINITY));
                        e.0 = e.0.min(dist);
                        for (x, y) in e.1.iter_mut().zip(u) {
                            *x += y;
                        }
                        e.2 += 1.;
                        e.3 = e.3.min(left);
                    }
                }
                (
                    raw[w[0]].scan,
                    passes.into_iter().collect(),
                    hits.into_iter().collect(),
                )
            })
            .collect();
        std::fs::create_dir_all(dir)?;
        let mut fp = std::io::BufWriter::new(std::fs::File::create(dir.join("passes.bin"))?);
        let mut fh = std::io::BufWriter::new(std::fs::File::create(dir.join("hits.bin"))?);
        for (scan, passes, hits) in &per_scan {
            for (k, (dist, dirs, n, left)) in passes {
                fp.write_all(&k.to_le_bytes())?;
                fp.write_all(&scan.to_le_bytes())?;
                for x in [*dist, dirs[0] / n, dirs[1] / n, dirs[2] / n, *left] {
                    fp.write_all(&(x as f32).to_le_bytes())?;
                }
            }
            for (k, (s, n)) in hits {
                fh.write_all(&k.to_le_bytes())?;
                fh.write_all(&scan.to_le_bytes())?;
                for x in s {
                    fh.write_all(&((x / n) as f32).to_le_bytes())?;
                }
            }
        }
        Ok(())
    }
    /// (scans that measured, scans that passed through) the voxel of `p`.
    pub fn at(&self, p: [f32; 3]) -> (u32, u32) {
        let k = key(p.map(|x| x as f64), 1. / self.voxel);
        let (hit, pass) = self.counts.get(&k).copied().unwrap_or((0, 0));
        (hit, if self.protected.contains(&k) { 0 } else { pass })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn o(p: [f32; 3], origin: [f32; 3], scan: u32) -> Observation {
        Observation {
            time: scan as f64,
            xyz: p,
            origin,
            range: 1.,
            intensity: 0,
            scan,
        }
    }
    #[test]
    fn a_point_seen_through_later_is_free_a_wall_is_not() {
        let mut raw = vec![];
        // Scan 0 measures a person at x=1; scans 1..=9 see through it to the wall at x=2.
        raw.push(o([1., 0., 0.], [0., 0., 0.], 0));
        for s in 0..10u32 {
            if s > 0 {
                raw.push(o([2., 0., 0.], [0., 0., 0.], s));
            }
        }
        let opt = Options {
            voxel: 0.03,
            margin: 0.06,
            pass_radius: 0.,
            pass_far: 0.,
            protect_direction_spread: 0.,
            protect_direction_scans: 4,
        };
        let f = FreeSpace::new(&raw, &[], &opt, &Job::stderr());
        assert_eq!(f.at([1., 0., 0.]), (1, 9));
        assert_eq!(f.at([2., 0., 0.]), (9, 0));
    }

    #[test]
    fn directional_support_requires_distinct_scans_and_angular_spread() {
        let q = [1.01, 0.01, 0.01];
        let mut raw = Vec::new();
        for s in 0..4u32 {
            for _ in 0..20 {
                raw.push(o(q, [0., s as f32, 0.], s));
            }
        }
        let bounds = [0, 20, 40, 60, 80];
        let k = key(q.map(|x| x as f64), 50.);
        assert!(direction_protection(&raw, &bounds, 0.02, 0.03, 4).contains(&k));
        assert!(direction_protection(&raw[..60], &bounds[..4], 0.02, 0.03, 4).is_empty());
        for obs in &mut raw {
            obs.origin = [0.; 3];
        }
        assert!(direction_protection(&raw, &bounds, 0.02, 0.03, 4).is_empty());
    }

    #[test]
    fn directional_protection_keeps_persistent_target_but_carves_transient() {
        let q = [1.01, 0.01, 0.01];
        let transient = [1.01, 0.07, 0.01];
        let mut raw = Vec::new();
        for s in 0..4u32 {
            raw.push(o(q, [0., s as f32, 0.], s));
            if s == 0 {
                raw.push(o(transient, [0.; 3], s));
            }
        }
        for s in 4..24u32 {
            for y in [0.01, 0.07] {
                raw.push(o([2.01, y, 0.01], [0., y, 0.01], s));
            }
        }
        let mut opt = Options {
            voxel: 0.02,
            margin: 0.06,
            pass_radius: 0.01,
            pass_far: 0.2,
            protect_direction_spread: 0.,
            protect_direction_scans: 4,
        };
        let off = FreeSpace::new(&raw, &[], &opt, &Job::stderr());
        assert_eq!(off.at(q), (4, 20));
        opt.protect_direction_spread = 0.03;
        let on = FreeSpace::new(&raw, &[], &opt, &Job::stderr());
        assert_eq!(on.at(q), (4, 0));
        assert_eq!(on.at(transient), (1, 20));
    }
}
