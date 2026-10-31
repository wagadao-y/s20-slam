//! The cloud made square by square, so that the memory follows the observations of a square and
//! not of the data (`tile_size`, `tile_obs`). The result is the same as all at once.
//!
//! A square (its own `out` cells) keeps the points whose final post-thinning cell lies in it. It
//! finishes the representatives within it grown by the farthest a point moves onto its plane
//! (`queries`), from the observations within that grown by what a representative depends on
//! (planes, shared fits and the isolation test: `obs`), and counts the free space of those
//! observations' voxels with the rays of every observation that may cross them. All three
//! rectangles are on the [`BASE`] grid, so each cell of the sampling, moments, voxels, shared
//! fits and post thinning lies whole in them.
//!
//! The bag is read twice: once to count the observations per small cell and choose the squares
//! (one of `tile_size` split in four while it holds more than `tile_obs` observations), once to
//! write each square's observations and rays to temporary files next to the output. The squares
//! are then finished one by one. The points the passing-object test is to look at are held back
//! and tested at the end over the scans read a third time ([`crate::transient::find`]): a cone
//! from a scan reaches past any margin.
use crate::{
    BASE, FinishParams, Observation, Part, PointResult, Rect, load, range_sigma, transient,
};
use anyhow::{Context, Result};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use s20_geom::so3::V3;
use s20_geom::traj::Traj;
use s20_io::dataset::LidarData;
use s20_io::job::Job;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Scans read at a time.
const CHUNK: usize = 256;
/// The squares are counted in cells of this many [`BASE`] cells.
const SMALL: i64 = 100;
/// A square is not split below this side (BASE cells).
const MIN_SIDE: i64 = 2 * SMALL;
/// Bytes of an observation in the temporary files.
const RECORD: usize = 46;

/// The margins (BASE cells): (queries beyond `out`, observations beyond `queries`).
pub fn margins(p: &FinishParams) -> (i64, i64) {
    let cells = |m: f64| (m / BASE).ceil() as i64 + 1;
    // a point moves at most to the end of its taper, the off-plane limit when farther
    let reach = p
        .mls_taper_end
        .max(p.reject_sigma * p.max_surface_sigma.max(range_sigma(p.max_range)));
    let moves = reach + p.post_leaf;
    // a query's plane: within the radius of the mean of its shared-fit cell, over moment cells
    // whose centroids are within; its isolation test: within the support radius
    let plane = p.mls_radius + p.anchor_grid * 3f64.sqrt() + p.moment_cell * 3f64.sqrt();
    let support = p.support_radius.max(p.support_range * p.max_range);
    let depends = plane.max(support) + p.moment_cell + p.free_voxel.max(p.free_fine_voxel);
    (cells(moves), cells(depends))
}

fn base_cell(x: [f32; 3]) -> [i64; 2] {
    [0, 1].map(|d| (x[d] as f64 / BASE).floor() as i64)
}

/// Whether the segment a-b (xy, in BASE units) meets rectangle `r` (Liang-Barsky).
fn crosses(a: [f64; 2], b: [f64; 2], r: &Rect) -> bool {
    let (mut t0, mut t1) = (0f64, 1f64);
    for d in 0..2 {
        let (lo, hi) = (r.lo[d] as f64, r.hi[d] as f64);
        let delta = b[d] - a[d];
        if delta.abs() < 1e-12 {
            if a[d] < lo || a[d] >= hi {
                return false;
            }
            continue;
        }
        let (mut u, mut v) = ((lo - a[d]) / delta, (hi - a[d]) / delta);
        if u > v {
            std::mem::swap(&mut u, &mut v);
        }
        t0 = t0.max(u);
        t1 = t1.min(v);
        if t0 > t1 {
            return false;
        }
    }
    true
}

fn put(o: &Observation, w: &mut impl Write) -> std::io::Result<()> {
    let mut b = [0u8; RECORD];
    b[..8].copy_from_slice(&o.time.to_le_bytes());
    for (k, x) in o
        .xyz
        .iter()
        .chain(&o.origin)
        .chain(std::iter::once(&o.range))
        .enumerate()
    {
        b[8 + 4 * k..12 + 4 * k].copy_from_slice(&x.to_le_bytes());
    }
    b[36..40].copy_from_slice(&o.scan.to_le_bytes());
    b[40..42].copy_from_slice(&o.intensity.to_le_bytes());
    w.write_all(&b[..42])
}

fn read_all(path: &Path) -> Result<Vec<Observation>> {
    let mut r = BufReader::new(
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?,
    );
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    let f = |b: &[u8], i: usize| f32::from_le_bytes(b[i..i + 4].try_into().unwrap());
    Ok(bytes
        .as_chunks::<42>()
        .0
        .iter()
        .map(|b| Observation {
            time: f64::from_le_bytes(b[..8].try_into().unwrap()),
            xyz: [f(b, 8), f(b, 12), f(b, 16)],
            origin: [f(b, 20), f(b, 24), f(b, 28)],
            range: f(b, 32),
            scan: u32::from_le_bytes(b[36..40].try_into().unwrap()),
            intensity: u16::from_le_bytes(b[40..42].try_into().unwrap()),
        })
        .collect())
}

/// What [`finish_tiled`] did: the observations, the representatives, the points by status and
/// the seconds by stage (summed over the squares; `profile.*` are summed over the threads).
pub struct Tiled {
    pub observations: usize,
    pub sampled: usize,
    pub status: std::collections::BTreeMap<String, usize>,
    pub seconds: std::collections::BTreeMap<String, f64>,
}

/// A square: its own cells, and the rectangles it finishes and reads.
struct Tile {
    part: Part,
    obs: Rect,
}

/// Removes the folder of the temporary files when the run ends, also on an error.
pub(crate) struct TempDir(pub(crate) PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The scans' observations, in scan order, chunk by chunk.
fn for_chunks(
    data: &LidarData,
    tr: &Traj,
    lever: V3,
    p: &FinishParams,
    job: &Job,
    stage: &str,
    mut each: impl FnMut(Vec<Vec<Observation>>) -> Result<()>,
) -> Result<()> {
    let left_out = load::set_down_scans(data, tr, p.trim_still);
    let n = data.len();
    for a in (0..n).step_by(CHUNK) {
        job.check()?;
        job.report(stage, a as u64, n as u64);
        let b = (a + CHUNK).min(n);
        let scans: Vec<Vec<Observation>> = (a..b)
            .into_par_iter()
            .map(|i| {
                if left_out[i] {
                    Vec::new()
                } else {
                    load::scan(data, tr, lever, p.max_range, i)
                }
            })
            .collect();
        each(scans)?;
    }
    job.report(stage, n as u64, n as u64);
    Ok(())
}

/// The cloud of `data` placed with `tr` (see [`load`]), made in squares; each square's points
/// (observations and results of the kept ones) go to `emit` as it finishes. `tmp`: a folder for
/// the temporary files (removed at the end).
pub fn finish_tiled(
    data: &LidarData,
    tr: &Traj,
    lever: [f64; 3],
    p: &FinishParams,
    tmp: &Path,
    job: &Job,
    mut emit: impl FnMut(&[Observation], &[PointResult]) -> Result<()>,
) -> Result<Tiled> {
    p.validate()?;
    let lever = V3::from(lever);
    let side = ((p.tile_size / BASE).round() as i64).max(MIN_SIDE);
    let (mq, ms) = margins(p);
    let mut seconds: std::collections::BTreeMap<String, f64> = Default::default();
    let mut clock = std::time::Instant::now();
    let mut lap = |seconds: &mut std::collections::BTreeMap<String, f64>, name: &str| {
        *seconds.entry(name.into()).or_default() += clock.elapsed().as_secs_f64();
        clock = std::time::Instant::now();
    };
    // 1. observations per small cell
    let mut counts: FxHashMap<[i64; 2], u64> = FxHashMap::default();
    let mut total = 0usize;
    // the progress: reading the scans twice (count, split) about 5% of the time, the squares the
    // rest, each by its observations (the park 2026-10-07 10-31: 19 of 415 s)
    for_chunks(
        data,
        tr,
        lever,
        p,
        &job.part(0.0, 0.01),
        "cloud count",
        |scans| {
            for o in scans.iter().flatten() {
                *counts
                    .entry(base_cell(o.xyz).map(|c| c.div_euclid(SMALL)))
                    .or_default() += 1;
                total += 1;
            }
            Ok(())
        },
    )?;
    anyhow::ensure!(total > 0, "no points within the trajectory");
    // the squares: tile_size, split in four while their observations (with the margin) are many
    let count_in = |r: &Rect| -> u64 {
        let (lo, hi) = (
            r.lo.map(|x| x.div_euclid(SMALL)),
            r.hi.map(|x| (x - 1).div_euclid(SMALL)),
        );
        let mut n = 0;
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                n += counts.get(&[i, j]).copied().unwrap_or(0);
            }
        }
        n
    };
    let mut roots: Vec<[i64; 2]> = counts
        .keys()
        .map(|c| c.map(|x| (x * SMALL).div_euclid(side)))
        .collect();
    roots.sort_unstable();
    roots.dedup();
    let mut tiles: Vec<Tile> = Vec::new();
    let mut stack: Vec<Rect> = roots
        .iter()
        .rev()
        .map(|r| Rect {
            lo: r.map(|x| x * side),
            hi: r.map(|x| (x + 1) * side),
        })
        .collect();
    while let Some(out) = stack.pop() {
        let obs = out.grow(mq + ms);
        let w = out.hi[0] - out.lo[0];
        if count_in(&out) == 0 {
            continue;
        }
        if count_in(&obs) as usize > p.tile_obs && w >= 2 * MIN_SIDE {
            let mid = [0, 1].map(|d| out.lo[d] + (out.hi[d] - out.lo[d]) / 2);
            for (x, y) in [(1, 1), (1, 0), (0, 1), (0, 0)] {
                let lo = [
                    if x == 0 { out.lo[0] } else { mid[0] },
                    if y == 0 { out.lo[1] } else { mid[1] },
                ];
                let hi = [
                    if x == 0 { mid[0] } else { out.hi[0] },
                    if y == 0 { mid[1] } else { out.hi[1] },
                ];
                stack.push(Rect { lo, hi });
            }
            continue;
        }
        tiles.push(Tile {
            part: Part {
                queries: out.grow(mq),
                out,
            },
            obs,
        });
    }
    let weight: Vec<u64> = tiles.iter().map(|t| count_in(&t.obs)).collect();
    drop(counts);
    job.log(&format!(
        "{total} observations; {} squares of up to {:.0} m (margins {:.2} + {:.2} m)",
        tiles.len(),
        side as f64 * BASE,
        mq as f64 * BASE,
        ms as f64 * BASE
    ));
    lap(&mut seconds, "tiles.count");
    // which squares' observation rectangles overlap each small cell
    let mut index: FxHashMap<[i64; 2], Vec<u32>> = FxHashMap::default();
    for (k, t) in tiles.iter().enumerate() {
        let (lo, hi) = (
            t.obs.lo.map(|x| x.div_euclid(SMALL)),
            t.obs.hi.map(|x| (x - 1).div_euclid(SMALL)),
        );
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                index.entry([i, j]).or_default().push(k as u32);
            }
        }
    }
    // 2. each square's observations and rays to its files
    std::fs::create_dir_all(tmp).with_context(|| format!("create {}", tmp.display()))?;
    let _cleanup = TempDir(tmp.to_path_buf());
    let file = |k: usize, kind: &str| tmp.join(format!("{k}.{kind}"));
    let mut writers: Vec<(BufWriter<std::fs::File>, BufWriter<std::fs::File>)> = (0..tiles.len())
        .map(|k| -> Result<_> {
            Ok((
                BufWriter::new(std::fs::File::create(file(k, "obs"))?),
                BufWriter::new(std::fs::File::create(file(k, "rays"))?),
            ))
        })
        .collect::<Result<_>>()?;
    let empty: Vec<u32> = Vec::new();
    for_chunks(
        data,
        tr,
        lever,
        p,
        &job.part(0.01, 0.05),
        "cloud split",
        |scans| {
            // per scan: (square, ray?) of each observation, in order
            let routes: Vec<Vec<(u32, bool)>> = scans
                .par_iter()
                .map(|obs| {
                    let mut out = Vec::new();
                    let mut seen: Vec<u32> = Vec::new();
                    for o in obs {
                        let c = base_cell(o.xyz);
                        let a = [o.origin[0] as f64 / BASE, o.origin[1] as f64 / BASE];
                        let b = [o.xyz[0] as f64 / BASE, o.xyz[1] as f64 / BASE];
                        // the small cells the ray's xy segment passes, by its bounding box
                        let (lo, hi) = (
                            [0, 1].map(|d| (a[d].min(b[d]).floor() as i64).div_euclid(SMALL)),
                            [0, 1].map(|d| (a[d].max(b[d]).floor() as i64).div_euclid(SMALL)),
                        );
                        seen.clear();
                        for i in lo[0]..=hi[0] {
                            for j in lo[1]..=hi[1] {
                                for &k in index.get(&[i, j]).unwrap_or(&empty) {
                                    if seen.contains(&k) {
                                        continue;
                                    }
                                    seen.push(k);
                                    let t = &tiles[k as usize];
                                    if t.obs.contains(c) {
                                        out.push((k, false));
                                    } else if crosses(a, b, &t.obs) {
                                        out.push((k, true));
                                    }
                                }
                            }
                        }
                        out.push((u32::MAX, false));
                    }
                    out
                })
                .collect();
            for (obs, route) in scans.iter().zip(&routes) {
                let mut it = route.iter();
                for o in obs {
                    for &(k, ray) in it.by_ref() {
                        if k == u32::MAX {
                            break;
                        }
                        let w = &mut writers[k as usize];
                        put(o, if ray { &mut w.1 } else { &mut w.0 })?;
                    }
                }
            }
            Ok(())
        },
    )?;
    for (a, b) in writers.drain(..) {
        a.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        b.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    }
    lap(&mut seconds, "tiles.split");
    // 3. the squares one by one
    let mut sampled = 0usize;
    let mut status: std::collections::BTreeMap<String, usize> = Default::default();
    let squares = job.part(0.05, 0.8);
    // the points to test for passing objects, held back from `emit`
    let (mut held_obs, mut held_res): (Vec<Observation>, Vec<PointResult>) =
        (Vec::new(), Vec::new());
    let (mut done, all) = (0, weight.iter().sum::<u64>());
    for (k, t) in tiles.iter().enumerate() {
        job.check()?;
        squares.report("cloud squares", done, all);
        done += weight[k];
        let obs = read_all(&file(k, "obs"))?;
        let rays = read_all(&file(k, "rays"))?;
        let _ = std::fs::remove_file(file(k, "obs"));
        let _ = std::fs::remove_file(file(k, "rays"));
        if obs.is_empty() {
            continue;
        }
        job.log(&format!(
            "square {} of {}: ({:.1}, {:.1}) m, {} observations, {} rays",
            k + 1,
            tiles.len(),
            t.part.out.lo[0] as f64 * BASE,
            t.part.out.lo[1] as f64 * BASE,
            obs.len(),
            rays.len()
        ));
        lap(&mut seconds, "tiles.read");
        let quiet = Job::new(|_, _, _| {}, |_| {});
        let f = crate::finish_part(&obs, &rays, Some(&t.part), p, &quiet)?;
        drop((obs, rays));
        for (k, s) in &f.timings {
            *seconds.entry(k.clone()).or_default() += s;
        }
        lap(&mut seconds, "tiles.finish");
        sampled += f.input.len();
        for (i, name) in crate::STATUS_NAMES.iter().enumerate() {
            let n = f.points.iter().filter(|r| r.status == i as u32).count();
            if n > 0 {
                *status.entry(name.to_string()).or_default() += n;
            }
        }
        let (mut o, mut r) = (Vec::new(), Vec::new());
        for ((ob, res), &c) in f.input.iter().zip(&f.points).zip(&f.candidates) {
            if crate::removed(res.status) {
                continue;
            }
            if c {
                held_obs.push(*ob);
                held_res.push(*res);
            } else {
                o.push(*ob);
                r.push(*res);
            }
        }
        emit(&o, &r)?;
        lap(&mut seconds, "tiles.write");
    }
    squares.report("cloud squares", all, all);
    // 4. passing objects among the points held back
    if !held_obs.is_empty() {
        job.log(&format!(
            "transient: testing {} points over the scans",
            held_obs.len()
        ));
        let points: Vec<[f32; 3]> = held_res.iter().map(|r| r.xyz).collect();
        let passing = transient::find(data, tr, lever, p, &points, &job.part(0.8, 1.0));
        let mut taken = 0;
        let (mut o, mut r) = (Vec::new(), Vec::new());
        for ((ob, res), t) in held_obs.iter().zip(&held_res).zip(passing) {
            if t {
                if let Some(n) = status.get_mut(crate::STATUS_NAMES[res.status as usize]) {
                    *n -= 1;
                }
                taken += 1;
            } else {
                o.push(*ob);
                r.push(*res);
            }
        }
        if taken > 0 {
            status.insert(
                crate::STATUS_NAMES[crate::TRANSIENT as usize].to_string(),
                taken,
            );
        }
        job.log(&format!(
            "transient: {taken} of {} points seen through by many more scans than saw them",
            held_obs.len()
        ));
        emit(&o, &r)?;
        lap(&mut seconds, "transient");
    }
    for (k, s) in &seconds {
        job.log(&format!("seconds {k}: {s:.1}"));
    }
    Ok(Tiled {
        observations: total,
        sampled,
        status,
        seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A floor and a wall measured from a walk, finished in two halves (each from its
    /// observations and the rays crossing them), give the points of the whole.
    #[test]
    fn two_halves_give_the_whole() {
        let mut raw = Vec::new();
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for scan in 0..60u32 {
            let o = [-1.5 + 0.05 * scan as f64, 0.0, 1.0];
            for k in 0..1500 {
                // the floor (z = 0) and a wall (y = 1.2), with a few cm of range noise
                let p = if k % 3 == 0 {
                    [o[0] + 2.0 * rnd(), 1.2, 1.0 + rnd()]
                } else {
                    [o[0] + 3.0 * rnd(), 1.1 * rnd() + 0.55, 0.0]
                };
                let d: Vec<f64> = (0..3).map(|i| p[i] - o[i]).collect();
                let r = d.iter().map(|x| x * x).sum::<f64>().sqrt();
                let e = 1.0 + 0.004 * rnd() / r;
                let x: [f32; 3] = std::array::from_fn(|i| (o[i] + d[i] * e) as f32);
                raw.push(Observation {
                    time: scan as f64 * 0.1,
                    xyz: x,
                    origin: o.map(|v| v as f32),
                    range: (r * e) as f32,
                    scan,
                    intensity: 0,
                });
            }
        }
        let p = FinishParams {
            trim_still: 0.,
            min_support: 20,
            ..Default::default()
        };
        let job = Job::new(|_, _, _| {}, |_| {});
        let whole = crate::finish(&raw, &p, &job).unwrap();
        let key = |r: &PointResult| r.xyz.map(|v| (v as f64 * 1e5).round() as i64);
        let mut a: Vec<_> = whole.kept().map(|(_, r)| key(r)).collect();
        let (mq, ms) = margins(&p);
        let mut b = Vec::new();
        for out in [
            Rect {
                lo: [-500, -500],
                hi: [0, 500],
            },
            Rect {
                lo: [0, -500],
                hi: [500, 500],
            },
        ] {
            let obs_rect = out.grow(mq + ms);
            let own: Vec<_> = raw
                .iter()
                .filter(|o| obs_rect.contains(base_cell(o.xyz)))
                .copied()
                .collect();
            let rays: Vec<_> = raw
                .iter()
                .filter(|o| {
                    !obs_rect.contains(base_cell(o.xyz))
                        && crosses(
                            [o.origin[0] as f64 / BASE, o.origin[1] as f64 / BASE],
                            [o.xyz[0] as f64 / BASE, o.xyz[1] as f64 / BASE],
                            &obs_rect,
                        )
                })
                .copied()
                .collect();
            let part = Part {
                queries: out.grow(mq),
                out,
            };
            let f = crate::finish_part(&own, &rays, Some(&part), &p, &job).unwrap();
            b.extend(f.kept().map(|(_, r)| key(r)));
        }
        a.sort_unstable();
        b.sort_unstable();
        assert!(a.len() > 10_000, "{}", a.len());
        assert_eq!(a, b);
    }
}
