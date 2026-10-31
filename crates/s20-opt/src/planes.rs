//! Adaptive voxel planes (port of `voxel_planes` in ba.py).
//!
//! Coarse to fine levels: points in a voxel are split by the side they were seen from (sign of
//! n . (sensor - point) with the voxel normal); a (voxel, side) group is a plane if it has enough points,
//! its thickness is below `max_std` and it is flat (lambda0 < ratio * lambda1). Accepted points are
//! removed before the next (finer) level.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PlaneParams {
    pub levels: Vec<f64>,
    pub min_pts: usize,
    pub max_std: f64,
    pub ratio: f64,
}

impl Default for PlaneParams {
    fn default() -> Self {
        PlaneParams {
            levels: vec![1.0, 0.5, 0.25, 0.125],
            min_pts: 20,
            max_std: 0.025,
            ratio: 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub n: Vector3<f64>,
    pub c: Vector3<f64>,
    pub std: f64,
    pub count: u32,
}

pub struct Planes {
    /// plane id per point (u32::MAX = none)
    pub pid: Vec<u32>,
    pub planes: Vec<Plane>,
}

pub const NONE: u32 = u32::MAX;

struct Group {
    start: usize,
    end: usize,
    mean: Vector3<f64>,
    ev: [f64; 3],
    normal: Vector3<f64>,
}

/// Runs of equal keys in sorted `keys`: (start, end) of each.
fn runs(keys: &[(u64, u32)]) -> Vec<(usize, usize)> {
    let starts: Vec<usize> = (0..keys.len())
        .into_par_iter()
        .filter(|&i| i == 0 || keys[i].0 != keys[i - 1].0)
        .collect();
    (0..starts.len())
        .into_par_iter()
        .map(|c| (starts[c], starts.get(c + 1).copied().unwrap_or(keys.len())))
        .collect()
}

/// Mean, eigenvalues (ascending) and normal of the points `pts` (in that order), which are
/// `start..start + pts.len()` of the sorted points.
fn stats(pts: &[[f32; 3]], start: usize) -> Group {
    let (a, b) = (start, start + pts.len());
    let n = (b - a) as f64;
    let mut m = Vector3::zeros();
    for p in pts {
        m += Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64);
    }
    m /= n;
    let mut c = Matrix3::zeros();
    for p in pts {
        let d = Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64) - m;
        c += d * d.transpose();
    }
    c /= n;
    let e = SymmetricEigen::new(c);
    let mut o = [0usize, 1, 2];
    o.sort_by(|&x, &y| e.eigenvalues[x].total_cmp(&e.eigenvalues[y]));
    Group {
        start: a,
        end: b,
        mean: m,
        ev: [
            e.eigenvalues[o[0]],
            e.eigenvalues[o[1]],
            e.eigenvalues[o[2]],
        ],
        normal: e.eigenvectors.column(o[0]).into_owned(),
    }
}

/// The voxel key with 21 bits per axis (the reference extraction's).
#[cfg(test)]
#[inline]
fn vkey(p: &[f32; 3], org: &[f64; 3], inv: f64) -> u64 {
    let mut k = 0u64;
    for d in 0..3 {
        let c = ((p[d] as f64 - org[d]) * inv).floor() as i64;
        k |= ((c.clamp(0, (1 << 21) - 1)) as u64) << (21 * d);
    }
    k
}

/// Voxel keys packed with only the bits the cells use: the axes in the order of `vkey` (x lowest),
/// so the keys order as `vkey` does, with fewer bits to sort.
struct DenseKey {
    org: [f64; 3],
    inv: f64,
    lim: [i64; 3],
    shift: [u32; 3],
    bits: u32,
}

impl DenseKey {
    /// `org`, `max`: bounds of the points.
    fn new(org: &[f64; 3], max: &[f64; 3], inv: f64) -> Self {
        let mut lim = [0i64; 3];
        let mut shift = [0u32; 3];
        let mut bits = 0;
        for d in 0..3 {
            // the largest cell (p <= max gives (p - org) * inv <= (max - org) * inv)
            lim[d] = (((max[d] - org[d]) * inv).floor() as i64).clamp(0, (1 << 21) - 1);
            shift[d] = bits;
            bits += 64 - (lim[d] as u64).leading_zeros();
        }
        DenseKey {
            org: *org,
            inv,
            lim,
            shift,
            bits,
        }
    }

    #[inline]
    fn key(&self, p: &[f32; 3]) -> u64 {
        let mut k = 0u64;
        for (d, &x) in p.iter().enumerate() {
            let c = ((x as f64 - self.org[d]) * self.inv).floor() as i64;
            k |= (c.clamp(0, self.lim[d]) as u64) << self.shift[d];
        }
        k
    }
}

/// Stable LSD radix sort by the low `bits` bits of the keys (in parallel).
fn radix_sort(v: &mut Vec<(u64, u32)>, bits: u32) {
    let n = v.len();
    if bits == 0 {
        return;
    }
    let passes = bits.div_ceil(11);
    let digit = bits.div_ceil(passes);
    let nb = 1usize << digit;
    let mask = (nb - 1) as u64;
    let chunk = n.div_ceil(rayon::current_num_threads() * 4).max(1 << 14);
    let mut buf = vec![(0u64, 0u32); n];
    for pass in 0..passes {
        let shift = pass * digit;
        let hist: Vec<Vec<usize>> = v
            .par_chunks(chunk)
            .map(|c| {
                let mut h = vec![0usize; nb];
                for e in c {
                    h[((e.0 >> shift) & mask) as usize] += 1;
                }
                h
            })
            .collect();
        // each chunk writes its elements of a digit after those of the chunks before it
        let mut offs = vec![vec![0usize; nb]; hist.len()];
        let mut s = 0;
        for d in 0..nb {
            for (o, h) in offs.iter_mut().zip(&hist) {
                o[d] = s;
                s += h[d];
            }
        }
        let out = Slots(buf.as_mut_ptr());
        v.par_chunks(chunk)
            .zip(offs.par_iter_mut())
            .for_each(|(c, o)| {
                for &e in c {
                    let d = ((e.0 >> shift) & mask) as usize;
                    // SAFETY: the offsets give every element its own slot.
                    unsafe { *out.at(o[d]) = e };
                    o[d] += 1;
                }
            });
        std::mem::swap(v, &mut buf);
    }
}

/// Raw pointer for disjoint parallel writes (each slot by one group).
struct Slots<T>(*mut T);
unsafe impl<T> Send for Slots<T> {}
unsafe impl<T> Sync for Slots<T> {}
impl<T> Slots<T> {
    fn at(&self, i: usize) -> *mut T {
        self.0.wrapping_add(i)
    }
}

/// `w`: world points, `sv`: vector point -> sensor (world). Only points with `use_pt` are considered.
///
/// Per level the points are sorted by voxel once; each voxel is then split by side keeping the
/// point order (side 0 first), which is the order a sort by (voxel, side, point) gives.
/// The sort is a stable radix sort of the points in index order: the order of a sort by
/// (voxel, point). Each voxel is then done in one task: its points are copied once, and the
/// voxel normal, the side split and the statistics of both sides read the copies (the same
/// values in the same order: the same planes).
pub fn extract(w: &[[f32; 3]], sv: &[[f32; 3]], prm: &PlaneParams) -> Planes {
    let n = w.len();
    let mut pid = vec![NONE; n];
    let mut planes: Vec<Plane> = Vec::new();
    let mut org = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in w {
        for d in 0..3 {
            org[d] = org[d].min(p[d] as f64);
            max[d] = max[d].max(p[d] as f64);
        }
    }
    for o in org.iter_mut() {
        *o -= 1.0;
    }
    let mut rem: Vec<u32> = (0..n as u32).collect();
    for &lev in &prm.levels {
        if rem.is_empty() {
            break;
        }
        let dk = DenseKey::new(&org, &max, 1.0 / lev);
        // pass 1: voxel normal for the side split (`rem` is in index order)
        let mut keys: Vec<(u64, u32)> = rem
            .par_iter()
            .map(|&i| (dk.key(&w[i as usize]), i))
            .collect();
        radix_sort(&mut keys, dk.bits);
        // per voxel: the voxel normal, then the (voxel, side) groups, side 0 then side 1, points
        // in order
        let mut keys2 = vec![(0u64, 0u32); keys.len()];
        let out = Slots(keys2.as_mut_ptr());
        type Scratch = (Vec<[f32; 3]>, Vec<u64>, Vec<[f32; 3]>);
        let g2: Vec<Group> = runs(&keys)
            .into_par_iter()
            .map_init(
                || (Vec::new(), Vec::new(), Vec::new()),
                |(pts, sides, split): &mut Scratch, (a, b)| {
                    let ks = &keys[a..b];
                    pts.clear();
                    pts.extend(ks.iter().map(|&(_, i)| w[i as usize]));
                    let nrm = stats(pts, a).normal;
                    sides.clear();
                    sides.extend(ks.iter().map(|&(_, i)| {
                        let s = sv[i as usize];
                        (nrm.x * s[0] as f64 + nrm.y * s[1] as f64 + nrm.z * s[2] as f64 > 0.0)
                            as u64
                    }));
                    let n0 = sides.iter().filter(|&&sd| sd == 0).count();
                    split.clear();
                    split.resize(b - a, [0.0; 3]);
                    let (mut a0, mut a1) = (0, n0);
                    for ((&(k, i), &sd), &p) in ks.iter().zip(sides.iter()).zip(pts.iter()) {
                        let at = if sd == 0 { &mut a0 } else { &mut a1 };
                        // SAFETY: the voxels' ranges a..b are disjoint.
                        unsafe { *out.at(a + *at) = (k * 2 + sd, i) };
                        split[*at] = p;
                        *at += 1;
                    }
                    [
                        (n0 > 0).then(|| stats(&split[..n0], a)),
                        (n0 < b - a).then(|| stats(&split[n0..], a + n0)),
                    ]
                },
            )
            .flat_map_iter(|g| g.into_iter().flatten())
            .collect();
        drop(keys);
        // planes in group order; the other points go to the next level
        let ok: Vec<bool> = g2
            .par_iter()
            .map(|g| {
                g.end - g.start >= prm.min_pts
                    && g.ev[0] < prm.max_std * prm.max_std
                    && g.ev[0] < prm.ratio * g.ev[1]
            })
            .collect();
        let mut ids = Vec::with_capacity(g2.len());
        for (g, &ok) in g2.iter().zip(&ok) {
            if ok {
                ids.push(planes.len() as u32);
                planes.push(Plane {
                    n: g.normal,
                    c: g.mean,
                    std: g.ev[0].max(0.0).sqrt(),
                    count: (g.end - g.start) as u32,
                });
            } else {
                ids.push(NONE);
            }
        }
        let slots = Slots(pid.as_mut_ptr());
        g2.par_iter().zip(&ids).for_each(|(g, &id)| {
            if id != NONE {
                for &(_, i) in &keys2[g.start..g.end] {
                    // SAFETY: every point is in one group only.
                    unsafe { *slots.at(i as usize) = id };
                }
            }
        });
        // the points not in a plane yet, in index order
        rem = (0..n as u32)
            .into_par_iter()
            .filter(|&i| pid[i as usize] == NONE)
            .collect();
    }
    Planes { pid, planes }
}

#[cfg(test)]
#[path = "planes_reference.rs"]
mod reference;
