//! Observations between two images anchored on the cloud: features of each image, their mutual
//! matches kept by RANSAC, and for both directions the anchor of one image's ray projected into the
//! other.

use crate::anchors::{Cloud, Pose, V3, ZGrid, associate, hint};
use crate::camera::Fisheye;
use crate::{image, matching, sift};
use anyhow::{Context, Result};
use nalgebra::{Quaternion, UnitQuaternion};
use rayon::prelude::*;
use std::path::Path;

pub struct Features {
    pub uv: Vec<[f64; 2]>,
    pub ray: Vec<[f64; 3]>,
    pub desc: Vec<[u8; 128]>,
}

/// SIFT on the image scaled to `max_size` on its long side, keypoints back in full pixels, those
/// within 80 degrees of the axis.
pub fn features(jpeg: &[u8], cam: &Fisheye, max_size: usize, count: usize) -> Result<Features> {
    let full = image::decode(jpeg)?;
    let s = (max_size as f64 / full.w.max(full.h) as f64).min(1.);
    let (w, h) = (
        ((full.w as f64) * s).round() as usize,
        ((full.h as f64) * s).round() as usize,
    );
    let small = image::resize_area(&full, w, h);
    let (kps, desc) = sift::detect(&small, count);
    let (sx, sy) = (w as f64 / full.w as f64, h as f64 / full.h as f64);
    let limit = 80f64.to_radians().cos();
    let mut f = Features {
        uv: Vec::new(),
        ray: Vec::new(),
        desc: Vec::new(),
    };
    for (k, d) in kps.iter().zip(desc) {
        let uv = [(k.x as f64 + 0.5) / sx - 0.5, (k.y as f64 + 0.5) / sy - 0.5];
        let r = cam.ray(uv);
        if r[2] > limit {
            f.uv.push(uv);
            f.ray.push(r);
            f.desc.push(d);
        }
    }
    Ok(f)
}

/// Matches kept by the fundamental matrix on normalised coordinates (threshold 3 px at `focal`).
pub fn matches(a: &Features, b: &Features, focal: f64) -> Vec<(usize, usize)> {
    let pairs = matching::mutual_ratio(&a.desc, &b.desc, 0.8);
    if pairs.len() < 12 {
        return Vec::new();
    }
    let na: Vec<[f64; 2]> = pairs
        .iter()
        .map(|&(i, _)| [a.ray[i][0] / a.ray[i][2], a.ray[i][1] / a.ray[i][2]])
        .collect();
    let nb: Vec<[f64; 2]> = pairs
        .iter()
        .map(|&(_, j)| [b.ray[j][0] / b.ray[j][2], b.ray[j][1] / b.ray[j][2]])
        .collect();
    let mask = matching::ransac_fundamental(&na, &nb, 3. / focal, 0.999, 10000);
    pairs
        .into_iter()
        .zip(mask)
        .filter(|(_, m)| *m)
        .map(|(p, _)| p)
        .collect()
}

/// One image's feature anchored on the cloud and seen at `uv_b` in the other image.
#[derive(Clone, Debug)]
pub struct Obs {
    pub a: usize,
    pub b: usize,
    /// the two features' indices in their images' `Features` (to join matches into tracks)
    pub fa: usize,
    pub fb: usize,
    pub uv_a: [f64; 2],
    pub uv_b: [f64; 2],
    pub world: V3,
    /// zero where the feature met no plane (`observe_guided` with `loose`)
    pub normal: V3,
    pub depth: f64,
}

impl Obs {
    /// anchored on a plane of the cloud
    pub fn planar(&self) -> bool {
        self.normal.norm_squared() > 0.5
    }
}

pub struct View<'a> {
    pub index: usize,
    pub pose: Pose,
    pub cam: &'a Fisheye,
    pub grid: &'a ZGrid,
    pub features: &'a Features,
}

/// Both directions of the matches between two views; `gate` drops what the given poses miss by
/// more pixels than that.
pub fn observe(a: &View, b: &View, cloud: &Cloud, focal: f64, gate: f64) -> Vec<Obs> {
    let m = matches(a.features, b.features, focal);
    let one_way = |x: &View, y: &View, pairs: Vec<(usize, usize)>| -> Vec<Obs> {
        pairs
            .par_iter()
            .filter_map(|&(i, j)| {
                let (p, q) = (x.features.uv[i], y.features.uv[j]);
                let an = associate(p, x.grid, &x.pose, x.cam, cloud)?;
                if cloud.occluders(&x.pose.p, &an.point, &an.normal) > 1
                    || cloud.occluders(&y.pose.p, &an.point, &an.normal) > 1
                {
                    return None;
                }
                let l = y.pose.r.transpose() * (an.point - y.pose.p);
                if l.xy().norm().atan2(l.z) > 80f64.to_radians() {
                    return None;
                }
                let pred = y.cam.project([l.x, l.y, l.z]);
                if (pred[0] - q[0]).hypot(pred[1] - q[1]) > gate {
                    return None;
                }
                Some(Obs {
                    a: x.index,
                    b: y.index,
                    fa: i,
                    fb: j,
                    uv_a: p,
                    uv_b: q,
                    world: an.point,
                    normal: an.normal,
                    depth: an.depth,
                })
            })
            .collect()
    };
    let mut out = one_way(a, b, m.clone());
    out.extend(one_way(b, a, m.into_iter().map(|(i, j)| (j, i)).collect()));
    out
}

fn descriptor_distance(a: &[u8; 128], b: &[u8; 128]) -> f32 {
    let mut s = 0i32;
    for k in 0..128 {
        let d = a[k] as i32 - b[k] as i32;
        s += d * d;
    }
    (s as f32).sqrt()
}

/// Both directions of observations matched where the poses and the cloud predict them: a feature
/// anchored on the cloud is compared only with the other image's features within `search` pixels
/// of its predicted position. The nearest must lie within `gate` pixels and pass the ratio test
/// (`ratio`) against the second nearest in the search area (or, alone there, an absolute distance); each
/// feature of the other image keeps its best proposal, and RANSAC drops what remains inconsistent.
/// `loose`: a feature on no plane is placed by `hint` instead (its observation's normal is zero).
#[allow(clippy::too_many_arguments)]
pub fn observe_guided(
    a: &View,
    b: &View,
    cloud: &Cloud,
    focal: f64,
    gate: f64,
    search: f64,
    ratio: f32,
    loose: bool,
) -> Vec<Obs> {
    const CELL: f64 = 32.;
    const ALONE: f32 = 250.;
    let one_way = |x: &View, y: &View| -> Vec<Obs> {
        let cols = (crate::anchors::WIDTH as f64 / CELL).ceil() as usize;
        let rows = (crate::anchors::HEIGHT as f64 / CELL).ceil() as usize;
        let mut cells: Vec<Vec<usize>> = vec![Vec::new(); cols * rows];
        for (j, uv) in y.features.uv.iter().enumerate() {
            let (cx, cy) = ((uv[0] / CELL) as usize, (uv[1] / CELL) as usize);
            if cx < cols && cy < rows {
                cells[cy * cols + cx].push(j);
            }
        }
        let reach = (search / CELL).ceil() as isize;
        let proposals: Vec<(usize, usize, f32, crate::anchors::Anchor)> = (0..x.features.uv.len())
            .into_par_iter()
            .filter_map(|i| {
                let uv = x.features.uv[i];
                let an = associate(uv, x.grid, &x.pose, x.cam, cloud).or_else(|| {
                    loose
                        .then(|| hint(uv, x.grid, &x.pose, x.cam, cloud))
                        .flatten()
                })?;
                let l = y.pose.r.transpose() * (an.point - y.pose.p);
                if l.xy().norm().atan2(l.z) > 80f64.to_radians() {
                    return None;
                }
                let pred = y.cam.project([l.x, l.y, l.z]);
                let (cx, cy) = (
                    (pred[0] / CELL).floor() as isize,
                    (pred[1] / CELL).floor() as isize,
                );
                let (mut best, mut d1, mut d2) = (usize::MAX, f32::INFINITY, f32::INFINITY);
                for gy in cy - reach..=cy + reach {
                    for gx in cx - reach..=cx + reach {
                        if gx < 0 || gy < 0 || gx >= cols as isize || gy >= rows as isize {
                            continue;
                        }
                        for &j in &cells[gy as usize * cols + gx as usize] {
                            let q = y.features.uv[j];
                            if (q[0] - pred[0]).hypot(q[1] - pred[1]) > search {
                                continue;
                            }
                            let d = descriptor_distance(&x.features.desc[i], &y.features.desc[j]);
                            if d < d1 {
                                d2 = d1;
                                d1 = d;
                                best = j;
                            } else if d < d2 {
                                d2 = d;
                            }
                        }
                    }
                }
                if best == usize::MAX {
                    return None;
                }
                let q = y.features.uv[best];
                let ok = (q[0] - pred[0]).hypot(q[1] - pred[1]) <= gate
                    && if d2.is_finite() {
                        d1 < ratio * d2
                    } else {
                        d1 < ALONE
                    };
                ok.then_some((i, best, d1, an))
            })
            .collect();
        // one proposal per feature of the other image (the closest descriptor) and RANSAC, first
        // for the features on planes alone (as without `loose`), then for the others, on the
        // features of the other image still free
        let norm = |r: &[f64; 3]| [r[0] / r[2], r[1] / r[2]];
        let select = |subset: Vec<usize>, taken: &std::collections::HashSet<usize>| -> Vec<usize> {
            let mut keep: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for k in subset {
                let p = &proposals[k];
                if taken.contains(&p.1) {
                    continue;
                }
                match keep.get(&p.1) {
                    Some(&o) if proposals[o].2 <= p.2 => {}
                    _ => {
                        keep.insert(p.1, k);
                    }
                }
            }
            let mut chosen: Vec<usize> = keep.into_values().collect();
            chosen.sort_unstable();
            if chosen.len() < 12 {
                return Vec::new();
            }
            let na: Vec<[f64; 2]> = chosen
                .iter()
                .map(|&k| norm(&x.features.ray[proposals[k].0]))
                .collect();
            let nb: Vec<[f64; 2]> = chosen
                .iter()
                .map(|&k| norm(&y.features.ray[proposals[k].1]))
                .collect();
            let mask = matching::ransac_fundamental(&na, &nb, 3. / focal, 0.999, 2000);
            chosen
                .into_iter()
                .zip(mask)
                .filter(|(_, m)| *m)
                .map(|(k, _)| k)
                .collect()
        };
        let planar = |k: &usize| proposals[*k].3.normal.norm_squared() > 0.5;
        let (on, off): (Vec<usize>, Vec<usize>) = (0..proposals.len()).partition(planar);
        let mut kept = select(on, &std::collections::HashSet::new());
        if !off.is_empty() {
            let taken = kept.iter().map(|&k| proposals[k].1).collect();
            kept.extend(select(off, &taken));
        }
        kept.par_iter()
            .filter_map(|&k| {
                let (i, j, _, an) = &proposals[k];
                // the occluders lie off the anchored plane: none to count without one
                if planar(&k)
                    && (cloud.occluders(&x.pose.p, &an.point, &an.normal) > 1
                        || cloud.occluders(&y.pose.p, &an.point, &an.normal) > 1)
                {
                    return None;
                }
                Some(Obs {
                    a: x.index,
                    b: y.index,
                    fa: *i,
                    fb: *j,
                    uv_a: x.features.uv[*i],
                    uv_b: y.features.uv[*j],
                    world: an.point,
                    normal: an.normal,
                    depth: an.depth,
                })
            })
            .collect()
    };
    let mut out = one_way(a, b);
    out.extend(one_way(b, a));
    out
}

/// `ImgPose.txt`: names, camera poses and stamps.
pub struct ImagePoses {
    pub names: Vec<String>,
    pub poses: Vec<Pose>,
    pub times: Vec<f64>,
    /// each row's text, to write a corrected file in the same layout
    pub rows: Vec<Vec<String>>,
}

pub fn read_image_poses(path: &Path) -> Result<ImagePoses> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = ImagePoses {
        names: Vec::new(),
        poses: Vec::new(),
        times: Vec::new(),
        rows: Vec::new(),
    };
    for line in text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with("index") && !l.starts_with('#'))
    {
        let v: Vec<String> = line.split_whitespace().map(str::to_string).collect();
        anyhow::ensure!(v.len() >= 12, "{}: bad line {line:?}", path.display());
        let f = |k: usize| -> Result<f64> { Ok(v[k].parse::<f64>()?) };
        let q = UnitQuaternion::from_quaternion(Quaternion::new(f(10)?, f(7)?, f(8)?, f(9)?));
        out.names.push(v[0].clone());
        out.poses.push(Pose {
            r: *q.to_rotation_matrix().matrix(),
            p: V3::new(f(1)?, f(2)?, f(3)?),
        });
        out.times.push(f(11)?);
        out.rows.push(v);
    }
    Ok(out)
}
