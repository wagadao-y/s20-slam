//! Matching descriptors between two images: nearest neighbours by L2 with Lowe's ratio test, kept
//! only when mutual, then the fundamental matrix by RANSAC on normalised image coordinates (x/z,
//! y/z of the rays), as `cv2.findFundamentalMat(FM_RANSAC)` with the larger of the two
//! point-to-epipolar-line distances.

use nalgebra::{DMatrix, Matrix3, Vector3};
use rayon::prelude::*;

/// For every descriptor of `a`, its nearest and second nearest in `b` (index, distances).
fn two_nearest(
    a: &[[f32; 128]],
    na: &[f32],
    b: &[[f32; 128]],
    nb: &[f32],
) -> Vec<(usize, f32, f32)> {
    a.par_iter()
        .zip(na)
        .map(|(x, &nx)| {
            let (mut best, mut d1, mut d2) = (usize::MAX, f32::INFINITY, f32::INFINITY);
            for (j, (y, &ny)) in b.iter().zip(nb).enumerate() {
                let mut dot = 0f32;
                for k in 0..128 {
                    dot += x[k] * y[k];
                }
                let d = (nx + ny - 2. * dot).max(0.);
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                    best = j;
                } else if d < d2 {
                    d2 = d;
                }
            }
            (best, d1.sqrt(), d2.sqrt())
        })
        .collect()
}

/// Index pairs (in `a`, in `b`) that pass the ratio test both ways and agree.
pub fn mutual_ratio(a: &[[u8; 128]], b: &[[u8; 128]], ratio: f32) -> Vec<(usize, usize)> {
    if a.len() < 2 || b.len() < 2 {
        return Vec::new();
    }
    let f = |d: &[[u8; 128]]| -> (Vec<[f32; 128]>, Vec<f32>) {
        let v: Vec<[f32; 128]> = d
            .iter()
            .map(|x| std::array::from_fn(|k| x[k] as f32))
            .collect();
        let n = v.iter().map(|x| x.iter().map(|q| q * q).sum()).collect();
        (v, n)
    };
    let ((fa, na), (fb, nb)) = (f(a), f(b));
    let fwd = two_nearest(&fa, &na, &fb, &nb);
    let bwd = two_nearest(&fb, &nb, &fa, &na);
    fwd.iter()
        .enumerate()
        .filter_map(|(i, &(j, d1, d2))| {
            let (back, e1, e2) = bwd[j];
            (d1 < ratio * d2 && e1 < ratio * e2 && back == i).then_some((i, j))
        })
        .collect()
}

/// Eight-point fundamental matrix (Hartley normalisation, rank two).
fn eight_point(p: &[[f64; 2]], q: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    let norm = |v: &[[f64; 2]]| -> Matrix3<f64> {
        let n = v.len() as f64;
        let (mx, my) = (
            v.iter().map(|x| x[0]).sum::<f64>() / n,
            v.iter().map(|x| x[1]).sum::<f64>() / n,
        );
        let d = v
            .iter()
            .map(|x| ((x[0] - mx).powi(2) + (x[1] - my).powi(2)).sqrt())
            .sum::<f64>()
            / n;
        let s = if d > 1e-12 {
            std::f64::consts::SQRT_2 / d
        } else {
            1.
        };
        Matrix3::new(s, 0., -s * mx, 0., s, -s * my, 0., 0., 1.)
    };
    let (tp, tq) = (norm(p), norm(q));
    let mut a = DMatrix::<f64>::zeros(p.len().max(9), 9);
    for (k, (x, y)) in p.iter().zip(q).enumerate() {
        let u = tp * Vector3::new(x[0], x[1], 1.);
        let v = tq * Vector3::new(y[0], y[1], 1.);
        for i in 0..3 {
            for j in 0..3 {
                a[(k, 3 * i + j)] = v[i] * u[j];
            }
        }
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let (idx, _) = svd
        .singular_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))?;
    let f = vt.row(idx);
    let f = Matrix3::new(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]);
    let mut s = f.svd(true, true);
    let (u, vt) = (s.u?, s.v_t?);
    let mut order: Vec<usize> = (0..3).collect();
    order.sort_by(|&i, &j| s.singular_values[j].total_cmp(&s.singular_values[i]));
    s.singular_values[order[2]] = 0.;
    let f = u * Matrix3::from_diagonal(&s.singular_values) * vt;
    Some(tq.transpose() * f * tp)
}

/// OpenCV's error for a fundamental matrix: the larger squared distance to the two epipolar lines.
fn error(f: &Matrix3<f64>, p: [f64; 2], q: [f64; 2]) -> f64 {
    let (x1, y1, x2, y2) = (p[0], p[1], q[0], q[1]);
    let l2 = f * Vector3::new(x1, y1, 1.);
    let d2 = x2 * l2[0] + y2 * l2[1] + l2[2];
    let s2 = 1. / (l2[0] * l2[0] + l2[1] * l2[1]);
    let l1 = f.transpose() * Vector3::new(x2, y2, 1.);
    let d1 = x1 * l1[0] + y1 * l1[1] + l1[2];
    let s1 = 1. / (l1[0] * l1[0] + l1[1] * l1[1]);
    (d1 * d1 * s1).max(d2 * d2 * s2)
}

/// Inliers of the fundamental matrix found by RANSAC (threshold in normalised coordinates).
pub fn ransac_fundamental(
    p: &[[f64; 2]],
    q: &[[f64; 2]],
    threshold: f64,
    confidence: f64,
    max_iters: usize,
) -> Vec<bool> {
    let n = p.len();
    if n < 8 {
        return vec![false; n];
    }
    let thr2 = threshold * threshold;
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut best: Vec<bool> = vec![false; n];
    let mut best_count = 0;
    let mut iters = max_iters;
    let mut it = 0;
    while it < iters {
        it += 1;
        let mut idx = [0usize; 8];
        let mut k = 0;
        while k < 8 {
            let c = (next() % n as u64) as usize;
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        let sp: Vec<[f64; 2]> = idx.iter().map(|&i| p[i]).collect();
        let sq: Vec<[f64; 2]> = idx.iter().map(|&i| q[i]).collect();
        let Some(f) = eight_point(&sp, &sq) else {
            continue;
        };
        let mask: Vec<bool> = (0..n).map(|i| error(&f, p[i], q[i]) <= thr2).collect();
        let count = mask.iter().filter(|&&m| m).count();
        if count > best_count {
            best_count = count;
            best = mask;
            let w = count as f64 / n as f64;
            let denom = (1. - w.powi(8)).max(1e-12).ln();
            if denom < 0. {
                iters = iters.min(((1. - confidence).ln() / denom).ceil().max(1.) as usize);
            }
        }
    }
    // one refit on the inliers
    if best_count >= 8 {
        let (sp, sq): (Vec<_>, Vec<_>) = (0..n).filter(|&i| best[i]).map(|i| (p[i], q[i])).unzip();
        if let Some(f) = eight_point(&sp, &sq) {
            let mask: Vec<bool> = (0..n).map(|i| error(&f, p[i], q[i]) <= thr2).collect();
            if mask.iter().filter(|&&m| m).count() >= best_count {
                best = mask;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ransac_keeps_a_rigid_motion_and_drops_outliers() {
        // points seen from two cameras 1 m apart, 20 % of the matches scrambled
        let mut p = Vec::new();
        let mut q = Vec::new();
        let rot = nalgebra::Rotation3::from_euler_angles(0.02, -0.1, 0.03);
        let t = Vector3::new(1., 0.1, 0.05);
        for i in 0..200 {
            let a = i as f64 * 0.61803;
            let x = Vector3::new(
                (a * 7.).sin() * 2.,
                (a * 3.).cos() * 1.5,
                4. + (a * 5.).sin(),
            );
            let y = rot * x + t;
            p.push([x[0] / x[2], x[1] / x[2]]);
            q.push([y[0] / y[2], y[1] / y[2]]);
        }
        for i in (0..200).step_by(5) {
            q[i] = [q[i][0] + 0.1, q[i][1] - 0.07];
        }
        let mask = ransac_fundamental(&p, &q, 1e-3, 0.999, 10000);
        let kept: Vec<usize> = (0..200).filter(|&i| mask[i]).collect();
        assert!(kept.len() >= 155, "{}", kept.len());
        assert!(kept.iter().all(|i| i % 5 != 0));
    }

    #[test]
    fn mutual_ratio_pairs_identical_descriptors() {
        let a: Vec<[u8; 128]> = (0..20)
            .map(|i| std::array::from_fn(|k| ((i * 37 + k * 11) % 251) as u8))
            .collect();
        let mut b = a.clone();
        b.reverse();
        let m = mutual_ratio(&a, &b, 0.8);
        assert_eq!(m.len(), 20);
        assert!(m.iter().all(|&(i, j)| j == 19 - i));
    }
}
