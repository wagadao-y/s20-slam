//! Trajectory comparison against a reference (e.g. LIO against the PPK-constrained optimisation).
//!
//! - Drift: the estimate is fitted rigidly to the reference over its first `align_s` seconds only, as an
//!   unreferenced survey would be, and the horizontal / vertical error is reported along the path.
//! - ATE: the same after a rigid fit over the whole trajectory.
//! - Relative errors over 1, 5 and 30 s windows (position in the start frame of the window, rotation).
//! - Jumps: 1 s windows whose relative error exceeds 5 cm or 1 degree, merged into intervals.

use anyhow::{Result, bail};
use nalgebra::{Matrix3, SVD};
use s20_geom::so3::{self, M3, V3};
use std::path::Path;

struct Pose {
    t: f64,
    r: M3,
    p: V3,
}

fn load(path: &Path) -> Result<Vec<Pose>> {
    Ok(s20_io::dataset::read_tum(path)?
        .iter()
        .map(|q| Pose {
            t: q[0],
            r: so3::from_quat([q[4], q[5], q[6], q[7]]),
            p: V3::new(q[1], q[2], q[3]),
        })
        .collect())
}

/// Rotation and translation that best map `a` onto `b` (Kabsch, no scale).
fn rigid_fit(a: &[V3], b: &[V3]) -> (M3, V3) {
    let n = a.len() as f64;
    let ma: V3 = a.iter().sum::<V3>() / n;
    let mb: V3 = b.iter().sum::<V3>() / n;
    let mut c = Matrix3::zeros();
    for (x, y) in a.iter().zip(b) {
        c += (y - mb) * (x - ma).transpose();
    }
    let svd = SVD::new(c, true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let d = Matrix3::from_diagonal(&V3::new(1.0, 1.0, (u * vt).determinant().signum()));
    let r = u * d * vt;
    (r, mb - r * ma)
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

pub fn run(est: &Path, reference: &Path, align_s: f64) -> Result<()> {
    let e = load(est)?;
    let rf = load(reference)?;
    // pair each estimated pose with the reference pose at the same time (both are knots at scan ends)
    let mut pairs: Vec<(&Pose, &Pose)> = Vec::new();
    for x in &e {
        let k = rf.partition_point(|y| y.t < x.t);
        let best = [k.saturating_sub(1), k.min(rf.len().saturating_sub(1))]
            .into_iter()
            .min_by(|&i, &j| (rf[i].t - x.t).abs().total_cmp(&(rf[j].t - x.t).abs()));
        if let Some(i) = best
            && (rf[i].t - x.t).abs() < 1e-3
        {
            pairs.push((x, &rf[i]));
        }
    }
    if pairs.len() < 10 {
        bail!("only {} poses at common times", pairs.len());
    }
    let t0 = pairs[0].0.t;
    let ts: Vec<f64> = pairs.iter().map(|(x, _)| x.t - t0).collect();
    let pe: Vec<V3> = pairs.iter().map(|(x, _)| x.p).collect();
    let pr: Vec<V3> = pairs.iter().map(|(_, y)| y.p).collect();
    let mut path = vec![0.0];
    for w in pr.windows(2) {
        path.push(path.last().unwrap() + (w[1] - w[0]).norm());
    }
    println!(
        "{} poses at common times, {:.0} s, path {:.0} m",
        pairs.len(),
        ts.last().unwrap(),
        path.last().unwrap()
    );

    // drift after fitting the start only
    let na = ts.partition_point(|&t| t < align_s).max(10);
    let (ra, ta) = rigid_fit(&pe[..na], &pr[..na]);
    println!("drift (fitted on the first {align_s:.0} s):");
    let mut worst_h: f64 = 0.0;
    let mut worst_v: f64 = 0.0;
    let mut next = 0;
    let marks = [
        60.0, 120.0, 300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2400.0, 3000.0,
    ];
    for k in 0..ts.len() {
        let d = ra * pe[k] + ta - pr[k];
        worst_h = worst_h.max(d.xy().norm());
        worst_v = worst_v.max(d.z.abs());
        let last = k + 1 == ts.len();
        if (next < marks.len() && ts[k] >= marks[next]) || last {
            println!(
                "  t {:5.0} s  path {:5.0} m  horiz {:6.2} m  vert {:6.2} m  ({:.2}% of path)",
                ts[k],
                path[k],
                d.xy().norm(),
                d.z.abs(),
                100.0 * d.norm() / path[k].max(1.0)
            );
            while next < marks.len() && ts[k] >= marks[next] {
                next += 1;
            }
        }
    }
    println!("  worst  horiz {worst_h:.2} m  vert {worst_v:.2} m");

    let (rg, tg) = rigid_fit(&pe, &pr);
    let mut ate: Vec<f64> = (0..ts.len())
        .map(|k| (rg * pe[k] + tg - pr[k]).norm())
        .collect();
    let rms = (ate.iter().map(|x| x * x).sum::<f64>() / ate.len() as f64).sqrt();
    println!(
        "ATE (fitted on all): rms {:.3} m  max {:.3} m",
        rms,
        quantile(&mut ate, 1.0)
    );

    // relative errors over windows
    let rel = |i: usize, j: usize| -> (f64, f64) {
        let (a, b) = (pairs[i], pairs[j]);
        let de = a.0.r.transpose() * (b.0.p - a.0.p);
        let dr = a.1.r.transpose() * (b.1.p - a.1.p);
        let qe = a.0.r.transpose() * b.0.r;
        let qr = a.1.r.transpose() * b.1.r;
        (
            (de - dr).norm(),
            so3::log(&(qe * qr.transpose())).norm().to_degrees(),
        )
    };
    println!("relative error:");
    let mut jumps = Vec::new();
    for w in [1.0, 5.0, 30.0] {
        let (mut dp, mut da) = (Vec::new(), Vec::new());
        for i in 0..ts.len() {
            let j = ts.partition_point(|&t| t < ts[i] + w);
            if j >= ts.len() {
                break;
            }
            let (p, a) = rel(i, j);
            if w == 1.0 && (p > 0.05 || a > 1.0) {
                jumps.push((ts[i], p, a));
            }
            dp.push(p);
            da.push(a);
        }
        let n = dp.len();
        let over = dp.iter().filter(|&&x| x > 0.03).count();
        println!(
            "  {w:2.0} s  pos median {:6.1} mm  p99 {:7.1} mm  max {:7.1} mm  | rot median {:.3}  p99 {:.3}  max {:.3} deg{}",
            1e3 * quantile(&mut dp, 0.5),
            1e3 * quantile(&mut dp, 0.99),
            1e3 * quantile(&mut dp, 1.0),
            quantile(&mut da, 0.5),
            quantile(&mut da, 0.99),
            quantile(&mut da, 1.0),
            if w == 1.0 {
                format!(
                    "  | >3 cm in {:.1}% of windows",
                    100.0 * over as f64 / n as f64
                )
            } else {
                String::new()
            }
        );
    }

    // merge the jump windows into intervals, worst first
    let mut iv: Vec<(f64, f64, f64, f64)> = Vec::new(); // start, end, worst pos, worst rot
    for (t, p, a) in jumps {
        match iv.last_mut() {
            Some(l) if t - l.1 <= 2.0 => {
                l.1 = t;
                l.2 = l.2.max(p);
                l.3 = l.3.max(a);
            }
            _ => iv.push((t, t, p, a)),
        }
    }
    let total: f64 = iv.iter().map(|x| x.1 - x.0 + 1.0).sum();
    println!(
        "jumps (1 s windows over 5 cm or 1 deg): {} intervals, {:.0} s in all",
        iv.len(),
        total
    );
    iv.sort_by(|a, b| b.2.total_cmp(&a.2));
    for x in iv.iter().take(12) {
        println!(
            "  {:7.1} - {:7.1} s  pos {:6.0} mm  rot {:.2} deg",
            x.0,
            x.1 + 1.0,
            1e3 * x.2,
            x.3
        );
    }
    Ok(())
}
