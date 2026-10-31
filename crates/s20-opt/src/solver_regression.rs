//! Frozen allocation-based PCG reference (double-precision preconditioner, full coupling vectors):
//! the fast solver must reach the same solution with comparable iterations.
use super::*;

#[test]
fn fast_pcg_matches_the_reference() {
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for n in [7, 128, 130, 193] {
                let mut h = BlockTridiag::new(n);
                for k in 0..n {
                    let a = M15::from_fn(|i, j| ((k * 31 + i * 17 + j * 7) as f64).sin() * 0.05);
                    h.d[k] = a * a.transpose() + M15::identity() * 5.0;
                    h.g[k] = V15::from_fn(|i, _| ((k * 13 + i * 3) as f64).cos());
                    if k + 1 < n {
                        h.u[k] = a * 0.1;
                    }
                }
                let normals: Vec<V3> = (0..41)
                    .map(|p| {
                        let t = p as f64 * 0.7;
                        V3::new(t.cos() * 0.6, t.sin() * 0.6, 0.8)
                    })
                    .collect();
                let mut entries = Vec::new();
                for k in (0..n - 2).rev() {
                    for plane in (0..40).rev() {
                        if (k + plane) % 11 == 0 {
                            continue;
                        }
                        let f = |i: usize| ((k * 19 + plane * 7 + i) as f64).sin() * 0.1;
                        entries.push(Coupling {
                            plane: plane as u32,
                            knot: k as u32,
                            a: V3::new(f(0), f(1), f(2)),
                            // plane 0 has no weight (an empty plane)
                            w: if plane == 0 { 0.0 } else { 0.5 + f(3).abs() },
                            wtau: f(4).abs() * 0.05,
                            wr: 0.0,
                        });
                    }
                }
                // the points behind each entry: sum w J J^T >= u u^T / w, which keeps the system
                // positive definite after the low-rank part (weighted Cauchy-Schwarz)
                for c in entries.iter().filter(|c| c.w > 0.0) {
                    let u = c.u(&normals[c.plane as usize]);
                    let mut d = h.d[c.knot as usize].fixed_view_mut::<9, 9>(0, 0);
                    d += u * u.transpose() / c.w;
                }
                for entries in [entries, Vec::new()] {
                    let pc = PlaneCoupling::new(entries.clone(), normals.clone());
                    let far = vec![(
                        0,
                        n - 1,
                        M15::from_fn(|i, j| ((i * 3 + j * 17) as f64).cos() * 0.07),
                    )];
                    for far in [far, Vec::new()] {
                        for lambda in [1e-6, 0.1] {
                            let (expected, ei) =
                                reference_solve(&entries, &normals, &h, lambda, 1e-10, 600, &far)
                                    .unwrap();
                            let (actual, ai) = pc.solve(&h, lambda, 1e-10, 600, &far).unwrap();
                            assert!(
                                ai <= ei + ei / 10 + 2,
                                "n {n}: {ai} iterations, reference {ei}"
                            );
                            if !far.is_empty() {
                                assert!(ai > 0, "n {n}: no iterations (reference {ei})");
                            }
                            let err: f64 = actual
                                .iter()
                                .zip(&expected)
                                .map(|(a, b)| (a - b).norm_squared())
                                .sum::<f64>()
                                .sqrt();
                            let size: f64 = expected
                                .iter()
                                .map(|b| b.norm_squared())
                                .sum::<f64>()
                                .sqrt();
                            assert!(err <= 1e-8 * size, "n {n}: error {err:e} of {size:e}");
                        }
                    }
                }
            }
        });
    }
}

fn reference_mul(h: &BlockTridiag, x: &[V15], lambda: f64) -> Vec<V15> {
    let n = h.d.len();
    (0..n)
        .map(|k| {
            let mut y = h.d[k] * x[k];
            for i in 0..15 {
                // same damping as the factorisation (which adds 1e-12 to the Jacobi-scaled diagonal)
                y[i] += (lambda + 1e-12) * h.d[k][(i, i)].abs() * x[k][i];
            }
            if k + 1 < n {
                y += h.u[k] * x[k + 1];
            }
            if k > 0 {
                y += h.u[k - 1].transpose() * x[k - 1];
            }
            y
        })
        .collect()
}

fn reference_lowrank(e: &[Coupling], normals: &[V3], x: &[V15], y: &mut [V15]) {
    let np = normals.len();
    let mut wsum = vec![0.0; np];
    let mut d = vec![0.0; np];
    for c in e {
        let p = c.plane as usize;
        wsum[p] += c.w;
        d[p] += c.u(&normals[p]).dot(&x[c.knot as usize].fixed_rows::<9>(0));
    }
    for c in e {
        let p = c.plane as usize;
        if wsum[p] > 0.0 {
            let f = d[p] / wsum[p];
            let u = c.u(&normals[p]);
            let mut yk = y[c.knot as usize].fixed_rows_mut::<9>(0);
            yk -= f * u;
        }
    }
}

fn reference_solve(
    e: &[Coupling],
    normals: &[V3],
    h: &BlockTridiag,
    lambda: f64,
    tol: f64,
    max_it: usize,
    far: &[(usize, usize, M15)],
) -> Result<(Vec<V15>, usize)> {
    let f = h.factor(lambda)?;
    let b: Vec<V15> = h.g.iter().map(|g| -g).collect();
    let dot = |a: &[V15], b: &[V15]| a.iter().zip(b).map(|(x, y)| x.dot(y)).sum::<f64>();
    let mul = |x: &[V15]| {
        let mut y = reference_mul(h, x, lambda);
        reference_lowrank(e, normals, x, &mut y);
        for (a, b, m) in far {
            y[*a] += m * x[*b];
            y[*b] += m.transpose() * x[*a];
        }
        y
    };
    let mut x = f.apply(&b);
    let ax = mul(&x);
    let mut r: Vec<V15> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let mut z = f.apply(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let rz0 = dot(&b, &x).max(1e-300);
    let mut it = 0;
    while it < max_it && rz > tol * tol * rz0 {
        let ap = mul(&p);
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            break;
        }

        let alpha = rz / pap;
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        z = f.apply(&r);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..p.len() {
            p[i] = z[i] + beta * p[i];
        }
        it += 1;
    }
    Ok((x, it))
}
