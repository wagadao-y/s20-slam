//! The original plane extraction (two sorts per level), kept to check the faster one.
use super::*;

struct Group {
    start: usize,
    end: usize,
    mean: Vector3<f64>,
    ev: [f64; 3],
    normal: Vector3<f64>,
}

/// Sort `idx` by key and compute statistics of each run of equal keys.
fn group_stats(keys: &mut [(u64, u32)], w: &[[f32; 3]]) -> Vec<Group> {
    keys.par_sort_unstable();
    let mut bounds = Vec::new();
    let mut s = 0;
    for i in 1..=keys.len() {
        if i == keys.len() || keys[i].0 != keys[s].0 {
            bounds.push((s, i));
            s = i;
        }
    }
    let keys = &*keys;
    bounds
        .par_iter()
        .map(|&(a, b)| {
            let n = (b - a) as f64;
            let mut m = Vector3::zeros();
            for &(_, i) in &keys[a..b] {
                let p = w[i as usize];
                m += Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            }
            m /= n;
            let mut c = Matrix3::zeros();
            for &(_, i) in &keys[a..b] {
                let p = w[i as usize];
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
        })
        .collect()
}

/// `w`: world points, `sv`: vector point -> sensor (world). Only points with `use_pt` are considered.
fn reference_extract(w: &[[f32; 3]], sv: &[[f32; 3]], prm: &PlaneParams) -> Planes {
    let n = w.len();
    let mut pid = vec![NONE; n];
    let mut planes: Vec<Plane> = Vec::new();
    let mut org = [f64::INFINITY; 3];
    for p in w {
        for d in 0..3 {
            org[d] = org[d].min(p[d] as f64);
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
        let inv = 1.0 / lev;
        // pass 1: voxel normal for the side split
        let mut keys: Vec<(u64, u32)> = rem
            .par_iter()
            .map(|&i| (vkey(&w[i as usize], &org, inv), i))
            .collect();
        let g1 = group_stats(&mut keys, w);
        let mut keys2: Vec<(u64, u32)> = g1
            .par_iter()
            .flat_map_iter(|g| {
                let nrm = g.normal;
                keys[g.start..g.end].iter().map(move |&(k, i)| {
                    let s = sv[i as usize];
                    let side = (nrm.x * s[0] as f64 + nrm.y * s[1] as f64 + nrm.z * s[2] as f64
                        > 0.0) as u64;
                    (k * 2 + side, i)
                })
            })
            .collect();
        // pass 2: (voxel, side) groups
        let g2 = group_stats(&mut keys2, w);
        let mut next = Vec::new();
        for g in &g2 {
            let cnt = g.end - g.start;
            let ok = cnt >= prm.min_pts
                && g.ev[0] < prm.max_std * prm.max_std
                && g.ev[0] < prm.ratio * g.ev[1];
            if ok {
                let id = planes.len() as u32;
                planes.push(Plane {
                    n: g.normal,
                    c: g.mean,
                    std: g.ev[0].max(0.0).sqrt(),
                    count: cnt as u32,
                });
                for &(_, i) in &keys2[g.start..g.end] {
                    pid[i as usize] = id;
                }
            } else {
                next.extend(keys2[g.start..g.end].iter().map(|&(_, i)| i));
            }
        }
        rem = next;
    }
    Planes { pid, planes }
}

/// Noisy walls and floors seen from both sides, with clutter.
fn scene() -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut seed = 5u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    };
    let (mut w, mut sv) = (Vec::new(), Vec::new());
    for i in 0..60000 {
        let (a, b, e) = (rnd() * 8.0, rnd() * 8.0, rnd() * 0.01);
        let (p, sensor) = match i % 4 {
            0 => ([a, b, e], [0.0, 0.0, 1.5]),
            1 => ([a, 2.0 + e, b.abs() * 0.4], [0.0, 0.0, 1.5]),
            2 => ([a, 2.0 + e, b.abs() * 0.4], [0.0, 5.0, 1.5]),
            _ => ([a, b, rnd().abs() * 3.0], [0.0, 0.0, 1.5]),
        };
        w.push([p[0] as f32 + 50.0, p[1] as f32 - 20.0, p[2] as f32 + 1.0]);
        sv.push([
            (sensor[0] - p[0]) as f32,
            (sensor[1] - p[1]) as f32,
            (sensor[2] - p[2]) as f32,
        ]);
    }
    (w, sv)
}

#[test]
fn faster_extraction_gives_the_same_planes() {
    let (w, sv) = scene();
    for prm in [
        PlaneParams::default(),
        PlaneParams {
            levels: vec![2.0, 1.0, 0.5],
            max_std: 0.08,
            ..Default::default()
        },
    ] {
        let a = reference_extract(&w, &sv, &prm);
        let b = extract(&w, &sv, &prm);
        assert!(a.planes.len() > 10, "{} planes", a.planes.len());
        assert_eq!(a.pid, b.pid);
        assert_eq!(a.planes.len(), b.planes.len());
        for (p, q) in a.planes.iter().zip(&b.planes) {
            assert_eq!(p.count, q.count);
            assert_eq!(p.std.to_bits(), q.std.to_bits());
            for d in 0..3 {
                assert_eq!(p.n[d].to_bits(), q.n[d].to_bits());
                assert_eq!(p.c[d].to_bits(), q.c[d].to_bits());
            }
        }
    }
}

#[test]
fn radix_sort_orders_as_a_sort_by_key_and_index() {
    let mut seed = 11u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed >> 1
    };
    for (n, bits) in [
        (0usize, 5u32),
        (100, 3),
        (200_000, 7),
        (200_000, 25),
        (200_000, 63),
    ] {
        let mut v: Vec<(u64, u32)> = (0..n as u32)
            .map(|i| (rnd() & ((1u64 << bits) - 1), i))
            .collect();
        let mut expected = v.clone();
        expected.sort_unstable();
        radix_sort(&mut v, bits);
        assert_eq!(v, expected, "n {n} bits {bits}");
    }
}
