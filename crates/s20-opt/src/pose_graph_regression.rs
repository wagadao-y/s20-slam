use super::*;

fn fixture(prm: &Params, rtk: bool) -> Problem {
    let n = 5;
    let states: Vec<_> = (0..n)
        .map(|k| State {
            r: M3::identity(),
            p: V3::new(k as f64 * 0.03, 0.0, 0.0),
            v: V3::new(0.3, 0.0, 0.0),
            bg: V3::zeros(),
            ba: V3::zeros(),
        })
        .collect();
    let rows: Vec<_> = (0..101)
        .map(|i| [i as f64 * 0.005, 0.01, 0.02, -0.005, 0.0, 0.0, 1.0])
        .collect();
    let mut pr = Problem {
        kt: (0..n).map(|k| k as f64 * 0.1).collect(),
        init: states.clone(),
        priors: vec![(0, states[0])],
        free: vec![false],
        states,
        linked: vec![true, true, false, true],
        pts: Points {
            x: vec![[1.0, 0.0, 0.0]; n - 1],
            tau: vec![0.025; n - 1],
            off: vec![0, 1, 2, 3, 4, 4],
        },
        imu: ImuData::from_rows(&rows, 0.0),
        session: vec![0; n],
        lambda: prm.lambda,
        loops: Vec::new(),
        rtk: Vec::new(),
        rtk_prm: RtkParams::default(),
    };
    pr.states[2].p.x += 0.002;
    pr.states[3].v.y += 0.01;
    if rtk {
        let pre = pr.preintegrate(&prm.noise);
        let p = pr.antenna(&pr.states[1], pre[1].as_ref().unwrap(), 0.025).0;
        pr.rtk.push(RtkObs {
            sd: V3::zeros(),
            t: 0.125,
            knot: 1,
            tau: 0.025,
            p: p + V3::new(0.02, 0.01, 0.03),
        });
    }
    pr
}

#[test]
fn no_plane_path_preserves_rtk_and_imu() {
    let prm = Params::default();
    for odometry in [false, true] {
        for rtk in [false, true] {
            for couple in [false, true] {
                let mut fast = fixture(&prm, rtk);
                let mut extracted = fixture(&prm, rtk);
                let stage = Stage {
                    iters: 1,
                    planes: relpose::no_planes(),
                    c_cauchy: 0.15,
                    couple,
                    rtk_c: None,
                    odometry,
                };
                // Exercise ordinary preintegration/point conversion with no accepted planes.
                let reference = Stage {
                    planes: PlaneParams {
                        levels: vec![1.0],
                        min_pts: usize::MAX,
                        ..Default::default()
                    },
                    ..stage.clone()
                };
                let a = fast.step(&prm, &stage).unwrap();
                let b = extracted.step(&prm, &reference).unwrap();
                assert_eq!(a.planes, 0);
                assert_eq!(a.used, 0.0);
                assert_eq!(a.cg_iters, b.cg_iters);
                assert_eq!(a.tries, b.tries);
                for (a, b) in [
                    (a.cost0, b.cost0),
                    (a.cost1, b.cost1),
                    (a.lambda, b.lambda),
                    (a.imu_chi2, b.imu_chi2),
                    (a.rtk_h_mm, b.rtk_h_mm),
                    (a.rtk_v_mm, b.rtk_v_mm),
                ] {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                assert_eq!(a.imu_chi2 > 0.0, !odometry);
                assert_eq!(a.rtk_h_mm > 0.0, rtk);
                assert_eq!(a.rtk_v_mm > 0.0, rtk);
                for (a, b) in fast.states.iter().zip(&extracted.states) {
                    let rotation = a.r.iter().zip(b.r.iter());
                    let vectors = [&a.p, &a.v, &a.bg, &a.ba]
                        .into_iter()
                        .zip([&b.p, &b.v, &b.bg, &b.ba])
                        .flat_map(|(a, b)| a.iter().zip(b.iter()));
                    for (a, b) in rotation.chain(vectors) {
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                }
            }
        }
    }
}

#[test]
fn gnss_uncertainty_scales_constraints_without_changing_legacy_rtk() {
    let prm = Params::default();
    let mut pr = fixture(&prm, true);
    let pre = pr.preintegrate(&prm.noise);
    let o = pr.rtk[0];
    pr.rtk[0].p = pr
        .antenna(&pr.states[o.knot], pre[o.knot].as_ref().unwrap(), o.tau)
        .0;
    let base = pr.rtk_terms(&pr.states, &pre, 0.05, false)[0].h.trace();
    pr.rtk[0].sd = V3::new(0.1, 0.1, 0.2);
    let uncertain = pr.rtk_terms(&pr.states, &pre, 0.05, false)[0].h.trace();
    assert!((uncertain / base - 0.01).abs() < 1e-10);
    pr.rtk[0].sd = V3::new(0.001, 0.001, 0.002);
    assert!((pr.rtk_terms(&pr.states, &pre, 0.05, false)[0].h.trace() / base - 1.).abs() < 1e-10);
}

#[test]
fn pose_graph_gnss_moves_the_pose_not_the_velocity() {
    let prm = Params::default();
    let mut pr = fixture(&prm, true);
    let pre = pr.preintegrate(&prm.noise);
    let o = pr.rtk[0];
    assert!(o.tau > 0.0);
    // without IMU factors, a free velocity would absorb the fix through v tau
    let t = &pr.rtk_terms(&pr.states, &pre, 0.05, true)[0];
    assert_eq!(t.h.fixed_view::<3, 9>(6, 0).norm(), 0.0);
    assert_eq!(t.g.fixed_rows::<3>(6).norm(), 0.0);
    let t = &pr.rtk_terms(&pr.states, &pre, 0.05, false)[0];
    assert!(t.h.fixed_view::<3, 3>(6, 6).norm() > 0.0);
    // Huber: a fix far from the trajectory keeps a constant pull (k / |u| of the weight)
    pr.rtk[0].p = pr
        .antenna(&pr.states[o.knot], pre[o.knot].as_ref().unwrap(), o.tau)
        .0;
    let near = pr.rtk_terms(&pr.states, &pre, 0.05, false)[0].h.trace();
    pr.rtk[0].p += V3::new(5.0, 0.0, 0.0);
    let far = &pr.rtk_terms(&pr.states, &pre, 0.05, false)[0];
    let k = 0.05 / pr.rtk_prm.sig_h;
    let u = 5.0 / pr.rtk_prm.sig_h;
    assert!((far.h.trace() / near - k / u).abs() < 1e-6);
    assert!((far.cost - (k * u - 0.5 * k * k)).abs() < 1e-6 * far.cost);
}
