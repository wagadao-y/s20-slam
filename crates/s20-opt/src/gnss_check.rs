//! False-fix screening of GNSS positions against the LIO odometry.
//!
//! The offset o = fix - antenna (LIO after the initial alignment) drifts slowly with the odometry, but a
//! false fix moves it by a step: the receiver changes integers while the LIO moves smoothly. Steps are only
//! attributed to the GNSS between consecutive epochs (an outage gives no comparison) and while the LIO is
//! healthy (its position agrees with its velocity), since an LIO failure also steps the offset.
//!
//! The fixes are split into segments at outages ("soft" edges) and GNSS steps ("hard" edges). At each hard
//! edge one side is wrong:
//! - the segment after the step when a later step returns to the level before it (an excursion),
//! - otherwise the side that began at an outage or at the start (wrong integers are usually chosen at a
//!   (re)initialisation), unless it is much longer than the other side.
//!
//! A rejection spreads across short outages to the neighbours that are still at the wrong level. Steps
//! smaller than the thresholds are left to the robust GNSS weights and the LiDAR terms.

use s20_geom::so3::V3;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CheckParams {
    pub enabled: bool,
    /// smallest horizontal / vertical step between consecutive fixes treated as a GNSS jump (m)
    pub step_h: f64,
    pub step_v: f64,
    /// LIO knots whose position disagrees with their velocity by more than this (m) excuse the steps
    /// within `LIO_WINDOW` seconds
    pub lio_tol: f64,
    /// a rejection spreads across outages up to this long (s)
    pub max_gap: f64,
}

impl Default for CheckParams {
    fn default() -> Self {
        CheckParams {
            enabled: true,
            step_h: 0.15,
            step_v: 0.30,
            lio_tol: 0.03,
            max_gap: 10.0,
        }
    }
}

/// Consecutive fixes further apart than this (s) are not compared.
const CONTINUOUS: f64 = 0.5;
/// Steps within this time of an unhealthy LIO knot are attributed to the LIO (s).
const LIO_WINDOW: f64 = 0.5;
/// The longer side of a step is kept when it is this many times longer.
const LONGER: f64 = 3.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Rejected {
    pub t0: f64,
    pub t1: f64,
    pub n: usize,
    /// offset from the wrong level to the accepted one (m)
    pub corr: V3,
}

#[derive(Clone, Copy, PartialEq)]
enum Edge {
    Soft,
    Hard(V3),
}

/// `t`: fix times (sorted), `o`: fix minus antenna, `lio_bad`: times of unhealthy LIO knots (sorted).
/// Returns which fixes to keep and the rejected segments.
pub fn screen(t: &[f64], o: &[V3], lio_bad: &[f64], p: &CheckParams) -> (Vec<bool>, Vec<Rejected>) {
    let n = t.len();
    if !p.enabled || n < 2 {
        return (vec![true; n], Vec::new());
    }
    let near_bad = |ta: f64, tb: f64| {
        let i = lio_bad.partition_point(|&x| x < ta - LIO_WINDOW);
        i < lio_bad.len() && lio_bad[i] <= tb + LIO_WINDOW
    };
    // segments [start, end) and the edges between them
    let mut segs = Vec::new();
    let mut edges = Vec::new();
    let mut i0 = 0;
    for i in 1..n {
        if t[i] - t[i - 1] > CONTINUOUS {
            segs.push((i0, i));
            edges.push(Edge::Soft);
            i0 = i;
            continue;
        }
        let d = o[i] - o[i - 1];
        let step = d.xy().norm() > p.step_h || d.z.abs() > p.step_v;
        if step && !near_bad(t[i - 1], t[i]) {
            segs.push((i0, i));
            edges.push(Edge::Hard(d));
            i0 = i;
        }
    }
    segs.push((i0, n));
    let ns = segs.len();
    let dur: Vec<f64> = segs.iter().map(|&(a, b)| t[b - 1] - t[a]).collect();
    // segment s began at the start or after an outage
    let after_outage = |s: usize| s == 0 || edges[s - 1] == Edge::Soft;
    let mut corr: Vec<Option<V3>> = vec![None; ns];
    let mut k = 0;
    while k < edges.len() {
        let Edge::Hard(j) = edges[k] else {
            k += 1;
            continue;
        };
        let (l, r) = (k, k + 1);
        if let Some(&Edge::Hard(j2)) = edges.get(r)
            && (j + j2).norm() <= (0.3 * j.norm()).max(p.step_h)
        {
            corr[r] = Some(-j);
            k += 2;
            continue;
        }
        if after_outage(l) && dur[l] <= LONGER * dur[r] {
            corr[l] = Some(j);
        } else {
            corr[r] = Some(-j);
        }
        k += 1;
    }
    // spread across short outages to neighbours closer to the wrong level than to the right one
    loop {
        let mut changed = false;
        for s in 0..ns {
            let Some(c) = corr[s] else { continue };
            for nb in [s.wrapping_sub(1), s + 1] {
                if nb >= ns || corr[nb].is_some() {
                    continue;
                }
                let e = s.min(nb);
                if edges[e] != Edge::Soft {
                    continue;
                }
                let (a, b) = (segs[e].1 - 1, segs[e + 1].0);
                if t[b] - t[a] > p.max_gap {
                    continue;
                }
                // offset of the neighbour relative to the wrong segment at their common outage
                let d = if nb > s { o[b] - o[a] } else { o[a] - o[b] };
                if d.norm() < (d - c).norm() {
                    corr[nb] = Some(c);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut keep = vec![true; n];
    let mut rejected = Vec::new();
    for (s, &(a, b)) in segs.iter().enumerate() {
        if let Some(c) = corr[s] {
            keep[a..b].iter_mut().for_each(|x| *x = false);
            rejected.push(Rejected {
                t0: t[a],
                t1: t[b - 1],
                n: b - a,
                corr: c,
            });
        }
    }
    (keep, rejected)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 10 Hz fixes with a slowly drifting offset (like the odometry), and an outage at 50..55 s.
    fn base() -> (Vec<f64>, Vec<V3>) {
        let t: Vec<f64> = (0..1200)
            .map(|i| i as f64 * 0.1)
            .filter(|&x| !(50.0..55.0).contains(&x))
            .collect();
        let o = t
            .iter()
            .map(|&x| V3::new(0.01 * x, -0.005 * x, 0.002 * (x * 0.3).sin()))
            .collect();
        (t, o)
    }

    fn add(t: &[f64], o: &mut [V3], a: f64, b: f64, d: V3) {
        for (x, y) in t.iter().zip(o.iter_mut()) {
            if (a..b).contains(x) {
                *y += d;
            }
        }
    }

    fn rejected_in(t: &[f64], keep: &[bool], a: f64, b: f64) -> (usize, usize) {
        let inside = t
            .iter()
            .zip(keep)
            .filter(|(x, k)| (a..b).contains(*x) && !**k)
            .count();
        let outside = t
            .iter()
            .zip(keep)
            .filter(|(x, k)| !(a..b).contains(*x) && !**k)
            .count();
        (inside, outside)
    }

    #[test]
    fn clean_fixes_are_kept() {
        let (t, o) = base();
        let (keep, rej) = screen(&t, &o, &[], &CheckParams::default());
        assert!(keep.iter().all(|k| *k) && rej.is_empty());
    }

    #[test]
    fn excursion_in_continuous_fix_is_rejected() {
        let (t, mut o) = base();
        add(&t, &mut o, 20.0, 30.0, V3::new(0.2, 0.0, 0.0));
        let (keep, _) = screen(&t, &o, &[], &CheckParams::default());
        assert_eq!(rejected_in(&t, &keep, 20.0, 30.0), (100, 0));
    }

    #[test]
    fn wrong_refix_after_outage_is_rejected_across_later_outages() {
        let (mut t, mut o) = base();
        // wrong from the re-fix at 55 s until 90 s, with a short outage at 70..72 s inside
        let keep_t: Vec<bool> = t.iter().map(|x| !(70.0..72.0).contains(x)).collect();
        let mut i = 0;
        t.retain(|_| {
            i += 1;
            keep_t[i - 1]
        });
        o = o
            .into_iter()
            .zip(&keep_t)
            .filter(|(_, k)| **k)
            .map(|(v, _)| v)
            .collect();
        add(&t, &mut o, 55.0, 90.0, V3::new(0.0, 0.5, 0.2));
        let (keep, rej) = screen(&t, &o, &[], &CheckParams::default());
        let n = t.iter().filter(|x| (55.0..90.0).contains(*x)).count();
        assert_eq!(rejected_in(&t, &keep, 55.0, 90.0), (n, 0));
        assert!(
            rej.iter()
                .all(|r| (r.corr - V3::new(0.0, -0.5, -0.2)).norm() < 0.02)
        );
    }

    #[test]
    fn step_at_an_unhealthy_lio_knot_is_not_a_gnss_jump() {
        let (t, mut o) = base();
        // the odometry jumps at 30 s and stays: every later offset moves
        add(&t, &mut o, 30.0, 1e9, V3::new(0.3, 0.0, 0.0));
        let (keep, _) = screen(&t, &o, &[29.95], &CheckParams::default());
        assert!(keep.iter().all(|k| *k));
    }

    #[test]
    fn small_steps_and_disabled_check_keep_everything() {
        let (t, mut o) = base();
        add(&t, &mut o, 20.0, 30.0, V3::new(0.1, 0.0, 0.2));
        assert!(
            screen(&t, &o, &[], &CheckParams::default())
                .0
                .iter()
                .all(|k| *k)
        );
        add(&t, &mut o, 20.0, 30.0, V3::new(1.0, 0.0, 0.0));
        let off = CheckParams {
            enabled: false,
            ..Default::default()
        };
        assert!(screen(&t, &o, &[], &off).0.iter().all(|k| *k));
    }
}
