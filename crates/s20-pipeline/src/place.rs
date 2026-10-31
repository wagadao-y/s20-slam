//! Placing the sessions in a common frame before the optimisation: from the GNSS fixes, or from
//! the links to a session already placed (two times at the same place, or matched points). See
//! `s20_opt::links`.

use crate::{At, Clock, PointAt};
use anyhow::{Context, Result, bail};
use s20_geom::so3::{self, M3, V3};
use s20_io::job::Job;
use s20_opt::Problem;
use s20_opt::links::{place_points, yaw_only};
use s20_opt::relpose::RelPose;

/// A picked point on the trajectory: knot `k` of session `s`, `x` in the body of that knot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pick {
    pub s: usize,
    pub k: usize,
    pub x: V3,
}

impl Pick {
    /// From a point as stored: in the body at its time on its session (the pose interpolated
    /// between the knots), or, without a time, in the session's world as given (its odometry: the
    /// current states, before any placement).
    pub fn resolve(p: &PointAt, pr: &Problem, clock: &Clock) -> Result<Pick> {
        let s = clock.session(&p.session)?;
        let v = V3::from(p.p);
        let (k, w) = match p.t {
            Some(t) => {
                let at = At::on(&p.session, t);
                let ta = clock.abs(&at)?;
                let k = pr
                    .knot_at(ta)
                    .filter(|&k| pr.session[k] == s)
                    .with_context(|| format!("{}: outside the trajectory", clock.label(&at)))?;
                let (a, b) = (&pr.states[k], pr.states.get(k + 1));
                let (r, pos) = match b {
                    Some(b) if pr.session.get(k + 1) == Some(&s) => {
                        let u = ((ta - pr.kt[k]) / (pr.kt[k + 1] - pr.kt[k])).clamp(0.0, 1.0);
                        (
                            a.r * so3::exp(&(so3::log(&(a.r.transpose() * b.r)) * u)),
                            a.p + (b.p - a.p) * u,
                        )
                    }
                    _ => (a.r, a.p),
                };
                (k, r * v + pos)
            }
            None => (
                pr.nearest_knot(s, &v).context("a session without knots")?,
                v,
            ),
        };
        let st = &pr.states[k];
        Ok(Pick {
            s,
            k,
            x: st.r.transpose() * (w - st.p),
        })
    }

    pub fn world(&self, pr: &Problem) -> V3 {
        let st = &pr.states[self.k];
        st.r * self.x + st.p
    }
}

/// Matched points of two sessions (side a, side b).
pub(crate) struct Matches {
    pub a: usize,
    pub b: usize,
    pub pairs: Vec<(Pick, Pick)>,
}

impl Matches {
    /// The matched points grouped by pair of sessions (in the order given).
    pub fn group(picks: Vec<(Pick, Pick)>) -> Result<Vec<Matches>> {
        let mut out: Vec<Matches> = Vec::new();
        for (x, y) in picks {
            if x.s == y.s {
                bail!("matched points on the same session (use a loop)");
            }
            match out
                .iter_mut()
                .find(|m| (m.a, m.b) == (x.s, y.s) || (m.a, m.b) == (y.s, x.s))
            {
                Some(m) if m.a == x.s => m.pairs.push((x, y)),
                Some(m) => m.pairs.push((y, x)),
                None => out.push(Matches {
                    a: x.s,
                    b: y.s,
                    pairs: vec![(x, y)],
                }),
            }
        }
        Ok(out)
    }

    /// The centres of the matched points (world, current states).
    fn centres(&self, pr: &Problem) -> (V3, V3) {
        let n = self.pairs.len() as f64;
        let ca = self.pairs.iter().map(|p| p.0.world(pr)).sum::<V3>() / n;
        let cb = self.pairs.iter().map(|p| p.1.world(pr)).sum::<V3>() / n;
        (ca, cb)
    }

    /// The initial transforms b -> a: from two pairs or more the yaw and translation, from one the
    /// translation with yaws every 10 degrees (the alignment starts with correspondences within
    /// 2 m: 5 degrees is 1 m at the edge of the region).
    fn guesses(&self, pr: &Problem) -> Vec<(M3, V3)> {
        let a: Vec<V3> = self.pairs.iter().map(|p| p.0.world(pr)).collect();
        let b: Vec<V3> = self.pairs.iter().map(|p| p.1.world(pr)).collect();
        if a.len() >= 2 {
            let (r, t) = place_points(&a, &b);
            // and a few degrees either way: points picked by hand are a few centimetres off
            [0.0f64, 3.0, -3.0]
                .iter()
                .map(|d| {
                    let rz = so3::exp(&V3::new(0.0, 0.0, d.to_radians()));
                    let ca = a.iter().sum::<V3>() / a.len() as f64;
                    (rz * r, rz * (t - ca) + ca)
                })
                .collect()
        } else {
            (0..36)
                .map(|i| {
                    let rz = so3::exp(&V3::new(0.0, 0.0, (i as f64 * 10.0).to_radians()));
                    (rz, a[0] - rz * b[0])
                })
                .collect()
        }
    }
}

/// Registers the regions of matched points: the transform b -> a, logged.
fn register_matches(
    pr: &Problem,
    m: &Matches,
    guesses: &[(M3, V3)],
    p: &s20_opt::Params,
    name: &str,
    job: &Job,
) -> Result<(M3, V3)> {
    let (ca, cb) = m.centres(pr);
    let f = pr.register_regions(m.a, &ca, m.b, &cb, guesses, &p.loop_reg, p.rmax)?;
    job.log(&format!(
        "{name}: matched {:.0}% of the points, guesses {:.2} m apart",
        f.inliers * 100.0,
        f.spread_m
    ));
    if let Some((d, ratio)) = f.weak {
        job.log(&format!(
            "{name}: WARNING the points hardly fix the direction ({:.2}, {:.2}, {:.2}) (weight {:.3} of the strongest): pick points that also fix it, or check the placement",
            d.x, d.y, d.z, ratio
        ));
    }
    if !f.accepted {
        bail!(
            "{name}: the regions do not match ({:.0}% of the points, guesses {:.2} m apart); pick other points",
            f.inliers * 100.0,
            f.spread_m
        );
    }
    Ok((f.r, f.t))
}

/// Moves session `s` by the yaw and translation of the transform (r, t) (the sessions agree on
/// gravity), about the world point `c`.
fn place(pr: &mut Problem, s: usize, r: &M3, t: &V3, c: &V3) -> (f64, f64) {
    let ry = yaw_only(r);
    // c maps where the full transform takes it
    let ty = r * c + t - ry * c;
    pr.move_session(s, &ry, &ty);
    pr.free_session(s);
    (so3::log(&ry).z.to_degrees(), ty.norm())
}

/// Places the sessions not `placed` (by GNSS, or the first) through the links to placed ones:
/// loops between sessions and matched points. Returns the factors of the matched points, measured
/// once all are placed.
pub(crate) fn place_sessions(
    pr: &mut Problem,
    clock: &Clock,
    mut placed: Vec<bool>,
    loops: &[(At, At)],
    matches: &[Matches],
    p: &s20_opt::Params,
    job: &Job,
) -> Result<Vec<RelPose>> {
    let name = |k: usize| clock.names[k].clone();
    let cross: Vec<(usize, usize, &(At, At))> = loops
        .iter()
        .map(|l| {
            Ok((
                clock.session(&l.0.session)?,
                clock.session(&l.1.session)?,
                l,
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|(a, b, _)| a != b)
        .collect();
    loop {
        let mut progress = false;
        for &(sa, sb, l) in &cross {
            if placed[sa] == placed[sb] {
                continue;
            }
            job.check()?;
            let (ta, tb) = (clock.abs(&l.0)?, clock.abs(&l.1)?);
            let what = format!("loop {} = {}", clock.label(&l.0), clock.label(&l.1));
            // the loop measured from the placed side
            let f = if placed[sa] {
                pr.register_loop(ta, tb, &p.loop_reg)?
            } else {
                pr.register_loop(tb, ta, &p.loop_reg)?
            };
            if !s20_opt::links::loop_accepted(&f, true, &p.loop_reg) {
                bail!(
                    "{what}: the local maps do not match ({:.0}% of the points, guesses {:.2} m apart); the session cannot be placed by it",
                    f.inliers * 100.0,
                    f.spread_m
                );
            }
            let (r, t) = pr.link_transform(&f.rel);
            let s = if placed[sa] { sb } else { sa };
            let c = pr.states[f.rel.b].p;
            let (yaw, d) = place(pr, s, &r, &t, &c);
            job.log(&format!(
                "session {} placed by {what} ({:.0}% matched): turned {yaw:.1} deg, moved {d:.2} m",
                name(s),
                f.inliers * 100.0
            ));
            placed[s] = true;
            progress = true;
        }
        for m in matches {
            if placed[m.a] == placed[m.b] {
                continue;
            }
            job.check()?;
            let what = format!("matched points {} - {}", name(m.a), name(m.b));
            let g = m.guesses(pr);
            let (r, t) = if placed[m.a] {
                register_matches(pr, m, &g, p, &what, job)?
            } else {
                // place a onto b: the inverse transforms
                let rev = Matches {
                    a: m.b,
                    b: m.a,
                    pairs: m.pairs.iter().map(|(x, y)| (*y, *x)).collect(),
                };
                let g = rev.guesses(pr);
                register_matches(pr, &rev, &g, p, &what, job)?
            };
            let s = if placed[m.a] { m.b } else { m.a };
            let (ca, cb) = m.centres(pr);
            let c = if s == m.b { cb } else { ca };
            let (yaw, d) = place(pr, s, &r, &t, &c);
            job.log(&format!(
                "session {} placed by {what}: turned {yaw:.1} deg, moved {d:.2} m",
                name(s)
            ));
            placed[s] = true;
            progress = true;
        }
        if !progress {
            break;
        }
    }
    let missing: Vec<String> = (0..placed.len())
        .filter(|&k| !placed[k])
        .map(name)
        .collect();
    if !missing.is_empty() {
        bail!(
            "sessions {} are not placed: without GNSS fixes, link each one to a placed session by a loop (two times at the same place) or matched points",
            missing.join(", ")
        );
    }
    // the matched points as factors, measured on the placed sessions
    let mut out = Vec::new();
    for m in matches {
        job.check()?;
        let what = format!("matched points {} - {}", name(m.a), name(m.b));
        let (r, t) = register_matches(pr, m, &[(M3::identity(), V3::zeros())], p, &what, job)?;
        let (ca, cb) = m.centres(pr);
        let ka = pr.nearest_knot(m.a, &ca).context("no knot")?;
        let kb = pr.nearest_knot(m.b, &cb).context("no knot")?;
        let mut rel = pr.link_factor(ka, kb, &r, &t, &p.loop_reg);
        let (sa, sb) = (&pr.states[ka], &pr.states[kb]);
        rel.region = Some((
            sa.r.transpose() * (ca - sa.p),
            sb.r.transpose() * (cb - sb.p),
        ));
        job.log(&format!(
            "{what}: {:.1} mm / {:.3} deg from the current placement",
            (r * cb + t - cb).norm() * 1e3,
            so3::log(&r).norm().to_degrees()
        ));
        out.push(rel);
    }
    Ok(out)
}
