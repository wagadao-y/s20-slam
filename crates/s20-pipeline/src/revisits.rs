//! Revisits found and registered automatically in the optimisation (see `s20_opt::revisit`), and
//! the check of a trajectory by them: how doubled the points of two passes at the same place are.

use crate::Clock;
use anyhow::{Context, Result};
use s20_io::job::Job;
use s20_opt::Problem;
use s20_opt::revisit::{Revisit, RevisitParams};
use std::path::{Path, PathBuf};

/// Makes the marked loops robust, then finds the revisits on the placed sessions and adds the
/// accepted ones as factors. Returns how many loops were marked (the factors after them are the
/// revisits).
pub(crate) fn add(pr: &mut Problem, rp: &RevisitParams, clock: &Clock, job: &Job) -> usize {
    for l in &mut pr.loops {
        l.robust = Some(rp.robust);
    }
    let marked = pr.loops.len();
    let pairs = pr.revisit_pairs(rp);
    job.log(&format!(
        "revisits: {} pairs of {} s submaps within {} m",
        pairs.len(),
        rp.len_s,
        rp.max_dist
    ));
    let found = pr.register_revisits(rp, &pairs);
    summary(pr, &found, "revisits found", job);
    job.log(&format!(
        "revisits found: {} of the accepted along a path ({} times the standard deviations)",
        found
            .iter()
            .filter(|f| f.accepted && rp.along_path(f.weak))
            .count(),
        rp.path_sd
    ));
    for f in &found {
        if f.accepted && f.gap_m > 0.3 {
            job.log(&format!(
                "  revisit {} = {}: {:.2} m / {:.2} deg from the trajectory ({:.0}% matched)",
                label(pr, clock, f.rel.a),
                label(pr, clock, f.rel.b),
                f.gap_m,
                f.gap_deg,
                f.inliers * 100.0
            ));
        }
    }
    pr.loops
        .extend(found.iter().filter(|f| f.accepted).map(|f| {
            let mut r = f.rel;
            r.robust = Some(rp.robust);
            r
        }));
    marked
}

/// After the optimisation: the factors the robust weighting let go (residual beyond its scale),
/// the worst first.
pub(crate) fn report(pr: &Problem, rp: &RevisitParams, marked: usize, clock: &Clock, job: &Job) {
    let mut off: Vec<(f64, usize)> = pr
        .loops
        .iter()
        .enumerate()
        .map(|(i, l)| (l.chi2(&pr.states).sqrt(), i))
        .filter(|(s, _)| *s > rp.robust)
        .collect();
    off.sort_by(|a, b| b.0.total_cmp(&a.0));
    job.log(&format!(
        "revisits: {} of {} factors ({} marked loops) are beyond {} standard deviations and pull little",
        off.len(),
        pr.loops.len(),
        marked,
        rp.robust
    ));
    for (s, i) in off.iter().take(15) {
        let l = &pr.loops[*i];
        let (deg, m) = l.error(&pr.states);
        job.log(&format!(
            "  {} {} = {}: {:.0} mm / {:.3} deg ({s:.1} sd)",
            if *i < marked { "loop" } else { "revisit" },
            label(pr, clock, l.a),
            label(pr, clock, l.b),
            m * 1e3,
            deg
        ));
    }
}

fn label(pr: &Problem, clock: &Clock, k: usize) -> String {
    clock.label_at(pr.session[k], pr.kt[k])
}

/// The gaps of the accepted registrations, within sessions and across them.
fn summary(pr: &Problem, found: &[Revisit], what: &str, job: &Job) {
    for across in [false, true] {
        let mut g: Vec<f64> = found
            .iter()
            .filter(|f| f.accepted && (pr.session[f.rel.a] != pr.session[f.rel.b]) == across)
            .map(|f| f.gap_m * 1e3)
            .collect();
        let n = found
            .iter()
            .filter(|f| (pr.session[f.rel.a] != pr.session[f.rel.b]) == across)
            .count();
        g.sort_by(f64::total_cmp);
        let q = |x: f64| {
            g.get(((g.len() as f64 - 1.0) * x).round() as usize)
                .copied()
        };
        job.log(&format!(
            "{what} {}: {} of {n} registered; gap to the trajectory median {:.0} mm, 90% {:.0} mm, max {:.0} mm",
            if across { "across sessions" } else { "within a session" },
            g.len(),
            q(0.5).unwrap_or(f64::NAN),
            q(0.9).unwrap_or(f64::NAN),
            q(1.0).unwrap_or(f64::NAN),
        ));
    }
}

/// Checks a trajectory (`knots`, as the optimisation writes them) by its revisits: registers the
/// pairs of submaps (those of `pairs`, a file this wrote, else found on this trajectory) and
/// writes per pair how far the registration is from the trajectory to `out` (CSV).
pub fn eval(
    data: &Path,
    knots: &Path,
    out: &Path,
    pairs: Option<&Path>,
    rp: &RevisitParams,
    job: &Job,
) -> Result<()> {
    let l = s20_io::dataset::LidarData::open(data)?;
    let imu = s20_io::dataset::load_imu(data)?;
    let dirs = s20_io::dataset::sessions(data)?;
    let ranges = l.session_ranges();
    let rows = s20_io::dataset::read_table(knots)?;
    // the knots of each session (times on the common axis)
    let mut sessions = vec![(Vec::new(), Vec::new()); ranges.len()];
    for r in rows.iter().filter(|r| r.len() >= 17) {
        let k = s20_io::dataset::session_at(&ranges, r[0]);
        sessions[k]
            .0
            .push([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]]);
        sessions[k].1.push(vec![
            r[0], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15], r[16],
        ]);
    }
    let p = s20_opt::Params::default();
    let pr = Problem::new(&l, &imu, &sessions, &p)?;
    let clock = Clock {
        names: dirs
            .iter()
            .map(|d| s20_io::dataset::session_name(&d.dir))
            .collect(),
        range: ranges.clone(),
    };
    let pairs: Vec<(usize, usize)> = match pairs {
        Some(f) => read_pairs(f)?
            .into_iter()
            .filter_map(|(a, b)| Some((pr.knot_at(a)?, pr.knot_at(b)?)))
            .collect(),
        None => pr.revisit_pairs(rp),
    };
    let found = pr.register_revisits(rp, &pairs);
    summary(&pr, &found, "revisits", job);
    let mut s = String::from("ta,tb,across,inliers,spread_m,accepted,gap_mm,gap_deg,weak,a,b\n");
    for f in &found {
        let (a, b) = (f.rel.a, f.rel.b);
        s += &format!(
            "{:.3},{:.3},{},{:.3},{:.3},{},{:.1},{:.3},{:.4},{},{}\n",
            pr.kt[a],
            pr.kt[b],
            pr.session[a] != pr.session[b],
            f.inliers,
            f.spread_m,
            f.accepted,
            f.gap_m * 1e3,
            f.gap_deg,
            f.weak,
            label(&pr, &clock, a),
            label(&pr, &clock, b)
        );
    }
    std::fs::write(out, s).with_context(|| format!("write {}", out.display()))?;
    job.log(&format!("wrote {}", out.display()));
    Ok(())
}

/// The pair times (ta, tb) of a CSV that [`eval`] wrote.
fn read_pairs(f: &Path) -> Result<Vec<(f64, f64)>> {
    let s = std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
    s.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut c = l.split(',');
            let mut num =
                || -> Result<f64> { Ok(c.next().context("short line")?.trim().parse::<f64>()?) };
            Ok((num()?, num()?))
        })
        .collect()
}

/// Where [`eval`] would write by default: `<knots without _knots.txt>_revisits.csv`.
pub fn default_out(knots: &Path) -> PathBuf {
    PathBuf::from(
        knots
            .to_string_lossy()
            .replace("_knots.txt", "")
            .replace(".txt", "")
            + "_revisits.csv",
    )
}
