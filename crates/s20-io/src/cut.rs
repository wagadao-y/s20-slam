//! A part of a raw folder as a raw folder of its own: short recordings for tests, cut from long
//! ones whose results are known. Only `all_*.bag` is cut (by record time); the other files the
//! processing reads (`info/`, `frame_pose.txt`, `BASE.BIN`, `ROVER_PPKRAW.bin`, ...) are looked up
//! by time or keep only the scans' span, and are copied whole.

use crate::bag::{Bag, BagWriter};
use crate::extract::find_bag;
use crate::job::Job;
use anyhow::{Context, Result, bail, ensure};
use std::fs;
use std::path::Path;

pub struct CutStats {
    pub messages: usize,
    pub bytes: u64,
    /// the record times of the first and last message written, seconds from the first scan
    pub span: [f64; 2],
}

/// Writes the messages of `raw`'s bag recorded from `from` to `to` seconds after its first scan
/// into `out` (a new or empty folder), under the same name, and copies the other files: those at
/// the top of `raw` and the `info` folder. The LiDAR odometry starts from the first seconds being
/// still, so `from` should fall a few seconds before the device starts moving.
pub fn cut_raw(raw: &Path, out: &Path, from: f64, to: f64, job: &Job) -> Result<CutStats> {
    ensure!(from >= 0.0 && to > from, "the span {from}..{to} s is empty");
    let src = find_bag(raw)?;
    if out.exists() && fs::read_dir(out)?.next().is_some() {
        bail!("{} is not empty", out.display());
    }
    fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let bag = Bag::open(&src)?;
    let t0 = bag
        .topic_messages("/livox/lidar")
        .first()
        .map(|m| m.time_ns)
        .context("no /livox/lidar messages")?;
    let at = |s: f64| t0 + (s * 1e9) as u64;
    let (lo, hi) = (at(from), at(to));
    let name = src.file_name().context("bag without a name")?;
    let mut w = BagWriter::create(&out.join(name))?;
    let total = bag.messages.len() as u64;
    let (mut messages, mut bytes, mut span) = (0usize, 0u64, [f64::INFINITY, f64::NEG_INFINITY]);
    for (i, m) in bag.messages.iter().enumerate() {
        if i % 10_000 == 0 {
            job.check()?;
            job.report("cut", i as u64, total);
        }
        if m.time_ns < lo || m.time_ns > hi {
            continue;
        }
        w.write(&bag, m)?;
        let s = (m.time_ns - t0) as f64 * 1e-9;
        span = [span[0].min(s), span[1].max(s)];
        messages += 1;
        bytes += m.len as u64;
    }
    w.finish()?;
    ensure!(messages > 0, "no messages from {from} to {to} s");
    // the other files, whole
    for e in fs::read_dir(raw)? {
        let p = e?.path();
        if p == src {
            continue;
        }
        if p.is_file() {
            fs::copy(&p, out.join(p.file_name().context("file without a name")?))
                .with_context(|| format!("copy {}", p.display()))?;
        }
    }
    let info = raw.join("info");
    if info.is_dir() {
        fs::create_dir_all(out.join("info"))?;
        for e in fs::read_dir(&info)? {
            let p = e?.path();
            if p.is_file() {
                fs::copy(&p, out.join("info").join(p.file_name().context("no name")?))
                    .with_context(|| format!("copy {}", p.display()))?;
            }
        }
    }
    Ok(CutStats {
        messages,
        bytes,
        span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bag::BagWriter;

    #[test]
    fn empty_span_and_non_empty_output_refused() {
        let d = std::env::temp_dir().join(format!("s20_cut_{}", std::process::id()));
        let (raw, out) = (d.join("raw"), d.join("out"));
        fs::create_dir_all(&raw).unwrap();
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join("x"), b"x").unwrap();
        let job = Job::new(|_, _, _| {}, |_| {});
        assert!(cut_raw(&raw, &out, 5.0, 5.0, &job).is_err());
        // an empty bag (no messages) is enough to reach the check of the output
        BagWriter::create(&raw.join("all_x.bag"))
            .unwrap()
            .finish()
            .unwrap();
        let e = cut_raw(&raw, &out, 0.0, 5.0, &job).err().unwrap();
        assert!(format!("{e:#}").contains("is not empty"), "{e:#}");
        fs::remove_dir_all(&d).unwrap();
    }
}
