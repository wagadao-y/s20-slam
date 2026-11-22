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
    let bag = Bag::open(&src)?;
    let t0 = bag
        .topic_messages("/livox/lidar")
        .first()
        .map(|m| m.time_ns)
        .context("no /livox/lidar messages")?;
    // (a time past the recording's end, even infinite, is its end)
    let at = |s: f64| t0.saturating_add((s * 1e9) as u64);
    let (lo, hi) = (at(from), at(to));
    // (checked before writing: a span outside the recording, or without the scans and IMU
    // samples a processing needs, leaves the folder as it was)
    for topic in ["/livox/lidar", "/livox/imu"] {
        ensure!(
            bag.topic_messages(topic)
                .iter()
                .any(|m| (lo..=hi).contains(&m.time_ns)),
            "no {topic} messages from {from} to {to} s"
        );
    }
    fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    // written into a folder of this run's own inside `out` (made new), moved up once complete:
    // canceled or failed, only that folder goes, nothing another run or the user put in `out`
    let mut n = 0u32;
    let stage = loop {
        let d = out.join(format!(".cut-{}-{n}", std::process::id()));
        match fs::create_dir(&d) {
            Ok(()) => break d,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => n += 1,
            Err(e) => {
                return Err(anyhow::Error::from(e).context(format!("create {}", d.display())));
            }
        }
    };
    let (mut messages, mut bytes, mut span) = (0usize, 0u64, [f64::INFINITY, f64::NEG_INFINITY]);
    let written = (|| -> Result<()> {
        let name = src.file_name().context("bag without a name")?;
        let mut w = BagWriter::create(&stage.join(name))?;
        let total = bag.messages.len() as u64;
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
        job.check()?;
        // the other files, whole
        for e in fs::read_dir(raw)? {
            let p = e?.path();
            if p == src {
                continue;
            }
            if p.is_file() {
                fs::copy(
                    &p,
                    stage.join(p.file_name().context("file without a name")?),
                )
                .with_context(|| format!("copy {}", p.display()))?;
            }
        }
        let info = raw.join("info");
        if info.is_dir() {
            fs::create_dir_all(stage.join("info"))?;
            for e in fs::read_dir(&info)? {
                let p = e?.path();
                if p.is_file() {
                    fs::copy(
                        &p,
                        stage.join("info").join(p.file_name().context("no name")?),
                    )
                    .with_context(|| format!("copy {}", p.display()))?;
                }
            }
        }
        job.check()?;
        // (still empty but for this run's folder: else another run or the user wrote into it)
        let others = fs::read_dir(out)?
            .filter_map(|e| e.ok())
            .any(|e| e.path() != stage);
        ensure!(!others, "{} is no longer empty", out.display());
        for e in fs::read_dir(&stage)? {
            let p = e?.path();
            let to = out.join(p.file_name().context("no name")?);
            // (a file come there meanwhile is not replaced: the rename would)
            ensure!(!to.exists(), "{} came there meanwhile", to.display());
            fs::rename(&p, &to).with_context(|| format!("move {}", p.display()))?;
        }
        fs::remove_dir(&stage).with_context(|| format!("remove {}", stage.display()))
    })();
    if let Err(e) = written {
        let _ = fs::remove_dir_all(&stage);
        return Err(e);
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
