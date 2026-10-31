//! Progress, cancellation and log messages of a long-running job, shared by the CLI and the GUI.

use anyhow::{Result, bail};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub type ProgressFn = dyn Fn(&str, u64, u64) + Send + Sync;
pub type LogFn = dyn Fn(&str) + Send + Sync;

#[derive(Clone)]
pub struct Job {
    pub cancel: Arc<AtomicBool>,
    progress: Arc<ProgressFn>,
    log: Arc<LogFn>,
}

/// The error a cancelled job returns.
#[derive(Debug)]
pub struct Cancelled;
impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}
impl std::error::Error for Cancelled {}

pub fn is_cancelled(e: &anyhow::Error) -> bool {
    e.downcast_ref::<Cancelled>().is_some()
}

impl Job {
    pub fn new(
        progress: impl Fn(&str, u64, u64) + Send + Sync + 'static,
        log: impl Fn(&str) + Send + Sync + 'static,
    ) -> Self {
        Job {
            cancel: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(progress),
            log: Arc::new(log),
        }
    }

    /// Log lines to stderr with the elapsed time, no progress display (the CLI).
    pub fn stderr() -> Self {
        let t0 = std::time::Instant::now();
        Job::new(
            |_, _, _| {},
            move |s| eprintln!("[{:7.1}s] {s}", t0.elapsed().as_secs_f64()),
        )
    }

    /// Fails with [`Cancelled`] once cancellation was requested.
    pub fn check(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            bail!(Cancelled);
        }
        Ok(())
    }

    pub fn report(&self, stage: &str, done: u64, total: u64) {
        (self.progress)(stage, done, total);
    }

    /// The same job whose progress takes the part `lo..hi` (fractions) of this one's: for a step
    /// made of a few pieces, each reporting from 0 to its end.
    pub fn part(&self, lo: f64, hi: f64) -> Job {
        const STEPS: f64 = 1e6;
        let parent = self.progress.clone();
        Job {
            cancel: self.cancel.clone(),
            progress: Arc::new(move |stage, done, total| {
                let f = if total > 0 {
                    done as f64 / total as f64
                } else {
                    0.0
                };
                parent(
                    stage,
                    ((lo + (hi - lo) * f.min(1.0)) * STEPS) as u64,
                    STEPS as u64,
                );
            }),
            log: self.log.clone(),
        }
    }

    /// The same job reporting all its progress under `stage`: the GUI follows a step by its
    /// name, while the pieces of the step report under their own.
    pub fn named(&self, stage: &str) -> Job {
        let parent = self.progress.clone();
        let stage = stage.to_string();
        Job {
            cancel: self.cancel.clone(),
            progress: Arc::new(move |_, done, total| parent(&stage, done, total)),
            log: self.log.clone(),
        }
    }

    pub fn log(&self, s: &str) {
        (self.log)(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn a_part_reports_within_its_share_and_shares_the_cancellation() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (s, all) = (seen.clone(), seen.clone());
        let job = Job::new(
            move |stage, done, total| {
                s.lock()
                    .unwrap()
                    .push((stage.to_string(), done as f64 / total as f64))
            },
            |_| {},
        );
        let part = job.part(0.9, 1.0);
        part.report("cloud", 0, 4);
        part.report("cloud", 2, 4);
        part.report("cloud", 4, 4);
        let seen = seen.lock().unwrap();
        let f: Vec<f64> = seen.iter().map(|s| s.1).collect();
        assert!(seen.iter().all(|s| s.0 == "cloud"));
        assert!(
            (f[0] - 0.9).abs() < 1e-9 && (f[1] - 0.95).abs() < 1e-9 && (f[2] - 1.0).abs() < 1e-9
        );
        job.cancel.store(true, Ordering::Relaxed);
        assert!(part.check().is_err());
        // a part of a named job reports under the name
        drop(seen);
        job.named("colorize")
            .part(0.0, 0.5)
            .report("colour images", 1, 2);
        let seen = all.lock().unwrap();
        assert_eq!(seen.last().unwrap().0, "colorize");
        assert!((seen.last().unwrap().1 - 0.25).abs() < 1e-9);
    }
}
