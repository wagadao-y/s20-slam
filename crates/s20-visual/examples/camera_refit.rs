//! Solve the cameras' changes together with every image's pose over several recordings, and check
//! each recording with the values fitted on the others:
//! `camera_refit <manifest.json> <report.json> [intrinsic|mount|both]`; the manifest lists
//! `{"name", "data", "imgpose", "calib", "cloud"}` per recording (all with the same calibration).

use anyhow::Result;
use s20_io::job::Job;
use s20_visual::joint::{Joint, compare};
use s20_visual::run::{Input, collect};
use serde_json::Value;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(3).map(String::as_str).unwrap_or("intrinsic");
    let (fit_camera, fit_mount) = match mode {
        "intrinsic" => (true, false),
        "mount" => (false, true),
        "both" => (true, true),
        _ => anyhow::bail!("mode: intrinsic, mount or both"),
    };
    let manifest: Vec<Value> = serde_json::from_slice(&std::fs::read(&a[1])?)?;
    let job = Job::stderr();
    let t0 = Instant::now();
    let mut recordings = Vec::new();
    for m in &manifest {
        let s = |k: &str| PathBuf::from(m[k].as_str().unwrap());
        let input = Input {
            data: s("data"),
            imgpose: s("imgpose"),
            calib: s("calib"),
            cloud: s("cloud"),
            step: 12,
            out: PathBuf::new(),
            holdout: 0.2,
            max_size: 1800,
            features: 8000,
            gate: 25.,
            search: 80.,
            ratio: 0.75,
            calib_out: None,
            camera_prior: s20_visual::run::CameraPrior::MAKER,
            tracks: None,
        };
        let c = collect(&input, &job)?;
        eprintln!(
            "[{:7.1}s] {}: {} observations",
            t0.elapsed().as_secs_f64(),
            m["name"],
            c.obs.len()
        );
        recordings.push((m["name"].as_str().unwrap().to_string(), c));
    }
    let joint = Joint::new(recordings);
    let log = |s: &str| eprintln!("[{:7.1}s] {s}", t0.elapsed().as_secs_f64());
    let (_, report) = compare(&joint, fit_camera, fit_mount, true, &log);
    std::fs::write(&a[2], serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
