//! The device's camera calibration (`s20 calibrate-cameras` in the GUI): solved from several of a
//! device's recordings, kept in a folder by the digest of the maker's calibration file (the raw
//! data carry no serial number, and each device has its own maker calibration), and used in place
//! of the maker's for its recordings (the export's image poses, the refinement and the colouring).

use crate::{IMG_POSE_FILE, RunConfig, export, file_digest, frames};
use anyhow::{Context, Result, ensure};
use s20_io::job::Job;
use std::path::{Path, PathBuf};

/// The device calibrations in folder `dir`: `<digest>.yaml` and what it was made from, `<digest>.json`.
pub struct Store {
    pub dir: PathBuf,
}

/// What a device calibration was made from.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Info {
    /// seconds since 1970
    pub made_at: u64,
    /// the work folders of the recordings
    pub recordings: Vec<PathBuf>,
    /// the cameras solved (the mounts' rotation, the principal points), as `s20 calibrate-cameras` reports them
    pub report: serde_json::Value,
}

impl Store {
    fn base(&self, maker: &Path) -> Option<PathBuf> {
        let d = file_digest(maker);
        (!d.is_empty()).then(|| self.dir.join(d))
    }

    /// The device calibration of the recordings with maker calibration `maker`, if there is one.
    pub fn find(&self, maker: &Path) -> Option<PathBuf> {
        self.base(maker)
            .map(|b| b.with_extension("yaml"))
            .filter(|p| p.is_file())
    }

    pub fn info(&self, maker: &Path) -> Option<Info> {
        let f = self.base(maker)?.with_extension("json");
        serde_json::from_slice(&std::fs::read(f).ok()?).ok()
    }

    pub fn save(&self, maker: &Path, calibration: &str, info: &Info) -> Result<PathBuf> {
        let base = self
            .base(maker)
            .with_context(|| format!("{} cannot be read", maker.display()))?;
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("create {}", self.dir.display()))?;
        std::fs::write(
            base.with_extension("json"),
            serde_json::to_vec_pretty(info)?,
        )?;
        let f = base.with_extension("yaml");
        std::fs::write(&f, calibration).with_context(|| format!("write {}", f.display()))?;
        Ok(f)
    }

    pub fn remove(&self, maker: &Path) -> Result<()> {
        if let Some(base) = self.base(maker) {
            for ext in ["yaml", "json"] {
                let f = base.with_extension(ext);
                if f.exists() {
                    std::fs::remove_file(&f).with_context(|| format!("remove {}", f.display()))?;
                }
            }
        }
        Ok(())
    }
}

/// Why a work folder cannot be used for the device calibration, if it cannot.
pub fn unusable(cfg: &RunConfig) -> Option<&'static str> {
    if cfg.sessions.is_empty() {
        return Some("セッションがありません");
    }
    if !cfg.calib().is_file() {
        return Some("生データにカメラの校正（info/calibration.yaml）がありません");
    }
    if !(cfg.done(crate::Step::Opt) && cfg.done(crate::Step::Cloud)) {
        return Some("点群生成まで済んでいません");
    }
    None
}

/// The device's cameras solved from the recordings of work folders `works` (one device: the same
/// maker calibration), with every image's pose: the calibration file (the maker's with the
/// cameras' changes) and a report.
pub fn calibrate(works: &[RunConfig], job: &Job) -> Result<(String, serde_json::Value)> {
    ensure!(!works.is_empty(), "no recordings");
    let maker = works[0].calib();
    let text = std::fs::read(&maker).with_context(|| maker.display().to_string())?;
    let cal = s20_io::calib::load(&maker)?;
    let mut recordings = Vec::new();
    for (i, cfg) in works.iter().enumerate() {
        if let Some(why) = unusable(cfg) {
            anyhow::bail!("{}: {why}", cfg.work.display());
        }
        ensure!(
            std::fs::read(cfg.calib())? == text,
            "{}: カメラの校正が最初の撮影と違います（別の機体）",
            cfg.work.display()
        );
        // the image poses with the maker's calibration (the export may have used a device calibration)
        let traj = cfg.opt_file();
        let tr = s20_geom::traj::Traj::from_rows(&s20_io::dataset::read_tum(&traj)?);
        let frame = frames::result_frame(&cfg.data(), &traj)?;
        let poses = export::camera_poses(
            &cfg.data(),
            &tr,
            &cal,
            frame.output_offset(),
            cal.lidar_imu_t,
        )?;
        let imgpose = cfg.runs().join(IMG_POSE_FILE.replace(".txt", "-maker.txt"));
        export::image_poses(&poses, &imgpose)?;
        let input = s20_visual::run::Input {
            data: cfg.session_data(0),
            imgpose,
            calib: maker.clone(),
            cloud: cfg.uncolored(),
            step: 12,
            holdout: 0.2,
            max_size: 1800,
            features: 8000,
            gate: 25.,
            search: 80.,
            ratio: 0.75,
            calib_out: None,
            camera_prior: s20_visual::run::CameraPrior::MAKER,
            tracks: None,
            out: PathBuf::new(),
        };
        job.log(&format!("{}: collecting observations", cfg.name));
        let part = job.part(
            0.9 * i as f64 / works.len() as f64,
            0.9 * (i + 1) as f64 / works.len() as f64,
        );
        recordings.push((cfg.name.clone(), s20_visual::run::collect(&input, &part)?));
    }
    job.log("solving the cameras with every image's pose");
    job.report("device calibration", 9, 10);
    let joint = s20_visual::joint::Joint::new(recordings);
    let (cameras, mut report) =
        s20_visual::joint::compare(&joint, true, true, false, &|s: &str| job.log(s));
    let cams = [
        s20_visual::camera::Fisheye::from(&cal.left),
        s20_visual::camera::Fisheye::from(&cal.right),
    ];
    let (written, error) =
        s20_visual::calibrate::write(&String::from_utf8(text)?, &cams, &cameras)?;
    report["source_calibration"] = serde_json::json!(maker);
    report["polynomial_refit_px"] = serde_json::json!(error);
    job.report("device calibration", 10, 10);
    Ok((written, report))
}
