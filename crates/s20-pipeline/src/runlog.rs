//! The processing log kept as a file: one per run, `<work folder>/logs/<local time>.log`, written
//! line by line as the run goes (a run that stops or fails keeps what it got to), with a header
//! saying what ran and with which settings.

use crate::{RunConfig, file_digest};
use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct LogFile {
    pub path: PathBuf,
    file: Mutex<std::fs::File>,
}

/// The local time as in the log files' names (`2026-10-28_15-42-10`).
pub fn stamp() -> String {
    chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string()
}

impl LogFile {
    /// A new log file in folder `dir`, named by the local time (`_2`, `_3` added when taken).
    pub fn create(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let base = stamp();
        let mut path = dir.join(format!("{base}.log"));
        let mut n = 2;
        while path.exists() {
            path = dir.join(format!("{base}_{n}.log"));
            n += 1;
        }
        let file =
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        Ok(LogFile {
            path,
            file: Mutex::new(file),
        })
    }

    /// Appends a line, at once (a failed write is ignored: the log must not stop the processing).
    pub fn line(&self, s: &str) {
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{s}");
            let _ = f.flush();
        }
    }
}

/// The lines a run's log starts with: the version, the time, the work folder and sessions, what
/// runs and the settings.
pub fn header(cfg: &RunConfig, what: &str) -> Vec<String> {
    let gnss = if cfg.ppk {
        "PPK"
    } else if cfg.rtk {
        "RTK（受信機）"
    } else {
        "なし"
    };
    let on = |b: bool| if b { "オン" } else { "オフ" };
    let device = match &cfg.device_calib {
        Some(d) => format!("{} ({})", d.display(), file_digest(d)),
        None => "なし（姿勢合わせで撮影ごとにカメラを求める）".to_string(),
    };
    let maker = if cfg.sessions.is_empty() {
        PathBuf::new()
    } else {
        cfg.calib()
    };
    vec![
        format!("s20 {}", env!("CARGO_PKG_VERSION")),
        format!("開始: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")),
        format!("作業フォルダー: {}", cfg.work.display()),
        format!(
            "セッション: {}",
            cfg.sessions
                .iter()
                .map(|s| format!("{} ({})", s.name, s.raw.display()))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        format!("実行: {what}"),
        format!("GNSS: {gnss}"),
        format!(
            "メーカーの校正: {} ({})",
            maker.display(),
            file_digest(&maker)
        ),
        format!("機体の校正: {device}"),
        format!("画像の姿勢合わせ: {}", on(cfg.refine)),
        format!("画像を出力: {}", on(cfg.images)),
        format!("点群の形式: {}", if cfg.laz { "LAZ" } else { "LAS" }),
        format!(
            "パラメータ: {}",
            serde_json::to_string(&cfg.params).unwrap_or_default()
        ),
        String::new(),
    ]
}

impl RunConfig {
    /// The folder of the run logs.
    pub fn log_dir(&self) -> PathBuf {
        self.work.join("logs")
    }
}
