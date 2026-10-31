//! The processing log kept as a file: one per run, `<work folder>/logs/<local time>.log`, written
//! line by line as the run goes (a run that stops or fails keeps what it got to), with a header
//! saying what ran and with which settings. Its paths are masked ([`PathMask`]), so that the log
//! can be shown to others without the user's name and folders.

use crate::{RunConfig, file_digest};
use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct LogFile {
    pub path: PathBuf,
    file: Mutex<std::fs::File>,
    mask: PathMask,
}

/// Rewrites the paths in log lines: those in the given folders relative to them or under a label,
/// and the rest of the user's folder as `~`.
#[derive(Clone)]
pub struct PathMask {
    /// (folder, label), the longest folder first; an empty label: relative to the folder
    rules: Vec<(Vec<char>, String)>,
}

impl PathMask {
    /// Only the user's folder (`%USERPROFILE%`) masked, as `~`.
    pub fn home() -> Self {
        let m = PathMask { rules: Vec::new() };
        match std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            Some(h) => m.with(Path::new(&h), "~"),
            None => m,
        }
    }

    /// The paths in `dir` written as `<label>\...`, or relative to `dir` when `label` is empty.
    pub fn with(mut self, dir: &Path, label: &str) -> Self {
        let d: Vec<char> = dir
            .to_string_lossy()
            .trim_end_matches(['\\', '/'])
            .chars()
            .collect();
        if !d.is_empty() {
            self.rules.push((d, label.to_string()));
            self.rules.sort_by_key(|r| std::cmp::Reverse(r.0.len()));
        }
        self
    }

    /// `s` with its paths rewritten. The folders match whatever the letter case and `/` or `\`.
    pub fn apply(&self, s: &str) -> String {
        let sep = |c: char| c == '\\' || c == '/';
        // a match must not end inside a name: `rec_01` is not the start of `rec_01b`
        let in_name = |c: char| c.is_alphanumeric() || matches!(c, '.' | '_' | '-');
        let c: Vec<char> = s.chars().collect();
        let mut out = String::with_capacity(s.len());
        let mut i = 0;
        'next: while i < c.len() {
            for (d, label) in &self.rules {
                let end = i + d.len();
                if end > c.len() || c.get(end).is_some_and(|&n| in_name(n)) {
                    continue;
                }
                let same = d
                    .iter()
                    .zip(&c[i..end])
                    .all(|(&a, &b)| a.eq_ignore_ascii_case(&b) || (sep(a) && sep(b)));
                if same {
                    let then_sep = c.get(end).is_some_and(|&n| sep(n));
                    i = end;
                    if !label.is_empty() {
                        out.push_str(label);
                    } else if then_sep {
                        i += 1;
                    } else {
                        out.push('.');
                    }
                    continue 'next;
                }
            }
            out.push(c[i]);
            i += 1;
        }
        out
    }
}

/// The local time as in the log files' names (`2026-10-28_15-42-10`).
pub fn stamp() -> String {
    chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string()
}

impl LogFile {
    /// A new log file in folder `dir`, named by the local time (`_2`, `_3` added when taken), its
    /// paths masked by `mask`.
    pub fn create(dir: &Path, mask: PathMask) -> Result<Self> {
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
            mask,
        })
    }

    /// Appends a line with its paths masked, at once (a failed write is ignored: the log must not
    /// stop the processing). Returns the line as written, for the same log shown on screen.
    pub fn line(&self, s: &str) -> String {
        let s = self.mask.apply(s);
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{s}");
            let _ = f.flush();
        }
        s
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
        "パス: 作業フォルダーの中は作業フォルダーから、生データフォルダーの中は <raw:セッション名> から"
            .to_string(),
        format!(
            "セッション: {}",
            cfg.sessions
                .iter()
                .map(|s| s.name.clone())
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

    /// The paths of a run's log: relative to the work folder, the raw folders as
    /// `<raw:session name>`, the rest of the user's folder as `~`.
    pub fn path_mask(&self) -> PathMask {
        self.sessions
            .iter()
            .fold(PathMask::home().with(&self.work, ""), |m, s| {
                m.with(&s.raw, &format!("<raw:{}>", s.name))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_relative_to_the_work_folder_or_labelled() {
        let m = PathMask { rules: Vec::new() }
            .with(Path::new(r"E:\home\someone"), "~")
            .with(Path::new(r"E:\home\someone\scans\work"), "")
            .with(Path::new(r"D:/raw/rec_01\"), "<raw:rec_01>");
        assert_eq!(
            m.apply(r"wrote E:\home\someone\scans\work\output\a.laz (5 points)"),
            r"wrote output\a.laz (5 points)"
        );
        assert_eq!(m.apply("done: e:/HOME/someone/scans/work"), "done: .");
        assert_eq!(
            m.apply(r"read D:\raw\rec_01\info\calibration.yaml"),
            r"read <raw:rec_01>\info\calibration.yaml"
        );
        assert_eq!(
            m.apply(r"E:\home\someone\AppData\x.yaml, E:\home\someone2\a"),
            r"~\AppData\x.yaml, E:\home\someone2\a"
        );
        // a folder whose name only starts the same
        assert_eq!(m.apply(r"D:\raw\rec_01b\x"), r"D:\raw\rec_01b\x");
    }
}
