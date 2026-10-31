//! Panics on the worker threads turned into errors that say what went wrong and where. The GUI has
//! no console, so the default report on stderr is lost; the hook keeps the places of the recent
//! panics for the error made when one is caught (a panic on a rayon thread reaches the caller
//! with the same message, but the place is only known to the hook). A panic on the main thread
//! ends the program: the hook writes it to a crash log, which the next start points to.
use crate::lang::tl;
use s20_pipeline::runlog::PathMask;
use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};

/// The message and the place (`file:line`) of the recent panics on any thread, the newest last.
static RECENT: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
const KEPT: usize = 8;

/// Keeps the place of every panic (the default report still runs). Calls after the first do
/// nothing.
pub fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let place = info
                .location()
                .map_or(String::new(), |l| format!("{}:{}", l.file(), l.line()));
            let message = info.payload_as_str().unwrap_or_default().to_string();
            if std::thread::current().name() == Some("main") {
                write_crash(&message, &place);
            }
            if let Ok(mut recent) = RECENT.lock() {
                if recent.len() == KEPT {
                    recent.remove(0);
                }
                recent.push((message, place));
            }
            default(info);
        }));
    });
}

/// Where a crash log goes: the open work folder (its text), its `logs` folder and the masking of
/// its log files.
struct CrashTo {
    work: String,
    dir: PathBuf,
    mask: PathMask,
}

static CRASH_TO: Mutex<Option<CrashTo>> = Mutex::new(None);

/// The folder of the GUI's own files (settings, device calibrations).
fn app_dir() -> Option<PathBuf> {
    if cfg!(test) {
        Some(std::env::temp_dir().join(format!("s20_gui_app_{}", std::process::id())))
    } else {
        eframe::storage_dir("S20 SLAM")
    }
}

/// The file naming the last crash log, until the next start has shown it.
fn last_crash_file() -> Option<PathBuf> {
    app_dir().map(|d| d.join("last_crash.txt"))
}

/// A crash from now on is written in work folder `work` (its `logs` folder), its paths masked by
/// `mask`; with no work folder (`work` empty), in the GUI's own folder. Cheap when `work` is
/// unchanged: `mask` is only made for a new one.
pub fn crash_to(work: &str, mask: impl FnOnce() -> PathMask) {
    let Ok(mut to) = CRASH_TO.lock() else { return };
    if to.as_ref().is_some_and(|t| t.work == work) {
        return;
    }
    let dir = if work.is_empty() {
        app_dir().map(|d| d.join("logs"))
    } else {
        Some(Path::new(work).join("logs"))
    };
    *to = dir.map(|dir| CrashTo {
        work: work.to_string(),
        dir,
        mask: mask(),
    });
}

/// Writes the crash log of a panic on the main thread and names it for the next start. Nothing
/// here may panic again: a failed write is skipped.
fn write_crash(message: &str, place: &str) {
    // the lock is never held across a panic; `try_lock` all the same, not to wait in a crash
    let Some((dir, mask)) = CRASH_TO
        .try_lock()
        .ok()
        .and_then(|t| t.as_ref().map(|t| (t.dir.clone(), t.mask.clone())))
        .or_else(|| app_dir().map(|d| (d.join("logs"), PathMask::home())))
    else {
        return;
    };
    let stamp = s20_pipeline::runlog::stamp();
    let version = env!("CARGO_PKG_VERSION");
    let text = mask.apply(&format!("s20-gui {version} {stamp}\n{message}\n{place}\n"));
    let path = dir.join(format!("crash_{stamp}.log"));
    if std::fs::create_dir_all(&dir).is_err() || std::fs::write(&path, text).is_err() {
        return;
    }
    // the GUI's folder may not exist yet: eframe makes it when it first saves the settings
    if let Some(f) = last_crash_file()
        && f.parent()
            .is_some_and(|d| std::fs::create_dir_all(d).is_ok())
    {
        let _ = std::fs::write(f, path.to_string_lossy().as_bytes());
    }
}

/// The crash log of the last run, if it ended by a panic and that was not shown yet (shown once).
pub fn take_last_crash() -> Option<PathBuf> {
    let f = last_crash_file()?;
    let path = std::fs::read_to_string(&f).ok()?;
    let _ = std::fs::remove_file(&f);
    Some(PathBuf::from(path.trim()))
}

/// A caught panic: its message, with the place when the hook saw it.
#[derive(Debug)]
pub struct Panic {
    pub detail: String,
}

impl std::fmt::Display for Panic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = tl("内部エラー（panic）", "internal error (panic)");
        write!(f, "{what}: {}", self.detail)
    }
}

impl std::error::Error for Panic {}

impl Panic {
    fn of(payload: &(dyn Any + Send)) -> Self {
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        // the newest panic with this message (other threads may have panicked since)
        let place = RECENT
            .lock()
            .ok()
            .and_then(|mut r| {
                let i = r.iter().rposition(|(m, _)| *m == message)?;
                Some(r.remove(i).1)
            })
            .filter(|p| !p.is_empty());
        let message = if message.is_empty() {
            tl("（メッセージなし）", "(no message)").to_string()
        } else {
            message
        };
        Panic {
            detail: match place {
                Some(p) => format!("{message} ({p})"),
                None => message,
            },
        }
    }
}

/// Runs `f` on this thread; a panic in it comes back as the error.
pub fn catch<T>(f: impl FnOnce() -> T) -> Result<T, Panic> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|p| Panic::of(&*p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caught_with_message_and_place() {
        install();
        let e = catch(|| -> u32 { panic!("boom {}", 7) }).unwrap_err();
        assert!(e.detail.starts_with("boom 7 ("), "{}", e.detail);
        assert!(e.detail.contains("panics.rs:"), "{}", e.detail);
    }

    #[test]
    fn caught_from_a_rayon_thread() {
        use rayon::prelude::*;
        install();
        let e = catch(|| {
            (0..64).into_par_iter().for_each(|i| {
                if i == 63 {
                    panic!("worker failed");
                }
            })
        })
        .unwrap_err();
        assert!(e.detail.starts_with("worker failed ("), "{}", e.detail);
    }

    #[test]
    fn crash_log_in_the_work_folder_and_named_once() {
        let work = std::env::temp_dir().join(format!("s20_crash_work_{}", std::process::id()));
        let w = work.display().to_string();
        crash_to(&w, || PathMask::home().with(&work, ""));
        write_crash(&format!("cannot read {w}\\output\\a.laz"), "src/x.rs:3");
        let log = take_last_crash().unwrap();
        assert_eq!(log.parent(), Some(work.join("logs").as_path()));
        let text = std::fs::read_to_string(&log).unwrap();
        assert!(
            text.contains("cannot read output\\a.laz\nsrc/x.rs:3"),
            "{text}"
        );
        assert!(take_last_crash().is_none());
        let _ = std::fs::remove_dir_all(&work);
        let _ = std::fs::remove_dir_all(app_dir().unwrap());
    }

    #[test]
    fn no_panic_passes_the_value() {
        assert_eq!(catch(|| 5).unwrap(), 5);
    }
}
