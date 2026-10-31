//! Panics on the worker threads turned into errors that say what went wrong and where. The GUI has
//! no console, so the default report on stderr is lost; the hook keeps the places of the recent
//! panics for the error made when one is caught (a panic on a rayon thread reaches the caller
//! with the same message, but the place is only known to the hook).
use crate::lang::tl;
use std::any::Any;
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
    fn no_panic_passes_the_value() {
        assert_eq!(catch(|| 5).unwrap(), 5);
    }
}
