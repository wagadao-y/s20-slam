//! The coordinate frame of each result, kept in one file of the data folder ([`FILE`]) so that an
//! old result keeps its own frame. An entry holds for the file as it was recorded (its size and
//! modification time); a result without one is in the legacy projected frame of the data folder.
//! Results are named relative to the data folder's parent (the work folder) when under it.

use anyhow::{Context, Result};
use s20_io::geodesy::CoordinateFrame;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const FILE: &str = "result_frames.json";
const LOCK: &str = "result_frames.lock";

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Entry {
    frame: CoordinateFrame,
    bytes: u64,
    /// nanoseconds since the Unix epoch
    modified: u128,
}

type Ledger = BTreeMap<String, Entry>;

fn stamp(path: &Path) -> Result<(u64, u128)> {
    let m = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    Ok((
        m.len(),
        m.modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
    ))
}

fn key(data: &Path, result: &Path) -> Result<String> {
    let result = std::path::absolute(result)?;
    let data = std::path::absolute(data)?;
    let base = data.parent().unwrap_or(&data);
    let p = result.strip_prefix(base).unwrap_or(&result);
    Ok(p.to_string_lossy().replace('\\', "/"))
}

fn load(data: &Path) -> Result<Ledger> {
    let path = data.join(FILE);
    if !path.exists() {
        return Ok(Ledger::new());
    }
    serde_json::from_slice(&fs::read(&path)?).with_context(|| format!("read {}", path.display()))
}

fn save(data: &Path, ledger: &Ledger) -> Result<()> {
    let path = data.join(FILE);
    let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    fs::create_dir_all(data)?;
    fs::write(&tmp, serde_json::to_vec_pretty(ledger)?)?;
    fs::rename(&tmp, &path).with_context(|| format!("write {}", path.display()))
}

/// The sidecar `<result>.frame.json` that held a result's frame before the ledger.
fn sidecar(result: &Path) -> PathBuf {
    PathBuf::from(format!("{}.frame.json", result.display()))
}

/// The frame `result` was recorded in, while it is the file recorded (else none); without an
/// entry, that of its old sidecar.
pub fn recorded(data: &Path, result: &Path) -> Result<Option<CoordinateFrame>> {
    if let Some(e) = load(data)?.get(&key(data, result)?) {
        let current = result.exists() && stamp(result)? == (e.bytes, e.modified);
        return Ok(current.then_some(e.frame));
    }
    let old = sidecar(result);
    if old.exists() {
        return Ok(Some(CoordinateFrame::read(&old)?));
    }
    Ok(None)
}

/// The ledgers a result made with data folder `data` may be in: its own, and when it is a
/// session's (`sessions/<name>/data`) the work folder's `data/` too.
pub fn ledgers(data: &Path) -> Vec<PathBuf> {
    // (written with `..` or `.`, the same folder: its parents read from the folder named)
    let mut normal = PathBuf::new();
    for c in std::path::absolute(data)
        .unwrap_or_else(|_| data.to_path_buf())
        .components()
    {
        match c {
            std::path::Component::ParentDir => {
                normal.pop();
            }
            std::path::Component::CurDir => {}
            c => normal.push(c),
        }
    }
    let data = normal.as_path();
    let work = data
        .parent()
        .and_then(Path::parent)
        .filter(|s| s.file_name().is_some_and(|n| n == "sessions"))
        .and_then(Path::parent)
        .map(|w| w.join("data"));
    std::iter::once(data.to_path_buf()).chain(work).collect()
}

/// The frame of `result` as [`recorded`] in the first of `ledgers` that has it, with that
/// ledger.
pub fn find(ledgers: &[PathBuf], result: &Path) -> Result<Option<(PathBuf, CoordinateFrame)>> {
    for d in ledgers {
        if let Some(frame) = recorded(d, result)? {
            return Ok(Some((d.clone(), frame)));
        }
    }
    Ok(None)
}

/// A cloud and image poses used together: in the same frame, or both of results without one
/// (older versions'); one known only, the other may be of anywhere.
pub fn ensure_together(
    cloud: Option<CoordinateFrame>,
    poses: Option<CoordinateFrame>,
) -> Result<()> {
    if cloud.is_some() || poses.is_some() {
        anyhow::ensure!(
            cloud.is_some() && poses.is_some(),
            "cloud and image poses must have coordinate metadata; re-run cloud generation and export"
        );
        anyhow::ensure!(
            cloud == poses,
            "cloud and image poses use different coordinate frames; re-run cloud generation and export"
        );
    }
    Ok(())
}

/// The frame of `result`: as [`recorded`], else the legacy projected frame of `data`.
pub fn result_frame(data: &Path, result: &Path) -> Result<CoordinateFrame> {
    Ok(match recorded(data, result)? {
        Some(f) => f,
        None => CoordinateFrame::LegacyProjected {
            origin: s20_io::dataset::load_origin(data),
        },
    })
}

/// Records `results` as they are now in `frame`, or forgets them (`None`); their old sidecars go.
pub fn record(data: &Path, results: &[&Path], frame: Option<CoordinateFrame>) -> Result<()> {
    // (read and replaced by one at a time: two runs of s20 refine-images of two sessions record
    // in the work folder's ledger both)
    fs::create_dir_all(data)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(data.join(LOCK))?;
    lock.lock()?;
    let mut ledger = load(data)?;
    for &r in results {
        let k = key(data, r)?;
        match frame {
            Some(frame) => {
                let (bytes, modified) = stamp(r)?;
                ledger.insert(
                    k,
                    Entry {
                        frame,
                        bytes,
                        modified,
                    },
                );
            }
            None => {
                ledger.remove(&k);
            }
        }
        let old = sidecar(r);
        if old.exists() {
            fs::remove_file(&old).with_context(|| format!("remove {}", old.display()))?;
        }
    }
    save(data, &ledger)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cloud_and_poses_go_together_in_one_frame_or_both_without() {
        let (l, p) = (
            Some(CoordinateFrame::LocalSlam),
            Some(CoordinateFrame::LegacyProjected { origin: [0.; 3] }),
        );
        assert!(ensure_together(l, l).is_ok());
        assert!(ensure_together(None, None).is_ok());
        assert!(ensure_together(l, p).is_err());
        assert!(ensure_together(l, None).is_err());
        assert!(ensure_together(None, l).is_err());
        // a folder written with `..`: the work folder's ledger found all the same
        let w = std::env::temp_dir().join("s20_ledgers").join("work");
        let session = w.join("sessions").join("A").join("data");
        let odd = session.join("..").join("data");
        assert_eq!(ledgers(&odd), ledgers(&session));
        assert_eq!(
            ledgers(&session)[1],
            std::path::absolute(w.join("data")).unwrap()
        );
    }

    #[test]
    fn entries_hold_while_the_file_is_unchanged_and_sidecars_are_migrated() {
        let work = std::env::temp_dir().join(format!("s20_frames_{}", std::process::id()));
        let data = work.join("data");
        fs::create_dir_all(work.join("output")).unwrap();
        fs::create_dir_all(&data).unwrap();
        let cloud = work.join("output").join("a.las");
        fs::write(&cloud, "points").unwrap();
        assert_eq!(recorded(&data, &cloud).unwrap(), None);
        // an old sidecar is read, then replaced by the entry
        CoordinateFrame::LocalSlam.write(&sidecar(&cloud)).unwrap();
        assert_eq!(
            recorded(&data, &cloud).unwrap(),
            Some(CoordinateFrame::LocalSlam)
        );
        let enu = CoordinateFrame::local_enu([38., 141., 40.]).unwrap();
        record(&data, &[&cloud], Some(enu)).unwrap();
        assert!(!sidecar(&cloud).exists());
        assert_eq!(recorded(&data, &cloud).unwrap(), Some(enu));
        let text = fs::read_to_string(data.join(FILE)).unwrap();
        assert!(text.contains("\"output/a.las\""), "{text}");
        // a file written again without recording is of no known frame
        fs::write(&cloud, "other points").unwrap();
        assert_eq!(recorded(&data, &cloud).unwrap(), None);
        assert_eq!(
            result_frame(&data, &cloud).unwrap().output_offset(),
            [0.; 3]
        );
        record(&data, &[&cloud], Some(enu)).unwrap();
        record(&data, &[&cloud], None).unwrap();
        assert_eq!(recorded(&data, &cloud).unwrap(), None);
        fs::write(data.join(FILE), "invalid").unwrap();
        assert!(recorded(&data, &cloud).is_err());
        fs::remove_dir_all(work).unwrap();
    }
}
