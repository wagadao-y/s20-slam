//! Access to a data directory written by [`crate::extract`], including multi-session sets
//! (`sessions.txt` lists per-session data directories, each with the shift added to its times; see
//! [`build_set`]). Points and camera images are read from the bag through a memory map, at the
//! positions in the indices.

use crate::msgs::{POINT_SIZE, livox_point};
use crate::npy;
use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[derive(Debug, Clone, Copy)]
pub struct ScanInfo {
    pub header_stamp: f64,
    pub timebase_ns: f64,
    /// byte position of the packed points in the session's bag
    pub offset: usize,
    pub count: usize,
    pub session: usize,
}

/// Points of one scan (structure of arrays).
#[derive(Default, Clone)]
pub struct Scan {
    pub t: Vec<f64>,
    pub xyz: Vec<[f64; 3]>,
    pub refl: Vec<u8>,
    pub tag: Vec<u8>,
    /// the laser of the point (the MID-360 fires 4 at once)
    pub line: Vec<u8>,
}

pub struct LidarData {
    pub scans: Vec<ScanInfo>,
    maps: Vec<Arc<Mmap>>,
    /// per session: added to its times (see [`build_set`])
    pub shift: Vec<f64>,
}

/// The bag a data directory indexes (`source.txt`: absolute path, then size in bytes).
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    pub bag: PathBuf,
    pub size: u64,
}

impl Source {
    pub const FILE: &str = "source.txt";

    pub fn write(&self, dir: &Path) -> Result<()> {
        fs::write(
            dir.join(Self::FILE),
            format!("{}\n{}\n", self.bag.display(), self.size),
        )?;
        Ok(())
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let f = dir.join(Self::FILE);
        let s = fs::read_to_string(&f).with_context(|| {
            format!(
                "read {} (data folders from older versions must be extracted again)",
                f.display()
            )
        })?;
        let mut it = s.lines();
        let bag = PathBuf::from(it.next().context("source.txt: no bag")?);
        let size = it
            .next()
            .and_then(|l| l.trim().parse().ok())
            .context("source.txt: no size")?;
        Ok(Source { bag, size })
    }

    /// Read-only map of the bag of a data directory, shared within the process.
    pub fn map(dir: &Path) -> Result<Arc<Mmap>> {
        let src = Self::read(dir)?;
        // Share active readers without keeping every recording mapped for the process lifetime.
        static MAPS: OnceLock<Mutex<HashMap<PathBuf, Weak<Mmap>>>> = OnceLock::new();
        let mut maps = MAPS.get_or_init(Default::default).lock().unwrap();
        if let Some(m) = maps.get(&src.bag).and_then(Weak::upgrade)
            && m.len() as u64 == src.size
        {
            return Ok(m.clone());
        }
        let f = fs::File::open(&src.bag)
            .with_context(|| format!("open the bag {} (moved?)", src.bag.display()))?;
        // SAFETY: read-only map of a recording that is not modified while open.
        let m = Arc::new(unsafe { Mmap::map(&f)? });
        if m.len() as u64 != src.size {
            bail!(
                "{} has {} bytes, the index in {} was made from {}: extract again",
                src.bag.display(),
                m.len(),
                dir.display(),
                src.size
            );
        }
        maps.retain(|_, m| m.strong_count() > 0);
        maps.insert(src.bag, Arc::downgrade(&m));
        Ok(m)
    }
}

/// One session of a data directory: its own data directory and the shift added to its times.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionDir {
    pub dir: PathBuf,
    pub shift: f64,
}

/// The sessions of a data directory: the lines of `sessions.txt` (`<dir> [shift]`, a relative
/// directory from the data directory), or the directory itself.
pub fn sessions(data: &Path) -> Result<Vec<SessionDir>> {
    let f = data.join("sessions.txt");
    if !f.exists() {
        return Ok(vec![SessionDir {
            dir: data.to_path_buf(),
            shift: 0.0,
        }]);
    }
    let s = fs::read_to_string(&f)?;
    Ok(s.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            // the directory may contain spaces: the shift is the last field when it parses
            let (d, shift) = match l.rsplit_once(char::is_whitespace) {
                Some((d, x)) if x.parse::<f64>().is_ok() => (d.trim(), x.parse().unwrap()),
                _ => (l, 0.0),
            };
            let p = PathBuf::from(d);
            SessionDir {
                dir: if p.is_absolute() { p } else { data.join(p) },
                shift,
            }
        })
        .collect())
}

pub fn session_dirs(data: &Path) -> Result<Vec<PathBuf>> {
    Ok(sessions(data)?.into_iter().map(|s| s.dir).collect())
}

/// A session's name: the folder holding its data directory (`sessions/<name>/data`), or the data
/// directory's own name.
pub fn session_name(dir: &Path) -> String {
    let d = if dir.file_name().is_some_and(|n| n == "data") {
        dir.parent().unwrap_or(dir)
    } else {
        dir
    };
    d.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// The session recording at time `t` (s, shifted), from the sessions' time ranges
/// ([`LidarData::session_ranges`]): the last one starting at or before it.
pub fn session_at(ranges: &[(f64, f64)], t: f64) -> usize {
    (0..ranges.len())
        .rev()
        .find(|&k| ranges[k].0 <= t)
        .unwrap_or(0)
}

/// Gap left between sessions whose sensor clocks overlap (s).
pub const SESSION_GAP: f64 = 60.0;

/// The shifts that put sessions (time ranges on their own sensor clocks, in order) one after the
/// other: none for the first, and none for a session starting after the previous one ended (as
/// with sessions recorded without switching the sensor off); else the session starts
/// [`SESSION_GAP`] after the previous one, at a whole second.
pub fn session_shifts(ranges: &[(f64, f64)]) -> Vec<f64> {
    let mut out = Vec::with_capacity(ranges.len());
    let mut end = f64::NEG_INFINITY;
    for &(a, b) in ranges {
        let shift = if a > end {
            0.0
        } else {
            (end + SESSION_GAP - a).ceil()
        };
        out.push(shift);
        end = b + shift;
    }
    out
}

/// Writes the data directory of a multi-session set: `sessions.txt` (the session directories as
/// given, relative to `out` or absolute, with their shifts) and the IMU and GNSS tables of all
/// sessions with shifted times. Preserve the shared ENU frame across session changes, as well as
/// `origin.txt` needed to display legacy results until they are regenerated. Returns the shifts.
pub fn build_set(out: &Path, dirs: &[PathBuf]) -> Result<Vec<f64>> {
    fs::create_dir_all(out)?;
    let full = |d: &PathBuf| {
        if d.is_absolute() {
            d.clone()
        } else {
            out.join(d)
        }
    };
    let mut imu = Vec::new();
    let mut ranges = Vec::new();
    for d in dirs {
        let rows = load_imu(&full(d))?;
        let (Some(a), Some(b)) = (rows.first(), rows.last()) else {
            bail!("{}: no IMU samples", full(d).display());
        };
        ranges.push((a[0], b[0]));
        imu.push(rows);
    }
    let shifts = session_shifts(&ranges);
    let mut list = String::from("# session data directory, shift added to its times (s)\n");
    let mut imu_all = Vec::new();
    let mut gnss_all = Vec::new();
    for ((d, rows), &shift) in dirs.iter().zip(&imu).zip(&shifts) {
        list.push_str(&format!(
            "{} {shift}\n",
            d.to_string_lossy().replace('\\', "/")
        ));
        for r in rows {
            imu_all.push(r[0] + shift);
            imu_all.extend_from_slice(&r[1..]);
        }
        // a session without GNSS (indoors) may have no table
        if let Ok(g) = load_gnss(&full(d)) {
            for r in g {
                gnss_all.push(r[0] + shift);
                gnss_all.extend_from_slice(&r[1..]);
            }
        }
    }
    npy::write_f64(&out.join("imu.npy"), &imu_all, 7)?;
    npy::write_f64(&out.join("gnss.npy"), &gnss_all, 14)?;
    fs::write(out.join("sessions.txt"), list)?;
    Ok(shifts)
}

impl LidarData {
    pub fn open(data: &Path) -> Result<Self> {
        let mut scans = Vec::new();
        let mut maps = Vec::new();
        let dirs = sessions(data)?;
        for (k, sd) in dirs.iter().enumerate() {
            let d = &sd.dir;
            let map = Source::map(d)?;
            let (a, rows, cols) = npy::read_f64(&d.join("lidar_index.npy"))?;
            assert_eq!(cols, 4);
            for r in 0..rows {
                let s = ScanInfo {
                    header_stamp: a[r * 4],
                    timebase_ns: a[r * 4 + 1],
                    offset: a[r * 4 + 2] as usize,
                    count: a[r * 4 + 3] as usize,
                    session: k,
                };
                if s.offset + s.count * POINT_SIZE > map.len() {
                    bail!("scan {r} of {} lies outside the bag", d.display());
                }
                scans.push(s);
            }
            maps.push(map);
        }
        Ok(LidarData {
            scans,
            maps,
            shift: dirs.iter().map(|s| s.shift).collect(),
        })
    }

    pub fn n_sessions(&self) -> usize {
        self.shift.len()
    }

    /// Per session: the start of its first scan and the end of its last (s, shifted).
    pub fn session_ranges(&self) -> Vec<(f64, f64)> {
        let mut r = vec![(f64::INFINITY, f64::NEG_INFINITY); self.n_sessions()];
        for i in 0..self.len() {
            let s = &mut r[self.scans[i].session];
            let t = self.scan_start(i);
            s.0 = s.0.min(t);
            s.1 = s.1.max(t + 0.1);
        }
        r
    }

    pub fn len(&self) -> usize {
        self.scans.len()
    }
    pub fn is_empty(&self) -> bool {
        self.scans.is_empty()
    }

    /// Scan start time in seconds (sensor clock, shifted for its session) = timebase.
    pub fn scan_start(&self, i: usize) -> f64 {
        self.scans[i].timebase_ns * 1e-9 + self.shift[self.scans[i].session]
    }

    pub fn scan(&self, i: usize) -> Scan {
        let s = &self.scans[i];
        let b = &self.maps[s.session][s.offset..s.offset + s.count * POINT_SIZE];
        self.decode_scan(i, b)
    }

    /// Read through a bounded file buffer instead of faulting the whole recording's mapped pages
    /// into the viewer's working set. The caller keeps one reader per session.
    pub fn read_scan(
        &self,
        i: usize,
        reader: &mut (impl std::io::Read + std::io::Seek),
    ) -> Result<Scan> {
        let s = &self.scans[i];
        let mut bytes = vec![0u8; s.count * POINT_SIZE];
        reader.seek(std::io::SeekFrom::Start(s.offset as u64))?;
        reader.read_exact(&mut bytes)?;
        Ok(self.decode_scan(i, &bytes))
    }

    fn decode_scan(&self, i: usize, b: &[u8]) -> Scan {
        let s = &self.scans[i];
        let mut out = Scan {
            t: Vec::with_capacity(s.count),
            xyz: Vec::with_capacity(s.count),
            refl: Vec::with_capacity(s.count),
            tag: Vec::with_capacity(s.count),
            line: Vec::with_capacity(s.count),
        };
        let tb = self.scan_start(i);
        for rec in b.as_chunks::<POINT_SIZE>().0 {
            let p = livox_point(rec);
            out.t.push(tb + p.offset_ns as f64 * 1e-9);
            out.xyz
                .push([p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64]);
            out.refl.push(p.reflectivity);
            out.tag.push(p.tag);
            out.line.push(p.line);
        }
        out
    }
}

/// One camera image in the bag: name relative to the (virtual) `images` folder of its data
/// directory (`left/left_00001.jpg`), sensor-clock time, byte range.
#[derive(Clone, Debug)]
pub struct ImageRef {
    pub name: String,
    pub t: f64,
    pub offset: usize,
    pub len: usize,
}

/// The camera images of one data directory (`images.txt`), in record time order.
pub fn image_index(dir: &Path) -> Result<Vec<ImageRef>> {
    let f = dir.join("images.txt");
    let s = fs::read_to_string(&f).with_context(|| format!("read {}", f.display()))?;
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<&str> = l.split_whitespace().collect();
            if v.len() != 4 {
                bail!("{}: bad line {l:?}", f.display());
            }
            Ok(ImageRef {
                name: v[0].to_string(),
                t: v[1].parse()?,
                offset: v[2].parse()?,
                len: v[3].parse()?,
            })
        })
        .collect()
}

/// JPEG bytes of an image named `<data dir>/images/<side>/<file>`: the file when it exists (for
/// image folders on disk), else the image in the bag of that data directory.
pub fn read_image(path: &Path) -> Result<Vec<u8>> {
    if let Ok(b) = fs::read(path) {
        return Ok(b);
    }
    let (_, dir, o, n) = bag_image(path)?;
    let map = Source::map(&dir)?;
    map.get(o..o + n)
        .map(<[u8]>::to_vec)
        .with_context(|| format!("image {} lies outside the bag", path.display()))
}

/// Where the JPEG bytes of an image are: its own file, or a range of the bag.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageLoc {
    File(PathBuf),
    Bag {
        bag: PathBuf,
        size: u64,
        offset: u64,
        len: usize,
    },
}

/// The location of an image named as for [`read_image`].
pub fn image_location(path: &Path) -> Result<ImageLoc> {
    if path.is_file() {
        return Ok(ImageLoc::File(path.to_path_buf()));
    }
    let (src, _, o, n) = bag_image(path)?;
    Ok(ImageLoc::Bag {
        bag: src.bag.clone(),
        size: src.size,
        offset: o as u64,
        len: n,
    })
}

/// The image index of the data directory of an image path (`<data>/images/<side>/<file>`),
/// shared within the process: its bag, the directory, and the image's offset and length.
fn bag_image(path: &Path) -> Result<(Arc<Source>, PathBuf, usize, usize)> {
    let (Some(side_dir), Some(file)) = (path.parent(), path.file_name()) else {
        bail!("no image {}", path.display());
    };
    let (Some(side), Some(images)) = (side_dir.file_name(), side_dir.parent()) else {
        bail!("no image {}", path.display());
    };
    let dir = images
        .parent()
        .context("image path without a data folder")?;
    let name = format!("{}/{}", side.to_string_lossy(), file.to_string_lossy());
    type Index = Arc<(Arc<Source>, HashMap<String, (usize, usize)>)>;
    static INDICES: OnceLock<Mutex<HashMap<PathBuf, Index>>> = OnceLock::new();
    let idx = {
        let mut all = INDICES.get_or_init(Default::default).lock().unwrap();
        match all.get(dir) {
            Some(i) => i.clone(),
            None => {
                let refs =
                    image_index(dir).with_context(|| format!("no image {}", path.display()))?;
                let i: Index = Arc::new((
                    Arc::new(Source::read(dir)?),
                    refs.into_iter()
                        .map(|r| (r.name, (r.offset, r.len)))
                        .collect(),
                ));
                all.insert(dir.to_path_buf(), i.clone());
                i
            }
        }
    };
    let &(o, n) = idx
        .1
        .get(&name)
        .with_context(|| format!("no image {name} in {}", dir.display()))?;
    Ok((idx.0.clone(), dir.to_path_buf(), o, n))
}

/// Reads images with plain file reads (no memory map): images read in bag order are read
/// in sequence, which a hard disk needs (page faults of a map from several threads seek).
#[derive(Default)]
pub struct ImageReader {
    bags: HashMap<PathBuf, fs::File>,
}

impl ImageReader {
    pub fn read(&mut self, loc: &ImageLoc) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        match loc {
            ImageLoc::File(p) => fs::read(p).with_context(|| format!("read {}", p.display())),
            ImageLoc::Bag {
                bag,
                size,
                offset,
                len,
            } => {
                let f = match self.bags.entry(bag.clone()) {
                    std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        let f = fs::File::open(bag)
                            .with_context(|| format!("open the bag {} (moved?)", bag.display()))?;
                        let have = f.metadata()?.len();
                        if have != *size {
                            bail!(
                                "{} has {have} bytes, the index was made from {size}: extract again",
                                bag.display()
                            );
                        }
                        e.insert(f)
                    }
                };
                if offset + *len as u64 > *size {
                    bail!("image at {offset} lies outside the bag {}", bag.display());
                }
                f.seek(SeekFrom::Start(*offset))?;
                let mut b = vec![0u8; *len];
                f.read_exact(&mut b)?;
                Ok(b)
            }
        }
    }
}

/// IMU samples [t, wx, wy, wz, ax, ay, az] (acc in g), concatenated over sessions.
/// LiDAR core temperature rows `[t, degrees C]` (sensor clock), when extracted (`temp.npy`).
pub fn load_temperature(data: &Path) -> Result<Vec<[f64; 2]>> {
    let (a, rows, cols) = npy::read_f64(&data.join("temp.npy"))?;
    anyhow::ensure!(cols == 2, "temp.npy must have 2 columns");
    Ok((0..rows).map(|r| [a[2 * r], a[2 * r + 1]]).collect())
}

pub fn load_imu(data: &Path) -> Result<Vec<[f64; 7]>> {
    let p = data.join("imu.npy");
    let (a, rows, cols) = npy::read_f64(&p)?;
    assert_eq!(cols, 7);
    Ok((0..rows)
        .map(|r| a[r * 7..r * 7 + 7].try_into().unwrap())
        .collect())
}

/// GNSS rows (see msgs::pvtsln), 14 columns.
pub fn load_gnss(data: &Path) -> Result<Vec<[f64; 14]>> {
    let (a, rows, cols) = npy::read_f64(&data.join("gnss.npy"))?;
    assert_eq!(
        cols, 14,
        "gnss.npy must have 14 columns (re-extract with single-point solution)"
    );
    Ok((0..rows)
        .map(|r| a[r * 14..r * 14 + 14].try_into().unwrap())
        .collect())
}

/// Frame metadata for GNSS and output coordinates. Missing metadata denotes an old project.
pub fn load_frame(data: &Path) -> Result<crate::geodesy::CoordinateFrame> {
    let path = data.join("coordinate_frame.json");
    if path.exists() {
        crate::geodesy::CoordinateFrame::read(&path)
    } else {
        Ok(crate::geodesy::CoordinateFrame::LegacyProjected {
            origin: load_origin(data),
        })
    }
}

/// Optional legacy projected offset. New ENU / SLAM results have no output translation.
pub fn load_origin(data: &Path) -> [f64; 3] {
    fs::read_to_string(data.join("origin.txt"))
        .ok()
        .and_then(|s| {
            let v: Vec<f64> = s
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            (v.len() == 3).then(|| [v[0], v[1], v[2]])
        })
        .unwrap_or([0.0; 3])
}

pub fn load_time_offset(data: &Path) -> f64 {
    fs::read_to_string(data.join("maker_time_offset.txt"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0)
}

/// Per session: subtracted from a (shifted) time to give the output clock, its shift plus its
/// maker offset.
pub fn output_offsets(data: &Path) -> Result<Vec<f64>> {
    Ok(sessions(data)?
        .iter()
        .map(|s| s.shift + load_time_offset(&s.dir))
        .collect())
}

/// Trajectory text file: rows `t x y z qx qy qz qw` ('#' comments). Returns rows.
pub fn read_tum(path: &Path) -> Result<Vec<[f64; 8]>> {
    let s = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    for l in s.lines() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let v: Vec<f64> = l
            .split_whitespace()
            .take(8)
            .map(|x| x.parse())
            .collect::<Result<_, _>>()?;
        if v.len() == 8 {
            out.push(v.try_into().unwrap());
        }
    }
    Ok(out)
}

pub fn write_tum(path: &Path, rows: &[[f64; 8]], header: &str) -> Result<()> {
    let mut s = format!("# {header}\n");
    for r in rows {
        s.push_str(&format!(
            "{:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}\n",
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
        ));
    }
    fs::write(path, s)?;
    Ok(())
}

/// Whitespace separated numeric table ('#' comments).
pub fn read_table(path: &Path) -> Result<Vec<Vec<f64>>> {
    let s = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    s.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            l.split_whitespace()
                .map(|x| x.parse::<f64>().map_err(Into::into))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_maps_are_shared_only_while_readers_are_alive() {
        let dir = std::env::temp_dir().join(format!("s20_map_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bag = dir.join("test.bag");
        fs::write(&bag, [1, 2, 3]).unwrap();
        Source { bag, size: 3 }.write(&dir).unwrap();
        let first = Source::map(&dir).unwrap();
        let second = Source::map(&dir).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let weak = Arc::downgrade(&first);
        drop((first, second));
        assert!(
            weak.upgrade().is_none(),
            "recording stays mapped without active readers"
        );
        assert_eq!(&Source::map(&dir).unwrap()[..], &[1, 2, 3]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sessions_follow_each_other_on_one_time_axis() {
        // recorded without switching off: kept on the sensor clock
        assert_eq!(
            session_shifts(&[(100.0, 200.0), (300.0, 400.0)]),
            vec![0.0, 0.0]
        );
        // the clock went back (switched off in between): moved after the previous one
        let s = session_shifts(&[(100.0, 200.0), (90.5, 150.0), (120.0, 130.0)]);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], (200.0 + SESSION_GAP - 90.5f64).ceil());
        assert!(120.0 + s[2] >= 150.0 + s[1] + SESSION_GAP);
        assert_eq!(s[2].fract(), 0.0);
    }

    #[test]
    fn sessions_txt_with_and_without_shifts() {
        let d = std::env::temp_dir().join(format!("s20_sessions_{}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        fs::write(
            d.join("sessions.txt"),
            "# comment\n../../park1/data\n../a b/data 61\n",
        )
        .unwrap();
        let s = sessions(&d).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(
            (s[0].dir.clone(), s[0].shift),
            (d.join("../../park1/data"), 0.0)
        );
        assert_eq!(
            (s[1].dir.clone(), s[1].shift),
            (d.join("../a b/data"), 61.0)
        );
        assert_eq!(session_name(&s[1].dir), "a b");
        assert_eq!(session_name(Path::new("x/park3")), "park3");
        fs::remove_dir_all(&d).unwrap();
    }
}
