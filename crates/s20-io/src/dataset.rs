//! Access to a data directory written by [`crate::extract`], including multi-session sets
//! (`sessions.txt` lists per-session data directories, each with the shift added to its times; see
//! [`build_set`]). Points and camera images are read from the bag through a memory map, at the
//! positions in the indices.

use crate::msgs::{POINT_SIZE, livox_point};
use crate::npy;
use anyhow::{Context, Result, bail, ensure};
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

/// A bag's fingerprint (`v2-<FNV-1a 64>-<CRC-32>`): FNV-1a over its size and every byte, one at
/// a time as the method is, and a CRC-32 of its bytes. Read whole, as a part read would miss a
/// chunk replaced; made by the extraction (which reads the bag anyway) and when a recording
/// moved, not at every run.
pub fn bag_fingerprint(bag: &Path) -> Result<String> {
    bag_fingerprint_checked(bag, &|| Ok(()))
}

/// [`bag_fingerprint`] that asks `check` between its blocks (a cancellation stops it).
pub fn bag_fingerprint_checked(bag: &Path, check: &dyn Fn() -> Result<()>) -> Result<String> {
    use std::io::Read;
    let mut f = fs::File::open(bag).with_context(|| format!("open {}", bag.display()))?;
    let size = f.metadata()?.len();
    let mut h: u64 = 0xcbf29ce484222325;
    let mut crc = crc32fast::Hasher::new();
    for b in size.to_le_bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    let mut buf = vec![0u8; 8 << 20];
    loop {
        check()?;
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &b in &buf[..n] {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
        crc.update(&buf[..n]);
    }
    Ok(format!("v2-{h:016x}-{:08x}", crc.finalize()))
}

/// The fingerprint of the bag data folder `dir` was extracted from (`bag_fingerprint.txt`), when
/// the extraction wrote one of this kind.
pub fn extracted_fingerprint(dir: &Path) -> Option<String> {
    fs::read_to_string(dir.join(FINGERPRINT_FILE))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| s.starts_with("v2-"))
}

pub const FINGERPRINT_FILE: &str = "bag_fingerprint.txt";

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
/// `gnss`: the device's GNSS tables are used (its RTK): one that cannot be read is an error; else
/// it is left out.
pub fn build_set(out: &Path, dirs: &[PathBuf], gnss: bool) -> Result<Vec<f64>> {
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
        // a session without GNSS (indoors) may have no table; one that cannot be read is an
        // error when the tables are used, not a session without fixes
        let table = if full(d).join("gnss.npy").exists() {
            match load_gnss(&full(d)) {
                Ok(t) => t,
                Err(e) if gnss => return Err(e),
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        };
        {
            for r in table {
                gnss_all.push(r[0] + shift);
                gnss_all.extend_from_slice(&r[1..]);
            }
        }
    }
    // (a file of the same contents left as it is, its time with it: made again at every
    // optimization, the steps after the import stay current when nothing changed)
    let same_or_write = |f: PathBuf, b: &[u8]| -> Result<()> {
        if fs::read(&f).is_ok_and(|old| old == b) {
            return Ok(());
        }
        fs::write(&f, b).with_context(|| format!("write {}", f.display()))
    };
    same_or_write(out.join("imu.npy"), &npy::to_bytes(&imu_all, 7)?)?;
    same_or_write(out.join("gnss.npy"), &npy::to_bytes(&gnss_all, 14)?)?;
    same_or_write(out.join("sessions.txt"), list.as_bytes())?;
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
            let f = d.join("lidar_index.npy");
            let (a, rows, cols) = npy::read_f64(&f)?;
            ensure!(cols == 4, "{}: {cols} columns, not 4", f.display());
            for r in 0..rows {
                // (a damaged index: whole, nonnegative and within the bag, or an error)
                let whole = |x: f64| {
                    (x.is_finite() && x >= 0. && x.fract() == 0. && x < 2f64.powi(53))
                        .then_some(x as usize)
                };
                let (Some(offset), Some(count)) = (whole(a[r * 4 + 2]), whole(a[r * 4 + 3])) else {
                    bail!("scan {r} of {}: a bad offset or count", f.display());
                };
                let s = ScanInfo {
                    header_stamp: a[r * 4],
                    timebase_ns: a[r * 4 + 1],
                    offset,
                    count,
                    session: k,
                };
                let end = count
                    .checked_mul(POINT_SIZE)
                    .and_then(|n| n.checked_add(offset));
                if end.is_none_or(|e| e > map.len()) {
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
            // `left/<file>` or `right/<file>`: joined to the folders read and written
            let ok = v[0].split_once('/').is_some_and(|(side, file)| {
                matches!(side, "left" | "right")
                    && !file.is_empty()
                    && file != ".."
                    && !file.contains(['/', '\\', ':'])
            });
            if !ok {
                bail!("{}: bad image name {:?}", f.display(), v[0]);
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
    let (src, _, o, n) = bag_image(path)?;
    if o.checked_add(n).is_none_or(|end| end as u64 > src.size) {
        bail!("image {} lies outside the bag", path.display());
    }
    // a plain read at the image's place, not through a map of the bag: the pages read stay in
    // the system's file cache instead of the process (4650 images: tens of GB in its working set)
    let bag = open_bag(&src)?;
    let mut buf = vec![0u8; n];
    read_exact_at(&bag, &mut buf, o as u64)
        .with_context(|| format!("image {} lies outside the bag", path.display()))?;
    Ok(buf)
}

/// The bag of `src`, opened once in the process while it keeps the size the index was made from
/// (checked on every use: a bag replaced at the same path is opened again).
fn open_bag(src: &Source) -> Result<Arc<fs::File>> {
    static BAGS: OnceLock<Mutex<HashMap<PathBuf, Arc<fs::File>>>> = OnceLock::new();
    let mut bags = BAGS.get_or_init(Default::default).lock().unwrap();
    // (the handle's file still the one at the path: the same size and time)
    let same = |f: &fs::File| -> bool {
        let (Ok(a), Ok(b)) = (f.metadata(), fs::metadata(&src.bag)) else {
            return false;
        };
        a.len() == src.size && b.len() == a.len() && a.modified().ok() == b.modified().ok()
    };
    if let Some(f) = bags.get(&src.bag)
        && same(f)
    {
        return Ok(f.clone());
    }
    let f = fs::File::open(&src.bag)
        .with_context(|| format!("open the bag {} (moved?)", src.bag.display()))?;
    let size = f.metadata()?.len();
    if size != src.size {
        bail!(
            "{} has {size} bytes, its index was made from {}: extract again",
            src.bag.display(),
            src.size
        );
    }
    let f = Arc::new(f);
    bags.insert(src.bag.clone(), f.clone());
    Ok(f)
}

/// `buf.len()` bytes of `f` from `offset`, from any thread (no shared position).
#[cfg(unix)]
fn read_exact_at(f: &fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, offset)
}

/// `buf.len()` bytes of `f` from `offset`, from any thread (each read says where; on Windows it
/// also moves the file's cursor, which nothing here relies on).
#[cfg(windows)]
fn read_exact_at(f: &fs::File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match f.seek_read(buf, offset) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "the bag ends before the image",
                ));
            }
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
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
        // (kept while the data folder still points at the same bag: relinked to a moved one, its
        // index is read again)
        let now = Source::read(dir).ok();
        match all.get(dir).filter(|i| now.as_ref() == Some(&*i.0)) {
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
    ensure!(cols == 7, "{}: {cols} columns, not 7", p.display());
    let mut v: Vec<[f64; 7]> = (0..rows)
        .map(|r| a[r * 7..r * 7 + 7].try_into().unwrap())
        .collect();
    // (finite, then in time order, for every step that integrates it: the bag holds the samples
    // by their record times, which may swap two; a NaN would be sorted to an end and taken for
    // the session's span)
    if let Some(k) = v.iter().position(|r| !r.iter().all(|x| x.is_finite())) {
        bail!("{}: sample {k} not finite", p.display());
    }
    v.sort_by(|a, b| a[0].total_cmp(&b[0]));
    Ok(v)
}

/// GNSS rows (see msgs::pvtsln), 14 columns.
pub fn load_gnss(data: &Path) -> Result<Vec<[f64; 14]>> {
    let (a, rows, cols) = npy::read_f64(&data.join("gnss.npy"))?;
    ensure!(
        cols == 14,
        "gnss.npy must have 14 columns, not {cols} (re-extract with single-point solution)"
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

/// A line of `ImgPose.txt` that is no row: blank, the header (`index ...`) or a comment (`#`).
/// The same for every reader, so a file read by one is read by all.
pub fn img_pose_skipped(line: &str) -> bool {
    let l = line.trim_start();
    l.is_empty() || l.starts_with("index") || l.starts_with('#')
}

/// The fields of a row of `ImgPose.txt` (`name x y z roll pitch yaw qx qy qz qw timestamp`): the
/// name first, then the numbers. The name is all before the numbers at the end of the row, so it
/// may hold spaces (the absolute path of a session's image). None without eleven numbers after a
/// name.
pub fn img_pose_fields(line: &str) -> Option<Vec<&str>> {
    // (a comment after the numbers left out: from the first '#' standing alone (a space before
    // it, a space or the end after it) that follows eleven numbers and is not followed by eleven
    // at the end (then it was a name's, the pose after it); one in a name ("#2") is the name's)
    let ends_in_numbers = |t: &str| {
        t.split_whitespace()
            .rev()
            .take(11)
            .filter(|t| t.parse::<f64>().is_ok())
            .count()
            == 11
    };
    // (every way to read it: no comment (the line ends in the numbers), or a comment from a
    // standalone '#' the numbers come before; read more than one way, or none, it is refused
    // rather than one taken)
    let mut readings: Vec<&str> = Vec::new();
    if ends_in_numbers(line) {
        readings.push(line);
    }
    for (i, c) in line.char_indices() {
        let before = &line[..i];
        // (a space before it, whatever after: "# refined" and "#refined" both)
        if c == '#' && before.ends_with(char::is_whitespace) && ends_in_numbers(before) {
            readings.push(before);
        }
    }
    let [line] = readings[..] else {
        return None;
    };
    let mut tokens: Vec<(usize, &str)> = Vec::new();
    let mut start = None;
    for (i, c) in line.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                tokens.push((s, &line[s..i]));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        tokens.push((s, &line[s..]));
    }
    let mut first = tokens.len();
    while first > 1 && tokens[first - 1].1.parse::<f64>().is_ok() {
        first -= 1;
    }
    if tokens.len() - first < 11 {
        return None;
    }
    let mut v = vec![line[..tokens[first].0].trim()];
    v.extend(tokens[first..].iter().map(|t| t.1));
    Some(v)
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
    fs::write(path, tum_text(rows, header))?;
    Ok(())
}

/// The text [`write_tum`] writes.
pub fn tum_text(rows: &[[f64; 8]], header: &str) -> String {
    let mut s = format!("# {header}\n");
    for r in rows {
        s.push_str(&format!(
            "{:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}\n",
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
        ));
    }
    s
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
    fn a_bag_differing_in_its_middle_has_another_fingerprint() {
        let dir = std::env::temp_dir().join(format!("s20_fp_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let mut a = vec![7u8; 5 << 20];
        let (fa, fb) = (dir.join("a.bag"), dir.join("b.bag"));
        fs::write(&fa, &a).unwrap();
        // the first and last MiB the same, a byte in the middle not
        a[(5 << 20) / 2] = 8;
        fs::write(&fb, &a).unwrap();
        assert_ne!(bag_fingerprint(&fa).unwrap(), bag_fingerprint(&fb).unwrap());
        assert_eq!(bag_fingerprint(&fa).unwrap(), bag_fingerprint(&fa).unwrap());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_image_name_with_spaces_stays_whole() {
        let nums = "1 2 3 0.1 0.2 0.3 0 0 0 1 12.5";
        let line = format!("C:/My work/A b/images/left/l_1.jpg {nums}");
        let v = img_pose_fields(&line).unwrap();
        assert_eq!(v[0], "C:/My work/A b/images/left/l_1.jpg");
        assert_eq!(v.len(), 12);
        assert_eq!(v[11], "12.5");
        let line = format!("left/l_1.jpg  {nums}");
        let v = img_pose_fields(&line).unwrap();
        assert_eq!(v[0], "left/l_1.jpg");
        assert!(img_pose_fields("left/l_1.jpg 1 2 3").is_none());
        assert!(img_pose_fields(nums).is_none());
        // a comment after the numbers, with a name of spaces
        let line = format!("C:/My work/l_1.jpg {nums}  # refined");
        let v = img_pose_fields(&line).unwrap();
        assert_eq!((v[0], v.len(), v[11]), ("C:/My work/l_1.jpg", 12, "12.5"));
        // a comment with numbers of its own: read two ways (the pose, or the comment's numbers)
        let line = format!("l_1.jpg {nums} # audit {nums} # end");
        assert!(img_pose_fields(&line).is_none());
        // a name's '#' and a comment: read two ways too
        let line = format!("C:/Survey 1 2 3 4 5 6 7 8 9 10 11 # East/l_1.jpg {nums} # refined");
        assert!(img_pose_fields(&line).is_none());
        // read two ways (a comment ending in eleven numbers, or a name of them and a '#'): refused
        let line = format!("C:/Survey 1 2 3 4 5 6 7 8 9 10 11 # East/l_1.jpg {nums}");
        assert!(img_pose_fields(&line).is_none());
        let line = format!("l_1.jpg {nums} # before {nums}");
        assert!(img_pose_fields(&line).is_none());
        // a comment written without a space after its '#'
        let line = format!("l_1.jpg {nums} #refined");
        assert_eq!(img_pose_fields(&line).unwrap()[0], "l_1.jpg");
        // a name's "#2" after eleven numbers reads two ways too
        let line = format!("C:/Survey 1 2 3 4 5 6 7 8 9 10 11 #2/l_1.jpg {nums}");
        assert!(img_pose_fields(&line).is_none());
        // a '#' after a space in the name is the name's
        let line = format!("C:/Survey #2/l_1.jpg {nums} # refined");
        let v = img_pose_fields(&line).unwrap();
        assert_eq!((v[0], v.len()), ("C:/Survey #2/l_1.jpg", 12));
    }

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
