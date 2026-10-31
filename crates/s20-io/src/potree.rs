//! Potree 2.0 octrees in the layout of PotreeConverter 2 (`metadata.json`, `hierarchy.bin`,
//! `octree.bin` in a folder named after the source file plus `_converted`), as the Potree viewer
//! reads them: written BROTLI-encoded from a LAS/LAZ file with bounded memory, read node by node
//! (BROTLI or DEFAULT).
//!
//! As PotreeConverter: a cube over the points is cut into chunks of at most [`CHUNK_POINTS`] by a
//! counting grid, the points of each going to a temporary file; each chunk becomes an octree of its
//! own, the levels above the chunks are made last. A node holds a Poisson-disk sample of its
//! subtree (no two points closer than its spacing, the cube's 1/128), taken bottom-up from its
//! children; a leaf keeps the rest of its points (at most [`LEAF_POINTS`]). Every point is in
//! exactly one node; the spacing halves at each level. The Potree viewer sizes the points by it
//! (its adaptive size): a denser sample would draw the coarse levels as clusters of large discs.
//!
//! A BROTLI node is, for its points in Morton order, attribute by attribute: the position as a 3D
//! Morton code of its integer coordinates (16 bytes: the code of the upper 16 bits of each, then
//! that of the lower), the intensity, the colour as the Morton code of its three 16-bit values
//! (8 bytes) and the GPS time (8-byte float), the last two when written; the whole compressed with Brotli (quality 5: PotreeConverter takes 6, slower for
//! files 0.05 % smaller). See ref/potree/src/modules/loader/2.0 (OctreeLoader.js,
//! DecoderWorker_brotli.js).

use crate::las::LasReader;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Points of a chunk at most (unless it is a cell of the counting grid).
pub const CHUNK_POINTS: usize = 1_000_000;
/// Points of a leaf at most (unless it is at [`MAX_DEPTH`]), as PotreeConverter.
pub const LEAF_POINTS: usize = 10_000;
/// Cells of a node's sampling grid along each axis: the root's spacing is the cube's 1/128.
pub const GRID: u32 = 128;
/// Levels below the root at most (the cube's 1/2^20: 0.2 mm for 200 m).
pub const MAX_DEPTH: u32 = 20;
/// Levels of the counting grid that cuts the chunks (128³ cells).
const COUNT_LEVEL: u32 = 7;
/// Bits of the key: a point's cell at this level along each axis (see [`Pt::key`]).
const KEY_BITS: u32 = 30;
const HIERARCHY_ENTRY: usize = 22;
/// The record of a point in a chunk file: coordinates (3 × i32), colour (3 × u16), intensity;
/// then the GPS time (f64) when it is written.
const RECORD: usize = 20;
const RECORD_TIME: usize = RECORD + 8;
/// Chunks built at once (each takes some 60 bytes a point).
const CHUNK_THREADS: usize = 16;

/// The folder PotreeConverter writes for `source`: its file name with `_converted` appended.
pub fn converted_dir(source: &Path) -> PathBuf {
    let mut name = source.file_name().unwrap_or_default().to_os_string();
    name.push("_converted");
    source.with_file_name(name)
}

/// How octrees are written, in their [`Stamp`]: 2 with Poisson-disk samples (1: one point a cell).
const WRITER: u32 = 2;

/// What a written octree was made from: the source's file name, size and modification time, and
/// how it was written ([`WRITER`]), kept in `metadata.json` (under `s20`; the Potree viewer reads
/// past it) to tell whether it is current.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Stamp {
    pub source: String,
    pub bytes: u64,
    /// nanoseconds since the Unix epoch
    pub modified: u128,
    /// none in the octrees of the first writer
    #[serde(default)]
    pub writer: u32,
}

impl Stamp {
    pub fn of(source: &Path) -> Result<Self> {
        let m = fs::metadata(source).with_context(|| format!("stat {}", source.display()))?;
        Ok(Stamp {
            source: source
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            bytes: m.len(),
            modified: m
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
            writer: WRITER,
        })
    }
}

/// Whether `dir` holds an octree written from `source` as it is now.
pub fn is_current(dir: &Path, source: &Path) -> bool {
    let Ok(stamp) = Stamp::of(source) else {
        return false;
    };
    dir.join("hierarchy.bin").exists()
        && dir.join("octree.bin").exists()
        && read_metadata(dir).is_ok_and(|m| m.s20 == Some(stamp))
}

/// What to keep of the source.
#[derive(Clone, Copy, Debug, Default)]
pub struct WriteOptions {
    /// the colour (else only the positions and intensities)
    pub rgb: bool,
    /// the GPS times (`gps-time`, as PotreeConverter's: the Potree viewer colours and filters by it)
    pub gps_time: bool,
    /// points of this class are left out (the dynamic points)
    pub skip_class: Option<u8>,
}

/// A point while the octree is built.
#[derive(Clone, Copy)]
struct Pt {
    /// integer coordinates: (x - offset) / scale
    x: [u32; 3],
    rgb: [u16; 3],
    intensity: u16,
    gps_time: f64,
    /// the point's cell along each axis at level [`KEY_BITS`]: a node's index at level l is
    /// `key >> (KEY_BITS - l)`
    key: [u32; 3],
}

/// The cube in integer coordinates: [0, size) along each axis.
#[derive(Clone, Copy)]
struct Cube {
    size: f64,
}

impl Cube {
    fn key(&self, x: [u32; 3]) -> [u32; 3] {
        let cells = (1u64 << KEY_BITS) as f64;
        x.map(|v| ((v as f64 * cells / self.size) as u64).min((1 << KEY_BITS) - 1) as u32)
    }
}

/// The child of a node at `level` that a point is in (Potree's order: x 4, y 2, z 1).
fn child_of(key: [u32; 3], level: u32) -> usize {
    let s = KEY_BITS - level - 1;
    (((key[0] >> s) & 1) << 2 | ((key[1] >> s) & 1) << 1 | ((key[2] >> s) & 1)) as usize
}

/// A node made and not yet written: its parent may still take points from it.
struct Pending {
    name: Vec<u8>,
    points: Vec<Pt>,
    leaf: bool,
    mask: u8,
}

#[derive(Clone, Copy)]
struct Entry {
    points: u32,
    offset: u64,
    size: u64,
    leaf: bool,
    mask: u8,
}

struct Build<'a> {
    rgb: bool,
    gps_time: bool,
    octree: Mutex<(BufWriter<File>, u64)>,
    entries: Mutex<HashMap<Vec<u8>, Entry>>,
    cancel: &'a AtomicBool,
    done: AtomicU64,
}

fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!(crate::job::Cancelled);
    }
    Ok(())
}

impl Build<'_> {
    /// The octree of `points`, the subtree of node `name` at `level`: all written but its root.
    fn build(&self, level: u32, name: Vec<u8>, points: Vec<Pt>) -> Result<Pending> {
        check(self.cancel)?;
        if points.len() <= LEAF_POINTS || level >= MAX_DEPTH {
            self.done.fetch_add(points.len() as u64, Ordering::Relaxed);
            return Ok(Pending {
                name,
                points,
                leaf: true,
                mask: 0,
            });
        }
        let mut parts: [Vec<Pt>; 8] = Default::default();
        for p in points {
            parts[child_of(p.key, level)].push(p);
        }
        let mut children = Vec::new();
        for (k, part) in parts.into_iter().enumerate() {
            if !part.is_empty() {
                let mut child = name.clone();
                child.push(k as u8);
                children.push(self.build(level + 1, child, part)?);
            }
        }
        self.join(level, name, children)
    }

    /// Node `name` at `level` over its children: it takes its sample from their points, which
    /// are then written (a leaf left without points goes). The sample is PotreeConverter's
    /// Poisson-disk one: the points by their distance to the node's centre, each taken unless one
    /// taken is closer than the spacing (found with a grid of the spacing).
    fn join(&self, level: u32, name: Vec<u8>, mut children: Vec<Pending>) -> Result<Pending> {
        check(self.cancel)?;
        // in key units: the spacing is a cell at level + 7, the node 2^(KEY_BITS - level)
        let shift = KEY_BITS - level - GRID.trailing_zeros();
        let spacing2 = 1i64 << (2 * shift);
        let node_bits = KEY_BITS - level;
        let half = 1i64 << (node_bits - 1);
        let mask = (1u64 << node_bits) - 1;
        let rel = |k: [u32; 3]| k.map(|v| (v as u64 & mask) as i64 - half);
        let mut candidates: Vec<(i64, u32, u32)> = Vec::new();
        for (c, child) in children.iter().enumerate() {
            for (i, p) in child.points.iter().enumerate() {
                let d = rel(p.key).iter().map(|d| d * d).sum::<i64>();
                candidates.push((d, c as u32, i as u32));
            }
        }
        candidates.sort_unstable();
        let mut taken: Vec<Vec<bool>> = children
            .iter()
            .map(|c| vec![false; c.points.len()])
            .collect();
        let mut points = Vec::new();
        // by cell: the points taken in it (no more than 8 can be a cell apart)
        let mut cells: HashMap<u32, (u8, [u32; 8]), std::hash::BuildHasherDefault<CellHasher>> =
            HashMap::default();
        let mut accepted: Vec<[i64; 3]> = Vec::new();
        let cell_id = |c: [i64; 3]| ((c[0] as u32) << 14) | ((c[1] as u32) << 7) | c[2] as u32;
        let last = GRID as i64 - 1;
        for &(_, c, i) in &candidates {
            let p = rel(children[c as usize].points[i as usize].key);
            let cell = p.map(|v| ((v + half) >> shift).clamp(0, last));
            let near = |q: [i64; 3]| {
                cells.get(&cell_id(q)).is_some_and(|(n, ids)| {
                    ids[..*n as usize].iter().any(|&a| {
                        let a = accepted[a as usize];
                        (0..3).map(|d| (a[d] - p[d]).pow(2)).sum::<i64>() < spacing2
                    })
                })
            };
            let mut free = true;
            'around: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let q = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                        if q.iter().all(|&v| (0..=last).contains(&v)) && near(q) {
                            free = false;
                            break 'around;
                        }
                    }
                }
            }
            if !free {
                continue;
            }
            let e = cells.entry(cell_id(cell)).or_insert((0, [0; 8]));
            if (e.0 as usize) < e.1.len() {
                e.1[e.0 as usize] = accepted.len() as u32;
                e.0 += 1;
            }
            accepted.push(p);
            taken[c as usize][i as usize] = true;
            points.push(children[c as usize].points[i as usize]);
        }
        drop(candidates);
        let mut mask = 0u8;
        for (child, taken) in children.iter_mut().zip(taken) {
            let mut t = taken.iter();
            child.points.retain(|_| !*t.next().unwrap());
            if child.points.is_empty() && child.leaf {
                continue;
            }
            mask |= 1 << child.name[child.name.len() - 1];
        }
        children
            .into_par_iter()
            .filter(|c| !(c.points.is_empty() && c.leaf))
            .try_for_each(|c| self.write(c))?;
        Ok(Pending {
            name,
            points,
            leaf: false,
            mask,
        })
    }

    /// Encode a node into octree.bin and enter it in the hierarchy.
    fn write(&self, node: Pending) -> Result<()> {
        let data = if node.points.is_empty() {
            Vec::new()
        } else {
            encode(&node.points, self.rgb, self.gps_time)?
        };
        let offset = {
            let mut o = self.octree.lock().unwrap();
            let at = o.1;
            o.0.write_all(&data)?;
            o.1 += data.len() as u64;
            at
        };
        self.entries.lock().unwrap().insert(
            node.name,
            Entry {
                points: node.points.len() as u32,
                offset,
                size: data.len() as u64,
                leaf: node.leaf,
                mask: node.mask,
            },
        );
        Ok(())
    }
}

/// A hasher for the cells of a node's grid (whole numbers below 2^21).
#[derive(Default)]
struct CellHasher(u64);

impl std::hash::Hasher for CellHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u32(&mut self, v: u32) {
        self.0 = (v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// Read into `buf` until it is full or the file ends; the bytes read.
fn read_full(f: &mut File, buf: &mut [u8]) -> Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        let k = f.read(&mut buf[n..])?;
        if k == 0 {
            break;
        }
        n += k;
    }
    Ok(n)
}

/// Spread the low 21 bits of `v` to every third bit.
fn spread(v: u64) -> u64 {
    let mut x = v & 0x1f_ffff;
    x = (x | x << 32) & 0x1f_0000_0000_ffff;
    x = (x | x << 16) & 0x1f_0000_ff00_00ff;
    x = (x | x << 8) & 0x100f_00f0_0f00_f00f;
    x = (x | x << 4) & 0x10c3_0c30_c30c_30c3;
    x = (x | x << 2) & 0x1249_2492_4924_9249;
    x
}

/// The inverse of [`spread`].
fn compact(v: u64) -> u64 {
    let mut x = v & 0x1249_2492_4924_9249;
    x = (x ^ (x >> 2)) & 0x10c3_0c30_c30c_30c3;
    x = (x ^ (x >> 4)) & 0x100f_00f0_0f00_f00f;
    x = (x ^ (x >> 8)) & 0x1f_0000_ff00_00ff;
    x = (x ^ (x >> 16)) & 0x1f_0000_0000_ffff;
    x = (x ^ (x >> 32)) & 0x1f_ffff;
    x
}

/// Morton code of three 16-bit values, the first at bit 0.
fn morton(v: [u32; 3]) -> u64 {
    spread(v[0] as u64) | spread(v[1] as u64) << 1 | spread(v[2] as u64) << 2
}

fn unmorton(m: u64) -> [u32; 3] {
    [
        compact(m) as u32,
        compact(m >> 1) as u32,
        compact(m >> 2) as u32,
    ]
}

/// A position's code: (upper 16 bits, lower 16 bits).
fn position_code(x: [u32; 3]) -> (u64, u64) {
    (morton(x.map(|v| v >> 16)), morton(x.map(|v| v & 0xffff)))
}

fn encode(points: &[Pt], rgb: bool, gps_time: bool) -> Result<Vec<u8>> {
    let mut order: Vec<((u64, u64), u32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (position_code(p.x), i as u32))
        .collect();
    order.sort_unstable();
    let n = points.len();
    let mut raw = Vec::with_capacity(n * (18 + 8 * rgb as usize + 8 * gps_time as usize));
    for &((upper, lower), _) in &order {
        raw.extend_from_slice(&upper.to_le_bytes());
        raw.extend_from_slice(&lower.to_le_bytes());
    }
    for &(_, i) in &order {
        raw.extend_from_slice(&points[i as usize].intensity.to_le_bytes());
    }
    if rgb {
        for &(_, i) in &order {
            let c = points[i as usize].rgb.map(u32::from);
            raw.extend_from_slice(&morton(c).to_le_bytes());
        }
    }
    if gps_time {
        for &(_, i) in &order {
            raw.extend_from_slice(&points[i as usize].gps_time.to_le_bytes());
        }
    }
    let mut out = Vec::with_capacity(raw.len() / 3);
    {
        let mut w = brotli::CompressorWriter::new(&mut out, 1 << 16, 5, 22);
        w.write_all(&raw)?;
    }
    Ok(out)
}

#[derive(serde::Serialize, serde::Deserialize)]
struct HierarchyMeta {
    #[serde(rename = "firstChunkSize")]
    first_chunk_size: u64,
    #[serde(rename = "stepSize")]
    step_size: u32,
    depth: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BoxMeta {
    min: [f64; 3],
    max: [f64; 3],
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct AttributeMeta {
    name: String,
    #[serde(default)]
    description: String,
    size: usize,
    #[serde(rename = "numElements")]
    num_elements: usize,
    #[serde(rename = "elementSize")]
    element_size: usize,
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    min: Vec<f64>,
    #[serde(default)]
    max: Vec<f64>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Metadata {
    version: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
    points: u64,
    #[serde(default)]
    projection: String,
    hierarchy: HierarchyMeta,
    offset: [f64; 3],
    scale: [f64; 3],
    spacing: f64,
    #[serde(rename = "boundingBox")]
    bounding_box: BoxMeta,
    encoding: String,
    attributes: Vec<AttributeMeta>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    s20: Option<Stamp>,
}

fn read_metadata(dir: &Path) -> Result<Metadata> {
    let path = dir.join("metadata.json");
    let text = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&text).with_context(|| format!("parse {}", path.display()))
}

fn attribute(
    name: &str,
    ty: &str,
    n: usize,
    size: usize,
    min: Vec<f64>,
    max: Vec<f64>,
) -> AttributeMeta {
    AttributeMeta {
        name: name.into(),
        description: String::new(),
        size: n * size,
        num_elements: n,
        element_size: size,
        ty: ty.into(),
        min,
        max,
    }
}

/// Write the octree of `source` (LAS/LAZ) into `dir`, replacing what is there once it is complete
/// (built in `dir` + `.tmp`). `progress` gets (done, total) points. Returns the points written.
pub fn write(
    source: &Path,
    dir: &Path,
    opt: WriteOptions,
    cancel: &AtomicBool,
    progress: &(dyn Fn(u64, u64) + Sync),
) -> Result<u64> {
    let stamp = Stamp::of(source)?;
    let mut tmp = dir.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    if tmp.exists() {
        fs::remove_dir_all(&tmp).with_context(|| format!("remove {}", tmp.display()))?;
    }
    fs::create_dir_all(tmp.join("chunks"))?;
    let result = write_into(source, &tmp, opt, CHUNK_POINTS, stamp, cancel, progress);
    let _ = fs::remove_dir_all(tmp.join("chunks"));
    let points = match result {
        Ok(n) => n,
        Err(e) => {
            let _ = fs::remove_dir_all(&tmp);
            return Err(e);
        }
    };
    if dir.exists() {
        fs::remove_dir_all(dir).with_context(|| format!("remove {}", dir.display()))?;
    }
    fs::rename(&tmp, dir).with_context(|| format!("rename to {}", dir.display()))?;
    Ok(points)
}

fn write_into(
    source: &Path,
    dir: &Path,
    opt: WriteOptions,
    chunk_points: usize,
    stamp: Stamp,
    cancel: &AtomicBool,
    progress: &(dyn Fn(u64, u64) + Sync),
) -> Result<u64> {
    const BATCH: usize = 1 << 20;
    let info = LasReader::open(source)?.info();
    let scale = info.scale;
    ensure!(
        scale[0] == scale[1] && scale[1] == scale[2] && scale[0] > 0.0,
        "{}: the axes have different scales",
        source.display()
    );
    let scale = scale[0];
    let [lo, hi] = info.bounds;
    // the cube's corner on the LAS grid, so that the integer coordinates are those of the file
    let offset: [f64; 3] = std::array::from_fn(|d| {
        let lo = if lo[d].is_finite() && lo[d] <= hi[d] {
            lo[d]
        } else {
            info.offset[d]
        };
        info.offset[d] + ((lo - info.offset[d]) / scale).floor() * scale
    });
    let extent = (0..3)
        .map(|d| ((hi[d] - offset[d]) / scale).ceil())
        .fold(1.0f64, f64::max);
    let cube = Cube { size: extent + 1.0 };
    ensure!(
        cube.size < (1u64 << 31) as f64,
        "{}: too large for its scale",
        source.display()
    );
    let keep = |p: &crate::las::LasPoint| opt.skip_class != Some(p.classification);
    let to_int = |p: &crate::las::LasPoint| -> [u32; 3] {
        std::array::from_fn(|d| {
            ((p.xyz[d] - offset[d]) / scale)
                .round()
                .clamp(0.0, cube.size - 1.0) as u32
        })
    };
    let cells = 1usize << COUNT_LEVEL;
    let cell_of = |k: [u32; 3]| {
        let s = KEY_BITS - COUNT_LEVEL;
        (((k[0] >> s) as usize * cells) + (k[1] >> s) as usize) * cells + (k[2] >> s) as usize
    };

    // the counting grid, the points kept into a file of records (read again faster than the LAZ)
    let mut rd = LasReader::open(source)?;
    let total = rd.len() as u64;
    let all_path = dir.join("chunks").join("all.bin");
    let mut all = BufWriter::with_capacity(1 << 22, File::create(&all_path)?);
    let mut counts = vec![0u32; cells * cells * cells];
    let mut read = 0u64;
    let (mut pmin, mut pmax) = ([u32::MAX; 3], [0u32; 3]);
    let (mut imin, mut imax) = (u16::MAX, 0u16);
    let (mut cmin, mut cmax) = ([u16::MAX; 3], [0u16; 3]);
    let (mut tmin, mut tmax) = (f64::INFINITY, f64::NEG_INFINITY);
    let rec = if opt.gps_time { RECORD_TIME } else { RECORD };
    let mut kept = 0u64;
    let mut records = Vec::new();
    loop {
        check(cancel)?;
        let part = rd.read(BATCH)?;
        if part.is_empty() {
            break;
        }
        read += part.len() as u64;
        records.clear();
        for p in part.iter().filter(|p| keep(p)) {
            let x = to_int(p);
            counts[cell_of(cube.key(x))] += 1;
            let mut r = [0u8; RECORD_TIME];
            for d in 0..3 {
                r[4 * d..4 * d + 4].copy_from_slice(&x[d].to_le_bytes());
                r[12 + 2 * d..14 + 2 * d].copy_from_slice(&p.rgb[d].to_le_bytes());
                pmin[d] = pmin[d].min(x[d]);
                pmax[d] = pmax[d].max(x[d]);
                cmin[d] = cmin[d].min(p.rgb[d]);
                cmax[d] = cmax[d].max(p.rgb[d]);
            }
            r[18..20].copy_from_slice(&p.intensity.to_le_bytes());
            imin = imin.min(p.intensity);
            imax = imax.max(p.intensity);
            r[20..28].copy_from_slice(&p.gps_time.to_le_bytes());
            tmin = tmin.min(p.gps_time);
            tmax = tmax.max(p.gps_time);
            records.extend_from_slice(&r[..rec]);
            kept += 1;
        }
        all.write_all(&records)?;
        // reading the source is about a quarter of the work
        progress(read / 4, total);
    }
    drop(rd);
    all.flush()?;
    drop(all);

    // chunks: the largest nodes of at most `chunk_points` (or cells of the grid)
    let mut pyramid = vec![counts];
    for l in (0..COUNT_LEVEL).rev() {
        let n = 1usize << l;
        let fine = pyramid.last().unwrap();
        let mut coarse = vec![0u32; n * n * n];
        for x in 0..2 * n {
            for y in 0..2 * n {
                for z in 0..2 * n {
                    coarse[((x / 2) * n + y / 2) * n + z / 2] += fine[(x * 2 * n + y) * 2 * n + z];
                }
            }
        }
        pyramid.push(coarse);
    }
    pyramid.reverse();
    let count_at = |l: u32, i: [usize; 3]| {
        let n = 1usize << l;
        pyramid[l as usize][(i[0] * n + i[1]) * n + i[2]]
    };
    struct Chunk {
        level: u32,
        index: [usize; 3],
    }
    let mut chunks = Vec::new();
    let mut stack = vec![(0u32, [0usize; 3])];
    while let Some((l, i)) = stack.pop() {
        let c = count_at(l, i);
        if c == 0 {
            continue;
        }
        if c as usize <= chunk_points || l == COUNT_LEVEL {
            chunks.push(Chunk { level: l, index: i });
        } else {
            for k in 0..8 {
                stack.push((
                    l + 1,
                    [
                        2 * i[0] + (k >> 2 & 1),
                        2 * i[1] + (k >> 1 & 1),
                        2 * i[2] + (k & 1),
                    ],
                ));
            }
        }
    }
    let mut chunk_of = vec![u32::MAX; cells * cells * cells];
    for (ci, c) in chunks.iter().enumerate() {
        let n = 1usize << (COUNT_LEVEL - c.level);
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let (x, y, z) = (c.index[0] * n + x, c.index[1] * n + y, c.index[2] * n + z);
                    chunk_of[(x * cells + y) * cells + z] = ci as u32;
                }
            }
        }
    }

    // the points into a file each
    let chunk_path = |ci: usize| dir.join("chunks").join(format!("{ci}.bin"));
    let mut files = chunks
        .iter()
        .enumerate()
        .map(|(ci, _)| {
            Ok(BufWriter::with_capacity(
                1 << 16,
                File::create(chunk_path(ci))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut all = File::open(&all_path)?;
    let mut block = vec![0u8; BATCH * rec];
    let mut moved = 0u64;
    loop {
        check(cancel)?;
        let n = read_full(&mut all, &mut block)?;
        if n == 0 {
            break;
        }
        for r in block[..n].chunks_exact(rec) {
            let x = std::array::from_fn(|d| {
                u32::from_le_bytes(r[4 * d..4 * d + 4].try_into().unwrap())
            });
            files[chunk_of[cell_of(cube.key(x))] as usize].write_all(r)?;
        }
        moved += (n / rec) as u64;
        progress(total / 4 + moved * total / kept.max(1) / 10, total);
    }
    for f in &mut files {
        f.flush()?;
    }
    drop(files);
    drop(all);
    let _ = fs::remove_file(&all_path);

    let octree = File::create(dir.join("octree.bin"))?;
    let b = Build {
        rgb: opt.rgb,
        gps_time: opt.gps_time,
        octree: Mutex::new((BufWriter::with_capacity(1 << 20, octree), 0)),
        entries: Mutex::new(HashMap::new()),
        cancel,
        done: AtomicU64::new(0),
    };
    let report = |b: &Build| {
        let done = b.done.load(Ordering::Relaxed);
        progress(
            total * 35 / 100 + done * 65 / 100 * total / kept.max(1),
            total,
        );
    };
    let name_of = |level: u32, index: [usize; 3]| -> Vec<u8> {
        (0..level)
            .map(|d| {
                let s = level - 1 - d;
                ((index[0] >> s & 1) << 2 | (index[1] >> s & 1) << 1 | (index[2] >> s & 1)) as u8
            })
            .collect()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(CHUNK_THREADS.min(rayon::current_num_threads()).max(1))
        .build()?;
    let roots: Vec<Pending> = pool.install(|| {
        chunks
            .par_iter()
            .enumerate()
            .map(|(ci, c)| {
                let mut bytes = Vec::new();
                File::open(chunk_path(ci))?.read_to_end(&mut bytes)?;
                let _ = fs::remove_file(chunk_path(ci));
                let points: Vec<Pt> = bytes
                    .chunks_exact(rec)
                    .map(|r| {
                        let u32_ = |q: usize| u32::from_le_bytes(r[q..q + 4].try_into().unwrap());
                        let u16_ = |q: usize| u16::from_le_bytes(r[q..q + 2].try_into().unwrap());
                        let x = [u32_(0), u32_(4), u32_(8)];
                        Pt {
                            x,
                            rgb: [u16_(12), u16_(14), u16_(16)],
                            intensity: u16_(18),
                            gps_time: if rec == RECORD_TIME {
                                f64::from_le_bytes(r[20..28].try_into().unwrap())
                            } else {
                                0.0
                            },
                            key: cube.key(x),
                        }
                    })
                    .collect();
                drop(bytes);
                let root = b.build(c.level, name_of(c.level, c.index), points)?;
                report(&b);
                Ok(root)
            })
            .collect::<Result<Vec<_>>>()
    })?;

    // the levels above the chunks: the nodes the chunks are in
    let above: std::collections::HashSet<Vec<u8>> = roots
        .iter()
        .flat_map(|r| (0..r.name.len()).map(|k| r.name[..k].to_vec()))
        .collect();
    let pending: Mutex<HashMap<Vec<u8>, Pending>> =
        Mutex::new(roots.into_iter().map(|r| (r.name.clone(), r)).collect());
    fn upper(
        b: &Build,
        above: &std::collections::HashSet<Vec<u8>>,
        pending: &Mutex<HashMap<Vec<u8>, Pending>>,
        name: Vec<u8>,
    ) -> Result<Option<Pending>> {
        if let Some(p) = pending.lock().unwrap().remove(&name) {
            return Ok(Some(p));
        }
        if !above.contains(&name) {
            return Ok(None);
        }
        let children = (0..8u8)
            .into_par_iter()
            .map(|k| {
                let mut child = name.clone();
                child.push(k);
                upper(b, above, pending, child)
            })
            .collect::<Result<Vec<_>>>()?;
        let children = children.into_iter().flatten().collect();
        b.join(name.len() as u32, name, children).map(Some)
    }
    let root = pool
        .install(|| upper(&b, &above, &pending, Vec::new()))?
        .unwrap_or(Pending {
            name: Vec::new(),
            points: Vec::new(),
            leaf: true,
            mask: 0,
        });
    b.write(root)?;
    {
        let mut o = b.octree.lock().unwrap();
        o.0.flush()?;
    }

    // the hierarchy, breadth first in one chunk
    let entries = b.entries.into_inner().unwrap();
    let mut hierarchy = Vec::with_capacity(entries.len() * HIERARCHY_ENTRY);
    let mut queue = std::collections::VecDeque::from([Vec::<u8>::new()]);
    let mut depth = 0;
    while let Some(name) = queue.pop_front() {
        let e = entries[&name];
        depth = depth.max(name.len());
        hierarchy.push(if e.leaf { 1u8 } else { 0 });
        hierarchy.push(e.mask);
        hierarchy.extend_from_slice(&e.points.to_le_bytes());
        hierarchy.extend_from_slice(&e.offset.to_le_bytes());
        hierarchy.extend_from_slice(&e.size.to_le_bytes());
        for k in 0..8u8 {
            if e.mask & (1 << k) != 0 {
                let mut child = name.clone();
                child.push(k);
                queue.push_back(child);
            }
        }
    }
    ensure!(
        hierarchy.len() == entries.len() * HIERARCHY_ENTRY,
        "octree nodes left out of the hierarchy"
    );
    fs::write(dir.join("hierarchy.bin"), &hierarchy)?;

    let world = |x: u32, d: usize| x as f64 * scale + offset[d];
    let side = cube.size * scale;
    let (pmin, pmax) = if kept == 0 {
        ([0; 3], [0; 3])
    } else {
        (pmin, pmax)
    };
    let mut attributes = vec![
        attribute(
            "position",
            "int32",
            3,
            4,
            (0..3).map(|d| world(pmin[d], d)).collect(),
            (0..3).map(|d| world(pmax[d], d)).collect(),
        ),
        attribute(
            "intensity",
            "uint16",
            1,
            2,
            vec![imin.min(imax) as f64],
            vec![imax as f64],
        ),
    ];
    if opt.rgb {
        attributes.push(attribute(
            "rgb",
            "uint16",
            3,
            2,
            (0..3).map(|d| cmin[d].min(cmax[d]) as f64).collect(),
            cmax.iter().map(|&v| v as f64).collect(),
        ));
    }
    if opt.gps_time {
        let (tmin, tmax) = if kept == 0 { (0.0, 0.0) } else { (tmin, tmax) };
        attributes.push(attribute(
            "gps-time",
            "double",
            1,
            8,
            vec![tmin],
            vec![tmax],
        ));
    }
    let meta = Metadata {
        version: "2.0".into(),
        name: stamp.source.clone(),
        description: String::new(),
        points: kept,
        projection: String::new(),
        hierarchy: HierarchyMeta {
            first_chunk_size: hierarchy.len() as u64,
            step_size: 4,
            depth: depth as u32,
        },
        offset,
        scale: [scale; 3],
        spacing: side / GRID as f64,
        bounding_box: BoxMeta {
            min: offset,
            max: offset.map(|o| o + side),
        },
        encoding: "BROTLI".into(),
        attributes,
        s20: Some(stamp),
    };
    fs::write(dir.join("metadata.json"), serde_json::to_vec_pretty(&meta)?)?;
    progress(total, total);
    Ok(kept)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Encoding {
    Default,
    Brotli,
}

/// A node of an octree read with [`Octree::open`].
#[derive(Clone, Debug)]
pub struct Node {
    pub level: u32,
    /// its place among the nodes of its level along each axis
    pub index: [u32; 3],
    pub points: u32,
    pub byte_offset: u64,
    pub byte_size: u64,
    pub parent: Option<usize>,
    /// by Potree's child index (x 4, y 2, z 1)
    pub children: [Option<usize>; 8],
}

/// The points of a node, in world coordinates.
#[derive(Clone, Debug, Default)]
pub struct Points {
    pub xyz: Vec<[f64; 3]>,
    pub intensity: Option<Vec<u16>>,
    pub rgb: Option<Vec<[u16; 3]>>,
    pub gps_time: Option<Vec<f64>>,
}

/// An octree read from its folder: the metadata and the whole hierarchy; points by node.
pub struct Octree {
    pub dir: PathBuf,
    pub points: u64,
    pub offset: [f64; 3],
    pub scale: [f64; 3],
    /// the cube's corner and size along each axis (world coordinates)
    pub min: [f64; 3],
    pub size: [f64; 3],
    /// the root's spacing (halved at each level)
    pub spacing: f64,
    pub nodes: Vec<Node>,
    pub stamp: Option<Stamp>,
    encoding: Encoding,
    attributes: Vec<AttributeMeta>,
}

impl Octree {
    pub fn open(dir: &Path) -> Result<Self> {
        let meta = read_metadata(dir)?;
        ensure!(
            meta.version == "2.0",
            "{}: Potree version {} is not supported",
            dir.display(),
            meta.version
        );
        let encoding = match meta.encoding.as_str() {
            "DEFAULT" => Encoding::Default,
            "BROTLI" => Encoding::Brotli,
            e => bail!("{}: encoding {e} is not supported", dir.display()),
        };
        ensure!(
            meta.attributes
                .first()
                .is_some_and(|a| a.name == "position" && a.ty == "int32"),
            "{}: the first attribute is not the position",
            dir.display()
        );
        let path = dir.join("hierarchy.bin");
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let mut nodes = vec![Node {
            level: 0,
            index: [0; 3],
            points: 0,
            byte_offset: 0,
            byte_size: 0,
            parent: None,
            children: [None; 8],
        }];
        // chunks of the hierarchy to read: (offset, size, the node of their first entry)
        let mut todo = vec![(0u64, meta.hierarchy.first_chunk_size, 0usize)];
        while let Some((offset, size, first)) = todo.pop() {
            let chunk = usize::try_from(offset)
                .ok()
                .zip(usize::try_from(offset + size).ok())
                .and_then(|(a, b)| bytes.get(a..b))
                .with_context(|| {
                    format!("{}: a hierarchy chunk is out of the file", path.display())
                })?;
            let mut local = vec![first];
            for (i, e) in chunk.as_chunks::<HIERARCHY_ENTRY>().0.iter().enumerate() {
                let current = *local
                    .get(i)
                    .with_context(|| format!("{}: entries without nodes", path.display()))?;
                let ty = e[0];
                let mask = e[1];
                let points = u32::from_le_bytes(e[2..6].try_into().unwrap());
                let offset = u64::from_le_bytes(e[6..14].try_into().unwrap());
                let size = u64::from_le_bytes(e[14..22].try_into().unwrap());
                nodes[current].points = points;
                if ty == 2 {
                    // a proxy: the node and its subtree are in another chunk
                    todo.push((offset, size, current));
                    continue;
                }
                nodes[current].byte_offset = offset;
                nodes[current].byte_size = size;
                if size == 0 {
                    nodes[current].points = 0;
                }
                for k in 0..8 {
                    if mask & (1 << k) == 0 {
                        continue;
                    }
                    let p = &nodes[current];
                    let index = [
                        2 * p.index[0] + (k >> 2 & 1),
                        2 * p.index[1] + (k >> 1 & 1),
                        2 * p.index[2] + (k & 1),
                    ];
                    let child = Node {
                        level: p.level + 1,
                        index,
                        points: 0,
                        byte_offset: 0,
                        byte_size: 0,
                        parent: Some(current),
                        children: [None; 8],
                    };
                    nodes.push(child);
                    let ci = nodes.len() - 1;
                    nodes[current].children[k as usize] = Some(ci);
                    local.push(ci);
                }
            }
        }
        Ok(Octree {
            dir: dir.to_path_buf(),
            points: meta.points,
            offset: meta.offset,
            scale: meta.scale,
            min: meta.bounding_box.min,
            size: std::array::from_fn(|d| meta.bounding_box.max[d] - meta.bounding_box.min[d]),
            spacing: meta.spacing,
            nodes,
            stamp: meta.s20,
            encoding,
            attributes: meta.attributes,
        })
    }

    pub fn has_rgb(&self) -> bool {
        self.attributes.iter().any(|a| a.name == "rgb")
    }

    pub fn octree_file(&self) -> PathBuf {
        self.dir.join("octree.bin")
    }

    /// The cube of node `i`: corner and size along each axis (world coordinates).
    pub fn cube(&self, i: usize) -> ([f64; 3], [f64; 3]) {
        let n = &self.nodes[i];
        let k = (1u64 << n.level) as f64;
        let size = self.size.map(|s| s / k);
        (
            std::array::from_fn(|d| self.min[d] + n.index[d] as f64 * size[d]),
            size,
        )
    }

    /// The spacing of node `i`'s points (with those of its ancestors).
    pub fn node_spacing(&self, i: usize) -> f64 {
        self.spacing / (1u64 << self.nodes[i].level) as f64
    }

    /// Read node `i`'s points from `file` (octree.bin).
    pub fn read(&self, file: &mut File, i: usize) -> Result<Points> {
        let data = self.read_bytes(file, i)?;
        self.decode(i, &data)
    }

    /// Node `i` as stored in `file` (octree.bin), to [`Octree::decode`] (apart, as it takes
    /// longer and needs no file).
    pub fn read_bytes(&self, file: &mut File, i: usize) -> Result<Vec<u8>> {
        let node = &self.nodes[i];
        let mut data = vec![0u8; node.byte_size as usize];
        file.seek(SeekFrom::Start(node.byte_offset))?;
        file.read_exact(&mut data)?;
        Ok(data)
    }

    /// The points of node `i` from its bytes.
    pub fn decode(&self, i: usize, data: &[u8]) -> Result<Points> {
        let node = &self.nodes[i];
        let n = node.points as usize;
        if n == 0 || node.byte_size == 0 {
            return Ok(Points::default());
        }
        let mut out = Points::default();
        let pos = |x: [u32; 3]| -> [f64; 3] {
            std::array::from_fn(|d| x[d] as i32 as f64 * self.scale[d] + self.offset[d])
        };
        match self.encoding {
            Encoding::Brotli => {
                let mut raw = Vec::new();
                brotli::Decompressor::new(data, 1 << 16).read_to_end(&mut raw)?;
                let mut at = 0;
                for a in &self.attributes {
                    let size = match a.name.as_str() {
                        "position" => 16,
                        "rgb" => 8,
                        _ => a.size,
                    };
                    let block = raw
                        .get(at..at + size * n)
                        .with_context(|| format!("node {i}: fewer bytes than points"))?;
                    at += size * n;
                    match a.name.as_str() {
                        "position" => {
                            out.xyz = block
                                .as_chunks::<16>()
                                .0
                                .iter()
                                .map(|b| {
                                    let upper =
                                        unmorton(u64::from_le_bytes(b[0..8].try_into().unwrap()));
                                    let lower =
                                        unmorton(u64::from_le_bytes(b[8..16].try_into().unwrap()));
                                    pos(std::array::from_fn(|d| upper[d] << 16 | lower[d]))
                                })
                                .collect();
                        }
                        "rgb" => {
                            out.rgb = Some(
                                block
                                    .as_chunks::<8>()
                                    .0
                                    .iter()
                                    .map(|b| unmorton(u64::from_le_bytes(*b)).map(|v| v as u16))
                                    .collect(),
                            );
                        }
                        "intensity" if a.ty == "uint16" => {
                            out.intensity = Some(
                                block
                                    .as_chunks::<2>()
                                    .0
                                    .iter()
                                    .map(|b| u16::from_le_bytes(*b))
                                    .collect(),
                            );
                        }
                        "gps-time" if a.ty == "double" => {
                            out.gps_time = Some(
                                block
                                    .as_chunks::<8>()
                                    .0
                                    .iter()
                                    .map(|b| f64::from_le_bytes(*b))
                                    .collect(),
                            );
                        }
                        _ => {}
                    }
                }
            }
            Encoding::Default => {
                let stride: usize = self.attributes.iter().map(|a| a.size).sum();
                ensure!(
                    data.len() >= stride * n,
                    "node {i}: fewer bytes than points"
                );
                let mut at = 0;
                for a in &self.attributes {
                    let field = |p: usize| &data[p * stride + at..p * stride + at + a.size];
                    match a.name.as_str() {
                        "position" => {
                            out.xyz = (0..n)
                                .map(|p| {
                                    let b = field(p);
                                    pos(std::array::from_fn(|d| {
                                        u32::from_le_bytes(b[4 * d..4 * d + 4].try_into().unwrap())
                                    }))
                                })
                                .collect();
                        }
                        "rgb" if a.ty == "uint16" => {
                            out.rgb = Some(
                                (0..n)
                                    .map(|p| {
                                        let b = field(p);
                                        std::array::from_fn(|d| {
                                            u16::from_le_bytes(
                                                b[2 * d..2 * d + 2].try_into().unwrap(),
                                            )
                                        })
                                    })
                                    .collect(),
                            );
                        }
                        "intensity" if a.ty == "uint16" => {
                            out.intensity = Some(
                                (0..n)
                                    .map(|p| u16::from_le_bytes(field(p).try_into().unwrap()))
                                    .collect(),
                            );
                        }
                        "gps-time" if a.ty == "double" => {
                            out.gps_time = Some(
                                (0..n)
                                    .map(|p| f64::from_le_bytes(field(p).try_into().unwrap()))
                                    .collect(),
                            );
                        }
                        _ => {}
                    }
                    at += a.size;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::las::LasPoint;

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let p = std::env::temp_dir().join(format!(
                "s20_potree_{}_{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A floor and a wall of points 5 mm apart, with a few dynamic points.
    fn cloud(n_side: usize) -> Vec<LasPoint> {
        let mut v = Vec::new();
        for i in 0..n_side {
            for j in 0..n_side {
                let (a, b) = (i as f64 * 0.005, j as f64 * 0.005);
                for (k, xyz) in [[a, b, 0.0], [a, 0.0, b]].into_iter().enumerate() {
                    v.push(LasPoint {
                        xyz: [xyz[0] + 8000.0, xyz[1] - 210_000.0, xyz[2] + 45.0],
                        intensity: ((i + j) % 200) as u16,
                        rgb: [(i * 100) as u16, (j * 100) as u16, (k * 30000) as u16],
                        classification: if (i * 7 + j) % 97 == 0 { 64 } else { 0 },
                        ..Default::default()
                    });
                }
            }
        }
        v
    }

    fn read_all(t: &Octree) -> Vec<(i64, i64, i64, u16, [u16; 3])> {
        let mut f = File::open(t.octree_file()).unwrap();
        let mut all = Vec::new();
        for i in 0..t.nodes.len() {
            let p = t.read(&mut f, i).unwrap();
            assert_eq!(p.xyz.len(), t.nodes[i].points as usize);
            let rgb = p.rgb.clone().unwrap_or_else(|| vec![[0; 3]; p.xyz.len()]);
            for ((x, inten), c) in p.xyz.iter().zip(p.intensity.unwrap()).zip(rgb) {
                // the point is inside its node's cube
                let (min, size) = t.cube(i);
                for d in 0..3 {
                    assert!(x[d] >= min[d] - 1e-6 && x[d] <= min[d] + size[d] + 1e-6);
                }
                all.push((
                    (x[0] * 1e4).round() as i64,
                    (x[1] * 1e4).round() as i64,
                    (x[2] * 1e4).round() as i64,
                    inten,
                    c,
                ));
            }
        }
        all.sort_unstable();
        all
    }

    #[test]
    fn morton_codes_invert_and_put_x_first() {
        assert_eq!(morton([1, 0, 0]), 1);
        assert_eq!(morton([0, 1, 0]), 2);
        assert_eq!(morton([0, 0, 1]), 4);
        for v in [[0, 0, 0], [65535, 1, 32768], [12345, 54321, 777]] {
            assert_eq!(unmorton(morton(v)), v);
        }
        let x = [0x1234_5678, 0x0abc_def0, 0x7fff_0001];
        let (upper, lower) = position_code(x);
        let (u, l) = (unmorton(upper), unmorton(lower));
        assert_eq!(std::array::from_fn::<u32, 3, _>(|d| u[d] << 16 | l[d]), x);
    }

    #[test]
    fn every_point_but_the_dynamic_ones_is_in_one_node_exactly() {
        let dir = Temp::new();
        let src = dir.0.join("a_colorized.laz");
        let pts = cloud(300);
        crate::las::write(&src, &pts, 1e-4, [8000.0, -210_000.0, 0.0]).unwrap();
        let out = converted_dir(&src);
        assert_eq!(out.file_name().unwrap(), "a_colorized.laz_converted");
        let opt = WriteOptions {
            rgb: true,
            skip_class: Some(64),
            ..Default::default()
        };
        // small chunks, so that there are levels above them
        fs::create_dir_all(out.join("chunks")).unwrap();
        let stamp = Stamp::of(&src).unwrap();
        let n = write_into(
            &src,
            &out,
            opt,
            5000,
            stamp,
            &AtomicBool::new(false),
            &|_, _| {},
        )
        .unwrap();
        let mut want: Vec<_> = pts
            .iter()
            .filter(|p| p.classification != 64)
            .map(|p| {
                (
                    (p.xyz[0] * 1e4).round() as i64,
                    (p.xyz[1] * 1e4).round() as i64,
                    (p.xyz[2] * 1e4).round() as i64,
                    p.intensity,
                    p.rgb,
                )
            })
            .collect();
        want.sort_unstable();
        assert_eq!(n as usize, want.len());
        let t = Octree::open(&out).unwrap();
        assert!(t.has_rgb());
        assert_eq!(t.points, n);
        assert!(t.nodes.len() > 9, "{} nodes", t.nodes.len());
        assert_eq!(read_all(&t), want);
        // a cube; leaves within their limit; the root a sample on its grid
        assert!(t.size.iter().all(|&s| (s - t.size[0]).abs() < 1e-9));
        assert!(
            t.nodes
                .iter()
                .all(|n| (n.points as usize) <= LEAF_POINTS.max((GRID * GRID * 4) as usize))
        );
        assert!(t.nodes[0].points > 1000);
        assert!(is_current(&out, &src));
        assert!(!dir.0.join("a_colorized.laz_converted.tmp").exists());
        // a changed source is told apart
        std::thread::sleep(std::time::Duration::from_millis(20));
        crate::las::write(&src, &pts[..1000], 1e-4, [8000.0, -210_000.0, 0.0]).unwrap();
        assert!(!is_current(&out, &src));
    }

    #[test]
    fn without_colour_and_rewritten_in_place() {
        let dir = Temp::new();
        let src = dir.0.join("a_uncolorized.las");
        let pts = cloud(120);
        crate::las::write(&src, &pts, 1e-4, [8000.0, -210_000.0, 0.0]).unwrap();
        let out = converted_dir(&src);
        let opt = WriteOptions::default();
        for _ in 0..2 {
            write(&src, &out, opt, &AtomicBool::new(false), &|_, _| {}).unwrap();
        }
        let t = Octree::open(&out).unwrap();
        assert!(!t.has_rgb());
        assert_eq!(read_all(&t).len(), pts.len());
        let cancel = AtomicBool::new(true);
        let e = write(&src, &out, opt, &cancel, &|_, _| {}).unwrap_err();
        assert!(crate::job::is_cancelled(&e));
        // a cancelled build leaves the octree there as it was
        assert!(is_current(&out, &src));
        assert!(!dir.0.join("a_uncolorized.las_converted.tmp").exists());
    }

    #[test]
    fn gps_times_are_kept_with_each_point_when_asked_for() {
        let dir = Temp::new();
        let src = dir.0.join("trajectory.las");
        let mut pts = cloud(150);
        for (i, p) in pts.iter_mut().enumerate() {
            p.gps_time = 1.47e9 + i as f64 * 0.1;
        }
        crate::las::write(&src, &pts, 1e-3, [8000.0, -210_000.0, 0.0]).unwrap();
        let out = converted_dir(&src);
        let key = |x: &[f64; 3], t: f64| {
            (
                (x[0] * 1e3).round() as i64,
                (x[1] * 1e3).round() as i64,
                (x[2] * 1e3).round() as i64,
                t.to_bits(),
            )
        };
        let read = |t: &Octree| {
            let mut f = File::open(t.octree_file()).unwrap();
            (0..t.nodes.len())
                .map(|i| t.read(&mut f, i).unwrap())
                .collect::<Vec<_>>()
        };
        write(
            &src,
            &out,
            WriteOptions::default(),
            &AtomicBool::new(false),
            &|_, _| {},
        )
        .unwrap();
        assert!(
            read(&Octree::open(&out).unwrap())
                .iter()
                .all(|p| p.gps_time.is_none())
        );
        // small chunks, so that there are levels above them
        fs::remove_dir_all(&out).unwrap();
        fs::create_dir_all(out.join("chunks")).unwrap();
        let opt = WriteOptions {
            gps_time: true,
            ..Default::default()
        };
        let stamp = Stamp::of(&src).unwrap();
        write_into(
            &src,
            &out,
            opt,
            5000,
            stamp,
            &AtomicBool::new(false),
            &|_, _| {},
        )
        .unwrap();
        let t = Octree::open(&out).unwrap();
        assert!(t.nodes.len() > 9, "{} nodes", t.nodes.len());
        let time = t.attributes.iter().find(|a| a.name == "gps-time").unwrap();
        assert_eq!((time.ty.as_str(), time.size), ("double", 8));
        assert_eq!(time.min, [pts[0].gps_time]);
        assert_eq!(time.max, [pts.last().unwrap().gps_time]);
        let mut got: Vec<_> = read(&t)
            .iter()
            .flat_map(|p| {
                p.xyz
                    .iter()
                    .zip(p.gps_time.as_ref().unwrap())
                    .map(|(x, &t)| key(x, t))
            })
            .collect();
        let mut want: Vec<_> = pts.iter().map(|p| key(&p.xyz, p.gps_time)).collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want);
    }
}
