//! LAS 1.4 writer, point data format 7 (XYZ, intensity, point source id, GPS time, RGB) with three float
//! extra bytes NormalX/NormalY/NormalZ: the layout of the maker's `*_uncolorized.las` / Python version.
//! A `.laz` path is written LASzip-compressed (laz-rs, chunk-parallel); the reader takes both.

use anyhow::Result;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone, Copy, Debug, Default)]
pub struct LasPoint {
    /// world coordinates (including the projection origin)
    pub xyz: [f64; 3],
    pub intensity: u16,
    pub source_id: u16,
    pub gps_time: f64,
    pub rgb: [u16; 3],
    pub normal: [f32; 3],
    pub classification: u8,
}

const HEADER_SIZE: u16 = 375;
const VLR_HEADER: usize = 54;
const EXTRA_DESC: usize = 192;
const BASE_LEN: usize = 36;

#[derive(Clone, Copy, Debug)]
pub enum ExtraKind {
    F32,
    F64,
}

impl ExtraKind {
    fn size(self) -> usize {
        match self {
            ExtraKind::F32 => 4,
            ExtraKind::F64 => 8,
        }
    }
    fn code(self) -> u8 {
        match self {
            ExtraKind::F32 => 9,
            ExtraKind::F64 => 10,
        }
    }
}

fn put_str(buf: &mut Vec<u8>, s: &str, n: usize) {
    let b = s.as_bytes();
    let k = b.len().min(n);
    buf.extend_from_slice(&b[..k]);
    buf.extend(std::iter::repeat_n(0u8, n - k));
}

fn extra_bytes_vlr(extras: &[(&str, ExtraKind)]) -> Vec<u8> {
    let mut v = Vec::new();
    for &(name, kind) in extras {
        let mut d = vec![0u8; 2];
        d.push(kind.code());
        d.push(0); // options
        put_str(&mut d, name, 32);
        d.extend([0u8; 4]);
        d.extend([0u8; 8 + 16 + 8 + 16 + 8 + 16]); // no_data, min, max (+ deprecated)
        d.extend(1.0f64.to_le_bytes()); // scale (unused: option bit not set)
        d.extend([0u8; 16]);
        d.extend(0.0f64.to_le_bytes()); // offset
        d.extend([0u8; 16]);
        put_str(&mut d, "", 32);
        debug_assert_eq!(d.len(), EXTRA_DESC);
        v.extend(d);
    }
    v
}

/// Write `pts` with normals as float extra bytes (coordinates are stored as i32 (x - offset) / scale).
pub fn write(path: &Path, pts: &[LasPoint], scale: f64, offset: [f64; 3]) -> Result<()> {
    write_extra(path, pts, scale, offset, &NORMALS, |i, b| {
        fill_normal(&pts[i], b)
    })
}

/// Write `pts` with arbitrary extra bytes, filled per point by `fill(i, bytes)`.
pub fn write_extra(
    path: &Path,
    pts: &[LasPoint],
    scale: f64,
    offset: [f64; 3],
    extras: &[(&str, ExtraKind)],
    fill: impl Fn(usize, &mut [u8]) + Sync,
) -> Result<()> {
    let mut w = LasWriter::create(path, scale, offset, extras)?;
    w.write(pts, fill)?;
    w.finish()?;
    Ok(())
}

/// The normals as extra bytes (the layout of [`write`]).
pub const NORMALS: [(&str, ExtraKind); 3] = [
    ("NormalX", ExtraKind::F32),
    ("NormalY", ExtraKind::F32),
    ("NormalZ", ExtraKind::F32),
];

/// Fills the normal extra bytes of point `p`.
pub fn fill_normal(p: &LasPoint, b: &mut [u8]) {
    for c in 0..3 {
        b[4 * c..4 * c + 4].copy_from_slice(&p.normal[c].to_le_bytes());
    }
}

enum Sink {
    Las(BufWriter<File>),
    Laz(laz::ParLasZipCompressor<BufWriter<File>>),
}

/// A LAS / LAZ file written in parts: the header (point count, bounds) is completed by
/// [`LasWriter::finish`]. The file is the same as one written at once.
pub struct LasWriter {
    sink: Option<Sink>,
    scale: f64,
    offset: [f64; 3],
    record_len: usize,
    n_extras: usize,
    /// header bytes before the counts and bounds are known
    vlrs: Vec<u8>,
    laz: bool,
    lo: [f64; 3],
    hi: [f64; 3],
    n: u64,
    /// records per batch (whole LAZ chunks per thread)
    batch: usize,
    buf: Vec<u8>,
}

impl LasWriter {
    pub fn create(
        path: &Path,
        scale: f64,
        offset: [f64; 3],
        extras: &[(&str, ExtraKind)],
    ) -> Result<Self> {
        let extra_len: usize = extras.iter().map(|e| e.1.size()).sum();
        let record_len = BASE_LEN + extra_len;
        let mut vlrs = vlr_header("LASF_Spec", 4, extras.len() * EXTRA_DESC, "extra bytes");
        vlrs.extend(extra_bytes_vlr(extras));
        let laz_vlr = if is_laz(path) {
            let v = laz::LazVlrBuilder::default()
                .with_point_format(7, extra_len as u16)?
                .build();
            let mut d = Vec::new();
            v.write_to(&mut d)?;
            vlrs.extend(vlr_header(
                laz::LazVlr::USER_ID,
                laz::LazVlr::RECORD_ID,
                d.len(),
                laz::LazVlr::DESCRIPTION,
            ));
            vlrs.extend(d);
            Some(v)
        } else {
            None
        };
        let mut w = LasWriter {
            sink: None,
            scale,
            offset,
            record_len,
            n_extras: extras.len(),
            vlrs,
            laz: laz_vlr.is_some(),
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
            n: 0,
            batch: 1 << 16,
            buf: Vec::new(),
        };
        let mut f = BufWriter::with_capacity(1 << 22, File::create(path)?);
        f.write_all(&w.header())?;
        f.write_all(&w.vlrs)?;
        w.sink = Some(match laz_vlr {
            None => Sink::Las(f),
            Some(v) => {
                w.batch = v.chunk_size() as usize * rayon::current_num_threads().max(1);
                Sink::Laz(laz::ParLasZipCompressor::new(f, v)?)
            }
        });
        Ok(w)
    }

    /// The public header block with the points written so far.
    fn header(&self) -> Vec<u8> {
        let (lo, hi) = if self.n == 0 {
            ([0.0; 3], [0.0; 3])
        } else {
            (self.lo, self.hi)
        };
        let mut h: Vec<u8> = Vec::with_capacity(HEADER_SIZE as usize);
        h.extend_from_slice(b"LASF");
        h.extend(0u16.to_le_bytes()); // file source id
        h.extend(16u16.to_le_bytes()); // global encoding: WKT bit (required for formats 6-10)
        h.extend([0u8; 16]); // GUID
        h.push(1);
        h.push(4);
        put_str(&mut h, "s20-slam", 32);
        put_str(&mut h, "s20-slam", 32);
        h.extend(1u16.to_le_bytes()); // creation day
        h.extend(2026u16.to_le_bytes());
        h.extend(HEADER_SIZE.to_le_bytes());
        h.extend(((HEADER_SIZE as usize + self.vlrs.len()) as u32).to_le_bytes());
        h.extend((1 + self.laz as u32).to_le_bytes()); // number of VLRs
        h.push(if self.laz { 0x80 | 7 } else { 7 }); // LAZ: compression bit
        h.extend((self.record_len as u16).to_le_bytes());
        h.extend(0u32.to_le_bytes()); // legacy point count
        h.extend([0u8; 20]); // legacy points by return
        for _ in 0..3 {
            h.extend(self.scale.to_le_bytes());
        }
        for o in self.offset {
            h.extend(o.to_le_bytes());
        }
        for d in 0..3 {
            h.extend(hi[d].to_le_bytes());
            h.extend(lo[d].to_le_bytes());
        }
        h.extend(0u64.to_le_bytes()); // waveform
        h.extend(0u64.to_le_bytes()); // first EVLR
        h.extend(0u32.to_le_bytes()); // number of EVLRs
        h.extend(self.n.to_le_bytes());
        h.extend(self.n.to_le_bytes()); // points by return: all first returns
        h.extend([0u8; 14 * 8]);
        debug_assert_eq!(h.len(), HEADER_SIZE as usize);
        h
    }

    /// Appends `pts`, their extra bytes filled by `fill(i, bytes)` (`i` indexes `pts`).
    pub fn write(
        &mut self,
        pts: &[LasPoint],
        fill: impl Fn(usize, &mut [u8]) + Sync,
    ) -> Result<()> {
        debug_assert!(self.n_extras > 0 || self.record_len == BASE_LEN);
        for p in pts {
            for d in 0..3 {
                self.lo[d] = self.lo[d].min(p.xyz[d]);
                self.hi[d] = self.hi[d].max(p.xyz[d]);
            }
        }
        self.n += pts.len() as u64;
        let (inv, offset, rl) = (1.0 / self.scale, self.offset, self.record_len);
        let record = |i: usize, rec: &mut [u8]| {
            let p = &pts[i];
            for d in 0..3 {
                let v = ((p.xyz[d] - offset[d]) * inv).round() as i32;
                rec[4 * d..4 * d + 4].copy_from_slice(&v.to_le_bytes());
            }
            rec[12..14].copy_from_slice(&p.intensity.to_le_bytes());
            rec[14] = 0x11; // return 1 of 1
            rec[15] = 0;
            rec[16] = p.classification;
            rec[17] = 0; // user data
            rec[18..20].copy_from_slice(&0i16.to_le_bytes());
            rec[20..22].copy_from_slice(&p.source_id.to_le_bytes());
            rec[22..30].copy_from_slice(&p.gps_time.to_le_bytes());
            for c in 0..3 {
                rec[30 + 2 * c..32 + 2 * c].copy_from_slice(&p.rgb[c].to_le_bytes());
            }
            fill(i, &mut rec[BASE_LEN..]);
        };
        for s in (0..pts.len()).step_by(self.batch) {
            let n = self.batch.min(pts.len() - s);
            self.buf.resize(n * rl, 0);
            self.buf
                .par_chunks_mut(rl)
                .enumerate()
                .for_each(|(k, rec)| record(s + k, rec));
            match self.sink.as_mut().expect("writer open") {
                Sink::Las(w) => w.write_all(&self.buf)?,
                Sink::Laz(c) => c.compress_many(&self.buf)?,
            }
        }
        Ok(())
    }

    /// Completes the file; returns the number of points.
    pub fn finish(mut self) -> Result<u64> {
        use std::io::{Seek, SeekFrom};
        let mut f = match self.sink.take().expect("writer open") {
            Sink::Las(w) => w,
            Sink::Laz(mut c) => {
                c.done()?;
                c.into_inner()
            }
        };
        f.flush()?;
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&self.header())?;
        f.flush()?;
        Ok(self.n)
    }
}

/// Whether `path` names a compressed (LAZ) file.
pub fn is_laz(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("laz"))
}

fn vlr_header(user: &str, record_id: u16, len: usize, description: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(VLR_HEADER);
    v.extend_from_slice(&0u16.to_le_bytes());
    put_str(&mut v, user, 16);
    v.extend_from_slice(&record_id.to_le_bytes());
    v.extend_from_slice(&(len as u16).to_le_bytes());
    put_str(&mut v, description, 32);
    debug_assert_eq!(v.len(), VLR_HEADER);
    v
}

/// Header values needed to rewrite a file with the same frame.
#[derive(Clone, Copy, Debug)]
pub struct LasInfo {
    pub scale: [f64; 3],
    pub offset: [f64; 3],
    pub format: u8,
    /// the header's bounds of the points: min, max
    pub bounds: [[f64; 3]; 2],
}

/// Read a LAS/LAZ 1.2-1.4 file with point format 6, 7 or 8 (RGB kept when present); float extra bytes named
/// NormalX/NormalY/NormalZ are read into `normal`.
pub fn read(path: &Path) -> Result<(Vec<LasPoint>, LasInfo)> {
    let mut r = LasReader::open(path)?;
    let pts = r.read(r.len())?;
    Ok((pts, r.info()))
}

enum Source {
    Las(std::io::BufReader<File>),
    Laz(Box<laz::ParLasZipDecompressor<std::io::BufReader<File>>>),
}

/// A LAS / LAZ file read in parts (as [`read`]).
pub struct LasReader {
    src: Source,
    info: LasInfo,
    n: usize,
    left: usize,
    rec_len: usize,
    normal_at: [Option<usize>; 3],
    buf: Vec<u8>,
}

impl LasReader {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with(path, ["NormalX", "NormalY", "NormalZ"])
    }

    /// As [`LasReader::open`], reading the float extra bytes `names` into the normals.
    pub fn open_with(path: &Path, names: [&str; 3]) -> Result<Self> {
        use anyhow::{Context, bail};
        use std::io::{Read, Seek, SeekFrom};
        let mut f = File::open(path).with_context(|| format!("read {}", path.display()))?;
        let mut b = vec![0u8; 375];
        let got = f.read(&mut b)?;
        if got < 227 || &b[0..4] != b"LASF" {
            bail!("{} is not a LAS file", path.display());
        }
        let u16_ = |b: &[u8], p: usize| u16::from_le_bytes(b[p..p + 2].try_into().unwrap());
        let u32_ = |b: &[u8], p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        let f64_ = |b: &[u8], p: usize| f64::from_le_bytes(b[p..p + 8].try_into().unwrap());
        let header_size = u16_(&b, 94) as usize;
        let data_off = u32_(&b, 96) as usize;
        // the header and the VLRs
        b.resize(data_off.max(got), 0);
        f.seek(SeekFrom::Start(0))?;
        f.read_exact(&mut b[..data_off])?;
        let n_vlr = u32_(&b, 100) as usize;
        let compressed = b[104] & 0x80 != 0;
        let format = b[104] & 0x3F;
        let rec_len = u16_(&b, 105) as usize;
        let scale = [f64_(&b, 131), f64_(&b, 139), f64_(&b, 147)];
        let offset = [f64_(&b, 155), f64_(&b, 163), f64_(&b, 171)];
        let bounds = [
            [f64_(&b, 187), f64_(&b, 203), f64_(&b, 219)],
            [f64_(&b, 179), f64_(&b, 195), f64_(&b, 211)],
        ];
        let minor = b[25];
        let n = if minor >= 4 && header_size >= 255 {
            u64::from_le_bytes(b[247..255].try_into().unwrap()) as usize
        } else {
            u32_(&b, 107) as usize
        };
        let base = match format {
            6 => 30,
            7 => 36,
            8 => 38,
            f => bail!("point format {f} is not supported"),
        };
        // extra bytes: locate the normals
        let mut normal_at: [Option<usize>; 3] = [None; 3];
        let mut laz_vlr = None;
        let mut p = header_size;
        for _ in 0..n_vlr {
            if p + 54 > data_off {
                break;
            }
            let user = String::from_utf8_lossy(&b[p + 2..p + 18])
                .trim_end_matches('\0')
                .to_string();
            let rid = u16_(&b, p + 18);
            let len = u16_(&b, p + 20) as usize;
            if user == "LASF_Spec" && rid == 4 {
                let mut at = base;
                for d in b[p + 54..p + 54 + len].as_chunks::<192>().0 {
                    let ty = d[2];
                    let name = String::from_utf8_lossy(&d[4..36])
                        .trim_end_matches('\0')
                        .to_string();
                    let size = match ty {
                        0 => d[3] as usize,
                        1 | 2 => 1,
                        3 | 4 => 2,
                        5 | 6 | 9 => 4,
                        7 | 8 | 10 => 8,
                        _ => bail!("unsupported extra bytes type {ty}"),
                    };
                    if ty == 9 {
                        for (k, nm) in names.iter().enumerate() {
                            if name == *nm {
                                normal_at[k] = Some(at);
                            }
                        }
                    }
                    at += size;
                }
            }
            if user == laz::LazVlr::USER_ID && rid == laz::LazVlr::RECORD_ID {
                laz_vlr = Some(laz::LazVlr::from_buffer(&b[p + 54..p + 54 + len])?);
            }
            p += 54 + len;
        }
        f.seek(SeekFrom::Start(data_off as u64))?;
        let r = std::io::BufReader::with_capacity(1 << 22, f);
        let src = if compressed {
            let v = laz_vlr.with_context(|| format!("{}: no LASzip VLR", path.display()))?;
            Source::Laz(Box::new(laz::ParLasZipDecompressor::new(r, v)?))
        } else {
            Source::Las(r)
        };
        Ok(LasReader {
            src,
            info: LasInfo {
                scale,
                offset,
                format,
                bounds,
            },
            n,
            left: n,
            rec_len,
            normal_at,
            buf: Vec::new(),
        })
    }

    /// Number of points in the file.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn info(&self) -> LasInfo {
        self.info
    }

    /// The next points, at most `max` (none at the end).
    pub fn read(&mut self, max: usize) -> Result<Vec<LasPoint>> {
        use std::io::Read;
        let k = max.min(self.left);
        self.left -= k;
        self.buf.resize(k * self.rec_len, 0);
        match &mut self.src {
            Source::Las(r) => r.read_exact(&mut self.buf)?,
            Source::Laz(d) => d.decompress_many(&mut self.buf)?,
        }
        let (scale, offset, format) = (self.info.scale, self.info.offset, self.info.format);
        let normal_at = self.normal_at;
        Ok(self
            .buf
            .par_chunks(self.rec_len)
            .map(|r| {
                let i32_ = |q: usize| i32::from_le_bytes(r[q..q + 4].try_into().unwrap()) as f64;
                let mut pt = LasPoint {
                    xyz: [
                        i32_(0) * scale[0] + offset[0],
                        i32_(4) * scale[1] + offset[1],
                        i32_(8) * scale[2] + offset[2],
                    ],
                    intensity: u16::from_le_bytes(r[12..14].try_into().unwrap()),
                    source_id: u16::from_le_bytes(r[20..22].try_into().unwrap()),
                    gps_time: f64::from_le_bytes(r[22..30].try_into().unwrap()),
                    classification: r[16],
                    ..Default::default()
                };
                if format >= 7 {
                    for c in 0..3 {
                        pt.rgb[c] =
                            u16::from_le_bytes(r[30 + 2 * c..32 + 2 * c].try_into().unwrap());
                    }
                }
                for (c, at) in normal_at.iter().enumerate() {
                    if let Some(a) = *at {
                        pt.normal[c] = f32::from_le_bytes(r[a..a + 4].try_into().unwrap());
                    }
                }
                pt
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join("s20_las_test.las");
        let pts: Vec<LasPoint> = (0..10)
            .map(|i| LasPoint {
                xyz: [1000.0 + i as f64 * 0.1234, -2000.5, 3.25 * i as f64],
                intensity: i as u16,
                source_id: 7,
                gps_time: 1e9 + i as f64,
                rgb: [1, 2, 3],
                classification: (i % 2) as u8 * 64,
                normal: [0.5, -0.25, i as f32],
            })
            .collect();
        write(&dir, &pts, 1e-4, [1000.0, -2000.0, 0.0]).unwrap();
        let (r, info) = read(&dir).unwrap();
        assert_eq!(info.format, 7);
        assert_eq!(r.len(), 10);
        for (a, b) in pts.iter().zip(&r) {
            for d in 0..3 {
                assert!((a.xyz[d] - b.xyz[d]).abs() < 1e-4);
            }
            assert_eq!(a.normal, b.normal);
            assert_eq!(a.gps_time, b.gps_time);
            assert_eq!(a.rgb, b.rgb);
            assert_eq!(a.classification, b.classification);
        }
    }

    /// LAZ reads back exactly what LAS stores, over several chunks and a partial last chunk.
    #[test]
    fn laz_matches_las() {
        let pts: Vec<LasPoint> = (0..120_001)
            .map(|i| {
                let f = i as f64;
                LasPoint {
                    xyz: [
                        500.0 + (f * 0.37).sin() * 20.0,
                        20.0 + f * 1e-4,
                        (f * 0.01).cos(),
                    ],
                    intensity: (i % 251) as u16,
                    source_id: (i / 1000) as u16,
                    gps_time: 1.7e9 + f * 1e-5,
                    rgb: [(i % 65536) as u16, 7, (i * 3 % 65536) as u16],
                    classification: (i % 3) as u8,
                    normal: [(f * 0.1).sin() as f32, (f * 0.1).cos() as f32, 0.5],
                }
            })
            .collect();
        let dir = std::env::temp_dir();
        let (a, b) = (dir.join("s20_las_test2.las"), dir.join("s20_las_test2.laz"));
        write(&a, &pts, 1e-4, [500.0, 0.0, 0.0]).unwrap();
        write(&b, &pts, 1e-4, [500.0, 0.0, 0.0]).unwrap();
        let (ra, _) = read(&a).unwrap();
        let (rb, ib) = read(&b).unwrap();
        assert_eq!(ib.format, 7);
        assert_eq!(ra.len(), pts.len());
        assert_eq!(rb.len(), pts.len());
        for (x, y) in ra.iter().zip(&rb) {
            assert_eq!(x.xyz, y.xyz);
            assert_eq!(
                (x.intensity, x.source_id, x.classification, x.rgb),
                (y.intensity, y.source_id, y.classification, y.rgb)
            );
            assert_eq!(x.gps_time, y.gps_time);
            assert_eq!(x.normal, y.normal);
        }
        assert!(std::fs::metadata(&b).unwrap().len() < std::fs::metadata(&a).unwrap().len());
        // written in parts: the same bytes
        for (whole, part) in [
            (&a, dir.join("s20_las_test3.las")),
            (&b, dir.join("s20_las_test3.laz")),
        ] {
            let mut w = LasWriter::create(&part, 1e-4, [500.0, 0.0, 0.0], &NORMALS).unwrap();
            for c in pts.chunks(33_333) {
                w.write(c, |i, b| fill_normal(&c[i], b)).unwrap();
            }
            assert_eq!(w.finish().unwrap(), pts.len() as u64);
            assert_eq!(std::fs::read(whole).unwrap(), std::fs::read(&part).unwrap());
            // read in parts: the same points
            let mut r = LasReader::open(&part).unwrap();
            let mut back = Vec::new();
            loop {
                let c = r.read(40_000).unwrap();
                if c.is_empty() {
                    break;
                }
                back.extend(c);
            }
            assert_eq!(back.len(), pts.len());
            for (x, y) in ra.iter().zip(&back) {
                assert_eq!(
                    (x.xyz, x.gps_time, x.normal, x.rgb),
                    (y.xyz, y.gps_time, y.normal, y.rgb)
                );
            }
        }
    }
}
