//! Colouring from the camera images (ported from slam-experiment, whose colorize_blend.py
//! reproduced PointClouds Studio's way).
//!
//! Per image (in parallel batches): the points in its field of view within `max_range`, a
//! z-buffer of 8 x 8 pixel cells, the points not behind it and facing the camera, with the
//! bilinear colour and a score of range, incidence, angle off the
//! optical axis and the device's turning speed. Per point the best `keep` views are kept
//! (merged in image order, so the result does not depend on the threads). Then: exposure gains
//! per image and channel from the cells several images share; views off the reference near view
//! (a third of the way up by brightness: a glare most views catch, and the operator's shadow a
//! few do, both lose) lose score; the best `views` are blended for the detail and all kept views
//! for each cell's colour (no seams where the camera's tone or white balance changed); a point
//! one view sees that is much brighter than its neighbours takes their colour; points no image
//! sees are dropped. (Judged by eye on the four recordings of 2026-10-28, with the refined poses:
//! a single view or fewer kept views leave seams, the median reference keeps a glare, 10 cm cells
//! leave the outline of the operator's shadow.)
//!
//! Done in squares (the square over all points, split in four while its points would take more
//! than `tile_memory` of the memory available) so that the memory follows the points of a
//! square: the views of its points and those within 0.3 m are gathered (the
//! occluders taken from the box of the camera and the square), kept on disk until the gains of
//! all images are known, then blended. The points' positions and normals stay in memory.
use crate::{BASE, Rect};
use anyhow::{Context, Result, ensure};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use s20_io::calib::FisheyeIntrinsics;
use s20_io::job::Job;
use std::path::{Path, PathBuf};

type V = Vector3<f64>;
const LUM: [f32; 3] = [0.299, 0.587, 0.114];
/// The z-buffer cell (pixels).
const ZCELL: usize = 8;
/// Points are grouped in blocks of this size (m) to skip whole blocks out of an image's view.
const BLOCK: f64 = 1.;
/// Kept colours are stored as u16 in 1/256 steps (0..256), before the exposure gains.
const CSCALE: f32 = 256.;
fn pack(c: [f32; 3]) -> [u16; 3] {
    c.map(|x| (x * CSCALE).round().clamp(0., 65535.) as u16)
}
fn unpack(c: [u16; 3], g: [f32; 3]) -> [f32; 3] {
    [
        c[0] as f32 / CSCALE * g[0],
        c[1] as f32 / CSCALE * g[1],
        c[2] as f32 / CSCALE * g[2],
    ]
}
/// No image in a kept slot.
const NO_IMAGE: u16 = u16::MAX;
/// The squares are counted in cells of this many [`BASE`] cells.
const SMALL: i64 = 100;
/// The points around a square whose views it gathers too ([`BASE`] cells): the cells of the
/// outlier test around its points (2 x `cell`) and the neighbours of the fill.
const PART_MARGIN: i64 = 15;
/// The memory of a point of a part while its views are gathered (bytes; the shrine 2026-09-23:
/// 18.5 GB more for 65 M points more, and 32 for the cells around it), and the points of a part
/// when the memory available is not known.
const BYTES_PER_POINT: f64 = 330.;
const FALLBACK_POINTS: usize = 12_000_000;
/// The z-buffer is filled where the coloured points of an image fall, grown by this many cells:
/// the occlusion test of a point looks up to 6 cells around it.
const ZB_MARGIN: i64 = 8;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ColorParams {
    /// Views blended per point.
    pub views: usize,
    /// Views kept per point before the outlier test.
    pub keep: usize,
    /// Beyond this angle off the optical axis (deg) a view is used only for a point no other sees.
    pub max_theta_deg: f64,
    /// Beyond this angle (deg) the score falls by `theta_slope` per radian.
    pub soft_theta_deg: f64,
    pub theta_slope: f64,
    /// Score lost per 30 deg/s of the device's turning (motion blur).
    pub turn_weight: f64,
    /// Sum |dRGB| of a view's cell mean off the reference at which the penalty is ln 2 x penalty (0: off).
    pub outlier: f64,
    pub outlier_penalty: f64,
    /// Cells (m, x facing) of the outlier test and the exposure gains.
    pub cell: f64,
    /// exp((score - best) / temperature) weighting of the blended views.
    pub temperature: f64,
    /// Blend in two bands (0: off): the detail (a view's colour off its cell's mean) from the
    /// blended views, the cell's colour from all kept views weighted by exp((score - best) /
    /// band_temperature), so that views of another tone or white balance leave no seams.
    pub band_temperature: f64,
    /// The outlier reference is taken among the views within this of the best score.
    pub dark_window: f64,
    /// The outlier reference is the view at this fraction of those views ordered by brightness
    /// (0.5 the median, so that a shadow or a glare in a few of them both lose; 0: the second
    /// darkest, so that only a glare loses).
    pub outlier_quantile: f64,
    /// Radius (m) of the neighbours a bright single-view point takes its colour from (0: off).
    pub fill: f64,
    /// Exposure / white balance gain per image and channel.
    pub gain: bool,
    /// Only the gains' differences from the images within this time window (s) are applied: the
    /// steps between neighbouring images and between the two cameras are levelled, while the
    /// exposure the camera chose for a place (darker in the sun, brighter in the shade) stays
    /// (0: the gains levelled over all images, which washes sunlit images out where most were
    /// taken in the shade; the shrine 2026-10-22).
    pub gain_window: f64,
    /// Only points within this distance (m) of a camera are coloured from it.
    pub max_range: f64,
    /// Point spacing (m): sets how many z-buffer cells around a point may hide it.
    pub spacing: f64,
    /// Allow the depth slope of a surface seen at a grazing angle in the occlusion test.
    pub graze: bool,
    /// The points are coloured in squares in xy, split in four while the points of one (with
    /// its margin) would take more than this fraction of the memory available (0: all at once);
    /// the result is the same. Fewer squares decode each image fewer times (the shrine
    /// 2026-09-23, 77 M points: 27 squares 534 s and 17 GB, all at once 438 s and 36 GB).
    pub tile_memory: f64,
}
impl Default for ColorParams {
    fn default() -> Self {
        Self {
            views: 3,
            keep: 8,
            max_theta_deg: 78.,
            soft_theta_deg: 65.,
            theta_slope: 20.,
            turn_weight: 2.,
            outlier: 80.,
            outlier_penalty: 3.,
            cell: 0.05,
            temperature: 2.,
            band_temperature: 8.,
            dark_window: 8.,
            outlier_quantile: 0.3,
            fill: 0.04,
            gain: true,
            gain_window: 60.,
            max_range: 40.,
            spacing: 0.007,
            graze: true,
            tile_memory: 0.5,
        }
    }
}

/// One image: path, camera side, position and camera -> world rotation, time.
struct Pose {
    name: String,
    left: bool,
    t: V,
    /// world -> camera (the transpose of the camera -> world rotation of ImgPose.txt)
    rcw: Matrix3<f64>,
    time: f64,
}

/// `ImgPose.txt`: name, x y z, roll pitch yaw, qx qy qz qw (camera to world, OpenCV camera
/// axes), timestamp; in the cloud's coordinates.
fn load_poses(imgpose: &Path) -> Result<Vec<Pose>> {
    let text =
        std::fs::read_to_string(imgpose).with_context(|| format!("read {}", imgpose.display()))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with("index") || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        ensure!(f.len() >= 12, "ImgPose.txt: short line {line}");
        let x = |i: usize| {
            f[i].parse::<f64>()
                .with_context(|| format!("ImgPose.txt: {line}"))
        };
        let q = UnitQuaternion::from_quaternion(Quaternion::new(x(10)?, x(7)?, x(8)?, x(9)?));
        let base = Path::new(f[0])
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        out.push(Pose {
            name: f[0].to_string(),
            left: base.starts_with("left"),
            t: V::new(x(1)?, x(2)?, x(3)?),
            rcw: q.to_rotation_matrix().matrix().transpose(),
            time: x(11)?,
        });
    }
    Ok(out)
}

/// Turning speed (deg/s) of the device around each image's time, from `trajectory.txt` of the
/// export step (x y z roll pitch yaw qx qy qz qw t, 10 Hz; the mean over the poses from two
/// before to two after).
fn turning(trajectory: &Path, poses: &[Pose]) -> Result<Vec<f64>> {
    let text = std::fs::read_to_string(trajectory)
        .with_context(|| format!("read {}", trajectory.display()))?;
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect()
        })
        .filter(|r: &Vec<f64>| r.len() >= 11)
        .collect();
    let ft: Vec<f64> = rows.iter().map(|r| r[10]).collect();
    let fq: Vec<UnitQuaternion<f64>> = rows
        .iter()
        .map(|r| UnitQuaternion::from_quaternion(Quaternion::new(r[9], r[6], r[7], r[8])))
        .collect();
    let mut wr: Vec<f64> = (1..fq.len())
        .map(|i| (fq[i] * fq[i - 1].inverse()).angle().to_degrees() / (ft[i] - ft[i - 1]))
        .collect();
    wr.push(0.);
    Ok(poses
        .iter()
        .map(|p| {
            let j = ft.partition_point(|&t| t < p.time);
            let (a, b) = (j.saturating_sub(2), (j + 2).min(wr.len()));
            if a < b {
                wr[a..b].iter().sum::<f64>() / (b - a) as f64
            } else {
                0.
            }
        })
        .collect())
}

/// RGB pixels of a JPEG and its width.
fn decode(data: &[u8]) -> Result<(Vec<u8>, usize)> {
    let mut d = zune_jpeg::JpegDecoder::new_with_options(
        zune_core::bytestream::ZCursor::new(data),
        zune_core::options::DecoderOptions::default()
            .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGB),
    );
    let px = d.decode().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let w = d.info().context("no image size")?.width as usize;
    ensure!(px.len() % (3 * w.max(1)) == 0, "not RGB");
    Ok((px, w))
}

/// Pixel (u, v) and the angle off the optical axis of a camera-frame point (polynomial fisheye).
fn project(c: &FisheyeIntrinsics, x: V) -> (f64, f64, f64) {
    let rho = x.x.hypot(x.y);
    let th = rho.atan2(x.z);
    let k = &c.k;
    let rr =
        th * (1. + th * (k[0] + th * (k[1] + th * (k[2] + th * (k[3] + th * (k[4] + th * k[5]))))));
    let (cp, sp) = if rho > 1e-9 {
        (x.x / rho, x.y / rho)
    } else {
        (1., 0.)
    };
    (
        c.a11 * rr * cp + c.a12 * rr * sp + c.u0,
        c.a22 * rr * sp + c.v0,
        th,
    )
}

/// The depth change per z-buffer cell around cell (ci, cj), summed over the two image axes: on
/// each axis the cells on both sides must be valid and run on smoothly (their second difference
/// within half the first), else that axis adds nothing.
fn depth_slope(zb: &[f32], zw: usize, zh: usize, ci: i64, cj: i64) -> f64 {
    let at = |i: i64, j: i64| -> Option<f64> {
        (i >= 0 && j >= 0 && (i as usize) < zh && (j as usize) < zw)
            .then(|| zb[i as usize * zw + j as usize] as f64)
            .filter(|z| z.is_finite())
    };
    let axis = |a: Option<f64>, c: Option<f64>, b: Option<f64>| -> f64 {
        let (Some(a), Some(c), Some(b)) = (a, c, b) else {
            return 0.;
        };
        let first = (b - a).abs() / 2.;
        if (a + b - 2. * c).abs() <= 0.5 * first {
            first
        } else {
            0.
        }
    };
    axis(at(ci - 1, cj), at(ci, cj), at(ci + 1, cj))
        + axis(at(ci, cj - 1), at(ci, cj), at(ci, cj + 1))
}

/// The points projected at a time by [`Lens::project`].
const LANES: usize = 16;

/// A camera's projection in single precision ([`project`] with the range and the field of view
/// of a view; u, v within about 1e-4 pixel), the points of [`LANES`] at a time.
struct Lens {
    r: [[f32; 3]; 3],
    k: [f32; 6],
    a11: f32,
    a12: f32,
    a22: f32,
    u0: f32,
    v0: f32,
    umax: f32,
    vmax: f32,
    max_range: f32,
    limit: f32,
}
/// Per point of [`Lens::project`]: u, v, range, angle off the axis, and whether it is within
/// range, the field of view and the image.
#[derive(Default)]
struct Lanes {
    u: [f32; LANES],
    v: [f32; LANES],
    r: [f32; LANES],
    th: [f32; LANES],
    ok: [bool; LANES],
}
/// atan(x) for x in [0, 1] (Cephes' atanf: within 2 ulp).
#[inline(always)]
fn atan01(x: f32) -> f32 {
    let big = x > 0.414_213_57;
    let y = if big { (x - 1.) / (x + 1.) } else { x };
    let z = y * y;
    // the coefficients as published
    #[allow(clippy::excessive_precision)]
    let p =
        (((8.053_744_5e-2 * z - 1.387_768_6e-1) * z + 1.997_771_1e-1) * z - 3.333_294_9e-1) * z * y
            + y;
    if big {
        p + std::f32::consts::FRAC_PI_4
    } else {
        p
    }
}
impl Lens {
    fn new(c: &FisheyeIntrinsics, pose: &Pose, max_range: f64, limit: f64) -> Self {
        let f = |x: f64| x as f32;
        Lens {
            r: [0, 1, 2].map(|i| [0, 1, 2].map(|j| f(pose.rcw[(i, j)]))),
            k: c.k.map(f),
            a11: f(c.a11),
            a12: f(c.a12),
            a22: f(c.a22),
            u0: f(c.u0),
            v0: f(c.v0),
            umax: (c.width - 2) as f32,
            vmax: (c.height - 2) as f32,
            max_range: f(max_range),
            limit: f(limit),
        }
    }
    /// The points `d` (from the camera, in the world's axes).
    fn project(&self, d: &[[f32; LANES]; 3], out: &mut Lanes) {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU has AVX2
            return unsafe { self.project_avx2(d, out) };
        }
        self.lanes(d, out)
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn project_avx2(&self, d: &[[f32; LANES]; 3], out: &mut Lanes) {
        self.lanes(d, out)
    }
    /// (without branches, so that the compiler makes it vector instructions)
    #[inline(always)]
    #[allow(clippy::needless_range_loop)] // `l` indexes the lanes of `d` and `out` together (vectorised)
    fn lanes(&self, d: &[[f32; LANES]; 3], out: &mut Lanes) {
        use std::f32::consts::{FRAC_PI_2, PI};
        let (r, k) = (&self.r, &self.k);
        for l in 0..LANES {
            let (dx, dy, dz) = (d[0][l], d[1][l], d[2][l]);
            let range = (dx * dx + dy * dy + dz * dz).sqrt();
            let x = r[0][0] * dx + r[0][1] * dy + r[0][2] * dz;
            let y = r[1][0] * dx + r[1][1] * dy + r[1][2] * dz;
            let z = r[2][0] * dx + r[2][1] * dy + r[2][2] * dz;
            let rho = (x * x + y * y).sqrt();
            // theta = atan2(rho, z)
            let az = z.abs();
            let (lo, hi) = (rho.min(az), rho.max(az));
            let q = if hi > 0. {
                lo / hi.max(f32::MIN_POSITIVE)
            } else {
                0.
            };
            let a = atan01(q);
            let a = if rho > az { FRAC_PI_2 - a } else { a };
            let th = if z < 0. { PI - a } else { a };
            let rr = th
                * (1.
                    + th * (k[0]
                        + th * (k[1] + th * (k[2] + th * (k[3] + th * (k[4] + th * k[5]))))));
            let tiny = rho <= 1e-9;
            let inv = 1. / rho.max(1e-9);
            let (cp, sp) = (
                if tiny { 1. } else { x * inv },
                if tiny { 0. } else { y * inv },
            );
            let u = self.a11 * rr * cp + self.a12 * rr * sp + self.u0;
            let v = self.a22 * rr * sp + self.v0;
            out.u[l] = u;
            out.v[l] = v;
            out.r[l] = range;
            out.th[l] = th;
            out.ok[l] = (range <= self.max_range)
                & (th < self.limit)
                & (u >= 1.)
                & (u < self.umax)
                & (v >= 1.)
                & (v < self.vmax);
        }
    }
}

/// Per cell of the w x h grid `z` the least within 1, 2, ... `n` cells around it (the windows
/// clipped to the grid): each the least of the previous within 1 cell.
fn window_minima(z: &[f32], w: usize, h: usize, n: usize) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut row = vec![0f32; w * h];
    for _ in 0..n {
        let prev = out.last().map_or(z, |v| v.as_slice());
        for i in 0..h {
            let line = &prev[i * w..(i + 1) * w];
            for j in 0..w {
                let (a, b) = (j.saturating_sub(1), (j + 1).min(w - 1));
                row[i * w + j] = line[a].min(line[j]).min(line[b]);
            }
        }
        let mut m = vec![0f32; w * h];
        for i in 0..h {
            let (a, b) = (i.saturating_sub(1), (i + 1).min(h - 1));
            for j in 0..w {
                m[i * w + j] = row[a * w + j].min(row[i * w + j]).min(row[b * w + j]);
            }
        }
        out.push(m);
    }
    out
}

/// Seconds of the views summed over the threads (nanoseconds): the coloured candidates, the
/// z-buffer, the occlusion test, reading and decoding the image, the colours and scores; and
/// the images whose view was empty or decoded.
static SECONDS: [std::sync::atomic::AtomicU64; 7] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 7];
const STAGE_NAMES: [&str; 5] = ["candidates", "z-buffer", "occlusion", "decode", "colours"];
fn lap(k: usize, t: &mut std::time::Instant) {
    SECONDS[k].fetch_add(
        t.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    *t = std::time::Instant::now();
}

/// The coloured points an image sees: (block order, u, v, range, theta).
type Seen = Vec<(u32, f32, f32, f32, f32)>;

/// The points of an image's view: slots of the points coloured, score, colour, and the views
/// beyond `max_theta`.
struct View {
    pts: Vec<u32>,
    score: Vec<f32>,
    rgb: Vec<[f32; 3]>,
    wide: Vec<bool>,
}

/// Points sorted into 1 m blocks: (point order, block start / end, block centres), and the
/// points' positions and normals in that order (read in turn, not across all memory: the
/// candidates and their occlusion tests take a tenth less time).
struct Blocks {
    order: Vec<u32>,
    bounds: Vec<(usize, usize)>,
    centres: Vec<V>,
    p: Vec<[f32; 3]>,
    n: Vec<[f32; 3]>,
}
fn blocks(p: &[[f32; 3]], n: &[[f32; 3]]) -> Blocks {
    let key = |x: &[f32; 3]| x.map(|v| (v as f64 / BLOCK).floor() as i64);
    let mut order: Vec<u32> = (0..p.len() as u32).collect();
    order.par_sort_unstable_by_key(|&i| (key(&p[i as usize]), i));
    let mut bounds = Vec::new();
    let mut centres = Vec::new();
    let mut a = 0;
    while a < order.len() {
        let k = key(&p[order[a] as usize]);
        let mut b = a + 1;
        while b < order.len() && key(&p[order[b] as usize]) == k {
            b += 1;
        }
        bounds.push((a, b));
        centres.push(V::new(
            (k[0] as f64 + 0.5) * BLOCK,
            (k[1] as f64 + 0.5) * BLOCK,
            (k[2] as f64 + 0.5) * BLOCK,
        ));
        a = b;
    }
    let (p, n) = order
        .par_iter()
        .map(|&i| (p[i as usize], n[i as usize]))
        .unzip();
    Blocks {
        order,
        bounds,
        centres,
        p,
        n,
    }
}

fn vf(x: [f32; 3]) -> V {
    V::new(x[0] as f64, x[1] as f64, x[2] as f64)
}

/// The points (local frame, see [`colorize`]), their normals and blocks, and the cameras.
struct Scene<'a> {
    p: &'a [[f32; 3]],
    n: &'a [[f32; 3]],
    blocks: Blocks,
    images: PathBuf,
    left: FisheyeIntrinsics,
    right: FisheyeIntrinsics,
    /// the points in block order on the GPU for the z-buffers (None: on the CPU)
    gpu: Option<crate::gpu::ZBuffer>,
}

/// The points a pass colours: `slot` of each point (u32::MAX: not coloured, an occluder at most),
/// and the xy box (m) they lie in.
struct Coloured<'a> {
    slot: &'a [u32],
    lo: [f64; 2],
    hi: [f64; 2],
}

impl Scene<'_> {
    fn view(&self, pose: &Pose, omega: f64, a: &ColorParams, c: &Coloured) -> Result<Option<View>> {
        let cam = if pose.left { &self.left } else { &self.right };
        let (w, h) = (cam.width as usize, cam.height as usize);
        let half_diag = 0.5 * 3f64.sqrt() * BLOCK;
        let limit = 100f64.to_radians();
        // A block's points within range whose cone reaches the field of view: (centre, angular
        // radius) on the image, or None.
        let reach = |cb: &V| -> Option<(f64, f64, f64)> {
            let d = cb - pose.t;
            let dist = d.norm();
            if dist > a.max_range + half_diag {
                return None;
            }
            let xc = pose.rcw * d;
            let spread = if dist > half_diag {
                (half_diag / dist).asin()
            } else {
                std::f64::consts::PI
            };
            let (u, v, th) = project(cam, xc);
            (th - spread <= limit).then_some((u, v, spread))
        };
        // (by position in the block order)
        let point = |j: usize| -> Option<(f32, f32, f32, f32)> {
            let x = vf(self.blocks.p[j]);
            let r = (x - pose.t).norm();
            if r > a.max_range {
                return None;
            }
            let (u, v, th) = project(cam, pose.rcw * (x - pose.t));
            (th < limit && u >= 1. && u < (w - 2) as f64 && v >= 1. && v < (h - 2) as f64)
                .then_some((u as f32, v as f32, r as f32, th as f32))
        };
        let (p, n) = (&self.blocks.p, &self.blocks.n);
        let has_n = |i: usize| {
            let m = n[i];
            m[0] * m[0] + m[1] * m[1] + m[2] * m[2] > 0.25
        };
        let mut t = std::time::Instant::now();
        // on the GPU when there is one (see `seen_on_gpu`), else:
        let on_gpu = self
            .gpu
            .as_ref()
            .filter(|g| g.no_normals.is_none())
            .and_then(|g| self.seen_on_gpu(g, pose, a, c, limit).ok());
        let seen: Seen = if let Some(seen) = on_gpu {
            lap(1, &mut t);
            if seen.is_empty() {
                SECONDS[5].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(None);
            }
            seen
        } else {
            // The coloured points it sees (from the blocks of their box) ...
            let mut cand: Vec<(u32, f32, f32, f32, f32)> = Vec::new(); // block order, u, v, r, theta
            let lens = Lens::new(cam, pose, a.max_range, limit);
            let mut out = Lanes::default();
            for (b, cb) in self.blocks.centres.iter().enumerate() {
                if (0..2).any(|d| cb[d] < c.lo[d] - half_diag || cb[d] > c.hi[d] + half_diag)
                    || reach(cb).is_none()
                {
                    continue;
                }
                let (s, e) = self.blocks.bounds[b];
                let (centre, to) = (
                    [0, 1, 2].map(|d| cb[d] as f32),
                    [0, 1, 2].map(|d| (cb[d] - pose.t[d]) as f32),
                );
                for at in (s..e).step_by(LANES) {
                    let m = (e - at).min(LANES);
                    // from the camera, as the block's centre plus the offset from it (exact in
                    // single precision)
                    let mut d = [[0f32; LANES]; 3];
                    // `l` fills one lane of the three coordinates (kept as plain lanes for speed)
                    #[allow(clippy::needless_range_loop)]
                    for l in 0..LANES {
                        let q = self.blocks.p[at + l.min(m - 1)];
                        for k in 0..3 {
                            d[k][l] = (q[k] - centre[k]) + to[k];
                        }
                    }
                    lens.project(&d, &mut out);
                    for l in 0..m {
                        if out.ok[l] && c.slot[self.blocks.order[at + l] as usize] != u32::MAX {
                            cand.push(((at + l) as u32, out.u[l], out.v[l], out.r[l], out.th[l]));
                        }
                    }
                }
            }
            lap(0, &mut t);
            if cand.is_empty() {
                SECONDS[5].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(None);
            }
            // ... and the z-buffer of all points where they fall, grown by ZB_MARGIN cells: the
            // cells their occlusion tests look at are as with all the points.
            let (zw, zh) = (w / ZCELL + 1, h / ZCELL + 1);
            let (mut clo, mut chi) = ([i64::MAX; 2], [i64::MIN; 2]);
            for &(_, u, v, _, _) in &cand {
                let k = [u as i64 / ZCELL as i64, v as i64 / ZCELL as i64];
                for d in 0..2 {
                    clo[d] = clo[d].min(k[d] - ZB_MARGIN);
                    chi[d] = chi[d].max(k[d] + ZB_MARGIN);
                }
            }
            let (plo, phi) = (
                clo.map(|x| (x * ZCELL as i64) as f64),
                chi.map(|x| ((x + 1) * ZCELL as i64) as f64),
            );
            // the blocks whose image, a disc of their angular radius (1.5 times for the lens), meets
            // the cells looked at
            let near: Vec<usize> = (0..self.blocks.centres.len())
                .filter(|&b| {
                    let Some((u, v, spread)) = reach(&self.blocks.centres[b]) else {
                        return false;
                    };
                    let rad = 1.5 * cam.a11 * spread + 2. * ZCELL as f64;
                    spread >= 1.
                        || !(u + rad < plo[0]
                            || u - rad > phi[0]
                            || v + rad < plo[1]
                            || v - rad > phi[1])
                })
                .collect();
            let cpu = || {
                let mut zb = vec![f32::INFINITY; zw * zh];
                for &b in &near {
                    let (s, e) = self.blocks.bounds[b];
                    for j in s..e {
                        if let Some((u, v, r, _)) = point(j) {
                            let k = [u as i64 / ZCELL as i64, v as i64 / ZCELL as i64];
                            if (0..2).all(|d| k[d] >= clo[d] && k[d] <= chi[d]) {
                                let cell = &mut zb[k[1] as usize * zw + k[0] as usize];
                                *cell = cell.min(r);
                            }
                        }
                    }
                }
                zb
            };
            let zb = match &self.gpu {
                Some(g) => {
                    // (single precision there: a point at a cell's border may fall on the other side)
                    let ranges: Vec<(u32, u32)> = near
                        .iter()
                        .map(|&b| self.blocks.bounds[b])
                        .map(|(s, e)| (s as u32, (e - s) as u32))
                        .collect();
                    g.fill(&self.zcamera(pose, a, limit), &ranges, zw, zh, clo, chi)
                        .unwrap_or_else(|_| cpu())
                }
                None => cpu(),
            };
            lap(1, &mut t);
            // the nearest within 1 to 6 cells around each cell, and the depth slopes
            let mins = window_minima(&zb, zw, zh, 6);
            let slopes: Vec<f64> = if a.graze {
                (0..zw * zh)
                    .map(|o| depth_slope(&zb, zw, zh, (o / zw) as i64, (o % zw) as i64))
                    .collect()
            } else {
                Vec::new()
            };
            let seen: Vec<_> = cand
                .into_iter()
                .filter(|&(i, u, v, r, _)| {
                    let (u, v, r) = (u as f64, v as f64, r as f64);
                    let reach = (cam.a11 * a.spacing / r.max(1e-3) / ZCELL as f64)
                        .ceil()
                        .clamp(1., 6.) as usize;
                    let cell = (v / ZCELL as f64) as usize * zw + (u / ZCELL as f64) as usize;
                    let near = mins[reach - 1][cell];
                    // A surface seen at a grazing angle (the ground far away) comes nearer across the
                    // window by its depth slope: allowed where the depths of the cells around run on
                    // smoothly, not where they jump (an occluder's edge).
                    let slope = if a.graze {
                        slopes[cell] * (reach + 1) as f64
                    } else {
                        0.
                    };
                    if r > near as f64 + 0.03f64.max(0.02 * r) + slope {
                        return false;
                    }
                    let i = i as usize;
                    if !has_n(i) {
                        return true;
                    }
                    let m = n[i];
                    let ray = pose.t - vf(p[i]);
                    m[0] as f64 * ray.x + m[1] as f64 * ray.y + m[2] as f64 * ray.z > 0.
                })
                .collect();
            lap(2, &mut t);
            seen
        };
        if seen.is_empty() {
            return Ok(None);
        }
        SECONDS[6].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let img_path = Path::new(&pose.name);
        let img_path = if img_path.is_absolute() {
            img_path.to_path_buf()
        } else {
            self.images.join(img_path)
        };
        let (img, iw) = decode(&s20_io::dataset::read_image(&img_path)?)
            .with_context(|| format!("decode {}", img_path.display()))?;
        let px = |x: usize, y: usize| {
            let o = 3 * (y * iw + x);
            [img[o] as f32, img[o + 1] as f32, img[o + 2] as f32]
        };
        lap(3, &mut t);
        let (soft, hard) = (a.soft_theta_deg.to_radians(), a.max_theta_deg.to_radians());
        let mut out = View {
            pts: Vec::with_capacity(seen.len()),
            score: Vec::new(),
            rgb: Vec::new(),
            wide: Vec::new(),
        };
        for (i, u, v, r, th) in seen {
            let (u, v, r, th) = (u as f64, v as f64, r as f64, th as f64);
            let (u0, v0) = (u.floor() as usize, v.floor() as usize);
            let (fu, fv) = ((u - u0 as f64) as f32, (v - v0 as f64) as f32);
            let (a00, a10, a01, a11) = (
                px(u0, v0),
                px(u0 + 1, v0),
                px(u0, v0 + 1),
                px(u0 + 1, v0 + 1),
            );
            let rgb: [f32; 3] = std::array::from_fn(|c| {
                a00[c] * (1. - fu) * (1. - fv)
                    + a10[c] * fu * (1. - fv)
                    + a01[c] * (1. - fu) * fv
                    + a11[c] * fu * fv
            });
            let iu = i as usize;
            let cos = if has_n(iu) {
                let m = n[iu];
                let ray = pose.t - vf(p[iu]);
                (m[0] as f64 * ray.x + m[1] as f64 * ray.y + m[2] as f64 * ray.z).abs()
                    / r.max(1e-6)
            } else {
                0.5
            };
            let s = 5.15 * -r.max(0.1).ln() + 10.6 * cos
                - 4.05 * th
                - a.turn_weight * omega / 30.
                - a.theta_slope * (th - soft).max(0.);
            out.pts.push(c.slot[self.blocks.order[iu] as usize]);
            out.score.push(s as f32);
            out.rgb.push(rgb);
            out.wide.push(th > hard);
        }
        lap(4, &mut t);
        Ok(Some(out))
    }

    /// The coloured points the image sees, on the GPU: the z-buffer over the whole image, its
    /// window minima and depth slopes on the CPU, then the points of the blocks of the coloured
    /// points' box tested there (as the CPU's path in [`Scene::view`]).
    fn seen_on_gpu(
        &self,
        g: &crate::gpu::ZBuffer,
        pose: &Pose,
        a: &ColorParams,
        c: &Coloured,
        limit: f64,
    ) -> Result<Seen> {
        let cam = if pose.left { &self.left } else { &self.right };
        let (w, h) = (cam.width as usize, cam.height as usize);
        let half_diag = 0.5 * 3f64.sqrt() * BLOCK;
        let (zw, zh) = (w / ZCELL + 1, h / ZCELL + 1);
        let (plo, phi) = ([0., 0.], [(zw * ZCELL) as f64, (zh * ZCELL) as f64]);
        let mut near = Vec::new();
        let mut blocks = Vec::new();
        for (b, cb) in self.blocks.centres.iter().enumerate() {
            let d = cb - pose.t;
            let dist = d.norm();
            if dist > a.max_range + half_diag {
                continue;
            }
            let spread = if dist > half_diag {
                (half_diag / dist).asin()
            } else {
                std::f64::consts::PI
            };
            let (u, v, th) = project(cam, pose.rcw * d);
            if th - spread > limit {
                continue;
            }
            let (s, e) = self.blocks.bounds[b];
            let rad = 1.5 * cam.a11 * spread + 2. * ZCELL as f64;
            if spread >= 1.
                || !(u + rad < plo[0] || u - rad > phi[0] || v + rad < plo[1] || v - rad > phi[1])
            {
                near.push((s as u32, (e - s) as u32));
            }
            if (0..2).all(|k| cb[k] >= c.lo[k] - half_diag && cb[k] <= c.hi[k] + half_diag) {
                blocks.push(crate::gpu::Block {
                    first: s as u32,
                    count: (e - s) as u32,
                    centre: [0, 1, 2].map(|k| cb[k] as f32),
                    to: [0, 1, 2].map(|k| d[k] as f32),
                });
            }
        }
        if blocks.is_empty() {
            return Ok(Vec::new());
        }
        let zc = self.zcamera(pose, a, limit);
        let zb = g.fill(&zc, &near, zw, zh, [0, 0], [zw as i64 - 1, zh as i64 - 1])?;
        let mut grids: Vec<f32> = window_minima(&zb, zw, zh, 6).concat();
        if a.graze {
            grids.extend(
                (0..zw * zh)
                    .map(|o| depth_slope(&zb, zw, zh, (o / zw) as i64, (o % zw) as i64) as f32),
            );
        } else {
            grids.extend(std::iter::repeat_n(0f32, zw * zh));
        }
        let reach = (cam.a11 * a.spacing / ZCELL as f64) as f32;
        let mut seen = g.visible(&zc, reach, a.graze, &blocks, zw, zh, &grids)?;
        seen.retain(|s| c.slot[self.blocks.order[s.0 as usize] as usize] != u32::MAX);
        Ok(seen)
    }

    /// The single precision camera of the GPU's passes.
    fn zcamera(&self, pose: &Pose, a: &ColorParams, limit: f64) -> crate::gpu::ZCamera {
        let cam = if pose.left { &self.left } else { &self.right };
        let f = |x: f64| x as f32;
        crate::gpu::ZCamera {
            rcw: [0, 1, 2].map(|r| [0, 1, 2].map(|c| f(pose.rcw[(r, c)]))),
            t: [0, 1, 2].map(|d| f(pose.t[d])),
            k: cam.k.map(f),
            a11: f(cam.a11),
            a12: f(cam.a12),
            a22: f(cam.a22),
            u0: f(cam.u0),
            v0: f(cam.v0),
            w: cam.width,
            h: cam.height,
            max_range: f(a.max_range),
            limit: f(limit),
        }
    }
}

/// Cell of position x facing for the outlier test and the gains.
struct Cells {
    inv: f64,
    of_point: Vec<u32>,
    /// per point the 8 cells (same facing) whose centres surround it, u32::MAX where there is
    /// none (found once, not per view: the views spent as long on them as on the occlusion test)
    around: Vec<[u32; 8]>,
    /// per cell: its key and whether the part owns it (its gain entries)
    key: Vec<u64>,
    owned: Vec<bool>,
}
fn cell_key(k: [i64; 3], face: u8) -> u64 {
    let b = |x: i64| ((x + (1 << 19)) as u64) & 0xf_ffff;
    (b(k[0]) << 43) | (b(k[1]) << 23) | (b(k[2]) << 3) | face as u64
}
/// The cells of the points `ids`; `owns(xy)`: whether the part owns a cell with this centre.
fn cells(
    p: &[[f32; 3]],
    n: &[[f32; 3]],
    ids: &[u32],
    cell: f64,
    owns: impl Fn([f64; 2]) -> bool + Sync,
) -> Cells {
    let inv = 1. / cell;
    let face: Vec<u8> = ids
        .par_iter()
        .map(|&i| {
            let m = n[i as usize];
            let ax = (0..3)
                .max_by(|&a, &b| m[a].abs().total_cmp(&m[b].abs()).then(b.cmp(&a)))
                .unwrap();
            if m[0] * m[0] + m[1] * m[1] + m[2] * m[2] > 0.25 {
                (ax * 2 + usize::from(m[ax] < 0.)) as u8
            } else {
                6
            }
        })
        .collect();
    let index = |i: u32| p[i as usize].map(|v| (v as f64 * inv).floor() as i64);
    let keys: Vec<u64> = ids
        .par_iter()
        .zip(&face)
        .map(|(&i, &f)| cell_key(index(i), f))
        .collect();
    let mut uniq: Vec<(u64, usize)> = keys.iter().copied().zip(0..).collect();
    uniq.par_sort_unstable();
    uniq.dedup_by_key(|x| x.0);
    let id: FxHashMap<u64, u32> = uniq
        .iter()
        .enumerate()
        .map(|(k, &(key, _))| (key, k as u32))
        .collect();
    let owned = uniq
        .iter()
        .map(|&(_, at)| {
            let k = index(ids[at]);
            owns([(k[0] as f64 + 0.5) / inv, (k[1] as f64 + 0.5) / inv])
        })
        .collect();
    let of_point = keys.par_iter().map(|k| id[k]).collect();
    let around = ids
        .par_iter()
        .zip(&face)
        .map(|(&i, &f)| {
            let b = p[i as usize].map(|x| (x as f64 * inv - 0.5).floor() as i64);
            std::array::from_fn(|o| {
                let off = [o & 1, (o >> 1) & 1, (o >> 2) & 1];
                let key = cell_key(std::array::from_fn(|d| b[d] + off[d] as i64), f);
                id.get(&key).copied().unwrap_or(u32::MAX)
            })
        })
        .collect();
    Cells {
        inv,
        of_point,
        around,
        key: uniq.iter().map(|x| x.0).collect(),
        owned,
    }
}

/// The best views kept per point of a part (by slot).
struct Kept {
    s: Vec<f32>,
    c: Vec<[u16; 3]>,
    m: Vec<[u16; 3]>,
    i: Vec<u16>,
    wide_s: Vec<f32>,
    wide_c: Vec<[u16; 3]>,
}
impl Kept {
    fn new(np: usize, kk: usize) -> Self {
        Kept {
            s: vec![f32::NEG_INFINITY; np * kk],
            c: vec![[0; 3]; np * kk],
            m: vec![[0; 3]; np * kk],
            i: vec![NO_IMAGE; np * kk],
            wide_s: vec![f32::NEG_INFINITY; np],
            wide_c: vec![[0; 3]; np],
        }
    }
    fn save(&self, path: &Path) -> Result<()> {
        let mut b = Vec::with_capacity(self.s.len() * 18 + self.wide_s.len() * 10);
        for x in &self.s {
            b.extend(x.to_le_bytes());
        }
        for x in self.c.iter().chain(&self.m).chain(&self.wide_c) {
            for v in x {
                b.extend(v.to_le_bytes());
            }
        }
        for x in &self.i {
            b.extend(x.to_le_bytes());
        }
        for x in &self.wide_s {
            b.extend(x.to_le_bytes());
        }
        std::fs::write(path, b).with_context(|| format!("write {}", path.display()))
    }
    fn load(path: &Path, np: usize, kk: usize) -> Result<Self> {
        let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let mut at = 0;
        let mut f32s = |n: usize| -> Vec<f32> {
            let v = b[at..at + 4 * n]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect();
            at += 4 * n;
            v
        };
        let s = f32s(np * kk);
        let tail = &b[4 * np * kk..];
        let rgb = |o: usize, n: usize| -> Vec<[u16; 3]> {
            tail[o..o + 6 * n]
                .as_chunks::<6>()
                .0
                .iter()
                .map(|x| std::array::from_fn(|d| u16::from_le_bytes([x[2 * d], x[2 * d + 1]])))
                .collect()
        };
        let (c, m, wide_c) = (
            rgb(0, np * kk),
            rgb(6 * np * kk, np * kk),
            rgb(12 * np * kk, np),
        );
        let o = 12 * np * kk + 6 * np;
        let i = tail[o..o + 2 * np * kk]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| u16::from_le_bytes(*x))
            .collect();
        let o = o + 2 * np * kk;
        let wide_s = tail[o..o + 4 * np]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        Ok(Kept {
            s,
            c,
            m,
            i,
            wide_s,
            wide_c,
        })
    }
}

/// A gain entry: cell, image, points, mean colour.
type Entry = (u64, u32, u32, [f32; 3]);

/// Exposure gain per image and channel: log mean colour of image k over cell c ~ L_c - g_k, by
/// alternating means weighted by the point counts (residuals clipped at 0.4), pulled to 0 by
/// the cells' median weight, mean 0 (as colorize_blend.gains). `entries` sorted.
fn gains(entries: &[Entry], images: usize) -> Vec<[f32; 3]> {
    let ok: Vec<_> = entries
        .iter()
        .filter(|e| e.3.iter().all(|&c| c > 8. && c < 245.) && e.2 >= 20)
        .collect();
    let mut count: FxHashMap<u64, u32> = FxHashMap::default();
    for e in &ok {
        *count.entry(e.0).or_default() += 1;
    }
    let ok: Vec<_> = ok.into_iter().filter(|e| count[&e.0] >= 2).collect();
    let mut dense: FxHashMap<u64, usize> = FxHashMap::default();
    let mut cs = Vec::with_capacity(ok.len());
    let mut uniq: Vec<u64> = ok.iter().map(|e| e.0).collect();
    uniq.sort_unstable();
    uniq.dedup();
    for (i, c) in uniq.iter().enumerate() {
        dense.insert(*c, i);
    }
    for e in &ok {
        cs.push(dense[&e.0]);
    }
    let nc = uniq.len();
    let ks: Vec<usize> = ok.iter().map(|e| e.1 as usize).collect();
    let ws: Vec<f64> = ok.iter().map(|e| e.2 as f64).collect();
    let ys: Vec<[f64; 3]> = ok.iter().map(|e| e.3.map(|c| (c as f64).ln())).collect();
    let mut sorted = ws.clone();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.is_empty() {
        0.
    } else if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2])
    };
    let mut g = vec![[0f64; 3]; images];
    for _ in 0..30 {
        let mut lnum = vec![[0f64; 3]; nc];
        let mut lden = vec![0f64; nc];
        for i in 0..ok.len() {
            for j in 0..3 {
                lnum[cs[i]][j] += ws[i] * (ys[i][j] + g[ks[i]][j]);
            }
            lden[cs[i]] += ws[i];
        }
        let mut num = vec![[0f64; 3]; images];
        let mut den = vec![0f64; images];
        for i in 0..ok.len() {
            for j in 0..3 {
                let l = lnum[cs[i]][j] / lden[cs[i]];
                let r = (l - ys[i][j]).clamp(g[ks[i]][j] - 0.4, g[ks[i]][j] + 0.4);
                num[ks[i]][j] += ws[i] * r;
            }
            den[ks[i]] += ws[i];
        }
        for k in 0..images {
            for j in 0..3 {
                g[k][j] = num[k][j] / (den[k] + median);
            }
        }
        let total: f64 = den.iter().map(|d| d + 1e-9).sum();
        for j in 0..3 {
            let mean = (0..images).map(|k| g[k][j] * (den[k] + 1e-9)).sum::<f64>() / total;
            for gk in g.iter_mut() {
                gk[j] -= mean;
            }
        }
    }
    g.iter().map(|gk| gk.map(|x| x.exp() as f32)).collect()
}

/// The gains divided by their geometric mean over the images within `window` / 2 of each.
fn local_gains(g: &[[f32; 3]], times: &[f64], window: f64) -> Vec<[f32; 3]> {
    (0..g.len())
        .map(|k| {
            let near: Vec<usize> = (0..g.len())
                .filter(|&j| (times[j] - times[k]).abs() <= window / 2.)
                .collect();
            std::array::from_fn(|c| {
                let mean =
                    near.iter().map(|&j| (g[j][c] as f64).ln()).sum::<f64>() / near.len() as f64;
                ((g[k][c] as f64).ln() - mean).exp() as f32
            })
        })
        .collect()
}

/// A part of the colouring: the points whose [`BASE`] cell lies in `out` are coloured, from the
/// views of those within `out` grown by [`PART_MARGIN`] (the cells of the outlier test around
/// them and the neighbours of the fill lie within).
struct ColourPart {
    out: Rect,
    ids: Vec<u32>,
}

/// The views of the points of `part` (their slots in `slot`), and the gain entries of the cells
/// the part owns.
#[allow(clippy::needless_range_loop)] // `j` indexes a view's points, scores and colours together
#[allow(clippy::too_many_arguments)]
fn gather(
    scene: &Scene,
    part: &ColourPart,
    slot: &[u32],
    poses: &[Pose],
    omega: &[f64],
    a: &ColorParams,
    job: &Job,
    entries: &mut Vec<Entry>,
) -> Result<(Kept, Cells)> {
    let p = scene.p;
    let grown = part.out.grow(PART_MARGIN);
    let lo = [0, 1].map(|d| grown.lo[d] as f64 * BASE);
    let hi = [0, 1].map(|d| grown.hi[d] as f64 * BASE);
    let coloured = Coloured { slot, lo, hi };
    let out = part.out;
    let cells = cells(p, scene.n, &part.ids, a.cell, |c| {
        out.contains([0, 1].map(|d| (c[d] / BASE).floor() as i64))
    });
    let np = part.ids.len();
    let kk = a.keep;
    let mut kept = Kept::new(np, kk);
    // the images whose camera can see the part
    let near: Vec<usize> = (0..poses.len())
        .filter(|&k| {
            let t = poses[k].t;
            let d: f64 = (0..2)
                .map(|d| (lo[d] - t[d]).max(t[d] - hi[d]).max(0.).powi(2))
                .sum();
            d.sqrt() <= a.max_range
        })
        .collect();
    // The merge is split by the points' slots in `shards` runs of `run` (the order of the images
    // stays per point).
    let threads = rayon::current_num_threads().max(1);
    let run = np.div_ceil(4 * threads).max(1);
    let shards = np.div_ceil(run);
    // per image: its view, the cell means around each point, the cells' means for the gains, and
    // the view's points by shard (where each shard's start)
    type Seen = (
        View,
        Vec<[f32; 3]>,
        Vec<(u32, u32, [f32; 3])>,
        Vec<u32>,
        Vec<usize>,
    );
    let work = |k: usize| -> Result<Option<Seen>> {
        let Some(v) = scene.view(&poses[k], omega[k], a, &coloured)? else {
            return Ok(None);
        };
        // The image's mean colour per cell it sees, and per point the means of the 8
        // cells (same facing) around it, trilinear: no step at the cell borders.
        let mut sums: FxHashMap<u32, ([f64; 3], u32)> = FxHashMap::default();
        for (&s, c) in v.pts.iter().zip(&v.rgb) {
            let e = sums
                .entry(cells.of_point[s as usize])
                .or_insert(([0.; 3], 0));
            for j in 0..3 {
                e.0[j] += c[j] as f64;
            }
            e.1 += 1;
        }
        let means: FxHashMap<u32, [f64; 3]> = sums
            .iter()
            .map(|(&c, (s, m))| (c, s.map(|x| x / *m as f64)))
            .collect();
        let pm: Vec<[f32; 3]> = v
            .pts
            .iter()
            .map(|&s| {
                let x = p[part.ids[s as usize] as usize];
                let g = x.map(|x| x as f64 * cells.inv - 0.5);
                let b = g.map(|x| x.floor());
                let fr = [g[0] - b[0], g[1] - b[1], g[2] - b[2]];
                let (mut acc, mut ws) = ([0f64; 3], 0f64);
                for (o, &c) in cells.around[s as usize].iter().enumerate() {
                    let off = [o & 1, (o >> 1) & 1, (o >> 2) & 1];
                    let Some(m) = means.get(&c) else { continue };
                    let w = (0..3)
                        .map(|d| if off[d] == 1 { fr[d] } else { 1. - fr[d] })
                        .product::<f64>()
                        + 1e-9;
                    for j in 0..3 {
                        acc[j] += w * m[j];
                    }
                    ws += w;
                }
                acc.map(|x| (x / ws.max(1e-12)) as f32)
            })
            .collect();
        let gain: Vec<(u32, u32, [f32; 3])> = sums
            .iter()
            .filter(|(c, _)| cells.owned[**c as usize])
            .map(|(&c, (s, m))| (c, *m, s.map(|x| (x / *m as f64) as f32)))
            .collect();
        let mut starts = vec![0usize; shards + 1];
        for &s in &v.pts {
            starts[s as usize / run + 1] += 1;
        }
        for q in 0..shards {
            starts[q + 1] += starts[q];
        }
        let mut fill = starts.clone();
        let mut by_shard = vec![0u32; v.pts.len()];
        for (j, &s) in v.pts.iter().enumerate() {
            let q = s as usize / run;
            by_shard[fill[q]] = j as u32;
            fill[q] += 1;
        }
        Ok(Some((v, pm, gain, by_shard, starts)))
    };
    // Merge in image order: replace the worst kept view when this one scores higher.
    let mut merge = |k: usize, r: Option<Seen>| {
        let Some((v, pm, gain, by_shard, starts)) = r else {
            return;
        };
        if a.gain {
            entries.extend(
                gain.into_iter()
                    .map(|(c, m, rgb)| (cells.key[c as usize], k as u32, m, rgb)),
            );
        }
        let Kept {
            s,
            c,
            m,
            i,
            wide_s,
            wide_c,
        } = &mut kept;
        let parts: Vec<_> = s
            .chunks_mut(run * kk)
            .zip(c.chunks_mut(run * kk))
            .zip(m.chunks_mut(run * kk))
            .zip(i.chunks_mut(run * kk))
            .zip(wide_s.chunks_mut(run))
            .zip(wide_c.chunks_mut(run))
            .enumerate()
            .collect();
        parts
            .into_par_iter()
            .for_each(|(q, (((((ks, kc), km), ki), ws), wc))| {
                for &j in &by_shard[starts[q]..starts[q + 1]] {
                    let j = j as usize;
                    let i = v.pts[j] as usize - q * run;
                    let s = v.score[j];
                    if v.wide[j] {
                        if s > ws[i] {
                            ws[i] = s;
                            wc[i] = pack(v.rgb[j]);
                        }
                        continue;
                    }
                    let row = &ks[i * kk..(i + 1) * kk];
                    let worst = (0..kk).fold(0, |w, x| if row[x] < row[w] { x } else { w });
                    if s > row[worst] {
                        let at = i * kk + worst;
                        ks[at] = s;
                        kc[at] = pack(v.rgb[j]);
                        km[at] = pack(pm[j]);
                        ki[at] = k as u16;
                    }
                }
            });
    };
    // Each thread takes the next image when free, at most `window` images ahead of the merge
    // (which waits for the images in order): no thread waits for the slowest of a batch.
    let window = 2 * threads;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let merged = (std::sync::Mutex::new(0usize), std::sync::Condvar::new());
    let stop = std::sync::atomic::AtomicBool::new(false);
    let mut failed = None;
    std::thread::scope(|sc| {
        let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<Option<Seen>>)>();
        for _ in 0..threads {
            let tx = tx.clone();
            let (next, merged, stop, work, near) = (&next, &merged, &stop, &work, &near);
            sc.spawn(move || {
                loop {
                    let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if j >= near.len() || stop.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let mut m = merged.0.lock().unwrap();
                    while j >= *m + window {
                        m = merged.1.wait(m).unwrap();
                    }
                    drop(m);
                    if tx.send((j, work(near[j]))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut pending = std::collections::BTreeMap::new();
        let mut at = 0;
        for (j, r) in rx {
            pending.insert(j, r);
            while let Some(r) = pending.remove(&at) {
                match r.and_then(|r| job.check().map(|_| r)) {
                    Ok(r) => merge(near[at], r),
                    Err(e) => {
                        failed.get_or_insert(e);
                        stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                at += 1;
                job.report("colour images", at as u64, near.len() as u64);
                // (all waiting threads go on once failed)
                *merged.0.lock().unwrap() = if failed.is_some() { usize::MAX / 2 } else { at };
                merged.1.notify_all();
            }
        }
    });
    if let Some(e) = failed {
        return Err(e);
    }
    Ok((kept, cells))
}

/// Debug-only provenance of direct blending in a world-space sphere, before the fill.
/// Enabled by S20_COLOR_AUDIT=<output.json> and S20_COLOR_AUDIT_SPHERE=x,y,z,r.
struct ColorAudit {
    centre: V,
    radius2: f64,
    output: PathBuf,
    points: usize,
    wide: usize,
    weights: Vec<f64>,
    counts: Vec<usize>,
    plane: Option<(V, f64)>,
}

impl ColorAudit {
    fn from_env(origin: [f64; 3], images: usize) -> Result<Option<Self>> {
        let Some(output) = std::env::var_os("S20_COLOR_AUDIT") else {
            return Ok(None);
        };
        let sphere = std::env::var("S20_COLOR_AUDIT_SPHERE")
            .context("S20_COLOR_AUDIT requires S20_COLOR_AUDIT_SPHERE=x,y,z,r")?;
        let s: Vec<f64> = sphere
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            s.len() == 4 && s.iter().all(|x| x.is_finite()) && s[3] > 0.,
            "invalid audit sphere"
        );
        let plane = if let Ok(text) = std::env::var("S20_COLOR_AUDIT_PLANE") {
            let v: Vec<f64> = text
                .split(',')
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()?;
            ensure!(
                v.len() == 4 && v.iter().all(|x| x.is_finite()) && v[3] > 0.,
                "invalid audit plane"
            );
            let n = V::new(v[0], v[1], v[2]);
            ensure!(n.norm() > 1e-9, "zero audit plane normal");
            Some((n.normalize(), v[3]))
        } else {
            None
        };
        Ok(Some(Self {
            centre: V::new(s[0], s[1], s[2]) - V::from(origin),
            radius2: s[3] * s[3],
            output: output.into(),
            points: 0,
            wide: 0,
            weights: vec![0.; images],
            counts: vec![0; images],
            plane,
        }))
    }

    fn write(&self, poses: &[Pose]) -> Result<()> {
        let images: Vec<_> = (0..poses.len()).filter(|&k| self.counts[k] > 0).map(|k| serde_json::json!({
            "name": poses[k].name, "time": poses[k].time, "points": self.counts[k], "weight_sum": self.weights[k],
        })).collect();
        let value = serde_json::json!({"scope": "direct blend before fill", "points": self.points,
            "plane_slab_half_width_m": self.plane.map(|(_, w)| w),
            "wide_fallback_points_without_image_id": self.wide, "images": images});
        std::fs::write(&self.output, serde_json::to_vec_pretty(&value)?)?;
        Ok(())
    }
}

/// Exactly the weights used by the colour and its diagnostic provenance.
fn blend_weights(s: &[f32], a: &ColorParams) -> Vec<(usize, f32)> {
    let mut o: Vec<usize> = (0..s.len()).filter(|&j| s[j].is_finite()).collect();
    o.sort_by(|&x, &y| s[y].total_cmp(&s[x]).then(x.cmp(&y)));
    let Some(&first) = o.first() else {
        return Vec::new();
    };
    let next = o.get(a.views).map_or(f32::NEG_INFINITY, |&j| s[j]);
    o.iter()
        .take(a.views)
        .map(|&j| {
            (
                j,
                (s[j] - next).clamp(1e-3, 1.)
                    * ((s[j] - s[first]) / a.temperature as f32).min(0.).exp(),
            )
        })
        .collect()
}

/// The colours of the points of `part` it owns, from their kept views and the gains `g`.
fn blend(
    p: &[[f32; 3]],
    part: &ColourPart,
    mut kept: Kept,
    g: &[[f32; 3]],
    a: &ColorParams,
    audit: Option<&mut ColorAudit>,
) -> Vec<(u32, [f32; 3])> {
    let np = part.ids.len();
    let kk = a.keep;
    let gain_of = |k: u16| {
        if k == NO_IMAGE {
            [1.; 3]
        } else {
            g[k as usize]
        }
    };
    let lum = |c: &[f32; 3]| c[0] * LUM[0] + c[1] * LUM[1] + c[2] * LUM[2];
    // Per point: the outlier reference (the second darkest view within `dark_window` of the
    // best, the darkest with fewer than 3; or with `outlier_quantile` that fraction of 3 or more
    // by brightness) and the count of those near views.
    let (refs, near_counts): (Vec<[f32; 3]>, Vec<u8>) = (0..np)
        .into_par_iter()
        .map(|i| {
            let s = &kept.s[i * kk..(i + 1) * kk];
            let m: Vec<[f32; 3]> = (0..kk)
                .map(|j| unpack(kept.m[i * kk + j], gain_of(kept.i[i * kk + j])))
                .collect();
            let top = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut cand: Vec<(f32, usize)> = (0..kk)
                .filter(|&j| s[j].is_finite() && s[j] >= top - a.dark_window as f32)
                .map(|j| (lum(&m[j]), j))
                .collect();
            cand.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
            let pick = match cand.len() {
                0 => 0,
                1 | 2 => cand[0].1,
                n if a.outlier_quantile > 0. => {
                    cand[((a.outlier_quantile * (n - 1) as f64).round() as usize).min(n - 1)].1
                }
                _ => cand[1].1,
            };
            (m[pick], cand.len() as u8)
        })
        .unzip();
    if a.outlier > 0. {
        kept.s
            .par_chunks_mut(kk)
            .zip(kept.m.par_chunks(kk).zip(kept.i.par_chunks(kk)))
            .zip(refs.par_iter())
            .for_each(|((s, (m, ki)), r)| {
                if s.iter().filter(|x| x.is_finite()).count() < 2 {
                    return;
                }
                for j in 0..kk {
                    if !s[j].is_finite() {
                        continue;
                    }
                    let mj = unpack(m[j], gain_of(ki[j]));
                    let dev: f64 = (0..3).map(|c| (mj[c] - r[c]).abs() as f64).sum();
                    s[j] -= (a.outlier_penalty * (dev / a.outlier).powi(4).ln_1p()) as f32;
                }
            });
    }
    if let Some(audit) = audit {
        for (i, &id) in part.ids.iter().enumerate() {
            let point = p[id as usize];
            if !part.out.contains(base_xy(point))
                || (vf(point) - audit.centre).norm_squared() > audit.radius2
            {
                continue;
            }
            if audit
                .plane
                .is_some_and(|(n, w)| (vf(point) - audit.centre).dot(&n).abs() > w)
            {
                continue;
            }
            let weights = blend_weights(&kept.s[i * kk..(i + 1) * kk], a);
            if weights.is_empty() {
                audit.wide += usize::from(kept.wide_s[i].is_finite());
                continue;
            }
            audit.points += 1;
            let sum = weights.iter().map(|(_, w)| w).sum::<f32>().max(1e-12);
            for (j, w) in weights {
                let k = kept.i[i * kk + j] as usize;
                audit.weights[k] += (w / sum) as f64;
                audit.counts[k] += 1;
            }
        }
    }
    // Blend the best `views`: exp((score - best) / temperature) x fade, the fade the margin over
    // the next view (at most 1), so a view fades out before another takes its place.
    let mut rgb: Vec<Option<[f32; 3]>> = (0..np)
        .into_par_iter()
        .map(|i| {
            let s = &kept.s[i * kk..(i + 1) * kk];
            let weights = blend_weights(s, a);
            if weights.is_empty() {
                // (as colorize_blend: the wide views are not exposure-corrected)
                return kept.wide_s[i]
                    .is_finite()
                    .then(|| unpack(kept.wide_c[i], [1.; 3]));
            }
            let band = a.band_temperature > 0.;
            let (mut acc, mut ws) = ([0f32; 3], 0f32);
            for (j, w) in weights {
                let gain = gain_of(kept.i[i * kk + j]);
                let c = unpack(kept.c[i * kk + j], gain);
                let m = if band {
                    unpack(kept.m[i * kk + j], gain)
                } else {
                    [0.; 3]
                };
                for d in 0..3 {
                    acc[d] += w * (c[d] - m[d]);
                }
                ws += w;
            }
            let mut out = acc.map(|x| x / ws.max(1e-12));
            if band {
                let top = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let (mut low, mut wl) = ([0f32; 3], 0f32);
                for j in (0..kk).filter(|&j| s[j].is_finite()) {
                    let w = ((s[j] - top) / a.band_temperature as f32).exp();
                    let m = unpack(kept.m[i * kk + j], gain_of(kept.i[i * kk + j]));
                    for d in 0..3 {
                        low[d] += w * m[d];
                    }
                    wl += w;
                }
                for d in 0..3 {
                    out[d] = (out[d] + low[d] / wl.max(1e-12)).max(0.);
                }
            }
            Some(out)
        })
        .collect();
    let pos = |i: usize| p[part.ids[i] as usize].map(|v| v as f64);
    let owned = |i: usize| part.out.contains(base_xy(p[part.ids[i] as usize]));
    // A point only one view sees cannot tell a glare: much brighter than the reference of its
    // neighbours several views see, it takes their colour.
    if a.fill > 0. && a.outlier > 0. {
        let multi: Vec<usize> = (0..np)
            .filter(|&i| rgb[i].is_some() && near_counts[i] >= 2)
            .collect();
        let single: Vec<usize> = (0..np)
            .filter(|&i| rgb[i].is_some() && near_counts[i] <= 1 && owned(i))
            .collect();
        if !multi.is_empty() && !single.is_empty() {
            let coords: Vec<[f64; 3]> = multi.iter().map(|&i| pos(i)).collect();
            let tree: kiddo::KdTree<f64, 3> = (&coords).into();
            let fixes: Vec<(usize, [f32; 3])> = single
                .par_iter()
                .filter_map(|&i| {
                    let nb = tree.nearest_n_within::<kiddo::SquaredEuclidean>(
                        &pos(i),
                        a.fill * a.fill,
                        std::num::NonZero::new(16).unwrap(),
                        true,
                    );
                    if nb.len() < 4 {
                        return None;
                    }
                    let mut l: Vec<f32> = nb
                        .iter()
                        .map(|m| lum(&refs[multi[m.item as usize]]))
                        .collect();
                    l.sort_by(f32::total_cmp);
                    let nref = if l.len() % 2 == 1 {
                        l[l.len() / 2]
                    } else {
                        0.5 * (l[l.len() / 2 - 1] + l[l.len() / 2])
                    };
                    let own = rgb[i].unwrap();
                    (lum(&own) > 1.25 * nref + 8.).then(|| {
                        let mut acc = [0f32; 3];
                        for m in &nb {
                            let c = rgb[multi[m.item as usize]].unwrap();
                            for d in 0..3 {
                                acc[d] += c[d];
                            }
                        }
                        (i, acc.map(|x| x / nb.len() as f32))
                    })
                })
                .collect();
            for (i, c) in fixes {
                rgb[i] = Some(c);
            }
        }
    }
    (0..np)
        .filter(|&i| owned(i))
        .filter_map(|i| rgb[i].map(|c| (part.ids[i], c)))
        .collect()
}

fn base_xy(x: [f32; 3]) -> [i64; 2] {
    [0, 1].map(|d| (x[d] as f64 / BASE).floor() as i64)
}

/// Colours of the points `p` (normals `n`), in a local frame: `origin` is added to them to give
/// the frame of the poses. `Some(rgb)` per coloured point, `None` where no image sees it.
/// `images`: the folder the names of `imgpose` are relative to (see
/// [`s20_io::dataset::read_image`]); `trajectory`: the export step's `trajectory.txt`; `tmp`: a
/// folder for the views of the parts while the gains are found (removed at the end).
#[allow(clippy::too_many_arguments)]
pub fn colorize(
    p: &[[f32; 3]],
    n: &[[f32; 3]],
    origin: [f64; 3],
    images: &Path,
    imgpose: &Path,
    trajectory: &Path,
    calibration: &Path,
    a: &ColorParams,
    tmp: &Path,
    job: &Job,
) -> Result<Vec<Option<[f32; 3]>>> {
    ensure!(p.len() == n.len(), "points and normals differ");
    ensure!(a.keep >= a.views && a.views > 0, "keep >= views > 0");
    let start = std::time::Instant::now();
    let cal = s20_io::calib::load(calibration)?;
    let mut poses = load_poses(imgpose)?;
    let mut audit = ColorAudit::from_env(origin, poses.len())?;
    ensure!(poses.len() < NO_IMAGE as usize, "too many images");
    let omega = turning(trajectory, &poses)?;
    for pose in &mut poses {
        pose.t -= V::from(origin);
    }
    let blocks = blocks(p, n);
    // the z-buffers and the points each image sees on the GPU when there is one (S20_NO_GPU
    // set: on the CPU)
    let gpu = match crate::gpu::shared().filter(|_| std::env::var_os("S20_NO_GPU").is_none()) {
        Some(g) => match crate::gpu::ZBuffer::new(g, p, n, &blocks.order) {
            Ok(z) => {
                job.log(&match &z.no_normals {
                    None => format!("colour: the points each image sees on the GPU ({})", g.name),
                    Some(e) => format!(
                        "colour: z-buffers on the GPU ({}), the points each image sees on the CPU                          (no room for the normals: {e})",
                        g.name
                    ),
                });
                Some(z)
            }
            Err(e) => {
                job.log(&format!("colour: z-buffers on the CPU (GPU: {e:#})"));
                None
            }
        },
        None => {
            job.log("colour: z-buffers on the CPU");
            None
        }
    };
    let scene = Scene {
        p,
        n,
        blocks,
        images: images.to_path_buf(),
        left: cal.left.clone(),
        right: cal.right.clone(),
        gpu,
    };
    // the parts: the square over all points split in four while they hold many points
    let max_points = match (a.tile_memory > 0.).then(crate::memory::available) {
        None => usize::MAX,
        Some(Some(free)) => {
            let n = ((free as f64 * a.tile_memory / BYTES_PER_POINT) as usize).max(1_000_000);
            job.log(&format!(
                "colour: up to {:.1} M points a part ({:.1} of {:.1} GB available)",
                n as f64 / 1e6,
                free as f64 * a.tile_memory / 1e9,
                free as f64 / 1e9
            ));
            n
        }
        Some(None) => FALLBACK_POINTS,
    };
    let mut counts: FxHashMap<[i64; 2], usize> = FxHashMap::default();
    for x in p {
        *counts
            .entry(base_xy(*x).map(|c| c.div_euclid(SMALL)))
            .or_default() += 1;
    }
    let count_in = |r: &Rect| -> usize {
        let (lo, hi) = (
            r.lo.map(|x| x.div_euclid(SMALL)),
            r.hi.map(|x| (x - 1).div_euclid(SMALL)),
        );
        let mut n = 0;
        for (k, c) in &counts {
            if (0..2).all(|d| k[d] >= lo[d] && k[d] <= hi[d]) {
                n += c;
            }
        }
        n
    };
    // (a square of a power of two small cells, so that its quarters are of whole ones)
    let (mut lo, mut hi) = ([i64::MAX; 2], [i64::MIN; 2]);
    for c in counts.keys() {
        for d in 0..2 {
            lo[d] = lo[d].min(c[d]);
            hi[d] = hi[d].max(c[d] + 1);
        }
    }
    let side = ((hi[0] - lo[0]).max(hi[1] - lo[1]).max(1) as u64).next_power_of_two() as i64;
    let mut stack = vec![Rect {
        lo: lo.map(|x| x * SMALL),
        hi: lo.map(|x| (x + side) * SMALL),
    }];
    let mut rects = Vec::new();
    while let Some(out) = stack.pop() {
        let w = out.hi[0] - out.lo[0];
        if count_in(&out.grow(PART_MARGIN)) > max_points && w >= 2 * SMALL {
            let mid = [0, 1].map(|d| out.lo[d] + (out.hi[d] - out.lo[d]) / 2);
            for (x, y) in [(1, 1), (1, 0), (0, 1), (0, 0)] {
                let lo = [
                    if x == 0 { out.lo[0] } else { mid[0] },
                    if y == 0 { out.lo[1] } else { mid[1] },
                ];
                let hi = [
                    if x == 0 { mid[0] } else { out.hi[0] },
                    if y == 0 { mid[1] } else { out.hi[1] },
                ];
                stack.push(Rect { lo, hi });
            }
        } else if count_in(&out) > 0 {
            rects.push(out);
        }
    }
    // the progress: the views about 80% of the time, the gains 3%, the colours the rest, each part
    // by its points (the park 2026-10-07 10-31: 325, 11 and 70 of 406 s)
    let all = rects.iter().map(&count_in).sum::<usize>().max(1) as f64;
    let mut before = vec![0.0];
    for r in &rects {
        before.push(before.last().unwrap() + count_in(r) as f64 / all);
    }
    let share = |k: usize, lo: f64, hi: f64| {
        job.part(lo + (hi - lo) * before[k], lo + (hi - lo) * before[k + 1])
    };
    drop(counts);
    job.log(&format!(
        "colour: {} points, {} images, {} blocks, {} parts ({:.1} s)",
        p.len(),
        poses.len(),
        scene.blocks.centres.len(),
        rects.len(),
        start.elapsed().as_secs_f64()
    ));
    // 1. the views of each part (kept on disk when there are several) and the gain entries
    let several = rects.len() > 1;
    if several {
        std::fs::create_dir_all(tmp).with_context(|| format!("create {}", tmp.display()))?;
    }
    let _cleanup = several.then(|| crate::tiles::TempDir(tmp.to_path_buf()));
    let mut slot = vec![u32::MAX; p.len()];
    let mut entries: Vec<Entry> = Vec::new();
    let mut memory: Vec<Kept> = Vec::new();
    let part_of = |out: Rect| {
        let grown = out.grow(PART_MARGIN);
        let ids: Vec<u32> = (0..p.len() as u32)
            .into_par_iter()
            .filter(|&i| grown.contains(base_xy(p[i as usize])))
            .collect();
        ColourPart { out, ids }
    };
    for (k, &out) in rects.iter().enumerate() {
        let part = part_of(out);
        for (s, &i) in part.ids.iter().enumerate() {
            slot[i as usize] = s as u32;
        }
        let (kept, _) = gather(
            &scene,
            &part,
            &slot,
            &poses,
            &omega,
            a,
            &share(k, 0.01, 0.81),
            &mut entries,
        )?;
        for &i in &part.ids {
            slot[i as usize] = u32::MAX;
        }
        if several {
            kept.save(&tmp.join(format!("{k}.views")))?;
        } else {
            memory.push(kept);
        }
        job.log(&format!(
            "colour: part {} of {}: {} points ({:.0} s)",
            k + 1,
            rects.len(),
            part.ids.len(),
            start.elapsed().as_secs_f64()
        ));
    }
    drop(slot);
    let gathered = start.elapsed().as_secs_f64();
    // 2. the gains
    entries.par_sort_unstable_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)));
    let mut g: Vec<[f32; 3]> = if a.gain {
        gains(&entries, poses.len())
    } else {
        vec![[1.; 3]; poses.len()]
    };
    if a.gain && a.gain_window > 0. {
        let times: Vec<f64> = poses.iter().map(|p| p.time).collect();
        g = local_gains(&g, &times, a.gain_window);
    }
    drop(entries);
    if a.gain {
        let mut q: Vec<[f32; 3]> = g.clone();
        let quantile = |q: &mut Vec<[f32; 3]>, j: usize, f: f64| {
            q.sort_by(|a, b| a[j].total_cmp(&b[j]));
            q[((q.len() - 1) as f64 * f).round() as usize][j]
        };
        let qs: Vec<String> = [0.05, 0.5, 0.95]
            .iter()
            .map(|&f| {
                format!(
                    "{:.3}/{:.3}/{:.3}",
                    quantile(&mut q, 0, f),
                    quantile(&mut q, 1, f),
                    quantile(&mut q, 2, f)
                )
            })
            .collect();
        job.log(&format!(
            "colour: gains (R/G/B) q05 {} q50 {} q95 {}",
            qs[0], qs[1], qs[2]
        ));
    }
    let gained = start.elapsed().as_secs_f64();
    job.report("colour gains", 84, 100);
    // 3. the colours, part by part
    let mut rgb: Vec<Option<[f32; 3]>> = vec![None; p.len()];
    let mut memory = memory.into_iter();
    for (k, &out) in rects.iter().enumerate() {
        job.check()?;
        share(k, 0.84, 1.0).report("colour blend", 0, 1);
        let part = part_of(out);
        let kept = if several {
            Kept::load(&tmp.join(format!("{k}.views")), part.ids.len(), a.keep)?
        } else {
            memory.next().unwrap()
        };
        for (i, c) in blend(p, &part, kept, &g, a, audit.as_mut()) {
            rgb[i as usize] = Some(c);
        }
    }
    if let Some(audit) = &audit {
        audit.write(&poses)?;
    }
    job.report("colour blend", 1, 1);
    let total = start.elapsed().as_secs_f64();
    let cpu: Vec<String> = STAGE_NAMES
        .iter()
        .enumerate()
        .map(|(k, n)| {
            format!(
                "{n} {:.0}",
                SECONDS[k].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9
            )
        })
        .collect();
    job.log(&format!(
        "colour seconds: gathering {gathered:.0} (CPU: {}; views empty {}, decoded {}), gains {:.0}, blending {:.0}",
        cpu.join(", "),
        SECONDS[5].swap(0, std::sync::atomic::Ordering::Relaxed),
        SECONDS[6].swap(0, std::sync::atomic::Ordering::Relaxed),
        gained - gathered,
        total - gained
    ));
    job.log(&format!(
        "colour: {} of {} points coloured ({:.1} s)",
        rgb.iter().filter(|c| c.is_some()).count(),
        p.len(),
        start.elapsed().as_secs_f64()
    ));
    Ok(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_weights_preserve_direct_blending_and_ties() {
        let a = ColorParams {
            views: 2,
            temperature: 2.,
            ..Default::default()
        };
        let weights = blend_weights(&[3., 1., -2., f32::NEG_INFINITY], &a);
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0], (0, 1.));
        assert!((weights[1].1 - (-1f32).exp()).abs() < 1e-7);
        assert_eq!(
            blend_weights(&[1., 1., 1.], &a),
            vec![(0, 0.001), (1, 0.001)]
        );
        assert!(blend_weights(&[f32::NEG_INFINITY; 3], &a).is_empty());
    }

    fn camera() -> FisheyeIntrinsics {
        FisheyeIntrinsics {
            k: [0.; 6],
            a11: 1000.,
            a12: 0.,
            a22: 1000.,
            u0: 1750.,
            v0: 2300.,
            width: 3504,
            height: 4672,
        }
    }

    #[test]
    fn single_precision_projection_is_within_a_thousandth_of_a_pixel() {
        let mut c = camera();
        c.k = [0.01, -0.02, 0.003, -0.001, 0.0002, -0.00001];
        let pose = Pose {
            name: String::new(),
            t: V::new(12.3, -45.6, 7.8),
            rcw: *UnitQuaternion::from_euler_angles(0.3, -1.1, 2.0)
                .to_rotation_matrix()
                .matrix(),
            time: 0.,
            left: true,
        };
        let lens = Lens::new(&c, &pose, 40., 100f64.to_radians());
        let mut out = Lanes::default();
        let mut worst = 0f64;
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..2000 {
            let mut d = [[0f32; LANES]; 3];
            let mut exact = Vec::new();
            // `l` fills one lane of the three coordinates, as the colouring does
            #[allow(clippy::needless_range_loop)]
            for l in 0..LANES {
                let x = V::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * 60.;
                for k in 0..3 {
                    d[k][l] = x[k] as f32;
                }
                exact.push(V::new(d[0][l] as f64, d[1][l] as f64, d[2][l] as f64));
            }
            lens.project(&d, &mut out);
            for (l, x) in exact.iter().enumerate() {
                let (u, v, th) = project(&c, pose.rcw * x);
                let ok = x.norm() <= 40.
                    && th < 100f64.to_radians()
                    && u >= 1.
                    && u < (c.width - 2) as f64
                    && v >= 1.
                    && v < (c.height - 2) as f64;
                if ok && out.ok[l] {
                    worst = worst
                        .max((u - out.u[l] as f64).abs())
                        .max((v - out.v[l] as f64).abs());
                    assert!((th - out.th[l] as f64).abs() < 1e-6);
                }
            }
        }
        assert!(worst < 1e-3, "{worst} px");
    }

    #[test]
    fn window_minima_are_the_least_around() {
        let (w, h) = (7, 5);
        let z: Vec<f32> = (0..w * h).map(|k| ((k * 37 + 11) % 23) as f32).collect();
        let m = window_minima(&z, w, h, 3);
        for (r, m) in m.iter().enumerate() {
            let r = r as i64 + 1;
            for i in 0..h as i64 {
                for j in 0..w as i64 {
                    let mut least = f32::INFINITY;
                    for di in -r..=r {
                        for dj in -r..=r {
                            let (y, x) = (
                                (i + di).clamp(0, h as i64 - 1),
                                (j + dj).clamp(0, w as i64 - 1),
                            );
                            least = least.min(z[(y * w as i64 + x) as usize]);
                        }
                    }
                    assert_eq!(m[(i * w as i64 + j) as usize], least);
                }
            }
        }
    }

    #[test]
    fn projection_is_equidistant_without_distortion() {
        let c = camera();
        let (u, v, th) = project(&c, V::new(0., 0., 2.));
        assert!((u - 1750.).abs() < 1e-9 && (v - 2300.).abs() < 1e-9 && th.abs() < 1e-12);
        // 45 degrees to the right: r = f theta.
        let (u, v, th) = project(&c, V::new(1., 0., 1.));
        assert!((th - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!(
            (u - (1750. + 1000. * std::f64::consts::FRAC_PI_4)).abs() < 1e-9
                && (v - 2300.).abs() < 1e-9
        );
    }

    #[test]
    fn gains_level_an_image_twice_as_bright() {
        // Ten cells seen by images 0 and 1; image 1 is twice as bright.
        let mut e = Vec::new();
        for c in 0..10u64 {
            let base = 40. + 5. * c as f32;
            e.push((c, 0, 50, [base; 3]));
            e.push((c, 1, 50, [2. * base; 3]));
        }
        let g = gains(&e, 2);
        let ratio = g[0][0] / g[1][0];
        assert!((ratio - 2.).abs() < 0.2, "gain ratio {ratio}");
        assert!((g[0][0] * g[1][0] - 1.).abs() < 0.2, "gains are centred");
    }

    #[test]
    fn local_gains_keep_the_exposure_of_a_place() {
        // Ten images in the shade (gain 0.6), ten in the sun (1.8), 2 s apart; image 5 is a
        // step brighter than its neighbours (half their gain).
        let mut g: Vec<[f32; 3]> = (0..20)
            .map(|k| [if k < 10 { 0.6 } else { 1.8 }; 3])
            .collect();
        g[5] = [0.3; 3];
        let times: Vec<f64> = (0..20).map(|k| 2. * k as f64).collect();
        let l = local_gains(&g, &times, 8.);
        assert!(
            (l[15][0] - 1.).abs() < 1e-3 && (l[2][0] - 1.).abs() < 0.2,
            "places keep their exposure: {l:?}"
        );
        assert!(l[5][0] < 0.6, "the step is levelled: {}", l[5][0]);
    }
}
