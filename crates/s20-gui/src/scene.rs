//! What the viewport shows: the raw scans placed with the current trajectory, or the finished point
//! cloud, and the trajectory. Coordinates are the local frame of the data folder (projected
//! coordinates for legacy results, ENU for new GNSS results).
//!
//! The scans: level of detail without an octree. Each node's points are in a random order, so any
//! prefix is a uniform subsample. The viewport draws a prefix as long as the node's share of the
//! point budget (by its size on screen). Prefixes are cut into chunks of growing size (2048, x4,
//! ...), each its own GPU buffer, so only what is drawn is uploaded.
//!
//! The output cloud: its Potree octree beside it in the output folder (see `s20_io::potree`), its
//! nodes drawn as Potree does (see [`allocate_octree`]).
//!
//! Previews keep metadata and an overview in RAM; the detail is streamed on demand.

use crate::lang::tl;
use crate::stream::{self, Streaming};
mod bag_preview;
use anyhow::Result;
use glam::DVec3;
use s20_geom::so3::{M3, V3};
use s20_geom::traj::Traj;
use s20_pipeline::RunConfig;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// A displayed point: f32 offset from its chunk's origin and an sRGB colour (16 bytes, a GPU vertex).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewPoint {
    pub offset: [f32; 3],
    pub color: [u8; 4],
}

/// A block of points uploaded and drawn together.
#[derive(Clone, Debug, PartialEq)]
pub struct NodePoints {
    pub origin: [f64; 3],
    pub points: Vec<ViewPoint>,
}
impl NodePoints {
    pub fn position(&self, i: usize) -> DVec3 {
        let o = self.points[i].offset;
        DVec3::from(self.origin) + DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64)
    }
    pub fn len(&self) -> usize {
        self.points.len()
    }
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// A node: points in random order, cut into chunks of growing size.
pub struct LodNode {
    pub center: DVec3,
    pub radius: f64,
    /// median nearest-neighbour distance among the first chunk's points
    pub sample_nn: f64,
    pub total: usize,
    pub chunks: Vec<Arc<NodePoints>>,
    /// the session of its scans (none for the finished cloud)
    pub session: Option<usize>,
    /// the first and last time of its scans (sensor clock; none for the finished cloud)
    pub time: Option<[f64; 2]>,
}

/// The grey of a reflectivity for the preview: the scans' colour, and every point's alpha (see
/// `render::Coloring`).
pub fn reflectivity_grey(reflectivity: u16) -> u8 {
    (40.0 + reflectivity.min(150) as f32 / 150.0 * 200.0) as u8
}

/// What the preview points are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PreviewSource {
    /// raw scans placed with the trajectory
    Scans,
    Uncolored,
    Colored,
}

/// Trajectory used to place raw scans.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrajectorySource {
    /// the device's own estimate while recording (`frame_pose.txt`), each session in its own frame
    Device,
    Lio,
    #[default]
    Optimized,
}

/// A session as shown: its time range on the common axis, and whether it is in the common frame
/// (else it is drawn beside the others, in its own odometry frame).
#[derive(Clone, Debug, PartialEq)]
pub struct SceneSession {
    pub name: String,
    pub start: f64,
    pub end: f64,
    pub placed: bool,
}

#[derive(Default)]
pub struct Scene {
    pub trajectory_source: TrajectorySource,
    pub nodes: Vec<LodNode>,
    pub streaming: Option<Streaming>,
    /// trajectory samples (time, position), of all sessions in time order
    pub traj: Vec<(f64, DVec3)>,
    /// orientation of each trajectory sample
    pub rot: Vec<glam::DQuat>,
    /// the sessions shown (one for a work folder of one session)
    pub sessions: Vec<SceneSession>,
    /// file the trajectory came from
    pub traj_file: Option<PathBuf>,
    /// per-knot quality of the optimised trajectory
    pub quality: Option<crate::quality::Quality>,
    /// Whether the loaded trajectory used RTK, computed once when its quality is loaded.
    pub rtk_used: bool,
    /// camera images of the recording
    pub stamps: Vec<crate::images::Stamp>,
    /// time shown as 0: the start of the first scan (as `--rtk_exclude` on the command line)
    pub t_origin: f64,
    /// RTK FIX epochs: time, position (local frame) and whether the validity check kept it
    pub rtk: Vec<(f64, DVec3, bool)>,
    /// for each of `rtk`, what it is and why it is used or not (for a tooltip)
    pub rtk_info: Vec<String>,
    pub ppk_summary: Option<String>,
    pub source: Option<PreviewSource>,
    /// why the cloud asked for is not shown (none shown instead)
    pub unavailable: Option<String>,
    /// whether the points carry image colours (else a height ramp is offered)
    pub has_color: bool,
    pub bounds: Option<(DVec3, DVec3)>,
    pub z_range: Option<[f64; 2]>,
    /// largest full-density neighbour distance used for adaptive sizes: 3 times the typical one
    pub nn_cap: f64,
    /// the nodes' octree, for the output cloud
    pub octree: Option<Arc<Octree>>,
}

impl Scene {
    /// Trajectory position at time `t` (nearest sample).
    pub fn position_at(&self, t: f64) -> Option<DVec3> {
        if self.traj.is_empty() {
            return None;
        }
        let i = self
            .traj
            .partition_point(|s| s.0 < t)
            .min(self.traj.len() - 1);
        Some(self.traj[i].1)
    }

    /// The session recording at time `t`.
    pub fn session_at(&self, t: f64) -> Option<usize> {
        session_of(&self.sessions, t)
    }

    /// Pose (rotation, position) of the trajectory sample nearest to time `t`.
    pub fn pose_at(&self, t: f64) -> Option<(glam::DQuat, DVec3)> {
        if self.traj.is_empty() {
            return None;
        }
        let i = self
            .traj
            .partition_point(|s| s.0 < t)
            .min(self.traj.len() - 1);
        Some((self.rot.get(i).copied().unwrap_or_default(), self.traj[i].1))
    }

    /// The nodes to draw for a point budget: (node, points), the largest on screen first (see
    /// [`allocate`], [`allocate_octree`]).
    pub fn allocate(
        &self,
        cam: &crate::camera::Camera,
        viewport_h: f64,
        budget: usize,
        visible: &dyn Fn(&LodNode) -> bool,
    ) -> Vec<(usize, usize)> {
        match &self.octree {
            Some(t) => allocate_octree(&self.nodes, t, cam, viewport_h, budget, visible),
            None => allocate(&self.nodes, cam, viewport_h, budget, visible),
        }
    }

    pub fn points(&self) -> usize {
        self.streaming
            .as_ref()
            .map_or_else(|| self.nodes.iter().map(|n| n.total).sum(), |s| s.total())
    }
    pub fn corners(&self) -> Vec<DVec3> {
        let Some((lo, hi)) = self.bounds else {
            return Vec::new();
        };
        (0..8)
            .map(|i| {
                DVec3::new(
                    if i & 1 == 0 { lo.x } else { hi.x },
                    if i & 2 == 0 { lo.y } else { hi.y },
                    if i & 4 == 0 { lo.z } else { hi.z },
                )
            })
            .collect()
    }
}

/// Small deterministic generator for the shuffles.
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut r = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for i in (1..v.len()).rev() {
        let j = (r.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

const FIRST_CHUNK: usize = 2048;

/// Points (already in random order) into a node of growing chunks.
pub fn lod_node(pts: &[([f64; 3], [u8; 4])]) -> Option<LodNode> {
    if pts.is_empty() {
        return None;
    }
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for (p, _) in pts {
        lo = lo.min(DVec3::from(*p));
        hi = hi.max(DVec3::from(*p));
    }
    let center = lo.midpoint(hi);
    let radius = (hi - lo).length() * 0.5 + 0.05;
    let o = center.to_array();
    let mut chunks = Vec::new();
    let (mut s, mut size) = (0, FIRST_CHUNK);
    while s < pts.len() {
        let e = (s + size).min(pts.len());
        chunks.push(Arc::new(NodePoints {
            origin: o,
            points: pts[s..e]
                .iter()
                .map(|(p, c)| ViewPoint {
                    offset: [
                        (p[0] - o[0]) as f32,
                        (p[1] - o[1]) as f32,
                        (p[2] - o[2]) as f32,
                    ],
                    color: *c,
                })
                .collect(),
        }));
        s = e;
        size *= 4;
    }
    let sample_nn = median_nn(&chunks[0]);
    Some(LodNode {
        center,
        radius,
        sample_nn,
        total: pts.len(),
        chunks,
        session: None,
        time: None,
    })
}

/// Median nearest-neighbour distance among a chunk's points (0 for fewer than two).
fn median_nn(c: &NodePoints) -> f64 {
    if c.len() < 2 {
        return 0.0;
    }
    let pts: Vec<[f64; 3]> = (0..c.len()).map(|i| c.position(i).to_array()).collect();
    let tree: kiddo::KdTree<f64, 3> = kiddo::KdTree::from(&pts);
    let mut d: Vec<f64> = pts
        .iter()
        .map(|p| {
            tree.nearest_n::<kiddo::SquaredEuclidean>(p, 2)
                .last()
                .map_or(0.0, |n| n.distance.sqrt())
        })
        .collect();
    let k = d.len() / 2;
    *d.select_nth_unstable_by(k, f64::total_cmp).1
}

impl LodNode {
    /// Neighbour distance with all points drawn. Points spread on surfaces: the distance shrinks
    /// with the square root of the point count.
    pub fn full_nn(&self) -> f64 {
        self.sample_nn * (self.total.min(FIRST_CHUNK) as f64 / self.total as f64).sqrt()
    }

    /// Grid spacing of `k` drawn points, for Potree's adaptive size: the full-density neighbour
    /// distance (at most `cap`, so sparse nodes of stray points are not drawn as large discs), grown
    /// by the square root of the thinning; the median neighbour distance of random points is about
    /// half the grid spacing.
    pub fn spacing(&self, k: usize, cap: f64) -> f32 {
        let full = self.full_nn().min(cap);
        (2.0 * full * (self.total as f64 / k.max(1) as f64).sqrt()) as f32
    }
}

/// Bounds of the bulk of the points (2..98 % per axis, a few far points through windows do not
/// shrink the view) and the height range for the ramp, from the first chunk of every node.
fn robust_bounds(nodes: &[LodNode]) -> Option<(DVec3, DVec3)> {
    let sample: Vec<DVec3> = nodes
        .iter()
        .flat_map(|n| {
            n.chunks
                .iter()
                .flat_map(|c| (0..c.len()).map(move |i| c.position(i)))
                .take(FIRST_CHUNK)
        })
        .collect();
    if sample.is_empty() {
        return None;
    }
    let pct = |d: usize, a: f64| {
        let mut v: Vec<f64> = sample.iter().map(|p| p[d]).collect();
        let k = ((v.len() - 1) as f64 * a) as usize;
        *v.select_nth_unstable_by(k, f64::total_cmp).1
    };
    let lo = DVec3::new(pct(0, 0.02), pct(1, 0.02), pct(2, 0.02));
    let hi = DVec3::new(pct(0, 0.98), pct(1, 0.98), pct(2, 0.98));
    Some((lo, hi))
}

/// The heights to colour over: those of the space the points take (cells of a quarter metre)
/// rather than of the points, so that where the scanner stayed long does not take the whole
/// ramp; the extreme 0.5 % of the cells at either end left out.
pub fn height_range(points: impl Iterator<Item = DVec3>) -> Option<[f64; 2]> {
    const CELL: f64 = 0.25;
    let cells: std::collections::HashSet<[i64; 3]> = points
        .map(|p| (p / CELL).floor().as_i64vec3().to_array())
        .collect();
    let mut z: Vec<i64> = cells.into_iter().map(|c| c[2]).collect();
    if z.is_empty() {
        return None;
    }
    z.sort_unstable();
    let k = (z.len() as f64 * 0.005) as usize;
    Some([z[k] as f64 * CELL, (z[z.len() - 1 - k] + 1) as f64 * CELL])
}

/// Scans grouped into nodes of this many (about one second).
const SCANS_PER_NODE: usize = 10;

/// One node of raw points: IMU-frame positions, time after `t0`, reflectivity, in random order.
struct ScanNode {
    t0: f64,
    t1: f64,
    x: Vec<[f32; 3]>,
    dt: Vec<f32>,
    refl: Vec<u8>,
}

/// Bounded raw scan working data, loaded from the bag when a node is needed.
struct ScanGroup {
    nodes: Vec<ScanNode>,
}

impl ScanGroup {
    fn read(
        l: &s20_io::dataset::LidarData,
        readers: &mut [std::io::BufReader<std::fs::File>],
        g: usize,
        end: usize,
        stride: usize,
    ) -> Result<Self> {
        let lever = V3::from(s20_pipeline::LIDAR_IMU_T);
        let stride = stride.max(1);
        let (mut x, mut t, mut refl) = (Vec::new(), Vec::new(), Vec::new());
        for i in g..end {
            let sc = l.read_scan(i, &mut readers[l.scans[i].session])?;
            // Keep the same deterministic sampling phase for either trajectory.
            for n in ((i % stride)..sc.t.len()).step_by(stride) {
                let q = V3::from(sc.xyz[n]);
                if sc.tag[n] & 0x3F != 0 || !(0.5..60.0).contains(&q.norm()) {
                    continue;
                }
                let p = q + lever;
                x.push([p.x as f32, p.y as f32, p.z as f32]);
                t.push(sc.t[n]);
                refl.push(sc.refl[n]);
            }
        }
        if x.is_empty() {
            return Ok(Self { nodes: Vec::new() });
        }
        let t0 = t.iter().copied().fold(f64::INFINITY, f64::min);
        let t1 = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut idx: Vec<u32> = (0..x.len() as u32).collect();
        shuffle(&mut idx, g as u64 + 1);
        Ok(Self {
            nodes: vec![ScanNode {
                t0,
                t1,
                x: idx.iter().map(|&k| x[k as usize]).collect(),
                dt: idx.iter().map(|&k| (t[k as usize] - t0) as f32).collect(),
                refl: idx.iter().map(|&k| refl[k as usize]).collect(),
            }],
        })
    }
}

/// The selected trajectory, when it exists.
pub fn trajectory_file(cfg: &RunConfig, source: TrajectorySource) -> Option<PathBuf> {
    let f = match source {
        TrajectorySource::Device => cfg.device_file_of(0),
        TrajectorySource::Lio => cfg.lio_file(),
        TrajectorySource::Optimized => cfg.opt_file(),
    };
    f.exists().then_some(f)
}

/// What a PPK solution is and why it is used or not: its state, the states of the forward and
/// backward solutions and how far apart they are, and its standard deviations against the limits.
fn ppk_info(
    r: &s20_pipeline::ppk::Solution,
    d: [Option<&s20_pipeline::ppk::Solution>; 2],
    p: &s20_pipeline::ppk::Params,
) -> String {
    use s20_pipeline::ppk::inspect::Fix;
    let label = |s: &s20_pipeline::ppk::Solution| {
        Fix::of_rtklib(s.status).map_or(tl("不明", "unknown"), Fix::label)
    };
    let state = Fix::of_ppk(r, d, p).map_or(tl("不明", "unknown"), Fix::label);
    let used = r.accepted(p);
    let cm = |m: f64| m * 100.0;
    let sd_h = r.sd[0].hypot(r.sd[1]);
    let mut why = Vec::new();
    if !(r.status == 1 || (p.use_float && r.status == 2)) {
        why.push(tr!(
            "{} は使わない設定",
            "{} not used by the settings",
            label(r)
        ));
    }
    if sd_h > p.max_sd_h {
        why.push(
            tl(
                "水平の標準偏差が上限を超えている",
                "horizontal standard deviation over the limit",
            )
            .to_string(),
        );
    }
    if r.sd[2] > p.max_sd_v {
        why.push(
            tl(
                "鉛直の標準偏差が上限を超えている",
                "vertical standard deviation over the limit",
            )
            .to_string(),
        );
    }
    if !r.age.is_finite() || r.age.abs() > 2.0 {
        why.push(
            tl(
                "基準局のデータが古い（2 秒超）",
                "base station data too old (over 2 s)",
            )
            .to_string(),
        );
    }
    let mut s = if used {
        tr!("PPK・{state}：採用", "PPK, {state}: accepted")
    } else if why.is_empty() {
        tr!("PPK・{state}：不採用", "PPK, {state}: rejected")
    } else {
        tr!(
            "PPK・{state}：不採用（{}）",
            "PPK, {state}: rejected ({})",
            why.join("、")
        )
    };
    if let [Some(f), Some(b)] = d {
        let (h, v) = f.gap_m(b);
        s += &tr!(
            "
前方解 {}・後方解 {}（ずれ 水平 {:.1} cm・鉛直 {:.1} cm）",
            "\nForward {}, backward {} (offset horizontal {:.1} cm, vertical {:.1} cm)",
            label(f),
            label(b),
            cm(h),
            cm(v.abs())
        );
        if r.status == 1 && f.status != b.status && !used {
            s += tl(
                "
片方向だけの FIX で、もう片方の解と離れているため、標準偏差をそのずれの分だけ大きくしています",
                "\nA one-sided FIX far from the other solution, so its standard deviation is enlarged by the offset",
            );
        }
    }
    s += &tr!(
        "
標準偏差 水平 {:.1} cm（上限 {:.0} cm）・鉛直 {:.1} cm（上限 {:.0} cm）",
        "\nStandard deviation horizontal {:.1} cm (limit {:.0} cm), vertical {:.1} cm (limit {:.0} cm)",
        cm(sd_h),
        cm(p.max_sd_h),
        cm(r.sd[2]),
        cm(p.max_sd_v)
    );
    s
}

/// RTK FIX epochs of a data folder in the local frame, with the result of the validity check.
fn rtk_fixes(data: &Path, frame: &s20_io::geodesy::CoordinateFrame) -> Vec<(f64, DVec3, bool)> {
    let Ok(g) = s20_io::dataset::load_gnss(data) else {
        return Vec::new();
    };
    let ok = s20_io::gnss::rtk_valid(&g, &s20_io::gnss::Validity::default());
    g.iter()
        .zip(ok)
        .filter(|(r, _)| r[1] == s20_io::gnss::FIX)
        .map(|(r, v)| {
            (
                r[0],
                DVec3::from(frame.position([r[2], r[3], r[4] + r[8]])),
                v,
            )
        })
        .collect()
}

/// The output cloud's octree as the preview has it: by node, its parent and children (Potree's
/// order: x 4, y 2, z 1), level and cube (local frame).
pub struct Octree {
    pub parent: Vec<Option<usize>>,
    pub children: Vec<[Option<usize>; 8]>,
    pub level: Vec<u32>,
    pub min: Vec<DVec3>,
    pub size: Vec<DVec3>,
    /// the root's spacing, halved at each level
    pub root_spacing: f64,
    /// Density-derived spacings, cached when nodes are decoded.
    adaptive_spacings: Vec<std::sync::OnceLock<f64>>,
    /// Previous selection for threshold hysteresis.
    selected: std::sync::Mutex<Vec<bool>>,
}

impl Octree {
    /// The spacing of node `i`'s points drawn with those of its ancestors.
    pub fn spacing(&self, i: usize) -> f64 {
        self.root_spacing / (1u64 << self.level[i]) as f64
    }

    /// The child of node `i` whose cube `p` is in.
    pub fn child_at(&self, i: usize, p: DVec3) -> Option<usize> {
        let mid = self.min[i] + self.size[i] * 0.5;
        let k = ((p.x >= mid.x) as usize) << 2
            | ((p.y >= mid.y) as usize) << 1
            | (p.z >= mid.z) as usize;
        self.children[i][k]
    }

    /// Potree 2 density is points per occupied cell of a 32^3 grid.
    /// Match its visible-node texture's quantization of the LOD offset.
    pub fn adaptive_spacing(&self, i: usize, chunk: &NodePoints) -> f64 {
        *self.adaptive_spacings[i].get_or_init(|| {
            let mut occupied = vec![false; 32 * 32 * 32];
            for j in 0..chunk.len() {
                let relative = (chunk.position(j) - self.min[i]) / self.size[i];
                let cell = relative
                    .to_array()
                    .map(|v| (v * 32.0).clamp(0.0, 31.0) as usize);
                occupied[cell[0] + 32 * cell[1] + 1024 * cell[2]] = true;
            }
            let cells = occupied.iter().filter(|&&v| v).count().max(1);
            let density = (chunk.len() / cells).max(1) as f64;
            let offset = ((density.log2() * 0.5 - 1.5 + 10.0) * 10.0)
                .clamp(0.0, 255.0)
                .floor()
                / 10.0
                - 10.0;
            self.spacing(i) / offset.exp2()
        })
    }

    /// Use the deepest drawn node's density and LOD at each point's position.
    pub fn point_spacings(
        &self,
        i: usize,
        chunk: &NodePoints,
        drawn: &[bool],
        spacings: &[f64],
    ) -> Arc<[f32]> {
        (0..chunk.len())
            .map(|j| {
                let p = chunk.position(j);
                let mut m = i;
                while let Some(c) = self.child_at(m, p).filter(|&c| drawn[c]) {
                    m = c;
                }
                spacings[m] as f32
            })
            .collect()
    }
}

/// Points of the octree's levels kept loaded from the start (whole levels from the root, the root
/// at least): the overview.
const OCTREE_RESIDENT: usize = 4 * stream::OVERVIEW_POINTS;

/// The output cloud's octree in `dir` as preview nodes: the overview loaded, the other nodes
/// streamed; colours as the cloud has them (`colored`), the reflectivity's grey in the alpha.
fn octree_preview(
    dir: &Path,
    origin: [f64; 3],
    colored: bool,
    cancel: &AtomicBool,
) -> Result<(Streaming, Vec<LodNode>, Octree)> {
    let t = s20_io::potree::Octree::open(dir)?;
    let o = DVec3::from(origin);
    let mut tree = Octree {
        parent: t.nodes.iter().map(|n| n.parent).collect(),
        children: t.nodes.iter().map(|n| n.children).collect(),
        level: t.nodes.iter().map(|n| n.level).collect(),
        min: Vec::with_capacity(t.nodes.len()),
        size: Vec::with_capacity(t.nodes.len()),
        root_spacing: t.spacing,
        adaptive_spacings: (0..t.nodes.len())
            .map(|_| std::sync::OnceLock::new())
            .collect(),
        selected: std::sync::Mutex::new(vec![false; t.nodes.len()]),
    };
    let mut nodes = Vec::with_capacity(t.nodes.len());
    for i in 0..t.nodes.len() {
        let (min, size) = t.cube(i);
        let (min, size) = (DVec3::from(min) - o, DVec3::from(size));
        tree.min.push(min);
        tree.size.push(size);
        nodes.push(LodNode {
            center: min + size * 0.5,
            radius: size.length() * 0.5,
            sample_nn: tree.spacing(i),
            total: t.nodes[i].points as usize,
            chunks: Vec::new(),
            session: None,
            time: None,
        });
    }
    let mut by_level = Vec::new();
    for (i, n) in t.nodes.iter().enumerate() {
        let l = n.level as usize;
        if by_level.len() <= l {
            by_level.resize(l + 1, 0);
        }
        by_level[l] += nodes[i].total;
    }
    let mut levels = 1;
    let mut sum = by_level.first().copied().unwrap_or(0);
    while levels < by_level.len() && sum + by_level[levels] <= OCTREE_RESIDENT {
        sum += by_level[levels];
        levels += 1;
    }
    let file = std::sync::Mutex::new(std::fs::File::open(t.octree_file())?);
    let t = Arc::new(t);
    let read = move |ni: usize, at: [f64; 3]| -> Result<Arc<NodePoints>> {
        let data = {
            let mut f = file
                .lock()
                .map_err(|_| anyhow::anyhow!("preview reader poisoned"))?;
            t.read_bytes(&mut f, ni)?
        };
        let p = t.decode(ni, &data)?;
        let at_world = DVec3::from(at) + o;
        let points = (0..p.xyz.len())
            .map(|j| {
                let grey = reflectivity_grey(p.intensity.as_ref().map_or(0, |v| v[j]));
                let color = match &p.rgb {
                    Some(c) if colored => {
                        let c = c[j];
                        [
                            (c[0] >> 8) as u8,
                            (c[1] >> 8) as u8,
                            (c[2] >> 8) as u8,
                            grey,
                        ]
                    }
                    _ => [200, 200, 200, grey],
                };
                ViewPoint {
                    offset: std::array::from_fn(|d| (p.xyz[j][d] - at_world[d]) as f32),
                    color,
                }
            })
            .collect();
        Ok(Arc::new(NodePoints { origin: at, points }))
    };
    for (i, n) in nodes.iter_mut().enumerate() {
        if (tree.level[i] as usize) < levels && n.total > 0 {
            stream::check_cancel(cancel)?;
            let points = read(i, n.center.to_array())?;
            n.chunks.push(points);
        }
    }
    let (streaming, nodes) = Streaming::whole_nodes(nodes, move |ni, _, _, at| read(ni, at));
    Ok((streaming, nodes, tree))
}

fn scan_dependencies(cfg: &RunConfig) -> Vec<PathBuf> {
    let mut paths = vec![cfg.data().join("sessions.txt")];
    for k in 0..cfg.sessions.len() {
        paths.push(cfg.session_data(k).join("source.txt"));
        paths.push(cfg.session_data(k).join("lidar_index.npy"));
    }
    paths
}

/// The trajectory rows of every session to show, in time order, with the sessions: the optimised
/// trajectory where it holds a session, else the session's odometry (or the device's trajectory,
/// for [`TrajectorySource::Device`]). With several sessions, those not in the optimised trajectory
/// (not placed yet, each in its own frame) are moved beside the others so that they do not overlap.
/// Trajectory rows of all sessions, the sessions, the file the trajectory came from.
type SessionRows = (Vec<[f64; 8]>, Vec<SceneSession>, Option<PathBuf>);

fn session_rows(cfg: &RunConfig, source: TrajectorySource) -> Result<SessionRows> {
    let opt = cfg.opt_file();
    let opt_rows = if source == TrajectorySource::Optimized && opt.exists() {
        s20_io::dataset::read_tum(&opt)?
    } else {
        Vec::new()
    };
    if !cfg.multi() {
        let Some(f) = trajectory_file(cfg, source) else {
            return Ok((Vec::new(), Vec::new(), None));
        };
        let rows = if f == opt {
            opt_rows
        } else {
            s20_io::dataset::read_tum(&f)?
        };
        let session = rows.first().map(|r| SceneSession {
            name: cfg.sessions[0].name.clone(),
            start: r[0],
            end: rows.last().unwrap()[0],
            placed: true,
        });
        return Ok((rows, session.into_iter().collect(), Some(f)));
    }
    // the sessions in the set of all of them, with their shifts and ranges
    let dirs = s20_io::dataset::sessions(&cfg.data()).unwrap_or_default();
    let ranges = s20_io::dataset::LidarData::open(&cfg.data())
        .map(|l| l.session_ranges())
        .unwrap_or_default();
    let mut parts: Vec<(SceneSession, Vec<[f64; 8]>)> = Vec::new();
    for (k, sc) in cfg.sessions.iter().enumerate() {
        let Some(j) = dirs
            .iter()
            .position(|d| s20_io::dataset::session_name(&d.dir) == sc.name)
        else {
            continue;
        };
        let Some(&(start, end)) = ranges.get(j) else {
            continue;
        };
        let inside: Vec<[f64; 8]> = opt_rows
            .iter()
            .filter(|r| r[0] >= start && r[0] <= end)
            .copied()
            .collect();
        let (placed, rows) = if inside.len() >= 10 {
            (true, inside)
        } else {
            let own = if source == TrajectorySource::Device {
                cfg.device_file_of(k)
            } else {
                cfg.lio_file_of(k)
            };
            if !own.exists() {
                continue;
            }
            let mut rows = s20_io::dataset::read_tum(&own)?;
            for r in &mut rows {
                r[0] += dirs[j].shift;
            }
            // the first session is the reference: its own frame is the common one
            (k == 0 && opt_rows.is_empty(), rows)
        };
        parts.push((
            SceneSession {
                name: sc.name.clone(),
                start,
                end,
                placed,
            },
            rows,
        ));
    }
    // the sessions not placed: beside those placed, one after the other along x
    let bounds = |rows: &[[f64; 8]]| {
        rows.iter().fold(
            (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
            |(lo, hi), r| {
                let p = DVec3::new(r[1], r[2], r[3]);
                (lo.min(p), hi.max(p))
            },
        )
    };
    let placed_rows: Vec<[f64; 8]> = parts
        .iter()
        .filter(|p| p.0.placed)
        .flat_map(|p| p.1.iter().copied())
        .collect();
    let (plo, phi) = if placed_rows.is_empty() {
        (DVec3::ZERO, DVec3::ZERO)
    } else {
        bounds(&placed_rows)
    };
    let mut x = phi.x + 5.0;
    for (s, rows) in parts.iter_mut().filter(|p| !p.0.placed) {
        let (lo, hi) = bounds(rows);
        let d = DVec3::new(x - lo.x, 0.5 * (plo.y + phi.y - lo.y - hi.y), 0.0);
        for r in rows.iter_mut() {
            r[1] += d.x;
            r[2] += d.y;
        }
        x += hi.x - lo.x + 5.0;
        s.placed = false;
    }
    let mut rows: Vec<[f64; 8]> = parts.iter().flat_map(|p| p.1.iter().copied()).collect();
    rows.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let file = if source == TrajectorySource::Optimized && opt.exists() {
        Some(opt)
    } else if source == TrajectorySource::Device {
        Some(cfg.device_file_of(0))
    } else {
        Some(cfg.lio_file())
    };
    Ok((rows, parts.into_iter().map(|p| p.0).collect(), file))
}

/// The camera images of every session (times on the common axis).
fn session_stamps(cfg: &RunConfig) -> Vec<crate::images::Stamp> {
    let Ok(dirs) = s20_io::dataset::sessions(&cfg.data()) else {
        return Vec::new();
    };
    let mut v: Vec<crate::images::Stamp> = dirs
        .iter()
        .flat_map(|d| crate::images::load_stamps_of(&d.dir, d.shift))
        .collect();
    v.sort_by(|a, b| a.t.total_cmp(&b.t));
    v
}

/// Load metadata and the overview; the detail is streamed. The output cloud's octree is made when
/// missing or out of date if `convert` (not while a step may be writing the cloud: the octree
/// there is then shown as it is, or the scans without one).
pub fn load(
    cfg: &RunConfig,
    want: PreviewSource,
    trajectory: TrajectorySource,
    stride: usize,
    convert: bool,
    cancel: &AtomicBool,
) -> Result<Scene> {
    // Finished clouds always belong to the optimized trajectory.
    let trajectory = if want == PreviewSource::Scans {
        trajectory
    } else {
        TrajectorySource::Optimized
    };
    let (rows, sessions, traj_file) = session_rows(cfg, trajectory)?;
    let mut s = Scene {
        trajectory_source: if traj_file.as_ref() == Some(&cfg.opt_file()) {
            TrajectorySource::Optimized
        } else if traj_file.as_ref() == Some(&cfg.device_file_of(0)) {
            TrajectorySource::Device
        } else {
            TrajectorySource::Lio
        },
        traj_file,
        stamps: session_stamps(cfg),
        sessions,
        traj: rows
            .iter()
            .map(|r| (r[0], DVec3::new(r[1], r[2], r[3])))
            .collect(),
        rot: rows
            .iter()
            .map(|r| glam::DQuat::from_xyzw(r[4], r[5], r[6], r[7]))
            .collect(),
        ..Default::default()
    };
    let opt = cfg.opt_file();
    if s.traj_file.as_ref() == Some(&opt) {
        let q = s20_pipeline::quality_file(&opt);
        if q.exists() {
            s.quality = crate::quality::Quality::load(&q).ok();
            s.rtk_used = s
                .quality
                .as_ref()
                .is_some_and(|q| q.v.iter().any(|r| r[3].is_finite()));
        }
    }
    let frame = s20_pipeline::frames::result_frame(&cfg.data(), &opt)?;
    s.t_origin = s
        .sessions
        .first()
        .map_or_else(|| s.traj.first().map_or(0.0, |f| f.0), |f| f.start);
    if !cfg.multi()
        && let Ok(l) = s20_io::dataset::LidarData::open(&cfg.data())
        && !l.is_empty()
    {
        s.t_origin = l.scan_start(0);
        if let Some(f) = s.sessions.first_mut() {
            f.start = s.t_origin;
        }
    }
    s.rtk = if frame == s20_io::geodesy::CoordinateFrame::LocalSlam {
        Vec::new()
    } else if cfg.ppk {
        let shifts: Vec<f64> = s20_io::dataset::sessions(&cfg.data())
            .map(|d| d.iter().map(|x| x.shift).collect())
            .unwrap_or_default();
        let mut v = Vec::new();
        let mut info = Vec::new();
        for k in 0..cfg.sessions.len() {
            let Ok(p) = s20_pipeline::ppk::read(&cfg.ppk_file_of(k)) else {
                continue;
            };
            info.extend(
                p.solutions
                    .iter()
                    .zip(p.directions())
                    .map(|(r, d)| ppk_info(r, d, &cfg.params.ppk)),
            );
            if k == 0 {
                s.ppk_summary = Some(p.summary());
            }
            let shift = shifts.get(k).copied().unwrap_or(0.0);
            v.extend(p.solutions.iter().map(|r| {
                (
                    r.t + shift,
                    DVec3::from(frame.position(r.llh)),
                    r.accepted(&cfg.params.ppk),
                )
            }));
        }
        s.rtk_info = info;
        v
    } else {
        let v = rtk_fixes(&cfg.data(), &frame);
        s.rtk_info = v
            .iter()
            .map(|f| {
                if f.2 {
                    tl("機体の RTK・FIX：採用", "Device RTK, FIX: accepted").to_string()
                } else {
                    tl("機体の RTK・FIX：不採用
単独測位の位置から離れているか、衛星数が少ない（偽の FIX の疑い）", "Device RTK, FIX: rejected\nFar from the single-point position, or few satellites (suspected false FIX)")
                        .to_string()
                }
            })
            .collect();
        v
    };
    let cache_root = cfg.work.join("cache").join("preview");
    // the output cloud asked for, only it (when it cannot be shown, nothing is, and why)
    let cloud = match want {
        PreviewSource::Scans => None,
        PreviewSource::Uncolored => Some((cfg.uncolored(), PreviewSource::Uncolored)),
        PreviewSource::Colored => Some((cfg.colored(), PreviewSource::Colored)),
    };
    let cloud = match cloud {
        Some((f, _)) if !f.exists() => {
            s.unavailable = Some(tl("出力点群がありません", "No output point cloud").into());
            None
        }
        Some((f, _))
            if !convert
                && !s20_io::potree::converted_dir(&f)
                    .join("hierarchy.bin")
                    .exists() =>
        {
            s.unavailable =
                Some(tl("処理の実行中は出力点群の表示データを作りません（終わると表示します）", "The output point cloud's display data isn't made while processing (it's shown when done)").into());
            None
        }
        // An old cloud must not be overlaid on a newly optimized trajectory in another frame.
        Some((f, _)) if s20_pipeline::frames::result_frame(&cfg.data(), &f)? != frame => {
            s.unavailable = Some(
                tl("出力点群の座標系が今の軌跡と違います（点群生成から実行し直してください）", "The output point cloud's coordinates differ from the current trajectory (run again from the point cloud step)").into(),
            );
            None
        }
        c => c,
    };
    if let Some((f, src)) = cloud {
        s.has_color = src == PreviewSource::Colored;
        // the caches the output cloud had before its octree
        for label in ["colored", "uncolored"] {
            Streaming::discard(&cache_root, label);
        }
        let dir = s20_io::potree::converted_dir(&f);
        if convert && !s20_io::potree::is_current(&dir, &f) {
            let opt = s20_io::potree::WriteOptions {
                rgb: s.has_color,
                ..Default::default()
            };
            s20_pipeline::potree(&f, opt, cancel, &|_, _| {})?;
        }
        let cloud_frame = s20_pipeline::frames::result_frame(&cfg.data(), &f)?;
        let (streaming, nodes, octree) =
            octree_preview(&dir, cloud_frame.output_offset(), s.has_color, cancel)?;
        s.streaming = Some(streaming);
        s.nodes = nodes;
        s.octree = Some(Arc::new(octree));
        s.source = Some(src);
    } else if want == PreviewSource::Scans
        && !rows.is_empty()
        && cfg.done(s20_pipeline::Step::Extract)
    {
        let (streaming, nodes) = bag_preview::open(cfg, &rows, &s.sessions, stride, cancel)?;
        s.streaming = Some(streaming);
        s.nodes = nodes;
        s.source = Some(PreviewSource::Scans);
    }
    s.z_range = height_range(
        s.nodes
            .iter()
            .flat_map(|n| &n.chunks)
            .flat_map(|c| (0..c.len()).map(move |i| c.position(i))),
    );
    if let Some(b) = robust_bounds(&s.nodes) {
        s.bounds = Some(b);
        // typical full-density neighbour distance, by points
        let mut v: Vec<(f64, usize)> = s.nodes.iter().map(|n| (n.full_nn(), n.total)).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let half = s.points() / 2;
        let mut acc = 0;
        for (d, n) in v {
            acc += n;
            if acc >= half {
                s.nn_cap = 3.0 * d;
                break;
            }
        }
    } else if !s.traj.is_empty() {
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for (_, q) in &s.traj {
            lo = lo.min(*q);
            hi = hi.max(*q);
        }
        s.bounds = Some((lo - 1.0, hi + 1.0));
    }
    Ok(s)
}

/// The session recording at time `t` (the last starting at or before it).
pub fn session_of(sessions: &[SceneSession], t: f64) -> Option<usize> {
    (0..sessions.len())
        .rev()
        .find(|&k| sessions[k].start <= t + 1e-6)
}

/// Points to draw per node for a point budget: nodes outside the view get none, the others a share
/// by their size on screen (water filling: the same points per pixel for all, capped by what a node
/// has). Returns (node index, points) for the nodes to draw, largest on screen first.
/// A node's radius on screen in pixels, none when it is out of view.
struct Projection {
    view: glam::DMat4,
    target: DVec3,
    tan: f64,
    focal: f64,
    half_h: f64,
    aspect: f64,
    ortho: bool,
    viewport_h: f64,
}

impl Projection {
    fn new(cam: &crate::camera::Camera, viewport_h: f64) -> Self {
        let tan = (cam.fov * 0.5).tan();
        Projection {
            view: cam.relative_view(),
            target: DVec3::from(cam.target),
            tan,
            focal: viewport_h * 0.5 / tan,
            half_h: cam.half_height(),
            aspect: cam.aspect,
            ortho: cam.ortho,
            viewport_h,
        }
    }

    fn pixels(&self, n: &LodNode) -> Option<f64> {
        let v = self.view.transform_point3(n.center - self.target);
        let depth = -v.z;
        let r = n.radius;
        if self.ortho {
            if v.x.abs() > self.half_h * self.aspect + r || v.y.abs() > self.half_h + r {
                return None;
            }
            Some(r / self.half_h * self.viewport_h * 0.5)
        } else {
            if depth + r < 0.0
                || v.x.abs() > depth * self.tan * self.aspect + r * 1.5
                || v.y.abs() > depth * self.tan + r * 1.5
            {
                return None;
            }
            Some(r / depth.max(r * 0.5) * self.focal)
        }
    }
}

pub fn allocate(
    nodes: &[LodNode],
    cam: &crate::camera::Camera,
    viewport_h: f64,
    budget: usize,
    visible: &dyn Fn(&LodNode) -> bool,
) -> Vec<(usize, usize)> {
    let projection = Projection::new(cam, viewport_h);
    let mut vis: Vec<(usize, f64)> = Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        if !visible(n) {
            continue;
        }
        let Some(pixels) = projection.pixels(n) else {
            continue;
        };
        vis.push((i, std::f64::consts::PI * pixels * pixels));
    }
    let total: usize = vis.iter().map(|&(i, _)| nodes[i].total).sum();
    let count = |d: f64| -> usize {
        vis.iter()
            .map(|&(i, a)| ((a * d) as usize).min(nodes[i].total))
            .sum()
    };
    let mut out: Vec<(usize, usize, f64)> = if total <= budget {
        vis.iter().map(|&(i, a)| (i, nodes[i].total, a)).collect()
    } else {
        // points per pixel so that the counts add up to the budget
        let (mut lo, mut hi) = (1e-9f64, 1e9f64);
        for _ in 0..60 {
            let mid = (lo * hi).sqrt();
            if count(mid) > budget {
                hi = mid
            } else {
                lo = mid
            }
        }
        vis.iter()
            .map(|&(i, a)| (i, ((a * lo) as usize).min(nodes[i].total), a))
            .filter(|&(_, c, _)| c > 0)
            .collect()
    };
    out.sort_by(|a, b| b.2.total_cmp(&a.2));
    out.into_iter().map(|(i, c, _)| (i, c)).collect()
}

/// Radius on screen in pixels below which an octree node is not drawn (Potree's `minNodeSize`):
/// At 30 pixels the nominal spacing of a cubic 128-cell node is about 0.27 px.
/// Potree's Viewer defaults to 30 px, overriding PointCloudOctree's initial
/// 150 px. Retain selected nodes down to 80% to avoid oscillation.
const MIN_NODE_PIXELS: f64 = 30.0;

/// The octree's nodes to draw for a point budget, as Potree: from the root, the largest on screen
/// first, each after its parent, until the next would pass the budget; nodes out of view or under
/// [`MIN_NODE_PIXELS`] are left out with their subtrees. Returns (node, its points), parents first.
pub fn allocate_octree(
    nodes: &[LodNode],
    tree: &Octree,
    cam: &crate::camera::Camera,
    viewport_h: f64,
    budget: usize,
    visible: &dyn Fn(&LodNode) -> bool,
) -> Vec<(usize, usize)> {
    let projection = Projection::new(cam, viewport_h);
    let mut selected = tree.selected.lock().unwrap();
    let mut queue = std::collections::BinaryHeap::new();
    let push = |queue: &mut std::collections::BinaryHeap<(u64, usize)>, i: usize| {
        if !visible(&nodes[i]) {
            return;
        }
        if projection.pixels(&nodes[i]).is_none() {
            return;
        }
        let v = projection
            .view
            .transform_point3(nodes[i].center - projection.target);
        let distance = v.length();
        let px = nodes[i].radius * projection.focal / distance.max(1e-9);
        let threshold = MIN_NODE_PIXELS * if selected[i] { 0.8 } else { 1.0 };
        if projection.ortho || px >= threshold || tree.parent[i].is_none() {
            let weight = if projection.ortho {
                nodes[i].radius * 2.0
            } else if distance < nodes[i].radius {
                f64::MAX
            } else {
                px
            };
            queue.push((weight.to_bits(), i));
        }
    };
    if !nodes.is_empty() {
        push(&mut queue, 0);
    }
    let mut out = Vec::new();
    let mut used = 0;
    while let Some((_, i)) = queue.pop() {
        if used + nodes[i].total > budget {
            break;
        }
        used += nodes[i].total;
        out.push((i, nodes[i].total));
        for c in tree.children[i].into_iter().flatten() {
            push(&mut queue, c);
        }
    }
    selected.fill(false);
    for &(i, _) in &out {
        selected[i] = true;
    }
    out
}

#[cfg(test)]
mod height_tests {
    use super::*;

    #[test]
    fn height_range_covers_a_floor_with_few_points() {
        // a long stay on the ground floor (0..2.5 m), a short one upstairs (3..5.5 m): 1 % of the
        // points, but a fifth of the space
        let mut points = Vec::new();
        for i in 0..99_000 {
            let (x, y) = ((i % 40) as f64 * 0.25, (i / 40 % 40) as f64 * 0.25);
            points.push(DVec3::new(x, y, (i % 10) as f64 * 0.25));
        }
        for i in 0..1_000 {
            let (x, y) = ((i % 20) as f64 * 0.25, (i / 20 % 20) as f64 * 0.25);
            points.push(DVec3::new(x, y, 3.0 + (i % 10) as f64 * 0.25));
        }
        let [low, high] = height_range(points.into_iter()).unwrap();
        assert!(low <= 0.25 && high >= 5.25, "{low} {high}");
        // a stray point far up does not take the ramp
        let mut points: Vec<DVec3> = (0..40_000)
            .map(|i| DVec3::new((i % 200) as f64 * 0.25, (i / 200) as f64 * 0.25, 1.0))
            .collect();
        points.push(DVec3::new(0.0, 0.0, 80.0));
        let [_, high] = height_range(points.into_iter()).unwrap();
        assert!(high < 2.0, "{high}");
        assert_eq!(height_range(std::iter::empty()), None);
    }
}

#[cfg(test)]
mod trajectory_tests {
    use super::*;
    use std::fs;

    struct Fixture(RunConfig);

    impl Fixture {
        fn new(count: usize) -> Self {
            let work = std::env::temp_dir().join(format!(
                "s20_preview_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let cfg = RunConfig::with_sessions(
                (0..count)
                    .map(|k| PathBuf::from(format!("raw{k}")))
                    .collect(),
                work,
            );
            fs::create_dir_all(cfg.data()).unwrap();
            fs::create_dir_all(cfg.runs()).unwrap();
            let mut list = String::new();
            let mut optimized = Vec::new();
            for k in 0..count {
                let data = cfg.session_data(k);
                fs::create_dir_all(&data).unwrap();
                fs::create_dir_all(cfg.session_runs(k)).unwrap();
                let bag = cfg.session_dir(k).join("fake.bag");
                let mut point = [0u8; 19];
                point[4..8].copy_from_slice(&1.0f32.to_le_bytes());
                fs::write(&bag, point).unwrap();
                s20_io::dataset::Source { bag, size: 19 }
                    .write(&data)
                    .unwrap();
                s20_io::npy::write_f64(
                    &data.join("lidar_index.npy"),
                    &[100.0, 100e9, 0.0, 1.0, 101.1, 101.1e9, 0.0, 1.0],
                    4,
                )
                .unwrap();
                let rows: Vec<[f64; 8]> = (0..12)
                    .map(|i| {
                        [
                            100.0 + i as f64 * 0.1,
                            i as f64,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ]
                    })
                    .collect();
                s20_io::dataset::write_tum(&cfg.lio_file_of(k), &rows, "lio").unwrap();
                let shift = k as f64 * 1000.0;
                list.push_str(&format!(
                    "../sessions/{}/data {shift}\n",
                    cfg.sessions[k].name
                ));
                optimized.extend(rows.into_iter().map(|mut r| {
                    r[0] += shift;
                    r[1] += 100.0;
                    r
                }));
            }
            fs::write(cfg.data().join("sessions.txt"), list).unwrap();
            s20_io::dataset::write_tum(&cfg.opt_file(), &optimized, "opt").unwrap();
            fs::write(
                s20_pipeline::quality_file(&cfg.opt_file()),
                "100 1 1 1 1 1 1 1\n",
            )
            .unwrap();
            Self(cfg)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0.work);
        }
    }

    #[test]
    fn enu_preview_and_trajectory_export_ignore_legacy_projected_origin() {
        let fixture = Fixture::new(1);
        let cfg = &fixture.0;
        fs::create_dir_all(cfg.output()).unwrap();
        fs::write(cfg.data().join("origin.txt"), "1000 -200000 0").unwrap();
        let frame = s20_io::geodesy::CoordinateFrame::local_enu([-33.9, 151.2, 35.]).unwrap();
        s20_pipeline::frames::record(&cfg.data(), &[&cfg.opt_file()], Some(frame)).unwrap();
        s20_pipeline::export(
            &cfg.data(),
            &cfg.opt_file(),
            &cfg.output(),
            None,
            &s20_io::job::Job::stderr(),
        )
        .unwrap();
        let rows = s20_io::dataset::read_table(&cfg.output().join("trajectory.txt")).unwrap();
        assert!(!rows.is_empty());
        assert!(rows[0][0] >= 100. && rows[0][0] < 120.);
        assert_eq!(rows[0][1], 0.);
        assert_eq!(
            s20_io::geodesy::CoordinateFrame::read(&cfg.output().join("coordinate_frame.json"))
                .unwrap(),
            frame
        );
        let points = [s20_io::las::LasPoint {
            xyz: [101., 2., 3.],
            ..Default::default()
        }];
        s20_io::las::write(&cfg.colored(), &points, 1e-4, [0.; 3]).unwrap();
        s20_pipeline::frames::record(&cfg.data(), &[&cfg.colored()], Some(frame)).unwrap();
        let scene = load(
            cfg,
            PreviewSource::Colored,
            TrajectorySource::Optimized,
            4,
            true,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!((scene.nodes[0].chunks[0].position(0) - DVec3::new(101., 2., 3.)).length() < 1e-6);
        // Migration: an old coloured cloud cannot be overlaid in the new frame.
        s20_pipeline::frames::record(&cfg.data(), &[&cfg.colored()], None).unwrap();
        let old = load(
            cfg,
            PreviewSource::Colored,
            TrajectorySource::Optimized,
            4,
            true,
            &AtomicBool::new(false),
        )
        .unwrap();
        // nothing else is shown in its place, and why is said
        assert_eq!(old.source, None);
        assert!(old.unavailable.unwrap().contains("座標系"));
        s20_io::las::write(&cfg.uncolored(), &points, 1e-4, [0.; 3]).unwrap();
        s20_pipeline::frames::record(&cfg.data(), &[&cfg.uncolored()], Some(frame)).unwrap();
        let load_cloud = |want| {
            load(
                cfg,
                want,
                TrajectorySource::Optimized,
                4,
                true,
                &AtomicBool::new(false),
            )
            .unwrap()
        };
        assert_eq!(load_cloud(PreviewSource::Colored).source, None);
        let current = load_cloud(PreviewSource::Uncolored);
        assert_eq!(current.source, Some(PreviewSource::Uncolored));
        assert_eq!(current.unavailable, None);
        // in the SLAM frame the outputs have no origin on the earth: no frame file beside them
        s20_pipeline::frames::record(
            &cfg.data(),
            &[&cfg.opt_file()],
            Some(s20_io::geodesy::CoordinateFrame::LocalSlam),
        )
        .unwrap();
        s20_pipeline::export(
            &cfg.data(),
            &cfg.opt_file(),
            &cfg.output(),
            None,
            &s20_io::job::Job::stderr(),
        )
        .unwrap();
        assert!(!cfg.output().join("coordinate_frame.json").exists());
    }

    #[test]
    fn scan_cache_places_same_points_and_keeps_quality_with_optimized_only() {
        let fixture = Fixture::new(1);
        let cfg = &fixture.0;
        let load_view = |want, trajectory| {
            load(cfg, want, trajectory, 4, true, &AtomicBool::new(false)).unwrap()
        };
        let lio = load_view(PreviewSource::Scans, TrajectorySource::Lio);
        let opt = load_view(PreviewSource::Scans, TrajectorySource::Optimized);
        assert_eq!(lio.source, Some(PreviewSource::Scans));
        let first_overview = |s: &Scene| s.nodes[0].chunks[0].position(0);
        assert!((first_overview(&opt) - first_overview(&lio) - DVec3::X * 100.0).length() < 1e-5);
        assert!(lio.quality.is_none());
        assert!(!lio.rtk_used);
        assert!(opt.quality.is_some());
        assert!(opt.rtk_used);
        let product_view = load_view(PreviewSource::Colored, TrajectorySource::Lio);
        assert_eq!(product_view.traj_file, Some(cfg.opt_file()));
        assert!(product_view.quality.is_some());
        // An explicit LIO choice must not even try to read the optimization.
        fs::write(cfg.opt_file(), "invalid").unwrap();
        assert!(session_rows(cfg, TrajectorySource::Lio).is_ok());
        // without the optimization, its trajectory is not replaced by the LIO
        fs::remove_file(cfg.opt_file()).unwrap();
        let (rows, _, path) = session_rows(cfg, TrajectorySource::Optimized).unwrap();
        assert!(rows.is_empty());
        assert_eq!(path, None);
    }

    #[test]
    fn multiple_lio_sessions_are_shifted_in_time_and_arranged_side_by_side() {
        let fixture = Fixture::new(2);
        let (lio, sessions, path) = session_rows(&fixture.0, TrajectorySource::Lio).unwrap();
        assert_eq!(path, Some(fixture.0.lio_file()));
        assert!(sessions[0].placed);
        assert!(!sessions[1].placed);
        assert_eq!(lio[0][0], 100.0);
        assert_eq!(lio[12][0], 1100.0);
        assert_eq!(lio[0][1], 0.0);
        assert!(lio[12][1] > lio[11][1]);
        let (opt, sessions, _) = session_rows(&fixture.0, TrajectorySource::Optimized).unwrap();
        assert!(sessions.iter().all(|s| s.placed));
        assert_eq!(opt[0][1], 100.0);
        assert_eq!(opt[12][1], 100.0);
    }

    #[test]
    fn buffered_scans_match_mapped_scans_and_cache_groups_do_not_cross_sessions() {
        let fixture = Fixture::new(2);
        let cfg = &fixture.0;
        let lidar = s20_io::dataset::LidarData::open(&cfg.data()).unwrap();
        let dirs = s20_io::dataset::sessions(&cfg.data()).unwrap();
        for i in 0..lidar.len() {
            let source = s20_io::dataset::Source::read(&dirs[lidar.scans[i].session].dir).unwrap();
            let mut file = std::fs::File::open(source.bag).unwrap();
            let buffered = lidar.read_scan(i, &mut file).unwrap();
            let mapped = lidar.scan(i);
            assert_eq!(buffered.t, mapped.t);
            assert_eq!(buffered.xyz, mapped.xyz);
            assert_eq!(buffered.refl, mapped.refl);
            assert_eq!(buffered.tag, mapped.tag);
            assert_eq!(buffered.line, mapped.line);
        }
        assert!(lidar.read_scan(0, &mut std::io::Cursor::new([])).is_err());
        let scene = load(
            cfg,
            PreviewSource::Scans,
            TrajectorySource::Lio,
            1,
            true,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(scene.nodes.len(), 2);
        assert_eq!(scene.nodes[0].session, Some(0));
        assert_eq!(scene.nodes[1].session, Some(1));
    }

    fn load_detail(scene: &mut Scene, node: usize) -> Vec<DVec3> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        let stream = scene.streaming.as_mut().unwrap();
        loop {
            stream.update(&mut scene.nodes, &[(node, 1000)], 500_000, || {});
            assert!(stream.error.is_none(), "{:?}", stream.error);
            if !stream.pending() {
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        scene.nodes[node]
            .chunks
            .iter()
            .flat_map(|c| (0..c.len()).map(|i| c.position(i)))
            .collect()
    }

    #[test]
    fn bag_detail_is_loaded_on_demand_and_summary_survives_trajectory_changes() {
        let fixture = Fixture::new(2);
        let cfg = &fixture.0;
        let mut lio = load(
            cfg,
            PreviewSource::Scans,
            TrajectorySource::Lio,
            4,
            true,
            &AtomicBool::new(false),
        )
        .unwrap();
        let index = cfg.work.join("cache/preview/scan-index-s4.json");
        let before = fs::read(&index).unwrap();
        assert!(
            before.len() < 4096,
            "summary contains more than small scan metadata"
        );
        // only the first parts, from the summary
        assert!(lio.nodes.iter().all(|n| n.chunks.len() == 1));
        let original = load_detail(&mut lio, 0);
        assert!(!original.is_empty());
        let mut opt = load(
            cfg,
            PreviewSource::Scans,
            TrajectorySource::Optimized,
            4,
            true,
            &AtomicBool::new(false),
        )
        .unwrap();
        let optimized = load_detail(&mut opt, 0);
        assert_eq!(original.len(), optimized.len());
        for (old, new) in original.iter().zip(&optimized) {
            assert!((*new - *old - DVec3::X * 100.0).length() < 1e-5);
        }
        assert_eq!(fs::read(&index).unwrap(), before);
        let bytes: Vec<_> = fs::read_dir(index.parent().unwrap())
            .unwrap()
            .map(|p| p.unwrap().file_name())
            .collect();
        assert!(
            !bytes.iter().any(|n| n.to_string_lossy().ends_with(".bin")),
            "full scan point cache was created"
        );
        // Rebuilding a corrupted small index does not require or create a full point cache.
        fs::write(&index, "partial index").unwrap();
        let _ = load(
            cfg,
            PreviewSource::Scans,
            TrajectorySource::Lio,
            4,
            true,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(fs::read(&index).unwrap(), before);
    }

    #[test]
    fn opening_previews_removes_their_obsolete_caches() {
        let fixture = Fixture::new(1);
        let cfg = &fixture.0;
        let root = cfg.work.join("cache/preview");
        fs::create_dir_all(&root).unwrap();
        for f in [
            "scans-lio-s4.json",
            "scans-lio-s4-0000000000000001.bin",
            "colored.json",
            "colored-0000000000000001.bin",
        ] {
            fs::write(root.join(f), "old").unwrap();
        }
        let scans = |cfg: &RunConfig| {
            load(
                cfg,
                PreviewSource::Scans,
                TrajectorySource::Lio,
                4,
                true,
                &AtomicBool::new(false),
            )
            .unwrap()
        };
        let _ = scans(cfg);
        assert!(!root.join("scans-lio-s4.json").exists());
        assert!(!root.join("scans-lio-s4-0000000000000001.bin").exists());
        assert!(root.join("colored-0000000000000001.bin").exists());
        // the output cloud's cache goes once its octree is shown
        fs::create_dir_all(cfg.output()).unwrap();
        let p = s20_io::las::LasPoint {
            xyz: [1.0, 2.0, 3.0],
            ..Default::default()
        };
        s20_io::las::write(&cfg.colored(), &[p], 1e-4, [0.0; 3]).unwrap();
        let scene = load(
            cfg,
            PreviewSource::Colored,
            TrajectorySource::Lio,
            4,
            true,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(scene.source, Some(PreviewSource::Colored));
        assert!(!root.join("colored.json").exists());
        assert!(!root.join("colored-0000000000000001.bin").exists());
    }

    #[test]
    fn the_octree_keeps_colors_and_origin_for_las_and_laz() {
        use s20_io::las::LasPoint;
        let fixture = Fixture::new(1);
        let cfg = &fixture.0;
        fs::create_dir_all(cfg.output()).unwrap();
        fs::write(cfg.data().join("origin.txt"), "1000 2000 3").unwrap();
        let points = [LasPoint {
            xyz: [1001.0, 2002.0, 6.0],
            rgb: [0xff00, 0x8000, 0x1000],
            intensity: 75,
            ..Default::default()
        }];
        for laz in [true, false] {
            let mut cfg = cfg.clone();
            cfg.laz = laz;
            s20_io::las::write(&cfg.colored(), &points, 1e-4, [1000.0, 2000.0, 3.0]).unwrap();
            let scene = load(
                &cfg,
                PreviewSource::Colored,
                TrajectorySource::Lio,
                4,
                true,
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_eq!(scene.points(), 1);
            assert_eq!(scene.source, Some(PreviewSource::Colored));
            assert!(scene.octree.is_some());
            let chunk = &scene.nodes[0].chunks[0];
            assert!((chunk.position(0) - DVec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
            // the reflectivity's grey in the alpha
            assert_eq!(chunk.points[0].color, [255, 128, 16, 140]);
            let dir = s20_io::potree::converted_dir(&cfg.colored());
            assert!(s20_io::potree::is_current(&dir, &cfg.colored()));
            // the octree of the other format goes when this one is written
            let other = s20_io::potree::converted_dir(&cfg.colored().with_extension(if laz {
                "las"
            } else {
                "laz"
            }));
            assert!(!other.exists());
        }
        // while a step may write the cloud, an octree out of date is shown as it is
        let before = s20_io::potree::converted_dir(&cfg.colored());
        std::thread::sleep(std::time::Duration::from_millis(20));
        s20_io::las::write(&cfg.colored(), &points[..1], 1e-4, [1000.0, 2000.0, 3.0]).unwrap();
        let mut cfg = cfg.clone();
        cfg.laz = false;
        let scene = load(
            &cfg,
            PreviewSource::Colored,
            TrajectorySource::Lio,
            4,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(scene.source, Some(PreviewSource::Colored));
        assert!(!s20_io::potree::is_current(&before, &cfg.colored()));
    }

    #[test]
    fn the_uncoloured_cloud_is_shown_when_chosen_with_the_points_colouring_dropped() {
        use s20_io::las::LasPoint;
        let fixture = Fixture::new(1);
        let cfg = &fixture.0;
        fs::create_dir_all(cfg.output()).unwrap();
        let points: Vec<LasPoint> = (0..10)
            .map(|i| LasPoint {
                xyz: [i as f64, 0.0, 0.0],
                ..Default::default()
            })
            .collect();
        s20_io::las::write(&cfg.uncolored(), &points, 1e-4, [0.0; 3]).unwrap();
        // colouring dropped three
        s20_io::las::write(&cfg.colored(), &points[3..], 1e-4, [0.0; 3]).unwrap();
        for (want, n) in [(PreviewSource::Uncolored, 10), (PreviewSource::Colored, 7)] {
            let scene = load(
                cfg,
                want,
                TrajectorySource::Lio,
                4,
                true,
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_eq!(scene.source, Some(want));
            assert_eq!(scene.has_color, want == PreviewSource::Colored);
            assert_eq!(scene.points(), n);
        }
    }

    #[test]
    fn octree_nodes_are_drawn_parents_first_within_the_budget_and_points_sized_by_the_detail() {
        use s20_io::las::LasPoint;
        let fixture = Fixture::new(1);
        let cfg = &fixture.0;
        fs::create_dir_all(cfg.output()).unwrap();
        // a floor of 300 x 300 points 1 cm apart
        let points: Vec<LasPoint> = (0..90_000)
            .map(|i| LasPoint {
                xyz: [(i % 300) as f64 * 0.01, (i / 300) as f64 * 0.01, 0.0],
                ..Default::default()
            })
            .collect();
        s20_io::las::write(&cfg.uncolored(), &points, 1e-4, [0.0; 3]).unwrap();
        let mut scene = load(
            cfg,
            PreviewSource::Uncolored,
            TrajectorySource::Lio,
            4,
            true,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(scene.source, Some(PreviewSource::Uncolored));
        assert_eq!(scene.points(), 90_000);
        let tree = scene.octree.clone().unwrap();
        assert!(scene.nodes.len() > 9);
        let mut camera = crate::camera::Camera::default();
        camera.frame(&scene.corners());
        camera.distance = 0.5;
        let all = scene.allocate(&camera, 900.0, 1_000_000, &|_| true);
        // each node after its parent
        let mut seen = vec![false; scene.nodes.len()];
        for &(i, _) in &all {
            assert!(tree.parent[i].is_none_or(|p| seen[p]));
            seen[i] = true;
        }
        let few = scene.allocate(&camera, 900.0, scene.nodes[0].total + 10, &|_| true);
        assert_eq!(few, vec![(0, scene.nodes[0].total)]);
        // drawn with all of its children, the root's points are as small as theirs
        let drawn: Vec<bool> = (0..scene.nodes.len())
            .map(|i| tree.parent[i].is_none() || tree.parent[i] == Some(0))
            .collect();
        let spacings: Vec<_> = (0..scene.nodes.len()).map(|i| tree.spacing(i)).collect();
        let s = tree.point_spacings(0, &scene.nodes[0].chunks[0], &drawn, &spacings);
        assert_eq!(s.len(), scene.nodes[0].total);
        assert!(
            s.iter()
                .all(|&v| (v as f64 - tree.root_spacing / 2.0).abs() < 1e-6)
        );
        let alone = tree.point_spacings(
            0,
            &scene.nodes[0].chunks[0],
            &vec![false; drawn.len()],
            &spacings,
        );
        assert!(
            alone
                .iter()
                .all(|&v| (v as f64 - tree.root_spacing).abs() < 1e-6)
        );
        // read when asked for
        let stream = scene.streaming.as_mut().unwrap();
        for _ in 0..1000 {
            stream.update(&mut scene.nodes, &all, 1_000_000, || {});
            if !stream.pending()
                && all
                    .iter()
                    .all(|&(i, n)| n == 0 || !scene.nodes[i].chunks.is_empty())
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let drawn: usize = all
            .iter()
            .map(|&(i, _)| scene.nodes[i].chunks.iter().map(|c| c.len()).sum::<usize>())
            .sum();
        assert_eq!(drawn, all.iter().map(|a| a.1).sum::<usize>());
    }
}

#[cfg(test)]
mod gnss_info_tests {
    use super::ppk_info;
    use s20_pipeline::ppk::{Params, Solution};

    fn sol(status: u8, sd: f64, lat: f64) -> Solution {
        Solution {
            utc: 0.0,
            t: 0.0,
            llh: [lat, 139.0, 50.0],
            sd: [sd, sd, sd],
            ratio: 10.0,
            age: 0.5,
            status,
            satellites: 20,
        }
    }

    #[test]
    fn a_fix_says_whether_it_is_used_and_why_not() {
        let p = Params::default();
        let used = ppk_info(
            &sol(1, 0.005, 35.0),
            [Some(&sol(1, 0.005, 35.0)), Some(&sol(1, 0.005, 35.0))],
            &p,
        );
        assert!(used.starts_with("PPK・FIX：採用"), "{used}");
        // fixed forward only, 12 cm away from the backward FLOAT: its deviation grown past the limit
        let one_sided = ppk_info(
            &sol(1, 0.09, 35.0),
            [Some(&sol(1, 0.005, 35.0)), Some(&sol(2, 0.2, 35.0000011))],
            &p,
        );
        assert!(
            one_sided.starts_with("PPK・FIX（不採用）：不採用（水平の標準偏差が上限を超えている）"),
            "{one_sided}"
        );
        assert!(
            one_sided.contains("前方解 FIX・後方解 FLOAT（ずれ 水平 12.2 cm"),
            "{one_sided}"
        );
        assert!(one_sided.contains("片方向だけの FIX"), "{one_sided}");
        let float = ppk_info(&sol(2, 0.3, 35.0), [None, None], &p);
        assert!(
            float.starts_with("PPK・FLOAT：不採用（FLOAT は使わない設定"),
            "{float}"
        );
    }
}

#[cfg(test)]
mod adaptive_tests {
    use super::*;

    fn tree() -> Octree {
        Octree {
            parent: vec![None, Some(0)],
            children: vec![
                [Some(1), None, None, None, None, None, None, None],
                [None; 8],
            ],
            level: vec![0, 1],
            min: vec![DVec3::ZERO; 2],
            size: vec![DVec3::ONE, DVec3::splat(0.5)],
            root_spacing: 1.0 / 128.0,
            adaptive_spacings: (0..2).map(|_| std::sync::OnceLock::new()).collect(),
            selected: std::sync::Mutex::new(vec![false; 2]),
        }
    }

    #[test]
    fn density_matches_potree_occupancy_and_texture_quantization() {
        let chunk = |n| NodePoints {
            origin: [0.0; 3],
            points: vec![
                ViewPoint {
                    offset: [0.01; 3],
                    color: [255; 4]
                };
                n
            ],
        };
        let sparse = tree().adaptive_spacing(0, &chunk(1));
        let dense = tree().adaptive_spacing(0, &chunk(8));
        assert!((sparse - (1.0 / 128.0) * 2.0f64.powf(1.5)).abs() < 1e-12);
        assert!((dense - 1.0 / 128.0).abs() < 1e-12);
        // log2(3)/2 - 1.5 is encoded as byte 92: offset -0.8.
        assert!(
            (tree().adaptive_spacing(0, &chunk(3)) - (1.0 / 128.0) * 2.0f64.powf(0.8)).abs()
                < 1e-12
        );
        // Points in different occupied cells have density one regardless of count.
        let spread = NodePoints {
            origin: [0.0; 3],
            points: vec![
                ViewPoint {
                    offset: [0.0; 3],
                    color: [255; 4],
                },
                ViewPoint {
                    offset: [1.0; 3],
                    color: [255; 4],
                },
            ],
        };
        assert!((tree().adaptive_spacing(0, &spread) - sparse).abs() < 1e-12);
    }

    #[test]
    fn sub_150_pixel_detail_uses_available_budget() {
        let tree = tree();
        let nodes: Vec<_> = (0..2)
            .map(|_| LodNode {
                center: DVec3::ZERO,
                radius: 1.0,
                sample_nn: 0.01,
                total: 500_000,
                chunks: vec![],
                session: None,
                time: None,
            })
            .collect();
        let mut cam = crate::camera::Camera::default();
        let focal = 900.0 * 0.5 / (cam.fov * 0.5).tan();
        cam.distance = focal / 60.0;
        let small = allocate_octree(&nodes, &tree, &cam, 900.0, 500_000, &|_| true);
        let large = allocate_octree(&nodes, &tree, &cam, 900.0, 1_000_000, &|_| true);
        assert_eq!(small.iter().map(|(_, n)| n).sum::<usize>(), 500_000);
        assert_eq!(large.iter().map(|(_, n)| n).sum::<usize>(), 1_000_000);
    }

    #[test]
    fn node_selection_has_hysteresis_and_obeys_budget_and_visibility() {
        let tree = tree();
        let nodes: Vec<_> = (0..2)
            .map(|_| LodNode {
                center: DVec3::ZERO,
                radius: 1.0,
                sample_nn: 0.01,
                total: 10,
                chunks: vec![],
                session: None,
                time: None,
            })
            .collect();
        let mut cam = crate::camera::Camera::default();
        let focal = 900.0 * 0.5 / (cam.fov * 0.5).tan();
        let mut select = |px| {
            cam.distance = focal / px;
            allocate_octree(&nodes, &tree, &cam, 900.0, 100, &|_| true).len()
        };
        assert_eq!(select(28.0), 1);
        assert_eq!(select(32.0), 2);
        assert_eq!(select(28.0), 2);
        assert_eq!(select(22.0), 1);
        assert_eq!(select(28.0), 1);
        cam.distance = 1.0;
        assert_eq!(
            allocate_octree(&nodes, &tree, &cam, 900.0, 10, &|_| true).len(),
            1
        );
        assert!(allocate_octree(&nodes, &tree, &cam, 900.0, 100, &|_| false).is_empty());
        cam.ortho = true;
        cam.distance = 1000.0;
        assert_eq!(
            allocate_octree(&nodes, &tree, &cam, 900.0, 100, &|_| true).len(),
            2
        );
    }
}
