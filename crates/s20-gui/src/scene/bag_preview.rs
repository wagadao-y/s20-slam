//! Small trajectory-independent scan summaries; detailed points always come from the original bag.
use super::*;
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::BufReader;
use std::sync::Mutex;

const VERSION: u32 = 1;
const STEP: f64 = 0.005;
/// Points between trajectory samples further apart (s) are not placed: the device's own trajectory
/// has gaps (about 10 s while it starts, and up to 1 s in a park), which are not bridged.
const MAX_GAP: f64 = 1.0;

/// The pose at `t`, unless it falls in a gap of the trajectory.
fn pose(traj: &Traj, t: f64) -> Option<(M3, V3)> {
    (traj.gap_at(t) <= MAX_GAP).then(|| traj.at(t))
}

#[derive(Serialize, Deserialize)]
struct Sample {
    xyz: [f32; 3],
    dt: f32,
    reflectivity: u8,
}

#[derive(Serialize, Deserialize)]
struct Group {
    start: usize,
    end: usize,
    count: usize,
    t0: f64,
    t1: f64,
    lo: [f32; 3],
    hi: [f32; 3],
    samples: Vec<Sample>,
}

#[derive(Serialize, Deserialize)]
struct Summary {
    version: u32,
    key: u64,
    groups: Vec<Group>,
}

fn ranges(lidar: &s20_io::dataset::LidarData) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut start = 0;
    while start < lidar.len() {
        let mut end = (start + SCANS_PER_NODE).min(lidar.len());
        if let Some(boundary) = (start + 1..end).find(|&i| {
            lidar.scans[i].session != lidar.scans[start].session
                || lidar.scan_start(i) - lidar.scan_start(start) > 1.2
        }) {
            end = boundary;
        }
        result.push((start, end));
        start = end;
    }
    result
}

fn readers(cfg: &RunConfig) -> Result<Vec<BufReader<File>>> {
    s20_io::dataset::sessions(&cfg.data())?
        .iter()
        .map(|d| {
            let source = s20_io::dataset::Source::read(&d.dir)?;
            Ok(BufReader::with_capacity(1 << 20, File::open(source.bag)?))
        })
        .collect()
}

fn summary(
    cfg: &RunConfig,
    lidar: &s20_io::dataset::LidarData,
    stride: usize,
    cancel: &AtomicBool,
) -> Result<Summary> {
    let key = stream::fingerprint(&scan_dependencies(cfg), (VERSION, stride, SCANS_PER_NODE))?;
    let root = cfg.work.join("cache").join("preview");
    let path = root.join(format!("scan-index-s{stride}.json"));
    let cached = fs::read(&path)
        .ok()
        .and_then(|data| serde_json::from_slice::<Summary>(&data).ok());
    if let Some(cached) = cached.filter(|s| {
        s.version == VERSION
            && s.key == key
            && s.groups.iter().all(|g| {
                g.start < g.end
                    && g.end <= lidar.len()
                    && g.count > 0
                    && g.t0.is_finite()
                    && g.t1.is_finite()
                    && g.t1 >= g.t0
                    && g.t1 - g.t0 < 2.0
                    && g.samples.len() <= 32
                    && g.lo.iter().chain(&g.hi).all(|v| v.is_finite())
                    && g.samples
                        .iter()
                        .all(|s| s.xyz.iter().all(|v| v.is_finite()) && s.dt.is_finite())
                    && g.count
                        <= lidar.scans[g.start..g.end]
                            .iter()
                            .map(|s| s.count)
                            .sum::<usize>()
            })
    }) {
        return Ok(cached);
    }
    let ranges = ranges(lidar);
    let samples = stream::OVERVIEW_POINTS
        .div_ceil(ranges.len().max(1))
        .clamp(1, 32);
    let mut files = readers(cfg)?;
    let mut groups = Vec::new();
    for (start, end) in ranges {
        stream::check_cancel(cancel)?;
        let group = ScanGroup::read(lidar, &mut files, start, end, stride)?;
        let Some(raw) = group.nodes.first() else {
            continue;
        };
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in &raw.x {
            for i in 0..3 {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        groups.push(Group {
            start,
            end,
            count: raw.x.len(),
            t0: raw.t0,
            t1: raw.t1,
            lo,
            hi,
            samples: (0..samples.min(raw.x.len()))
                .map(|i| Sample {
                    xyz: raw.x[i],
                    dt: raw.dt[i],
                    reflectivity: raw.refl[i],
                })
                .collect(),
        });
    }
    let index = Summary {
        version: VERSION,
        key,
        groups,
    };
    stream::check_cancel(cancel)?;
    fs::create_dir_all(&root)?;
    // Summary writes are small. A unique temporary file keeps interrupted requests separate.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let temp = root.join(format!(
        "scan-index-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        fs::write(&temp, serde_json::to_vec(&index)?)?;
        stream::check_cancel(cancel)?;
        fs::rename(&temp, &path)?;
        Ok(())
    })();
    let _ = fs::remove_file(temp);
    result?;
    Ok(index)
}

/// Conservative transformed bounds: every deskew pose transforms the entire local bounding box.
/// This covers points even when a rapid turn moves them outside the sampled overview's bounds.
fn bounds(group: &Group, traj: &Traj) -> (DVec3, f64) {
    let low = V3::from(group.lo.map(f64::from));
    let high = V3::from(group.hi.map(f64::from));
    let center = (low + high) * 0.5;
    let half = (high - low) * 0.5;
    let (mut lo, mut hi) = (V3::repeat(f64::INFINITY), V3::repeat(f64::NEG_INFINITY));
    let steps = ((group.t1 - group.t0) / STEP).ceil() as usize + 1;
    for k in 0..steps {
        let (r, p) = traj.at(group.t0 + k as f64 * STEP);
        let c = r * center + p;
        let extent = r.map(f64::abs) * half;
        lo = lo.inf(&(c - extent));
        hi = hi.sup(&(c + extent));
    }
    let c = (lo + hi) * 0.5;
    (DVec3::new(c.x, c.y, c.z), (hi - lo).norm() * 0.5 + 0.05)
}

/// A one-second group being read: its node, raw scans and the poses of its scans (none in a gap).
type CurrentGroup = (usize, ScanGroup, Vec<Option<(M3, V3)>>);

struct Reader {
    lidar: s20_io::dataset::LidarData,
    files: Vec<BufReader<File>>,
    ranges: Vec<(usize, usize)>,
    stride: usize,
    trajectory: Traj,
    /// Bounded working data for the current one-second group; never written as a point cache.
    current: Option<CurrentGroup>,
}

impl Reader {
    fn read(
        &mut self,
        node: usize,
        offset: u64,
        count: usize,
        origin: [f64; 3],
    ) -> Result<Arc<NodePoints>> {
        if self.current.as_ref().is_none_or(|(n, _, _)| *n != node) {
            let &(start, end) = self.ranges.get(node).context("scan group outside index")?;
            let raw = ScanGroup::read(&self.lidar, &mut self.files, start, end, self.stride)?;
            let poses = raw.nodes.first().map_or_else(Vec::new, |n| {
                let steps = ((n.t1 - n.t0) / STEP).ceil() as usize + 1;
                (0..steps)
                    .map(|k| pose(&self.trajectory, n.t0 + k as f64 * STEP))
                    .collect()
            });
            self.current = Some((node, raw, poses));
        }
        let (_, raw, poses) = self.current.as_ref().unwrap();
        let mut points = Vec::with_capacity(count);
        if let Some(raw) = raw.nodes.first() {
            let start = (offset as usize).min(raw.x.len());
            let end = start.saturating_add(count).min(raw.x.len());
            for i in start..end {
                let t = raw.t0 + raw.dt[i] as f64;
                if t < self.trajectory.start() || t > self.trajectory.end() {
                    continue;
                }
                let k = ((raw.dt[i] as f64 / STEP).round() as usize).min(poses.len() - 1);
                let Some((r, p)) = &poses[k] else {
                    continue;
                };
                let w = r * V3::from(raw.x[i].map(f64::from)) + p;
                let g = super::reflectivity_grey(raw.refl[i].into());
                points.push(ViewPoint {
                    offset: [
                        (w.x - origin[0]) as f32,
                        (w.y - origin[1]) as f32,
                        (w.z - origin[2]) as f32,
                    ],
                    color: [g; 4],
                });
            }
        }
        Ok(Arc::new(NodePoints { origin, points }))
    }
}

pub(super) fn open(
    cfg: &RunConfig,
    rows: &[[f64; 8]],
    sessions: &[SceneSession],
    stride: usize,
    cancel: &AtomicBool,
) -> Result<(Streaming, Vec<LodNode>)> {
    ensure!(stride > 0, "scan stride must be positive");
    let lidar = s20_io::dataset::LidarData::open(&cfg.data())?;
    let summary = summary(cfg, &lidar, stride, cancel)?;
    let trajectory = Traj::from_rows(rows);
    let mut nodes = Vec::new();
    let mut ranges = Vec::new();
    // the summary's samples are the first point slots of each group: its resident first part
    let mut first = Vec::new();
    for group in summary.groups {
        stream::check_cancel(cancel)?;
        if group.t1 < trajectory.start() || group.t0 > trajectory.end() {
            continue;
        }
        let session = session_of(sessions, group.t0);
        let (center, radius) = bounds(&group, &trajectory);
        let steps = ((group.t1 - group.t0) / STEP).ceil() as usize + 1;
        let slots = group.samples.len();
        let mut sample = Vec::new();
        for s in group.samples {
            let t = group.t0 + s.dt as f64;
            if t < trajectory.start() || t > trajectory.end() {
                continue;
            }
            let k = ((s.dt as f64 / STEP).round() as usize).min(steps - 1);
            let Some((r, p)) = pose(&trajectory, group.t0 + k as f64 * STEP) else {
                continue;
            };
            let w = r * V3::from(s.xyz.map(f64::from)) + p;
            let g = super::reflectivity_grey(s.reflectivity.into());
            sample.push(([w.x, w.y, w.z], [g; 4]));
        }
        let spacing = lod_node(&sample).map_or(0.05, |n| n.sample_nn);
        let sample_nn =
            spacing * (sample.len().max(1) as f64 / group.count.min(FIRST_CHUNK) as f64).sqrt();
        let origin = center.to_array();
        let points = sample
            .iter()
            .map(|(w, color)| ViewPoint {
                offset: std::array::from_fn(|i| (w[i] - origin[i]) as f32),
                color: *color,
            })
            .collect();
        nodes.push(LodNode {
            center,
            radius,
            sample_nn,
            total: group.count,
            chunks: vec![Arc::new(NodePoints { origin, points })],
            session,
            time: Some([group.t0, group.t1]),
        });
        first.push(slots);
        ranges.push((group.start, group.end));
    }
    let files = readers(cfg)?;
    let reader = Mutex::new(Reader {
        lidar,
        files,
        ranges,
        stride,
        trajectory,
        current: None,
    });
    let result = Streaming::from_source(nodes, &first, move |ni, offset, count, origin| {
        reader
            .lock()
            .map_err(|_| anyhow::anyhow!("scan reader poisoned"))?
            .read(ni, offset, count, origin)
    });
    // These were generated by the previous full-point cache implementation. They are obsolete.
    for stride in [1, 2, 4, 8, 16] {
        let root = cfg.work.join("cache").join("preview");
        for source in ["lio", "opt"] {
            Streaming::discard(&root, &format!("scans-{source}-s{stride}"));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_cover_all_deskewed_corners_during_a_turn() {
        let group = Group {
            start: 0,
            end: 10,
            count: 1000,
            t0: 0.0,
            t1: 0.9,
            lo: [-2.0, -1.0, -0.5],
            hi: [5.0, 4.0, 2.0],
            samples: Vec::new(),
        };
        let trajectory = Traj::from_rows(&[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ]);
        let (center, radius) = bounds(&group, &trajectory);
        for k in 0..=180 {
            let (r, p) = trajectory.at(k as f64 * STEP);
            for bits in 0..8 {
                let corner = V3::new(
                    if bits & 1 == 0 { -2.0 } else { 5.0 },
                    if bits & 2 == 0 { -1.0 } else { 4.0 },
                    if bits & 4 == 0 { -0.5 } else { 2.0 },
                );
                let w = r * corner + p;
                assert!(DVec3::new(w.x, w.y, w.z).distance(center) <= radius);
            }
        }
    }
}
