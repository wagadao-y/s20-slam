//! Preview nodes read on demand from their source (the bag, the Potree octree). Only each node's
//! resident parts (the overview, together) and the requested parts live in RAM.
use crate::lang::tl;
use crate::scene::{LodNode, NodePoints};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

/// In the identity of the summaries derived from files (see [`fingerprint`]).
const VERSION: u32 = 2;
/// Points kept resident for the whole view, as the first parts of all nodes.
pub const OVERVIEW_POINTS: usize = 32_768;
/// Bounded staging space, independent of the dataset size.
const READ_BATCH: usize = 500_000;
const MAX_CHUNK: usize = 32_768;

#[derive(Clone)]
struct Part {
    offset: u64,
    count: usize,
}

struct Node {
    parts: Vec<Part>,
    /// the first parts that stay loaded
    resident: usize,
}

/// The parts of `total` points from `start` (`stride` apart): the resident first one of `first`
/// points, then fourfold larger ones up to MAX_CHUNK. As a node's points are in random order,
/// every prefix of parts is an even sample of it.
fn split(start: u64, total: usize, first: usize, stride: u64) -> Vec<Part> {
    let mut parts = Vec::new();
    let (mut at, mut size) = (0, first.max(1));
    while at < total {
        let end = (at + size).min(total);
        parts.push(Part {
            offset: start + at as u64 * stride,
            count: end - at,
        });
        at = end;
        size = (size * 4).min(MAX_CHUNK);
    }
    parts
}

/// File identity includes the path, size and modification time, including missing dependencies.
pub fn fingerprint(paths: &[PathBuf], options: impl Hash) -> Result<u64> {
    let mut h = DefaultHasher::new();
    VERSION.hash(&mut h);
    options.hash(&mut h);
    for path in paths {
        path.hash(&mut h);
        match fs::metadata(path) {
            Ok(m) => {
                m.len().hash(&mut h);
                m.modified()?
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos()
                    .hash(&mut h);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0u8.hash(&mut h),
            Err(e) => return Err(e).with_context(|| format!("stat {}", path.display())),
        }
    }
    Ok(h.finish())
}

pub fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!("preview cancelled");
    }
    Ok(())
}

type ReadResult = Result<Vec<(usize, usize, Arc<NodePoints>)>>;
type ReadPoints = dyn Fn(usize, u64, usize, [f64; 3]) -> Result<Arc<NodePoints>> + Send + Sync;

/// One reader at a time, with at most READ_BATCH points in flight. Old views can be discarded
/// without retaining their points in GPU bookkeeping or a worker queue.
pub struct Streaming {
    nodes: Vec<Node>,
    reader: Arc<ReadPoints>,
    max_nodes_per_batch: usize,
    /// the parts of a batch read at once (else in order: the bag is read forwards)
    parallel: bool,
    pending: Option<mpsc::Receiver<ReadResult>>,
    /// the last update had nothing to read: what the view asks for is loaded
    idle: bool,
    cancel: Arc<AtomicBool>,
    pub error: Option<String>,
}

impl Drop for Streaming {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Streaming {
    /// Stream directly from an indexed source, keeping only node metadata and the first parts:
    /// each node comes with its first `first[i]` point slots read (its only chunk), the rest is
    /// read on demand. Offsets here are point slots; a source may return fewer points after
    /// range/time filtering.
    pub fn from_source(
        nodes: Vec<LodNode>,
        first: &[usize],
        reader: impl Fn(usize, u64, usize, [f64; 3]) -> Result<Arc<NodePoints>> + Send + Sync + 'static,
    ) -> (Self, Vec<LodNode>) {
        let index = nodes
            .iter()
            .zip(first)
            .map(|(n, &first)| Node {
                parts: split(0, n.total, first, 1),
                resident: 1,
            })
            .collect();
        (Self::new(index, reader, 8, false), nodes)
    }

    /// Nodes read whole (the nodes of an octree, each compressed as one): those that come with
    /// their points stay, the others are read when asked for.
    pub fn whole_nodes(
        nodes: Vec<LodNode>,
        reader: impl Fn(usize, u64, usize, [f64; 3]) -> Result<Arc<NodePoints>> + Send + Sync + 'static,
    ) -> (Self, Vec<LodNode>) {
        let index = nodes
            .iter()
            .map(|n| Node {
                parts: if n.total > 0 {
                    vec![Part {
                        offset: 0,
                        count: n.total,
                    }]
                } else {
                    Vec::new()
                },
                resident: n.chunks.len().min(1),
            })
            .collect();
        (Self::new(index, reader, 64, true), nodes)
    }

    fn new(
        nodes: Vec<Node>,
        reader: impl Fn(usize, u64, usize, [f64; 3]) -> Result<Arc<NodePoints>> + Send + Sync + 'static,
        max_nodes_per_batch: usize,
        parallel: bool,
    ) -> Self {
        Self {
            nodes,
            reader: Arc::new(reader),
            max_nodes_per_batch,
            parallel,
            pending: None,
            idle: false,
            cancel: Arc::new(AtomicBool::new(false)),
            error: None,
        }
    }

    /// Discard obsolete generated point caches, without following paths from their manifests.
    pub fn discard(root: &Path, label: &str) {
        let path = root.join(format!("{label}.json"));
        if let Ok(files) = fs::read_dir(root) {
            for file in files.flatten() {
                let name = file.file_name();
                let name = name.to_string_lossy();
                if let Some(key) = name
                    .strip_prefix(&format!("{label}-"))
                    .and_then(|s| s.strip_suffix(".bin"))
                    && key.len() == 16
                    && key.bytes().all(|c| c.is_ascii_hexdigit())
                {
                    let _ = fs::remove_file(file.path());
                }
            }
        }
        let _ = fs::remove_file(path);
    }

    pub fn total(&self) -> usize {
        self.nodes
            .iter()
            .flat_map(|n| &n.parts)
            .map(|p| p.count)
            .sum()
    }
    pub fn pending(&self) -> bool {
        self.pending.is_some()
    }
    /// Whether what the last view asked for is loaded (nothing pending, nothing left to read).
    #[cfg(test)]
    pub fn idle(&self) -> bool {
        self.idle && self.pending.is_none()
    }

    /// Internal memory allowance for this view: its requested prefixes rounded
    /// to whole read parts, plus the pinned overview (counted only once).
    /// Unrequested detail is not cached. This prevents an independent memory
    /// limit or off-screen overview from starving the selected draw budget.
    pub fn view_budget(&self, allocation: &[(usize, usize)]) -> usize {
        let mut required: Vec<usize> = self
            .nodes
            .iter()
            .map(|n| n.parts.iter().take(n.resident).map(|p| p.count).sum())
            .collect();
        for &(i, count) in allocation {
            let mut covered = required[i];
            for part in self.nodes[i].parts.iter().skip(self.nodes[i].resident) {
                if covered >= count {
                    break;
                }
                covered += part.count;
            }
            required[i] = covered;
        }
        required.iter().sum()
    }

    /// Evict before scheduling reads, and fetch only bounded prefixes selected for this view.
    pub fn update(
        &mut self,
        nodes: &mut [LodNode],
        allocation: &[(usize, usize)],
        budget: usize,
        repaint: impl Fn() + Send + 'static,
    ) {
        if let Some(rx) = &self.pending {
            match rx.try_recv() {
                Ok(Ok(parts)) => {
                    for (ni, ci, points) in parts {
                        if nodes[ni].chunks.len() == ci {
                            nodes[ni].chunks.push(points);
                        }
                    }
                    self.pending = None;
                }
                Ok(Err(e)) => {
                    self.error = Some(format!("{e:#}"));
                    self.pending = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.error = Some(
                        tl(
                            "表示データの読み込みが終了しました",
                            "Reading the display data ended",
                        )
                        .into(),
                    );
                    self.pending = None;
                }
                Err(mpsc::TryRecvError::Empty) => return,
            }
        }
        // the resident parts stay: the overview
        let pinned: usize = nodes
            .iter()
            .zip(&self.nodes)
            .flat_map(|(n, s)| n.chunks.iter().take(s.resident))
            .map(|c| c.len())
            .sum();
        let mut left = budget.saturating_sub(pinned);
        let mut desired: Vec<usize> = self.nodes.iter().map(|n| n.resident).collect();
        for &(ni, count) in allocation {
            let parts = &self.nodes[ni].parts;
            let mut covered: usize = parts[..desired[ni]].iter().map(|p| p.count).sum();
            for part in &parts[desired[ni]..] {
                if covered >= count || part.count > left {
                    break;
                }
                desired[ni] += 1;
                covered += part.count;
                left -= part.count;
            }
        }
        for (n, &keep) in nodes.iter_mut().zip(&desired) {
            n.chunks.truncate(keep);
        }
        self.idle = true;
        if self.error.is_some() {
            return;
        }
        let mut jobs = Vec::new();
        let mut loading = 0;
        let mut source_nodes = 0;
        for &(ni, _) in allocation {
            if nodes[ni].chunks.len() >= desired[ni] {
                continue;
            }
            if source_nodes >= self.max_nodes_per_batch {
                break;
            }
            source_nodes += 1;
            for ci in nodes[ni].chunks.len()..desired[ni] {
                let part = &self.nodes[ni].parts[ci];
                // a part larger than the batch is read by itself
                if loading > 0 && loading + part.count > READ_BATCH {
                    break;
                }
                loading += part.count;
                jobs.push((ni, ci, part.clone(), nodes[ni].center.to_array()));
            }
        }
        if jobs.is_empty() {
            return;
        }
        self.idle = false;
        let reader = self.reader.clone();
        let cancel = self.cancel.clone();
        let parallel = self.parallel;
        let (tx, rx) = mpsc::channel();
        self.pending = Some(rx);
        std::thread::spawn(move || {
            let read = |(ni, ci, part, origin): (usize, usize, Part, [f64; 3])| {
                check_cancel(&cancel)?;
                Ok((ni, ci, reader(ni, part.offset, part.count, origin)?))
            };
            let result = crate::panics::catch(|| {
                if parallel {
                    jobs.into_par_iter().map(read).collect()
                } else {
                    jobs.into_iter().map(read).collect()
                }
            });
            let _ = tx.send(result.map_err(anyhow::Error::from).and_then(|r| r));
            repaint();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{ViewPoint, lod_node};
    use glam::DVec3;
    use std::time::{Duration, Instant};

    /// Three nodes of 60 000 points 100 m apart, read from memory by point slot.
    fn source() -> (Streaming, Vec<LodNode>) {
        let all: Vec<Vec<([f64; 3], [u8; 4])>> = (0..3)
            .map(|k| {
                (0..60_000)
                    .map(|i| {
                        (
                            [
                                k as f64 * 100.0 + (i % 100) as f64 * 0.01,
                                (i / 100) as f64 * 0.01,
                                0.0,
                            ],
                            [k as u8, 20, 30, 255],
                        )
                    })
                    .collect()
            })
            .collect();
        let first = 32;
        let nodes: Vec<LodNode> = all
            .iter()
            .enumerate()
            .map(|(k, pts)| {
                let mut n = lod_node(pts).unwrap();
                n.session = Some(k);
                n.chunks = vec![n.chunks[0].clone()];
                let c = Arc::make_mut(&mut n.chunks[0]);
                c.points.truncate(first);
                n
            })
            .collect();
        let firsts = vec![first; 3];
        let all = Arc::new(all);
        Streaming::from_source(nodes, &firsts, move |ni, offset, count, origin| {
            let points = all[ni][offset as usize..offset as usize + count]
                .iter()
                .map(|(p, c)| ViewPoint {
                    offset: std::array::from_fn(|d| (p[d] - origin[d]) as f32),
                    color: *c,
                })
                .collect();
            Ok(Arc::new(NodePoints { origin, points }))
        })
    }

    fn resident(nodes: &[LodNode]) -> usize {
        nodes.iter().flat_map(|n| &n.chunks).map(|c| c.len()).sum()
    }

    fn settle(
        stream: &mut Streaming,
        nodes: &mut [LodNode],
        allocation: &[(usize, usize)],
        budget: usize,
    ) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            stream.update(nodes, allocation, budget, || {});
            assert!(stream.error.is_none(), "{:?}", stream.error);
            if !stream.pending() {
                break;
            }
            assert!(Instant::now() < deadline, "preview reader stalled");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(resident(nodes) <= budget);
    }

    #[test]
    fn view_budget_covers_read_rounding_and_overview_and_shrinks_with_the_view() {
        let (mut stream, mut nodes) = source();
        let allocation = [(0, 100)];
        let budget = stream.view_budget(&allocation);
        // 32 + 128 covers the 100 drawn points, plus 32 from each other node.
        assert_eq!(budget, 224);
        settle(&mut stream, &mut nodes, &allocation, budget);
        assert_eq!(resident(&nodes), budget);
        assert!(nodes[0].chunks.iter().map(|c| c.len()).sum::<usize>() >= 100);
        let allocation = [(1, 60_000)];
        let budget = stream.view_budget(&allocation);
        assert_eq!(budget, 60_000 + 64);
        settle(&mut stream, &mut nodes, &allocation, budget);
        assert_eq!(nodes[0].chunks.len(), 1);
        assert_eq!(
            nodes[1].chunks.iter().map(|c| c.len()).sum::<usize>(),
            60_000
        );
        let budget = stream.view_budget(&[]);
        assert_eq!(budget, 96);
        settle(&mut stream, &mut nodes, &[], budget);
        assert_eq!(resident(&nodes), 96);
    }

    #[test]
    fn first_parts_stay_as_the_overview_and_detail_moves_with_the_view_under_budget() {
        let (mut stream, mut nodes) = source();
        assert_eq!(stream.total(), 180_000);
        // every node has its first part, with its own session
        assert_eq!(resident(&nodes), 3 * 32);
        assert!(nodes.iter().all(|n| n.chunks.len() == 1));
        for (k, n) in nodes.iter().enumerate() {
            assert_eq!(n.session, Some(k));
        }
        settle(&mut stream, &mut nodes, &[(0, 60_000)], 50_000);
        assert!(nodes[0].chunks.len() > 1);
        let old_points = Arc::downgrade(&nodes[0].chunks[1]);
        settle(&mut stream, &mut nodes, &[(2, 60_000)], 50_000);
        assert_eq!(nodes[0].chunks.len(), 1);
        assert!(
            old_points.upgrade().is_none(),
            "evicted payload is still held"
        );
        // the second part of node 2, read from its place in the source
        assert!((200.0..201.0).contains(&nodes[2].chunks[1].position(0).x));
        assert_eq!(nodes[2].chunks[1].points[0].color[0], 2);
        // Reducing the budget also evicts existing details; the first parts remain.
        settle(&mut stream, &mut nodes, &[(2, 60_000)], 3 * 32 + 100);
        assert_eq!(nodes[2].chunks.iter().map(|c| c.len()).sum::<usize>(), 32);
        settle(&mut stream, &mut nodes, &[], 50_000);
        assert_eq!(resident(&nodes), 3 * 32);
    }

    #[test]
    fn whole_nodes_are_read_and_evicted_as_one_and_those_given_with_points_stay() {
        let node = |x: f64, n: usize, loaded: bool| {
            let pts: Vec<_> = (0..n)
                .map(|i| ([x + i as f64 * 1e-3, 0.0, 0.0], [255; 4]))
                .collect();
            let mut node = lod_node(&pts).unwrap();
            if !loaded {
                node.chunks.clear();
            }
            node
        };
        let nodes = vec![
            node(0.0, 100, true),
            node(10.0, 5000, false),
            node(20.0, 7000, false),
        ];
        let (mut stream, mut nodes) = Streaming::whole_nodes(nodes, |ni, _, count, origin| {
            Ok(Arc::new(NodePoints {
                origin,
                points: vec![
                    ViewPoint {
                        offset: [ni as f32, 0.0, 0.0],
                        color: [0; 4],
                    };
                    count
                ],
            }))
        });
        assert_eq!(stream.total(), 12_100);
        settle(&mut stream, &mut nodes, &[(1, 5000), (2, 7000)], 20_000);
        assert_eq!(nodes[1].chunks[0].len(), 5000);
        assert_eq!(nodes[2].chunks[0].len(), 7000);
        assert_eq!(nodes[2].chunks[0].points[0].offset[0], 2.0);
        // too little left for node 2 as a whole: it goes, node 0 stays
        settle(&mut stream, &mut nodes, &[(1, 5000), (2, 7000)], 10_000);
        assert_eq!(resident(&nodes), 5100);
        assert!(nodes[2].chunks.is_empty());
        settle(&mut stream, &mut nodes, &[], 10_000);
        assert_eq!(resident(&nodes), 100);
        assert_eq!(nodes[0].center, DVec3::new(0.0495, 0.0, 0.0));
    }

    #[test]
    fn parts_grow_fourfold_and_cover_every_node() {
        let parts = split(100, 1000, 8, 16);
        assert_eq!(parts[0].count, 8);
        assert_eq!(parts[1].count, 32);
        assert_eq!(parts[1].offset, 100 + 8 * 16);
        assert_eq!(parts.iter().map(|p| p.count).sum::<usize>(), 1000);
    }

    #[test]
    fn input_changes_change_identity() {
        let dir = std::env::temp_dir().join(format!("s20_stream_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("trajectory.txt");
        let _ = fs::remove_file(&missing);
        let key = fingerprint(std::slice::from_ref(&missing), 4).unwrap();
        fs::write(&missing, "new trajectory").unwrap();
        assert_ne!(fingerprint(&[missing], 4).unwrap(), key);
        let _ = fs::remove_dir_all(&dir);
    }
}
