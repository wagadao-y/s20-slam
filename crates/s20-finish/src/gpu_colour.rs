//! The colours of an image's visible points on the GPU (`colour.wgsl`), after its visible pass
//! ([`ZBuffer::encode_visible`]): their colours, scores and grazing flags, the image's cell means
//! around each point and the cells the gains take, as the CPU's `Scene::view` and `gather` in
//! color.rs. The cells' sums are integers (the packed colours), so the threads' order does not
//! matter and a run gives the same as the next; they and the score in single precision differ
//! a little from the CPU's (double).

use crate::gpu::{Block, ZBuffer, ZCamera};
use anyhow::{Context, Result};
use std::num::NonZeroU64;
use std::sync::{Condvar, Mutex};
use wgpu::util::DeviceExt;

/// Sets of the per-image buffers at most (each some 500 MB on the GPU outdoors), so that the
/// threads take turns rather than fill the GPU (4650 images: three gathered in 299 s, two in
/// 326 s and 2 GB less at the highest).
const SETS: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    t: [f32; 4],
    angles: [f32; 4],
    size: [u32; 4],
    first: [u32; 4],
}

/// The colour passes and their layout.
pub(crate) struct Pipes {
    layout: wgpu::BindGroupLayout,
    colour: wgpu::ComputePipeline,
    around_mean: wgpu::ComputePipeline,
    gain_cells: wgpu::ComputePipeline,
}

impl Pipes {
    fn new(z: &ZBuffer) -> Result<Self> {
        let gpu = z.gpu;
        anyhow::ensure!(
            gpu.limits.max_storage_buffers_per_shader_stage >= 12,
            "too few storage buffers for the colour passes"
        );
        let dev = &gpu.device;
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("colour"),
            source: wgpu::ShaderSource::Wgsl(include_str!("colour.wgsl").into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("colour"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, ro),
                entry(2, ro),
                entry(3, ro),
                entry(4, rw),
                entry(5, ro),
                entry(6, ro),
                entry(7, ro),
                entry(8, ro),
                entry(9, ro),
                entry(10, rw),
                entry(11, rw),
                entry(12, rw),
            ],
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("colour"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Pipes {
            colour: pipeline("colour"),
            around_mean: pipeline("around_mean"),
            gain_cells: pipeline("gain_cells"),
            layout,
        })
    }
}

/// One image's buffers on the GPU, kept for the next image (made larger when short).
struct Set {
    img: (u64, wgpu::Buffer),
    out: (u64, wgpu::Buffer),
    res: (u64, wgpu::Buffer),
    gains: (u64, wgpu::Buffer),
    back: (u64, wgpu::Buffer),
    sums: wgpu::Buffer,
    counts: wgpu::Buffer,
}

/// A part's points and cells on the GPU for [`ZBuffer::colours`], and the sets of buffers.
pub struct ColourPart {
    slot_of: wgpu::Buffer,
    cell_of: wgpu::Buffer,
    around: wgpu::Buffer,
    owned: wgpu::Buffer,
    cells: u32,
    sets: Mutex<(Vec<Set>, usize)>,
    free: Condvar,
}

/// An image's coloured points: slot, score, grazing, colour and the cells' mean (packed), and
/// its gain entries (cell, points, mean colour).
pub struct Colours {
    pub pts: Vec<u32>,
    pub score: Vec<f32>,
    pub wide: Vec<bool>,
    pub c: Vec<[u16; 3]>,
    pub m: Vec<[u16; 3]>,
    pub gain: Vec<(u32, u32, [f32; 3])>,
}

/// The camera and the score's terms of an image for [`ZBuffer::colours`].
pub struct Shot<'a> {
    pub cam: &'a ZCamera,
    pub reach: f32,
    pub graze: bool,
    /// the camera's centre (the points' frame) and the turn term (turn weight x omega / 30)
    pub centre: [f32; 3],
    pub turn: f32,
    /// soft and hard angles off the axis (the hard one the largest f32 not beyond it, so that
    /// "beyond" is as in double), the slope beyond the soft one
    pub soft: f32,
    pub hard: f32,
    pub slope: f32,
    /// RGB8, `width` x `height`
    pub image: &'a [u8],
    pub width: u32,
    pub height: u32,
}

fn storage(dev: &wgpu::Device, size: u64, usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage,
        mapped_at_creation: false,
    })
}

/// `data` into a new storage buffer, a few million words at a time.
fn upload(z: &ZBuffer, data: &[u32], label: &str) -> Result<wgpu::Buffer> {
    let gpu = z.gpu;
    let buf = storage(
        &gpu.device,
        (data.len() * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        label,
    );
    const STEP: usize = 1 << 22;
    for (k, c) in data.chunks(STEP).enumerate() {
        gpu.queue
            .write_buffer(&buf, (k * STEP * 4) as u64, bytemuck::cast_slice(c));
        let index = gpu.queue.submit([]);
        gpu.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .context("GPU poll")?;
    }
    Ok(buf)
}

/// `buf` when it holds `size` bytes, else a new one a quarter larger (at most `most`).
fn grown(
    dev: &wgpu::Device,
    kept: &mut (u64, wgpu::Buffer),
    size: u64,
    most: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) {
    if kept.0 < size {
        let cap = (size + size / 4).next_multiple_of(256).min(most).max(size);
        *kept = (cap, storage(dev, cap, usage, label));
    }
}

impl ZBuffer {
    /// The colour passes, made once (None: they could not be).
    fn pipes(&self) -> Option<&Pipes> {
        self.colour
            .get_or_init(|| {
                self.gpu
                    .checked(|| Pipes::new(self))
                    .map_err(|e| eprintln!("colours on the CPU (GPU: {e:#})"))
                    .ok()
            })
            .as_ref()
    }

    /// A part's points and cells for [`ZBuffer::colours`]: each point's slot (in block order,
    /// u32::MAX where not coloured), each slot's cell and its place among the 8 cells around it
    /// (the grid's fractions, 10 bits each), those cells, the cells the part owns, and the most
    /// points of a cell (its colour sums must not overflow).
    pub fn colour_part(
        &self,
        slot_of: &[u32],
        cell_of: &[[u32; 2]],
        around: &[[u32; 8]],
        owned: &[bool],
        most_in_cell: usize,
    ) -> Result<ColourPart> {
        anyhow::ensure!(self.no_normals.is_none(), "no normals on the GPU");
        // a cell's sum of colours x 256 (at most 65535 a point) within 32 bits
        anyhow::ensure!(
            (most_in_cell as u64) * 65535 <= u32::MAX as u64,
            "{most_in_cell} points in a cell: their colour sums would overflow"
        );
        self.pipes().context("no colour passes")?;
        let most = self.max_places() * 20;
        anyhow::ensure!(
            (around.len() * 32) as u64 <= most && (owned.len() * 16) as u64 <= most,
            "too many points or cells for one GPU buffer"
        );
        self.gpu.checked(|| {
            let owned: Vec<u32> = owned.iter().map(|&o| o as u32).collect();
            Ok(ColourPart {
                slot_of: upload(self, slot_of, "colour slots")?,
                cell_of: upload(self, cell_of.as_flattened(), "colour cells")?,
                around: upload(self, around.as_flattened(), "colour around")?,
                owned: upload(self, &owned, "colour owned")?,
                cells: owned.len() as u32,
                sets: Mutex::new((Vec::new(), 0)),
                free: Condvar::new(),
            })
        })
    }

    /// The coloured points of an image (see [`Colours`]): its visible pass over `blocks`
    /// (`grids`: the z-buffer's window minima, then its depth slopes), then the colour passes.
    #[allow(clippy::too_many_arguments)]
    pub fn colours(
        &self,
        part: &ColourPart,
        shot: &Shot,
        blocks: &[Block],
        zw: usize,
        zh: usize,
        grids: &[f32],
    ) -> Result<Colours> {
        // a set of buffers: a free one, a new one while there are fewer than SETS, else wait
        let mut set = {
            let mut s = part.sets.lock().unwrap();
            loop {
                if let Some(set) = s.0.pop() {
                    break Some(set);
                }
                if s.1 < SETS {
                    s.1 += 1;
                    break None;
                }
                s = part.free.wait(s).unwrap();
            }
        };
        let r = self.gpu.checked(|| {
            if set.is_none() {
                set = Some(self.new_set(part));
            }
            self.colours_with(part, set.as_mut().unwrap(), shot, blocks, zw, zh, grids)
        });
        let mut s = part.sets.lock().unwrap();
        match (set, r.is_ok()) {
            (Some(set), true) => s.0.push(set),
            // (a set that failed is not used again)
            _ => s.1 -= 1,
        }
        part.free.notify_one();
        r
    }

    fn new_set(&self, part: &ColourPart) -> Set {
        let dev = &self.gpu.device;
        use wgpu::BufferUsages as U;
        let empty = |label: &str, usage: U| (0, storage(dev, 16, usage, label));
        Set {
            img: empty("colour image", U::STORAGE | U::COPY_DST),
            out: empty("colour visible", U::STORAGE | U::COPY_SRC),
            res: empty("colour results", U::STORAGE | U::COPY_SRC),
            gains: empty("colour gains", U::STORAGE | U::COPY_SRC),
            back: empty("colour read back", U::MAP_READ | U::COPY_DST),
            sums: storage(
                dev,
                part.cells as u64 * 16,
                U::STORAGE | U::COPY_DST,
                "colour sums",
            ),
            counts: storage(
                dev,
                16,
                U::STORAGE | U::COPY_SRC | U::COPY_DST,
                "colour counts",
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn colours_with(
        &self,
        part: &ColourPart,
        set: &mut Set,
        shot: &Shot,
        blocks: &[Block],
        zw: usize,
        zh: usize,
        grids: &[f32],
    ) -> Result<Colours> {
        let dev = &self.gpu.device;
        let pipes = self.pipes().context("no colour passes")?;
        let total: u64 = blocks.iter().map(|b| b.count as u64).sum();
        let empty = Colours {
            pts: Vec::new(),
            score: Vec::new(),
            wide: Vec::new(),
            c: Vec::new(),
            m: Vec::new(),
            gain: Vec::new(),
        };
        if total == 0 {
            return Ok(empty);
        }
        // the points the results' read-back buffer holds with their gain entries (21 words of 20
        // for each 20 points) within the GPU's largest buffer
        let most = self
            .max_places()
            .min(self.gpu.limits.max_buffer_size.saturating_sub(16) / 21)
            .max(1);
        use wgpu::BufferUsages as U;
        // the image, padded to whole words
        let bytes = (shot.image.len() as u64).next_multiple_of(4);
        grown(
            dev,
            &mut set.img,
            bytes,
            bytes.max(most * 20),
            U::STORAGE | U::COPY_DST,
            "colour image",
        );
        if shot.image.len().is_multiple_of(4) {
            self.gpu.queue.write_buffer(&set.img.1, 0, shot.image);
        } else {
            let mut padded = shot.image.to_vec();
            padded.resize(bytes as usize, 0);
            self.gpu.queue.write_buffer(&set.img.1, 0, &padded);
        }
        let grid = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colour grids"),
            contents: bytemuck::cast_slice(grids),
            usage: U::STORAGE,
        });
        let last = self.last_seen.load(std::sync::atomic::Ordering::Relaxed);
        let mut places = (total / 4)
            .max(last + last / 4)
            .max(1 << 16)
            .min(total.max(1 << 16))
            .min(most);
        loop {
            for (buf, label, usage) in [
                (&mut set.out, "colour visible", U::STORAGE | U::COPY_SRC),
                (&mut set.res, "colour results", U::STORAGE | U::COPY_SRC),
            ] {
                grown(dev, buf, places * 20, most * 20, usage, label);
            }
            // a gain entry takes 20 points of the image or more
            let room = places / 20 + 1;
            grown(
                dev,
                &mut set.gains,
                room * 20,
                most * 20,
                U::STORAGE | U::COPY_SRC,
                "colour gains",
            );
            grown(
                dev,
                &mut set.back,
                16 + (places + room) * 20,
                16 + (most + most / 20 + 1) * 20,
                U::MAP_READ | U::COPY_DST,
                "colour read back",
            );
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("colour"),
            });
            enc.clear_buffer(&set.counts, 0, None);
            enc.clear_buffer(&set.sums, 0, None);
            let kept = self.encode_visible(
                &mut enc,
                shot.cam,
                shot.reach,
                shot.graze,
                blocks,
                zw,
                zh,
                &grid,
                &set.counts,
                &set.out.1,
                places,
            )?;
            let mut binds = Vec::new();
            for (k, (first, pts, nrm)) in self.parts.iter().enumerate() {
                let nrm = nrm.as_ref().context("no normals")?;
                let end = self.parts.get(k + 1).map_or(u32::MAX, |p| p.0);
                let params = Params {
                    t: [shot.centre[0], shot.centre[1], shot.centre[2], shot.turn],
                    angles: [shot.soft, shot.hard, shot.slope, 0.],
                    size: [shot.width, shot.height, places as u32, part.cells],
                    first: [*first, end, room as u32, 0],
                };
                let uni = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("colour params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: U::UNIFORM,
                });
                fn sized(b: &wgpu::Buffer, bytes: u64) -> wgpu::BindingResource<'_> {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: b,
                        offset: 0,
                        size: NonZeroU64::new(bytes),
                    })
                }
                let res = [
                    uni.as_entire_binding(),
                    pts.as_entire_binding(),
                    nrm.as_entire_binding(),
                    sized(&set.out.1, places * 20),
                    set.counts.as_entire_binding(),
                    part.slot_of.as_entire_binding(),
                    part.cell_of.as_entire_binding(),
                    part.around.as_entire_binding(),
                    part.owned.as_entire_binding(),
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &set.img.1,
                        offset: 0,
                        size: NonZeroU64::new(bytes),
                    }),
                    set.sums.as_entire_binding(),
                    sized(&set.res.1, places * 20),
                    sized(&set.gains.1, room * 20),
                ];
                let entries: Vec<wgpu::BindGroupEntry> = res
                    .into_iter()
                    .enumerate()
                    .map(|(i, resource)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource,
                    })
                    .collect();
                let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("colour"),
                    layout: &pipes.layout,
                    entries: &entries,
                });
                binds.push((uni, bind));
            }
            let groups = |n: u64| {
                let g = (n.div_ceil(64)).max(1) as u32;
                let x = g.min(65535);
                (x, g.div_ceil(x))
            };
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("colour"),
                    timestamp_writes: None,
                });
                // the colours (and the cells' sums) of every points' buffer, then their means
                let (x, y) = groups(places);
                for pipe in [&pipes.colour, &pipes.around_mean] {
                    for (_, bind) in &binds {
                        pass.set_pipeline(pipe);
                        pass.set_bind_group(0, bind, &[]);
                        pass.dispatch_workgroups(x, y, 1);
                    }
                }
                let (x, y) = groups(part.cells as u64);
                pass.set_pipeline(&pipes.gain_cells);
                pass.set_bind_group(0, &binds[0].1, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
            enc.copy_buffer_to_buffer(&set.counts, 0, &set.back.1, 0, 16);
            let index = self.gpu.queue.submit([enc.finish()]);
            let counts: Vec<u32> = self.fetch(&set.back.1, index, 16, |w| w.to_vec())?;
            drop(kept);
            let (n, g) = (counts[0] as u64, counts[1] as u64);
            if n > places {
                anyhow::ensure!(
                    n <= most,
                    "{n} visible points: more than a GPU buffer holds"
                );
                places = n;
                continue;
            }
            self.last_seen
                .store(n, std::sync::atomic::Ordering::Relaxed);
            let g = g.min(room);
            // the results and the gain entries, read back together
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("colour read"),
            });
            if n > 0 {
                enc.copy_buffer_to_buffer(&set.res.1, 0, &set.back.1, 16, n * 20);
            }
            if g > 0 {
                enc.copy_buffer_to_buffer(&set.gains.1, 0, &set.back.1, 16 + n * 20, g * 20);
            }
            let index = self.gpu.queue.submit([enc.finish()]);
            return self.fetch(&set.back.1, index, 16 + (n + g) * 20, |w| {
                let (n, g) = (n as usize, g as usize);
                let mut out = Colours {
                    pts: Vec::with_capacity(n),
                    score: Vec::with_capacity(n),
                    wide: Vec::with_capacity(n),
                    c: Vec::with_capacity(n),
                    m: Vec::with_capacity(n),
                    gain: Vec::with_capacity(g),
                };
                for r in w[4..4 + 5 * n].as_chunks::<5>().0 {
                    if r[0] == u32::MAX {
                        continue;
                    }
                    let lo = |x: u32| (x & 0xffff) as u16;
                    let hi = |x: u32| (x >> 16) as u16;
                    out.pts.push(r[0] & 0x7fff_ffff);
                    out.wide.push(r[0] >> 31 == 1);
                    out.score.push(f32::from_bits(r[1]));
                    out.c.push([lo(r[2]), hi(r[2]), lo(r[3])]);
                    out.m.push([hi(r[3]), lo(r[4]), hi(r[4])]);
                }
                for e in w[4 + 5 * n..4 + 5 * (n + g)].as_chunks::<5>().0 {
                    out.gain
                        .push((e[0], e[1], [e[2], e[3], e[4]].map(f32::from_bits)));
                }
                out
            });
        }
    }

    /// `f` of the first `bytes` of `back` (mapped) once submission `index` is done.
    fn fetch<T>(
        &self,
        back: &wgpu::Buffer,
        index: wgpu::SubmissionIndex,
        bytes: u64,
        f: impl FnOnce(&[u32]) -> T,
    ) -> Result<T> {
        let dev = &self.gpu.device;
        let slice = back.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .context("GPU poll")?;
        rx.recv().context("GPU map")?.context("GPU map")?;
        let r = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("GPU map: {e:?}"))?;
            f(bytemuck::cast_slice::<u8, u32>(&view))
        };
        back.unmap();
        Ok(r)
    }
}
