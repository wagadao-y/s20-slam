//! GPU compute (wgpu, no window) for the heavy loops. Without a usable GPU the callers keep
//! their CPU paths. The GPU computes in single precision, the CPU in double: a value at a
//! cell's border may fall on the other side.
use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

pub use s20_gpu::{Gpu, shared};

/// A buffer of `v` in the order `order`, written a few million at a time.
fn put(gpu: &Gpu, order: &[u32], v: &[[f32; 3]], label: &str) -> Result<wgpu::Buffer> {
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: order.len() as u64 * 12,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    const STEP: usize = 1 << 22;
    for (j, s) in order.chunks(STEP).enumerate() {
        let x: Vec<[f32; 3]> = s.iter().map(|&i| v[i as usize]).collect();
        gpu.queue
            .write_buffer(&buf, (j * STEP * 12) as u64, bytemuck::cast_slice(&x));
        gpu.queue.submit([]);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("GPU poll")?;
    }
    Ok(buf)
}

/// The camera of a z-buffer (single precision): world -> camera rotation rows, its centre in
/// the points' frame, the polynomial fisheye, the image size and the range limit.
#[derive(Clone, Copy, Debug)]
pub struct ZCamera {
    pub rcw: [[f32; 3]; 3],
    pub t: [f32; 3],
    pub k: [f32; 6],
    pub a11: f32,
    pub a12: f32,
    pub a22: f32,
    pub u0: f32,
    pub v0: f32,
    pub w: u32,
    pub h: u32,
    pub max_range: f32,
    /// the largest angle off the optical axis (rad)
    pub limit: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ZParams {
    r0: [f32; 4],
    r1: [f32; 4],
    r2: [f32; 4],
    t: [f32; 4],
    k03: [f32; 4],
    k45: [f32; 4],
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    cells: [i32; 4],
    d: [u32; 4],
}

/// The points (and their normals) on the GPU in parts that fit a binding, and the pipelines of
/// the z-buffer and of the points an image sees.
pub struct ZBuffer {
    gpu: &'static Gpu,
    /// (first point, positions, normals); the normals only for [`ZBuffer::visible`], None when
    /// they did not fit (the z-buffers stay on the GPU)
    parts: Vec<(u32, wgpu::Buffer, Option<wgpu::Buffer>)>,
    /// why the normals are not there
    pub no_normals: Option<String>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    visible: wgpu::ComputePipeline,
    visible_layout: wgpu::BindGroupLayout,
    /// the points the last image saw (the room the next one is given first)
    last_seen: std::sync::atomic::AtomicU64,
}

/// A block of points for [`ZBuffer::visible`]: its points (first, count) and its centre, and
/// the centre less the camera's (the points are projected from it, as the CPU's `Lens`).
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub first: u32,
    pub count: u32,
    pub centre: [f32; 3],
    pub to: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VParams {
    r0: [f32; 4],
    r1: [f32; 4],
    r2: [f32; 4],
    k03: [f32; 4],
    k45: [f32; 4],
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [u32; 4],
    e: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VRange {
    first: u32,
    thread: u32,
    pad: [u32; 2],
    centre: [f32; 4],
    to: [f32; 4],
}

/// The points an image sees: (block order, u, v, range, theta), in block order.
pub type Seen = Vec<(u32, f32, f32, f32, f32)>;

thread_local! {
    /// Each thread's buffers of the visible pass: its points on the GPU, and their read-back.
    static VISIBLE_OUT: std::cell::RefCell<Option<(u64, wgpu::Buffer)>> =
        const { std::cell::RefCell::new(None) };
    static VISIBLE_BACK: std::cell::RefCell<Option<(u64, wgpu::Buffer)>> =
        const { std::cell::RefCell::new(None) };
}

/// The thread's buffer of `cell`, made again (a quarter larger, up to `most` bytes) when smaller
/// than `size`.
fn kept_buffer(
    cell: &'static std::thread::LocalKey<std::cell::RefCell<Option<(u64, wgpu::Buffer)>>>,
    dev: &wgpu::Device,
    size: u64,
    most: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) -> wgpu::Buffer {
    cell.with_borrow_mut(|kept| {
        if let Some((cap, b)) = kept.as_ref()
            && *cap >= size
        {
            return b.clone();
        }
        let cap = (size + size / 4).next_multiple_of(256).min(most).max(size);
        let b = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: cap,
            usage,
            mapped_at_creation: false,
        });
        *kept = Some((cap, b.clone()));
        b
    })
}

/// The points the visible pass wrote (index, u, v, range, angle: five words each, in any order)
/// in the order of their indices, each index once: a bit set per index, each point placed at
/// the number of the bits before its own (as sorting by the index, without the sort's passes).
fn in_index_order(out: &[u32]) -> Seen {
    let rows = out.as_chunks::<5>().0;
    let most = rows.iter().map(|w| w[0]).max().unwrap_or(0) as usize;
    let mut bits = vec![0u64; most / 64 + 1];
    for w in rows {
        bits[w[0] as usize / 64] |= 1 << (w[0] % 64);
    }
    let mut before = Vec::with_capacity(bits.len());
    let mut n = 0u32;
    for b in &bits {
        before.push(n);
        n += b.count_ones();
    }
    let f = f32::from_bits;
    let mut seen = vec![(0u32, 0f32, 0f32, 0f32, 0f32); rows.len()];
    for w in rows {
        let (word, bit) = (w[0] as usize / 64, w[0] % 64);
        let at = before[word] + (bits[word] & ((1u64 << bit) - 1)).count_ones();
        seen[at as usize] = (w[0], f(w[1]), f(w[2]), f(w[3]), f(w[4]));
    }
    seen
}

impl ZBuffer {
    /// Uploads `points`, then `normals`, in the order `order` (the order the ranges of
    /// [`ZBuffer::fill`] refer to), a few million at a time (no full copy on either side).
    /// Without room for the normals only the z-buffers are on the GPU (see `no_normals`).
    pub fn new(
        gpu: &'static Gpu,
        points: &[[f32; 3]],
        normals: &[[f32; 3]],
        order: &[u32],
    ) -> Result<Self> {
        let mut z = gpu.checked(|| Self::upload(gpu, points, order))?;
        let max = z.part_size();
        match gpu.checked(|| {
            order
                .chunks(max)
                .map(|c| put(gpu, c, normals, "normals"))
                .collect::<Result<Vec<_>>>()
        }) {
            Ok(bufs) => {
                for (p, b) in z.parts.iter_mut().zip(bufs) {
                    p.2 = Some(b);
                }
            }
            Err(e) => z.no_normals = Some(format!("{e:#}")),
        }
        Ok(z)
    }
    /// The points in a part (the most a binding holds).
    fn part_size(&self) -> usize {
        let l = &self.gpu.limits;
        (l.max_storage_buffer_binding_size.min(l.max_buffer_size) / 12) as usize
    }
    fn upload(gpu: &'static Gpu, points: &[[f32; 3]], order: &[u32]) -> Result<Self> {
        let max = gpu
            .limits
            .max_storage_buffer_binding_size
            .min(gpu.limits.max_buffer_size)
            / 12;
        anyhow::ensure!(max > 0, "no storage buffers");
        let mut parts = Vec::new();
        for (k, c) in order.chunks(max as usize).enumerate() {
            parts.push((
                (k as u64 * max) as u32,
                put(gpu, c, points, "points")?,
                None,
            ));
        }
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zbuffer"),
                source: wgpu::ShaderSource::Wgsl(include_str!("zbuffer.wgsl").into()),
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
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("zbuffer"),
                entries: &[
                    entry(0, wgpu::BufferBindingType::Uniform),
                    entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
        let pl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("zbuffer"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("zbuffer"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let vmodule = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("visible"),
                source: wgpu::ShaderSource::Wgsl(include_str!("visible.wgsl").into()),
            });
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let visible_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("visible"),
                    entries: &[
                        entry(0, wgpu::BufferBindingType::Uniform),
                        entry(1, ro),
                        entry(2, ro),
                        entry(3, ro),
                        entry(4, ro),
                        entry(5, rw),
                        entry(6, rw),
                    ],
                });
        let vpl = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visible"),
                bind_group_layouts: &[Some(&visible_layout)],
                immediate_size: 0,
            });
        let visible = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("visible"),
                layout: Some(&vpl),
                module: &vmodule,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(ZBuffer {
            gpu,
            parts,
            no_normals: None,
            pipeline,
            layout,
            visible,
            visible_layout,
            last_seen: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// The points of `blocks` that image `cam` sees: within range, the field of view and the
    /// image, not behind `grids` (the z-buffer's minima within 1..6 cells, then its depth slopes,
    /// each `zw` x `zh`; the window from `reach` = a11 x spacing / cell, the slopes with `graze`)
    /// and facing it where they have a normal.
    #[allow(clippy::too_many_arguments)]
    pub fn visible(
        &self,
        cam: &ZCamera,
        reach: f32,
        graze: bool,
        blocks: &[Block],
        zw: usize,
        zh: usize,
        grids: &[f32],
    ) -> Result<Seen> {
        let r = self
            .gpu
            .checked(|| self.run_visible(cam, reach, graze, blocks, zw, zh, grids));
        if r.is_err() {
            // (a buffer that failed is not kept for the next image)
            VISIBLE_OUT.with_borrow_mut(|b| *b = None);
            VISIBLE_BACK.with_borrow_mut(|b| *b = None);
        }
        r
    }
    #[allow(clippy::too_many_arguments)]
    fn run_visible(
        &self,
        cam: &ZCamera,
        reach: f32,
        graze: bool,
        blocks: &[Block],
        zw: usize,
        zh: usize,
        grids: &[f32],
    ) -> Result<Seen> {
        if let Some(e) = &self.no_normals {
            anyhow::bail!("no normals on the GPU: {e}");
        }
        let dev = &self.gpu.device;
        let total: u64 = blocks.iter().map(|b| b.count as u64).sum();
        if total == 0 {
            return Ok(Vec::new());
        }
        let grid = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("visible grids"),
            contents: bytemuck::cast_slice(grids),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // room for a quarter of the points, or for a quarter more than the last image saw (a
        // wide outdoor recording's images see a quarter of a cloud: two passes each before),
        // all of them when that is short
        let last = self.last_seen.load(std::sync::atomic::Ordering::Relaxed);
        let most = self.max_places();
        let mut places = (total / 4)
            .max(last + last / 4)
            .max(1 << 16)
            .min(total.max(1 << 16))
            .min(most);
        loop {
            let (count, seen) =
                self.dispatch_visible(cam, reach, graze, blocks, zw, zh, &grid, places)?;
            let Some(seen) = seen else {
                anyhow::ensure!(
                    (count as u64) <= most,
                    "{count} visible points: more than a GPU buffer holds"
                );
                places = count as u64;
                continue;
            };
            self.last_seen
                .store(count as u64, std::sync::atomic::Ordering::Relaxed);
            return Ok(seen);
        }
    }
    /// One pass of [`ZBuffer::visible`] with room for `places` points: their number, and those
    /// written (all when they fit).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_visible(
        &self,
        cam: &ZCamera,
        reach: f32,
        graze: bool,
        blocks: &[Block],
        zw: usize,
        zh: usize,
        grid: &wgpu::Buffer,
        places: u64,
    ) -> Result<(u32, Option<Seen>)> {
        let dev = &self.gpu.device;
        let count = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("visible count"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        // the thread's buffer for the points (100 MB and more an image outdoors), kept
        let out = kept_buffer(
            &VISIBLE_OUT,
            dev,
            places * 20,
            self.max_places() * 20,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "visible out",
        );
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("visible"),
        });
        let mut kept = Vec::new();
        for (k, (first, pts, nrm)) in self.parts.iter().enumerate() {
            let nrm = nrm.as_ref().context("no normals")?;
            let end = self.parts.get(k + 1).map_or(u32::MAX, |p| p.0);
            let mut list: Vec<VRange> = Vec::new();
            let mut threads = 0u32;
            for b in blocks {
                let (s, e) = (b.first.max(*first), (b.first + b.count).min(end));
                if s < e {
                    list.push(VRange {
                        first: s,
                        thread: threads,
                        pad: [0; 2],
                        centre: [b.centre[0], b.centre[1], b.centre[2], 0.],
                        to: [b.to[0], b.to[1], b.to[2], 0.],
                    });
                    threads += e - s;
                }
            }
            if threads == 0 {
                continue;
            }
            let r = cam.rcw;
            let params = VParams {
                r0: [r[0][0], r[0][1], r[0][2], 0.],
                r1: [r[1][0], r[1][1], r[1][2], 0.],
                r2: [r[2][0], r[2][1], r[2][2], 0.],
                k03: [cam.k[0], cam.k[1], cam.k[2], cam.k[3]],
                k45: [cam.k[4], cam.k[5], 0., 0.],
                a: [cam.a11, cam.a12, cam.a22, cam.u0],
                b: [
                    cam.v0,
                    (cam.w - 2) as f32,
                    (cam.h - 2) as f32,
                    cam.max_range,
                ],
                c: [cam.limit, reach, if graze { 1. } else { 0. }, 0.],
                d: [zw as u32, list.len() as u32, threads, *first],
                e: [zh as u32, places as u32, 0, 0],
            };
            let uni = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("visible params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let lst = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("visible ranges"),
                contents: bytemuck::cast_slice(&list),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let res = [
                uni.as_entire_binding(),
                pts.as_entire_binding(),
                nrm.as_entire_binding(),
                lst.as_entire_binding(),
                grid.as_entire_binding(),
                count.as_entire_binding(),
                // (as much as this image is given: the kept buffer may be larger)
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &out,
                    offset: 0,
                    size: std::num::NonZeroU64::new(places * 20),
                }),
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
                label: Some("visible"),
                layout: &self.visible_layout,
                entries: &entries,
            });
            let groups = threads.div_ceil(64);
            let x = groups.min(65535);
            let y = groups.div_ceil(x);
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("visible"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.visible);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
            kept.push((uni, lst, bind));
        }
        self.gpu.queue.submit([enc.finish()]);
        let n = self.read(&count, 4)?[0];
        if n as u64 > places {
            return Ok((n, None));
        }
        // in block order (the threads wrote them in any order), straight from the read-back
        // buffer: each point's place among them by its index, which only it has
        let seen = if n > 0 {
            let back = kept_buffer(
                &VISIBLE_BACK,
                dev,
                n as u64 * 20,
                self.max_places() * 20,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                "visible read back",
            );
            self.read_into(&out, &back, n as u64 * 20, in_index_order)?
        } else {
            Vec::new()
        };
        drop(kept);
        Ok((n, Some(seen)))
    }

    /// The points a visible pass can write (20 bytes each, within one buffer and one binding).
    fn max_places(&self) -> u64 {
        let l = &self.gpu.limits;
        (l.max_storage_buffer_binding_size.min(l.max_buffer_size) / 20).max(1)
    }

    /// `f` of the first `bytes` of `buf` (a multiple of 4) as words, read through `back` (a
    /// buffer to map, at least that large) without copying them out.
    fn read_into<T>(
        &self,
        buf: &wgpu::Buffer,
        back: &wgpu::Buffer,
        bytes: u64,
        f: impl FnOnce(&[u32]) -> T,
    ) -> Result<T> {
        let dev = &self.gpu.device;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read"),
        });
        enc.copy_buffer_to_buffer(buf, 0, back, 0, bytes);
        let index = self.gpu.queue.submit([enc.finish()]);
        let slice = back.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let r = dev
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .context("GPU poll")
            .and_then(|_| rx.recv().context("GPU map")?.context("GPU map"))
            .and_then(|_| {
                let view = slice
                    .get_mapped_range()
                    .map_err(|e| anyhow::anyhow!("GPU map: {e:?}"))?;
                Ok(f(bytemuck::cast_slice::<u8, u32>(&view)))
            });
        // (left mapped, the thread's buffer could not be used again)
        if r.is_ok() {
            back.unmap();
        } else {
            VISIBLE_BACK.with_borrow_mut(|b| *b = None);
        }
        r
    }
    /// The first `bytes` of `buf` (a multiple of 4), as words.
    fn read(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u32>> {
        let dev = &self.gpu.device;
        let back = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read back"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read"),
        });
        enc.copy_buffer_to_buffer(buf, 0, &back, 0, bytes);
        let index = self.gpu.queue.submit([enc.finish()]);
        let slice = back.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // this thread's work only (not the other threads' submitted since)
        dev.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .context("GPU poll")?;
        rx.recv().context("GPU map")?.context("GPU map")?;
        let words = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("GPU map: {e:?}"))?;
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        back.unmap();
        Ok(words)
    }

    /// The nearest range per 8 x 8 pixel cell of image `cam` over the points of
    /// `ranges` (first point, count), in cells `lo..=hi` (others infinite); `zw` x `zh` cells.
    #[allow(clippy::too_many_arguments)]
    pub fn fill(
        &self,
        cam: &ZCamera,
        ranges: &[(u32, u32)],
        zw: usize,
        zh: usize,
        lo: [i64; 2],
        hi: [i64; 2],
    ) -> Result<Vec<f32>> {
        self.gpu.checked(|| self.run(cam, ranges, zw, zh, lo, hi))
    }
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        cam: &ZCamera,
        ranges: &[(u32, u32)],
        zw: usize,
        zh: usize,
        lo: [i64; 2],
        hi: [i64; 2],
    ) -> Result<Vec<f32>> {
        let dev = &self.gpu.device;
        let n = zw * zh;
        let zb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("zb"),
            contents: bytemuck::cast_slice(&vec![f32::INFINITY.to_bits(); n]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let back = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zb back"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc =
            dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zb") });
        let mut kept = Vec::new();
        for (k, (first, buf, _)) in self.parts.iter().enumerate() {
            let end = self.parts.get(k + 1).map_or(u32::MAX, |p| p.0);
            // the ranges' points in this part, by (first point, first thread)
            let mut list: Vec<[u32; 2]> = Vec::new();
            let mut total = 0u32;
            for &(s, c) in ranges {
                let (a, b) = (s.max(*first), (s + c).min(end));
                if a < b {
                    list.push([a, total]);
                    total += b - a;
                }
            }
            if total == 0 {
                continue;
            }
            let r = cam.rcw;
            let params = ZParams {
                r0: [r[0][0], r[0][1], r[0][2], 0.],
                r1: [r[1][0], r[1][1], r[1][2], 0.],
                r2: [r[2][0], r[2][1], r[2][2], 0.],
                t: [cam.t[0], cam.t[1], cam.t[2], 0.],
                k03: [cam.k[0], cam.k[1], cam.k[2], cam.k[3]],
                k45: [cam.k[4], cam.k[5], 0., 0.],
                a: [cam.a11, cam.a12, cam.a22, cam.u0],
                b: [
                    cam.v0,
                    (cam.w - 2) as f32,
                    (cam.h - 2) as f32,
                    cam.max_range,
                ],
                c: [cam.limit, 0., 0., 0.],
                cells: [lo[0] as i32, lo[1] as i32, hi[0] as i32, hi[1] as i32],
                d: [zw as u32, list.len() as u32, total, *first],
            };
            let uni = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("zb params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let lst = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("zb ranges"),
                contents: bytemuck::cast_slice(&list),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zb"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uni.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: zb.as_entire_binding(),
                    },
                ],
            });
            let groups = total.div_ceil(64);
            let x = groups.min(65535);
            let y = groups.div_ceil(x);
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("zb"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
            kept.push((uni, lst, bind));
        }
        enc.copy_buffer_to_buffer(&zb, 0, &back, 0, (n * 4) as u64);
        let index = self.gpu.queue.submit([enc.finish()]);
        let slice = back.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // this thread's work only (not the other threads' submitted since)
        dev.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .context("GPU poll")?;
        rx.recv().context("GPU map")?.context("GPU map")?;
        let out: Vec<f32> = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("GPU map: {e:?}"))?;
            bytemuck::cast_slice::<u8, u32>(&view)
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect()
        };
        back.unmap();
        drop(kept);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_points_come_in_the_order_of_their_indices_as_sorted() {
        // indices once each, in a pseudo-random order, each with its four values
        let mut seed = 7u64;
        let mut idx: Vec<u32> = (0..100_000u32).map(|k| k * 3 + (k % 7)).collect();
        for k in (1..idx.len()).rev() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            idx.swap(k, (seed >> 33) as usize % (k + 1));
        }
        let out: Vec<u32> = idx
            .iter()
            .flat_map(|&i| {
                [
                    i,
                    (i as f32).to_bits(),
                    1f32.to_bits(),
                    2f32.to_bits(),
                    3f32.to_bits(),
                ]
            })
            .collect();
        let got = super::in_index_order(&out);
        let mut want = idx.clone();
        want.sort_unstable();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!((g.0, g.1), (*w, *w as f32));
        }
    }
}
