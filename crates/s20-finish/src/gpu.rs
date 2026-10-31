//! GPU compute (wgpu, no window) for the heavy loops. Without a usable GPU the callers keep
//! their CPU paths. The GPU computes in single precision, the CPU in double: a value at a
//! cell's border may fall on the other side.
use anyhow::{Context, Result};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

/// The process-wide GPU, created on first use (None when there is no usable one).
pub fn shared() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| Gpu::new().ok()).as_ref()
}

/// A compute device and its queue.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
    pub limits: wgpu::Limits,
}

impl Gpu {
    /// The high-performance adapter with its own limits (large buffers), or an error. The backend
    /// is fixed (DX12 on Windows), so that the same adapter and shader compiler are used.
    pub fn new() -> Result<Self> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = if cfg!(windows) {
            wgpu::Backends::DX12
        } else {
            wgpu::Backends::PRIMARY
        };
        let instance = wgpu::Instance::new(desc.with_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter")?;
        let limits = adapter.limits();
        let info = adapter.get_info();
        let name = format!("{}, {:?}", info.name, info.backend);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("s20 compute"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .context("GPU device")?;
        // Errors outside a scope (see `checked`) and a lost device are noted, not a panic (wgpu's
        // default): the GPU is not used again.
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            eprintln!("GPU error: {e}");
            BROKEN.store(true, Ordering::Relaxed);
        }));
        device.set_device_lost_callback(|reason, msg| {
            eprintln!("GPU lost: {reason:?} {msg}");
            BROKEN.store(true, Ordering::Relaxed);
        });
        Ok(Gpu {
            device,
            queue,
            name,
            limits,
        })
    }

    /// `f`'s GPU work with its validation, out-of-memory and internal errors as an error (the
    /// caller goes on on the CPU), not a panic.
    pub fn checked<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        anyhow::ensure!(!BROKEN.load(Ordering::Relaxed), "the GPU failed before");
        let scopes = [
            wgpu::ErrorFilter::OutOfMemory,
            wgpu::ErrorFilter::Validation,
            wgpu::ErrorFilter::Internal,
        ]
        .map(|e| self.device.push_error_scope(e));
        let r = f();
        let mut first = None;
        for s in scopes.into_iter().rev() {
            if let Some(e) = pollster::block_on(s.pop()) {
                first.get_or_insert(e);
            }
        }
        if let Some(e) = first {
            anyhow::bail!("GPU: {e}");
        }
        anyhow::ensure!(!BROKEN.load(Ordering::Relaxed), "the GPU failed");
        r
    }
}

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

/// Set once the GPU failed outside a checked scope or was lost.
static BROKEN: AtomicBool = AtomicBool::new(false);

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

/// Sorts `keys` by their high 32 bits (least significant digit first, 11 bits at a time, only
/// as many digits as the largest key needs; stable, so the low bits break no ties).
fn radix_sort_high(keys: &mut Vec<u64>) {
    let most = keys.iter().map(|&k| k >> 32).max().unwrap_or(0);
    let bits = 64 - most.leading_zeros();
    let mut tmp = vec![0u64; keys.len()];
    let mut shift = 32;
    while shift < 32 + bits {
        let mut count = [0usize; 1 << 11];
        for &k in keys.iter() {
            count[((k >> shift) & 0x7ff) as usize] += 1;
        }
        let mut at = 0;
        for c in count.iter_mut() {
            (*c, at) = (at, at + *c);
        }
        for &k in keys.iter() {
            let d = ((k >> shift) & 0x7ff) as usize;
            tmp[count[d]] = k;
            count[d] += 1;
        }
        std::mem::swap(keys, &mut tmp);
        shift += 11;
    }
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
        self.gpu
            .checked(|| self.run_visible(cam, reach, graze, blocks, zw, zh, grids))
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
        // a quarter of the points first (an image sees some 13% of the points it may), all of
        // them when that is short
        let mut places = (total / 4).max(1 << 16);
        loop {
            let (count, out) =
                self.dispatch_visible(cam, reach, graze, blocks, zw, zh, &grid, places)?;
            if count as u64 > places {
                places = count as u64;
                continue;
            }
            // in block order (the threads wrote them in any order): sorted by the block order
            // with the place written, a radix sort of 11 bits at a time
            let mut keys: Vec<u64> = out
                .as_chunks::<5>()
                .0
                .iter()
                .enumerate()
                .map(|(k, w)| ((w[0] as u64) << 32) | k as u64)
                .collect();
            radix_sort_high(&mut keys);
            let f = f32::from_bits;
            return Ok(keys
                .iter()
                .map(|&x| {
                    let w = &out[5 * (x as u32 as usize)..];
                    (w[0], f(w[1]), f(w[2]), f(w[3]), f(w[4]))
                })
                .collect());
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
    ) -> Result<(u32, Vec<u32>)> {
        let dev = &self.gpu.device;
        let count = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("visible count"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let out = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visible out"),
            size: places * 20,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
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
                out.as_entire_binding(),
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
        let written = (n as u64).min(places);
        let out = if written > 0 {
            self.read(&out, written * 20)?
        } else {
            Vec::new()
        };
        drop(kept);
        Ok((n, out))
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
        self.gpu.queue.submit([enc.finish()]);
        let slice = back.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::PollType::wait_indefinitely())
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
        self.gpu.queue.submit([enc.finish()]);
        let slice = back.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::PollType::wait_indefinitely())
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
    fn radix_sort_orders_by_the_high_bits() {
        let mut seed = 7u64;
        let mut keys: Vec<u64> = (0..100_000u64)
            .map(|k| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 37) << 32) | k
            })
            .collect();
        let mut want = keys.clone();
        want.sort_by_key(|k| k >> 32);
        super::radix_sort_high(&mut keys);
        assert_eq!(keys, want);
    }
}
