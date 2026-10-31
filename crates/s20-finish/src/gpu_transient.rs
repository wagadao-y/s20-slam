//! The passing-object cone test on the GPU (see transient.rs and transient.wgsl). Single
//! precision: a ray at the cone's edge may fall on the other side.
use crate::gpu::Gpu;
use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

/// The settings of the cone test, as transient.wgsl reads them.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub cos_angle: f32,
    pub angle: f32,
    pub far: f32,
    pub hit: f32,
    pub max_range: f32,
    pub hit_span: f32,
    pub rows: u32,
    pub cols: u32,
    pub max_hits: u32,
    pub ranges: u32,
    pub pairs: u32,
    pub width: u32,
}

/// A scan of a batch: its sensor position, its number (for the times), its first ray and its
/// first cell start in the batch's arrays.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Scan {
    pub origin: [f32; 4],
    pub index: u32,
    pub rays: u32,
    pub starts: u32,
    pub pad: u32,
}

/// The scans of one dispatch: their rays sorted by cell (`starts`: rows × cols + 1 per scan, a
/// ray's length minus the reach for a shot without a return), and the ranges of points near
/// each (first in the cell order, count, scan in the batch, first pair).
#[derive(Default)]
pub struct Batch {
    pub scans: Vec<Scan>,
    pub starts: Vec<u32>,
    pub rays: Vec<[f32; 4]>,
    pub ranges: Vec<[u32; 4]>,
    pub pairs: u32,
}

fn map_err(e: impl std::fmt::Debug) -> anyhow::Error {
    anyhow::anyhow!("GPU map: {e:?}")
}

/// Per point (6 each): the scans with a ray beyond it, those seeing cleanly through, those
/// measuring it, the first and last of them, and whether it is settled. `points`: the tested
/// points; `order`: their indices sorted by the cells the ranges refer to; `times`: per scan
/// (s); `next(batch)` fills the next batch and returns false at the end; `progress()` after each.
pub fn counts(
    gpu: &Gpu,
    points: &[[f32; 3]],
    order: &[u32],
    times: &[f32],
    params: Params,
    next: impl FnMut(&mut Batch) -> bool,
    progress: impl FnMut() -> Result<()>,
) -> Result<Vec<u32>> {
    gpu.checked(|| run(gpu, points, order, times, params, next, progress))
}

fn run(
    gpu: &Gpu,
    points: &[[f32; 3]],
    order: &[u32],
    times: &[f32],
    mut params: Params,
    mut next: impl FnMut(&mut Batch) -> bool,
    mut progress: impl FnMut() -> Result<()>,
) -> Result<Vec<u32>> {
    let dev = &gpu.device;
    let n = points.len().max(1);
    let limit = gpu
        .limits
        .max_storage_buffer_binding_size
        .min(gpu.limits.max_buffer_size);
    let storage = |label, contents: &[u8]| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let p4: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 0.]).collect();
    anyhow::ensure!((n as u64) * 24 <= limit, "too many points for the GPU");
    let pbuf = storage("points", bytemuck::cast_slice(&p4));
    drop(p4);
    let obuf = storage("order", bytemuck::cast_slice(order));
    let tbuf = storage("times", bytemuck::cast_slice(times));
    let init: Vec<u32> = (0..n).flat_map(|_| [0, 0, 0, u32::MAX, 0, 0]).collect();
    let cbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("counts"),
        contents: bytemuck::cast_slice(&init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    drop(init);
    let uni = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cone params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("transient"),
        source: wgpu::ShaderSource::Wgsl(include_str!("transient.wgsl").into()),
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
        label: Some("transient"),
        entries: &[
            entry(0, wgpu::BufferBindingType::Uniform),
            entry(1, ro),
            entry(2, ro),
            entry(3, ro),
            entry(4, ro),
            entry(5, ro),
            entry(6, ro),
            entry(7, ro),
            entry(8, rw),
        ],
    });
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("transient"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cone"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("cone"),
        compilation_options: Default::default(),
        cache: None,
    });
    // the batch buffers, grown when a batch needs more: (scans, starts, rays, ranges) and the
    // bind group over them
    let mut caps = [0u64; 4];
    let mut bufs: Option<([wgpu::Buffer; 4], wgpu::BindGroup)> = None;
    let (mut cur, mut ahead) = (Batch::default(), Batch::default());
    let mut more = next(&mut cur);
    while !cur.scans.is_empty() {
        let sizes = [
            cur.scans.len() as u64 * 32,
            cur.starts.len() as u64 * 4,
            cur.rays.len().max(1) as u64 * 16,
            cur.ranges.len() as u64 * 16,
        ];
        if bufs.is_none() || (0..4).any(|k| sizes[k] > caps[k]) {
            for k in 0..4 {
                caps[k] = caps[k].max(sizes[k].next_power_of_two());
                anyhow::ensure!(
                    caps[k] <= limit,
                    "a batch too large for the GPU ({} bytes)",
                    caps[k]
                );
            }
            let make = |label, size| {
                dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let b = [
                make("scans", caps[0]),
                make("starts", caps[1]),
                make("rays", caps[2]),
                make("ranges", caps[3]),
            ];
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("transient"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uni.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: obuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: tbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: b[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: b[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: b[2].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: b[3].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: cbuf.as_entire_binding(),
                    },
                ],
            });
            bufs = Some((b, bind));
        }
        let (b, bind) = bufs.as_ref().unwrap();
        gpu.queue
            .write_buffer(&b[0], 0, bytemuck::cast_slice(&cur.scans));
        gpu.queue
            .write_buffer(&b[1], 0, bytemuck::cast_slice(&cur.starts));
        if !cur.rays.is_empty() {
            gpu.queue
                .write_buffer(&b[2], 0, bytemuck::cast_slice(&cur.rays));
        }
        gpu.queue
            .write_buffer(&b[3], 0, bytemuck::cast_slice(&cur.ranges));
        let groups = cur.pairs.div_ceil(64);
        let x = groups.clamp(1, 65535);
        params.ranges = cur.ranges.len() as u32;
        params.pairs = cur.pairs;
        params.width = x * 64;
        gpu.queue.write_buffer(&uni, 0, bytemuck::bytes_of(&params));
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("transient"),
        });
        if cur.pairs > 0 {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cone"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(x, groups.div_ceil(x).max(1), 1);
        }
        gpu.queue.submit([enc.finish()]);
        // the next batch while the GPU tests this one
        if more {
            more = next(&mut ahead);
        } else {
            ahead = Batch::default();
        }
        dev.poll(wgpu::PollType::wait_indefinitely())
            .context("GPU poll")?;
        progress()?;
        std::mem::swap(&mut cur, &mut ahead);
    }
    let bytes = n as u64 * 24;
    let back = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("counts back"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("counts"),
    });
    enc.copy_buffer_to_buffer(&cbuf, 0, &back, 0, bytes);
    gpu.queue.submit([enc.finish()]);
    let slice = back.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    dev.poll(wgpu::PollType::wait_indefinitely())
        .context("GPU poll")?;
    rx.recv().context("GPU map")?.context("GPU map")?;
    let out = {
        let view = slice.get_mapped_range().map_err(map_err)?;
        bytemuck::cast_slice::<u8, u32>(&view)[..points.len() * 6].to_vec()
    };
    back.unmap();
    Ok(out)
}
