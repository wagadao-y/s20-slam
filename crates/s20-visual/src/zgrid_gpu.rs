//! [`crate::anchors::zgrid`] for many images on the GPU (`zgrid.wgsl`), the cloud sent once. In
//! single precision (the points about their centre): a point at a cell's border, or as near as
//! another, may go the other way than on the CPU.

use crate::anchors::{CELL, Cloud, HEIGHT, Pose, WIDTH, ZGrid, zgrid};
use crate::camera::Fisheye;
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use s20_gpu::Gpu;
use std::num::NonZeroU64;

/// Images a submission.
const BATCH: usize = 16;
const SLOT: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    r: [[f32; 4]; 3],
    t: [f32; 4],
    k03: [f32; 4],
    k45: [f32; 4],
    a: [f32; 4],
    b: [f32; 4],
    d: [u32; 4],
}

/// The z-buffers of `poses` (each with its camera): on the GPU when there is one (and
/// `S20_NO_GPU` is not set), else, or when it fails, on the CPU; and where they were made.
pub fn zgrids(cloud: &Cloud, poses: &[(Pose, &Fisheye)]) -> (Vec<ZGrid>, &'static str) {
    if std::env::var_os("S20_NO_GPU").is_none()
        && let Some(gpu) = s20_gpu::shared()
    {
        match gpu.checked(|| on_gpu(gpu, cloud, poses)) {
            Ok(g) => return (g, "GPU"),
            Err(e) => eprintln!("z-buffers on the CPU (GPU: {e:#})"),
        }
    }
    let g = poses
        .par_iter()
        .map(|(pose, cam)| zgrid(cloud, pose, cam))
        .collect();
    (g, "CPU")
}

fn on_gpu(gpu: &Gpu, cloud: &Cloud, poses: &[(Pose, &Fisheye)]) -> Result<Vec<ZGrid>> {
    let n = cloud.xyz.len();
    ensure!(n > 0 && n < u32::MAX as usize, "no points or too many");
    let (w, h) = (WIDTH.div_ceil(CELL), HEIGHT.div_ceil(CELL));
    let cells = w * h;
    // the points about their centre, in single precision
    let mut centre = [0f64; 3];
    for p in &cloud.xyz {
        for d in 0..3 {
            centre[d] += p[d];
        }
    }
    centre = centre.map(|c| c / n as f64);
    let pts: Vec<f32> = cloud
        .xyz
        .iter()
        .flat_map(|p| [0, 1, 2].map(|d| (p[d] - centre[d]) as f32))
        .collect();
    let dev = &gpu.device;
    let buffer = |size: u64, usage: wgpu::BufferUsages, label: &str| {
        dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    use wgpu::BufferUsages as U;
    let points = buffer(
        (pts.len() * 4) as u64,
        U::STORAGE | U::COPY_DST,
        "zgrid points",
    );
    gpu.queue
        .write_buffer(&points, 0, bytemuck::cast_slice(&pts));
    drop(pts);
    let grid_bytes = (cells * 4) as u64;
    // each image's grid at an aligned offset
    let stride = grid_bytes.next_multiple_of(gpu.limits.min_storage_buffer_offset_alignment as u64);
    let zmin = buffer(
        stride * BATCH as u64,
        U::STORAGE | U::COPY_DST,
        "zgrid ranges",
    );
    let zidx = buffer(
        stride * BATCH as u64,
        U::STORAGE | U::COPY_DST | U::COPY_SRC,
        "zgrid indices",
    );
    let back = buffer(
        stride * BATCH as u64,
        U::MAP_READ | U::COPY_DST,
        "zgrid read back",
    );
    let slot = SLOT.max(gpu.limits.min_uniform_buffer_offset_alignment as u64);
    let uniforms = buffer(
        slot * BATCH as u64,
        U::UNIFORM | U::COPY_DST,
        "zgrid params",
    );
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("zgrid"),
        source: wgpu::ShaderSource::Wgsl(include_str!("zgrid.wgsl").into()),
    });
    let entry = |binding, ty, dynamic, size: Option<u64>| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: dynamic,
            min_binding_size: size.and_then(NonZeroU64::new),
        },
        count: None,
    };
    let rw = wgpu::BufferBindingType::Storage { read_only: false };
    let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("zgrid"),
        entries: &[
            entry(0, wgpu::BufferBindingType::Uniform, true, None),
            entry(
                1,
                wgpu::BufferBindingType::Storage { read_only: true },
                false,
                None,
            ),
            entry(2, rw, true, None),
            entry(3, rw, true, None),
        ],
    });
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("zgrid"),
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
    let (nearest, index) = (pipeline("nearest_pass"), pipeline("index_pass"));
    let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zgrid"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniforms,
                    offset: 0,
                    size: NonZeroU64::new(SLOT),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: points.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &zmin,
                    offset: 0,
                    size: NonZeroU64::new(grid_bytes),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &zidx,
                    offset: 0,
                    size: NonZeroU64::new(grid_bytes),
                }),
            },
        ],
    });
    let groups = (n as u32).div_ceil(64);
    let (gx, gy) = (groups.min(65535), groups.div_ceil(65535));
    let mut out = Vec::with_capacity(poses.len());
    for batch in poses.chunks(BATCH) {
        let mut bytes = vec![0u8; slot as usize * batch.len()];
        for (k, (pose, cam)) in batch.iter().enumerate() {
            let rt = pose.r.transpose();
            let row = |i: usize| [rt[(i, 0)] as f32, rt[(i, 1)] as f32, rt[(i, 2)] as f32, 0.];
            let p = Params {
                r: [row(0), row(1), row(2)],
                t: [
                    (pose.p.x - centre[0]) as f32,
                    (pose.p.y - centre[1]) as f32,
                    (pose.p.z - centre[2]) as f32,
                    0.,
                ],
                k03: [cam.k[0], cam.k[1], cam.k[2], cam.k[3]].map(|v| v as f32),
                k45: [cam.k[4] as f32, cam.k[5] as f32, 0., 0.],
                a: [cam.a11, cam.a12, cam.a22, cam.u0].map(|v| v as f32),
                b: [
                    cam.v0 as f32,
                    WIDTH as f32,
                    HEIGHT as f32,
                    82f32.to_radians(),
                ],
                d: [w as u32, n as u32, CELL as u32, 0],
            };
            let at = k * slot as usize;
            let p = bytemuck::bytes_of(&p);
            bytes[at..at + p.len()].copy_from_slice(p);
        }
        gpu.queue.write_buffer(&uniforms, 0, &bytes);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("zgrid"),
        });
        let used = stride * batch.len() as u64;
        enc.clear_buffer(&zmin, 0, Some(used));
        enc.clear_buffer(&zidx, 0, Some(used));
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zgrid"),
                timestamp_writes: None,
            });
            for k in 0..batch.len() {
                let offsets = [
                    (k as u64 * slot) as u32,
                    (k as u64 * stride) as u32,
                    (k as u64 * stride) as u32,
                ];
                pass.set_bind_group(0, &bind, &offsets);
                pass.set_pipeline(&nearest);
                pass.dispatch_workgroups(gx, gy, 1);
                pass.set_pipeline(&index);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        enc.copy_buffer_to_buffer(&zidx, 0, &back, 0, used);
        let submitted = gpu.queue.submit([enc.finish()]);
        let slice = back.slice(..used);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::PollType::Wait {
            submission_index: Some(submitted),
            timeout: None,
        })
        .context("GPU poll")?;
        rx.recv().context("GPU map")?.context("GPU map")?;
        {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("GPU map: {e:?}"))?;
            let words: &[u32] = bytemuck::cast_slice(&view);
            let step = (stride / 4) as usize;
            for k in 0..batch.len() {
                let idx: Vec<i32> = words[k * step..k * step + cells]
                    .iter()
                    .map(|&v| if v == 0 { -1 } else { (!v) as i32 })
                    .collect();
                out.push(ZGrid::from_cells(w, h, idx));
            }
        }
        back.unmap();
    }
    Ok(out)
}
