//! The free-space ray walk on the GPU (see freespace.rs and freewalk.wgsl). Single precision:
//! a step near a voxel's border may fall on the other side.
use crate::gpu::Gpu;
use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

/// The slot of a key in the open-addressing tables (as `slot_hash` in freewalk.wgsl).
fn slot_hash(k: u64) -> u32 {
    let (hi, lo) = ((k >> 32) as u32, k as u32);
    let mut h = hi.wrapping_mul(0x9e37_79b1) ^ lo;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// Open addressing of `keys` (no key has all of its high half set): `stride` u32 per slot, the
/// key's halves and, with stride 3, its index in `keys`. Returns the slots and their mask.
fn table(keys: &[u64], stride: usize) -> (Vec<u32>, u32) {
    let slots = (keys.len() + keys.len() / 2).max(2).next_power_of_two();
    let mask = (slots - 1) as u32;
    let mut t = vec![0u32; slots * stride];
    for s in t.chunks_mut(stride) {
        s[0] = u32::MAX;
    }
    for (n, &k) in keys.iter().enumerate() {
        let mut i = slot_hash(k) & mask;
        while t[i as usize * stride] != u32::MAX {
            i = (i + 1) & mask;
        }
        let s = &mut t[i as usize * stride..(i as usize + 1) * stride];
        s[0] = (k >> 32) as u32;
        s[1] = k as u32;
        if stride == 3 {
            s[2] = n as u32;
        }
    }
    (t, mask)
}

/// The settings of a walk (see `freespace::Options`).
pub struct Walk {
    pub voxel: f64,
    pub margin: f64,
    pub pass_far: f64,
    pub pass_radius: f64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    inv: f32,
    step: f32,
    margin: f32,
    far: f32,
    r2: f32,
    centroid: u32,
    rays: u32,
    vmask: u32,
    bmask: u32,
    voxels: u32,
    pad: [u32; 2],
}

/// The rays of one dispatch at most (well under a second of the GPU at 1 cm voxels).
const WALK_RAYS: usize = 1 << 18;

fn map_err(e: impl std::fmt::Debug) -> anyhow::Error {
    anyhow::anyhow!("GPU map: {e:?}")
}

/// Where a walk's time went: the tables (on the CPU) and the buffers, the batches, the waits for
/// the GPU (the walk and the count of each batch), and the read-back.
#[derive(Debug, Default, Clone, Copy)]
pub struct WalkTimes {
    pub tables: f64,
    pub batches: usize,
    pub rays: usize,
    pub waits: f64,
    pub total: f64,
}

/// The scans whose rays passed through each voxel of `keys` (the measured voxels; `blocks`: the
/// blocks of 8 voxels per axis holding them; `centroids`: per voxel, for `pass_radius`), over
/// `scans` scans whose rays `rays_of(scan, out)` appends as (sensor, 0), (point, 0).
/// `progress(scans done)` after each batch.
#[allow(clippy::too_many_arguments)]
pub fn passes(
    gpu: &Gpu,
    keys: &[u64],
    blocks: &[u64],
    centroids: Option<&[[f32; 3]]>,
    w: &Walk,
    scans: usize,
    rays_of: impl Fn(usize, &mut Vec<[f32; 4]>),
    progress: impl FnMut(usize) -> Result<()>,
) -> Result<(Vec<u32>, WalkTimes)> {
    gpu.checked(|| walk_all(gpu, keys, blocks, centroids, w, scans, rays_of, progress))
}

#[allow(clippy::too_many_arguments)]
fn walk_all(
    gpu: &Gpu,
    keys: &[u64],
    blocks: &[u64],
    centroids: Option<&[[f32; 3]]>,
    w: &Walk,
    scans: usize,
    rays_of: impl Fn(usize, &mut Vec<[f32; 4]>),
    mut progress: impl FnMut(usize) -> Result<()>,
) -> Result<(Vec<u32>, WalkTimes)> {
    let start = std::time::Instant::now();
    let mut times = WalkTimes::default();
    let dev = &gpu.device;
    let (vt, vmask) = table(keys, 3);
    let (bt, bmask) = table(blocks, 2);
    let n = keys.len().max(1);
    let limit = gpu
        .limits
        .max_storage_buffer_binding_size
        .min(gpu.limits.max_buffer_size);
    for bytes in [
        vt.len() as u64 * 4,
        bt.len() as u64 * 4,
        n as u64 * 12,
        WALK_RAYS as u64 * 32,
    ] {
        anyhow::ensure!(
            bytes <= limit,
            "free-space tables too large for the GPU ({bytes} bytes)"
        );
    }
    let storage = |label, contents: &[u8]| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let vbuf = storage("voxels", bytemuck::cast_slice(&vt));
    drop(vt);
    let bbuf = storage("blocks", bytemuck::cast_slice(&bt));
    drop(bt);
    let cbuf = match centroids {
        Some(c) => storage("centroids", bytemuck::cast_slice(c)),
        None => storage("centroids", &[0u8; 16]),
    };
    // (wgpu starts buffers zeroed)
    let zeroed = |label, usage| {
        dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: n as u64 * 4,
            usage,
            mapped_at_creation: false,
        })
    };
    let bits = zeroed("bits", wgpu::BufferUsages::STORAGE);
    let passes = zeroed(
        "passes",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let uni = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("walk params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("freewalk"),
        source: wgpu::ShaderSource::Wgsl(include_str!("freewalk.wgsl").into()),
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
        label: Some("freewalk"),
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
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("freewalk"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = |entry| {
        dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pl),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let (walk, count) = (pipeline("walk"), pipeline("count"));
    times.tables = start.elapsed().as_secs_f64();
    let mut params = Params {
        inv: (1. / w.voxel) as f32,
        step: (0.5 * w.voxel) as f32,
        margin: w.margin as f32,
        far: w.pass_far as f32,
        r2: (w.pass_radius * w.pass_radius) as f32,
        centroid: (w.pass_radius > 0. && centroids.is_some()) as u32,
        rays: 0,
        vmask,
        bmask,
        voxels: keys.len() as u32,
        pad: [0; 2],
    };
    let dispatch = |pass: &mut wgpu::ComputePass, groups: u32| {
        let x = groups.clamp(1, 65535);
        pass.dispatch_workgroups(x, groups.div_ceil(x).max(1), 1);
    };
    // the rays of the next batch: up to 32 scans (one bit each) and about WALK_RAYS rays
    let mut next_scan = 0;
    let mut batch = |out: &mut Vec<[f32; 4]>| -> usize {
        out.clear();
        let first = next_scan;
        while next_scan < scans
            && next_scan - first < 32
            && (next_scan == first || out.len() / 2 < WALK_RAYS)
        {
            let from = out.len();
            rays_of(next_scan, out);
            let bit = f32::from_bits((next_scan - first) as u32);
            for r in out[from..].iter_mut().step_by(2) {
                r[3] = bit;
            }
            next_scan += 1;
        }
        next_scan
    };
    let mut rays: Option<(wgpu::Buffer, wgpu::BindGroup, usize)> = None;
    let (mut cur, mut ahead) = (Vec::new(), Vec::new());
    let mut done = batch(&mut cur);
    while !cur.is_empty() {
        let nr = cur.len() / 2;
        if rays.as_ref().is_none_or(|r| r.2 < nr) {
            let cap = nr.max(WALK_RAYS);
            anyhow::ensure!(
                cap as u64 * 32 <= limit,
                "too many rays in a scan for the GPU"
            );
            let buf = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rays"),
                size: cap as u64 * 32,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("freewalk"),
                layout: &layout,
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
                        resource: vbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: cbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: bits.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: passes.as_entire_binding(),
                    },
                ],
            });
            rays = Some((buf, bind, cap));
        }
        let (buf, bind, _) = rays.as_ref().unwrap();
        params.rays = nr as u32;
        gpu.queue.write_buffer(&uni, 0, bytemuck::bytes_of(&params));
        gpu.queue.write_buffer(buf, 0, bytemuck::cast_slice(&cur));
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("freewalk"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("walk"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&walk);
            pass.set_bind_group(0, bind, &[]);
            dispatch(&mut pass, (nr as u32).div_ceil(64));
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&count);
            pass.set_bind_group(0, bind, &[]);
            dispatch(&mut pass, (keys.len() as u32).div_ceil(256));
        }
        gpu.queue.submit([enc.finish()]);
        // the next batch while the GPU walks this one
        let next = batch(&mut ahead);
        let t = std::time::Instant::now();
        dev.poll(wgpu::PollType::wait_indefinitely())
            .context("GPU poll")?;
        times.waits += t.elapsed().as_secs_f64();
        times.batches += 1;
        times.rays += nr;
        progress(done)?;
        done = next;
        std::mem::swap(&mut cur, &mut ahead);
    }
    let back = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("passes back"),
        size: n as u64 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("passes"),
    });
    enc.copy_buffer_to_buffer(&passes, 0, &back, 0, n as u64 * 4);
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
        bytemuck::cast_slice::<u8, u32>(&view)[..keys.len()].to_vec()
    };
    back.unmap();
    times.total = start.elapsed().as_secs_f64();
    Ok((out, times))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_gpu_error_is_an_error_not_a_panic() {
        let Some(g) = crate::gpu::shared() else {
            return;
        };
        let r = g.checked(|| {
            let _b = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("too large"),
                size: g.limits.max_buffer_size.saturating_add(4) & !3,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            Ok(())
        });
        assert!(r.is_err());
        // and the GPU still works
        assert!(g.checked(|| Ok(())).is_ok());
    }
    /// freewalk.wgsl's walk in single precision: the voxels of the occupied blocks its steps look
    /// up, stepping one by one or skipping the empty blocks as the shader does.
    fn walked(
        a: [f32; 3],
        b: [f32; 3],
        voxel: f32,
        occupied: &impl Fn([i32; 3]) -> bool,
        skip: bool,
    ) -> Vec<[i32; 3]> {
        let (inv, step, margin) = (1. / voxel, 0.5 * voxel, 0.06f32);
        let d = [0, 1, 2].map(|i| b[i] - a[i]);
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let stop = len - margin;
        let u = d.map(|x| x / len);
        let inside_until = |a: f32, b: f32, u: f32, lo: f32, hi: f32| -> f32 {
            if u == 0. {
                return 3.4e38;
            }
            let bound = if u > 0. { hi } else { lo };
            let eps = 1e-6 * (a.abs() + b.abs() + bound.abs());
            (bound - a) / u - eps / u.abs()
        };
        let (mut last, mut empty) = (None, None);
        let mut out = Vec::new();
        let mut k = 1u32;
        loop {
            let s = k as f32 * step;
            // (as the shader: a NaN stops too)
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(s < stop) {
                break;
            }
            k += 1;
            let c = [0, 1, 2].map(|i| ((a[i] + d[i] * (s / len)) * inv).floor() as i32);
            if last == Some(c) {
                continue;
            }
            last = Some(c);
            let bc = c.map(|x| x >> 3);
            if empty == Some(bc) {
                continue;
            }
            if !occupied(bc) {
                empty = Some(bc);
                if skip {
                    let size = 8. / inv;
                    let lo = bc.map(|x| x as f32 * size);
                    let t = (0..3)
                        .map(|i| inside_until(a[i], b[i], u[i], lo[i], lo[i] + size))
                        .fold(f32::INFINITY, f32::min);
                    let j = (t.min(stop) / step).floor() - 1.;
                    if j > k as f32 {
                        k = j as u32;
                    }
                }
                continue;
            }
            out.push(c);
        }
        out
    }

    #[test]
    fn skipping_the_empty_blocks_looks_up_the_same_voxels() {
        // occupied: a pseudo-random third of the blocks
        let occupied = |b: [i32; 3]| {
            let h = (b[0] as u32).wrapping_mul(0x9e37_79b1)
                ^ (b[1] as u32).wrapping_mul(0x85eb_ca6b)
                ^ (b[2] as u32).wrapping_mul(0xc2b2_ae35);
            (h >> 7).is_multiple_of(3)
        };
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f32 / (1u64 << 53) as f32
        };
        // the review's ray: x on a block's border, nearly along it
        let mut rays = vec![([0.159_999_98f32, 0., 0.01], [0.160_000_01f32, 1., 0.01])];
        for _ in 0..3000 {
            let a = [0, 1, 2].map(|_| (rnd() - 0.5) * 200.);
            let mut b = [0, 1, 2].map(|i| a[i] + (rnd() - 0.5) * 60.);
            // a third nearly along an axis's borders: that axis on a block's border at both ends
            if rnd() < 0.33 {
                let i = (rnd() * 3.) as usize % 3;
                let border = (a[i] / 0.16).round() * 0.16;
                b[i] = border + (rnd() - 0.5) * 1e-5;
                let mut a = a;
                a[i] = border + (rnd() - 0.5) * 1e-5;
                rays.push((a, b));
                continue;
            }
            rays.push((a, b));
        }
        for (a, b) in rays {
            for voxel in [0.02f32, 0.01] {
                assert_eq!(
                    walked(a, b, voxel, &occupied, true),
                    walked(a, b, voxel, &occupied, false),
                    "{a:?} -> {b:?}, voxel {voxel}"
                );
            }
        }
    }

    #[test]
    fn every_key_is_found_in_its_table() {
        let keys: Vec<u64> = (0..1000u64)
            .map(|i| i.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 1)
            .collect();
        let (t, mask) = table(&keys, 3);
        for (n, &k) in keys.iter().enumerate() {
            let mut i = slot_hash(k) & mask;
            while t[i as usize * 3] != (k >> 32) as u32 || t[i as usize * 3 + 1] != k as u32 {
                assert_ne!(t[i as usize * 3], u32::MAX);
                i = (i + 1) & mask;
            }
            assert_eq!(t[i as usize * 3 + 2], n as u32);
        }
    }
}
