//! [`super::detect_with`] on the GPU (`sift.wgsl`): the pyramid, the extrema, their orientations
//! and the descriptors there; the duplicates and the strongest chosen on the CPU as there
//! ([`super::select`]). One image at a time (the threads wait for the GPU in turn). Single
//! precision as on the CPU, but the GPU may fuse multiplications and additions and its `exp`
//! differs in the last bits: a keypoint at a threshold may come or go.

use super::{Keypoint, LAYERS, SIGMA, select};
use crate::image::Gray;
use anyhow::{Context, Result, ensure};
use s20_gpu::Gpu;
use std::num::NonZeroU64;
use std::sync::{Mutex, OnceLock};

/// Room for the candidates and their orientations of one image, and for the keypoints kept.
const MAX_CANDS: u32 = 1 << 18;
const MAX_KEPT: u32 = 1 << 16;
/// Uniform slots of one image's passes.
const SLOTS: usize = 512;
const SLOT: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    src_off: u32,
    dst_off: u32,
    w: u32,
    h: u32,
    src_w: u32,
    src_h: u32,
    k_off: u32,
    half: u32,
    dog_off: u32,
    octave: u32,
    layer: u32,
    threshold: f32,
    contrast: f32,
    base: f32,
    max_out: u32,
    count: u32,
    g_off: [u32; 16],
    g_w: [u32; 16],
    g_h: [u32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Cand {
    x: f32,
    y: f32,
    size: f32,
    response: f32,
    ox: f32,
    oy: f32,
    ix: i32,
    iy: i32,
    octave: u32,
    layer: u32,
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Ori {
    cand: u32,
    angle: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Kept {
    octave: u32,
    layer: u32,
    ox: f32,
    oy: f32,
    size: f32,
    angle: f32,
    pad: [u32; 2],
}

/// The blur kernels as the CPU's `blur` makes them: for each sigma, its half width and taps.
fn kernel(sigma: f32) -> Vec<f32> {
    let half = ((sigma * 8. + 1.).round() as usize | 1) / 2;
    let mut k: Vec<f32> = (0..=2 * half)
        .map(|i| (-((i as f32 - half as f32).powi(2)) / (2. * sigma * sigma)).exp())
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// The sigmas of the pyramid: the base's (doubled, as it is), then those between the layers.
fn sigmas() -> Vec<f32> {
    let init = super::INIT_SIGMA;
    let k = 2f32.powf(1. / LAYERS as f32);
    let mut v = vec![
        (SIGMA * SIGMA - (2. * init) * (2. * init)).max(0.01).sqrt(),
        (SIGMA * SIGMA - init * init).max(0.01).sqrt(),
    ];
    for i in 1..LAYERS + 3 {
        let prev = k.powi(i as i32 - 1) * SIGMA;
        v.push(((prev * k).powi(2) - prev * prev).sqrt());
    }
    v
}

/// The buffers for one size of image.
struct Sized {
    key: (usize, usize, bool),
    gauss: wgpu::Buffer,
    _dog: wgpu::Buffer,
    cands: wgpu::Buffer,
    counters: wgpu::Buffer,
    oris: wgpu::Buffer,
    kept: wgpu::Buffer,
    desc: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
    /// the octaves' sizes and offsets: (w, h, gauss offset, dog offset)
    octaves: Vec<(usize, usize, usize, usize)>,
    scratch: usize,
    src: usize,
}

pub(super) struct GpuSift {
    gpu: &'static Gpu,
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 9],
    kernels: wgpu::Buffer,
    /// each sigma's (offset into the kernels, half width)
    taps: Vec<(u32, u32)>,
    sized: Mutex<Option<Sized>>,
}

const DOUBLE: usize = 0;
const COPY: usize = 1;
const BLUR_H: usize = 2;
const BLUR_V: usize = 3;
const HALVE: usize = 4;
const DIFFERENCE: usize = 5;
const EXTREMA: usize = 6;
const ORIENT: usize = 7;
const DESCRIBE: usize = 8;

/// The process's GPU SIFT, when there is a GPU and `S20_NO_GPU` is not set (none otherwise, or
/// when its shaders do not build).
pub(super) fn shared() -> Option<&'static GpuSift> {
    static SIFT: OnceLock<Option<GpuSift>> = OnceLock::new();
    SIFT.get_or_init(|| {
        if std::env::var_os("S20_NO_GPU").is_some() {
            return None;
        }
        let gpu = s20_gpu::shared()?;
        match gpu.checked(|| GpuSift::new(gpu)) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("SIFT on the CPU (GPU: {e:#})");
                None
            }
        }
    })
    .as_ref()
}

/// The GPU's name, when SIFT runs there.
pub fn name() -> Option<&'static str> {
    shared().map(|s| s.gpu.name.as_str())
}

impl GpuSift {
    fn new(gpu: &'static Gpu) -> Result<Self> {
        ensure!(
            gpu.limits.max_storage_buffers_per_shader_stage >= 8,
            "too few storage buffers"
        );
        let dev = &gpu.device;
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sift"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sift.wgsl").into()),
        });
        let buffer = |binding, ty, dynamic| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sift"),
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform, true),
                buffer(1, rw, false),
                buffer(2, rw, false),
                buffer(3, ro, false),
                buffer(4, rw, false),
                buffer(5, rw, false),
                buffer(6, rw, false),
                buffer(7, ro, false),
                buffer(8, rw, false),
            ],
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sift"),
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
        let pipelines = [
            "double_img",
            "copy_img",
            "blur_h",
            "blur_v",
            "halve",
            "difference",
            "extrema",
            "orient",
            "describe",
        ]
        .map(pipeline);
        let mut all = Vec::new();
        let mut taps = Vec::new();
        for s in sigmas() {
            let k = kernel(s);
            taps.push((all.len() as u32, (k.len() / 2) as u32));
            all.extend(k);
        }
        let kernels = storage(gpu, (all.len() * 4) as u64, "sift kernels");
        gpu.queue
            .write_buffer(&kernels, 0, bytemuck::cast_slice(&all));
        Ok(GpuSift {
            gpu,
            layout,
            pipelines,
            kernels,
            taps,
            sized: Mutex::new(None),
        })
    }

    fn sized(&self, w: usize, h: usize, double_first: bool) -> Sized {
        let gpu = self.gpu;
        let (bw, bh) = if double_first { (w * 2, h * 2) } else { (w, h) };
        let count = ((bw.min(bh) as f32).log2().round() as isize - 2).max(1) as usize;
        let mut octaves = Vec::with_capacity(count);
        let (mut ow, mut oh, mut g, mut d) = (bw, bh, 0usize, 0usize);
        for o in 0..count {
            if o > 0 {
                (ow, oh) = (ow / 2, oh / 2);
            }
            octaves.push((ow, oh, g, d));
            g += (LAYERS + 3) * ow * oh;
            d += (LAYERS + 2) * ow * oh;
        }
        let scratch = g;
        let src = scratch + bw * bh;
        let gauss = storage(gpu, ((src + w * h) * 4) as u64, "sift gauss");
        let dog = storage(gpu, (d.max(1) * 4) as u64, "sift dog");
        let cands = storage(gpu, MAX_CANDS as u64 * 48, "sift candidates");
        let counters = storage(gpu, 16, "sift counters");
        let oris = storage(gpu, MAX_CANDS as u64 * 8, "sift orientations");
        let kept = storage(gpu, MAX_KEPT as u64 * 32, "sift kept");
        let desc = storage(gpu, MAX_KEPT as u64 * 128, "sift descriptors");
        let slot = SLOT.max(gpu.limits.min_uniform_buffer_offset_alignment as u64);
        let uniforms = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sift params"),
            size: slot * SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        fn whole(b: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
            b.as_entire_binding()
        }
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sift"),
            layout: &self.layout,
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
                    resource: whole(&gauss),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: whole(&dog),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: whole(&self.kernels),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: whole(&cands),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: whole(&counters),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: whole(&oris),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: whole(&kept),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: whole(&desc),
                },
            ],
        });
        Sized {
            key: (w, h, double_first),
            gauss,
            _dog: dog,
            cands,
            counters,
            oris,
            kept,
            desc,
            uniforms,
            bind,
            octaves,
            scratch,
            src,
        }
    }

    /// Keypoints and descriptors as [`super::detect_with`].
    pub(super) fn detect(
        &self,
        img: &Gray,
        max_features: usize,
        contrast: f32,
        double_first: bool,
    ) -> Result<(Vec<Keypoint>, Vec<[u8; 128]>)> {
        let mut guard = self.sized.lock().unwrap();
        self.gpu.checked(|| {
            if guard
                .as_ref()
                .is_none_or(|s| s.key != (img.w, img.h, double_first))
            {
                *guard = Some(self.sized(img.w, img.h, double_first));
            }
            self.run(
                guard.as_ref().unwrap(),
                img,
                max_features,
                contrast,
                double_first,
            )
        })
    }

    fn run(
        &self,
        s: &Sized,
        img: &Gray,
        max_features: usize,
        contrast: f32,
        double_first: bool,
    ) -> Result<(Vec<Keypoint>, Vec<[u8; 128]>)> {
        let gpu = self.gpu;
        let base = if double_first { 0.5 } else { 1. };
        let threshold = (0.5 * contrast / LAYERS as f32 * 255.).floor();
        ensure!(s.octaves.len() <= 16, "too many octaves");
        let mut octs = Params::default();
        for (o, &(w, h, g, _)) in s.octaves.iter().enumerate() {
            octs.g_off[o] = g as u32;
            octs.g_w[o] = w as u32;
            octs.g_h[o] = h as u32;
        }
        octs.base = base;
        // 1. the pyramid and the extrema
        let mut params: Vec<Params> = Vec::new();
        let mut steps: Vec<(usize, usize, u32, u32)> = Vec::new();
        let mut pass = |p: Params, pipeline: usize, gx: u32, gy: u32| {
            steps.push((pipeline, params.len(), gx, gy));
            params.push(p);
        };
        let groups = |w: usize, h: usize| ((w as u32).div_ceil(16), (h as u32).div_ceil(16));
        let image = |src: usize, dst: usize, w: usize, h: usize| Params {
            src_off: src as u32,
            dst_off: dst as u32,
            w: w as u32,
            h: h as u32,
            ..octs
        };
        let blur = |pass: &mut dyn FnMut(Params, usize, u32, u32),
                    from: usize,
                    to: usize,
                    w: usize,
                    h: usize,
                    sigma: usize| {
            let (k_off, half) = self.taps[sigma];
            let (gx, gy) = groups(w, h);
            pass(
                Params {
                    k_off,
                    half,
                    ..image(from, s.scratch, w, h)
                },
                BLUR_H,
                gx,
                gy,
            );
            pass(
                Params {
                    k_off,
                    half,
                    ..image(s.scratch, to, w, h)
                },
                BLUR_V,
                gx,
                gy,
            );
        };
        let (w0, h0, g0, _) = s.octaves[0];
        let (gx, gy) = groups(w0, h0);
        if double_first {
            pass(
                Params {
                    src_w: img.w as u32,
                    src_h: img.h as u32,
                    ..image(s.src, g0, w0, h0)
                },
                DOUBLE,
                gx,
                gy,
            );
            blur(&mut pass, g0, g0, w0, h0, 0);
        } else {
            pass(image(s.src, g0, w0, h0), COPY, gx, gy);
            blur(&mut pass, g0, g0, w0, h0, 1);
        }
        for (o, &(w, h, g, d)) in s.octaves.iter().enumerate() {
            let n = w * h;
            let (gx, gy) = groups(w, h);
            if o > 0 {
                let (pw, _, pg, _) = s.octaves[o - 1];
                pass(
                    Params {
                        src_w: pw as u32,
                        ..image(pg + LAYERS * pw * s.octaves[o - 1].1, g, w, h)
                    },
                    HALVE,
                    gx,
                    gy,
                );
            }
            for i in 1..LAYERS + 3 {
                blur(&mut pass, g + (i - 1) * n, g + i * n, w, h, i + 1);
            }
            for i in 0..LAYERS + 2 {
                pass(image(g + i * n, d + i * n, w, h), DIFFERENCE, gx, gy);
            }
        }
        for (o, &(w, h, _, d)) in s.octaves.iter().enumerate() {
            let (gx, gy) = groups(w, h);
            for l in 1..=LAYERS {
                pass(
                    Params {
                        w: w as u32,
                        h: h as u32,
                        dog_off: d as u32,
                        octave: o as u32,
                        layer: l as u32,
                        threshold,
                        contrast,
                        max_out: MAX_CANDS,
                        ..octs
                    },
                    EXTREMA,
                    gx,
                    gy,
                );
            }
        }
        ensure!(params.len() + 2 <= SLOTS, "too many passes");
        gpu.queue
            .write_buffer(&s.gauss, (s.src * 4) as u64, bytemuck::cast_slice(&img.px));
        gpu.queue
            .write_buffer(&s.counters, 0, bytemuck::cast_slice(&[0u32; 4]));
        self.write_params(s, &params);
        self.dispatch(s, &steps);
        let counts = read::<u32>(gpu, &s.counters, 16)?;
        ensure!(
            counts[0] <= MAX_CANDS,
            "too many candidates ({})",
            counts[0]
        );
        let n = counts[0];
        // 2. their orientations
        let p = Params {
            count: n,
            max_out: MAX_CANDS,
            ..octs
        };
        self.write_params(s, &[p]);
        self.dispatch(s, &[(ORIENT, 0, n.div_ceil(64).max(1), 1)]);
        let counts = read::<u32>(gpu, &s.counters, 16)?;
        ensure!(
            counts[1] <= MAX_CANDS,
            "too many orientations ({})",
            counts[1]
        );
        let cands: Vec<Cand> = read(gpu, &s.cands, n as u64 * 48)?;
        let oris: Vec<Ori> = read(gpu, &s.oris, counts[1] as u64 * 8)?;
        let kps: Vec<Keypoint> = oris
            .iter()
            .map(|o| {
                let c = &cands[o.cand as usize];
                Keypoint {
                    x: c.x,
                    y: c.y,
                    size: c.size,
                    angle: o.angle,
                    response: c.response,
                    octave: c.octave as usize,
                    layer: c.layer as usize,
                    ox: c.ox,
                    oy: c.oy,
                    ix: c.ix as isize,
                    iy: c.iy as isize,
                    base,
                }
            })
            .collect();
        // 3. the strongest, described
        let kps = select(kps, max_features);
        ensure!(kps.len() <= MAX_KEPT as usize, "too many keypoints kept");
        if kps.is_empty() {
            return Ok((kps, Vec::new()));
        }
        let kept: Vec<Kept> = kps
            .iter()
            .map(|k| Kept {
                octave: k.octave as u32,
                layer: k.layer as u32,
                ox: k.ox,
                oy: k.oy,
                size: k.size,
                angle: k.angle,
                pad: [0; 2],
            })
            .collect();
        gpu.queue
            .write_buffer(&s.kept, 0, bytemuck::cast_slice(&kept));
        let p = Params {
            count: kps.len() as u32,
            ..octs
        };
        self.write_params(s, &[p]);
        self.dispatch(s, &[(DESCRIBE, 0, (kps.len() as u32).div_ceil(64), 1)]);
        let words: Vec<u32> = read(gpu, &s.desc, kps.len() as u64 * 128)?;
        let desc = words
            .as_chunks::<32>()
            .0
            .iter()
            .map(|w| {
                let mut d = [0u8; 128];
                for (k, word) in w.iter().enumerate() {
                    d[4 * k..4 * k + 4].copy_from_slice(&word.to_le_bytes());
                }
                d
            })
            .collect();
        Ok((kps, desc))
    }

    fn write_params(&self, s: &Sized, params: &[Params]) {
        let slot = SLOT.max(self.gpu.limits.min_uniform_buffer_offset_alignment as u64) as usize;
        let mut bytes = vec![0u8; params.len() * slot];
        for (k, p) in params.iter().enumerate() {
            bytes[k * slot..k * slot + SLOT as usize].copy_from_slice(bytemuck::bytes_of(p));
        }
        self.gpu.queue.write_buffer(&s.uniforms, 0, &bytes);
    }

    /// The passes (pipeline, uniform slot, workgroups) in order.
    fn dispatch(&self, s: &Sized, steps: &[(usize, usize, u32, u32)]) {
        let slot = SLOT.max(self.gpu.limits.min_uniform_buffer_offset_alignment as u64);
        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sift"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sift"),
                timestamp_writes: None,
            });
            for &(pipeline, k, gx, gy) in steps {
                pass.set_pipeline(&self.pipelines[pipeline]);
                pass.set_bind_group(0, &s.bind, &[(k as u64 * slot) as u32]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        self.gpu.queue.submit([enc.finish()]);
    }
}

fn storage(gpu: &Gpu, size: u64, label: &str) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// The first `bytes` of `buf`, as `T`s.
fn read<T: bytemuck::Pod>(gpu: &Gpu, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<T>> {
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let dev = &gpu.device;
    let back = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sift read back"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("sift read"),
    });
    enc.copy_buffer_to_buffer(buf, 0, &back, 0, bytes);
    gpu.queue.submit([enc.finish()]);
    let slice = back.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    dev.poll(wgpu::PollType::wait_indefinitely())
        .context("GPU poll")?;
    rx.recv().context("GPU map")?.context("GPU map")?;
    let v = {
        let view = slice
            .get_mapped_range()
            .map_err(|e| anyhow::anyhow!("GPU map: {e:?}"))?;
        bytemuck::cast_slice::<u8, T>(&view).to_vec()
    };
    back.unmap();
    Ok(v)
}
