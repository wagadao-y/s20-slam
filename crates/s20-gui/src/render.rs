//! Point and line renderer with eye-dome lighting, adapted from Geemil Workbench.
use crate::camera::Camera;
use crate::scene::{NodePoints, ViewPoint};
use eframe::{egui, egui_wgpu::wgpu};
use glam::{DMat4, DVec3};
use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};
use wgpu::util::DeviceExt;

// The point pass writes linear RGB to an sRGB target, so the hardware encodes
// it and the EDL pass reads linear RGB back with textureLoad.
const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
// egui-wgpu (0.36) samples native textures as plain Rgba8Unorm holding sRGB
// code values, so edl.wgsl encodes its linear result before writing here.
const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Points uploaded per frame at most (8 MB of vertices); the rest follow in
/// later frames so a large view does not stall one.
const UPLOAD_PER_FRAME: usize = 500_000;
/// Per-pixel view depth for EDL; 0 marks background.
const VIEW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Per pixel, the frontmost point's segment plus one and its index, for
/// picking; 0 marks background.
const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
/// Picks look this many physical pixels around the pointer at most.
const MAX_PICK_REACH: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    matrix: [[f32; 4]; 4],
    viewport: [f32; 2],
    /// Splat diameter in physical pixels, or with `adaptive` above zero the
    /// smallest one.
    size: f32,
    /// Potree's adaptive size factor; 0 draws every splat `size` wide.
    adaptive: f32,
    tint: [f32; 4],
    /// View depth of a vertex for EDL as `dot(depth, (position, 1))`.
    depth: [f32; 4],
    /// Maps a vertex into box coordinates; points inside `[-1, 1]` are
    /// highlighted when `box_highlight_on` is 1.
    highlight_box: [[f32; 4]; 4],
    box_highlight_on: f32,
    /// How the points are coloured: `Coloring::code`.
    color_mode: f32,
    /// Physical pixels a metre spans at a clip-space w of 1.
    pixels_per_metre: f32,
    /// The draw's place in the frame, written to the id target.
    segment: u32,
    /// Height scaled to 0..1 over the ramp as `dot(ramp, (position, 1))`.
    ramp: [f32; 4],
    /// Spacing of the drawn points for adaptive size when positive (else the per-point spacings).
    node_spacing: f32,
    max_size: f32,
    /// Points whose height is outside this range are not drawn.
    clip: [f32; 2],
    /// Height (project Z) as `dot(height, (position, 1))`.
    height: [f32; 4],
}
/// A line drawn behind points in front of it, `width` physical pixels wide.
#[derive(Clone, Copy, PartialEq)]
pub struct Line {
    pub a: DVec3,
    pub b: DVec3,
    /// sRGB code values.
    pub color: [u8; 4],
    pub width: f32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    /// Relative to the camera target.
    a: [f32; 3],
    b: [f32; 3],
    color: [u8; 4],
    width: f32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineUniform {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    viewport: [f32; 2],
    pixels_per_metre: f32,
    ortho: f32,
    depth_offset: f32,
    padding: [f32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EdlUniform {
    radius: f32,
    strength: f32,
    padding: [f32; 2],
}
/// Eye-Dome Lighting. A strength of 0 shows the plain colours.
#[derive(Clone, Copy, PartialEq)]
pub struct Edl {
    /// Neighbour distance in physical pixels.
    pub radius: f32,
    pub strength: f32,
}
/// Size-dependent render targets, recreated when the viewport resizes.
struct Targets {
    size: [u32; 2],
    scene: wgpu::TextureView,
    view_depth: wgpu::TextureView,
    ids: wgpu::Texture,
    ids_view: wgpu::TextureView,
    depth: wgpu::TextureView,
    output: wgpu::TextureView,
    edl_bind: wgpu::BindGroup,
}
/// How the points are coloured. The points carry an sRGB colour and, in its alpha, their grey
/// by reflectivity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coloring {
    /// their own colour
    Rgb,
    /// grey by reflectivity
    Reflectivity,
    /// by height from the first to the second value (project Z)
    Height([f64; 2]),
    /// each node's tint (its session), shaded by reflectivity
    Tint,
    /// one grey
    Plain,
}

impl Coloring {
    /// The mode as points.wgsl reads it.
    fn code(self) -> f32 {
        match self {
            Self::Rgb => 0.0,
            Self::Reflectivity => 1.0,
            Self::Height(_) => 2.0,
            Self::Tint => 3.0,
            Self::Plain => 4.0,
        }
    }
}

/// How to draw the uploaded points.
pub struct DrawOptions<'a> {
    /// Splat diameter in physical pixels.
    pub point_size: f32,
    /// Potree's adaptive point size with this size factor: each point is as
    /// wide as 1.7 times its spacing (see `DrawNode::spacings`), but at least
    /// `min_size` physical pixels.
    pub adaptive: Option<f32>,
    pub min_size: f32,
    pub max_size: f32,
    pub edl: Edl,
    /// Maps the project frame into a box whose interior points are highlighted.
    pub highlight_box: Option<DMat4>,
    pub coloring: Coloring,
    /// Points below the first or above the second height (project Z) are not drawn; infinite
    /// for no limit.
    pub height_clip: [f64; 2],
    pub nodes: &'a [DrawNode<'a>],
    /// Lines that points in front of them hide, in the project frame.
    pub lines: &'a Arc<[Line]>,
    /// Draw the lines over the points instead of behind nearer ones.
    pub lines_on_top: bool,
}
/// A display octree node to draw: its points in scan coordinates, placed in
/// the project frame by `world`. Its view points are uploaded as they are:
/// an f32 offset from the node's origin and sRGB code values, which
/// points.wgsl decodes before writing to SCENE_FORMAT (alpha stays a linear
/// coverage value).
pub struct DrawNode<'a> {
    /// The caller's name for the node, which picks return.
    pub id: usize,
    pub points: &'a Arc<NodePoints>,
    pub world: DMat4,
    /// sRGB colour (0..1) mixed into the points, by the fourth component.
    pub tint: [f32; 4],
    /// Points to highlight, e.g. those a move would take; none when no point
    /// is. New marks come in a new array.
    pub marks: Option<&'a Arc<[bool]>>,
    /// Each point's spacing in metres, for adaptive point size.
    pub spacings: Option<&'a Arc<[f32]>>,
    /// Draw only the first `limit` points (all of them are uploaded).
    pub limit: u32,
    /// Spacing of the drawn points in metres for adaptive size; 0 uses `spacings`.
    pub spacing: f32,
}
/// GPU bookkeeping holds a weak reference: the identity allocation stays unique, while evicting
/// CPU points releases their payload immediately.
struct GpuNode {
    _points: Weak<NodePoints>,
    origin: DVec3,
    vertices: wgpu::Buffer,
    /// One u32 per point; bit 0 highlights it. None draws `zeros` instead,
    /// while no point is highlighted.
    flags: Option<wgpu::Buffer>,
    /// One f32 per point: its spacing for adaptive point size. None draws
    /// `zeros` instead, before any spacings are given.
    spacings: Option<wgpu::Buffer>,
    /// The spacings written, kept so their address identifies them.
    spacings_written: Option<Arc<[f32]>>,
    /// The marks written, likewise.
    marks_written: Option<Arc<[bool]>>,
    count: u32,
    used: u64,
}
/// Per node: samples, marks and spacings by address, world, tint, drawn points and spacing.
type ShownNode = ([usize; 3], DMat4, [f32; 4], u32, f32);
/// Retains the immutable lines, so pointer equality is enough to detect a changed source.
struct SharedLines(Arc<[Line]>);
impl PartialEq for SharedLines {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
/// What a frame showed. An unchanged frame is not drawn again.
#[derive(PartialEq)]
struct Shown {
    camera: Camera,
    size: [u32; 2],
    point_size: f32,
    adaptive: Option<f32>,
    min_size: f32,
    max_size: f32,
    edl: Edl,
    highlight_box: Option<DMat4>,
    coloring: Coloring,
    height_clip: [f64; 2],
    /// Per node: samples, marks and spacings by address, world and tint.
    nodes: Vec<ShownNode>,
    lines: SharedLines,
    lines_on_top: bool,
}
pub struct PointRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    /// One `Uniform` per segment, `uniform_stride` bytes apart, selected with
    /// a dynamic offset.
    uniform: wgpu::Buffer,
    uniform_stride: u64,
    uniform_slots: usize,
    layout: wgpu::BindGroupLayout,
    bind: wgpu::BindGroup,
    /// Zeros for nodes without flags or spacings, as long as the largest node.
    zeros: wgpu::Buffer,
    /// Resident nodes by the address of their samples.
    nodes: HashMap<usize, GpuNode>,
    frame: u64,
    resident: usize,
    /// Points kept resident at most, besides those drawn this frame.
    limit: usize,
    /// Whether nodes were left for later frames to upload.
    pending: bool,
    /// The last frame drawn, and per segment the id of its node.
    shown: Option<Shown>,
    drawn: Vec<usize>,
    line_pipeline: wgpu::RenderPipeline,
    /// lines drawn over the points
    line_on_top: wgpu::RenderPipeline,
    line_uniform: wgpu::Buffer,
    line_bind: wgpu::BindGroup,
    /// Grows when necessary, instead of allocating a GPU buffer for every camera movement.
    line_buffer: Option<wgpu::Buffer>,
    line_vertices: Vec<LineVertex>,
    line_geometry: Option<(Camera, SharedLines)>,
    edl_pipeline: wgpu::RenderPipeline,
    edl_layout: wgpu::BindGroupLayout,
    edl_uniform: wgpu::Buffer,
    targets: Option<Targets>,
    id: Option<egui::TextureId>,
}
impl PointRenderer {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point splats"),
            source: wgpu::ShaderSource::Wgsl(include_str!("points.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(UNIFORM_SIZE),
                },
                count: None,
            }],
        });
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let uniform_stride = UNIFORM_SIZE.div_ceil(alignment) * alignment;
        let (uniform, bind) = uniform_slots(&device, &layout, uniform_stride, 1);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point splats"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ViewPoint>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0=>Float32x3,1=>Unorm8x4],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 4,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![2=>Uint32],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 4,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![3=>Float32],
                    }),
                ],
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[
                    Some(SCENE_FORMAT.into()),
                    Some(VIEW_DEPTH_FORMAT.into()),
                    Some(ID_FORMAT.into()),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        let (line_pipeline, line_on_top, line_uniform, line_bind) = line_pipeline(&device);
        let (edl_pipeline, edl_layout) = edl_pipeline(&device);
        let edl_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edl"),
            size: std::mem::size_of::<EdlUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let zeros = zeros(&device, 0);
        Self {
            device,
            queue,
            pipeline,
            uniform,
            uniform_stride,
            uniform_slots: 1,
            layout,
            bind,
            zeros,
            nodes: HashMap::new(),
            frame: 0,
            resident: 0,
            limit: 0,
            pending: false,
            shown: None,
            drawn: vec![],
            line_pipeline,
            line_on_top,
            line_uniform,
            line_bind,
            line_buffer: None,
            line_vertices: Vec::new(),
            line_geometry: None,
            edl_pipeline,
            edl_layout,
            edl_uniform,
            targets: None,
            id: None,
        }
    }
    /// Keeps about `points` resident, like Potree's point load limit.
    pub fn set_point_limit(&mut self, points: usize) {
        self.limit = points;
    }
    /// Whether some nodes still wait for upload in a later frame.
    pub fn pending(&self) -> bool {
        self.pending
    }
    /// Uploads a node unless resident, and brings its marks up to date.
    /// Returns false when it has to wait for a later frame.
    fn resident(&mut self, node: &DrawNode, uploaded: &mut usize) -> bool {
        let key = Arc::as_ptr(node.points) as usize;
        if !self.nodes.contains_key(&key) {
            if *uploaded > 0 && *uploaded + node.points.len() > UPLOAD_PER_FRAME {
                self.pending = true;
                return false;
            }
            *uploaded += node.points.len();
            let gpu = GpuNode {
                _points: Arc::downgrade(node.points),
                origin: DVec3::from(node.points.origin),
                vertices: self
                    .vertex_buffer("node points", bytemuck::cast_slice(&node.points.points)),
                flags: None,
                spacings: None,
                spacings_written: None,
                marks_written: None,
                count: node.points.len() as u32,
                used: 0,
            };
            self.resident += node.points.len();
            self.nodes.insert(key, gpu);
        }
        let gpu = &self.nodes[&key];
        let spacings = node.spacings.filter(|spacings| {
            gpu.spacings_written
                .as_ref()
                .is_none_or(|w| !Arc::ptr_eq(w, spacings))
        });
        let spacings = spacings.map(|spacings| {
            let buffer = self.vertex_buffer("node spacings", bytemuck::cast_slice(spacings));
            (buffer, spacings.clone())
        });
        let marks_changed = match (node.marks, &gpu.marks_written) {
            (None, None) => false,
            (Some(new), Some(old)) => !Arc::ptr_eq(new, old),
            _ => true,
        };
        let flags = marks_changed.then(|| {
            let flags = node.marks.map(|marks| {
                let flags: Vec<u32> = (0..gpu.count as usize)
                    .map(|i| marks.get(i).is_some_and(|m| *m) as u32)
                    .collect();
                self.vertex_buffer("node flags", bytemuck::cast_slice(&flags))
            });
            (flags, node.marks.cloned())
        });
        let gpu = self.nodes.get_mut(&key).unwrap();
        gpu.used = self.frame;
        if let Some((buffer, written)) = spacings {
            gpu.spacings = Some(buffer);
            gpu.spacings_written = Some(written);
        }
        if let Some((flags, written)) = flags {
            gpu.flags = flags;
            gpu.marks_written = written;
        }
        true
    }
    fn vertex_buffer(&self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::VERTEX,
            })
    }
    /// Drops the least recently drawn nodes beyond the limit.
    fn evict(&mut self) {
        self.nodes.retain(|_, node| {
            let keep = node._points.strong_count() > 0;
            if !keep {
                self.resident -= node.count as usize;
            }
            keep
        });
        if self.resident <= self.limit {
            return;
        }
        let mut old: Vec<_> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.used < self.frame)
            .map(|(k, n)| (n.used, *k))
            .collect();
        old.sort_unstable();
        for (_, key) in old {
            if self.resident <= self.limit {
                break;
            }
            let node = self.nodes.remove(&key).unwrap();
            self.resident -= node.count as usize;
        }
    }
    /// Draws the nodes, uploading those not resident yet.
    pub fn draw(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        camera: &Camera,
        size: [u32; 2],
        options: &DrawOptions,
    ) -> egui::TextureId {
        let DrawOptions {
            point_size,
            adaptive,
            min_size,
            max_size,
            edl,
            highlight_box,
            coloring,
            height_clip,
            nodes,
            lines,
            lines_on_top,
        } = *options;
        let limit = self.device.limits().max_texture_dimension_2d;
        let size = size.map(|v| v.clamp(1, limit));
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            let targets = self.create_targets(size);
            if let Some(id) = self.id {
                rs.renderer.write().update_egui_texture_from_wgpu_texture(
                    &self.device,
                    &targets.output,
                    wgpu::FilterMode::Nearest,
                    id,
                );
            } else {
                self.id = Some(rs.renderer.write().register_native_texture(
                    &self.device,
                    &targets.output,
                    wgpu::FilterMode::Nearest,
                ));
            }
            self.targets = Some(targets);
            self.shown = None;
        }
        let address = |p: *const ()| p as usize;
        let shown = Shown {
            camera: *camera,
            size,
            point_size,
            adaptive,
            min_size,
            max_size,
            edl,
            highlight_box,
            coloring,
            height_clip,
            nodes: nodes
                .iter()
                .map(|n| {
                    let addresses = [
                        address(Arc::as_ptr(n.points) as *const ()),
                        n.marks.map_or(0, |m| address(Arc::as_ptr(m) as *const ())),
                        n.spacings
                            .map_or(0, |s| address(Arc::as_ptr(s) as *const ())),
                    ];
                    (addresses, n.world, n.tint, n.limit, n.spacing)
                })
                .collect(),
            lines: SharedLines(lines.clone()),
            lines_on_top,
        };
        // Drawing again would show the same; other panels repaint often.
        if !self.pending && self.shown.as_ref() == Some(&shown) {
            return self.id.unwrap();
        }
        self.shown = Some(shown);
        self.frame += 1;
        self.pending = false;
        let mut uploaded = 0;
        let segments: Vec<_> = nodes
            .iter()
            .filter(|n| !n.points.is_empty())
            .filter(|n| self.resident(n, &mut uploaded))
            .map(|n| (Arc::as_ptr(n.points) as usize, n))
            .collect();
        self.drawn = segments.iter().map(|(_, n)| n.id).collect();
        self.evict();
        let longest = segments.iter().map(|(key, _)| self.nodes[key].count);
        let longest = longest.max().unwrap_or(0) as u64;
        if self.zeros.size() < longest * 4 {
            self.zeros = zeros(&self.device, longest);
        }
        let targets = self.targets.as_ref().unwrap();
        if segments.len() > self.uniform_slots {
            self.uniform_slots = segments.len().next_power_of_two();
            (self.uniform, self.bind) = uniform_slots(
                &self.device,
                &self.layout,
                self.uniform_stride,
                self.uniform_slots,
            );
        }
        // Vertices are relative to their node's origin; compose in f64 so
        // large coordinates keep their precision before narrowing to f32.
        let view = camera.relative_matrix() * DMat4::from_translation(-DVec3::from(camera.target));
        // Parallel projection sees behind the eye; measure EDL depth from a
        // plane as far behind the eye as the target is in front, so it stays
        // positive and EDL shades like perspective does at the target.
        let mut depth_row = camera.depth_row();
        if camera.ortho {
            depth_row.w += camera.distance;
        }
        // A metre at clip-space w = 1 spans this many pixels vertically: the
        // projection's y scale over the half-height in pixels.
        let pixels_per_metre = (if camera.ortho {
            1. / camera.half_height()
        } else {
            1. / (camera.fov * 0.5).tan()
        } * size[1] as f64
            * 0.5) as f32;
        let mut uniforms = vec![0u8; segments.len() * self.uniform_stride as usize];
        for (i, ((key, segment), slot)) in segments
            .iter()
            .zip(uniforms.chunks_mut(self.uniform_stride as usize))
            .enumerate()
        {
            let to_world = segment.world * DMat4::from_translation(self.nodes[key].origin);
            let matrix = view * to_world;
            let depth = to_world.transpose() * depth_row;
            let uniform = Uniform {
                matrix: matrix.as_mat4().to_cols_array_2d(),
                viewport: [size[0] as f32, size[1] as f32],
                size: if adaptive.is_some() {
                    min_size
                } else {
                    point_size
                },
                adaptive: adaptive.unwrap_or(0.),
                tint: segment.tint,
                depth: depth.as_vec4().to_array(),
                highlight_box: highlight_box
                    .map_or(DMat4::IDENTITY, |c| c * to_world)
                    .as_mat4()
                    .to_cols_array_2d(),
                box_highlight_on: highlight_box.is_some() as u32 as f32,
                color_mode: coloring.code(),
                pixels_per_metre,
                segment: i as u32,
                ramp: match coloring {
                    Coloring::Height([low, high]) => {
                        let z = to_world.row(2) - glam::DVec4::new(0., 0., 0., low);
                        z / (high - low).max(1e-9)
                    }
                    _ => glam::DVec4::ZERO,
                }
                .as_vec4()
                .to_array(),
                node_spacing: segment.spacing,
                max_size,
                // finite: no limit as a height no point has
                clip: height_clip.map(|z| z.clamp(-1e30, 1e30) as f32),
                height: to_world.row(2).as_vec4().to_array(),
            };
            slot[..UNIFORM_SIZE as usize].copy_from_slice(bytemuck::bytes_of(&uniform));
        }
        if !uniforms.is_empty() {
            self.queue.write_buffer(&self.uniform, 0, &uniforms);
        }
        // Lines relative to the target, cut just in front of the eye so their
        // parts behind it do not wrap around.
        let target = DVec3::from(camera.target);
        let near = if camera.ortho {
            f64::NEG_INFINITY
        } else {
            camera.distance * 1e-3
        };
        let line_geometry = (*camera, SharedLines(lines.clone()));
        if self.line_geometry.as_ref() != Some(&line_geometry) {
            self.line_vertices.clear();
            self.line_vertices.extend(lines.iter().filter_map(|line| {
                let (da, db) = (
                    depth_row.dot(line.a.extend(1.)),
                    depth_row.dot(line.b.extend(1.)),
                );
                let (a, b) = match (da >= near, db >= near) {
                    (true, true) => (line.a, line.b),
                    (false, false) => return None,
                    (true, false) => (
                        line.a,
                        line.a + (line.b - line.a) * ((da - near) / (da - db)),
                    ),
                    (false, true) => (
                        line.b + (line.a - line.b) * ((db - near) / (db - da)),
                        line.b,
                    ),
                };
                Some(LineVertex {
                    a: (a - target).as_vec3().to_array(),
                    b: (b - target).as_vec3().to_array(),
                    color: line.color,
                    width: line.width,
                })
            }));
            if !self.line_vertices.is_empty() {
                let bytes = bytemuck::cast_slice(&self.line_vertices);
                if self
                    .line_buffer
                    .as_ref()
                    .is_none_or(|b| b.size() < bytes.len() as u64)
                {
                    self.line_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("lines"),
                        size: (bytes.len() as u64).next_power_of_two(),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                }
                self.queue
                    .write_buffer(self.line_buffer.as_ref().unwrap(), 0, bytes);
            }
            self.line_geometry = Some(line_geometry);
        }
        if !self.line_vertices.is_empty() {
            self.queue.write_buffer(
                &self.line_uniform,
                0,
                bytemuck::bytes_of(&LineUniform {
                    view: camera.relative_view().as_mat4().to_cols_array_2d(),
                    projection: camera.projection().as_mat4().to_cols_array_2d(),
                    viewport: [size[0] as f32, size[1] as f32],
                    pixels_per_metre,
                    ortho: camera.ortho as u32 as f32,
                    depth_offset: if camera.ortho {
                        camera.distance as f32
                    } else {
                        0.
                    },
                    padding: [lines_on_top as u32 as f32, 0., 0.],
                }),
            );
        }
        let edl = EdlUniform {
            radius: edl.radius,
            strength: edl.strength,
            padding: [0.; 2],
        };
        self.queue
            .write_buffer(&self.edl_uniform, 0, bytemuck::bytes_of(&edl));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("point splats"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.scene,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.025,
                                g: 0.035,
                                b: 0.05,
                                a: 1.,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.view_depth,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.ids_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (i, (key, segment)) in segments.iter().enumerate() {
                let node = &self.nodes[key];
                pass.set_bind_group(0, &self.bind, &[(i as u64 * self.uniform_stride) as u32]);
                pass.set_vertex_buffer(0, node.vertices.slice(..));
                pass.set_vertex_buffer(1, node.flags.as_ref().unwrap_or(&self.zeros).slice(..));
                let spacings = node.spacings.as_ref().unwrap_or(&self.zeros);
                pass.set_vertex_buffer(2, spacings.slice(..));
                pass.draw(0..6, 0..node.count.min(segment.limit));
            }
            if !self.line_vertices.is_empty() {
                let buffer = self.line_buffer.as_ref().unwrap();
                pass.set_pipeline(if lines_on_top {
                    &self.line_on_top
                } else {
                    &self.line_pipeline
                });
                pass.set_bind_group(0, &self.line_bind, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..6, 0..self.line_vertices.len() as u32);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("eye-dome lighting"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.output,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.edl_pipeline);
            pass.set_bind_group(0, &targets.edl_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.id.unwrap()
    }
    fn create_targets(&self, size: [u32; 2]) -> Targets {
        let raw = |label, format, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let texture = |label, format, usage| {
            let texture: wgpu::Texture = raw(label, format, usage);
            texture.create_view(&Default::default())
        };
        let sampled = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let scene = texture("point colors", SCENE_FORMAT, sampled);
        let view_depth = texture("point view depth", VIEW_DEPTH_FORMAT, sampled);
        let ids = raw(
            "point ids",
            ID_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let ids_view = ids.create_view(&Default::default());
        let depth = texture(
            "point depth",
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let output = texture(
            "point viewport",
            OUTPUT_FORMAT,
            sampled | wgpu::TextureUsages::COPY_SRC,
        );
        let edl_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("edl"),
            layout: &self.edl_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.edl_uniform.as_entire_binding(),
                },
            ],
        });
        Targets {
            size,
            scene,
            view_depth,
            ids,
            ids_view,
            depth,
            output,
            edl_bind,
        }
    }
    /// The point drawn last at `click` (normalized viewport coordinates),
    /// else the one closest to it within `reach` physical pixels: the id of
    /// its node and its index there. Waits for the GPU to finish the frame.
    pub fn pick(&self, click: [f64; 2], reach: u32) -> Option<(usize, usize)> {
        let targets = self.targets.as_ref()?;
        let [width, height] = targets.size.map(|v| v as i64);
        let x = (click[0] * width as f64).floor() as i64;
        let y = (click[1] * height as f64).floor() as i64;
        if x < 0 || y < 0 || x >= width || y >= height {
            return None;
        }
        let reach = reach.min(MAX_PICK_REACH) as i64;
        let (x0, y0) = ((x - reach).max(0), (y - reach).max(0));
        let (x1, y1) = ((x + reach).min(width - 1), (y + reach).min(height - 1));
        let (w, h) = ((x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32);
        let row = (w * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("picked ids"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &targets.ids,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: x0 as u32,
                    y: y0 as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        buffer.map_async(wgpu::MapMode::Read, .., |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let data = buffer.get_mapped_range(..).ok()?;
        let ids: Vec<[u32; 2]> = data
            .chunks(row as usize)
            .flat_map(|line| bytemuck::cast_slice::<u8, [u32; 2]>(&line[..w as usize * 8]))
            .copied()
            .collect();
        let center = [(x - x0) as u32, (y - y0) as u32];
        let [segment, index] = nearest_id(&ids, w, center)?;
        let id = *self.drawn.get(segment as usize - 1)?;
        Some((id, index as usize))
    }
}

const UNIFORM_SIZE: u64 = std::mem::size_of::<Uniform>() as u64;

/// In a window of ids `width` wide, the one at `center`, else the closest
/// to it; 0 in the first component marks background.
fn nearest_id(ids: &[[u32; 2]], width: u32, center: [u32; 2]) -> Option<[u32; 2]> {
    ids.iter()
        .enumerate()
        .filter(|(_, id)| id[0] != 0)
        .min_by_key(|(i, _)| {
            let (x, y) = ((*i as u32 % width) as i64, (*i as u32 / width) as i64);
            let (dx, dy) = (x - center[0] as i64, y - center[1] as i64);
            dx * dx + dy * dy
        })
        .map(|(_, id)| *id)
}

/// A vertex buffer of four zero bytes per point for at least `points`;
/// wgpu clears new buffers.
fn zeros(device: &wgpu::Device, points: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zeros"),
        size: points.max(1024).next_power_of_two() * 4,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

fn uniform_slots(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    stride: u64,
    slots: usize,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("point segments"),
        size: stride * slots as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("point segments"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: wgpu::BufferSize::new(UNIFORM_SIZE),
            }),
        }],
    });
    (buffer, bind)
}

/// The pipeline for depth-tested lines, its uniform and bind group.
/// Lines tested against the points' depth, and lines drawn over them; their uniform and bind group.
fn line_pipeline(
    device: &wgpu::Device,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::Buffer,
    wgpu::BindGroup,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lines"),
        source: wgpu::ShaderSource::Wgsl(include_str!("lines.wgsl").into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lines"),
        size: std::mem::size_of::<LineUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lines"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let make = |compare| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0=>Float32x3, 1=>Float32x3, 2=>Unorm8x4, 3=>Float32
                    ],
                })],
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                // Lines leave the ids of the points behind them.
                targets: &[
                    Some(SCENE_FORMAT.into()),
                    Some(VIEW_DEPTH_FORMAT.into()),
                    Some(wgpu::ColorTargetState {
                        format: ID_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    (
        make(wgpu::CompareFunction::Less),
        make(wgpu::CompareFunction::Always),
        uniform,
        bind,
    )
}

fn edl_pipeline(device: &wgpu::Device) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("eye-dome lighting"),
        source: wgpu::ShaderSource::Wgsl(include_str!("edl.wgsl").into()),
    });
    // Both inputs are read with textureLoad, so no sampler or filtering is needed.
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("edl"),
        entries: &[
            texture(0),
            texture(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("edl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("eye-dome lighting"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(OUTPUT_FORMAT.into())],
        }),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, layout)
}

#[cfg(test)]
mod tests {
    use super::nearest_id;

    #[test]
    fn picks_the_point_under_the_pointer_else_the_closest() {
        // A 4x3 window, pointer at (1, 1).
        let mut ids = [[0, 0]; 12];
        ids[3] = [1, 7]; // (3, 0): 5 away squared
        ids[6] = [2, 9]; // (2, 1): 1 away
        assert_eq!(nearest_id(&ids, 4, [1, 1]), Some([2, 9]));
        ids[5] = [3, 4]; // under the pointer
        assert_eq!(nearest_id(&ids, 4, [1, 1]), Some([3, 4]));
        assert_eq!(nearest_id(&[[0, 0]; 12], 4, [1, 1]), None);
    }
}
