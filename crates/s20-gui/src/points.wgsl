struct Camera {
    matrix: mat4x4<f32>, viewport: vec2<f32>, size: f32, adaptive: f32, tint: vec4<f32>,
    depth: vec4<f32>, highlight_box: mat4x4<f32>, box_highlight_on: f32,
    // 0 own colour, 1 grey by reflectivity (the alpha), 2 height, 3 tint shaded by reflectivity, 4 grey
    color_mode: f32,
    pixels_per_metre: f32, segment: u32,
    // Height in the ramp's 0..1 range as dot(ramp, (position, 1)).
    ramp: vec4<f32>,
    // Spacing of the drawn points for adaptive size, when the whole draw has one (else per point).
    node_spacing: f32,
    max_size: f32,
    // Points with a height outside this range are not drawn; the height as dot(height, (position, 1)).
    clip: vec2<f32>,
    height: vec4<f32>,
};
// Turbo colour map (Google), polynomial fit, sRGB output: of t in 0..1, the part 0.12..0.88 (the
// ends hardly show on the dark background; as TURBO_RANGE in app.rs).
fn turbo(u: f32) -> vec3<f32> {
    let t = 0.12 + u * 0.76;
    let r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
    let g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
    let b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
    return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}
@group(0) @binding(0) var<uniform> camera: Camera;
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) corner: vec2<f32>,
    @location(2) depth: f32,
    @location(3) @interpolate(flat) id: vec2<u32>,
};
// sRGB highlight matching the box edges.
const BOX_HIGHLIGHT = vec3<f32>(1.0, 0.863, 0.314);
// sRGB colour of points a move would take (flag bit 0).
const HIGHLIGHT = vec3<f32>(1.0, 0.188, 0.188);
@vertex fn vertex(@builtin(vertex_index) id: u32, @builtin(instance_index) point: u32, @location(0) position: vec3<f32>, @location(1) color: vec4<f32>, @location(2) flags: u32, @location(3) spacing: f32) -> Out {
    let corners = array<vec2<f32>, 6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
    var out: Out;
    let z = dot(camera.height, vec4(position, 1.0));
    if z < camera.clip.x || z > camera.clip.y {
        // outside the clip volume: the splat is not drawn (nor picked)
        out.position = vec4(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    out.position = camera.matrix * vec4(position,1.0);
    var size = camera.size;
    if camera.adaptive > 0.0 {
        // Potree's adaptive size: 1.7 times the point's spacing, on screen.
        let s = select(spacing, camera.node_spacing, camera.node_spacing > 0.0);
        let world = camera.adaptive * 1.7 * s;
        size = clamp(world * camera.pixels_per_metre / out.position.w, camera.size, camera.max_size);
    }
    out.position = vec4(out.position.xy + corners[id] * size / camera.viewport * out.position.w, out.position.zw);
    out.corner = corners[id];
    var base = color.rgb;
    let mode = u32(camera.color_mode + 0.5);
    if mode == 1u {
        base = vec3(color.a);
    } else if mode == 2u {
        base = turbo(clamp(dot(camera.ramp, vec4(position, 1.0)), 0.0, 1.0));
    } else if mode == 3u {
        base = camera.tint.rgb * (0.45 + 0.55 * color.a);
    } else if mode == 4u {
        base = vec3(0.78);
    }
    // Mix the tint in sRGB, like the colours it replaces (as the colour itself when by tint).
    let tint = select(camera.tint.a, 0.0, mode == 3u);
    // the alpha carries the reflectivity: the splats are opaque
    out.color = vec4(mix(base, camera.tint.rgb, tint), 1.0);
    // Keep the surrounding points visible; mark points inside the oriented box.
    if camera.box_highlight_on > 0.5 && all(abs((camera.highlight_box * vec4(position, 1.0)).xyz) <= vec3(1.0)) {
        out.color = vec4(BOX_HIGHLIGHT, 1.0);
    }
    // Selection previews take priority over the box highlight.
    if (flags & 1u) != 0u {
        out.color = vec4(HIGHLIGHT, 1.0);
    }
    // View-space distance along the camera axis; screen-aligned splats keep it
    // constant. 0 marks background, so keep drawn points above it.
    out.depth = max(dot(camera.depth, vec4(position, 1.0)), 1e-6);
    // Which point this is, for picking.
    out.id = vec2(camera.segment + 1u, point);
    return out;
}
fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let low = srgb / vec3<f32>(12.92);
    let high = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(high, low, srgb <= vec3<f32>(0.04045));
}
struct Target { @location(0) color: vec4<f32>, @location(1) depth: f32, @location(2) id: vec2<u32> };
@fragment fn fragment(in: Out) -> Target {
    if dot(in.corner,in.corner) > 1.0 { discard; }
    return Target(vec4(srgb_to_linear(in.color.rgb), in.color.a), in.depth, in.id);
}
