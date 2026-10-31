// Screen-width lines tested against the points' depth, e.g. box edges that
// points in front of them hide.
struct Lines {
    // Positions are relative to the camera target, like `view`.
    view: mat4x4<f32>, projection: mat4x4<f32>, viewport: vec2<f32>,
    // Physical pixels a metre spans at a view depth of 1 (perspective) or at
    // any depth (parallel projection).
    pixels_per_metre: f32, ortho: f32,
    // Added to the view depth for EDL, as for the points.
    depth_offset: f32,
    // 1: drawn over the points; they leave no depth, so eye-dome lighting does not shade them
    on_top: f32, padding2: f32, padding3: f32,
};
@group(0) @binding(0) var<uniform> lines: Lines;
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
};
// Lines are pulled this many pixels' worth toward the eye, so points lying
// on a box face (a tight box touches its outermost points) do not hide it.
const BIAS_PIXELS = 4.0;
fn toward_eye(p: vec3<f32>) -> vec4<f32> {
    var v = lines.view * vec4(p, 1.0);
    var metres = BIAS_PIXELS / lines.pixels_per_metre;
    if lines.ortho < 0.5 {
        metres = metres * max(-v.z, 0.0);
    }
    v.z = v.z + metres;
    return v;
}
@vertex fn vertex(@builtin(vertex_index) id: u32, @location(0) a: vec3<f32>, @location(1) b: vec3<f32>, @location(2) color: vec4<f32>, @location(3) width: f32) -> Out {
    let ends = array<f32, 6>(0.0, 1.0, 1.0, 0.0, 1.0, 0.0);
    let sides = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let va = toward_eye(a);
    let vb = toward_eye(b);
    let ca = lines.projection * va;
    let cb = lines.projection * vb;
    let half = lines.viewport * 0.5;
    let sa = ca.xy / ca.w * half;
    let sb = cb.xy / cb.w * half;
    var along = sb - sa;
    if dot(along, along) < 1e-12 {
        along = vec2(1.0, 0.0);
    }
    along = normalize(along);
    let across = vec2(-along.y, along.x);
    let end = ends[id];
    var c = mix(ca, cb, end);
    // Half the width across, and as far past each end, so corners close.
    let offset = (across * sides[id] + along * (end * 2.0 - 1.0)) * width * 0.5;
    c = vec4(c.xy + offset / half * c.w, c.zw);
    var out: Out;
    out.position = c;
    out.color = color;
    out.depth = select(max(-mix(va, vb, end).z + lines.depth_offset, 1e-6), 0.0, lines.on_top > 0.5);
    return out;
}
fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let low = srgb / vec3<f32>(12.92);
    let high = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(high, low, srgb <= vec3<f32>(0.04045));
}
struct Target { @location(0) color: vec4<f32>, @location(1) depth: f32 };
@fragment fn fragment(in: Out) -> Target {
    return Target(vec4(srgb_to_linear(in.color.rgb), in.color.a), in.depth);
}
