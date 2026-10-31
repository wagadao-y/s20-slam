// Eye-Dome Lighting, after Potree's edl.fs (adapted from CloudCompare's qEDL by
// Christian Boucheny). Darkens pixels that lie behind their neighbours in log2
// view depth, so shape reads without normals or colour. Depth 0 is background.
struct Edl { radius: f32, strength: f32, padding: vec2<f32> };
@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var depth: texture_2d<f32>;
@group(0) @binding(2) var<uniform> edl: Edl;

@vertex fn vertex(@builtin(vertex_index) id: u32) -> @builtin(position) vec4<f32> {
    // One triangle covering the viewport.
    let uv = vec2(f32((id << 1u) & 2u), f32(id & 2u));
    return vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}

// The output is a plain Rgba8Unorm texture that egui samples as sRGB code values.
fn srgb_from_linear(rgb: vec3<f32>) -> vec3<f32> {
    let low = rgb * 12.92;
    let high = 1.055 * pow(rgb, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, rgb <= vec3<f32>(0.0031308));
}

@fragment fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    // Linear RGB: the scene target is sRGB-encoded and decoded on load.
    let base = textureLoad(color, pixel, 0);
    let w = textureLoad(depth, pixel, 0).r;
    if w <= 0.0 || edl.strength <= 0.0 {
        return vec4(srgb_from_linear(base.rgb), base.a);
    }
    let center = log2(w);
    let size = vec2<i32>(textureDimensions(depth));
    var sum = 0.0;
    for (var i = 0; i < 8; i++) {
        let angle = f32(i) * 0.7853982;
        let offset = vec2(cos(angle), sin(angle)) * edl.radius;
        let neighbour = clamp(vec2<i32>(round(position.xy + offset)), vec2(0), size - 1);
        let neighbour_w = textureLoad(depth, neighbour, 0).r;
        // Background neighbours do not darken silhouettes, as in Potree.
        if neighbour_w > 0.0 {
            sum += max(0.0, center - log2(neighbour_w));
        }
    }
    let shade = exp(-sum / 8.0 * 300.0 * edl.strength);
    return vec4(srgb_from_linear(base.rgb * shade), base.a);
}
