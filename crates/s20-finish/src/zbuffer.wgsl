// The nearest range per 8 x 8 pixel cell of one fisheye image (see gpu.rs and the CPU loop in
// color.rs). One thread per point of the listed ranges: project, keep the range in its cell
// (positive floats order as their bits, so an atomic minimum of the bits).

struct Params {
    r0: vec4f,    // world -> camera rotation rows
    r1: vec4f,
    r2: vec4f,
    t: vec4f,     // camera centre
    k03: vec4f,   // fisheye polynomial k0..k3
    k45: vec4f,   // k4, k5
    a: vec4f,     // a11, a12, a22, u0
    b: vec4f,     // v0, w - 2, h - 2, max range
    c: vec4f,     // largest angle off the axis
    cells: vec4i, // cells looked at: lo x, lo y, hi x, hi y (inclusive)
    d: vec4u,     // cells across, ranges, threads, first point of this buffer
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
// per range: (first point, first thread), sorted by first thread
@group(0) @binding(2) var<storage, read> ranges: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> zb: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let t = (wid.y * nwg.x + wid.x) * 64u + lid.x;
    if (t >= P.d.z) {
        return;
    }
    var lo = 0u;
    var hi = P.d.y;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (ranges[mid].y <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let i = ranges[lo].x + (t - ranges[lo].y) - P.d.w;
    let x = vec3f(pts[3u * i], pts[3u * i + 1u], pts[3u * i + 2u]) - P.t.xyz;
    let r = length(x);
    if (r > P.b.w) {
        return;
    }
    let xc = vec3f(dot(P.r0.xyz, x), dot(P.r1.xyz, x), dot(P.r2.xyz, x));
    let rho = length(xc.xy);
    let th = atan2(rho, xc.z);
    if (th >= P.c.x) {
        return;
    }
    let rr = th * (1.0 + th * (P.k03.x + th * (P.k03.y + th * (P.k03.z + th * (P.k03.w + th * (P.k45.x + th * P.k45.y))))));
    var cp = 1.0;
    var sp = 0.0;
    if (rho > 1e-9) {
        cp = xc.x / rho;
        sp = xc.y / rho;
    }
    let u = P.a.x * rr * cp + P.a.y * rr * sp + P.a.w;
    let v = P.a.z * rr * sp + P.b.x;
    if (!(u >= 1.0 && u < P.b.y && v >= 1.0 && v < P.b.z)) {
        return;
    }
    let cx = i32(u) / 8;
    let cy = i32(v) / 8;
    if (cx < P.cells.x || cy < P.cells.y || cx > P.cells.z || cy > P.cells.w) {
        return;
    }
    atomicMin(&zb[u32(cy) * P.d.x + u32(cx)], bitcast<u32>(r));
}
