// The points of one fisheye image's view that pass the occlusion test (see gpu.rs and
// Scene::view in color.rs). One thread per point of the listed ranges: project as the CPU's
// Lens (from the block's centre, the same atan), keep it within range, the field of view and
// the image, test it against the z-buffer's window minimum and depth slope, and against its
// normal; write (block order, u, v, range, theta) at the next free place.

struct Params {
    r0: vec4f,    // world -> camera rotation rows
    r1: vec4f,
    r2: vec4f,
    k03: vec4f,   // fisheye polynomial k0..k3
    k45: vec4f,   // k4, k5
    a: vec4f,     // a11, a12, a22, u0
    b: vec4f,     // v0, w - 2, h - 2, max range
    c: vec4f,     // largest angle off the axis, a11 * spacing / cell, graze (0 or 1)
    d: vec4u,     // cells across, ranges, threads, first point of this buffer
    e: vec4u,     // cells down, places for the output
}

// per range: first point, first thread, the block's centre and the centre less the camera's
struct Range {
    first: u32,
    thread: u32,
    pad0: u32,
    pad1: u32,
    centre: vec4f,
    to: vec4f,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read> nrm: array<f32>;
@group(0) @binding(3) var<storage, read> ranges: array<Range>;
// the window minima of 1..6 cells, one grid after the other, then the depth slopes
@group(0) @binding(4) var<storage, read> grids: array<f32>;
@group(0) @binding(5) var<storage, read_write> count: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<u32>;

fn atan01(x: f32) -> f32 {
    let big = x > 0.41421357;
    let y = select(x, (x - 1.0) / (x + 1.0), big);
    let z = y * y;
    let p = (((0.080537445 * z - 0.13877686) * z + 0.19977711) * z - 0.33332949) * z * y + y;
    return select(p, p + 0.78539816, big);
}

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
        if (ranges[mid].thread <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let rg = ranges[lo];
    let j = rg.first + (t - rg.thread);
    let i = j - P.d.w;
    let q = vec3f(pts[3u * i], pts[3u * i + 1u], pts[3u * i + 2u]);
    let d = (q - rg.centre.xyz) + rg.to.xyz;
    let r = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    if (!(r <= P.b.w)) {
        return;
    }
    let x = P.r0.x * d.x + P.r0.y * d.y + P.r0.z * d.z;
    let y = P.r1.x * d.x + P.r1.y * d.y + P.r1.z * d.z;
    let z = P.r2.x * d.x + P.r2.y * d.y + P.r2.z * d.z;
    let rho = sqrt(x * x + y * y);
    let az = abs(z);
    let lo2 = min(rho, az);
    let hi2 = max(rho, az);
    var qq = 0.0;
    if (hi2 > 0.0) {
        qq = lo2 / hi2;
    }
    var a = atan01(qq);
    a = select(a, 1.5707964 - a, rho > az);
    let th = select(a, 3.1415927 - a, z < 0.0);
    if (!(th < P.c.x)) {
        return;
    }
    let rr = th * (1.0 + th * (P.k03.x + th * (P.k03.y + th * (P.k03.z + th * (P.k03.w + th * (P.k45.x + th * P.k45.y))))));
    var cp = 1.0;
    var sp = 0.0;
    if (rho > 1e-9) {
        let inv = 1.0 / rho;
        cp = x * inv;
        sp = y * inv;
    }
    let u = P.a.x * rr * cp + P.a.y * rr * sp + P.a.w;
    let v = P.a.z * rr * sp + P.b.x;
    if (!(u >= 1.0 && u < P.b.y && v >= 1.0 && v < P.b.z)) {
        return;
    }
    // the occlusion test
    let reach = u32(clamp(ceil(P.c.y / max(r, 1e-3)), 1.0, 6.0));
    let n = P.d.x * P.e.x;
    let cell = u32(v / 8.0) * P.d.x + u32(u / 8.0);
    let near = grids[(reach - 1u) * n + cell];
    var slope = 0.0;
    if (P.c.z > 0.5) {
        slope = grids[6u * n + cell] * f32(reach + 1u);
    }
    if (r > near + max(0.03, 0.02 * r) + slope) {
        return;
    }
    // facing the camera, where the point has a normal
    let m = vec3f(nrm[3u * i], nrm[3u * i + 1u], nrm[3u * i + 2u]);
    if (dot(m, m) > 0.25 && dot(m, -d) <= 0.0) {
        return;
    }
    let k = atomicAdd(&count, 1u);
    if (k >= P.e.y) {
        return;
    }
    out[5u * k] = j;
    out[5u * k + 1u] = bitcast<u32>(u);
    out[5u * k + 2u] = bitcast<u32>(v);
    out[5u * k + 3u] = bitcast<u32>(r);
    out[5u * k + 4u] = bitcast<u32>(th);
}
