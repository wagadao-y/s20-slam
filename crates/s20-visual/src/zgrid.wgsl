// The z-buffers of the refinement (`anchors::zgrid` on the CPU): per 8 x 8 pixel cell of an image,
// the nearest cloud point within 0.3 to 25 m and 82 degrees of the axis. Two passes over the
// points: the nearest range of each cell, then the lowest index of a point at that range. Both
// kept inverted (`~`), so that an atomic maximum keeps the least and a cleared buffer holds none.

struct Params {
    r0: vec4f,   // world -> camera rotation rows
    r1: vec4f,
    r2: vec4f,
    t: vec4f,    // camera centre (in the points' frame)
    k03: vec4f,  // fisheye polynomial k0..k3
    k45: vec4f,  // k4, k5
    a: vec4f,    // a11, a12, a22, u0
    b: vec4f,    // v0, width, height, largest angle off the axis
    d: vec4u,    // cells across, points, cell size, -
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read_write> zmin: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> zidx: array<atomic<u32>>;

// the point's cell and range bits, or none (cell u32 max)
fn hit(i: u32) -> vec2u {
    let none = vec2u(0xffffffffu, 0u);
    let x = vec3f(pts[3u * i], pts[3u * i + 1u], pts[3u * i + 2u]) - P.t.xyz;
    let r = length(x);
    if r > 25.0 || r <= 0.3 {
        return none;
    }
    let l = vec3f(dot(P.r0.xyz, x), dot(P.r1.xyz, x), dot(P.r2.xyz, x));
    let rho = length(l.xy);
    let th = atan2(rho, l.z);
    if th >= P.b.w {
        return none;
    }
    let rr = th * (1.0 + th * (P.k03.x + th * (P.k03.y + th * (P.k03.z + th * (P.k03.w + th * (P.k45.x + th * P.k45.y))))));
    var c = 1.0;
    var s = 0.0;
    if rho > 1e-15 {
        c = l.x / rho;
        s = l.y / rho;
    }
    let u = P.a.x * rr * c + P.a.y * rr * s + P.a.w;
    let v = P.a.z * rr * s + P.b.x;
    if !(u >= 0.0 && u < P.b.y && v >= 0.0 && v < P.b.z) {
        return none;
    }
    let cell = (u32(v) / P.d.z) * P.d.x + u32(u) / P.d.z;
    return vec2u(cell, bitcast<u32>(r));
}

fn thread(wid: vec3u, lid: vec3u, nwg: vec3u) -> u32 {
    return (wid.y * nwg.x + wid.x) * 64u + lid.x;
}

@compute @workgroup_size(64)
fn nearest_pass(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = thread(wid, lid, nwg);
    if i >= P.d.y {
        return;
    }
    let h = hit(i);
    if h.x != 0xffffffffu {
        atomicMax(&zmin[h.x], ~h.y);
    }
}

@compute @workgroup_size(64)
fn index_pass(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = thread(wid, lid, nwg);
    if i >= P.d.y {
        return;
    }
    let h = hit(i);
    if h.x != 0xffffffffu && ~h.y == atomicLoad(&zmin[h.x]) {
        atomicMax(&zidx[h.x], ~i);
    }
}
