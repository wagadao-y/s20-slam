// The colours of one image's visible points on the GPU (see gpu.rs `ZBuffer::colours` and the
// CPU's `Scene::view` and `gather` in color.rs): after the visible pass wrote (point, u, v, range,
// angle) per point, each point's colour (bilinear, as the CPU), score and grazing flag; the
// image's colour sums per cell (integers: the colours x 256 as packed, so the order of the threads
// does not matter); each point's mean of the 8 cells around it (trilinear); and the cells the
// gains take. Results per point: slot | wide << 31, score, then the colour and the cells' mean
// packed as u16 (x 256): (r | g << 16), (b | mean r << 16), (mean g | mean b << 16).

struct Params {
    t: vec4f,      // the camera's centre, the image's turn term (turn weight x omega / 30)
    angles: vec4f, // soft angle, hard angle (the largest f32 not beyond it), theta slope, -
    size: vec4u,   // image width, height, places, cells
    first: vec4u,  // first point of this points' buffer, the end of its range, room for gain entries, -
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pts: array<f32>;
@group(0) @binding(2) var<storage, read> nrm: array<f32>;
// (point, u, v, range, angle) per visible point, and their number
@group(0) @binding(3) var<storage, read> vis: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
// per point (block order): its slot, u32 max where it is not coloured
@group(0) @binding(5) var<storage, read> slot_of: array<u32>;
// per slot: its cell and its place among the 8 cells around it (the fractions of the cells'
// grid, found on the CPU in double: 10 bits each), then those 8 cells (u32 max: none)
@group(0) @binding(6) var<storage, read> cell_of: array<vec2u>;
@group(0) @binding(7) var<storage, read> around: array<u32>;
// per cell: whether the part owns it (its gain entries)
@group(0) @binding(8) var<storage, read> owned: array<u32>;
// the image, RGB8 four bytes to a word
@group(0) @binding(9) var<storage, read> img: array<u32>;
// per cell: the colour sums (x 256) and the number of points
@group(0) @binding(10) var<storage, read_write> sums: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> res: array<u32>;
// per gain entry: cell, points, mean r, g, b (f32)
@group(0) @binding(12) var<storage, read_write> gains: array<u32>;

const NONE: u32 = 0xffffffffu;

fn thread(wid: vec3u, lid: vec3u, nwg: vec3u) -> u32 {
    return (wid.y * nwg.x + wid.x) * 64u + lid.x;
}

fn byte(o: u32) -> f32 {
    return f32((img[o >> 2u] >> ((o & 3u) * 8u)) & 255u);
}

fn px(x: u32, y: u32) -> vec3f {
    let o = 3u * (y * P.size.x + x);
    return vec3f(byte(o), byte(o + 1u), byte(o + 2u));
}

// as the CPU's `pack`: x 256, rounded (halves up: the values are not negative), within u16
fn pack1(x: f32) -> u32 {
    return u32(clamp(floor(x * 256.0 + 0.5), 0.0, 65535.0));
}

// the visible points this pass of the points' buffer covers
fn record(k: u32) -> bool {
    let n = min(atomicLoad(&counts[0]), P.size.z);
    if k >= n {
        return false;
    }
    let j = vis[5u * k];
    return j >= P.first.x && j < P.first.y;
}

@compute @workgroup_size(64)
fn colour(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let k = thread(wid, lid, nwg);
    if !record(k) {
        return;
    }
    let j = vis[5u * k];
    let slot = slot_of[j];
    if slot == NONE {
        res[5u * k] = NONE;
        return;
    }
    let u = bitcast<f32>(vis[5u * k + 1u]);
    let v = bitcast<f32>(vis[5u * k + 2u]);
    let r = bitcast<f32>(vis[5u * k + 3u]);
    let th = bitcast<f32>(vis[5u * k + 4u]);
    let u0 = u32(floor(u));
    let v0 = u32(floor(v));
    let fu = u - f32(u0);
    let fv = v - f32(v0);
    let rgb = px(u0, v0) * (1.0 - fu) * (1.0 - fv) + px(u0 + 1u, v0) * fu * (1.0 - fv)
        + px(u0, v0 + 1u) * (1.0 - fu) * fv + px(u0 + 1u, v0 + 1u) * fu * fv;
    let i = j - P.first.x;
    let m = vec3f(nrm[3u * i], nrm[3u * i + 1u], nrm[3u * i + 2u]);
    var cosine = 0.5;
    if dot(m, m) > 0.25 {
        let q = vec3f(pts[3u * i], pts[3u * i + 1u], pts[3u * i + 2u]);
        cosine = abs(dot(m, P.t.xyz - q)) / max(r, 1e-6);
    }
    let s = 5.15 * -log(max(r, 0.1)) + 10.6 * cosine - 4.05 * th - P.t.w
        - P.angles.z * max(th - P.angles.x, 0.0);
    let c = vec3u(pack1(rgb.x), pack1(rgb.y), pack1(rgb.z));
    let wide = select(0u, 1u << 31u, th > P.angles.y);
    res[5u * k] = slot | wide;
    res[5u * k + 1u] = bitcast<u32>(s);
    res[5u * k + 2u] = c.x | (c.y << 16u);
    res[5u * k + 3u] = c.z;
    let cell = cell_of[slot].x;
    atomicAdd(&sums[4u * cell], c.x);
    atomicAdd(&sums[4u * cell + 1u], c.y);
    atomicAdd(&sums[4u * cell + 2u], c.z);
    atomicAdd(&sums[4u * cell + 3u], 1u);
}

@compute @workgroup_size(64)
fn around_mean(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let k = thread(wid, lid, nwg);
    if !record(k) {
        return;
    }
    let slot = res[5u * k];
    if slot == NONE {
        return;
    }
    let s = slot & 0x7fffffffu;
    let f = cell_of[s].y;
    let fr = vec3f(f32(f & 1023u), f32((f >> 10u) & 1023u), f32((f >> 20u) & 1023u)) / 1023.0;
    var acc = vec3f(0.0);
    var ws = 0.0;
    for (var o = 0u; o < 8u; o++) {
        let c = around[8u * s + o];
        if c == NONE {
            continue;
        }
        let n = atomicLoad(&sums[4u * c + 3u]);
        if n == 0u {
            continue;
        }
        let mean = vec3f(
            f32(atomicLoad(&sums[4u * c])),
            f32(atomicLoad(&sums[4u * c + 1u])),
            f32(atomicLoad(&sums[4u * c + 2u])),
        ) / (256.0 * f32(n));
        let off = vec3f(f32(o & 1u), f32((o >> 1u) & 1u), f32((o >> 2u) & 1u));
        let wv = select(1.0 - fr, fr, off > vec3f(0.5));
        let w = wv.x * wv.y * wv.z + 1e-9;
        acc += w * mean;
        ws += w;
    }
    let pm = acc / max(ws, 1e-12);
    res[5u * k + 3u] = res[5u * k + 3u] | (pack1(pm.x) << 16u);
    res[5u * k + 4u] = pack1(pm.y) | (pack1(pm.z) << 16u);
}

@compute @workgroup_size(64)
fn gain_cells(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let c = thread(wid, lid, nwg);
    if c >= P.size.w || owned[c] == 0u {
        return;
    }
    let n = atomicLoad(&sums[4u * c + 3u]);
    if n < 20u {
        return;
    }
    let mean = vec3f(
        f32(atomicLoad(&sums[4u * c])),
        f32(atomicLoad(&sums[4u * c + 1u])),
        f32(atomicLoad(&sums[4u * c + 2u])),
    ) / (256.0 * f32(n));
    if !(all(mean > vec3f(8.0)) && all(mean < vec3f(245.0))) {
        return;
    }
    let e = atomicAdd(&counts[1], 1u);
    if e >= P.first.z {
        return;
    }
    gains[5u * e] = c;
    gains[5u * e + 1u] = n;
    gains[5u * e + 2u] = bitcast<u32>(mean.x);
    gains[5u * e + 3u] = bitcast<u32>(mean.y);
    gains[5u * e + 4u] = bitcast<u32>(mean.z);
}
