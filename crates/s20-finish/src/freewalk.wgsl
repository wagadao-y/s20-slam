// The free-space ray walk (see freespace.rs, the CPU loop of FreeSpace::new). One thread per
// ray: half-voxel steps from the sensor to `margin` before the point; a measured voxel passed
// gets the bit of the ray's scan (one vote per scan and voxel). `count` then adds the bits of
// each voxel to its passes and clears them for the next scans.

struct Params {
    inv: f32,          // 1 / voxel
    step: f32,         // half a voxel
    margin: f32,
    far: f32,          // pass_far (0: off)
    r2: f32,           // pass_radius^2
    centroid: u32,     // 1: test the distance to the voxel's centroid
    rays: u32,
    vmask: u32,        // voxel table slots - 1
    bmask: u32,        // block table slots - 1
    voxels: u32,
    lmask: u32,        // large block table slots - 1
    pad0: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
// per ray: (sensor, bit of its scan as u32 bits), (point, 0)
@group(0) @binding(1) var<storage, read> rays: array<vec4f>;
// open addressing, 3 u32 per slot: key high, key low, voxel index (high 0xffffffff: empty)
@group(0) @binding(2) var<storage, read> vtable: array<u32>;
// open addressing, (key high, key low) per slot
@group(0) @binding(3) var<storage, read> btable: array<vec2u>;
@group(0) @binding(4) var<storage, read> centroids: array<f32>;
@group(0) @binding(5) var<storage, read_write> bits: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> passes: array<u32>;
// open addressing, (key high, key low) per slot: the large blocks (4 blocks per axis) holding blocks
@group(0) @binding(7) var<storage, read> ltable: array<vec2u>;

const EMPTY: u32 = 0xffffffffu;
const NONE: u32 = 0xffffffffu;

// the key of freespace.rs `pack` as (high, low) halves
fn pack(c: vec3i) -> vec2u {
    let k = vec3u(c + vec3i(1 << 20)) & vec3u(0x1fffffu);
    return vec2u((k.x << 10u) | (k.y >> 11u), (k.y << 21u) | k.z);
}

// as gpu.rs `slot_hash`
fn slot_hash(k: vec2u) -> u32 {
    var h = (k.x * 0x9e3779b1u) ^ k.y;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    return h ^ (h >> 16u);
}

fn voxel(k: vec2u) -> u32 {
    var i = slot_hash(k) & P.vmask;
    loop {
        let hi = vtable[3u * i];
        if (hi == k.x && vtable[3u * i + 1u] == k.y) {
            return vtable[3u * i + 2u];
        }
        if (hi == EMPTY) {
            return NONE;
        }
        i = (i + 1u) & P.vmask;
    }
    return NONE;
}

fn has_large(k: vec2u) -> bool {
    var i = slot_hash(k) & P.lmask;
    loop {
        let s = ltable[i];
        if (s.x == k.x && s.y == k.y) {
            return true;
        }
        if (s.x == EMPTY) {
            return false;
        }
        i = (i + 1u) & P.lmask;
    }
    return false;
}

fn has_block(k: vec2u) -> bool {
    var i = slot_hash(k) & P.bmask;
    loop {
        let s = btable[i];
        if (s.x == k.x && s.y == k.y) {
            return true;
        }
        if (s.x == EMPTY) {
            return false;
        }
        i = (i + 1u) & P.bmask;
    }
    return false;
}

// how far along a ray from `a` to `b` (`d` = b - a, `len` long) its steps surely stay in the slab
// lo..hi of one axis: where it leaves the slab, less the rounding of a step's coordinate (well
// over that of `a + d * (s / len)` and of the voxel's `floor(x * inv)`, in single precision, fused
// or not) over the ray's slope on the axis: on an axis the ray nearly runs along, a coordinate's
// rounding moves its crossing far along the ray. By `d` as the steps are (a slope `d / len` can
// be subnormal and flushed to zero); a `d` near the subnormals: no skip on its account.
fn inside_until(a: f32, b: f32, d: f32, len: f32, lo: f32, hi: f32) -> f32 {
    if (d == 0.0) {
        return 3.4e38;
    }
    if (abs(d) < 1e-30) {
        return 0.0;
    }
    let bound = select(lo, hi, d > 0.0);
    let eps = 1e-6 * (abs(a) + abs(b) + abs(bound));
    return ((bound - a) / d) * len - (eps / abs(d)) * len;
}

// the next step of a ray at step `k` in the empty box `at` of `cells` voxels per axis: two
// before it surely leaves the box
fn skipped(k: u32, a: vec3f, d: vec3f, len: f32, stop: f32, at: vec3i, cells: f32) -> u32 {
    let size = cells / P.inv;
    let lo = vec3f(at) * size;
    let hi = lo + vec3f(size);
    let b = a + d;
    let t_in = min(min(inside_until(a.x, b.x, d.x, len, lo.x, hi.x), inside_until(a.y, b.y, d.y, len, lo.y, hi.y)), inside_until(a.z, b.z, d.z, len, lo.z, hi.z));
    let j = floor(min(t_in, stop) / P.step) - 1.0;
    if (j > f32(k)) {
        return u32(j);
    }
    return k;
}

fn index(wid: vec3u, lid: vec3u, nwg: vec3u, size: u32) -> u32 {
    return (wid.y * nwg.x + wid.x) * size + lid.x;
}

@compute @workgroup_size(64)
fn walk(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let r = index(wid, lid, nwg, 64u);
    if (r >= P.rays) {
        return;
    }
    let ra = rays[2u * r];
    let a = ra.xyz;
    let bit = 1u << bitcast<u32>(ra.w);
    let d = rays[2u * r + 1u].xyz - a;
    let len = length(d);
    let stop = len - P.margin;
    let u = d / len;
    var last = vec3i(0);
    var has_last = false;
    var empty = vec3i(0);
    var has_empty = false;
    var empty_large = vec3i(0);
    var has_empty_large = false;
    var k = 1u;
    loop {
        let s = f32(k) * P.step;
        if (!(s < stop)) {
            break;
        }
        k = k + 1u;
        let c = vec3i(floor((a + d * (s / len)) * P.inv));
        if (has_last && all(c == last)) {
            continue;
        }
        last = c;
        has_last = true;
        // an empty large block, then an empty block: on to two steps before the ray surely leaves
        // it (the steps skipped are all in it: the same steps as one by one, without their
        // lookups, which would find no measured voxel there)
        // (3 + gpu_free.rs LARGE_BITS)
        let lc = c >> vec3u(5u);
        if (has_empty_large && all(lc == empty_large)) {
            continue;
        }
        if (!has_large(pack(lc))) {
            empty_large = lc;
            has_empty_large = true;
            k = skipped(k, a, d, len, stop, lc, 32.0);
            continue;
        }
        let bc = c >> vec3u(3u);
        if (has_empty && all(bc == empty)) {
            continue;
        }
        if (!has_block(pack(bc))) {
            empty = bc;
            has_empty = true;
            k = skipped(k, a, d, len, stop, bc, 8.0);
            continue;
        }
        let v = voxel(pack(c));
        if (v == NONE) {
            continue;
        }
        var ok = P.far > 0.0 && len - s >= P.far;
        if (!ok) {
            if (P.centroid == 0u) {
                ok = true;
            } else {
                // the distance of the centroid to the ray by the cross product: |m|^2 - (m.u)^2
                // loses all of a centimeter to cancellation in single precision at tens of meters
                let m = vec3f(centroids[3u * v], centroids[3u * v + 1u], centroids[3u * v + 2u]) - a;
                let q = cross(m, u);
                ok = dot(q, q) <= P.r2;
            }
        }
        if (ok) {
            atomicOr(&bits[v], bit);
        }
    }
}

@compute @workgroup_size(256)
fn count(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let v = index(wid, lid, nwg, 256u);
    if (v >= P.voxels) {
        return;
    }
    let b = atomicLoad(&bits[v]);
    if (b != 0u) {
        passes[v] = passes[v] + countOneBits(b);
        atomicStore(&bits[v], 0u);
    }
}
