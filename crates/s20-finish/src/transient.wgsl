// The passing-object cone test on the GPU (see transient.rs, gpu_transient.rs): one thread per
// pair of a scan of the batch and a point near it. Each scan's rays are sorted by the cells of a
// latitude-longitude grid (`starts`: the first ray of each cell, per scan); a ray's length is
// minus the reach for a shot without a return.

struct Params {
    cos_angle: f32,
    angle: f32,
    far: f32,
    hit: f32,
    max_range: f32,
    hit_span: f32,
    rows: u32,
    cols: u32,
    max_hits: u32,
    ranges: u32,
    pairs: u32,
    width: u32,
};

// origin.w: unused; index: the scan's number (for `times`); rays: its first ray; starts: its
// first cell start
struct Scan {
    origin: vec4<f32>,
    index: u32,
    rays: u32,
    starts: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read> times: array<f32>;
@group(0) @binding(4) var<storage, read> scans: array<Scan>;
@group(0) @binding(5) var<storage, read> starts: array<u32>;
@group(0) @binding(6) var<storage, read> rays: array<vec4<f32>>;
// (first in `order`, count, scan in the batch, first pair)
@group(0) @binding(7) var<storage, read> ranges: array<vec4<u32>>;
// per point: beyond, clean, hit, first scan, last scan, settled
@group(0) @binding(8) var<storage, read_write> counts: array<atomic<u32>>;

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

fn row_of(z: f32) -> u32 {
    return min(u32((asin(clamp(z, -1.0, 1.0)) + 0.5 * PI) / PI * f32(P.rows)), P.rows - 1u);
}
fn col_of(x: f32, y: f32) -> u32 {
    return min(u32((atan2(y, x) + PI) / TAU * f32(P.cols)), P.cols - 1u);
}

@compute @workgroup_size(64)
fn cone(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x + gid.y * P.width;
    if (tid >= P.pairs) {
        return;
    }
    // the range holding this pair
    var lo = 0u;
    var hi = P.ranges;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (ranges[mid].w <= tid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let rg = ranges[lo];
    let s = order[rg.x + (tid - rg.w)];
    let base = 6u * s;
    if (atomicLoad(&counts[base + 5u]) != 0u) {
        return;
    }
    let sc = scans[rg.z];
    let d = points[s].xyz - sc.origin.xyz;
    let r = length(d);
    if (r > P.max_range || r < 1e-3) {
        return;
    }
    let u = d / r;
    let row = row_of(u.z);
    let r0 = select(row - 1u, 0u, row == 0u);
    let r1 = min(row + 1u, P.rows - 1u);
    let lat0 = f32(r0) / f32(P.rows) * PI - 0.5 * PI;
    let lat1 = f32(r1 + 1u) / f32(P.rows) * PI - 0.5 * PI;
    let c = cos(max(abs(lat0), abs(lat1)));
    let width = TAU / f32(P.cols);
    var dc = P.cols;
    if (c >= 1e-3) {
        dc = u32(ceil(P.angle / (width * c)));
    }
    let col = col_of(u.x, u.y);
    let all = 2u * dc + 1u >= P.cols;
    var ncols = 2u * dc + 1u;
    if (all) {
        ncols = P.cols;
    }
    var beyond = false;
    var short = false;
    var seen = false;
    for (var i = r0; i <= r1; i++) {
        for (var k = 0u; k < ncols; k++) {
            var j = k;
            if (!all) {
                j = (col + P.cols + k - dc) % P.cols;
            }
            let cell = sc.starts + i * P.cols + j;
            for (var q = starts[cell]; q < starts[cell + 1u]; q++) {
                let v = rays[sc.rays + q];
                if (dot(v.xyz, u) < P.cos_angle) {
                    continue;
                }
                if (v.w < 0.0) {
                    // a shot without a return: nothing up to its gap's reach
                    beyond = beyond || -v.w > r + P.far;
                    continue;
                }
                beyond = beyond || v.w > r + P.far;
                short = short || v.w <= r - P.hit;
                seen = seen || abs(v.w - r) < P.hit;
            }
        }
    }
    if (beyond) {
        atomicAdd(&counts[base], 1u);
        if (!short) {
            atomicAdd(&counts[base + 1u], 1u);
        }
    }
    if (seen) {
        let n = atomicAdd(&counts[base + 2u], 1u) + 1u;
        let a = min(atomicMin(&counts[base + 3u], sc.index), sc.index);
        let z = max(atomicMax(&counts[base + 4u], sc.index), sc.index);
        if (n > P.max_hits || times[z] - times[a] > P.hit_span) {
            atomicStore(&counts[base + 5u], 1u);
        }
    }
}
