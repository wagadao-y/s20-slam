// SIFT on the GPU, as `sift.rs` on the CPU (OpenCV 4.x): the pyramid (doubling, separable
// Gaussian blurs with BORDER_REFLECT_101, halving, differences), the extrema refined with the
// contrast and edge tests, their orientations, and the descriptors of the keypoints kept. The
// sums run in the CPU's order; single precision as there.

struct Params {
    // an image pass: from `src` (offset, width, height) to `dst` (offset, width, height)
    src_off: u32,
    dst_off: u32,
    w: u32,
    h: u32,
    src_w: u32,
    src_h: u32,
    // the blur's kernel: offset into `kernels` and half width
    k_off: u32,
    half: u32,
    // extrema: the octave's first difference layer, the octave and the layer, the threshold
    dog_off: u32,
    octave: u32,
    layer: u32,
    threshold: f32,
    contrast: f32,
    // input pixels per pixel of octave 0
    base: f32,
    // the room for candidates, oriented keypoints
    max_out: u32,
    // the octaves' first Gaussian layers (offsets into `gauss`), widths and heights
    count: u32,
    g_off: array<vec4<u32>, 4>,
    g_w: array<vec4<u32>, 4>,
    g_h: array<vec4<u32>, 4>,
}

struct Cand {
    x: f32,
    y: f32,
    size: f32,
    response: f32,
    ox: f32,
    oy: f32,
    ix: i32,
    iy: i32,
    octave: u32,
    layer: u32,
    pad0: u32,
    pad1: u32,
}

struct Ori {
    cand: u32,
    angle: f32,
}

// a keypoint kept, for its descriptor
struct Kept {
    octave: u32,
    layer: u32,
    ox: f32,
    oy: f32,
    size: f32,
    angle: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> gauss: array<f32>;
@group(0) @binding(2) var<storage, read_write> dog: array<f32>;
@group(0) @binding(3) var<storage, read> kernels: array<f32>;
@group(0) @binding(4) var<storage, read_write> cands: array<Cand>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>, 4>;
@group(0) @binding(6) var<storage, read_write> oris: array<Ori>;
@group(0) @binding(7) var<storage, read> kept: array<Kept>;
@group(0) @binding(8) var<storage, read_write> desc: array<u32>;

const LAYERS: i32 = 3;
const SIGMA: f32 = 1.6;
const EDGE: f32 = 10.0;
const BORDER: i32 = 5;
const MAX_STEPS: i32 = 5;
const ORI_BINS: i32 = 36;
const ORI_PEAK: f32 = 0.8;
const ORI_SIG: f32 = 1.5;
const ORI_RADIUS: f32 = 4.5;
const D: i32 = 4;
const N: i32 = 8;
const DESCR_SCL: f32 = 3.0;
const DESCR_MAG: f32 = 0.2;
const DESCR_INT: f32 = 512.0;
const EPS: f32 = 1.1920929e-7;

fn reflect(i: i32, n: i32) -> i32 {
    if n == 1 {
        return 0;
    }
    var j = i;
    loop {
        if j >= 0 && j < n {
            break;
        }
        if j < 0 {
            j = -j;
        } else {
            j = 2 * n - 2 - j;
        }
    }
    return j;
}

// Rust's f32::round: halves away from zero
fn round_away(x: f32) -> f32 {
    return sign(x) * floor(abs(x) + 0.5);
}

@compute @workgroup_size(16, 16)
fn double_img(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= P.w || id.y >= P.h {
        return;
    }
    let sw = P.src_w;
    let sh = P.src_h;
    let fy = max((f32(id.y) + 0.5) * 0.5 - 0.5, 0.0);
    let y0 = min(u32(floor(fy)), sh - 1u);
    let y1 = min(y0 + 1u, sh - 1u);
    let ty = fy - f32(y0);
    let fx = max((f32(id.x) + 0.5) * 0.5 - 0.5, 0.0);
    let x0 = min(u32(floor(fx)), sw - 1u);
    let x1 = min(x0 + 1u, sw - 1u);
    let tx = fx - f32(x0);
    let s = P.src_off;
    let a = gauss[s + y0 * sw + x0] * (1.0 - tx) + gauss[s + y0 * sw + x1] * tx;
    let b = gauss[s + y1 * sw + x0] * (1.0 - tx) + gauss[s + y1 * sw + x1] * tx;
    gauss[P.dst_off + id.y * P.w + id.x] = a * (1.0 - ty) + b * ty;
}

@compute @workgroup_size(16, 16)
fn copy_img(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= P.w || id.y >= P.h {
        return;
    }
    gauss[P.dst_off + id.y * P.w + id.x] = gauss[P.src_off + id.y * P.w + id.x];
}

// rows, into `dst` (the scratch region)
@compute @workgroup_size(16, 16)
fn blur_h(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= P.w || id.y >= P.h {
        return;
    }
    let w = i32(P.w);
    let x = i32(id.x);
    let row = P.src_off + id.y * P.w;
    let half = i32(P.half);
    let k = P.k_off + P.half;
    var o = gauss[row + id.x] * kernels[k];
    for (var i = 1; i <= half; i++) {
        let a = gauss[row + u32(reflect(x - i, w))];
        let b = gauss[row + u32(reflect(x + i, w))];
        o += (a + b) * kernels[k + u32(i)];
    }
    gauss[P.dst_off + id.y * P.w + id.x] = o;
}

// columns, from the scratch region into `dst`
@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= P.w || id.y >= P.h {
        return;
    }
    let h = i32(P.h);
    let y = i32(id.y);
    let half = i32(P.half);
    let k = P.k_off + P.half;
    var o = gauss[P.src_off + id.y * P.w + id.x] * kernels[k];
    for (var i = 1; i <= half; i++) {
        let a = gauss[P.src_off + u32(reflect(y - i, h)) * P.w + id.x];
        let b = gauss[P.src_off + u32(reflect(y + i, h)) * P.w + id.x];
        o += (a + b) * kernels[k + u32(i)];
    }
    gauss[P.dst_off + id.y * P.w + id.x] = o;
}

@compute @workgroup_size(16, 16)
fn halve(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= P.w || id.y >= P.h {
        return;
    }
    gauss[P.dst_off + id.y * P.w + id.x] = gauss[P.src_off + (2u * id.y) * P.src_w + 2u * id.x];
}

// layer i + 1 less layer i, at `src_off` (layer i) into `dst_off` of `dog`
@compute @workgroup_size(16, 16)
fn difference(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= P.w || id.y >= P.h {
        return;
    }
    let n = P.w * P.h;
    let k = id.y * P.w + id.x;
    dog[P.dst_off + k] = gauss[P.src_off + n + k] - gauss[P.src_off + k];
}

fn at(layer: i32, x: i32, y: i32) -> f32 {
    return dog[P.dog_off + u32(layer) * P.w * P.h + u32(y) * P.w + u32(x)];
}

// the 3x3 system h s = g (Cramer's rule, without arrays: the shader compiler unrolls their
// writes); none (w = 0) when singular
fn solve3(hm: mat3x3<f32>, g: vec3<f32>) -> vec4<f32> {
    let det = determinant(hm);
    if det == 0.0 {
        return vec4<f32>(0.0);
    }
    let sx = determinant(mat3x3<f32>(g, hm[1], hm[2]));
    let sy = determinant(mat3x3<f32>(hm[0], g, hm[2]));
    let sz = determinant(mat3x3<f32>(hm[0], hm[1], g));
    return vec4<f32>(sx / det, sy / det, sz / det, 1.0);
}

// the octave's layer `P.layer`: the extrema, refined (OpenCV's adjustLocalExtrema)
@compute @workgroup_size(16, 16)
fn extrema(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = i32(P.w);
    let h = i32(P.h);
    let x0 = i32(id.x);
    let y0 = i32(id.y);
    if w <= 2 * BORDER || h <= 2 * BORDER || x0 < BORDER || x0 >= w - BORDER || y0 < BORDER
        || y0 >= h - BORDER {
        return;
    }
    let l0 = i32(P.layer);
    let v = at(l0, x0, y0);
    if abs(v) <= P.threshold {
        return;
    }
    var ext = true;
    for (var dl = -1; dl <= 1; dl++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if dl == 0 && dx == 0 && dy == 0 {
                    continue;
                }
                let q = at(l0 + dl, x0 + dx, y0 + dy);
                if (v > 0.0 && v < q) || (v <= 0.0 && v > q) {
                    ext = false;
                }
            }
        }
    }
    if !ext {
        return;
    }
    let img_scale = 1.0 / 255.0;
    let ds = img_scale * 0.5;
    let d2s = img_scale;
    let dxs = img_scale * 0.25;
    var layer = l0;
    var x = x0;
    var y = y0;
    var sol = vec3<f32>(0.0);
    var converged = false;
    for (var step = 0; step < MAX_STEPS; step++) {
        let v2 = at(layer, x, y) * 2.0;
        let grad = vec3<f32>(
            (at(layer, x + 1, y) - at(layer, x - 1, y)) * ds,
            (at(layer, x, y + 1) - at(layer, x, y - 1)) * ds,
            (at(layer + 1, x, y) - at(layer - 1, x, y)) * ds,
        );
        let dxx = (at(layer, x + 1, y) + at(layer, x - 1, y) - v2) * d2s;
        let dyy = (at(layer, x, y + 1) + at(layer, x, y - 1) - v2) * d2s;
        let dss = (at(layer + 1, x, y) + at(layer - 1, x, y) - v2) * d2s;
        let dxy = (at(layer, x + 1, y + 1) - at(layer, x - 1, y + 1) - at(layer, x + 1, y - 1)
            + at(layer, x - 1, y - 1)) * dxs;
        let dxs_ = (at(layer + 1, x + 1, y) - at(layer + 1, x - 1, y) - at(layer - 1, x + 1, y)
            + at(layer - 1, x - 1, y)) * dxs;
        let dys = (at(layer + 1, x, y + 1) - at(layer + 1, x, y - 1) - at(layer - 1, x, y + 1)
            + at(layer - 1, x, y - 1)) * dxs;
        let hm = mat3x3<f32>(
            vec3<f32>(dxx, dxy, dxs_),
            vec3<f32>(dxy, dyy, dys),
            vec3<f32>(dxs_, dys, dss),
        );
        let s = solve3(hm, grad);
        if s.w == 0.0 {
            return;
        }
        sol = -s.xyz;
        if abs(sol.x) < 0.5 && abs(sol.y) < 0.5 && abs(sol.z) < 0.5 {
            converged = true;
            break;
        }
        let big = f32(715827882);
        if abs(sol.x) > big || abs(sol.y) > big || abs(sol.z) > big {
            return;
        }
        x += i32(round_away(sol.x));
        y += i32(round_away(sol.y));
        layer += i32(round_away(sol.z));
        if layer < 1 || layer > LAYERS || x < BORDER || x >= w - BORDER || y < BORDER
            || y >= h - BORDER {
            return;
        }
    }
    if !converged {
        return;
    }
    let grad = vec3<f32>(
        (at(layer, x + 1, y) - at(layer, x - 1, y)) * ds,
        (at(layer, x, y + 1) - at(layer, x, y - 1)) * ds,
        (at(layer + 1, x, y) - at(layer - 1, x, y)) * ds,
    );
    let contr = at(layer, x, y) * img_scale + 0.5 * (grad.x * sol.x + grad.y * sol.y + grad.z * sol.z);
    if abs(contr) * f32(LAYERS) < P.contrast {
        return;
    }
    let v2 = at(layer, x, y) * 2.0;
    let dxx = (at(layer, x + 1, y) + at(layer, x - 1, y) - v2) * d2s;
    let dyy = (at(layer, x, y + 1) + at(layer, x, y - 1) - v2) * d2s;
    let dxy = (at(layer, x + 1, y + 1) - at(layer, x - 1, y + 1) - at(layer, x + 1, y - 1)
        + at(layer, x - 1, y - 1)) * dxs;
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    if det <= 0.0 || tr * tr * EDGE >= (EDGE + 1.0) * (EDGE + 1.0) * det {
        return;
    }
    let scale = f32(1u << P.octave) * P.base;
    let ox = f32(x) + sol.x;
    let oy = f32(y) + sol.y;
    let i = atomicAdd(&counters[0], 1u);
    if i >= P.max_out {
        return;
    }
    cands[i] = Cand(
        ox * scale,
        oy * scale,
        SIGMA * pow(2.0, (f32(layer) + sol.z) / f32(LAYERS)) * 2.0 * scale,
        abs(contr),
        ox,
        oy,
        x,
        y,
        P.octave,
        u32(layer),
        0u,
        0u,
    );
}

// OpenCV's fastAtan2 (degrees in [0, 360))
fn fast_atan2_deg(y: f32, x: f32) -> f32 {
    let r = 180.0 / 3.14159265358979;
    let p1 = 0.99978784 * r;
    let p3 = -0.3258084 * r;
    let p5 = 0.15557865 * r;
    let p7 = -0.044326556 * r;
    let ax = abs(x);
    let ay = abs(y);
    var a: f32;
    if ax >= ay {
        let c = ay / (ax + EPS);
        let c2 = c * c;
        a = (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c;
    } else {
        let c = ax / (ay + EPS);
        let c2 = c * c;
        a = 90.0 - (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c;
    }
    if x < 0.0 {
        a = 180.0 - a;
    }
    if y < 0.0 {
        a = 360.0 - a;
    }
    return a;
}

fn gpx(octave: u32, layer: u32, x: i32, y: i32) -> f32 {
    let o = octave / 4u;
    let c = octave % 4u;
    let w = P.g_w[o][c];
    let h = P.g_h[o][c];
    return gauss[P.g_off[o][c] + layer * w * h + u32(y) * w + u32(x)];
}

fn gsize(octave: u32) -> vec2<i32> {
    let o = octave / 4u;
    let c = octave % 4u;
    return vec2<i32>(i32(P.g_w[o][c]), i32(P.g_h[o][c]));
}

// a candidate's dominant orientations (OpenCV's calcOrientationHist and the peak search)
@compute @workgroup_size(64)
fn orient(@builtin(global_invocation_id) id: vec3<u32>) {
    // the candidates found (dispatched for as many as there is room for)
    if id.x >= min(atomicLoad(&counters[0]), P.max_out) {
        return;
    }
    let kp = cands[id.x];
    let base = P.base;
    let scl = kp.size * 0.5 / (base * f32(1u << kp.octave));
    let radius = i32(floor(ORI_RADIUS * scl + 0.5));
    let sigma = ORI_SIG * scl;
    let expf = -1.0 / (2.0 * sigma * sigma);
    let sz = gsize(kp.octave);
    var hist: array<f32, 36>;
    for (var b = 0; b < ORI_BINS; b++) {
        hist[b] = 0.0;
    }
    for (var i = -radius; i <= radius; i++) {
        let y = kp.iy + i;
        if y <= 0 || y >= sz.y - 1 {
            continue;
        }
        for (var j = -radius; j <= radius; j++) {
            let x = kp.ix + j;
            if x <= 0 || x >= sz.x - 1 {
                continue;
            }
            let dx = gpx(kp.octave, kp.layer, x + 1, y) - gpx(kp.octave, kp.layer, x - 1, y);
            let dy = gpx(kp.octave, kp.layer, x, y - 1) - gpx(kp.octave, kp.layer, x, y + 1);
            let wgt = exp(f32(i * i) * expf) * exp(f32(j * j) * expf);
            let ori = fast_atan2_deg(dy, dx);
            let mag = sqrt(dx * dx + dy * dy);
            var bin = i32(floor(f32(ORI_BINS) / 360.0 * ori + 0.5));
            if bin >= ORI_BINS {
                bin -= ORI_BINS;
            }
            if bin < 0 {
                bin += ORI_BINS;
            }
            hist[bin] += wgt * mag;
        }
    }
    var smoothed: array<f32, 36>;
    var mx = 0.0;
    for (var b = 0; b < ORI_BINS; b++) {
        let g0 = hist[(b + ORI_BINS - 2) % ORI_BINS];
        let g1 = hist[(b + ORI_BINS - 1) % ORI_BINS];
        let g3 = hist[(b + 1) % ORI_BINS];
        let g4 = hist[(b + 2) % ORI_BINS];
        smoothed[b] = (g0 + g4) * (1.0 / 16.0) + (g1 + g3) * (4.0 / 16.0) + hist[b] * (6.0 / 16.0);
        mx = max(mx, smoothed[b]);
    }
    let thr = mx * ORI_PEAK;
    for (var j = 0; j < ORI_BINS; j++) {
        let l = smoothed[(j + ORI_BINS - 1) % ORI_BINS];
        let r2 = smoothed[(j + 1) % ORI_BINS];
        let c = smoothed[j];
        if c > l && c > r2 && c >= thr {
            var bin = f32(j) + 0.5 * (l - r2) / (l - 2.0 * c + r2);
            if bin < 0.0 {
                bin += f32(ORI_BINS);
            } else if bin >= f32(ORI_BINS) {
                bin -= f32(ORI_BINS);
            }
            var angle = 360.0 - (360.0 / f32(ORI_BINS)) * bin;
            if abs(angle - 360.0) < EPS {
                angle = 0.0;
            }
            let k = atomicAdd(&counters[1], 1u);
            if k < P.max_out {
                oris[k] = Ori(id.x, angle);
            }
        }
    }
}

// a kept keypoint's 128 bytes (OpenCV's calcSIFTDescriptor), four to a word: a workgroup a
// keypoint, its threads taking the samples in turn and adding to the histogram in fixed point
// (integers: the same sums in any order), the first thread normalizing it
var<workgroup> whist: array<atomic<u32>, 360>;

@compute @workgroup_size(64)
fn describe(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let key = wid.y * nwg.x + wid.x;
    if key >= P.count {
        return;
    }
    for (var k = lid; k < 360u; k += 64u) {
        atomicStore(&whist[k], 0u);
    }
    workgroupBarrier();
    let kp = kept[key];
    let sz = gsize(kp.octave);
    let scl = kp.size * 0.5 / (P.base * f32(1u << kp.octave));
    var ori = 360.0 - kp.angle;
    if abs(ori - 360.0) < EPS {
        ori = 0.0;
    }
    let px = i32(floor(kp.ox + 0.5));
    let py = i32(floor(kp.oy + 0.5));
    let rad = radians(ori);
    var cos_t = cos(rad);
    var sin_t = sin(rad);
    let bins_per_deg = f32(N) / 360.0;
    let exp_scale = -1.0 / (f32(D) * f32(D) * 0.5);
    let hist_width = DESCR_SCL * scl;
    // (squared in floating point: a long side's square overflows i32)
    let diag = i32(sqrt(f32(sz.x) * f32(sz.x) + f32(sz.y) * f32(sz.y)));
    let radius = min(i32(floor(hist_width * 1.41421356 * (f32(D) + 1.0) * 0.5 + 0.5)), diag);
    cos_t /= hist_width;
    sin_t /= hist_width;
    let e = exp_scale / (hist_width * hist_width);
    let side = 2 * radius + 1;
    let samples = u32(side * side);
    // the fixed point: a sample adds its magnitude (at most 255 * sqrt 2 on 0..255 images) over
    // eight bins, so no bin's sum passes 2^32 - 1
    let fixed = 4.0e9 / (f32(samples) * 362.0);
    for (var s = lid; s < samples; s += 64u) {
        let i = i32(s) / side - radius;
        let j = i32(s) % side - radius;
        let x_rot = f32(j) * cos_t - f32(i) * sin_t;
        let y_rot = f32(j) * sin_t + f32(i) * cos_t;
        let rbin = y_rot + f32(D) / 2.0 - 0.5;
        let cbin = x_rot + f32(D) / 2.0 - 0.5;
        let r = py + i;
        let c = px + j;
        if !(rbin > -1.0 && rbin < f32(D) && cbin > -1.0 && cbin < f32(D) && r > 0
            && r < sz.y - 1 && c > 0 && c < sz.x - 1) {
            continue;
        }
        let dx = gpx(kp.octave, kp.layer, c + 1, r) - gpx(kp.octave, kp.layer, c - 1, r);
        let dy = gpx(kp.octave, kp.layer, c, r - 1) - gpx(kp.octave, kp.layer, c, r + 1);
        let wgt = exp(f32(i * i) * e) * exp(f32(j * j) * e);
        let mag = sqrt(dx * dx + dy * dy) * wgt;
        let obin = (fast_atan2_deg(dy, dx) - ori) * bins_per_deg;
        let r0 = floor(rbin);
        let c0 = floor(cbin);
        var o0 = floor(obin);
        let rb = rbin - r0;
        let cb = cbin - c0;
        let ob = obin - o0;
        if o0 < 0.0 {
            o0 += f32(N);
        }
        if o0 >= f32(N) {
            o0 -= f32(N);
        }
        let v_r1 = mag * rb;
        let v_r0 = mag - v_r1;
        let v_rc11 = v_r1 * cb;
        let v_rc10 = v_r1 - v_rc11;
        let v_rc01 = v_r0 * cb;
        let v_rc00 = v_r0 - v_rc01;
        let v_rco111 = v_rc11 * ob;
        let v_rco110 = v_rc11 - v_rco111;
        let v_rco101 = v_rc10 * ob;
        let v_rco100 = v_rc10 - v_rco101;
        let v_rco011 = v_rc01 * ob;
        let v_rco010 = v_rc01 - v_rco011;
        let v_rco001 = v_rc00 * ob;
        let v_rco000 = v_rc00 - v_rco001;
        let idx = u32(((i32(r0) + 1) * (D + 2) + (i32(c0) + 1)) * (N + 2) + i32(o0));
        let d1 = u32(D + 2) * u32(N + 2);
        let d2 = u32(D + 3) * u32(N + 2);
        atomicAdd(&whist[idx], u32(max(v_rco000, 0.0) * fixed + 0.5));
        atomicAdd(&whist[idx + 1u], u32(max(v_rco001, 0.0) * fixed + 0.5));
        atomicAdd(&whist[idx + u32(N + 2)], u32(max(v_rco010, 0.0) * fixed + 0.5));
        atomicAdd(&whist[idx + u32(N + 3)], u32(max(v_rco011, 0.0) * fixed + 0.5));
        atomicAdd(&whist[idx + d1], u32(max(v_rco100, 0.0) * fixed + 0.5));
        atomicAdd(&whist[idx + d1 + 1u], u32(max(v_rco101, 0.0) * fixed + 0.5));
        atomicAdd(&whist[idx + d2], u32(max(v_rco110, 0.0) * fixed + 0.5));
        atomicAdd(&whist[idx + d2 + 1u], u32(max(v_rco111, 0.0) * fixed + 0.5));
    }
    workgroupBarrier();
    if lid != 0u {
        return;
    }
    var raw: array<f32, 128>;
    for (var i = 0; i < D; i++) {
        for (var j = 0; j < D; j++) {
            let idx = u32(((i + 1) * (D + 2) + (j + 1)) * (N + 2));
            // (the bins past the last orientation wrap onto the first two)
            for (var k = 0; k < N; k++) {
                var v = f32(atomicLoad(&whist[idx + u32(k)]));
                if k < 2 {
                    v += f32(atomicLoad(&whist[idx + u32(N + k)]));
                }
                raw[(i * D + j) * N + k] = v / fixed;
            }
        }
    }
    var nrm = 0.0;
    for (var k = 0; k < 128; k++) {
        nrm += raw[k] * raw[k];
    }
    let thr = sqrt(nrm) * DESCR_MAG;
    var nrm2 = 0.0;
    for (var k = 0; k < 128; k++) {
        raw[k] = min(raw[k], thr);
        nrm2 += raw[k] * raw[k];
    }
    let f = DESCR_INT / max(sqrt(nrm2), EPS);
    for (var k = 0; k < 32; k++) {
        var word = 0u;
        for (var b = 0; b < 4; b++) {
            let v = u32(clamp(floor(raw[k * 4 + b] * f + 0.5), 0.0, 255.0));
            word |= v << (8u * u32(b));
        }
        desc[key * 32u + u32(k)] = word;
    }
}
