//! SIFT keypoints and descriptors, following OpenCV's implementation (4.x `sift.simd.hpp`): the
//! image doubled first, 3 layers per octave from sigma 1.6, contrast threshold 0.04, edge ratio 10,
//! 36-bin orientations with peaks above 80 % of the highest, 4x4x8 descriptors clipped at 0.2 and
//! stored as bytes (x512).

use crate::image::Gray;
use rayon::prelude::*;

const LAYERS: usize = 3;
const SIGMA: f32 = 1.6;
const INIT_SIGMA: f32 = 0.5;
const CONTRAST: f32 = 0.04;
const EDGE: f32 = 10.;
const BORDER: usize = 5;
const MAX_STEPS: usize = 5;
const ORI_BINS: usize = 36;
const ORI_PEAK: f32 = 0.8;
const ORI_SIG: f32 = 1.5;
const ORI_RADIUS: f32 = 3. * ORI_SIG;
const D: usize = 4;
const N: usize = 8;
const DESCR_SCL: f32 = 3.;
const DESCR_MAG: f32 = 0.2;
const DESCR_INT: f32 = 512.;

#[derive(Clone, Debug)]
pub struct Keypoint {
    /// pixel position in the input image
    pub x: f32,
    pub y: f32,
    /// diameter in input pixels, as OpenCV's `size`
    pub size: f32,
    /// degrees, OpenCV's convention
    pub angle: f32,
    pub response: f32,
    /// octave of the doubled image (0 = the doubled image), layer, and the position in that octave
    octave: usize,
    layer: usize,
    ox: f32,
    oy: f32,
    /// the refined integer position in that octave (the orientation histogram's centre)
    ix: isize,
    iy: isize,
}

/// Separable Gaussian blur, OpenCV's kernel size for float images (8 sigma + 1) and border
/// reflection without repeating the edge (BORDER_REFLECT_101).
fn blur(src: &Gray, sigma: f32) -> Gray {
    let half = ((sigma * 8. + 1.).round() as usize | 1) / 2;
    let mut k: Vec<f32> = (0..=2 * half)
        .map(|i| (-((i as f32 - half as f32).powi(2)) / (2. * sigma * sigma)).exp())
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    let reflect = |i: isize, n: usize| -> usize {
        let n = n as isize;
        let mut i = i;
        if n == 1 {
            return 0;
        }
        while i < 0 || i >= n {
            i = if i < 0 { -i } else { 2 * n - 2 - i };
        }
        i as usize
    };
    let (w, h) = (src.w, src.h);
    // rows padded by reflection, then a branch-free symmetric convolution
    let mut tmp = Gray::new(w, h);
    tmp.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let line = &src.px[y * w..(y + 1) * w];
        let padded: Vec<f32> = (0..w + 2 * half)
            .map(|i| line[reflect(i as isize - half as isize, w)])
            .collect();
        let centre = k[half];
        for (o, v) in row.iter_mut().zip(&padded[half..half + w]) {
            *o = v * centre;
        }
        // tap by tap over the whole row, so the inner loop runs over contiguous pixels
        for i in 1..=half {
            let (lo, hi, kv) = (
                &padded[half - i..half - i + w],
                &padded[half + i..half + i + w],
                k[half + i],
            );
            for ((o, a), b) in row.iter_mut().zip(lo).zip(hi) {
                *o += (a + b) * kv;
            }
        }
    });
    let mut out = Gray::new(w, h);
    out.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let line = |d: isize| {
            let yy = reflect(y as isize + d, h);
            &tmp.px[yy * w..(yy + 1) * w]
        };
        for (o, v) in row.iter_mut().zip(line(0)) {
            *o = v * k[half];
        }
        for i in 1..=half as isize {
            let (up, down, kv) = (line(-i), line(i), k[half + i as usize]);
            for ((o, a), b) in row.iter_mut().zip(up).zip(down) {
                *o += (a + b) * kv;
            }
        }
    });
    out
}

/// Bilinear doubling (OpenCV INTER_LINEAR with half-pixel centres).
fn double(src: &Gray) -> Gray {
    let (w, h) = (src.w * 2, src.h * 2);
    let mut out = Gray::new(w, h);
    out.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = ((y as f32 + 0.5) * 0.5 - 0.5).max(0.);
        let y0 = (fy.floor() as usize).min(src.h - 1);
        let y1 = (y0 + 1).min(src.h - 1);
        let ty = fy - y0 as f32;
        for (x, o) in row.iter_mut().enumerate() {
            let fx = ((x as f32 + 0.5) * 0.5 - 0.5).max(0.);
            let x0 = (fx.floor() as usize).min(src.w - 1);
            let x1 = (x0 + 1).min(src.w - 1);
            let tx = fx - x0 as f32;
            let a = src.at(x0, y0) * (1. - tx) + src.at(x1, y0) * tx;
            let b = src.at(x0, y1) * (1. - tx) + src.at(x1, y1) * tx;
            *o = a * (1. - ty) + b * ty;
        }
    });
    out
}

fn halve(src: &Gray) -> Gray {
    let (w, h) = (src.w / 2, src.h / 2);
    let mut out = Gray::new(w, h);
    out.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = src.at(2 * x, 2 * y);
        }
    });
    out
}

struct Pyramid {
    gauss: Vec<Vec<Gray>>,
    dog: Vec<Vec<Gray>>,
}

fn pyramid(img: &Gray) -> Pyramid {
    let base = double(img);
    let diff = (SIGMA * SIGMA - (2. * INIT_SIGMA).powi(2)).max(0.01).sqrt();
    let base = blur(&base, diff);
    let octaves = ((base.w.min(base.h) as f32).log2().round() as isize - 2).max(1) as usize;
    let k = 2f32.powf(1. / LAYERS as f32);
    let sig: Vec<f32> = (0..LAYERS + 3)
        .map(|i| {
            if i == 0 {
                SIGMA
            } else {
                let prev = k.powi(i as i32 - 1) * SIGMA;
                ((prev * k).powi(2) - prev * prev).sqrt()
            }
        })
        .collect();
    let mut gauss: Vec<Vec<Gray>> = Vec::with_capacity(octaves);
    for o in 0..octaves {
        let mut layers = Vec::with_capacity(LAYERS + 3);
        layers.push(if o == 0 {
            base.clone()
        } else {
            halve(&gauss[o - 1][LAYERS])
        });
        for i in 1..LAYERS + 3 {
            let next = blur(&layers[i - 1], sig[i]);
            layers.push(next);
        }
        gauss.push(layers);
    }
    let dog = gauss
        .iter()
        .map(|layers| {
            (0..LAYERS + 2)
                .map(|i| Gray {
                    w: layers[i].w,
                    h: layers[i].h,
                    px: layers[i + 1]
                        .px
                        .iter()
                        .zip(&layers[i].px)
                        .map(|(a, b)| a - b)
                        .collect(),
                })
                .collect()
        })
        .collect();
    Pyramid { gauss, dog }
}

/// Sub-pixel refinement and the contrast and edge tests (OpenCV's adjustLocalExtrema).
fn refine(dog: &[Gray], octave: usize, layer0: usize, x0: usize, y0: usize) -> Option<Keypoint> {
    let img_scale = 1. / 255.;
    let (ds, d2s, dxs) = (img_scale * 0.5, img_scale, img_scale * 0.25);
    let (mut layer, mut x, mut y) = (layer0 as isize, x0 as isize, y0 as isize);
    let (w, h) = (dog[0].w as isize, dog[0].h as isize);
    let mut sol = [0f32; 3];
    let mut converged = false;
    for _ in 0..MAX_STEPS {
        let (p, c, n) = (
            &dog[layer as usize - 1],
            &dog[layer as usize],
            &dog[layer as usize + 1],
        );
        let at = |g: &Gray, xx: isize, yy: isize| g.at(xx as usize, yy as usize);
        let v2 = at(c, x, y) * 2.;
        let grad = [
            (at(c, x + 1, y) - at(c, x - 1, y)) * ds,
            (at(c, x, y + 1) - at(c, x, y - 1)) * ds,
            (at(n, x, y) - at(p, x, y)) * ds,
        ];
        let dxx = (at(c, x + 1, y) + at(c, x - 1, y) - v2) * d2s;
        let dyy = (at(c, x, y + 1) + at(c, x, y - 1) - v2) * d2s;
        let dss = (at(n, x, y) + at(p, x, y) - v2) * d2s;
        let dxy = (at(c, x + 1, y + 1) - at(c, x - 1, y + 1) - at(c, x + 1, y - 1)
            + at(c, x - 1, y - 1))
            * dxs;
        let dxs_ = (at(n, x + 1, y) - at(n, x - 1, y) - at(p, x + 1, y) + at(p, x - 1, y)) * dxs;
        let dys = (at(n, x, y + 1) - at(n, x, y - 1) - at(p, x, y + 1) + at(p, x, y - 1)) * dxs;
        let hess = nalgebra::Matrix3::new(
            dxx as f64,
            dxy as f64,
            dxs_ as f64,
            dxy as f64,
            dyy as f64,
            dys as f64,
            dxs_ as f64,
            dys as f64,
            dss as f64,
        );
        let g = nalgebra::Vector3::new(grad[0] as f64, grad[1] as f64, grad[2] as f64);
        let s = hess.lu().solve(&g)?;
        sol = [-s[0] as f32, -s[1] as f32, -s[2] as f32];
        if sol.iter().all(|v| v.abs() < 0.5) {
            converged = true;
            break;
        }
        if sol.iter().any(|v| v.abs() > (i32::MAX / 3) as f32) {
            return None;
        }
        x += sol[0].round() as isize;
        y += sol[1].round() as isize;
        layer += sol[2].round() as isize;
        if layer < 1
            || layer > LAYERS as isize
            || x < BORDER as isize
            || x >= w - BORDER as isize
            || y < BORDER as isize
            || y >= h - BORDER as isize
        {
            return None;
        }
    }
    if !converged {
        return None;
    }
    let (p, c, n) = (
        &dog[layer as usize - 1],
        &dog[layer as usize],
        &dog[layer as usize + 1],
    );
    let at = |g: &Gray, xx: isize, yy: isize| g.at(xx as usize, yy as usize);
    let grad = [
        (at(c, x + 1, y) - at(c, x - 1, y)) * ds,
        (at(c, x, y + 1) - at(c, x, y - 1)) * ds,
        (at(n, x, y) - at(p, x, y)) * ds,
    ];
    let contr =
        at(c, x, y) * img_scale + 0.5 * (grad[0] * sol[0] + grad[1] * sol[1] + grad[2] * sol[2]);
    if contr.abs() * (LAYERS as f32) < CONTRAST {
        return None;
    }
    let v2 = at(c, x, y) * 2.;
    let dxx = (at(c, x + 1, y) + at(c, x - 1, y) - v2) * d2s;
    let dyy = (at(c, x, y + 1) + at(c, x, y - 1) - v2) * d2s;
    let dxy = (at(c, x + 1, y + 1) - at(c, x - 1, y + 1) - at(c, x + 1, y - 1)
        + at(c, x - 1, y - 1))
        * dxs;
    let (tr, det) = (dxx + dyy, dxx * dyy - dxy * dxy);
    if det <= 0. || tr * tr * EDGE >= (EDGE + 1.).powi(2) * det {
        return None;
    }
    let scale = (1 << octave) as f32 * 0.5; // octave 0 is the doubled image
    let (ox, oy) = (x as f32 + sol[0], y as f32 + sol[1]);
    Some(Keypoint {
        x: ox * scale,
        y: oy * scale,
        size: SIGMA
            * 2f32.powf((layer as f32 + sol[2]) / LAYERS as f32)
            * (1 << octave) as f32
            * 2.
            * 0.5,
        angle: 0.,
        response: contr.abs(),
        octave,
        layer: layer as usize,
        ox,
        oy,
        ix: x,
        iy: y,
    })
}

/// OpenCV's fastAtan2: degrees in [0, 360) from a 7th-order polynomial (about 0.01 degrees off).
fn fast_atan2_deg(y: f32, x: f32) -> f32 {
    const P1: f32 = 0.999_787_84 * (180. / std::f32::consts::PI);
    const P3: f32 = -0.325_808_4 * (180. / std::f32::consts::PI);
    const P5: f32 = 0.155_578_65 * (180. / std::f32::consts::PI);
    const P7: f32 = -0.044_326_556 * (180. / std::f32::consts::PI);
    let (ax, ay) = (x.abs(), y.abs());
    let mut a = if ax >= ay {
        let c = ay / (ax + f32::EPSILON);
        let c2 = c * c;
        (((P7 * c2 + P5) * c2 + P3) * c2 + P1) * c
    } else {
        let c = ax / (ay + f32::EPSILON);
        let c2 = c * c;
        90. - (((P7 * c2 + P5) * c2 + P3) * c2 + P1) * c
    };
    if x < 0. {
        a = 180. - a;
    }
    if y < 0. {
        a = 360. - a;
    }
    a
}

/// exp(k^2 * s) for k = 0..=n: a Gaussian weight over (i, j) is the product of two of these.
fn gauss_table(n: usize, s: f32) -> Vec<f32> {
    (0..=n).map(|k| ((k * k) as f32 * s).exp()).collect()
}

/// Dominant orientations of a keypoint (OpenCV's calcOrientationHist and the peak search).
fn orientations(img: &Gray, kp: &Keypoint) -> Vec<f32> {
    let scl = kp.size * 2. * 0.5 / (1 << kp.octave) as f32; // size on the octave, before the doubling unpack
    let (px, py) = (kp.ix, kp.iy);
    let radius = (ORI_RADIUS * scl).round() as isize;
    let sigma = ORI_SIG * scl;
    let expf = -1. / (2. * sigma * sigma);
    let wt = gauss_table(radius.max(0) as usize, expf);
    let mut hist = [0f32; ORI_BINS];
    for i in -radius..=radius {
        let y = py + i;
        if y <= 0 || y >= img.h as isize - 1 {
            continue;
        }
        for j in -radius..=radius {
            let x = px + j;
            if x <= 0 || x >= img.w as isize - 1 {
                continue;
            }
            let (xu, yu) = (x as usize, y as usize);
            let dx = img.at(xu + 1, yu) - img.at(xu - 1, yu);
            let dy = img.at(xu, yu - 1) - img.at(xu, yu + 1);
            let w = wt[i.unsigned_abs()] * wt[j.unsigned_abs()];
            let ori = fast_atan2_deg(dy, dx);
            let mag = (dx * dx + dy * dy).sqrt();
            let mut bin = ((ORI_BINS as f32 / 360.) * ori).round() as isize;
            if bin >= ORI_BINS as isize {
                bin -= ORI_BINS as isize;
            }
            if bin < 0 {
                bin += ORI_BINS as isize;
            }
            hist[bin as usize] += w * mag;
        }
    }
    let n = ORI_BINS;
    let g = |i: isize| hist[((i % n as isize + n as isize) % n as isize) as usize];
    let smooth: Vec<f32> = (0..n as isize)
        .map(|i| {
            (g(i - 2) + g(i + 2)) * (1. / 16.)
                + (g(i - 1) + g(i + 1)) * (4. / 16.)
                + g(i) * (6. / 16.)
        })
        .collect();
    let max = smooth.iter().cloned().fold(0f32, f32::max);
    let thr = max * ORI_PEAK;
    let mut out = Vec::new();
    for j in 0..n {
        let l = smooth[(j + n - 1) % n];
        let r2 = smooth[(j + 1) % n];
        let c = smooth[j];
        if c > l && c > r2 && c >= thr {
            let mut bin = j as f32 + 0.5 * (l - r2) / (l - 2. * c + r2);
            if bin < 0. {
                bin += n as f32;
            } else if bin >= n as f32 {
                bin -= n as f32;
            }
            let mut angle = 360. - (360. / n as f32) * bin;
            if (angle - 360.).abs() < f32::EPSILON {
                angle = 0.;
            }
            out.push(angle);
        }
    }
    out
}

/// The 128-byte descriptor (OpenCV's calcSIFTDescriptor).
fn descriptor(img: &Gray, kp: &Keypoint) -> [u8; 128] {
    let scl = kp.size / (1 << kp.octave) as f32;
    let mut ori = 360. - kp.angle;
    if (ori - 360.).abs() < f32::EPSILON {
        ori = 0.;
    }
    let (px, py) = (kp.ox.round() as isize, kp.oy.round() as isize);
    let (mut cos_t, mut sin_t) = (ori.to_radians().cos(), ori.to_radians().sin());
    let bins_per_deg = N as f32 / 360.;
    let exp_scale = -1. / (D as f32 * D as f32 * 0.5);
    let hist_width = DESCR_SCL * scl;
    let radius = ((hist_width * std::f32::consts::SQRT_2 * (D as f32 + 1.) * 0.5).round() as isize)
        .min(((img.w * img.w + img.h * img.h) as f64).sqrt() as isize);
    cos_t /= hist_width;
    sin_t /= hist_width;
    // the rotation keeps the distance, so the weight separates into rows and columns
    let wt = gauss_table(
        radius.max(0) as usize,
        exp_scale / (hist_width * hist_width),
    );
    let mut hist = vec![0f32; (D + 2) * (D + 2) * (N + 2)];
    for i in -radius..=radius {
        for j in -radius..=radius {
            let x_rot = j as f32 * cos_t - i as f32 * sin_t;
            let y_rot = j as f32 * sin_t + i as f32 * cos_t;
            let rbin = y_rot + D as f32 / 2. - 0.5;
            let cbin = x_rot + D as f32 / 2. - 0.5;
            let (r, c) = (py + i, px + j);
            if !(rbin > -1.
                && rbin < D as f32
                && cbin > -1.
                && cbin < D as f32
                && r > 0
                && r < img.h as isize - 1
                && c > 0
                && c < img.w as isize - 1)
            {
                continue;
            }
            let (ru, cu) = (r as usize, c as usize);
            let dx = img.at(cu + 1, ru) - img.at(cu - 1, ru);
            let dy = img.at(cu, ru - 1) - img.at(cu, ru + 1);
            let w = wt[i.unsigned_abs()] * wt[j.unsigned_abs()];
            let mag = (dx * dx + dy * dy).sqrt() * w;
            let obin = (fast_atan2_deg(dy, dx) - ori) * bins_per_deg;
            let (r0, c0, mut o0) = (rbin.floor(), cbin.floor(), obin.floor());
            let (rb, cb, ob) = (rbin - r0, cbin - c0, obin - o0);
            if o0 < 0. {
                o0 += N as f32;
            }
            if o0 >= N as f32 {
                o0 -= N as f32;
            }
            let (r0, c0, o0) = (r0 as isize, c0 as isize, o0 as usize);
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
            let idx = (((r0 + 1) as usize * (D + 2) + (c0 + 1) as usize) * (N + 2)) + o0;
            hist[idx] += v_rco000;
            hist[idx + 1] += v_rco001;
            hist[idx + (N + 2)] += v_rco010;
            hist[idx + (N + 3)] += v_rco011;
            hist[idx + (D + 2) * (N + 2)] += v_rco100;
            hist[idx + (D + 2) * (N + 2) + 1] += v_rco101;
            hist[idx + (D + 3) * (N + 2)] += v_rco110;
            hist[idx + (D + 3) * (N + 2) + 1] += v_rco111;
        }
    }
    let mut raw = [0f32; 128];
    for i in 0..D {
        for j in 0..D {
            let idx = ((i + 1) * (D + 2) + (j + 1)) * (N + 2);
            hist[idx] += hist[idx + N];
            hist[idx + 1] += hist[idx + N + 1];
            for k in 0..N {
                raw[(i * D + j) * N + k] = hist[idx + k];
            }
        }
    }
    let nrm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    let thr = nrm * DESCR_MAG;
    let mut nrm2 = 0f32;
    for v in raw.iter_mut() {
        *v = v.min(thr);
        nrm2 += *v * *v;
    }
    let nrm2 = DESCR_INT / nrm2.sqrt().max(f32::EPSILON);
    let mut out = [0u8; 128];
    for (o, v) in out.iter_mut().zip(raw) {
        *o = (v * nrm2).round().clamp(0., 255.) as u8;
    }
    out
}

/// Keypoints (in `img` pixels) and descriptors, the `max_features` strongest.
pub fn detect(img: &Gray, max_features: usize) -> (Vec<Keypoint>, Vec<[u8; 128]>) {
    let pyr = pyramid(img);
    let threshold = (0.5 * CONTRAST / LAYERS as f32 * 255.).floor();
    let mut kps: Vec<Keypoint> = (0..pyr.dog.len())
        .into_par_iter()
        .flat_map(|o| {
            let dog = &pyr.dog[o];
            let (w, h) = (dog[0].w, dog[0].h);
            (1..=LAYERS)
                .into_par_iter()
                .flat_map(move |l| {
                    let mut found = Vec::new();
                    if w <= 2 * BORDER || h <= 2 * BORDER {
                        return found;
                    }
                    let (p, c, n) = (&dog[l - 1], &dog[l], &dog[l + 1]);
                    // the 8 neighbours in the layer, then the 9 above and below (offsets in the row-major pixels)
                    let around: [isize; 8] = [
                        -(w as isize) - 1,
                        -(w as isize),
                        -(w as isize) + 1,
                        -1,
                        1,
                        w as isize - 1,
                        w as isize,
                        w as isize + 1,
                    ];
                    for y in BORDER..h - BORDER {
                        for x in BORDER..w - BORDER {
                            let k = y * w + x;
                            let v = c.px[k];
                            if v.abs() <= threshold {
                                continue;
                            }
                            let at = |g: &Gray, d: isize| g.px[(k as isize + d) as usize];
                            let ext = if v > 0. {
                                around.iter().all(|&d| v >= at(c, d))
                                    && v >= p.px[k]
                                    && v >= n.px[k]
                                    && around.iter().all(|&d| v >= at(p, d) && v >= at(n, d))
                            } else {
                                around.iter().all(|&d| v <= at(c, d))
                                    && v <= p.px[k]
                                    && v <= n.px[k]
                                    && around.iter().all(|&d| v <= at(p, d) && v <= at(n, d))
                            };
                            if ext && let Some(kp) = refine(dog, o, l, x, y) {
                                found.push(kp);
                            }
                        }
                    }
                    found
                })
                .collect::<Vec<_>>()
        })
        .collect();
    // orientations: one keypoint per dominant direction
    kps = kps
        .into_par_iter()
        .flat_map(|kp| {
            let img = &pyr.gauss[kp.octave][kp.layer];
            orientations(img, &kp)
                .into_iter()
                .map(|a| Keypoint {
                    angle: a,
                    ..kp.clone()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    // duplicates, then the strongest
    kps.sort_by(|a, b| {
        a.x.total_cmp(&b.x)
            .then(a.y.total_cmp(&b.y))
            .then(a.size.total_cmp(&b.size))
            .then(a.angle.total_cmp(&b.angle))
    });
    kps.dedup_by(|a, b| a.x == b.x && a.y == b.y && a.size == b.size && a.angle == b.angle);
    if kps.len() > max_features {
        kps.sort_by(|a, b| b.response.total_cmp(&a.response));
        let cut = kps[max_features - 1].response;
        let keep = kps.iter().take_while(|k| k.response >= cut).count();
        kps.truncate(keep);
    }
    let desc = kps
        .par_iter()
        .map(|kp| descriptor(&pyr.gauss[kp.octave][kp.layer], kp))
        .collect();
    (kps, desc)
}
