//! Greyscale images in f32 (0..255): JPEG decoding and area downscaling.

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub px: Vec<f32>,
}

impl Gray {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            px: vec![0.; w * h],
        }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.px[y * self.w + x]
    }
}

/// The luma of a JPEG.
pub fn decode(jpeg: &[u8]) -> Result<Gray> {
    let mut d = zune_jpeg::JpegDecoder::new_with_options(
        zune_core::bytestream::ZCursor::new(jpeg),
        zune_core::options::DecoderOptions::default()
            .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::Luma),
    );
    let px = d.decode().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let info = d.info().context("no image size")?;
    let (w, h) = (info.width as usize, info.height as usize);
    ensure!(px.len() == w * h, "not one channel");
    Ok(Gray {
        w,
        h,
        px: px.into_iter().map(f32::from).collect(),
    })
}

/// Pixel-area resampling to `w` x `h` (no larger than the source), as OpenCV's INTER_AREA.
pub fn resize_area(src: &Gray, w: usize, h: usize) -> Gray {
    let (sx, sy) = (src.w as f64 / w as f64, src.h as f64 / h as f64);
    // per output column/row: the source cells it covers with their weights
    let spans = |n: usize, s: f64, limit: usize| -> Vec<Vec<(usize, f32)>> {
        (0..n)
            .map(|i| {
                let (a, b) = (i as f64 * s, ((i + 1) as f64 * s).min(limit as f64));
                let mut v = Vec::new();
                let mut x = a.floor() as usize;
                while (x as f64) < b {
                    let cover = (b.min(x as f64 + 1.) - a.max(x as f64)) / (b - a);
                    if cover > 1e-9 {
                        v.push((x, cover as f32));
                    }
                    x += 1;
                }
                v
            })
            .collect()
    };
    let (cols, rows) = (spans(w, sx, src.w), spans(h, sy, src.h));
    let mut out = Gray::new(w, h);
    out.px.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
        for (x, o) in line.iter_mut().enumerate() {
            let mut s = 0f32;
            for &(sy_, wy) in &rows[y] {
                let base = sy_ * src.w;
                let mut r = 0f32;
                for &(sx_, wx) in &cols[x] {
                    r += src.px[base + sx_] * wx;
                }
                s += r * wy;
            }
            *o = s;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_resize_averages_whole_and_partial_cells() {
        let src = Gray {
            w: 4,
            h: 2,
            px: vec![0., 4., 8., 12., 0., 4., 8., 12.],
        };
        let half = resize_area(&src, 2, 1);
        assert_eq!(half.px, vec![2., 10.]);
        let third = resize_area(
            &Gray {
                w: 3,
                h: 1,
                px: vec![0., 3., 6.],
            },
            2,
            1,
        );
        assert!(
            (third.px[0] - 1.).abs() < 1e-6 && (third.px[1] - 5.).abs() < 1e-6,
            "{:?}",
            third.px
        );
    }
}
