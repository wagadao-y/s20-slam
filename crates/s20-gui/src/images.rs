//! Camera images of the recording (`data/images.txt`, JPEGs in the bag), decoded off the UI thread
//! and shrunk for display.

use eframe::egui;
use std::path::{Path, PathBuf};

/// One image: side ("left"/"right"), sensor-clock time, file.
#[derive(Clone, Debug)]
pub struct Stamp {
    pub left: bool,
    pub t: f64,
    pub file: PathBuf,
}

/// The images of one session's data folder, their times shifted by `shift`.
pub fn load_stamps_of(data: &Path, shift: f64) -> Vec<Stamp> {
    let dir = data.join("images");
    let Ok(index) = s20_io::dataset::image_index(data) else {
        return Vec::new();
    };
    let mut v: Vec<Stamp> = index
        .into_iter()
        .map(|im| Stamp {
            left: im.name.starts_with("left"),
            t: im.t + shift,
            file: dir.join(&im.name),
        })
        .collect();
    v.sort_by(|a, b| a.t.total_cmp(&b.t));
    v
}

/// The image of a side nearest to time `t`.
pub fn nearest(stamps: &[Stamp], left: bool, t: f64) -> Option<&Stamp> {
    stamps
        .iter()
        .filter(|s| s.left == left)
        .min_by(|a, b| (a.t - t).abs().total_cmp(&(b.t - t).abs()))
}

/// Decodes a JPEG and shrinks it by `factor` (box filter).
pub fn decode_small(path: &Path, factor: usize) -> Option<egui::ColorImage> {
    let data = s20_io::dataset::read_image(path).ok()?;
    let mut d = zune_jpeg::JpegDecoder::new(zune_core::bytestream::ZCursor::new(data));
    let px = d.decode().ok()?;
    let info = d.info()?;
    let (w, h) = (info.width as usize, info.height as usize);
    let f = factor.max(1);
    let (sw, sh) = (w / f, h / f);
    let mut out = vec![0u8; sw * sh * 4];
    for y in 0..sh {
        for x in 0..sw {
            let mut acc = [0u32; 3];
            for dy in 0..f {
                let row = (y * f + dy) * w;
                for dx in 0..f {
                    let i = (row + x * f + dx) * 3;
                    for c in 0..3 {
                        acc[c] += px[i + c] as u32;
                    }
                }
            }
            let o = (y * sw + x) * 4;
            for c in 0..3 {
                out[o + c] = (acc[c] / (f * f) as u32) as u8;
            }
            out[o + 3] = 255;
        }
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([sw, sh], &out))
}
