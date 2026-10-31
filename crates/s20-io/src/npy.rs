//! Minimal .npy (version 1.0) reader/writer for little-endian float64 2-D arrays.

use anyhow::{Context, Result, bail};
use std::fs;
use std::io::Write;
use std::path::Path;

pub fn write_f64(path: &Path, data: &[f64], cols: usize) -> Result<()> {
    if cols == 0 || !data.len().is_multiple_of(cols) {
        bail!("data length {} is not a multiple of {cols}", data.len());
    }
    let rows = data.len() / cols;
    let mut header =
        format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // magic(6) + version(2) + len(2) + header + '\n' must be a multiple of 64
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut f = fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    let mut buf = Vec::with_capacity(data.len() * 8);
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)?;
    Ok(())
}

/// Returns (data row-major, rows, cols).
pub fn read_f64(path: &Path) -> Result<(Vec<f64>, usize, usize)> {
    let b = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if !b.starts_with(b"\x93NUMPY") {
        bail!("{} is not an npy file", path.display());
    }
    let (hlen, hstart) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes(b[8..12].try_into()?) as usize, 12),
        v => bail!("unsupported npy version {v}"),
    };
    let header = std::str::from_utf8(&b[hstart..hstart + hlen])?;
    if !header.contains("'<f8'") || header.contains("'fortran_order': True") {
        bail!(
            "{}: only C-order float64 arrays are supported ({header})",
            path.display()
        );
    }
    let shape_s = header
        .split("'shape': (")
        .nth(1)
        .context("no shape")?
        .split(')')
        .next()
        .context("bad shape")?;
    let dims: Vec<usize> = shape_s
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let (rows, cols) = match dims.as_slice() {
        [r] => (*r, 1),
        [r, c] => (*r, *c),
        _ => bail!("unsupported shape {dims:?}"),
    };
    let body = &b[hstart + hlen..];
    if body.len() < rows * cols * 8 {
        bail!("{} truncated", path.display());
    }
    let data = body[..rows * cols * 8]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    Ok((data, rows, cols))
}
