//! Minimal .npy (version 1.0) reader/writer for little-endian float64 2-D arrays.

use anyhow::{Context, Result, bail};
use std::fs;
use std::io::Write;
use std::path::Path;

pub fn write_f64(path: &Path, data: &[f64], cols: usize) -> Result<()> {
    let b = to_bytes(data, cols)?;
    let mut f = fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(&b)?;
    Ok(())
}

/// The bytes of [`write_f64`]'s file.
pub fn to_bytes(data: &[f64], cols: usize) -> Result<Vec<u8>> {
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
    let mut b = Vec::with_capacity(10 + header.len() + data.len() * 8);
    b.extend_from_slice(b"\x93NUMPY\x01\x00");
    b.extend_from_slice(&(header.len() as u16).to_le_bytes());
    b.extend_from_slice(header.as_bytes());
    for v in data {
        b.extend_from_slice(&v.to_le_bytes());
    }
    Ok(b)
}

/// Returns (data row-major, rows, cols).
pub fn read_f64(path: &Path) -> Result<(Vec<f64>, usize, usize)> {
    let b = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if !b.starts_with(b"\x93NUMPY") {
        bail!("{} is not an npy file", path.display());
    }
    // (a short or damaged file an error, not a panic)
    let short = || anyhow::anyhow!("{} truncated", path.display());
    let (hlen, hstart) = match b.get(6).ok_or_else(short)? {
        1 => (
            u16::from_le_bytes(b.get(8..10).ok_or_else(short)?.try_into()?) as usize,
            10,
        ),
        2 | 3 => (
            u32::from_le_bytes(b.get(8..12).ok_or_else(short)?.try_into()?) as usize,
            12,
        ),
        v => bail!("unsupported npy version {v}"),
    };
    let header = std::str::from_utf8(b.get(hstart..hstart + hlen).ok_or_else(short)?)?;
    // (read without spaces and quotes: a Python literal may have either kind, spaces or not;
    // a structured array (its descr a list) is refused, not read as plain numbers)
    let compact: String = header
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '\'' && *c != '"')
        .collect();
    // (each key's value up to the next ',' or '}': the key itself, after '{' or ',', not one
    // ending in its name; present, and as read here)
    let value = |key: &str| {
        let at = ["{", ","]
            .iter()
            .find_map(|p| compact.find(&format!("{p}{key}:")).map(|i| i + p.len()))?
            + key.len()
            + 1;
        let rest = &compact[at..];
        Some(&rest[..rest.find([',', '}']).unwrap_or(rest.len())])
    };
    if value("descr") != Some("<f8") || value("fortran_order") != Some("False") {
        // (not the header itself: names in it could tell a place)
        bail!(
            "{}: only C-order float64 arrays are supported",
            path.display()
        );
    }
    let shape_s = [",shape:(", "{shape:("]
        .iter()
        .find_map(|k| compact.split_once(k).map(|(_, r)| r))
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
    let n = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(8))
        .context("shape too large")?;
    if body.len() < n {
        bail!("{} truncated", path.display());
    }
    let data = body[..n]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    Ok((data, rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An npy file of version 1 with header `h` (padded) and `n` zero doubles.
    fn npy_of(path: &Path, h: &str, n: usize) {
        let mut h = h.to_string();
        while !(10 + h.len() + 1).is_multiple_of(64) {
            h.push(' ');
        }
        h.push('\n');
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend((h.len() as u16).to_le_bytes());
        b.extend(h.as_bytes());
        b.extend(vec![0u8; 8 * n]);
        fs::write(path, b).unwrap();
    }

    #[test]
    fn headers_read_whatever_their_quotes_and_spaces() {
        let f = std::env::temp_dir().join(format!("s20_npy_{}.npy", std::process::id()));
        npy_of(
            &f,
            r#"{"descr":"<f8","fortran_order":False,"shape":(2,3)}"#,
            6,
        );
        assert_eq!(read_f64(&f).unwrap().1..read_f64(&f).unwrap().2, 2..3);
        // column order, written either way, refused
        npy_of(
            &f,
            r#"{"descr":"<f8","fortran_order":True,"shape":(2,3)}"#,
            6,
        );
        assert!(read_f64(&f).is_err());
        // the keys in another order, the last without a comma; a key missing
        npy_of(
            &f,
            "{'shape': (2, 3), 'fortran_order': False, 'descr': '<f8'}",
            6,
        );
        assert_eq!(read_f64(&f).unwrap().1, 2);
        npy_of(&f, "{'descr': '<f8', 'shape': (2, 3), }", 6);
        assert!(read_f64(&f).is_err());
        // a key that only ends in a name read is not that key
        npy_of(
            &f,
            "{'not_descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }",
            6,
        );
        assert!(read_f64(&f).is_err());
        // a structured array is not plain numbers
        npy_of(
            &f,
            "{'descr': [('t', '<f8'), ('x', '<f8')], 'fortran_order': False, 'shape': (3,), }",
            6,
        );
        assert!(read_f64(&f).is_err());
        write_f64(&f, &[1., 2., 3., 4.], 2).unwrap();
        assert_eq!(read_f64(&f).unwrap(), (vec![1., 2., 3., 4.], 2, 2));
        let _ = fs::remove_file(&f);
    }
}
