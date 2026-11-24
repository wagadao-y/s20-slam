//! Minimal .npy (version 1.0) reader/writer for little-endian float64 2-D arrays.

use anyhow::{Context, Result, bail, ensure};
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

/// The items of a Python dictionary literal `{key: value, ...}` (a top level one), keys and
/// values without their quotes and spaces; None when it is not one.
fn top_items(h: &str) -> Option<Vec<(String, String)>> {
    let h = h.trim();
    let body = h.strip_prefix('{')?.strip_suffix('}')?;
    let (mut items, mut cur, mut depth, mut quote) = (Vec::new(), String::new(), 0i32, None);
    for c in body.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => {
                quote = None;
                cur.push(c);
            }
            (Some(_), c) => cur.push(c),
            (None, '\'' | '"') => {
                quote = Some(c);
                cur.push(c);
            }
            (None, '(' | '[' | '{') => {
                depth += 1;
                cur.push(c);
            }
            (None, ')' | ']' | '}') => {
                depth -= 1;
                cur.push(c);
            }
            (None, ',') if depth == 0 => items.push(std::mem::take(&mut cur)),
            (None, c) => cur.push(c),
        }
    }
    items.push(cur);
    let clean = |s: &str| {
        s.chars()
            .filter(|c| !c.is_whitespace() && *c != '\'' && *c != '"')
            .collect::<String>()
    };
    let mut out = Vec::new();
    for item in items.iter().filter(|i| !i.trim().is_empty()) {
        // the key is quoted (an ASCII quote, so a byte): its colon is the first outside it; the
        // value kept as written (a string quoted, False bare), without its spaces
        let item = item.trim();
        let q = item.chars().next()?;
        if q != '\'' && q != '"' {
            return None;
        }
        let end = item[1..].find(q)? + 1;
        let (k, rest) = item.split_at(end + 1);
        let v = rest.trim_start().strip_prefix(':')?;
        let v: String = v.chars().filter(|c| !c.is_whitespace()).collect();
        out.push((clean(k), v));
    }
    Some(out)
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
    // (the dictionary's own items: split at its commas outside quotes and brackets, each key
    // and value read without quotes and spaces; a key once only; a structured array (its descr
    // a list) refused, not read as plain numbers)
    let items = top_items(header).context("not a dictionary")?;
    let mut map = std::collections::HashMap::new();
    for (k, v) in items {
        ensure!(
            map.insert(k.clone(), v).is_none(),
            "{}: a key twice in the header",
            path.display()
        );
    }
    let value = |key: &str| map.get(key).map(String::as_str);
    // (descr the string '<f8', fortran_order the bare False, not the string "False")
    if !matches!(value("descr"), Some("'<f8'" | "\"<f8\""))
        || value("fortran_order") != Some("False")
    {
        // (not the header itself: names in it could tell a place)
        bail!(
            "{}: only C-order float64 arrays are supported",
            path.display()
        );
    }
    let shape_s = value("shape")
        .and_then(|v| v.strip_prefix('('))
        .and_then(|v| v.strip_suffix(')'))
        .context("no shape")?;
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
        // a key in a value's text is not that key
        npy_of(
            &f,
            "{'descr': '<f8', 'note': ',fortran_order:False,', 'fortran_order': True, 'shape': (2, 3), }",
            6,
        );
        assert!(read_f64(&f).is_err());
        // a string "False" is not False; a key not quoted, of other characters than ASCII, an
        // error and no panic
        npy_of(
            &f,
            r#"{"descr":"<f8","fortran_order":"False","shape":(2,3)}"#,
            6,
        );
        assert!(read_f64(&f).is_err());
        npy_of(
            &f,
            "{日本: 1, 'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }",
            6,
        );
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
