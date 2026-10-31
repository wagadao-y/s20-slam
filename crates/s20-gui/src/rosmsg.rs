//! Bounded ROS1 message decoding from the definitions embedded in bags, for the raw data view
//! (from S20 Raw Inspector). Large arrays are counted, with a short preview, never expanded.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::collections::HashMap;

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("wire offset overflow")?;
        let b = self
            .data
            .get(self.pos..end)
            .context("truncated ROS message")?;
        self.pos = end;
        Ok(b)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into()?))
    }
    /// A ROS time in nanoseconds.
    fn time(&mut self) -> Result<i64> {
        let s = self.u32()? as i64;
        let n = self.u32()? as i64;
        ensure!(n < 1_000_000_000, "invalid ROS nanoseconds");
        Ok(s * 1_000_000_000 + n)
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

/// The fields of a message type and of the types it uses, from its definition.
#[derive(Clone)]
pub struct Schema {
    root: String,
    types: HashMap<String, Vec<(String, String)>>,
}

impl Schema {
    pub fn new(root: &str, definition: &str) -> Self {
        let mut types = HashMap::new();
        let mut current = root.to_string();
        for line in definition.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if let Some(name) = line.strip_prefix("MSG:") {
                current = name.trim().into();
                continue;
            }
            // constants are not on the wire
            if line.contains('=') {
                continue;
            }
            let mut parts = line.split_whitespace();
            if let (Some(ty), Some(name)) = (parts.next(), parts.next()) {
                types
                    .entry(current.clone())
                    .or_insert_with(Vec::new)
                    .push((ty.into(), name.into()));
            }
        }
        Self {
            root: root.into(),
            types,
        }
    }

    fn resolve(&self, ty: &str, owner: &str) -> String {
        if ty == "Header" {
            "std_msgs/Header".into()
        } else if ty.contains('/') {
            ty.into()
        } else {
            format!("{}/{}", owner.split('/').next().unwrap_or(""), ty)
        }
    }

    /// The wire size of a type when fixed.
    fn fixed(&self, ty: &str, owner: &str, depth: usize) -> Option<usize> {
        if depth > 16 {
            return None;
        }
        match ty {
            "bool" | "byte" | "char" | "int8" | "uint8" => Some(1),
            "int16" | "uint16" => Some(2),
            "int32" | "uint32" | "float32" => Some(4),
            "int64" | "uint64" | "float64" | "time" | "duration" => Some(8),
            "string" => None,
            _ => {
                if let Some((base, rest)) = ty.split_once('[') {
                    let n = rest.strip_suffix(']')?.parse::<usize>().ok()?;
                    return self.fixed(base, owner, depth + 1)?.checked_mul(n);
                }
                let name = self.resolve(ty, owner);
                self.types.get(&name)?.iter().try_fold(0usize, |a, (t, _)| {
                    a.checked_add(self.fixed(t, &name, depth + 1)?)
                })
            }
        }
    }

    fn value(&self, c: &mut Cursor<'_>, ty: &str, owner: &str, depth: usize) -> Result<Value> {
        ensure!(depth < 20, "ROS schema recursion limit");
        if let Some((base, rest)) = ty.split_once('[') {
            let length = rest.strip_suffix(']').context("invalid array type")?;
            let n = if length.is_empty() {
                c.u32()? as usize
            } else {
                length.parse()?
            };
            ensure!(n <= 100_000_000, "ROS array exceeds preview limit");
            let mut preview = Vec::new();
            for _ in 0..n.min(3) {
                preview.push(self.value(c, base, owner, depth + 1)?);
            }
            if n > 3 {
                if let Some(size) = self.fixed(base, owner, depth + 1) {
                    c.take(size.checked_mul(n - 3).context("array overflow")?)?;
                } else {
                    ensure!(n < 1024, "dynamic array exceeds preview limit");
                    for _ in 3..n {
                        self.value(c, base, owner, depth + 1)?;
                    }
                }
            }
            return Ok(json!({"count": n, "preview": preview}));
        }
        Ok(match ty {
            "bool" => json!(c.u8()? != 0),
            "char" | "uint8" => json!(c.u8()?),
            "byte" | "int8" => json!(c.u8()? as i8),
            "uint16" => json!(c.u16()?),
            "int16" => json!(c.u16()? as i16),
            "uint32" => json!(c.u32()?),
            "int32" => json!(c.u32()? as i32),
            "uint64" => json!(c.u64()?),
            "int64" => json!(c.u64()? as i64),
            "float32" => json!(c.f32()?),
            "float64" => json!(c.f64()?),
            "string" => json!(c.string()?),
            "time" => json!({"ns": c.time()?}),
            "duration" => {
                let s = c.u32()? as i32;
                let n = c.u32()? as i32;
                json!({"sec": s, "nsec": n})
            }
            _ => {
                let name = if self.types.contains_key(ty) {
                    ty.into()
                } else {
                    self.resolve(ty, owner)
                };
                let fields = self
                    .types
                    .get(&name)
                    .with_context(|| format!("unknown ROS type {name}"))?;
                let mut obj = serde_json::Map::new();
                for (t, n) in fields {
                    obj.insert(n.clone(), self.value(c, t, &name, depth + 1)?);
                }
                Value::Object(obj)
            }
        })
    }

    pub fn decode(&self, data: &[u8]) -> Result<Value> {
        self.value(&mut Cursor { data, pos: 0 }, &self.root, &self.root, 0)
    }
}

/// The header stamp of a decoded message, in seconds (none without a header).
pub fn stamp(v: &Value) -> Option<f64> {
    let ns = v.get("header")?.get("stamp")?.get("ns")?.as_i64()?;
    Some(ns as f64 * 1e-9)
}

/// The main values of a decoded message, on one line.
pub fn brief(v: &Value, ty: &str) -> String {
    let number = |key: &str| v.get(key).and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
    if ty.ends_with("NavSatFix") {
        return format!(
            "lat={:.8} lon={:.8} h={:.3}",
            number("latitude"),
            number("longitude"),
            number("altitude")
        );
    }
    if ty.ends_with("PVTSLNMsg") {
        return format!(
            "lat={:.8} lon={:.8} h={:.3} type={}",
            number("bestpos_lat"),
            number("bestpos_lon"),
            number("bestpos_hgt"),
            v["bestpos_type"]["type"]
        );
    }
    if ty.ends_with("Imu") {
        let vector = |key: &str| {
            let q = &v[key];
            format!(
                "({:.4},{:.4},{:.4})",
                q["x"].as_f64().unwrap_or(f64::NAN),
                q["y"].as_f64().unwrap_or(f64::NAN),
                q["z"].as_f64().unwrap_or(f64::NAN)
            )
        };
        return format!(
            "gyro={} accel={}",
            vector("angular_velocity"),
            vector("linear_acceleration")
        );
    }
    let keys: &[&str] = if ty.ends_with("CustomMsg") {
        &["point_num", "timebase"]
    } else if ty.ends_with("CompressedImage") {
        &["format"]
    } else if ty.ends_with("ShareSyncMsg") {
        &["sync_time", "pps_time"]
    } else {
        &[]
    };
    keys.iter()
        .filter_map(|key| v.get(*key).map(|value| format!("{key}={value}")))
        .collect::<Vec<_>>()
        .join("  ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrays_are_counted_with_a_preview_and_truncation_fails() {
        let s = Schema::new("test/Msg", "uint8[] bytes\nfloat64 value");
        let mut b = 4u32.to_le_bytes().to_vec();
        b.extend([1, 2, 3, 4]);
        b.extend(1.5f64.to_le_bytes());
        let v = s.decode(&b).unwrap();
        assert_eq!(v["bytes"]["count"], 4);
        assert_eq!(v["bytes"]["preview"].as_array().unwrap().len(), 3);
        assert_eq!(v["value"], 1.5);
        assert!(s.decode(&b[..6]).is_err());
    }

    #[test]
    fn headers_resolve_to_std_msgs_and_give_the_stamp() {
        let def = "Header header\nfloat64 x\n\
                   ================\nMSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n";
        let s = Schema::new("test/Point", def);
        let mut b = 7u32.to_le_bytes().to_vec();
        b.extend(12u32.to_le_bytes());
        b.extend(500_000_000u32.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        b.extend(b"map");
        b.extend(2.0f64.to_le_bytes());
        let v = s.decode(&b).unwrap();
        assert_eq!(v["header"]["frame_id"], "map");
        assert_eq!(stamp(&v), Some(12.5));
        assert_eq!(v["x"], 2.0);
    }
}
