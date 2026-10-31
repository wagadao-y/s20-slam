//! Minimal ROS1 bag (v2.0) reader for S20 recordings.
//!
//! The file is memory-mapped. Only uncompressed chunks are supported (S20 writes them uncompressed);
//! message payloads are returned as slices into the map (zero copy).

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const MAGIC: &[u8] = b"#ROSBAG V2.0\n";
const OP_MSG_DATA: u8 = 0x02;
const OP_BAG_HEADER: u8 = 0x03;
const OP_INDEX_DATA: u8 = 0x04;
const OP_CHUNK: u8 = 0x05;
const OP_CHUNK_INFO: u8 = 0x06;
const OP_CONNECTION: u8 = 0x07;

#[derive(Debug, Clone)]
pub struct Connection {
    pub id: u32,
    pub topic: String,
    pub msg_type: String,
    pub message_definition: String,
}

/// One message record: connection id, record time (ns) and payload byte range in the file.
#[derive(Debug, Clone, Copy)]
pub struct MsgRef {
    pub conn: u32,
    pub time_ns: u64,
    pub start: usize,
    pub len: usize,
}

pub struct Bag {
    map: Mmap,
    pub connections: HashMap<u32, Connection>,
    /// All message records in file order.
    pub messages: Vec<MsgRef>,
}

struct Record<'a> {
    header: Vec<(&'a [u8], &'a [u8])>,
    data_start: usize,
    data_len: usize,
    end: usize,
}

fn read_u32(b: &[u8], p: usize) -> Result<u32> {
    let s = b.get(p..p + 4).context("truncated u32")?;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}

fn parse_fields(h: &[u8]) -> Result<Vec<(&[u8], &[u8])>> {
    let mut out = Vec::new();
    let mut p = 0;
    while p < h.len() {
        let n = read_u32(h, p)? as usize;
        p += 4;
        let f = h.get(p..p + n).context("truncated header field")?;
        p += n;
        let eq = f
            .iter()
            .position(|&c| c == b'=')
            .context("header field without '='")?;
        out.push((&f[..eq], &f[eq + 1..]));
    }
    Ok(out)
}

fn read_record(b: &[u8], p: usize) -> Result<Record<'_>> {
    let hlen = read_u32(b, p)? as usize;
    let h = b
        .get(p + 4..p + 4 + hlen)
        .context("truncated record header")?;
    let header = parse_fields(h)?;
    let dpos = p + 4 + hlen;
    let dlen = read_u32(b, dpos)? as usize;
    if dpos + 4 + dlen > b.len() {
        bail!("truncated record data");
    }
    Ok(Record {
        header,
        data_start: dpos + 4,
        data_len: dlen,
        end: dpos + 4 + dlen,
    })
}

fn field<'a>(r: &Record<'a>, name: &[u8]) -> Option<&'a [u8]> {
    r.header.iter().find(|(k, _)| *k == name).map(|(_, v)| *v)
}

fn op(r: &Record) -> Result<u8> {
    let v = field(r, b"op").context("record without op")?;
    Ok(v[0])
}

fn time_ns(v: &[u8]) -> u64 {
    let s = u32::from_le_bytes(v[0..4].try_into().unwrap()) as u64;
    let n = u32::from_le_bytes(v[4..8].try_into().unwrap()) as u64;
    s * 1_000_000_000 + n
}

impl Bag {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the file is opened read-only and not modified while mapped.
        let map = unsafe { Mmap::map(&f)? };
        if !map.starts_with(MAGIC) {
            bail!("{} is not a ROS1 bag v2.0", path.display());
        }
        let mut connections = HashMap::new();
        let mut messages = Vec::new();
        let b: &[u8] = &map;
        let mut p = MAGIC.len();
        let mut trailing = 0usize;
        while p < b.len() {
            let r = match read_record(b, p) {
                Ok(r) => r,
                Err(_) => {
                    // S20 bags can end with a few hundred bytes that do not form a record
                    trailing = b.len() - p;
                    break;
                }
            };
            match op(&r)? {
                OP_BAG_HEADER | OP_INDEX_DATA | OP_CHUNK_INFO => {}
                OP_CONNECTION => {
                    let c = parse_connection(b, &r)?;
                    connections.entry(c.id).or_insert(c);
                }
                OP_CHUNK => {
                    let comp = field(&r, b"compression").unwrap_or(b"none");
                    if comp != b"none" {
                        bail!(
                            "compressed chunks ({}) are not supported",
                            String::from_utf8_lossy(comp)
                        );
                    }
                    let mut q = r.data_start;
                    let end = r.data_start + r.data_len;
                    while q < end {
                        let m = read_record(b, q)?;
                        match op(&m)? {
                            OP_MSG_DATA => {
                                let conn = u32::from_le_bytes(
                                    field(&m, b"conn").context("msg without conn")?.try_into()?,
                                );
                                let t = time_ns(field(&m, b"time").context("msg without time")?);
                                messages.push(MsgRef {
                                    conn,
                                    time_ns: t,
                                    start: m.data_start,
                                    len: m.data_len,
                                });
                            }
                            OP_CONNECTION => {
                                let c = parse_connection(b, &m)?;
                                connections.entry(c.id).or_insert(c);
                            }
                            _ => {}
                        }
                        q = m.end;
                    }
                }
                OP_MSG_DATA => {
                    // message outside a chunk (not written by rosbag, but allowed)
                    let conn = u32::from_le_bytes(
                        field(&r, b"conn").context("msg without conn")?.try_into()?,
                    );
                    let t = time_ns(field(&r, b"time").context("msg without time")?);
                    messages.push(MsgRef {
                        conn,
                        time_ns: t,
                        start: r.data_start,
                        len: r.data_len,
                    });
                }
                o => bail!("unknown record op {o:#x} at {p}"),
            }
            p = r.end;
        }
        if trailing > 0 {
            eprintln!(
                "warning: {} ignored {trailing} trailing bytes",
                path.display()
            );
        }
        Ok(Bag {
            map,
            connections,
            messages,
        })
    }

    pub fn data(&self, m: &MsgRef) -> &[u8] {
        &self.map[m.start..m.start + m.len]
    }

    /// Byte position in the file of a slice returned by [`Bag::data`] (or a part of it).
    pub fn offset_of(&self, s: &[u8]) -> usize {
        let o = (s.as_ptr() as usize).wrapping_sub(self.map.as_ptr() as usize);
        assert!(o + s.len() <= self.map.len(), "slice is not in the bag");
        o
    }

    /// Connection ids of a topic.
    pub fn topic_conns(&self, topic: &str) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .connections
            .values()
            .filter(|c| c.topic == topic)
            .map(|c| c.id)
            .collect();
        v.sort();
        v
    }

    /// Messages of a topic sorted by record time (stable: ties keep file order).
    pub fn topic_messages(&self, topic: &str) -> Vec<MsgRef> {
        let ids = self.topic_conns(topic);
        let mut v: Vec<MsgRef> = self
            .messages
            .iter()
            .filter(|m| ids.contains(&m.conn))
            .copied()
            .collect();
        v.sort_by_key(|m| m.time_ns);
        v
    }
}

fn parse_connection(b: &[u8], r: &Record) -> Result<Connection> {
    let id = u32::from_le_bytes(
        field(r, b"conn")
            .context("connection without conn")?
            .try_into()?,
    );
    let topic = String::from_utf8_lossy(field(r, b"topic").context("connection without topic")?)
        .into_owned();
    let inner = parse_fields(&b[r.data_start..r.data_start + r.data_len])?;
    let get = |k: &[u8]| {
        inner
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| String::from_utf8_lossy(v).into_owned())
            .unwrap_or_default()
    };
    Ok(Connection {
        id,
        topic,
        msg_type: get(b"type"),
        message_definition: get(b"message_definition"),
    })
}
