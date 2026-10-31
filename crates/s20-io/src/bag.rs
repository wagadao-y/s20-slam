//! Minimal ROS1 bag (v2.0) reader for S20 recordings.
//!
//! The file is memory-mapped. Only uncompressed chunks are supported (S20 writes them uncompressed);
//! message payloads are returned as slices into the map (zero copy). [`BagWriter`] writes a bag
//! this reader reads back: a part of a recording, as a short raw folder for tests.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
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
    /// The byte range in the file of each connection's (first) record, header and data.
    conn_records: HashMap<u32, (usize, usize)>,
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
    v.first().copied().context("empty op field")
}

fn time_ns(v: &[u8]) -> Result<u64> {
    let s = read_u32(v, 0).context("short time field")? as u64;
    let n = read_u32(v, 4).context("short time field")? as u64;
    Ok(s * 1_000_000_000 + n)
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
        let mut conn_records = HashMap::new();
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
                    conn_records.entry(c.id).or_insert((p, r.end));
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
                                let t = time_ns(field(&m, b"time").context("msg without time")?)?;
                                messages.push(MsgRef {
                                    conn,
                                    time_ns: t,
                                    start: m.data_start,
                                    len: m.data_len,
                                });
                            }
                            OP_CONNECTION => {
                                let c = parse_connection(b, &m)?;
                                conn_records.entry(c.id).or_insert((q, m.end));
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
                    let t = time_ns(field(&r, b"time").context("msg without time")?)?;
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
            conn_records,
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

    /// The record of connection `id` as it is in the file (header and data), for [`BagWriter`].
    pub fn connection_record(&self, id: u32) -> Option<&[u8]> {
        self.conn_records.get(&id).map(|&(a, b)| &self.map[a..b])
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

/// Header fields (`name=value`), each after its length.
fn header(fields: &[(&str, &[u8])]) -> Vec<u8> {
    let mut h = Vec::new();
    for (k, v) in fields {
        h.extend((k.len() as u32 + 1 + v.len() as u32).to_le_bytes());
        h.extend(k.as_bytes());
        h.push(b'=');
        h.extend(*v);
    }
    h
}

/// A record: its header fields and data.
fn record(fields: &[(&str, &[u8])], data: &[u8]) -> Vec<u8> {
    let h = header(fields);
    let mut r = (h.len() as u32).to_le_bytes().to_vec();
    r.extend(h);
    r.extend((data.len() as u32).to_le_bytes());
    r.extend(data);
    r
}

/// Chunks are closed at about this size (a record's data length is a u32).
const CHUNK_BYTES: usize = 32 << 20;

/// Writes messages of an open [`Bag`] into a new bag: uncompressed chunks, each connection's
/// record (as in the source) before its first message, no index (this reader does not use one;
/// rosbag's tools want it).
pub struct BagWriter {
    file: BufWriter<File>,
    chunk: Vec<u8>,
    written: std::collections::HashSet<u32>,
}

impl BagWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
        let mut file = BufWriter::new(f);
        file.write_all(MAGIC)?;
        // no index: its position and the counts are 0
        let header = record(
            &[
                ("op", &[OP_BAG_HEADER]),
                ("index_pos", &0u64.to_le_bytes()),
                ("conn_count", &0u32.to_le_bytes()),
                ("chunk_count", &0u32.to_le_bytes()),
            ],
            &[],
        );
        file.write_all(&header)?;
        Ok(BagWriter {
            file,
            chunk: Vec::new(),
            written: Default::default(),
        })
    }

    /// Appends message `m` of `bag`, its connection first when it is new.
    pub fn write(&mut self, bag: &Bag, m: &MsgRef) -> Result<()> {
        if self.written.insert(m.conn) {
            let c = bag
                .connection_record(m.conn)
                .with_context(|| format!("no record of connection {}", m.conn))?;
            self.chunk.extend(c);
        }
        let mut time = ((m.time_ns / 1_000_000_000) as u32).to_le_bytes().to_vec();
        time.extend(((m.time_ns % 1_000_000_000) as u32).to_le_bytes());
        self.chunk.extend(record(
            &[
                ("op", &[OP_MSG_DATA]),
                ("conn", &m.conn.to_le_bytes()),
                ("time", &time),
            ],
            bag.data(m),
        ));
        if self.chunk.len() >= CHUNK_BYTES {
            self.close_chunk()?;
        }
        Ok(())
    }

    fn close_chunk(&mut self) -> Result<()> {
        if self.chunk.is_empty() {
            return Ok(());
        }
        let size = u32::try_from(self.chunk.len()).context("chunk over 4 GB")?;
        let r = record(
            &[
                ("op", &[OP_CHUNK]),
                ("compression", b"none"),
                ("size", &size.to_le_bytes()),
            ],
            &self.chunk,
        );
        self.file.write_all(&r)?;
        self.chunk.clear();
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.close_chunk()?;
        self.file.flush()?;
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The message times of a bag of one message outside a chunk, with header fields `op` and
    /// `time` as given.
    fn times(name: &str, op: &[u8], time: &[u8]) -> Result<Vec<u64>> {
        let mut b = MAGIC.to_vec();
        let conn = 0u32.to_le_bytes();
        b.extend(record(&[("op", op), ("conn", &conn), ("time", time)], b"x"));
        let path = std::env::temp_dir().join(format!("s20_bag_{name}_{}.bag", std::process::id()));
        std::fs::write(&path, &b).unwrap();
        // the bag (its map) is closed before the file is removed
        let t = Bag::open(&path).map(|bag| bag.messages.iter().map(|m| m.time_ns).collect());
        let _ = std::fs::remove_file(&path);
        t
    }

    #[test]
    fn message_time_read() {
        let mut t = 3u32.to_le_bytes().to_vec();
        t.extend(5u32.to_le_bytes());
        assert_eq!(times("ok", &[OP_MSG_DATA], &t).unwrap(), [3_000_000_005]);
    }

    #[test]
    fn written_back_as_read() {
        let dir = std::env::temp_dir().join(format!("s20_bag_write_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let conn = 7u32.to_le_bytes();
        // a connection's data: the topic's fields
        let fields = header(&[("topic", b"/a"), ("type", b"std_msgs/String")]);
        let mut b = MAGIC.to_vec();
        b.extend(record(
            &[("op", &[OP_CONNECTION]), ("conn", &conn), ("topic", b"/a")],
            &fields,
        ));
        for (k, payload) in [b"one".as_slice(), b"two", b"three"]
            .into_iter()
            .enumerate()
        {
            let mut t = (10 + k as u32).to_le_bytes().to_vec();
            t.extend(0u32.to_le_bytes());
            b.extend(record(
                &[("op", &[OP_MSG_DATA]), ("conn", &conn), ("time", &t)],
                payload,
            ));
        }
        let src = dir.join("src.bag");
        std::fs::write(&src, &b).unwrap();
        let out = dir.join("out.bag");
        {
            let bag = Bag::open(&src).unwrap();
            let mut w = BagWriter::create(&out).unwrap();
            for m in &bag.messages[1..] {
                w.write(&bag, m).unwrap();
            }
            w.finish().unwrap();
            let back = Bag::open(&out).unwrap();
            let got: Vec<(u64, &[u8])> = back
                .messages
                .iter()
                .map(|m| (m.time_ns, back.data(m)))
                .collect();
            assert_eq!(
                got,
                [
                    (11_000_000_000, b"two".as_slice()),
                    (12_000_000_000, b"three")
                ]
            );
            let c = &back.connections[&7];
            assert_eq!(
                (c.topic.as_str(), c.msg_type.as_str()),
                ("/a", "std_msgs/String")
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn short_fields_are_errors() {
        let e = times("short_time", &[OP_MSG_DATA], &[1, 2, 3])
            .err()
            .unwrap();
        assert!(format!("{e:#}").contains("short time field"), "{e:#}");
        let e = times("empty_op", &[], &[0; 8]).err().unwrap();
        assert!(format!("{e:#}").contains("empty op field"), "{e:#}");
    }
}
