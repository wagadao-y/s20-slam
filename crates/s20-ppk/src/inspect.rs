//! What the GNSS raw files of a recording hold, for viewing: the frames and message types of the
//! rover (`ROVER_PPKRAW.bin`, Unicore) and the base (`BASE.BIN`, RTCM3), the satellites of each
//! observation epoch (decoded by RTKLIB, apart from PPK) and the base's antenna position.
use super::{Frames, SOLVER, clocks, device_time, frames};
use anyhow::{Context, Result, ensure};
use s20_io::job::Job;
use std::collections::BTreeMap;
use std::path::Path;

/// The satellite systems counted per epoch.
pub const SYSTEMS: [&str; 6] = ["GPS", "GLONASS", "Galileo", "BeiDou", "QZSS", "その他"];

/// One observation epoch: when, and what was tracked.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RawEpoch {
    /// UTC (s since 1970)
    pub utc: f64,
    /// satellites by system, as [`SYSTEMS`]
    pub sats: [u16; 6],
    /// mean over the satellites of their best signal strength (dB-Hz)
    pub snr: f32,
    /// signals tracked over all satellites
    pub signals: u16,
    /// satellites with a slipped signal
    pub slips: u16,
}

impl RawEpoch {
    pub fn total(&self) -> u16 {
        self.sats.iter().sum()
    }
}

/// What a GNSS raw file holds.
#[derive(Clone, Debug, Default)]
pub struct RawSummary {
    pub bytes: u64,
    /// valid frames, and bytes skipped over broken or truncated ones (counted where they start)
    pub frames: usize,
    pub bad: usize,
    /// frames by message type (Unicore message id, RTCM message number)
    pub messages: BTreeMap<u16, usize>,
    /// frames RTKLIB could not decode
    pub errors: usize,
    pub epochs: Vec<RawEpoch>,
    /// the base's antenna reference point (RTCM 1005/1006): ECEF (m), and latitude, longitude
    /// (deg) and ellipsoidal height (m)
    pub position: Option<([f64; 3], [f64; 3])>,
}

impl RawSummary {
    /// The longest interval between two epochs (s).
    pub fn max_gap(&self) -> f64 {
        self.epochs
            .windows(2)
            .map(|w| w[1].utc - w[0].utc)
            .fold(0.0, f64::max)
    }

    /// The usual interval between epochs (s): the median.
    pub fn interval(&self) -> Option<f64> {
        let mut d: Vec<f64> = self
            .epochs
            .windows(2)
            .map(|w| w[1].utc - w[0].utc)
            .collect();
        d.sort_by(f64::total_cmp);
        d.get(d.len() / 2).copied()
    }
}

/// The state of a position solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Fix {
    Fixed,
    /// PPK: fixed in one direction only (forward or backward), and used
    FixedOneSided,
    /// PPK: fixed, but not used: in one direction only and away from the other one's solution
    /// (or else too uncertain)
    FixedUnused,
    Float,
    Dgps,
    Single,
}

impl Fix {
    pub const ALL: [Self; 6] = [
        Self::Fixed,
        Self::FixedOneSided,
        Self::FixedUnused,
        Self::Float,
        Self::Dgps,
        Self::Single,
    ];

    /// From RTKLIB's solution status (1 FIX, 2 FLOAT, 4 DGPS, 5 SINGLE).
    pub fn of_rtklib(status: u8) -> Option<Self> {
        match status {
            1 => Some(Self::Fixed),
            2 => Some(Self::Float),
            4 => Some(Self::Dgps),
            5 => Some(Self::Single),
            _ => None,
        }
    }

    /// Of a PPK solution, with its forward and backward ones ([`super::ResultFile::directions`]):
    /// a FIX is told apart by the directions that fixed it, and whether it is used.
    pub fn of_ppk(
        s: &super::Solution,
        directions: [Option<&super::Solution>; 2],
        p: &super::Params,
    ) -> Option<Self> {
        let f = Self::of_rtklib(s.status)?;
        Some(if f != Self::Fixed {
            f
        } else if !s.accepted(p) {
            Self::FixedUnused
        } else if directions.iter().flatten().any(|d| d.status != 1) {
            Self::FixedOneSided
        } else {
            Self::Fixed
        })
    }

    /// From the receiver's position type (NovAtel / Unicore BESTPOS: 48..50 integer, 32..34
    /// float, 17 and 18 differential, 16 single).
    pub fn of_receiver(kind: u8) -> Option<Self> {
        match kind {
            48..=50 => Some(Self::Fixed),
            32..=34 => Some(Self::Float),
            17 | 18 => Some(Self::Dgps),
            16 => Some(Self::Single),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Fixed => "FIX",
            Self::FixedOneSided => "片方向 FIX",
            Self::FixedUnused => "FIX（不採用）",
            Self::Float => "FLOAT",
            Self::Dgps => "DGPS",
            Self::Single => "単独",
        }
    }

    /// What the state means, for a tooltip.
    pub fn hint(self) -> &'static str {
        match self {
            Self::Fixed => {
                "整数の不確定性が解けた解（PPK では前方解と後方解の両方で FIX）。軌跡の最適化で使う"
            }
            Self::FixedOneSided => {
                "前方解か後方解の片方だけで FIX。もう片方の解と近いので、軌跡の最適化で使う"
            }
            Self::FixedUnused => {
                "FIX だが軌跡の最適化で使わない解。ほとんどは片方向だけの FIX で、もう片方の解と離れていたもの\
                 （どちらが正しいか決められないため）。標準偏差が上限を超えたものも含む"
            }
            Self::Float => "整数の不確定性が解けていない解。軌跡の最適化では使わない（設定による）",
            Self::Dgps => "コード差分の解",
            Self::Single => "単独測位の解",
        }
    }
}

/// The position solutions over time: their UTC (s since 1970) and state, by time.
#[derive(Clone, Debug, Default)]
pub struct Positioning {
    /// where they come from: the PPK result or the receiver's real-time RTK
    pub source: String,
    pub states: Vec<(f64, Fix)>,
}

impl Positioning {
    /// The share of each state, as [`Fix::ALL`].
    pub fn shares(&self) -> [f64; Fix::ALL.len()] {
        let n = self.states.len().max(1) as f64;
        Fix::ALL.map(|f| self.states.iter().filter(|s| s.1 == f).count() as f64 / n)
    }

    /// The state at UTC `t`: of the nearest solution within 2 s.
    pub fn at(&self, t: f64) -> Option<Fix> {
        let i = self.states.partition_point(|s| s.0 < t);
        [i.checked_sub(1), Some(i)]
            .into_iter()
            .flatten()
            .filter_map(|i| self.states.get(i))
            .filter(|s| (s.0 - t).abs() <= 2.0)
            .min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs()))
            .map(|s| s.1)
    }
}

/// The receiver's clock against UTC, from the synchronisation in the raw folder's info bags.
pub struct SyncClock {
    /// (device time, UTC), by device time
    by_device: Vec<(f64, f64)>,
}

impl SyncClock {
    pub fn read(raw: &Path, job: &Job) -> Result<Self> {
        let mut info: Vec<_> = std::fs::read_dir(raw.join("info"))?
            .filter_map(|f| f.ok().map(|f| f.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "bag"))
            .collect();
        info.sort();
        let mut by_device: Vec<(f64, f64)> = clocks(&info, job)?
            .into_iter()
            .map(|(u, d)| (d, u))
            .collect();
        by_device.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(Self { by_device })
    }

    /// The UTC of a device time (none outside the synchronisation or across a break in it).
    pub fn utc(&self, device: f64) -> Option<f64> {
        device_time(&self.by_device, device)
    }
}

/// What a message type is: Unicore message ids of the rover, RTCM3 message numbers of the base.
pub fn message_name(rover: bool, id: u16) -> &'static str {
    if rover {
        return match id {
            12 => "OBSVM 観測",
            106 => "GPSEPH 軌道",
            107 => "GLOEPH 軌道",
            108 => "BDSEPH 軌道",
            109 => "GALEPH 軌道",
            110 => "QZSSEPH 軌道",
            112 => "IRNSSEPH 軌道",
            2999 => "BD3EPH 軌道",
            _ => "",
        };
    }
    match id {
        1005 => "基準局の位置",
        1006 => "基準局の位置（アンテナ高つき）",
        1007 | 1008 | 1033 => "アンテナ・受信機の情報",
        1019 => "GPS 軌道",
        1020 => "GLONASS 軌道",
        1042 => "BeiDou 軌道",
        1044 => "QZSS 軌道",
        1045 | 1046 => "Galileo 軌道",
        1074..=1077 => "GPS 観測（MSM）",
        1084..=1087 => "GLONASS 観測（MSM）",
        1094..=1097 => "Galileo 観測（MSM）",
        1104..=1107 => "SBAS 観測（MSM）",
        1114..=1117 => "QZSS 観測（MSM）",
        1124..=1127 => "BeiDou 観測（MSM）",
        1230 => "GLONASS コードと位相のずれ",
        _ => "",
    }
}

/// `len` bits from bit `pos` of `b`, big endian.
fn bits(b: &[u8], pos: usize, len: usize) -> u64 {
    (pos..pos + len).fold(0, |v, i| (v << 1) | ((b[i / 8] >> (7 - i % 8)) & 1) as u64)
}

fn signed_bits(b: &[u8], pos: usize, len: usize) -> i64 {
    let v = bits(b, pos, len) as i64;
    if v >> (len - 1) & 1 == 1 {
        v - (1 << len)
    } else {
        v
    }
}

/// The antenna reference point of an RTCM 1005/1006 message (a whole frame), ECEF (m).
fn reference_point(frame: &[u8]) -> Option<[f64; 3]> {
    let p = 24; // after the frame header
    if frame.len() < 3 + 19 + 3 || !matches!(bits(frame, p, 12), 1005 | 1006) {
        return None;
    }
    let x = signed_bits(frame, p + 34, 38);
    let y = signed_bits(frame, p + 74, 38);
    let z = signed_bits(frame, p + 114, 38);
    Some([x, y, z].map(|v| v as f64 * 1e-4))
}

unsafe extern "C" {
    fn s20_inspect_begin(rcv: i32, week: i32, tow: f64) -> i32;
    fn s20_inspect_frame(
        b: *const u8,
        len: i32,
        rcv: i32,
        utc: *mut f64,
        sat: *mut i32,
        sys: *mut i32,
        snr: *mut f64,
        sig: *mut i32,
        slp: *mut i32,
        max: i32,
    ) -> i32;
    fn s20_inspect_end();
    fn s20_ecef_to_llh(r: *const f64, llh: *mut f64);
}

struct InspectGuard;
impl Drop for InspectGuard {
    fn drop(&mut self) {
        unsafe { s20_inspect_end() }
    }
}

/// The index of a system in [`SYSTEMS`] from RTKLIB's flag.
fn system(flag: i32) -> usize {
    match flag {
        0x01 => 0,
        0x04 => 1,
        0x08 => 2,
        0x20 => 3,
        0x10 => 4,
        _ => 5,
    }
}

/// Satellites of an epoch being gathered (a base epoch comes in MSM fragments).
#[derive(Default)]
struct Gathering {
    utc: f64,
    sats: BTreeMap<i32, (usize, f64, i32, bool)>,
}

impl Gathering {
    fn epoch(&self) -> RawEpoch {
        let mut e = RawEpoch {
            utc: self.utc,
            ..Default::default()
        };
        let mut snr = 0.0;
        for &(sys, s, sig, slip) in self.sats.values() {
            e.sats[sys] += 1;
            snr += s;
            e.signals += sig as u16;
            e.slips += slip as u16;
        }
        e.snr = (snr / self.sats.len().max(1) as f64) as f32;
        e
    }
}

/// The GPS week and time of week of the first rover frame.
fn rover_time(b: &[u8], f: &Frames) -> Option<(i32, f64)> {
    let &(i, _) = f.good.first()?;
    let week = u16::from_le_bytes([b[i + 10], b[i + 11]]) as i32;
    let tow = u32::from_le_bytes(b[i + 12..i + 16].try_into().ok()?) as f64 * 0.001;
    (1800..=8191).contains(&week).then_some((week, tow))
}

/// GPS week and time of week of a UTC time (s since 1970), to start a decoder near it.
fn gps_time(utc: f64) -> (i32, f64) {
    // GPS time is ahead of UTC by the leap seconds (18 since 2017)
    let gps = utc - 315_964_800.0 + 18.0;
    let week = (gps / 604_800.0).floor();
    (week as i32, gps - week * 604_800.0)
}

fn summarise(b: &[u8], rover: bool, time: (i32, f64), job: &Job) -> Result<RawSummary> {
    let f = frames(b, rover, job)?;
    let mut s = RawSummary {
        bytes: b.len() as u64,
        frames: f.good.len(),
        bad: f.bad,
        ..Default::default()
    };
    let rcv = if rover { 1 } else { 2 };
    ensure!(
        unsafe { s20_inspect_begin(rcv, time.0, time.1) } != 0,
        "GNSS decoder allocation failed"
    );
    let _guard = InspectGuard;
    const MAX: usize = 256;
    let (mut sat, mut sys, mut sig, mut slp) = ([0i32; MAX], [0i32; MAX], [0i32; MAX], [0i32; MAX]);
    let mut snr = [0f64; MAX];
    let mut gathering: Option<Gathering> = None;
    for (k, &(i, n)) in f.good.iter().enumerate() {
        if k % 1024 == 0 {
            job.check()?;
            job.report("GNSS", k as u64, f.good.len() as u64);
        }
        let frame = &b[i..i + n];
        let id = if rover {
            u16::from_le_bytes([frame[4], frame[5]])
        } else {
            bits(frame, 24, 12) as u16
        };
        *s.messages.entry(id).or_default() += 1;
        if !rover
            && s.position.is_none()
            && let Some(r) = reference_point(frame)
        {
            let mut llh = [0.0; 3];
            unsafe { s20_ecef_to_llh(r.as_ptr(), llh.as_mut_ptr()) };
            s.position = Some((r, llh));
        }
        let mut utc = 0.0;
        let got = unsafe {
            s20_inspect_frame(
                frame.as_ptr(),
                n as i32,
                rcv,
                &mut utc,
                sat.as_mut_ptr(),
                sys.as_mut_ptr(),
                snr.as_mut_ptr(),
                sig.as_mut_ptr(),
                slp.as_mut_ptr(),
                MAX as i32,
            )
        };
        if got < 0 {
            s.errors += 1;
            continue;
        }
        if got == 0 {
            continue;
        }
        // a new time ends the epoch gathered
        if gathering
            .as_ref()
            .is_some_and(|g| (g.utc - utc).abs() > 1e-4)
        {
            s.epochs.push(gathering.take().unwrap().epoch());
        }
        let g = gathering.get_or_insert_with(|| Gathering {
            utc,
            ..Default::default()
        });
        for j in 0..got as usize {
            let e = g
                .sats
                .entry(sat[j])
                .or_insert((system(sys[j]), 0.0, 0, false));
            e.1 = e.1.max(snr[j]);
            e.2 = e.2.max(sig[j]);
            e.3 |= slp[j] != 0;
        }
    }
    if let Some(g) = gathering {
        s.epochs.push(g.epoch());
    }
    // epochs in time order (a decoder may repeat or reorder a few)
    s.epochs.sort_by(|a, b| a.utc.total_cmp(&b.utc));
    s.epochs.dedup_by(|a, b| (a.utc - b.utc).abs() < 1e-4);
    Ok(s)
}

/// The rover's and the base's raw files of a raw folder, each when present. The base's decoder
/// starts at the rover's first time (RTCM gives only times of week), else at the file's date.
pub fn inspect(raw: &Path, job: &Job) -> Result<(Option<RawSummary>, Option<RawSummary>)> {
    let read = |name: &str| -> Result<Option<Vec<u8>>> {
        let p = raw.join(name);
        if !p.exists() {
            return Ok(None);
        }
        Ok(Some(
            std::fs::read(&p).with_context(|| format!("read {}", p.display()))?,
        ))
    };
    let rover = read("ROVER_PPKRAW.bin")?;
    let base = read("BASE.BIN")?;
    // RTKLIB decoders share state with PPK
    let _lock = SOLVER
        .lock()
        .map_err(|_| anyhow::anyhow!("GNSS decoder lock poisoned"))?;
    let mut start = None;
    let rover = match rover {
        Some(b) => {
            let f = frames(&b, true, job)?;
            start = rover_time(&b, &f);
            let time = start.context("no GNSS time in the rover's first frame")?;
            Some(summarise(&b, true, time, job)?)
        }
        None => None,
    };
    let base = match base {
        Some(b) => {
            let time = start.unwrap_or_else(|| {
                let modified = std::fs::metadata(raw.join("BASE.BIN"))
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0.0, |d| d.as_secs_f64());
                gps_time(modified)
            });
            Some(summarise(&b, false, time, job)?)
        }
        None => None,
    };
    Ok((rover, base))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtcm_reference_points_are_read_from_their_bits() {
        // a 1005 frame: station 1, X = 1.2345 m, Y = -2.5 m, Z = 3 m
        let mut bits_out: Vec<u8> = Vec::new();
        let mut push = |v: i64, len: usize| {
            for i in (0..len).rev() {
                bits_out.push(((v >> i) & 1) as u8);
            }
        };
        push(1005, 12);
        push(1, 12);
        push(0, 6);
        push(0, 4);
        push(12345, 38);
        push(0, 2);
        push(-25000, 38);
        push(0, 2);
        push(30000, 38);
        let payload: Vec<u8> = bits_out
            .chunks(8)
            .map(|c| c.iter().enumerate().fold(0u8, |v, (i, b)| v | b << (7 - i)))
            .collect();
        let mut frame = vec![0xd3, 0, payload.len() as u8];
        frame.extend(&payload);
        frame.extend([0, 0, 0]);
        let r = reference_point(&frame).unwrap();
        assert!(
            (r[0] - 1.2345).abs() < 1e-9 && (r[1] + 2.5).abs() < 1e-9 && (r[2] - 3.0).abs() < 1e-9
        );
        assert_eq!(message_name(false, 1077), "GPS 観測（MSM）");
        assert_eq!(message_name(true, 12), "OBSVM 観測");
    }

    #[test]
    fn solution_states_and_their_shares() {
        assert_eq!(Fix::of_receiver(50), Some(Fix::Fixed));
        assert_eq!(Fix::of_receiver(34), Some(Fix::Float));
        assert_eq!(Fix::of_receiver(0), None);
        assert_eq!(Fix::of_rtklib(2), Some(Fix::Float));
        let p = Positioning {
            source: "PPK".into(),
            states: vec![
                (10.0, Fix::Fixed),
                (11.0, Fix::Fixed),
                (12.0, Fix::Float),
                (20.0, Fix::Single),
            ],
        };
        assert_eq!(p.shares(), [0.5, 0.0, 0.0, 0.25, 0.0, 0.25]);
        assert_eq!(p.at(11.6), Some(Fix::Float));
        assert_eq!(p.at(16.0), None);
    }

    #[test]
    fn ppk_fixes_are_told_apart_by_the_directions_that_fixed_them() {
        let p = super::super::Params::default();
        let sol = |utc: f64, status: u8, sd: f64| super::super::Solution {
            utc,
            t: utc,
            llh: [35.0, 139.0, 50.0],
            sd: [sd, sd, sd],
            ratio: 10.0,
            age: 0.5,
            status,
            satellites: 20,
        };
        let r = super::super::ResultFile {
            engine: String::new(),
            params: p.clone(),
            inputs: Vec::new(),
            rover_bad_frames: 0,
            base_bad_frames: 0,
            solutions: vec![
                sol(1.0, 1, 0.01),
                sol(2.0, 1, 0.01),
                sol(3.0, 1, 0.13),
                sol(4.0, 2, 0.3),
            ],
            forward_solutions: vec![
                sol(1.0, 1, 0.01),
                sol(2.0, 1, 0.01),
                sol(3.0, 1, 0.01),
                sol(4.0, 2, 0.3),
            ],
            // the backward ones in reverse order, a little off in time
            backward_solutions: vec![
                sol(4.01, 2, 0.3),
                sol(3.01, 2, 0.3),
                sol(2.01, 2, 0.3),
                sol(1.01, 1, 0.01),
            ],
            forward_fixed_epochs: 0,
            backward_fixed_epochs: 0,
            inconsistent_fixed_epochs: 0,
            inconsistent_mixed_epochs: 0,
        };
        let d = r.directions();
        assert!(d.iter().all(|x| x[0].is_some() && x[1].is_some()));
        assert_eq!(d[1][1].unwrap().utc, 2.01);
        let states: Vec<Option<Fix>> = r
            .solutions
            .iter()
            .zip(d)
            .map(|(s, d)| Fix::of_ppk(s, d, &p))
            .collect();
        assert_eq!(
            states,
            [
                Some(Fix::Fixed),
                Some(Fix::FixedOneSided),
                Some(Fix::FixedUnused),
                Some(Fix::Float)
            ]
        );
        // without the directional solutions (older results) a used FIX is a FIX
        assert_eq!(
            Fix::of_ppk(&r.solutions[1], [None, None], &p),
            Some(Fix::Fixed)
        );
        let (h, v) = sol(0.0, 1, 0.0).gap_m(&super::super::Solution {
            llh: [35.000001, 139.0, 50.1],
            ..sol(0.0, 1, 0.0)
        });
        assert!(
            (h - 0.111).abs() < 0.001 && (v + 0.1).abs() < 1e-9,
            "{h} {v}"
        );
        // the device clock back to UTC
        let c = SyncClock {
            by_device: vec![(100.0, 1000.0), (101.0, 1001.0)],
        };
        assert_eq!(c.utc(100.5), Some(1000.5));
        assert_eq!(c.utc(99.0), None);
    }

    #[test]
    fn utc_turns_into_gps_week_and_time_of_week() {
        // 2026-10-03 05:08:34 UTC
        let (week, tow) = gps_time(1_791_004_114.0);
        assert_eq!(week, 2438);
        assert!((tow - 536_932.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod real_data {
    use super::*;

    #[test]
    #[ignore = "reads the raw folders of the local multi_park work folder; prints their GNSS summary"]
    fn summarise_a_recording() {
        let work = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/multi_park/project.json");
        let project: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(work).unwrap()).unwrap();
        for session in project["sessions"].as_array().unwrap() {
            summarise_one(Path::new(session["raw"].as_str().unwrap()));
        }
    }

    fn summarise_one(raw: &Path) {
        eprintln!("{}", raw.display());
        let started = std::time::Instant::now();
        let (rover, base) = inspect(raw, &Job::stderr()).unwrap();
        eprintln!("{:.1} s", started.elapsed().as_secs_f64());
        for (name, s, is_rover) in [("rover", rover, true), ("base", base, false)] {
            let Some(s) = s else {
                eprintln!("{name}: none");
                continue;
            };
            let first = s.epochs.first().map_or(0.0, |e| e.utc);
            let last = s.epochs.last().map_or(0.0, |e| e.utc);
            eprintln!(
                "{name}: {} bytes, {} frames, {} bad, {} errors, {} epochs over {:.0} s, interval {:?}, max gap {:.1} s, position {:?}",
                s.bytes,
                s.frames,
                s.bad,
                s.errors,
                s.epochs.len(),
                last - first,
                s.interval(),
                s.max_gap(),
                s.position
            );
            for (id, n) in &s.messages {
                eprintln!("  {id} {} x{n}", message_name(is_rover, *id));
            }
            if let Some(e) = s.epochs.get(s.epochs.len() / 2) {
                eprintln!("  middle epoch: {e:?}");
            }
        }
    }
}
