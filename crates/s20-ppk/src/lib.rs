//! Embedded RTKLIB PPK for S20 Unicore rover + RTCM3 base recordings.
//! Solutions retain UTC, device time, ellipsoidal height, status and ENU uncertainty.
use anyhow::{Context, Result, ensure};
use s20_io::{
    bag::Bag,
    job::Job,
    msgs::{Cursor, header},
};
use serde::{Deserialize, Serialize};
use std::{
    ffi::c_void,
    path::{Path, PathBuf},
    sync::Mutex,
};

pub mod inspect;

pub const ENGINE: &str = "rtklib-demo5-75a2e562/s20-16";
pub const LICENSE: &str = include_str!("../vendor/rtklib/license.txt");
static SOLVER: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Params {
    /// Rover output/filter sampling interval; all input frames still decoded for cycle slips.
    pub interval: f64,
    pub elevation_deg: f64,
    pub ar_ratio: f64,
    /// Lower bounds (m) on the horizontal / vertical standard deviation used when
    /// forward and backward solutions are checked for agreement (4 sigma, ENU).
    /// FIX covariance is often millimetres, below the accuracy of the corrections.
    pub fix_check_sd_h: f64,
    pub fix_check_sd_v: f64,
    /// Lower bounds (m) on the FLOAT side's horizontal / vertical standard deviation, added to
    /// the above, when a FIX of one direction is checked against FLOAT of the other. FLOAT
    /// covariance is millimetres while converging FLOAT is often decimetres off.
    pub float_check_sd_h: f64,
    pub float_check_sd_v: f64,
    /// SNR-dependent phase error at RTKLIB's 52 dB-Hz reference; zero disables it.
    pub phase_snr_error: f64,
    /// Prefer base GPS 2X for satellites tracked only as 2S by the rover.
    pub prefer_base_l2c: bool,
    /// Estimate differential ionosphere (useful for longer baselines).
    pub estimate_ionosphere: bool,
    pub estimate_troposphere: bool,
    pub frequencies: u8,
    pub gps: bool,
    pub glonass: bool,
    pub galileo: bool,
    pub beidou: bool,
    pub qzss: bool,
    pub use_float: bool,
    pub float_sigma_h: f64,
    pub float_sigma_v: f64,
    pub max_sd_h: f64,
    pub max_sd_v: f64,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            interval: 0.1,
            elevation_deg: 15.,
            ar_ratio: 3.,
            fix_check_sd_h: 0.01,
            fix_check_sd_v: 0.02,
            float_check_sd_h: 0.10,
            float_check_sd_v: 0.20,
            phase_snr_error: 0.,
            prefer_base_l2c: false,
            estimate_ionosphere: false,
            estimate_troposphere: false,
            frequencies: 3,
            gps: true,
            glonass: true,
            galileo: true,
            beidou: true,
            qzss: true,
            use_float: false,
            float_sigma_h: 0.20,
            float_sigma_v: 0.40,
            max_sd_h: 0.10,
            max_sd_v: 0.20,
        }
    }
}
impl Params {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.interval.is_finite() && (0.05..=10.).contains(&self.interval),
            "PPK interval must be 0.05..10 s"
        );
        ensure!(
            (self.interval / 0.05 - (self.interval / 0.05).round()).abs() < 1e-6,
            "PPK interval must be a multiple of the S20 observation period (0.05 s)"
        );
        ensure!(
            self.elevation_deg.is_finite() && (5.0..=45.).contains(&self.elevation_deg),
            "PPK elevation must be 5..45 deg"
        );
        ensure!(
            self.ar_ratio.is_finite() && self.ar_ratio >= 2.,
            "PPK AR ratio must be >= 2"
        );
        ensure!(
            self.fix_check_sd_h.is_finite()
                && (0.0..=0.1).contains(&self.fix_check_sd_h)
                && self.fix_check_sd_v.is_finite()
                && (0.0..=0.2).contains(&self.fix_check_sd_v),
            "PPK forward/backward check sigma floors must be 0..0.1 m (horizontal) and 0..0.2 m (vertical)"
        );
        ensure!(
            self.float_check_sd_h.is_finite()
                && (0.0..=0.5).contains(&self.float_check_sd_h)
                && self.float_check_sd_v.is_finite()
                && (0.0..=1.0).contains(&self.float_check_sd_v),
            "PPK FIX/FLOAT check sigma floors must be 0..0.5 m (horizontal) and 0..1 m (vertical)"
        );
        ensure!(
            self.phase_snr_error.is_finite() && (0.0..=0.1).contains(&self.phase_snr_error),
            "PPK SNR phase error must be 0..0.1 m"
        );
        ensure!(
            self.max_sd_h.is_finite()
                && self.max_sd_h > 0.
                && self.max_sd_v.is_finite()
                && self.max_sd_v > 0.,
            "PPK uncertainty limits must be positive"
        );
        ensure!(
            self.float_sigma_h.is_finite()
                && self.float_sigma_h > 0.
                && self.float_sigma_v.is_finite()
                && self.float_sigma_v > 0.,
            "PPK FLOAT uncertainty floors must be positive"
        );
        ensure!(
            (1..=3).contains(&self.frequencies)
                && (!self.estimate_ionosphere || self.frequencies >= 2),
            "PPK needs 1..3 frequencies; ionosphere estimation needs >=2"
        );
        ensure!(
            self.systems() != 0,
            "PPK requires at least one satellite constellation"
        );
        Ok(())
    }
    fn systems(&self) -> i32 {
        self.gps as i32
            | ((self.glonass as i32) * 4)
            | ((self.galileo as i32) * 8)
            | ((self.qzss as i32) * 16)
            | ((self.beidou as i32) * 32)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Solution {
    pub utc: f64,
    pub t: f64,
    pub llh: [f64; 3],
    /// East, north, up standard deviations in metres.
    pub sd: [f64; 3],
    pub ratio: f64,
    pub age: f64,
    /// RTKLIB: 1 FIX, 2 FLOAT, 4 DGPS, 5 SINGLE.
    pub status: u8,
    pub satellites: u8,
}
impl Solution {
    pub fn accepted(&self, p: &Params) -> bool {
        (self.status == 1 || (p.use_float && self.status == 2))
            && self.llh.iter().chain(&self.sd).all(|x| x.is_finite())
            && self.sd.iter().all(|x| *x > 0.)
            && self.sd[0].hypot(self.sd[1]) <= p.max_sd_h
            && self.sd[2] <= p.max_sd_v
            && self.age.is_finite()
            && self.age.abs() <= 2.
    }

    /// The horizontal and vertical distances (m) from `other` (on a sphere: for showing).
    pub fn gap_m(&self, other: &Solution) -> (f64, f64) {
        const R: f64 = 6_378_137.0;
        let dn = (self.llh[0] - other.llh[0]).to_radians() * R;
        let de = (self.llh[1] - other.llh[1]).to_radians() * R * self.llh[0].to_radians().cos();
        (dn.hypot(de), self.llh[2] - other.llh[2])
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Input {
    path: PathBuf,
    bytes: u64,
    modified_ns: u128,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResultFile {
    pub engine: String,
    pub params: Params,
    inputs: Vec<Input>,
    pub rover_bad_frames: usize,
    pub base_bad_frames: usize,
    pub solutions: Vec<Solution>,
    /// Independent directional solutions for diagnosing convergence and false fixes.
    #[serde(default)]
    pub forward_solutions: Vec<Solution>,
    #[serde(default)]
    pub backward_solutions: Vec<Solution>,
    #[serde(default)]
    pub forward_fixed_epochs: usize,
    #[serde(default)]
    pub backward_fixed_epochs: usize,
    #[serde(default)]
    pub inconsistent_fixed_epochs: usize,
    /// FIX/FLOAT disagreements whose covariance includes directional spread.
    #[serde(default)]
    pub inconsistent_mixed_epochs: usize,
}
impl ResultFile {
    /// The forward and backward solutions of each combined one (the same epoch, within 25 ms),
    /// where the result keeps them (older results do not).
    pub fn directions(&self) -> Vec<[Option<&Solution>; 2]> {
        fn by_time(v: &[Solution]) -> Vec<&Solution> {
            let mut s: Vec<&Solution> = v.iter().collect();
            s.sort_by(|a, b| a.utc.total_cmp(&b.utc));
            s
        }
        fn at<'a>(v: &[&'a Solution], t: f64) -> Option<&'a Solution> {
            let i = v.partition_point(|s| s.utc < t);
            [i.checked_sub(1), Some(i)]
                .into_iter()
                .flatten()
                .filter_map(|i| v.get(i).copied())
                .filter(|s| (s.utc - t).abs() <= 0.025)
                .min_by(|a, b| (a.utc - t).abs().total_cmp(&(b.utc - t).abs()))
        }
        let (f, b) = (
            by_time(&self.forward_solutions),
            by_time(&self.backward_solutions),
        );
        self.solutions
            .iter()
            .map(|s| [at(&f, s.utc), at(&b, s.utc)])
            .collect()
    }

    pub fn fixes(&self, frame: &s20_io::geodesy::CoordinateFrame) -> Vec<s20_io::gnss::Fix> {
        self.solutions
            .iter()
            .filter(|s| s.accepted(&self.params) && s20_io::geodesy::valid_llh(s.llh))
            .map(|s| s20_io::gnss::Fix {
                t: s.t,
                p: frame.position(s.llh),
                sd: if s.status == 2 {
                    [
                        s.sd[0].max(self.params.float_sigma_h),
                        s.sd[1].max(self.params.float_sigma_h),
                        s.sd[2].max(self.params.float_sigma_v),
                    ]
                } else {
                    s.sd
                },
            })
            .collect()
    }
    pub fn summary(&self) -> String {
        let count = |status| self.solutions.iter().filter(|s| s.status == status).count();
        format!(
            "PPK: {} epochs, FIX {}, FLOAT {}, accepted {}; damaged frames rover {} / base {}; FIX/FIX rejected {}, FIX/FLOAT uncertainty increased {}",
            self.solutions.len(),
            count(1),
            count(2),
            self.solutions
                .iter()
                .filter(|s| s.accepted(&self.params) && s20_io::geodesy::valid_llh(s.llh))
                .count(),
            self.rover_bad_frames,
            self.base_bad_frames,
            self.inconsistent_fixed_epochs,
            self.inconsistent_mixed_epochs,
        )
    }
}
pub fn read(path: &Path) -> Result<ResultFile> {
    serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(path)?))
        .context("read PPK result")
}

fn write(path: &Path, result: &ResultFile) -> Result<()> {
    use std::io::Write;
    if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    serde_json::to_writer(&mut w, result)?;
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Piecewise-linear UTC -> device clock. Never silently extrapolate across missing sync data.
fn device_time(clock: &[(f64, f64)], utc: f64) -> Option<f64> {
    let i = clock.partition_point(|x| x.0 < utc);
    if i < clock.len() && (clock[i].0 - utc).abs() < 1e-6 {
        return Some(clock[i].1);
    }
    if i == 0 || i == clock.len() {
        return None;
    }
    let (a, b) = (clock[i - 1], clock[i]);
    if b.0 - a.0 > 3. || b.0 <= a.0 || b.1 <= a.1 || ((b.0 - a.0) - (b.1 - a.1)).abs() > 0.01 {
        return None;
    }
    Some(a.1 + (utc - a.0) / (b.0 - a.0) * (b.1 - a.1))
}
fn clocks(paths: &[PathBuf], job: &Job) -> Result<Vec<(f64, f64)>> {
    let mut out = Vec::new();
    for p in paths {
        job.check()?;
        let b = Bag::open(p)?;
        for m in b.topic_messages("/ssync/sync_time") {
            let mut c = Cursor::new(b.data(&m));
            header(&mut c)?;
            c.u16()?;
            c.u16()?;
            let device = c.u32()? as f64 + c.u32()? as f64 * 1e-9;
            c.bytes()?;
            let utc = c.u32()? as f64 + c.u32()? as f64 * 1e-9;
            if utc > 1_500_000_000. && device > 1_000_000_000. {
                out.push((utc, device));
            }
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out.dedup_by(|a, b| a.0 == b.0);
    ensure!(
        out.len() >= 2,
        "PPK requires /ssync/sync_time in info/*.bag; cannot align GNSS UTC with SLAM device clock"
    );
    Ok(out)
}

#[derive(Default, Debug)]
struct Frames {
    good: Vec<(usize, usize)>,
    bad: usize,
}
fn crc32(b: &[u8]) -> u32 {
    let mut c = 0u32;
    for &x in b {
        c ^= x as u32;
        for _ in 0..8 {
            c = (c >> 1) ^ if c & 1 != 0 { 0xedb88320 } else { 0 };
        }
    }
    c
}
fn crc24(b: &[u8]) -> u32 {
    let mut c = 0u32;
    for &x in b {
        c ^= (x as u32) << 16;
        for _ in 0..8 {
            c <<= 1;
            if c & 0x1000000 != 0 {
                c ^= 0x1864cfb;
            }
        }
    }
    c & 0xffffff
}
fn frames(b: &[u8], rover: bool, job: &Job) -> Result<Frames> {
    let mut f = Frames::default();
    let mut i = 0;
    while i < b.len() {
        if i % 65536 == 0 {
            job.check()?;
        }
        let sync = if rover {
            b[i..].starts_with(&[0xaa, 0x44, 0xb5])
        } else {
            b[i] == 0xd3
        };
        if !sync {
            i += 1;
            continue;
        }
        let n = if rover {
            if i + 24 > b.len() {
                f.bad += 1;
                break;
            }
            24 + u16::from_le_bytes([b[i + 6], b[i + 7]]) as usize + 4
        } else {
            if i + 3 > b.len() {
                f.bad += 1;
                break;
            }
            if b[i + 1] & 0xfc != 0 {
                i += 1;
                continue;
            }
            3 + (((b[i + 1] & 3) as usize) << 8) + b[i + 2] as usize + 3
        };
        if i + n > b.len() {
            f.bad += 1;
            i += 1;
            continue;
        }
        let v = &b[i..i + n];
        let valid = if rover {
            crc32(&v[..n - 4]) == u32::from_le_bytes(v[n - 4..].try_into().unwrap())
        } else {
            crc24(&v[..n - 3])
                == ((v[n - 3] as u32) << 16) | ((v[n - 2] as u32) << 8) | v[n - 1] as u32
        };
        if valid {
            f.good.push((i, n));
            i += n;
        } else {
            f.bad += 1;
            i += 1;
        }
    }
    ensure!(
        !f.good.is_empty(),
        "no valid {} frames",
        if rover { "Unicore" } else { "RTCM3" }
    );
    Ok(f)
}

struct ContextState<'a> {
    job: &'a Job,
    raw: Vec<[f64; 12]>,
    first: f64,
    last: f64,
    last_pass: i32,
    tick: std::time::Instant,
}
unsafe extern "C" fn progress(ctx: *mut c_void, pass: i32, utc: f64) -> i32 {
    // SAFETY: native callbacks are synchronous, under SOLVER, during process().
    let c = unsafe { &mut *(ctx as *mut ContextState<'_>) };
    if (1..=2).contains(&pass) && pass != c.last_pass {
        c.last_pass = pass;
        c.job.log(if pass == 1 {
            "PPK: forward pass started"
        } else {
            "PPK: backward pass started"
        });
    }
    if utc >= 0. && c.tick.elapsed().as_millis() >= 250 {
        let span = (c.last - c.first).max(1.);
        let f = if pass == 2 {
            (c.last - utc) / span
        } else {
            (utc - c.first) / span
        };
        c.job.report(
            if pass == 2 {
                "ppk backward"
            } else {
                "ppk forward"
            },
            (f.clamp(0., 1.) * 1000.) as u64,
            1000,
        );
        c.tick = std::time::Instant::now();
    }
    c.job.cancel.load(std::sync::atomic::Ordering::Relaxed) as i32
}
unsafe extern "C" fn solution(ctx: *mut c_void, v: *const f64) {
    let c = unsafe { &mut *(ctx as *mut ContextState<'_>) };
    let mut row = [0.; 12];
    row.copy_from_slice(unsafe { std::slice::from_raw_parts(v, 12) });
    c.raw.push(row);
}
unsafe extern "C" {
    fn s20_ppk_begin(
        week: i32,
        tow: f64,
        dt: f64,
        ctx: *mut c_void,
        p: unsafe extern "C" fn(*mut c_void, i32, f64) -> i32,
        s: unsafe extern "C" fn(*mut c_void, *const f64),
    ) -> i32;
    fn s20_ppk_frame(b: *const u8, len: i32, rcv: i32) -> i32;
    fn s20_ppk_solve(
        elevation: f64,
        ratio: f64,
        estimate_ionosphere: i32,
        systems: i32,
        frequencies: i32,
        estimate_troposphere: i32,
        phase_snr_error: f64,
        prefer_base_l2c: i32,
        fix_check_sd_h: f64,
        fix_check_sd_v: f64,
        float_check_sd_h: f64,
        float_check_sd_v: f64,
    ) -> i32;
    fn s20_ppk_stats(stats: *mut i32);
    fn s20_ppk_end();
}
struct NativeGuard;
impl Drop for NativeGuard {
    fn drop(&mut self) {
        unsafe {
            s20_ppk_end();
        }
    }
}

pub fn process(raw: &Path, out: &Path, p: &Params, job: &Job) -> Result<ResultFile> {
    p.validate()?;
    job.check()?;
    let rover_path = raw.join("ROVER_PPKRAW.bin");
    let base_path = raw.join("BASE.BIN");
    let mut info: Vec<_> = std::fs::read_dir(raw.join("info"))?
        .filter_map(|f| f.ok().map(|f| f.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "bag"))
        .collect();
    info.sort();
    let mut inputs = Vec::new();
    for path in [&rover_path, &base_path].into_iter().chain(info.iter()) {
        let m = std::fs::metadata(path).with_context(|| format!("PPK input {}", path.display()))?;
        inputs.push(Input {
            path: std::fs::canonicalize(path)?,
            bytes: m.len(),
            modified_ns: m
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos(),
        });
    }
    if let Ok(mut r) = read(out)
        && r.engine == ENGINE
        && r.params.interval == p.interval
        && r.params.elevation_deg == p.elevation_deg
        && r.params.ar_ratio == p.ar_ratio
        && r.params.fix_check_sd_h == p.fix_check_sd_h
        && r.params.fix_check_sd_v == p.fix_check_sd_v
        && r.params.float_check_sd_h == p.float_check_sd_h
        && r.params.float_check_sd_v == p.float_check_sd_v
        && r.params.phase_snr_error == p.phase_snr_error
        && r.params.prefer_base_l2c == p.prefer_base_l2c
        && r.params.estimate_ionosphere == p.estimate_ionosphere
        && r.params.estimate_troposphere == p.estimate_troposphere
        && r.params.systems() == p.systems()
        && r.params.frequencies == p.frequencies
        && r.inputs == inputs
        && !r.solutions.is_empty()
    {
        if r.params != *p {
            r.params = p.clone();
            write(out, &r)?;
        }
        job.log("PPK: using cached solution");
        job.log(&r.summary());
        return Ok(r);
    }
    let clock = clocks(&info, job)?;
    let rb = std::fs::read(&rover_path)?;
    let bb = std::fs::read(&base_path)?;
    job.log("PPK: validating Unicore / RTCM3 frames");
    let rf = frames(&rb, true, job)?;
    let bf = frames(&bb, false, job)?;
    let first = rf.good.first().unwrap().0;
    let week = u16::from_le_bytes([rb[first + 10], rb[first + 11]]) as i32;
    let tow = u32::from_le_bytes(rb[first + 12..first + 16].try_into()?) as f64 * 0.001;
    ensure!((1800..=8191).contains(&week), "invalid GNSS week {week}");
    let mut c = ContextState {
        job,
        raw: Vec::new(),
        first: clock[0].0,
        last: clock.last().unwrap().0,
        last_pass: 0,
        tick: std::time::Instant::now(),
    };
    // RTKLIB postpos globals are never accessed concurrently.
    let _lock = SOLVER
        .lock()
        .map_err(|_| anyhow::anyhow!("PPK solver lock poisoned"))?;
    let _guard = NativeGuard;
    ensure!(
        unsafe {
            s20_ppk_begin(
                week,
                tow,
                p.interval,
                &mut c as *mut _ as *mut c_void,
                progress,
                solution,
            )
        } != 0,
        "PPK engine allocation failed"
    );
    for (rcv, data, f) in [(1, &rb, &rf), (2, &bb, &bf)] {
        let mut errors = 0;
        let mut error_messages = std::collections::BTreeMap::<u16, usize>::new();
        for (k, &(i, n)) in f.good.iter().enumerate() {
            if k % 128 == 0 {
                job.check()?;
                job.report("ppk decode", k as u64, f.good.len() as u64);
            }
            let ret = unsafe { s20_ppk_frame(data[i..i + n].as_ptr(), n as i32, rcv) };
            ensure!(ret != 0, "PPK observation allocation failed");
            if ret < 0 {
                errors += 1;
                let message = if rcv == 1 {
                    u16::from_le_bytes([data[i + 4], data[i + 5]])
                } else {
                    ((data[i + 3] as u16) << 4) | (data[i + 4] as u16 >> 4)
                };
                *error_messages.entry(message).or_default() += 1;
            }
        }
        job.log(&format!("PPK {}: {} valid frames, {} CRC/truncation errors, {} unsupported/invalid decoded messages",
            if rcv==1 {"rover"}else{"base"},f.good.len(),f.bad,errors));
        if !error_messages.is_empty() {
            job.log(&format!(
                "PPK decoded error counts by message ID: {error_messages:?}"
            ));
        }
    }
    job.log("PPK: forward / backward kinematic solution (GPS, GLONASS, Galileo, BeiDou, QZSS)");
    let n = unsafe {
        s20_ppk_solve(
            p.elevation_deg,
            p.ar_ratio,
            p.estimate_ionosphere as i32,
            p.systems(),
            p.frequencies as i32,
            p.estimate_troposphere as i32,
            p.phase_snr_error,
            p.prefer_base_l2c as i32,
            p.fix_check_sd_h,
            p.fix_check_sd_v,
            p.float_check_sd_h,
            p.float_check_sd_v,
        )
    };
    let mut stats = [0i32; 4];
    unsafe {
        s20_ppk_stats(stats.as_mut_ptr());
    }
    job.check()?;
    ensure!(
        n > 0 && !c.raw.is_empty(),
        "PPK produced no solution (code {n}); check common observations, ephemerides and base position"
    );
    let mut directions = [Vec::new(), Vec::new(), Vec::new()];
    for r in c
        .raw
        .into_iter()
        .filter(|r| r.iter().all(|x| x.is_finite()))
        .filter_map(|r| {
            device_time(&clock, r[0]).map(|t| {
                (
                    r[11] as usize,
                    Solution {
                        utc: r[0],
                        t,
                        llh: [r[1], r[2], r[3]],
                        sd: [r[4], r[5], r[6]],
                        ratio: r[7],
                        age: r[8],
                        status: r[9] as u8,
                        satellites: r[10] as u8,
                    },
                )
            })
        })
    {
        if (1..=3).contains(&r.0) {
            directions[r.0 - 1].push(r.1);
        }
    }
    let [forward_solutions, backward_solutions, solutions] = directions;
    ensure!(
        !solutions.is_empty(),
        "PPK solution has no overlap with valid SLAM clock synchronisation"
    );
    let result = ResultFile {
        engine: ENGINE.into(),
        params: p.clone(),
        inputs,
        rover_bad_frames: rf.bad,
        base_bad_frames: bf.bad,
        solutions,
        forward_solutions,
        backward_solutions,
        forward_fixed_epochs: stats[0] as usize,
        backward_fixed_epochs: stats[1] as usize,
        inconsistent_fixed_epochs: stats[2] as usize,
        inconsistent_mixed_epochs: stats[3] as usize,
    };
    write(out, &result)?;
    job.log(&result.summary());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_combination_validates_fix_float_in_both_directions() {
        unsafe extern "C" {
            fn s20_ppk_test_mixed_consistency() -> i32;
        }
        let _lock = SOLVER.lock().unwrap();
        assert_eq!(unsafe { s20_ppk_test_mixed_consistency() }, 1);
    }
    #[test]
    fn native_fix_check_accepts_centimetres_and_rejects_decimetres() {
        unsafe extern "C" {
            fn s20_ppk_test_fix_floor() -> i32;
        }
        let _lock = SOLVER.lock().unwrap();
        assert_eq!(unsafe { s20_ppk_test_fix_floor() }, 1);
    }
    #[test]
    fn native_fix_float_check_adds_float_floors() {
        unsafe extern "C" {
            fn s20_ppk_test_float_floor() -> i32;
        }
        let _lock = SOLVER.lock().unwrap();
        assert_eq!(unsafe { s20_ppk_test_float_floor() }, 1);
    }
    #[test]
    fn native_bds3_high_prns_decode_observations_and_ephemerides() {
        unsafe extern "C" {
            fn s20_ppk_test_bds3() -> i32;
        }
        let _lock = SOLVER.lock().unwrap();
        assert_eq!(unsafe { s20_ppk_test_bds3() }, 1);
    }
    #[test]
    fn snr_weight_settings_are_validated_and_missing_keys_take_defaults() {
        let mut p: Params = serde_json::from_str(r#"{"interval":1.0}"#).unwrap();
        assert_eq!(p.phase_snr_error, Params::default().phase_snr_error);
        assert!(!p.prefer_base_l2c);
        assert!(p.validate().is_ok());
        for invalid in [-0.001, 0.101, f64::NAN, f64::INFINITY] {
            p.phase_snr_error = invalid;
            assert!(p.validate().is_err());
        }
    }
    #[test]
    fn fix_check_floors_default_to_centimetres_and_are_validated() {
        let p: Params = serde_json::from_str(r#"{"interval":1.0}"#).unwrap();
        assert_eq!((p.fix_check_sd_h, p.fix_check_sd_v), (0.01, 0.02));
        assert!(p.validate().is_ok());
        for (h, v) in [
            (-0.001, 0.02),
            (0.101, 0.02),
            (f64::NAN, 0.02),
            (0.01, 0.201),
        ] {
            let q = Params {
                fix_check_sd_h: h,
                fix_check_sd_v: v,
                ..p.clone()
            };
            assert!(q.validate().is_err());
        }
        let off = Params {
            fix_check_sd_h: 0.,
            fix_check_sd_v: 0.,
            ..p
        };
        assert!(off.validate().is_ok());
    }
    #[test]
    fn float_check_floors_default_to_decimetres_and_are_validated() {
        let p: Params = serde_json::from_str(r#"{"interval":1.0}"#).unwrap();
        assert_eq!((p.float_check_sd_h, p.float_check_sd_v), (0.10, 0.20));
        for (h, v) in [(-0.001, 0.2), (0.501, 0.2), (f64::NAN, 0.2), (0.1, 1.001)] {
            let q = Params {
                float_check_sd_h: h,
                float_check_sd_v: v,
                ..p.clone()
            };
            assert!(q.validate().is_err());
        }
        let off = Params {
            float_check_sd_h: 0.,
            float_check_sd_v: 0.,
            ..p
        };
        assert!(off.validate().is_ok());
    }
    #[test]
    fn native_base_l2_matches_rover_without_tracking_family_switches() {
        unsafe extern "C" {
            fn s20_ppk_test_l2_selection() -> i32;
        }
        let _lock = SOLVER.lock().unwrap();
        assert_eq!(unsafe { s20_ppk_test_l2_selection() }, 1);
    }
    #[test]
    fn native_glonass_ar_off_excludes_integer_constraints() {
        unsafe extern "C" {
            fn s20_ppk_test_ar_off() -> i32;
        }
        let _lock = SOLVER.lock().unwrap();
        assert_eq!(unsafe { s20_ppk_test_ar_off() }, 1);
    }
    #[test]
    fn native_signal_mapping_and_slips_across_sampling() {
        unsafe extern "C" {
            fn s20_ppk_test_signals() -> i32;
        }
        let _lock = SOLVER.lock().unwrap();
        assert_eq!(unsafe { s20_ppk_test_signals() }, 1);
    }
    #[test]
    fn clock_interpolation_drift_and_gaps() {
        let c = [(100., 10.), (101., 10.999996), (105., 15.)];
        assert!((device_time(&c, 100.5).unwrap() - 10.499998).abs() < 1e-9);
        assert!(device_time(&c, 99.9).is_none());
        assert!(device_time(&c, 103.).is_none());
    }
    #[test]
    fn damaged_unicore_resynchronises() {
        let mut b = vec![0; 28];
        b[..3].copy_from_slice(&[0xaa, 0x44, 0xb5]);
        let crc = crc32(&b[..24]);
        b[24..].copy_from_slice(&crc.to_le_bytes());
        let mut bad = b.clone();
        bad[6] = 0xff;
        bad.extend_from_slice(&b);
        bad.extend_from_slice(&b[..16]);
        let f = frames(&bad, true, &Job::stderr()).unwrap();
        assert_eq!(f.good, vec![(28, 28)]);
        assert_eq!(f.bad, 2);
    }
    #[test]
    fn float_and_uncertainty_are_gated() {
        let mut s = Solution {
            utc: 0.,
            t: 0.,
            llh: [38., 141., 40.],
            sd: [0.01, 0.01, 0.02],
            ratio: 4.,
            age: 0.,
            status: 2,
            satellites: 20,
        };
        let mut p = Params::default();
        assert!(!s.accepted(&p));
        p.use_float = true;
        assert!(s.accepted(&p));
        s.sd[2] = 0.3;
        assert!(!s.accepted(&p));
        s.sd[2] = f64::NAN;
        assert!(!s.accepted(&p));
    }

    #[test]
    fn ppk_positions_use_the_same_local_frame_as_receiver_rtk() {
        let origin = [-33.9, 151.2, 35.];
        let frame = s20_io::geodesy::CoordinateFrame::local_enu(origin).unwrap();
        let mut file: ResultFile = serde_json::from_value(serde_json::json!({
            "engine": ENGINE, "inputs": [], "rover_bad_frames": 0, "base_bad_frames": 0,
            "params": Params::default(),
            "solutions": [{
                "utc": 0., "t": 1., "llh": origin, "sd": [0.01, 0.01, 0.02],
                "ratio": 4., "age": 0., "status": 1, "satellites": 20
            }]
        }))
        .unwrap();
        let fixes = file.fixes(&frame);
        assert_eq!(fixes.len(), 1);
        assert_eq!(fixes[0].p, [0.; 3]);
        file.solutions[0].llh[2] += 10.;
        assert!((file.fixes(&frame)[0].p[2] - 10.).abs() < 1e-8);
    }
}
