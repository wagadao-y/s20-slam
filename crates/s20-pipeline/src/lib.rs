//! The processing steps as library functions, shared by the CLI and the GUI:
//! extract -> lio -> opt -> cloud -> export -> colorize. Every step reports progress and checks
//! cancellation through a [`Job`].

use anyhow::{Context, Result, bail, ensure};
use s20_io::geodesy::CoordinateFrame;
use s20_io::job::Job;
use std::path::{Path, PathBuf};

mod at;
pub mod device;
pub mod export;
pub mod frames;
mod place;
pub mod revisits;
pub mod runlog;
pub use at::{At, Clock, PointAt};
pub use s20_io::job;
pub use s20_lio::LIDAR_IMU_T;
pub use s20_ppk as ppk;

/// Odometry: writes `<out>.txt`, `<out>_scans.txt`, `<out>_state.txt`.
/// `hold`: heading hold intervals, seconds from the first scan (negative: back from the last), see
/// [`Edits::heading_hold`].
pub fn lio(
    data: &Path,
    out: &Path,
    p: &s20_lio::Params,
    init_traj: Option<&Path>,
    hold: &[(f64, f64)],
    job: &Job,
) -> Result<()> {
    let l = s20_io::dataset::LidarData::open(data)?;
    let imu = s20_io::dataset::load_imu(data)?;
    let (t_first, t_last) = (l.scan_start(0), l.scan_start(l.len().saturating_sub(1)));
    let abs = |t: f64| if t < 0.0 { t_last + t } else { t_first + t };
    let hold: Vec<(f64, f64)> = hold.iter().map(|&(a, b)| (abs(a), abs(b))).collect();
    job.log(&format!(
        "{} scans, {} imu samples; {p:?}",
        l.len(),
        imu.len()
    ));
    let init = match init_traj {
        Some(f) => {
            let poses = s20_io::dataset::read_tum(f)?;
            let st = s20_io::dataset::read_table(&PathBuf::from(
                f.to_string_lossy().replace(".txt", "_state.txt"),
            ))?;
            Some(s20_lio::init_from(
                &l,
                p.scans.map_or(0, |s| s.0),
                &poses,
                &st,
            )?)
        }
        None => None,
    };
    let res = s20_lio::run(&l, &imu, p, init, &hold, job)?;
    s20_lio::save(out, &res)?;
    job.log(&format!(
        "saved {} scans {} imu-only updates {} heading held {} taken from the gate's second search {}",
        out.display(),
        res.poses.len(),
        res.n_skip,
        res.n_held,
        res.n_retried
    ));
    Ok(())
}

/// RTK use in the optimisation.
#[derive(Clone, Debug)]
pub struct RtkOptions {
    pub params: s20_opt::RtkParams,
    /// excluded intervals
    pub exclude: Vec<(At, At)>,
    /// Use computed PPK results (one per session) instead of receiver RTK. Positions are mutually
    /// exclusive.
    pub ppk: Vec<PathBuf>,
}

fn enu_frame(
    data: &Path,
    positions: impl IntoIterator<Item = [f64; 3]>,
) -> Result<CoordinateFrame> {
    if let CoordinateFrame::LocalEnu { origin_llh } = s20_io::dataset::load_frame(data)? {
        return CoordinateFrame::local_enu(origin_llh);
    }
    let origin = positions
        .into_iter()
        .find(|p| s20_io::geodesy::valid_llh(*p))
        .context("no usable GNSS positions for the local ENU origin")?;
    CoordinateFrame::local_enu(origin)
}

/// Unified optimisation of one or more sessions' LIO results (one file per session of `data`, see
/// `s20_io::dataset::sessions`): writes the dense trajectory `out` and `<out>_knots.txt`.
/// Loops are pairs of times at the same place (`start-end` is (0.5, -0.5)).
pub fn opt(
    data: &Path,
    lio_files: &[PathBuf],
    out: &Path,
    p: &s20_opt::Params,
    rtk: Option<&RtkOptions>,
    edits: &Edits,
    job: &Job,
) -> Result<()> {
    let loops = &edits.loops;
    let l = s20_io::dataset::LidarData::open(data)?;
    let imu = s20_io::dataset::load_imu(data)?;
    let dirs = s20_io::dataset::sessions(data)?;
    if lio_files.len() != dirs.len() {
        bail!(
            "{} has {} sessions, {} LIO results given",
            data.display(),
            dirs.len(),
            lio_files.len()
        );
    }
    let mut sessions = Vec::new();
    for (f, sd) in lio_files.iter().zip(&dirs) {
        // the LIO of a session runs on its own clock
        let mut poses = s20_io::dataset::read_tum(f)?;
        let mut st = s20_io::dataset::read_table(&PathBuf::from(
            f.to_string_lossy().replace(".txt", "_state.txt"),
        ))?;
        for r in &mut poses {
            r[0] += sd.shift;
        }
        for r in &mut st {
            if let Some(t) = r.first_mut() {
                *t += sd.shift;
            }
        }
        sessions.push((poses, st));
    }
    job.log(&format!("{p:?}"));
    let mut pr = s20_opt::Problem::new(&l, &imu, &sessions, p)?;
    job.log(&format!("{} knots, {} points", pr.kt.len(), pr.pts.len()));
    let scans = l.session_ranges();
    let clock = Clock {
        names: dirs
            .iter()
            .map(|d| s20_io::dataset::session_name(&d.dir))
            .collect(),
        range: (0..dirs.len())
            .map(|k| {
                let end = (0..pr.kt.len())
                    .rev()
                    .find(|&j| pr.session[j] == k)
                    .map_or(scans[k].1, |j| pr.kt[j]);
                (scans[k].0, end)
            })
            .collect(),
    };
    if dirs.len() > 1 {
        for (k, d) in dirs.iter().enumerate() {
            job.log(&format!(
                "session {k} {}: {:.1} s, shift {} s",
                clock.names[k],
                clock.range[k].1 - clock.range[k].0,
                d.shift
            ));
        }
    }
    // matched points, on the odometry as given (before any placement)
    let picks = edits
        .matches
        .iter()
        .map(|(a, b)| {
            Ok((
                place::Pick::resolve(a, &pr, &clock)?,
                place::Pick::resolve(b, &pr, &clock)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let matches = place::Matches::group(picks)?;
    // the sessions in the common frame: those placed by GNSS, else the first (the reference)
    let mut placed = vec![false; dirs.len()];
    placed[0] = rtk.is_none();
    let mut frame = CoordinateFrame::LocalSlam;
    if let Some(r) = rtk {
        let g = if r.ppk.is_empty() {
            s20_io::dataset::load_gnss(data)?
        } else {
            Vec::new()
        };
        let ppk = r
            .ppk
            .iter()
            .map(|f| s20_ppk::read(f))
            .collect::<Result<Vec<_>>>()?;
        if !ppk.is_empty() && ppk.len() != dirs.len() {
            bail!("{} PPK results for {} sessions", ppk.len(), dirs.len());
        }
        // PPK positions of every session on the common time axis
        let ppk_fixes = |frame: &CoordinateFrame| -> Vec<s20_io::gnss::Fix> {
            ppk.iter()
                .zip(&dirs)
                .flat_map(|(p, d)| {
                    p.fixes(frame).into_iter().map(|mut f| {
                        f.t += d.shift;
                        f
                    })
                })
                .collect()
        };
        frame = if ppk.is_empty() {
            let ok = s20_io::gnss::rtk_valid(&g, &s20_io::gnss::Validity::default());
            enu_frame(
                data,
                g.iter()
                    .zip(ok)
                    .filter(|(_, ok)| *ok)
                    .map(|(r, _)| [r[2], r[3], r[4] + r[8]]),
            )?
        } else {
            enu_frame(
                data,
                ppk.iter().flat_map(|p| {
                    p.solutions
                        .iter()
                        .filter(|s| s.accepted(&p.params))
                        .map(|s| s.llh)
                }),
            )?
        };
        job.log(&format!(
            "GNSS coordinate frame: {frame:?} (metres; z is local up, not elevation)"
        ));
        let mut fixes = if ppk.is_empty() {
            s20_io::gnss::load_fixes(&g, &frame, &s20_io::gnss::Validity::default())
        } else {
            ppk_fixes(&frame)
        };
        let exclude = r
            .exclude
            .iter()
            .map(|(a, b)| Ok((clock.abs(a)?, clock.abs(b)?)))
            .collect::<Result<Vec<_>>>()?;
        let n0 = fixes.len();
        fixes.retain(|f| !exclude.iter().any(|&(x, y)| f.t >= x && f.t < y));
        let g_set = pr.set_gnss(&fixes, p, r.params, !ppk.is_empty())?;
        let what = if ppk.is_empty() { "rtk" } else { "ppk" };
        let n_rej: usize = g_set.rejected.iter().map(|s| s.n).sum();
        let epochs = if ppk.is_empty() {
            g.len()
        } else {
            ppk.iter().map(|p| p.solutions.len()).sum()
        };
        job.log(&format!(
            "{what}: {} valid fixes of {epochs} epochs ({} excluded), {} used inside the trajectory ({n_rej} rejected as false fixes); initial alignment rms {:.3} m",
            n0,
            n0 - fixes.len(),
            g_set.used,
            g_set.rms
        ));
        for s in &g_set.rejected {
            let k = s20_io::dataset::session_at(&scans, s.t0);
            job.log(&format!(
                "{what}: false fix {} - {} ({} fixes), {:.2} m off the odometry",
                clock.label_at(k, s.t0),
                clock.label_at(k, s.t1),
                s.n,
                s.corr.norm()
            ));
        }
        for (k, pl) in placed.iter_mut().enumerate() {
            *pl = !g_set.unplaced.contains(&k);
        }
        for &k in &g_set.unplaced {
            job.log(&format!(
                "{what}: session {} has too few fixes, it is placed by its links",
                clock.names[k]
            ));
        }
        // pull the drifted odometry onto the fixes before the loops are measured on it
        pr.gnss_pose_graph(p, job)?;
    }
    // the other sessions through their links to the placed ones
    let matched = if dirs.len() > 1 {
        place::place_sessions(&mut pr, &clock, placed, loops, &matches, p, job)?
    } else {
        Vec::new()
    };
    // every loop is checked before stopping, so all the failing ones are reported at once
    let mut rejected = Vec::new();
    for (k, (a, b)) in loops.iter().enumerate() {
        job.check()?;
        job.report("opt loops", k as u64, loops.len() as u64);
        let name = format!("loop {} = {}", clock.label(a), clock.label(b));
        let across = clock.session(&a.session).ok() != clock.session(&b.session).ok();
        let fit = clock
            .abs(a)
            .and_then(|ta| Ok((ta, clock.abs(b)?)))
            .and_then(|(ta, tb)| pr.register_loop(ta, tb, &p.loop_reg));
        match fit {
            Ok(f) if s20_opt::links::loop_accepted(&f, across, &p.loop_reg) => {
                job.log(&format!(
                    "{name}: matched {:.0}% of the points, {:.2} m / {:.1} deg from the current trajectory",
                    f.inliers * 100.0,
                    f.gap_m,
                    f.gap_deg
                ));
                pr.loops.push(f.rel);
            }
            Ok(f) => {
                let s = format!(
                    "{name}: the local maps do not match ({:.0}% of the points, guesses {:.2} m apart)",
                    f.inliers * 100.0,
                    f.spread_m
                );
                job.log(&s);
                rejected.push(s);
            }
            Err(e) => {
                let s = format!("{name}: {e:#}");
                job.log(&s);
                rejected.push(s);
            }
        }
    }
    // a loop the user asked for and that cannot be used would give a result they do not expect:
    // stop before the (long) optimisation instead of running without it
    if !rejected.is_empty() {
        bail!(
            "{} of {} loops could not be used; move or remove them and run again:\n{}",
            rejected.len(),
            loops.len(),
            rejected.join("\n")
        );
    }
    pr.loops.extend(matched);
    let marked = p
        .revisits
        .enabled
        .then(|| revisits::add(&mut pr, &p.revisits, &clock, job));
    pr.run(p, job)?;
    if let Some(m) = marked {
        revisits::report(&pr, &p.revisits, m, &clock, job);
    }
    for lp in &pr.loops {
        let (deg, m) = lp.error(&pr.states);
        job.log(&format!(
            "loop {} = {}: closed to {:.1} mm / {:.3} deg",
            clock.label_at(pr.session[lp.a], pr.kt[lp.a]),
            clock.label_at(pr.session[lp.b], pr.kt[lp.b]),
            m * 1e3,
            deg
        ));
    }
    s20_io::dataset::write_tum(
        out,
        &pr.dense(p, 0.01),
        "t x y z qx qy qz qw (world_T_imu, sensor clock; unified optimisation)",
    )?;
    let knots = pr.knots();
    if !knots.is_empty() {
        let med = |c: usize| {
            let mut v: Vec<f64> = knots.iter().map(|r| r[c]).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        job.log(&format!(
            "accelerometer bias (median over the knots): {:.3},{:.3},{:.3} m/s^2 \
             (the LIO's ba0 when the fixes held the optimisation)",
            med(14),
            med(15),
            med(16)
        ));
    }
    let mut s = String::from("# t x y z qx qy qz qw vx vy vz bgx bgy bgz bax bay baz\n");
    for r in &knots {
        s.push_str(
            &r.iter()
                .map(|v| format!("{v:.9}"))
                .collect::<Vec<_>>()
                .join(" "),
        );
        s.push('\n');
    }
    std::fs::write(out.to_string_lossy().replace(".txt", "_knots.txt"), s)?;
    let mut q = format!("# {}\n", s20_opt::quality::HEADER);
    for r in pr.quality(p) {
        q.push_str(&r.row());
        q.push('\n');
    }
    std::fs::write(quality_file(out), q)?;
    frames::record(data, &[out], Some(frame))?;
    frame.write(&data.join("coordinate_frame.json"))?;
    job.log(&format!("saved {}", out.display()));
    Ok(())
}

/// Per-knot quality written next to an optimised trajectory.
pub fn quality_file(traj: &Path) -> PathBuf {
    PathBuf::from(traj.to_string_lossy().replace(".txt", "_quality.txt"))
}

/// A result written under a temporary name of the same extension (`<stem>.partial.<ext>`), which
/// replaces it only once complete, so an interrupted step keeps the earlier result; the
/// temporary file goes when not kept.
struct Partial {
    path: PathBuf,
    out: PathBuf,
}

impl Partial {
    fn new(out: &Path) -> Self {
        let stem = out.file_stem().unwrap_or_default().to_string_lossy();
        let name = match out.extension() {
            Some(e) => format!("{stem}.partial.{}", e.to_string_lossy()),
            None => format!("{stem}.partial"),
        };
        Partial {
            path: out.with_file_name(name),
            out: out.to_path_buf(),
        }
    }

    fn keep(self) -> Result<()> {
        std::fs::rename(&self.path, &self.out)
            .with_context(|| format!("rename to {}", self.out.display()))
    }
}

impl Drop for Partial {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Point cloud from the raw scans and a trajectory (see [`s20_finish`]), written as LAS / LAZ with
/// the normals of the planes the points were projected onto (zero where none), and a `.json` of
/// the counts by status. Made in squares (`tile_size`), written as they finish.
pub fn cloud(
    data: &Path,
    traj: &Path,
    out: &Path,
    p: &s20_finish::FinishParams,
    job: &Job,
) -> Result<()> {
    let l = s20_io::dataset::LidarData::open(data)?;
    let tr = s20_geom::traj::Traj::from_rows(&s20_io::dataset::read_tum(traj)?);
    let frame = frames::result_frame(data, traj)?;
    let origin = frame.output_offset();
    let off = s20_io::dataset::output_offsets(data)?;
    job.log(&format!("{p:?}"));
    let las_off = origin.map(|v| (v / 100.0).round() * 100.0);
    let session = |scan: u32| l.scans[scan as usize].session;
    let to_las = |o: &s20_finish::Observation, r: &s20_finish::PointResult| s20_io::las::LasPoint {
        xyz: std::array::from_fn(|d| r.xyz[d] as f64 + origin[d]),
        intensity: o.intensity,
        source_id: (o.scan % 65535) as u16,
        gps_time: o.time - off[session(o.scan)],
        normal: r.normal,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let part = Partial::new(out);
    let (total, sampled, status, written, stages) = if p.tile_size > 0.0 {
        let mut w =
            s20_io::las::LasWriter::create(&part.path, 1e-4, las_off, &s20_io::las::NORMALS)?;
        let tmp = PathBuf::from(format!("{}.tiles", out.display()));
        let t = s20_finish::tiles::finish_tiled(
            &l,
            &tr,
            s20_lio::LIDAR_IMU_T,
            p,
            &tmp,
            job,
            |obs, res| {
                let pts: Vec<s20_io::las::LasPoint> =
                    obs.iter().zip(res).map(|(o, r)| to_las(o, r)).collect();
                w.write(&pts, |i, b| s20_io::las::fill_normal(&pts[i], b))
            },
        )?;
        (
            t.observations,
            t.sampled,
            t.status,
            w.finish()? as usize,
            t.seconds,
        )
    } else {
        let obs = s20_finish::load::observations(&l, &tr, s20_lio::LIDAR_IMU_T, p);
        ensure!(!obs.is_empty(), "no points within the trajectory");
        let total = obs.len();
        // its stages each report from 0: the progress stays at the start
        let mut f = s20_finish::finish(&obs, p, &job.part(0.0, 0.0))?;
        drop(obs);
        let held: Vec<usize> = (0..f.points.len())
            .filter(|&i| f.candidates[i] && !s20_finish::removed(f.points[i].status))
            .collect();
        if !held.is_empty() {
            let points: Vec<[f32; 3]> = held.iter().map(|&i| f.points[i].xyz).collect();
            let passing = s20_finish::transient::find(
                &l,
                &tr,
                s20_geom::so3::V3::from(s20_lio::LIDAR_IMU_T),
                p,
                &points,
                &job.part(0.8, 1.0),
            );
            let mut taken = 0;
            for (&i, t) in held.iter().zip(passing) {
                if t {
                    f.points[i].status = s20_finish::TRANSIENT;
                    taken += 1;
                }
            }
            job.log(&format!(
                "transient: {taken} of {} points seen through by many more scans than saw them",
                held.len()
            ));
        }
        let pts: Vec<s20_io::las::LasPoint> = f.kept().map(|(o, r)| to_las(o, r)).collect();
        s20_io::las::write(&part.path, &pts, 1e-4, las_off)?;
        let status = s20_finish::STATUS_NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| {
                (
                    name.to_string(),
                    f.points.iter().filter(|r| r.status == i as u32).count(),
                )
            })
            .filter(|(_, n)| *n > 0)
            .collect();
        (total, f.input.len(), status, pts.len(), f.timings.clone())
    };
    part.keep()?;
    std::fs::write(
        format!("{}.json", out.display()),
        serde_json::to_string_pretty(&serde_json::json!({
            "observations": total,
            "sampled": sampled,
            "output": written,
            "status": status,
            "seconds": t0.elapsed().as_secs_f64(),
            "stages": stages,
        }))?,
    )?;
    frames::record(data, &[out], Some(frame))?;
    job.log(&format!(
        "wrote {} ({written} points of {total} observations)",
        out.display()
    ));
    Ok(())
}

/// The dense trajectory as TUM rows, written by [`export`] beside the trajectory it is made from.
pub const TUM_FILE: &str = "trajectory_100hz_tum.txt";
/// The camera poses colouring reads, written by [`export`] beside the trajectory (the maker's are
/// in the `images/` folder, when it is asked for).
pub const IMG_POSE_FILE: &str = "ImgPose.txt";

/// Trajectory files in the maker's formats with the Potree octree of `trajectory.las` (and
/// `coordinate_frame.json` unless in the SLAM frame) into `out`; beside `traj` the dense
/// trajectory ([`TUM_FILE`]) and, when a calibration is given, the camera poses
/// ([`IMG_POSE_FILE`]), with the maker's `images/` folder in `out` if `images` (else one of an
/// earlier export goes, as it would not follow the trajectory).
pub fn export(data: &Path, traj: &Path, out: &Path, calib: Option<&Path>, job: &Job) -> Result<()> {
    let l = s20_io::dataset::LidarData::open(data)?;
    let tr = s20_geom::traj::Traj::from_rows(&s20_io::dataset::read_tum(traj)?);
    let frame = frames::result_frame(data, traj)?;
    let origin = frame.output_offset();
    let off = s20_io::dataset::output_offsets(data)?;
    let tum = traj.with_file_name(TUM_FILE);
    let n = export::trajectory(&l, &tr, origin, &off, out, &tum)?;
    // the frame's origin, to place the outputs on the earth (none in the SLAM frame)
    let frame_file = out.join("coordinate_frame.json");
    if frame == CoordinateFrame::LocalSlam {
        if frame_file.exists() {
            std::fs::remove_file(&frame_file)
                .with_context(|| format!("remove {}", frame_file.display()))?;
        }
    } else {
        frame.write(&frame_file)?;
    }
    let las = out.join("trajectory.las");
    frames::record(
        data,
        &[&out.join("trajectory.txt"), &las, &tum],
        Some(frame),
    )?;
    // where earlier exports wrote them
    for (name, now) in [
        (TUM_FILE, tum.clone()),
        (IMG_POSE_FILE, traj.with_file_name(IMG_POSE_FILE)),
    ] {
        let old = out.join(name);
        if old != now && old.exists() {
            std::fs::remove_file(&old).with_context(|| format!("remove {}", old.display()))?;
            frames::record(data, &[&old], None)?;
        }
    }
    job.log(&format!("trajectory: {n} poses -> {}", out.display()));
    if n > 0 {
        // with the times, by which the Potree viewer colours the trajectory
        let opt = s20_io::potree::WriteOptions {
            gps_time: true,
            ..Default::default()
        };
        potree_step(&las, opt, "export", job)?;
    }
    if let Some(c) = calib {
        let cal = s20_io::calib::load(c)?;
        let poses = export::camera_poses(data, &tr, &cal, origin, cal.lidar_imu_t)?;
        let f = traj.with_file_name(IMG_POSE_FILE);
        let n = export::image_poses(&poses, &f)?;
        frames::record(data, &[&f], Some(frame))?;
        job.log(&format!("image poses: {n} -> {}", f.display()));
    }
    Ok(())
}

/// The photos with their poses in output folder `out`: the maker's `images/` (the photos, their
/// poses as `ImgPose.txt` and `xyzopk.txt`, the cameras as `Left.opt` and `Right.opt`) and a COLMAP
/// model beside it (`sparse/0`, `masks/`, `gauge.txt`; its images those of `images/`), so that the
/// output folder opens as a dataset in Gaussian Splatting tools. `imgpose` and `calib`: the image
/// poses and the calibration of the colouring (refined), `colored` the coloured cloud (the COLMAP
/// model's seed points). `colmap`: other poses for the COLMAP model (the tracks' of the refinement,
/// written beside the photos as `ImgPose-3dgs.txt`).
#[allow(clippy::too_many_arguments)]
pub fn images(
    data: &Path,
    traj: &Path,
    imgpose: &Path,
    colmap: Option<&Path>,
    calib: &Path,
    colored: &Path,
    out: &Path,
    job: &Job,
) -> Result<()> {
    let cal = s20_io::calib::load(calib)?;
    let tr = s20_geom::traj::Traj::from_rows(&s20_io::dataset::read_tum(traj)?);
    let frame = frames::result_frame(data, traj)?;
    let mut poses = export::camera_poses(data, &tr, &cal, frame.output_offset(), cal.lidar_imu_t)?;
    // the poses refined (the image names as image_poses writes them)
    let multi = poses.iter().any(|c| c.dir != poses[0].dir);
    let take = |poses: &mut [export::CameraPose], file: &Path| -> Result<()> {
        let refined = s20_visual::pairs::read_image_poses(file)?;
        let by_name: std::collections::HashMap<&str, usize> = refined
            .names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        for c in poses {
            let name = if multi {
                c.dir
                    .join("images")
                    .join(&c.name)
                    .to_string_lossy()
                    .replace('\\', "/")
            } else {
                c.name.clone()
            };
            if let Some(&i) = by_name.get(name.as_str()) {
                c.r = refined.poses[i].r;
                c.p = refined.poses[i].p;
            }
        }
        Ok(())
    };
    take(&mut poses, imgpose)?;
    let colmap_poses = match colmap {
        Some(file) => {
            let mut other =
                export::camera_poses(data, &tr, &cal, frame.output_offset(), cal.lidar_imu_t)?;
            take(&mut other, file)?;
            Some(other)
        }
        None => None,
    };
    let dir = out.join("images");
    // made beside, replacing the folder once complete
    let tmp = out.join("images.partial");
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)?;
    }
    let t0 = std::time::Instant::now();
    let n = match export::image_set(&poses, &cal, &tmp, &job.part(0.0, 0.7)) {
        Ok(n) => n,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(e);
        }
    };
    if dir.exists() {
        std::fs::remove_dir_all(&dir).with_context(|| format!("remove {}", dir.display()))?;
    }
    std::fs::rename(&tmp, &dir).with_context(|| format!("rename to {}", dir.display()))?;
    let colmap_imgpose = match &colmap_poses {
        Some(other) => {
            let file = dir.join("ImgPose-3dgs.txt");
            export::set_poses(other, &file)?;
            file
        }
        None => dir.join(IMG_POSE_FILE),
    };
    job.log(&format!(
        "images: {n} -> {} ({:.1} s)",
        dir.display(),
        t0.elapsed().as_secs_f64()
    ));
    let input = s20_visual::colmap::Input {
        data: out.to_path_buf(),
        imgpose: colmap_imgpose,
        calib: calib.to_path_buf(),
        cloud: colored.to_path_buf(),
        out: out.to_path_buf(),
        points: 500_000,
        max_angle: 85.,
    };
    s20_visual::colmap::export(&input, &job.part(0.7, 1.0))?;
    Ok(())
}

/// Where the Potree octree starts in the progress of the steps that write a cloud.
const POTREE_FROM: f64 = 0.9;

/// The Potree octree of output cloud `cloud` in the folder beside it named as PotreeConverter's
/// (`<file name>_converted`, see [`s20_io::potree`]), with what `opt` keeps. The octree of the
/// cloud in the other format (LAS / LAZ), of an earlier run, goes. Returns the points written.
pub fn potree(
    cloud: &Path,
    opt: s20_io::potree::WriteOptions,
    cancel: &std::sync::atomic::AtomicBool,
    progress: &(dyn Fn(u64, u64) + Sync),
) -> Result<u64> {
    use s20_io::potree;
    let n = potree::write(cloud, &potree::converted_dir(cloud), opt, cancel, progress)?;
    let other = cloud.with_extension(if s20_io::las::is_laz(cloud) {
        "las"
    } else {
        "laz"
    });
    let old = potree::converted_dir(&other);
    if old.exists() {
        std::fs::remove_dir_all(&old).with_context(|| format!("remove {}", old.display()))?;
    }
    Ok(n)
}

/// [`potree`] as part of a step, its progress reported under `stage`.
pub fn potree_step(
    cloud: &Path,
    opt: s20_io::potree::WriteOptions,
    stage: &str,
    job: &Job,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let n = potree(cloud, opt, &job.cancel, &|done, total| {
        job.report(stage, done, total)
    })?;
    job.log(&format!(
        "wrote {} ({n} points, {:.1} s)",
        s20_io::potree::converted_dir(cloud).display(),
        t0.elapsed().as_secs_f64()
    ));
    Ok(())
}

/// Colours a LAS / LAZ file from the fisheye images (see [`s20_finish::color`]); points no image
/// sees are left out. `imgpose` and `trajectory`: the export step's `ImgPose.txt` and
/// `trajectory.txt`; `data`: the folder whose [`frames`] hold those of the cloud and the poses.
/// Only the points' positions and normals are held; the other fields are read again to write.
#[allow(clippy::too_many_arguments)]
pub fn colorize(
    data: &Path,
    inp: &Path,
    out: &Path,
    images: &Path,
    imgpose: &Path,
    trajectory: &Path,
    calib: &Path,
    p: &s20_finish::color::ColorParams,
    job: &Job,
) -> Result<()> {
    job.log(&format!("{p:?}"));
    let cloud_frame = frames::recorded(data, inp)?;
    let poses_frame = frames::recorded(data, imgpose)?;
    if cloud_frame.is_some() || poses_frame.is_some() {
        ensure!(
            cloud_frame.is_some() && poses_frame.is_some(),
            "cloud and image poses must have coordinate metadata; re-run cloud generation and export"
        );
        ensure!(
            cloud_frame == poses_frame,
            "cloud and image poses use different coordinate frames; re-run cloud generation and export"
        );
    }
    const CHUNK: usize = 1 << 20;
    let mut rd = s20_io::las::LasReader::open(inp)?;
    let info = rd.info();
    let origin = info.offset;
    let total = rd.len() as u64;
    // the progress: reading and writing the cloud about 4% and 5% of the time (the park
    // 2026-10-07 10-31: 16 and 23 of 446 s)
    let (read, write) = (job.part(0.0, 0.035), job.part(0.95, 1.0));
    let (mut xyz, mut nrm) = (Vec::with_capacity(rd.len()), Vec::with_capacity(rd.len()));
    loop {
        read.report("colour read", xyz.len() as u64, total);
        let pts = rd.read(CHUNK)?;
        if pts.is_empty() {
            break;
        }
        for q in &pts {
            xyz.push(std::array::from_fn(|d| (q.xyz[d] - origin[d]) as f32));
            nrm.push(q.normal);
        }
    }
    let tmp = PathBuf::from(format!("{}.tiles", out.display()));
    let rgb = s20_finish::color::colorize(
        &xyz,
        &nrm,
        origin,
        images,
        imgpose,
        trajectory,
        calib,
        p,
        &tmp,
        &job.part(0.035, 0.95),
    )?;
    drop((xyz, nrm));
    let mut rd = s20_io::las::LasReader::open(inp)?;
    let part = Partial::new(out);
    let mut w = s20_io::las::LasWriter::create(
        &part.path,
        info.scale[0],
        info.offset,
        &s20_io::las::NORMALS,
    )?;
    let mut at = 0;
    loop {
        write.report("colour write", at as u64, total);
        let mut pts = rd.read(CHUNK)?;
        if pts.is_empty() {
            break;
        }
        let colours = &rgb[at..at + pts.len()];
        at += pts.len();
        let mut c = colours.iter();
        pts.retain_mut(|q| match c.next().copied().flatten() {
            Some(c) => {
                q.rgb = las_rgb(c);
                true
            }
            None => false,
        });
        w.write(&pts, |i, b| s20_io::las::fill_normal(&pts[i], b))?;
    }
    let n = w.finish()?;
    part.keep()?;
    frames::record(data, &[out], cloud_frame)?;
    job.log(&format!("wrote {} ({n} points)", out.display()));
    Ok(())
}

/// The 16-bit LAS colour of an 8-bit-scale colour, in whole 8-bit levels: Potree reads a channel
/// of 255 or less as 8-bit, so a near-black fraction (1..=255 in 16 bits) would show at full
/// brightness.
fn las_rgb(c: [f32; 3]) -> [u16; 3] {
    c.map(|x| x.clamp(0., 255.).round() as u16 * 257)
}

/// Corrections the user made on the trajectory, kept in the work folder (`s20_edits.json`).
/// Times are on a session (see [`At`]), as on the command line.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Edits {
    /// RTK fixes in these intervals are not used
    pub rtk_exclude: Vec<(At, At)>,
    /// the odometry keeps the heading (rotation about the vertical) and the gyro bias of the IMU in these
    /// intervals: places where the LiDAR fixes the heading badly (round hills, featureless slopes)
    pub heading_hold: Vec<(At, At)>,
    /// pairs of times at the same place
    pub loops: Vec<(At, At)>,
    /// points picked at the same features in two sessions (placing a session without GNSS)
    pub matches: Vec<(PointAt, PointAt)>,
}

/// A saved state of the corrections, kept in the work folder (`edits_revisions/<id>.json`): saved
/// by the user, or when the optimisation ran with them.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EditsRevision {
    /// the file's name: the time saved (ms) and a counter, so that the names sort by time
    pub id: String,
    pub name: String,
    /// seconds since 1970
    pub saved_at: u64,
    /// saved when the optimisation ran (not by the user)
    #[serde(default)]
    pub by_run: bool,
    pub edits: Edits,
}

/// The last run of a step: whether it finished, and the settings it used.
#[derive(serde::Serialize, serde::Deserialize)]
struct StepRecord {
    done: bool,
    params: serde_json::Value,
}

/// A step's finished result before it runs again: its time, and its record as written.
struct PreviousResult {
    modified: std::time::SystemTime,
    record: Option<String>,
}

/// Why a step's result is out of date.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stale {
    /// an earlier step's result is newer
    Earlier,
    /// the corrections changed after it (optimisation)
    Edits,
    /// its settings changed since it ran
    Params,
}

/// A revision id names a file in the revisions folder: digits and dashes only.
fn revision_id_ok(id: &str) -> bool {
    !id.is_empty() && id.bytes().all(|c| c.is_ascii_digit() || c == b'-')
}

/// A file's contents as a short digest (CRC-32, hexadecimal; empty when it cannot be read), to tell
/// when a calibration file changed.
pub fn file_digest(p: &Path) -> String {
    std::fs::read(p)
        .map(|b| format!("{:08x}", crc32fast::hash(&b)))
        .unwrap_or_default()
}

fn json<T: serde::Serialize>(x: &T) -> serde_json::Value {
    serde_json::to_value(x).expect("parameters serialise")
}

/// The steps of a whole run, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Step {
    Extract,
    Lio,
    Opt,
    Cloud,
    Export,
    /// every image's pose refined against the cloud (and the cameras without a device calibration)
    Refine,
    Colorize,
    /// the photos with their poses as the maker's (`images/`) and as a COLMAP model beside them
    Images,
}

impl Step {
    /// Steps run for each session (the others run once for all of them).
    pub fn per_session(self) -> bool {
        matches!(self, Step::Extract | Step::Lio)
    }

    /// Steps that write their result only once they have finished: a run cancelled or failed
    /// before that leaves the previous result whole.
    fn writes_at_end(self) -> bool {
        matches!(self, Step::Lio | Step::Opt)
    }

    pub const ALL: [Step; 8] = [
        Step::Extract,
        Step::Lio,
        Step::Opt,
        Step::Cloud,
        Step::Export,
        Step::Refine,
        Step::Colorize,
        Step::Images,
    ];

    /// Stage name used in progress reports.
    pub fn key(self) -> &'static str {
        match self {
            Step::Extract => "extract",
            Step::Lio => "lio",
            Step::Opt => "opt",
            Step::Cloud => "cloud",
            Step::Export => "export",
            Step::Refine => "refine",
            Step::Colorize => "colorize",
            Step::Images => "images",
        }
    }
}

/// The parameters that do not change a step's result (how the cloud or the colouring is cut in
/// tiles), left out of its record and of the comparison with it (older records may hold them).
const UNCOMPARED: [&str; 4] = ["tile_size", "tile_points", "tile_obs", "tile_memory"];
fn comparable(mut v: serde_json::Value) -> serde_json::Value {
    if let Some(o) = v.as_object_mut() {
        for k in UNCOMPARED {
            o.remove(k);
        }
    }
    v
}

/// The parameters of every processing step, saved with the GUI settings and as JSON files.
/// Missing keys take the default values (the indoor preset), so files written by older versions
/// still load.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProcessParams {
    pub lio: s20_lio::Params,
    /// unified optimisation, with the registration of the loops
    pub opt: s20_opt::Params,
    pub rtk: s20_opt::RtkParams,
    pub ppk: s20_ppk::Params,
    pub cloud: s20_finish::FinishParams,
    pub color: s20_finish::color::ColorParams,
}

impl Default for ProcessParams {
    fn default() -> Self {
        Self::preset(false)
    }
}

impl ProcessParams {
    /// Parameters still at a default an earlier version had take today's (for settings carried
    /// over to new work: a work folder's own settings stay, so its results stay current):
    /// `anchor_grid` 2 cm -> 4 cm (2026-10-29).
    pub fn update_old_defaults(&mut self) -> bool {
        let new = s20_finish::FinishParams::default().anchor_grid;
        let old = self.cloud.anchor_grid == 0.02 && new != 0.02;
        if old {
            self.cloud.anchor_grid = new;
        }
        old
    }

    /// The indoor or outdoor preset.
    pub fn preset(outdoor: bool) -> Self {
        if !outdoor {
            return ProcessParams {
                lio: s20_lio::Params::indoor(),
                opt: s20_opt::Params::default(),
                rtk: s20_opt::RtkParams::default(),
                ppk: s20_ppk::Params::default(),
                cloud: s20_finish::FinishParams::default(),
                color: s20_finish::color::ColorParams::default(),
            };
        }
        ProcessParams {
            lio: s20_lio::Params::outdoor(),
            opt: s20_opt::Params::default(),
            rtk: s20_opt::RtkParams::default(),
            ppk: s20_ppk::Params::default(),
            // the same cloud indoors and outdoors
            cloud: s20_finish::FinishParams::default(),
            color: s20_finish::color::ColorParams::default(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("parameters serialise")
    }

    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("not a parameter file")
    }
}

/// The processing settings of a work folder (`settings.json`): the preset the parameters started
/// from, the parameters, which GNSS is used and the point cloud format. Kept with the work, since
/// they differ between places (indoors without GNSS, outdoors with it).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WorkSettings {
    pub outdoor: bool,
    pub params: ProcessParams,
    pub rtk: bool,
    pub ppk: bool,
    /// point clouds written as LAZ (else LAS)
    pub laz: bool,
    /// the export writes the photos with their poses (`images/`)
    pub images: bool,
    /// every image's pose is refined against the cloud before the colouring
    pub refine: bool,
}

impl Default for WorkSettings {
    fn default() -> Self {
        WorkSettings {
            outdoor: false,
            params: ProcessParams::preset(false),
            rtk: false,
            ppk: false,
            laz: true,
            images: false,
            refine: true,
        }
    }
}

impl WorkSettings {
    fn file(work: &Path) -> PathBuf {
        work.join("settings.json")
    }

    /// The settings saved in work folder `work`, if it has them.
    pub fn load(work: &Path) -> Option<Result<Self>> {
        let f = Self::file(work);
        f.is_file().then(|| {
            let s = std::fs::read_to_string(&f).with_context(|| format!("read {}", f.display()))?;
            serde_json::from_str(&s).with_context(|| format!("{} cannot be read", f.display()))
        })
    }

    pub fn save(&self, work: &Path) -> Result<()> {
        let f = Self::file(work);
        std::fs::write(&f, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("write {}", f.display()))
    }
}

/// A session of a run: one raw S20 folder.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SessionCfg {
    /// its folder in the work folder (`sessions/<name>`), and its name in the corrections
    pub name: String,
    pub raw: PathBuf,
}

/// Settings of a whole run from one or more raw S20 folders.
///
/// Each session keeps its extraction and odometry in `sessions/<name>/{data,runs}`, one session
/// as well as several; `data/` holds the set of all sessions (`sessions.txt`, the joined IMU and
/// GNSS tables, the common origin), `runs/` the optimisation of all of them and `output/` the
/// products. `project.json` lists the sessions (none in a work folder whose last one was taken
/// out).
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub sessions: Vec<SessionCfg>,
    pub work: PathBuf,
    pub name: String,
    /// parameters of the steps
    pub params: ProcessParams,
    /// use RTK when the receiver has fixes
    pub rtk: bool,
    /// Compute raw PPK and use it instead of receiver RTK in the optimisation.
    pub ppk: bool,
    pub rtk_exclude: Vec<(At, At)>,
    /// whether a run continues to the colorize step (running that step itself always colours)
    pub color: bool,
    /// point clouds written LASzip-compressed (`.laz`) rather than as `.las`
    pub laz: bool,
    /// the last step writes the maker's `images/` folder (the photos with their poses) and a COLMAP
    /// model beside it in the output folder ([`Step::Images`])
    pub images: bool,
    /// every image's pose is refined against the cloud ([`Step::Refine`]) and the colouring uses
    /// them; without `device_calib` the cameras are solved too, from the recording alone
    pub refine: bool,
    /// the device's calibration (`s20 calibrate-cameras`), in place of the maker's
    pub device_calib: Option<PathBuf>,
}

fn folder_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "s20".into())
}

/// What a folder chosen or dropped in the app is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FolderKind {
    Work,
    Raw,
    Other,
}

/// The list of a work folder's sessions (`project.json`).
#[derive(serde::Serialize, serde::Deserialize)]
struct Project {
    sessions: Vec<SessionCfg>,
}

impl RunConfig {
    pub fn new(raw: PathBuf, work: PathBuf) -> Self {
        Self::with_sessions(vec![raw], work)
    }

    /// A run of several raw folders (sessions), named by their folders (see
    /// [`RunConfig::new_session`]). The first session is the reference.
    pub fn with_sessions(raws: Vec<PathBuf>, work: PathBuf) -> Self {
        let mut c = Self::of_sessions(Vec::new(), work);
        for raw in raws {
            let s = c.new_session(raw);
            c.sessions.push(s);
        }
        c.name = Self::run_name(&c.sessions, &c.work);
        c
    }

    /// A run of sessions as listed (in `project.json`).
    pub fn of_sessions(sessions: Vec<SessionCfg>, work: PathBuf) -> Self {
        RunConfig {
            name: Self::run_name(&sessions, &work),
            sessions,
            work,
            params: ProcessParams::default(),
            rtk: true,
            ppk: false,
            rtk_exclude: Vec::new(),
            color: true,
            laz: true,
            images: false,
            refine: true,
            device_calib: None,
        }
    }

    /// The name of the products: of the session for one, else of the work folder.
    fn run_name(sessions: &[SessionCfg], work: &Path) -> String {
        match sessions {
            [s] => s.name.clone(),
            _ => folder_name(work),
        }
    }

    /// A new session of raw folder `raw`, named by the folder (`_2`, `_3` added to a name in use,
    /// or one whose folder holds the results of another recording: a session taken out keeps its
    /// folder, and adding it again finds its results).
    pub fn new_session(&self, raw: PathBuf) -> SessionCfg {
        let base = folder_name(&raw);
        let mut name = base.clone();
        let mut n = 2;
        while self.sessions.iter().any(|s| s.name == name)
            || !Self::holds(&self.work.join("sessions").join(&name).join("data"), &raw)
        {
            name = format!("{base}_{n}");
            n += 1;
        }
        SessionCfg { name, raw }
    }

    /// Whether data folder `data` was extracted from the recording in raw folder `raw` (a bag of
    /// the same name and size), or not extracted at all.
    fn holds(data: &Path, raw: &Path) -> bool {
        let Ok(src) = s20_io::dataset::Source::read(data) else {
            return true;
        };
        s20_io::extract::find_bag(raw).is_ok_and(|bag| {
            bag.file_name() == src.bag.file_name()
                && std::fs::metadata(&bag).is_ok_and(|m| m.len() == src.size)
        })
    }

    /// Whether raw folder `raw` holds the recording session `k` was extracted from (it was moved
    /// there), so its results stay valid.
    pub fn same_recording(&self, k: usize, raw: &Path) -> bool {
        Self::holds(&self.session_data(k), raw)
    }

    /// Points session `k` at raw folder `raw`, where its recording was moved
    /// ([`RunConfig::same_recording`]): the extraction keeps, reading the bag there.
    pub fn move_raw(&mut self, k: usize, raw: PathBuf) -> Result<()> {
        let data = self.session_data(k);
        if let Ok(src) = s20_io::dataset::Source::read(&data) {
            s20_io::dataset::Source {
                bag: s20_io::extract::find_bag(&raw)?,
                size: src.size,
            }
            .write(&data)?;
        }
        self.sessions[k].raw = raw;
        self.write_project()
    }

    /// The sessions listed in a work folder (`project.json`), if any.
    pub fn project_sessions(work: &Path) -> Option<Vec<SessionCfg>> {
        let s = std::fs::read_to_string(work.join("project.json")).ok()?;
        serde_json::from_str::<Project>(&s).ok().map(|p| p.sessions)
    }

    /// What a folder is: a work folder (it lists its sessions, or is one of an older layout), a
    /// raw S20 folder (it holds the bag), or neither.
    pub fn folder_kind(dir: &Path) -> FolderKind {
        if dir.join("project.json").is_file()
            || dir
                .join("data")
                .join(s20_io::dataset::Source::FILE)
                .is_file()
        {
            FolderKind::Work
        } else if s20_io::extract::find_bag(dir).is_ok() {
            FolderKind::Raw
        } else {
            FolderKind::Other
        }
    }

    /// The sessions of a work folder being opened (none when the last was taken out).
    pub fn open_work(work: &Path) -> Result<Vec<SessionCfg>> {
        match Self::project_sessions(work) {
            Some(v) => Ok(v),
            None if work.join("project.json").exists() => {
                bail!("{}: project.json cannot be read", work.display())
            }
            None => bail!(
                "{} is a work folder of an older version (no project.json): make a new one",
                work.display()
            ),
        }
    }

    /// Makes a new work folder for the raw folder `raw` (it lists the raw folder as its session).
    pub fn create_work(raw: &Path, work: &Path) -> Result<()> {
        s20_io::extract::find_bag(raw)?;
        if work.exists() && std::fs::read_dir(work)?.next().is_some() {
            bail!("{} exists already and is not empty", work.display());
        }
        Self::new(raw.to_path_buf(), work.to_path_buf()).write_project()
    }

    /// Saves the session list (`project.json`). The results of each session stay in its folder,
    /// so sessions can be added, taken out and reordered; the set of all of them is made again by
    /// the extraction.
    pub fn save_sessions(&self) -> Result<()> {
        self.write_project()
    }

    /// Session `k`'s result of a step of each session is on disk and finished.
    pub fn session_done(&self, step: Step, k: usize) -> bool {
        self.done_of(step, Some(k))
    }

    fn write_project(&self) -> Result<()> {
        std::fs::create_dir_all(&self.work)?;
        let p = Project {
            sessions: self.sessions.clone(),
        };
        std::fs::write(
            self.work.join("project.json"),
            serde_json::to_string_pretty(&p)?,
        )?;
        Ok(())
    }

    pub fn multi(&self) -> bool {
        self.sessions.len() > 1
    }

    /// The folder of session `k`'s extraction and odometry.
    pub fn session_dir(&self, k: usize) -> PathBuf {
        self.work.join("sessions").join(&self.sessions[k].name)
    }
    pub fn session_data(&self, k: usize) -> PathBuf {
        self.session_dir(k).join("data")
    }
    pub fn session_runs(&self, k: usize) -> PathBuf {
        self.session_dir(k).join("runs")
    }
    /// The data folder the optimisation and the later steps read (all sessions).
    pub fn data(&self) -> PathBuf {
        self.work.join("data")
    }
    pub fn runs(&self) -> PathBuf {
        self.work.join("runs")
    }
    pub fn output(&self) -> PathBuf {
        self.work.join("output")
    }
    /// The maker's camera calibration (of the first session: one device).
    pub fn calib(&self) -> PathBuf {
        self.sessions[0].raw.join("info").join("calibration.yaml")
    }
    /// The calibration the image poses are exported with: the device's, else the maker's.
    pub fn camera_calib(&self) -> PathBuf {
        self.device_calib.clone().unwrap_or_else(|| self.calib())
    }
    /// The image poses refined by [`Step::Refine`].
    pub fn refined_poses(&self) -> PathBuf {
        self.runs().join("ImgPose-refined.txt")
    }
    /// The image poses of [`Step::Refine`]'s tracks: the images agree with one another, for the
    /// COLMAP model (Gaussian Splatting), while [`RunConfig::refined_poses`] fit the cloud.
    pub fn tracked_poses(&self) -> PathBuf {
        self.runs().join("ImgPose-3dgs.txt")
    }
    /// The cameras [`Step::Refine`] solved from the recording (without a device calibration).
    pub fn refined_calib(&self) -> PathBuf {
        self.runs().join("calibration-refined.yaml")
    }
    /// The image poses and the calibration the colouring uses.
    pub fn colour_inputs(&self) -> (PathBuf, PathBuf) {
        if !self.refine {
            return (
                self.opt_file().with_file_name(IMG_POSE_FILE),
                self.camera_calib(),
            );
        }
        let calib = if self.device_calib.is_some() {
            self.camera_calib()
        } else {
            self.refined_calib()
        };
        (self.refined_poses(), calib)
    }
    pub fn lio_file_of(&self, k: usize) -> PathBuf {
        self.session_runs(k).join("lio.txt")
    }
    /// The odometry of the first session.
    pub fn lio_file(&self) -> PathBuf {
        self.lio_file_of(0)
    }
    pub fn opt_file(&self) -> PathBuf {
        self.runs().join("opt.txt")
    }
    /// The trajectory the device of session `k` estimated while recording (made by the extraction).
    pub fn device_file_of(&self, k: usize) -> PathBuf {
        self.session_data(k).join(s20_io::extract::DEVICE_POSE)
    }
    pub fn ppk_file_of(&self, k: usize) -> PathBuf {
        self.session_runs(k).join("ppk.json")
    }
    /// The PPK result of the first session.
    pub fn ppk_file(&self) -> PathBuf {
        self.ppk_file_of(0)
    }
    fn cloud_ext(&self) -> &'static str {
        if self.laz { "laz" } else { "las" }
    }
    pub fn uncolored(&self) -> PathBuf {
        self.output()
            .join(format!("{}_uncolorized.{}", self.name, self.cloud_ext()))
    }
    pub fn colored(&self) -> PathBuf {
        self.output()
            .join(format!("{}_colorized.{}", self.name, self.cloud_ext()))
    }

    /// The sessions a step runs for: each session for [`Step::per_session`], else one (`None`).
    fn units(&self, step: Step) -> Vec<Option<usize>> {
        if step.per_session() {
            (0..self.sessions.len()).map(Some).collect()
        } else {
            vec![None]
        }
    }

    /// The file a step writes (its presence means the step was run); for a step of each session,
    /// that of session `k`.
    fn result_of(&self, step: Step, k: Option<usize>) -> PathBuf {
        let k = k.unwrap_or(0);
        match step {
            Step::Extract => self.session_data(k).join(s20_io::dataset::Source::FILE),
            Step::Lio => self.lio_file_of(k),
            Step::Opt => self.opt_file(),
            Step::Cloud => self.uncolored(),
            Step::Export => self.output().join("trajectory.txt"),
            Step::Refine => self.refined_poses(),
            Step::Colorize => self.colored(),
            Step::Images => self.output().join("sparse").join("0").join("images.txt"),
        }
    }

    /// The file a step writes (of the first session for a step of each session).
    pub fn result_file(&self, step: Step) -> PathBuf {
        self.result_of(step, Some(0))
    }

    /// The files a step writes: of every session, and the session set after the extraction.
    fn results(&self, step: Step) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = self
            .units(step)
            .into_iter()
            .map(|k| self.result_of(step, k))
            .collect();
        if step == Step::Extract {
            v.push(self.set_file());
        }
        v
    }

    /// The list of the sessions in the data folder of all of them.
    fn set_file(&self) -> PathBuf {
        self.data().join("sessions.txt")
    }

    /// The session data folders as `sessions.txt` lists them (relative to `data/`).
    fn set_dirs(&self) -> Vec<PathBuf> {
        self.sessions
            .iter()
            .map(|s| PathBuf::from("../sessions").join(&s.name).join("data"))
            .collect()
    }

    /// Whether `data/` holds the set of the current sessions.
    fn set_current(&self) -> bool {
        let want: Vec<PathBuf> = self
            .set_dirs()
            .iter()
            .map(|d| self.data().join(d))
            .collect();
        s20_io::dataset::session_dirs(&self.data()).is_ok_and(|d| d == want)
            && self.set_file().exists()
    }

    /// The record of a step's last run (`runs/<step>.step.json`, of the session for a step of
    /// each session).
    pub fn record_file(&self, step: Step) -> PathBuf {
        self.record_of(step, Some(0))
    }

    fn record_of(&self, step: Step, k: Option<usize>) -> PathBuf {
        let runs = match k {
            Some(k) if step.per_session() => self.session_runs(k),
            _ => self.runs(),
        };
        runs.join(format!("{}.step.json", step.key()))
    }

    fn record(&self, step: Step, k: Option<usize>) -> Option<StepRecord> {
        let s = std::fs::read_to_string(self.record_of(step, k)).ok()?;
        serde_json::from_str(&s).ok()
    }

    fn write_record(&self, step: Step, k: Option<usize>, done: bool) -> Result<()> {
        let r = StepRecord {
            done,
            params: self.step_params(step),
        };
        let f = self.record_of(step, k);
        if let Some(d) = f.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(&f, serde_json::to_string_pretty(&r)?)
            .with_context(|| format!("write {}", f.display()))
    }

    /// The settings a step's result depends on (compared with those of its last run).
    pub fn step_params(&self, step: Step) -> serde_json::Value {
        match step {
            Step::Extract => serde_json::Value::Null,
            // without the images as before, keeping older records current
            Step::Export => match &self.device_calib {
                Some(d) => serde_json::json!({"device_calibration": file_digest(d)}),
                None => serde_json::Value::Null,
            },
            Step::Images => {
                let mut v = serde_json::json!({"refined_poses": self.refine, "colmap": true});
                // the COLMAP model of the tracks' poses: older exports are redone
                if self.refine {
                    v["colmap_poses"] = "tracks".into();
                }
                v
            }
            // the tracks' adjustment after the refinement: older refinements are rerun
            Step::Refine => match &self.device_calib {
                Some(d) => serde_json::json!({"device_calibration": file_digest(d), "tracks": 1}),
                None => serde_json::json!({"fit_camera": true, "tracks": 1}),
            },
            Step::Lio => {
                // without hold intervals as before, keeping older records current
                let hold = self.edits().heading_hold;
                if hold.is_empty() {
                    json(&self.params.lio)
                } else {
                    serde_json::json!({"lio": json(&self.params.lio), "heading_hold": json(&hold)})
                }
            }
            Step::Opt => {
                let mut v = serde_json::json!({
                "opt": json(&self.params.opt),
                "rtk": json(&self.params.rtk),
                "use_rtk": self.rtk,
                "coordinate_frame": "local_enu_v1",
                });
                // Keep pre-PPK records current when the new feature is disabled.
                if self.ppk {
                    v["use_ppk"] = true.into();
                    v["ppk"] = json(&self.params.ppk);
                    v["ppk_engine"] = s20_ppk::ENGINE.into();
                }
                v
            }
            Step::Cloud => comparable(json(&self.params.cloud)),
            Step::Colorize => {
                let mut v = comparable(json(&self.params.color));
                if let Some(o) = v.as_object_mut() {
                    // without refined poses as before, keeping older records current
                    if self.refine {
                        o.insert("refined_poses".into(), true.into());
                    }
                }
                v
            }
        }
    }

    /// Session `k`'s result of a step is on disk and its last run finished (an interrupted run
    /// leaves its record unfinished). Without a record (results of older versions) the file decides.
    fn done_of(&self, step: Step, k: Option<usize>) -> bool {
        self.result_of(step, k).exists() && self.record(step, k).is_none_or(|r| r.done)
    }

    /// The step's results are on disk (of every session) and its last runs finished.
    pub fn done(&self, step: Step) -> bool {
        // the image poses are not refined, or the images not written, when not asked for: nothing to do
        if (step == Step::Refine && !self.refine) || (step == Step::Images && !self.images) {
            return true;
        }
        self.units(step).into_iter().all(|k| self.done_of(step, k))
            && (step != Step::Extract || self.set_current())
    }

    /// Whether a step's result is older than what it was made from: the result of an earlier step
    /// (run again since), or, for the optimisation, the user's corrections.
    pub fn stale(&self, step: Step) -> bool {
        self.staleness(step).is_some()
    }

    /// Why a step's result is out of date, if it is (for a step of each session, of any session).
    pub fn staleness(&self, step: Step) -> Option<Stale> {
        self.units(step)
            .into_iter()
            .find_map(|k| self.staleness_of(step, k))
    }

    fn staleness_of(&self, step: Step, k: Option<usize>) -> Option<Stale> {
        if (step == Step::Refine && !self.refine) || (step == Step::Images && !self.images) {
            return None;
        }
        let time = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
        let mine = time(&self.result_of(step, k))?;
        // a session's step follows that session's earlier steps; the others follow all sessions'
        let earlier = Step::ALL
            .into_iter()
            .filter(|&s| s < step && (s != Step::Refine || self.refine))
            .flat_map(|s| match k {
                Some(_) if step.per_session() => vec![self.result_of(s, k)],
                _ => self.results(s),
            })
            .filter_map(|f| time(&f))
            .any(|t| t > mine);
        if earlier {
            return Some(Stale::Earlier);
        }
        if step == Step::Opt && time(&self.edits_file()).is_some_and(|t| t > mine) {
            return Some(Stale::Edits);
        }
        if step == Step::Opt && self.ppk {
            for (j, s) in self.sessions.iter().enumerate() {
                let inputs_newer = [
                    s.raw.join("BASE.BIN"),
                    s.raw.join("ROVER_PPKRAW.bin"),
                    self.ppk_file_of(j),
                ]
                .iter()
                .any(|p| time(p).is_none_or(|t| t > mine));
                let bags_newer = std::fs::read_dir(s.raw.join("info"))
                    .map(|it| {
                        it.filter_map(|e| e.ok())
                            .map(|e| e.path())
                            .filter(|p| p.extension().is_some_and(|x| x == "bag"))
                            .any(|p| time(&p).is_some_and(|t| t > mine))
                    })
                    .unwrap_or(true);
                if inputs_newer || bags_newer {
                    return Some(Stale::Earlier);
                }
            }
        }
        if step == Step::Opt
            && frames::recorded(&self.data(), &self.opt_file())
                .ok()
                .flatten()
                .is_none()
        {
            return Some(Stale::Params);
        }
        let params = self
            .record(step, k)
            .is_some_and(|r| r.done && comparable(r.params.clone()) != self.step_params(step));
        (params || (step == Step::Opt && self.ppk && self.record(step, k).is_none()))
            .then_some(Stale::Params)
    }

    pub fn lio_params(&self) -> s20_lio::Params {
        self.params.lio.clone()
    }
    pub fn opt_params(&self) -> s20_opt::Params {
        self.params.opt.clone()
    }
    pub fn cloud_params(&self) -> s20_finish::FinishParams {
        self.params.cloud.clone()
    }
    pub fn color_params(&self) -> s20_finish::color::ColorParams {
        self.params.color.clone()
    }

    /// RTK options when enabled and the receivers reported enough fixes.
    pub fn rtk_options(&self) -> Option<RtkOptions> {
        let mut exclude = self.rtk_exclude.clone();
        exclude.extend(self.edits().rtk_exclude);
        if self.ppk {
            return Some(RtkOptions {
                params: self.params.rtk,
                exclude,
                ppk: (0..self.sessions.len())
                    .map(|k| self.ppk_file_of(k))
                    .collect(),
            });
        }
        if !self.rtk {
            return None;
        }
        let g = s20_io::dataset::load_gnss(&self.data()).ok()?;
        (g.iter().filter(|r| r[1] == s20_io::gnss::FIX).count() >= 100).then_some(RtkOptions {
            params: self.params.rtk,
            exclude,
            ppk: Vec::new(),
        })
    }

    /// Session `k`'s heading hold intervals from the corrections, in seconds as for [`At`].
    pub fn heading_hold_of(&self, k: usize) -> Vec<(f64, f64)> {
        let clock = Clock {
            names: self.sessions.iter().map(|s| s.name.clone()).collect(),
            range: Vec::new(),
        };
        let on_k = |a: &At| clock.session(&a.session).is_ok_and(|s| s == k);
        self.edits()
            .heading_hold
            .iter()
            .filter(|(a, b)| on_k(a) && on_k(b))
            .map(|(a, b)| (a.t, b.t))
            .collect()
    }

    pub fn edits_file(&self) -> PathBuf {
        self.work.join("s20_edits.json")
    }

    /// The user's corrections (none when the file is missing or unreadable).
    pub fn edits(&self) -> Edits {
        std::fs::read_to_string(self.edits_file())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save_edits(&self, e: &Edits) -> Result<()> {
        std::fs::create_dir_all(&self.work)?;
        std::fs::write(self.edits_file(), serde_json::to_string_pretty(e)?)?;
        Ok(())
    }

    pub fn revisions_dir(&self) -> PathBuf {
        self.work.join("edits_revisions")
    }

    /// The saved revisions of the corrections, oldest first (unreadable files left out).
    pub fn revisions(&self) -> Vec<EditsRevision> {
        let mut out: Vec<EditsRevision> = std::fs::read_dir(self.revisions_dir())
            .into_iter()
            .flatten()
            .flatten()
            .filter(|f| f.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|f| std::fs::read_to_string(f.path()).ok())
            .filter_map(|s| serde_json::from_str::<EditsRevision>(&s).ok())
            .filter(|r| revision_id_ok(&r.id))
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// Saves `e` as a new revision.
    pub fn save_revision(&self, name: &str, by_run: bool, e: &Edits) -> Result<EditsRevision> {
        let dir = self.revisions_dir();
        std::fs::create_dir_all(&dir)?;
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
        let ms = now.as_millis();
        let id = (0..1000)
            .map(|k| format!("{ms:014}-{k:03}"))
            .find(|id| !dir.join(format!("{id}.json")).exists())
            .context("too many revisions saved at once")?;
        let r = EditsRevision {
            id,
            name: name.to_string(),
            saved_at: now.as_secs(),
            by_run,
            edits: e.clone(),
        };
        self.write_revision(&r)?;
        Ok(r)
    }

    pub fn rename_revision(&self, id: &str, name: &str) -> Result<()> {
        let mut r = self
            .revisions()
            .into_iter()
            .find(|r| r.id == id)
            .context("no such revision")?;
        r.name = name.to_string();
        self.write_revision(&r)
    }

    pub fn delete_revision(&self, id: &str) -> Result<()> {
        if !revision_id_ok(id) {
            bail!("bad revision id {id}");
        }
        std::fs::remove_file(self.revisions_dir().join(format!("{id}.json")))?;
        Ok(())
    }

    fn write_revision(&self, r: &EditsRevision) -> Result<()> {
        if !revision_id_ok(&r.id) {
            bail!("bad revision id {}", r.id);
        }
        let path = self.revisions_dir().join(format!("{}.json", r.id));
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, serde_json::to_string_pretty(r)?)?;
        std::fs::rename(&temp, &path)?;
        Ok(())
    }

    /// Runs one step (for every session when it runs per session). Its record says "running"
    /// until the step finishes, so a run that is cancelled or fails is not taken for a finished one
    /// even when it left files behind.
    pub fn run_step(&self, step: Step, job: &Job) -> Result<()> {
        self.run_step_for(step, false, job)
    }

    /// As [`RunConfig::run_step`]; `missing`: only for the sessions without a finished result.
    fn run_step_for(&self, step: Step, missing: bool, job: &Job) -> Result<()> {
        job.check()?;
        job.log(&format!("== {}", step.key()));
        self.write_project()?;
        std::fs::create_dir_all(self.runs())?;
        std::fs::create_dir_all(self.output())?;
        for k in self.units(step) {
            if missing && self.done_of(step, k) {
                continue;
            }
            if let (Some(k), true) = (k, self.multi()) {
                job.log(&format!("-- {}", self.sessions[k].name));
                std::fs::create_dir_all(self.session_runs(k))?;
            }
            let previous = self.previous_result(step, k);
            self.write_record(step, k, false)?;
            if let Err(e) = self.run_step_inner(step, k.unwrap_or(0), job) {
                self.restore_result(step, k, previous, job);
                return Err(e);
            }
            self.write_record(step, k, true)?;
        }
        if step == Step::Extract {
            let shifts = s20_io::dataset::build_set(&self.data(), &self.set_dirs())?;
            if self.multi() {
                for (s, x) in self.sessions.iter().zip(shifts) {
                    job.log(&format!("session {}: times shifted by {x} s", s.name));
                }
            }
        }
        Ok(())
    }

    /// A finished result of a step that writes it at the end, before running it again: the time of
    /// the result and its record as written (none for results of older versions).
    fn previous_result(&self, step: Step, k: Option<usize>) -> Option<PreviousResult> {
        if !step.writes_at_end() || !self.done_of(step, k) {
            return None;
        }
        let modified = std::fs::metadata(self.result_of(step, k))
            .and_then(|m| m.modified())
            .ok()?;
        let record = std::fs::read_to_string(self.record_of(step, k)).ok();
        Some(PreviousResult { modified, record })
    }

    /// After a run that did not finish: the previous result stays finished if the run did not
    /// touch it (its record as before), so a cancelled run does not lose it.
    fn restore_result(
        &self,
        step: Step,
        k: Option<usize>,
        previous: Option<PreviousResult>,
        job: &Job,
    ) {
        let Some(p) = previous else { return };
        let now = std::fs::metadata(self.result_of(step, k)).and_then(|m| m.modified());
        if now.ok() != Some(p.modified) {
            return;
        }
        let f = self.record_of(step, k);
        let restored = match &p.record {
            Some(r) => std::fs::write(&f, r),
            None => std::fs::remove_file(&f),
        };
        if restored.is_ok() {
            job.log(&format!("{}: the previous result is kept", step.key()));
        }
    }

    /// Runs `step`; for a step of each session, for session `k`.
    fn run_step_inner(&self, step: Step, k: usize, job: &Job) -> Result<()> {
        match step {
            Step::Extract => {
                let st =
                    s20_io::extract::extract(&self.sessions[k].raw, &self.session_data(k), job)?;
                job.log(&format!(
                    "extracted {} scans, {} imu, {} gnss, {} images",
                    st.scans, st.imu, st.gnss, st.images
                ));
            }
            Step::Lio => lio(
                &self.session_data(k),
                &self.lio_file_of(k),
                &self.lio_params(),
                None,
                &self.heading_hold_of(k),
                job,
            )?,
            Step::Opt => {
                if self.ppk {
                    for (j, s) in self.sessions.iter().enumerate() {
                        std::fs::create_dir_all(self.session_runs(j))?;
                        s20_ppk::process(&s.raw, &self.ppk_file_of(j), &self.params.ppk, job)?;
                    }
                }
                // The UI can switch phases before loading and assembling the optimisation.
                job.report("opt load", 0, 0);
                let lio_files: Vec<PathBuf> = (0..self.sessions.len())
                    .map(|j| self.lio_file_of(j))
                    .collect();
                opt(
                    &self.data(),
                    &lio_files,
                    &self.opt_file(),
                    &self.opt_params(),
                    self.rtk_options().as_ref(),
                    &self.edits(),
                    job,
                )?;
            }
            Step::Cloud => {
                cloud(
                    &self.data(),
                    &self.opt_file(),
                    &self.uncolored(),
                    &self.cloud_params(),
                    &job.named(step.key()).part(0.0, POTREE_FROM),
                )?;
                potree_step(
                    &self.uncolored(),
                    Default::default(),
                    step.key(),
                    &job.part(POTREE_FROM, 1.0),
                )?;
            }
            Step::Export => {
                let calib = self.camera_calib();
                export(
                    &self.data(),
                    &self.opt_file(),
                    &self.output(),
                    calib.exists().then_some(calib.as_path()),
                    job,
                )?;
                // the copy earlier exports made (the cameras are in images/ when asked for)
                let copy = self.output().join("calibration.yaml");
                if copy.exists() {
                    std::fs::remove_file(&copy)
                        .with_context(|| format!("remove {}", copy.display()))?;
                }
            }
            Step::Refine => {
                let imgpose = self.opt_file().with_file_name(IMG_POSE_FILE);
                let input = s20_visual::run::Input {
                    // the image names of one session are relative to its folder, those of several
                    // are absolute
                    data: self.session_data(0),
                    imgpose: imgpose.clone(),
                    calib: self.camera_calib(),
                    cloud: self.uncolored(),
                    step: 12,
                    holdout: 0.2,
                    max_size: 1800,
                    features: 8000,
                    gate: 25.,
                    search: 80.,
                    ratio: 0.75,
                    calib_out: self.device_calib.is_none().then(|| self.refined_calib()),
                    camera_prior: s20_visual::run::CameraPrior::MAKER,
                    tracks: Some(self.tracked_poses()),
                    out: self.refined_poses(),
                };
                s20_visual::run::run(&input, &job.named(step.key()))?;
                let frame = frames::recorded(&self.data(), &imgpose)?;
                let tracked = self.tracked_poses();
                frames::record(&self.data(), &[&input.out, &tracked], frame)?;
            }
            Step::Images => {
                let (imgpose, calib) = self.colour_inputs();
                // the COLMAP model with the poses of the tracks, when refined
                let tracked = self.tracked_poses();
                let colmap = (self.refine && tracked.exists()).then_some(tracked.as_path());
                images(
                    &self.data(),
                    &self.opt_file(),
                    &imgpose,
                    colmap,
                    &calib,
                    &self.colored(),
                    &self.output(),
                    &job.named(step.key()),
                )?;
            }
            Step::Colorize => {
                let (imgpose, calib) = self.colour_inputs();
                colorize(
                    &self.data(),
                    &self.uncolored(),
                    &self.colored(),
                    // the image names of one session are relative to its folder, those of several
                    // are absolute
                    &self.session_data(0).join("images"),
                    &imgpose,
                    &self.output().join("trajectory.txt"),
                    &calib,
                    &self.color_params(),
                    &job.named(step.key()).part(0.0, POTREE_FROM),
                )?;
                potree_step(
                    &self.colored(),
                    s20_io::potree::WriteOptions {
                        rgb: true,
                        ..Default::default()
                    },
                    step.key(),
                    &job.part(POTREE_FROM, 1.0),
                )?;
            }
        }
        Ok(())
    }

    /// Runs the steps `first..=last` in order.
    pub fn run_range(&self, first: Step, last: Step, job: &Job) -> Result<()> {
        for step in Step::ALL.into_iter().filter(|&s| {
            (first..=last).contains(&s)
                && (s != Step::Refine || self.refine)
                && (s != Step::Images || self.images)
        }) {
            self.run_step(step, job)?;
        }
        job.log(&format!("done: {}", self.output().display()));
        Ok(())
    }

    /// Runs every step from `first` to the end; steps before it only when their result is missing
    /// (for a step of each session, only for the sessions without one).
    pub fn run_from(&self, first: Step, job: &Job) -> Result<()> {
        self.run_until(first, Step::Images, job)
    }

    /// As [`RunConfig::run_from`], stopping after `last`.
    pub fn run_until(&self, first: Step, last: Step, job: &Job) -> Result<()> {
        for step in Step::ALL.into_iter().filter(|&s| s <= last) {
            if step < first && self.done(step) {
                continue;
            }
            // without colouring, a run stops before the colorize step (unless that step is asked for)
            if step == Step::Colorize && !self.color && first != Step::Colorize {
                continue;
            }
            // the image poses are refined for the colouring, when asked for
            if step == Step::Refine && (!self.refine || (!self.color && first != Step::Refine)) {
                continue;
            }
            // the images follow the colouring (the seed points of the COLMAP model), when asked for
            if step == Step::Images && (!self.images || (!self.color && first != Step::Images)) {
                continue;
            }
            self.run_step_for(step, step < first, job)?;
        }
        job.log(&format!("done: {}", self.output().display()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn las_colours_are_whole_8_bit_levels_never_in_the_8_bit_range_unless_black() {
        assert_eq!(las_rgb([0.9, 0.4, 5.9]), [257, 0, 6 * 257]);
        assert_eq!(las_rgb([-3., 255., 300.]), [0, 65535, 65535]);
        for k in 0..=2550 {
            let c = las_rgb([k as f32 / 10.; 3])[0];
            assert!(c == 0 || c > 255, "{c}");
            assert_eq!(c % 257, 0);
        }
    }

    #[test]
    fn params_round_trip_through_json() {
        let mut p = ProcessParams::preset(true);
        p.opt.loop_reg.half_s = 10.0;
        p.opt.stages[1].rtk_c = None;
        p.cloud.free_ratio = 3.0;
        assert_eq!(ProcessParams::from_json(&p.to_json()).unwrap(), p);
    }

    #[test]
    fn missing_keys_take_defaults() {
        let p = ProcessParams::from_json(r#"{"opt": {"loop_reg": {"half_s": 8.0}}}"#).unwrap();
        assert_eq!(p.opt.loop_reg.half_s, 8.0);
        assert_eq!(p.opt.loop_reg.min_inliers, 0.4);
        assert_eq!(p.opt.stages.len(), 3);
        assert_eq!(p.lio, s20_lio::Params::indoor());
        assert!(ProcessParams::from_json("[1, 2]").is_err());
    }

    #[test]
    fn a_partial_result_replaces_the_earlier_one_only_when_kept() {
        let dir = std::env::temp_dir().join(format!("s20_partial_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("a_uncolorized.laz");
        std::fs::write(&out, "earlier").unwrap();
        let part = Partial::new(&out);
        assert_eq!(part.path, dir.join("a_uncolorized.partial.laz"));
        std::fs::write(&part.path, "interrupted").unwrap();
        drop(part);
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "earlier");
        assert!(!dir.join("a_uncolorized.partial.laz").exists());
        let part = Partial::new(&out);
        std::fs::write(&part.path, "new").unwrap();
        part.keep().unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "new");
        assert!(!dir.join("a_uncolorized.partial.laz").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn enu_origin_is_shared_persisted_and_legacy_results_keep_their_frame() {
        let work = std::env::temp_dir().join(format!("s20_enu_{}", std::process::id()));
        std::fs::create_dir_all(&work).unwrap();
        std::fs::write(work.join("origin.txt"), "1000 -200000 0").unwrap();
        let old = work.join("old.txt");
        assert_eq!(
            frames::result_frame(&work, &old).unwrap().output_offset(),
            [1000., -200000., 0.]
        );
        let frame = enu_frame(&work, [[-33.9, 151.2, 35.]]).unwrap();
        frame.write(&work.join("coordinate_frame.json")).unwrap();
        // Re-running or adding a session does not move the existing origin.
        assert_eq!(enu_frame(&work, [[38., 141., 50.]]).unwrap(), frame);
        assert_eq!(enu_frame(&work, []).unwrap(), frame);
        assert_eq!(
            frames::result_frame(&work, &old).unwrap().output_offset(),
            [1000., -200000., 0.]
        );
        let new = work.join("new.txt");
        std::fs::write(&new, "poses").unwrap();
        frames::record(&work, &[&new], Some(frame)).unwrap();
        assert_eq!(frames::result_frame(&work, &new).unwrap(), frame);
        assert_eq!(
            frames::result_frame(&work, &new).unwrap().output_offset(),
            [0.; 3]
        );
        std::fs::remove_file(work.join("coordinate_frame.json")).unwrap();
        assert!(enu_frame(&work, []).is_err());
        std::fs::remove_dir_all(work).unwrap();
    }

    #[test]
    fn revisions_are_saved_listed_renamed_and_deleted_in_order() {
        let work = std::env::temp_dir().join(format!("s20_revisions_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work);
        let c = RunConfig::new(PathBuf::from("raw"), work.clone());
        assert!(c.revisions().is_empty());
        let one = Edits {
            loops: vec![(At::new(1.0), At::new(9.0))],
            ..Default::default()
        };
        let a = c.save_revision("保存 1", false, &Edits::default()).unwrap();
        let b = c.save_revision("最適化の実行時", true, &one).unwrap();
        assert_ne!(a.id, b.id);
        let list = c.revisions();
        assert_eq!(list, vec![a.clone(), b.clone()]);
        c.rename_revision(&b.id, "ループを足した").unwrap();
        assert_eq!(c.revisions()[1].name, "ループを足した");
        assert_eq!(c.revisions()[1].edits, one);
        c.delete_revision(&a.id).unwrap();
        assert_eq!(c.revisions().len(), 1);
        // ids never leave the folder
        assert!(c.delete_revision("../s20_edits").is_err());
        std::fs::remove_dir_all(&work).unwrap();
    }

    #[test]
    fn records_tell_interrupted_and_outdated_steps() {
        let work = std::env::temp_dir().join(format!("s20_records_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work);
        let mut c = RunConfig::new(PathBuf::from("raw"), work.clone());
        std::fs::create_dir_all(c.runs()).unwrap();
        std::fs::create_dir_all(c.session_runs(0)).unwrap();
        std::fs::write(c.lio_file(), "x").unwrap();
        // a result of an older version (no record): done, not stale
        assert!(c.done(Step::Lio));
        assert_eq!(c.staleness(Step::Lio), None);
        // interrupted: the record still says running
        c.write_record(Step::Lio, Some(0), false).unwrap();
        assert!(!c.done(Step::Lio));
        // finished, then the settings change
        c.write_record(Step::Lio, Some(0), true).unwrap();
        assert!(c.done(Step::Lio));
        assert_eq!(c.staleness(Step::Lio), None);
        c.params.lio.down *= 2.0;
        assert_eq!(c.staleness(Step::Lio), Some(Stale::Params));
        // the optimisation also follows the RTK switch
        c.params.lio.down /= 2.0;
        std::fs::write(c.opt_file(), "x").unwrap();
        c.write_record(Step::Opt, None, true).unwrap();
        c.rtk = !c.rtk;
        assert_eq!(c.staleness(Step::Opt), Some(Stale::Params));
        std::fs::remove_dir_all(&work).unwrap();
    }

    #[test]
    fn a_run_that_stops_before_writing_keeps_the_previous_result() {
        let work = std::env::temp_dir().join(format!("s20_keep_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work);
        let c = RunConfig::new(PathBuf::from("raw"), work.clone());
        std::fs::create_dir_all(c.runs()).unwrap();
        std::fs::write(c.opt_file(), "x").unwrap();
        c.write_record(Step::Opt, None, true).unwrap();
        // the optimisation fails (no odometry) before it writes its result
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let l = log.clone();
        let job = Job::new(|_, _, _| {}, move |s| l.lock().unwrap().push(s.into()));
        assert!(c.run_step(Step::Opt, &job).is_err());
        assert!(c.done(Step::Opt));
        assert!(
            log.lock()
                .unwrap()
                .iter()
                .any(|s| s.contains("previous result is kept"))
        );
        // a step that did not finish before stays unfinished
        c.write_record(Step::Opt, None, false).unwrap();
        assert!(
            c.run_step(Step::Opt, &Job::new(|_, _, _| {}, |_| {}))
                .is_err()
        );
        assert!(!c.done(Step::Opt));
        // the result written, then the run failing: not the previous result any more
        let previous = c.previous_result(Step::Opt, None);
        assert!(previous.is_none());
        c.write_record(Step::Opt, None, true).unwrap();
        let previous = c.previous_result(Step::Opt, None);
        c.write_record(Step::Opt, None, false).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(c.opt_file(), "y").unwrap();
        c.restore_result(Step::Opt, None, previous, &Job::new(|_, _, _| {}, |_| {}));
        assert!(!c.done(Step::Opt));
        std::fs::remove_dir_all(&work).unwrap();
    }

    #[test]
    fn heading_holds_go_to_their_session_and_outdate_the_odometry() {
        let work = std::env::temp_dir().join(format!("s20_hold_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work);
        let c = RunConfig::with_sessions(vec!["x/hill".into(), "x/room".into()], work.clone());
        std::fs::create_dir_all(c.session_runs(0)).unwrap();
        std::fs::write(c.lio_file_of(0), "x").unwrap();
        c.write_record(Step::Lio, Some(0), true).unwrap();
        assert_eq!(c.staleness(Step::Lio), None);
        let e = Edits {
            heading_hold: vec![
                (At::new(10.0), At::new(20.0)),
                (At::on("room", 5.0), At::on("room", 8.0)),
                (At::on("hill", -30.0), At::on("hill", -1.0)),
            ],
            ..Default::default()
        };
        c.save_edits(&e).unwrap();
        assert_eq!(c.heading_hold_of(0), [(10.0, 20.0), (-30.0, -1.0)]);
        assert_eq!(c.heading_hold_of(1), [(5.0, 8.0)]);
        assert_eq!(c.staleness(Step::Lio), Some(Stale::Params));
        std::fs::remove_dir_all(&work).unwrap();
    }

    #[test]
    fn sessions_keep_their_steps_apart() {
        let work = std::env::temp_dir().join(format!("s20_sessions_run_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&work);
        let c = RunConfig::with_sessions(
            vec!["x/garden".into(), "y/garden".into(), "x/room".into()],
            work.clone(),
        );
        let names: Vec<&str> = c.sessions.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["garden", "garden_2", "room"]);
        assert_eq!(c.name, format!("s20_sessions_run_{}", std::process::id()));
        assert_eq!(
            c.lio_file_of(1),
            work.join("sessions/garden_2/runs/lio.txt")
        );
        assert_eq!(c.opt_file(), work.join("runs/opt.txt"));
        let t0 = std::time::SystemTime::now();
        let touch = |p: &Path, secs: u64| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "x").unwrap();
            let f = std::fs::File::options().write(true).open(p).unwrap();
            f.set_modified(t0 + std::time::Duration::from_secs(secs))
                .unwrap();
        };
        for k in 0..3 {
            touch(&c.result_of(Step::Extract, Some(k)), 0);
            touch(&c.lio_file_of(k), 10);
        }
        // the set of all sessions is part of the extraction
        assert!(!c.done(Step::Extract));
        let dirs = c.set_dirs();
        std::fs::create_dir_all(c.data()).unwrap();
        std::fs::write(
            c.set_file(),
            dirs.iter()
                .map(|d| format!("{} 0\n", d.display()))
                .collect::<String>(),
        )
        .unwrap();
        assert!(c.done(Step::Extract) && c.done(Step::Lio));
        assert_eq!(c.staleness(Step::Lio), None);
        // extracting one session again makes only its odometry stale
        touch(&c.result_of(Step::Extract, Some(2)), 20);
        assert_eq!(c.staleness_of(Step::Lio, Some(0)), None);
        assert_eq!(c.staleness_of(Step::Lio, Some(2)), Some(Stale::Earlier));
        // the optimisation follows every session
        touch(&c.opt_file(), 15);
        assert_eq!(c.staleness(Step::Opt), Some(Stale::Earlier));
        touch(&c.opt_file(), 40);
        frames::record(
            &c.data(),
            &[&c.opt_file()],
            Some(CoordinateFrame::LocalSlam),
        )
        .unwrap();
        assert_eq!(c.staleness(Step::Opt), None);
        touch(&c.lio_file_of(1), 50);
        assert_eq!(c.staleness(Step::Opt), Some(Stale::Earlier));
        // a session left out: the set is not current
        let fewer =
            RunConfig::with_sessions(vec!["x/garden".into(), "y/garden".into()], work.clone());
        assert!(!fewer.done(Step::Extract));
        std::fs::remove_dir_all(&work).unwrap();
    }

    #[test]
    fn sessions_keep_their_results_when_added_and_taken_out() {
        let base = std::env::temp_dir().join(format!("s20_adopt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (raw, raw2) = (base.join("room"), base.join("room2"));
        for r in [&raw, &raw2] {
            std::fs::create_dir_all(r).unwrap();
            std::fs::write(r.join("all_0.bag"), "bag").unwrap();
        }
        let work = base.join("work");
        RunConfig::create_work(&raw, &work).unwrap();
        let one = RunConfig::of_sessions(RunConfig::open_work(&work).unwrap(), work.clone());
        // one session is kept in its folder too
        assert_eq!(one.session_dir(0), work.join("sessions").join("room"));
        std::fs::create_dir_all(one.session_data(0)).unwrap();
        std::fs::create_dir_all(one.session_runs(0)).unwrap();
        s20_io::dataset::Source {
            bag: raw.join("all_0.bag"),
            size: 3,
        }
        .write(&one.session_data(0))
        .unwrap();
        std::fs::write(one.lio_file(), "x").unwrap();
        let mut v = one.sessions.clone();
        v.push(one.new_session(raw2.clone()));
        let two = RunConfig::of_sessions(v, work.clone());
        two.save_sessions().unwrap();
        assert!(two.session_done(Step::Lio, 0));
        // without the first, the second alone; the first's results stay for adding it again
        let alone = RunConfig::of_sessions(vec![two.sessions[1].clone()], work.clone());
        alone.save_sessions().unwrap();
        assert_eq!(alone.session_dir(0), work.join("sessions").join("room2"));
        assert!(one.lio_file().exists());
        let back = alone.new_session(raw.clone());
        assert_eq!(back.name, "room");
        // none: the work folder still opens
        RunConfig::of_sessions(Vec::new(), work.clone())
            .save_sessions()
            .unwrap();
        assert!(RunConfig::open_work(&work).unwrap().is_empty());
        // another recording of the same folder name does not take the results of the first
        let other = base.join("x").join("room");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("all_1.bag"), "other").unwrap();
        let empty = RunConfig::of_sessions(Vec::new(), work.clone());
        assert_eq!(empty.new_session(other.clone()).name, "room_2");
        assert_eq!(empty.new_session(raw.clone()).name, "room");
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn a_moved_recording_keeps_its_extraction() {
        let base = std::env::temp_dir().join(format!("s20_move_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (raw, moved, other) = (base.join("a/rec"), base.join("b/rec"), base.join("c/rec"));
        for (r, bag) in [(&raw, "bag"), (&moved, "bag"), (&other, "other bag")] {
            std::fs::create_dir_all(r).unwrap();
            std::fs::write(r.join("all_0.bag"), bag).unwrap();
        }
        let work = base.join("work");
        RunConfig::create_work(&raw, &work).unwrap();
        let mut c = RunConfig::of_sessions(RunConfig::open_work(&work).unwrap(), work.clone());
        // not extracted yet: any raw folder will do
        assert!(c.same_recording(0, &other));
        std::fs::create_dir_all(c.session_data(0)).unwrap();
        s20_io::dataset::Source {
            bag: raw.join("all_0.bag"),
            size: 3,
        }
        .write(&c.session_data(0))
        .unwrap();
        assert!(c.same_recording(0, &moved));
        assert!(!c.same_recording(0, &other));
        c.move_raw(0, moved.clone()).unwrap();
        assert_eq!(RunConfig::open_work(&work).unwrap()[0].raw, moved);
        let src = s20_io::dataset::Source::read(&c.session_data(0)).unwrap();
        assert_eq!((src.bag, src.size), (moved.join("all_0.bag"), 3));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn work_folders_list_their_raw_folder() {
        let base = std::env::temp_dir().join(format!("s20_work_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let raw = base.join("rec");
        std::fs::create_dir_all(&raw).unwrap();
        assert_eq!(RunConfig::folder_kind(&raw), FolderKind::Other);
        let work = base.join("rec_s20");
        assert!(RunConfig::create_work(&raw, &work).is_err());
        std::fs::write(raw.join("all_0.bag"), "").unwrap();
        assert_eq!(RunConfig::folder_kind(&raw), FolderKind::Raw);
        RunConfig::create_work(&raw, &work).unwrap();
        assert_eq!(RunConfig::folder_kind(&work), FolderKind::Work);
        assert_eq!(RunConfig::open_work(&work).unwrap()[0].raw, raw);
        // not over a folder in use
        assert!(RunConfig::create_work(&raw, &work).is_err());
        // a work folder of an older version is told apart, not opened
        let old = base.join("old");
        std::fs::create_dir_all(old.join("data")).unwrap();
        std::fs::write(
            old.join("data").join(s20_io::dataset::Source::FILE),
            "bag\n1\n",
        )
        .unwrap();
        assert_eq!(RunConfig::folder_kind(&old), FolderKind::Work);
        assert!(RunConfig::open_work(&old).is_err());
        assert!(RunConfig::open_work(&base.join("none")).is_err());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn presets_keep_the_previous_values() {
        let o = ProcessParams::preset(true);
        assert_eq!(
            (o.opt.every, o.cloud.free_ratio, o.cloud.sample_leaf),
            (8, 2.0, 0.005)
        );
        assert_eq!(o.cloud, ProcessParams::preset(false).cloud);
        assert_eq!(o.color.max_range, 40.0);
        let i = ProcessParams::preset(false);
        assert_eq!((i.opt.every, i.lio.down), (8, 0.1));
        assert_eq!(i.color, o.color);
    }

    #[test]
    fn work_settings_are_saved_in_the_work_folder() {
        let work = std::env::temp_dir().join(format!("s20_settings_{}", std::process::id()));
        std::fs::create_dir_all(&work).unwrap();
        assert!(WorkSettings::load(&work).is_none());
        let w = WorkSettings {
            outdoor: true,
            params: ProcessParams::preset(true),
            rtk: true,
            ppk: false,
            laz: false,
            images: true,
            refine: false,
        };
        w.save(&work).unwrap();
        assert_eq!(WorkSettings::load(&work).unwrap().unwrap(), w);
        std::fs::write(work.join("settings.json"), "{").unwrap();
        assert!(WorkSettings::load(&work).unwrap().is_err());
        std::fs::remove_dir_all(&work).unwrap();
    }

    #[test]
    fn ppk_is_explicit_and_does_not_silently_fall_back_to_rtk() {
        let mut c = RunConfig::new("missing_raw".into(), "missing_work".into());
        c.rtk = false;
        assert!(c.rtk_options().is_none());
        let original = c.step_params(Step::Opt);
        assert!(original.get("use_ppk").is_none());
        c.ppk = true;
        c.rtk_exclude.push((At::new(10.), At::new(20.)));
        let source = c.rtk_options().unwrap();
        assert_eq!(source.ppk, vec![c.ppk_file()]);
        assert_eq!(source.exclude, vec![(At::new(10.), At::new(20.))]);
        assert_ne!(original, c.step_params(Step::Opt));
    }
    #[test]
    fn settings_at_an_old_default_take_the_new_one() {
        let mut p = ProcessParams::preset(true);
        assert!(!p.update_old_defaults());
        p.cloud.anchor_grid = 0.02;
        assert!(p.update_old_defaults());
        assert_eq!(p, ProcessParams::preset(true));
        p.cloud.anchor_grid = 0.03;
        assert!(!p.update_old_defaults());
        assert_eq!(p.cloud.anchor_grid, 0.03);
    }
    #[test]
    fn the_tiles_leave_the_cloud_and_colour_records_current() {
        let mut c = RunConfig::new(PathBuf::from("raw"), PathBuf::from("work"));
        let (cloud, colour) = (c.step_params(Step::Cloud), c.step_params(Step::Colorize));
        // a record of an older version, with the tile settings
        let mut old = json(&c.params.cloud);
        assert!(old.get("tile_obs").is_some());
        old["tile_points"] = 12_000_000.into();
        assert_eq!(comparable(old), cloud);
        c.params.cloud.tile_obs /= 2;
        c.params.cloud.tile_size = 25.;
        c.params.color.tile_memory = 0.2;
        assert_eq!(c.step_params(Step::Cloud), cloud);
        assert_eq!(c.step_params(Step::Colorize), colour);
        c.params.cloud.anchor_grid = 0.02;
        assert_ne!(c.step_params(Step::Cloud), cloud);
    }
    #[test]
    fn ppk_engine_update_invalidates_completed_optimisation() {
        let work = std::env::temp_dir().join(format!("s20_ppk_engine_{}", std::process::id()));
        let raw = work.join("raw");
        let mut c = RunConfig::new(raw.clone(), work.clone());
        c.ppk = true;
        std::fs::create_dir_all(c.runs()).unwrap();
        std::fs::create_dir_all(c.session_runs(0)).unwrap();
        std::fs::create_dir_all(raw.join("info")).unwrap();
        for path in [
            raw.join("BASE.BIN"),
            raw.join("ROVER_PPKRAW.bin"),
            c.ppk_file(),
            c.opt_file(),
        ] {
            std::fs::write(path, "fixture").unwrap();
        }
        let enu = CoordinateFrame::local_enu([38., 141., 40.]).unwrap();
        frames::record(&c.data(), &[&c.opt_file()], Some(enu)).unwrap();
        // A completed old record has identical user settings but no engine revision.
        let mut old = c.step_params(Step::Opt);
        old.as_object_mut().unwrap().remove("ppk_engine");
        std::fs::write(
            c.record_file(Step::Opt),
            serde_json::to_vec(&StepRecord {
                done: true,
                params: old,
            })
            .unwrap(),
        )
        .unwrap();
        assert!(c.done(Step::Opt));
        assert_eq!(c.staleness(Step::Opt), Some(Stale::Params));
        c.write_record(Step::Opt, None, true).unwrap();
        assert_eq!(c.staleness(Step::Opt), None);
        // Hand-created/legacy PPK outputs without a record cannot prove their engine.
        std::fs::remove_file(c.record_file(Step::Opt)).unwrap();
        assert_eq!(c.staleness(Step::Opt), Some(Stale::Params));
        c.ppk = false;
        assert_eq!(c.staleness(Step::Opt), None);
        std::fs::remove_dir_all(work).unwrap();
    }

    /// The cloud and colouring of a real work folder (`S20_PROGRESS_WORK`; its output is made
    /// again) report their progress under the step's name, only going forward, and along the way
    /// rather than all at the end.
    #[test]
    #[ignore]
    fn cloud_and_colour_progress_goes_forward() {
        use std::sync::{Arc, Mutex};
        let work = PathBuf::from(std::env::var("S20_PROGRESS_WORK").expect("S20_PROGRESS_WORK"));
        let cfg = RunConfig::of_sessions(RunConfig::open_work(&work).unwrap(), work);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let t0 = std::time::Instant::now();
        let job = Job::new(
            move |stage, done, total| {
                if total > 0 {
                    let f = done as f64 / total as f64;
                    s.lock()
                        .unwrap()
                        .push((stage.to_string(), f, t0.elapsed().as_secs_f64()));
                }
            },
            |l| eprintln!("{l}"),
        );
        for step in [Step::Cloud, Step::Export, Step::Colorize] {
            cfg.run_step_for(step, false, &job).unwrap();
        }
        let seen = seen.lock().unwrap();
        for key in ["cloud", "colorize"] {
            let r: Vec<(f64, f64)> = seen
                .iter()
                .filter(|x| x.0 == key)
                .map(|x| (x.1, x.2))
                .collect();
            assert!(r.windows(2).all(|w| w[1].0 >= w[0].0), "{key} went back");
            let (a, b) = (r.first().unwrap().1, r.last().unwrap().1);
            for k in 1..10 {
                let t = a + (b - a) * k as f64 / 10.0;
                let f = r.iter().take_while(|x| x.1 <= t).last().unwrap().0;
                eprintln!("{key}: {}% of the time, {:.0}% shown", k * 10, f * 100.0);
            }
            let along = r.iter().filter(|x| x.0 > 0.0 && x.0 < POTREE_FROM).count();
            assert!(along >= 10, "{key}: {along} reports before the octree");
        }
    }
}
