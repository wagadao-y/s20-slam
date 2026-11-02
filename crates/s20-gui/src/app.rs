//! The S20 SLAM window: input and settings, the processing steps with progress, the log, and a 3D
//! preview of the trajectory and the point cloud.

mod device_calib;
mod history;
mod match_edit;
mod raw_view;
mod traj_edit;
mod welcome;
mod workspace;
use crate::lang::{Lang, tl};
use match_edit::MATCH;
use traj_edit::{EditKind, EditRef, Placing};
use workspace::{InspectorTab, PointColor, ScanRange};

use crate::camera::Camera;
use crate::notice::Level;
use crate::quality::{self as q, METRICS};
use crate::render::{Coloring, DrawNode, DrawOptions, Edl, Line, PointRenderer};
pub use crate::scene::PreviewSource;
use crate::scene::{self, NodePoints, Scene, TrajectorySource};
use eframe::egui;
use egui_phosphor::regular as icon;
use glam::{DMat4, DVec3};
use s20_pipeline::job::{Job, is_cancelled};
use s20_pipeline::runlog::PathMask;
use s20_pipeline::{
    At, Edits, FolderKind, NameProblem, PointAt, ProcessParams, RunConfig, Step, WorkSettings,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Instant;

/// Dropping a preview request cancels its bounded cache build.
struct Loading {
    rx: mpsc::Receiver<anyhow::Result<Scene>>,
    cancel: Arc<AtomicBool>,
}
impl Drop for Loading {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

const SETTINGS_KEY: &str = "s20-gui-settings";
/// Preview points at most, by default and the slider range.
const PREVIEW_DEFAULT: usize = 5_000_000;
const PREVIEW_RANGE: std::ops::RangeInclusive<usize> = 500_000..=30_000_000;

/// The mutually exclusive GNSS choices; saved settings retain the existing boolean fields.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GnssMode {
    None,
    Rtk,
    Ppk,
}

impl GnssMode {
    fn label(self) -> &'static str {
        match self {
            Self::None => tl("GNSSを使わない", "Don't use GNSS"),
            Self::Rtk => tl("機体の RTK を使う", "Use the device's RTK"),
            Self::Ppk => tl("PPKを計算して使う", "Compute and use PPK"),
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Self::None => tl(
                "GNSSの位置を軌跡の補正に使いません。",
                "GNSS positions are not used to correct the trajectory.",
            ),
            Self::Rtk => tl(
                "機体がリアルタイムで記録した RTK の FIX 位置で軌跡を補正します。",
                "The trajectory is corrected with the RTK FIX positions the device recorded in real time.",
            ),
            Self::Ppk => tl(
                "BASE / ROVERの生観測から前方解・後方解のPPK測位を行い、品質条件を満たす位置で軌跡を補正します。計算結果は再利用されます。",
                "Forward and backward PPK solutions are computed from the BASE / ROVER raw observations, and the positions that meet the quality conditions correct the trajectory. The results are reused.",
            ),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    work: String,
    /// work folders opened last, the latest first
    recent: Vec<PathBuf>,
    /// where the last new work folder was made
    last_location: Option<PathBuf>,
    /// the preset the parameters started from (changed values are shown against it)
    outdoor: bool,
    /// parameters of the steps; None in settings saved before they could be edited (the preset)
    params: Option<ProcessParams>,
    rtk: bool,
    ppk: bool,
    /// point clouds written as LAZ (else LAS)
    laz: bool,
    /// the export writes the photos with their poses (`images/`)
    images: bool,
    /// every image's pose is refined against the cloud before the colouring
    refine: bool,
    point_size: f32,
    /// Potree's adaptive point size (by the spacing of the drawn points): the output cloud, the
    /// scans (off by default: thinned hard and of very uneven density, their discs would be large
    /// and hide the misalignments the scans are looked at for)
    adaptive: bool,
    adaptive_scans: bool,
    /// preview points at most
    preview_points: usize,
    /// what the preview shows: the scans placed with the trajectory, or the finished cloud
    view: PreviewSource,
    scan_trajectory: TrajectorySource,
    /// every n-th raw point is kept for the scan preview
    scan_stride: usize,
    edl: bool,
    show_traj: bool,
    /// trajectory drawn over the points (ceilings would hide it indoors)
    traj_on_top: bool,
    /// RTK fixes on the trajectory
    show_rtk: bool,
    /// the corrections' intervals in their colours on the trajectory (else its own colours show
    /// through; the interval being placed is coloured all the same)
    show_edit_spans: bool,
    /// the camera follows the time slider
    follow: bool,
    /// camera images of the slider time
    show_images: bool,
    /// trajectory colours: 0 time, else quality metric index + 1
    traj_metric: usize,
    /// value shown red, per metric
    metric_bad: [f64; 7],
    show_points: bool,
    /// how the points are coloured: the scans, the output cloud
    point_color_scans: PointColor,
    point_color_output: PointColor,
    /// the scans drawn by time: all, around the marked time or up to it
    scan_range: ScanRange,
    /// seconds before and after the marked time, around it
    scan_window: f64,
    show_project: bool,
    show_log: bool,
    inspector_open: bool,
    inspector_tab: InspectorTab,
    /// the interface's language; None (never chosen) follows Windows' display language
    lang: Option<Lang>,
}
impl Default for Settings {
    fn default() -> Self {
        Settings {
            work: String::new(),
            recent: Vec::new(),
            last_location: None,
            outdoor: false,
            params: None,
            rtk: true,
            ppk: false,
            laz: true,
            images: false,
            refine: true,
            point_size: 2.0,
            adaptive: true,
            adaptive_scans: false,
            preview_points: PREVIEW_DEFAULT,
            view: PreviewSource::Colored,
            scan_trajectory: TrajectorySource::Optimized,
            scan_stride: 4,
            edl: true,
            show_traj: true,
            traj_on_top: true,
            show_rtk: true,
            show_edit_spans: true,
            follow: true,
            show_images: true,
            traj_metric: 0,
            metric_bad: METRICS.map(|m| m.bad),
            show_points: true,
            point_color_scans: PointColor::Reflectivity,
            point_color_output: PointColor::Rgb,
            scan_range: ScanRange::All,
            scan_window: 2.0,
            show_project: true,
            show_log: false,
            inspector_open: false,
            inspector_tab: InspectorTab::Display,
            lang: None,
        }
    }
}

impl Settings {
    /// Puts `work` first in the recent work folders.
    fn remember(&mut self, work: &Path) {
        self.recent.retain(|p| p != work);
        self.recent.insert(0, work.to_path_buf());
        self.recent.truncate(10);
    }

    fn gnss_mode(&self) -> GnssMode {
        if self.ppk {
            GnssMode::Ppk
        } else if self.rtk {
            GnssMode::Rtk
        } else {
            GnssMode::None
        }
    }

    fn set_gnss_mode(&mut self, mode: GnssMode) {
        self.rtk = mode == GnssMode::Rtk;
        self.ppk = mode == GnssMode::Ppk;
    }

    fn params(&mut self) -> &mut ProcessParams {
        let outdoor = self.outdoor;
        self.params
            .get_or_insert_with(|| ProcessParams::preset(outdoor))
    }

    /// The processing settings, as kept in the work folder.
    fn work_settings(&self) -> WorkSettings {
        WorkSettings {
            outdoor: self.outdoor,
            params: self
                .params
                .clone()
                .unwrap_or_else(|| ProcessParams::preset(self.outdoor)),
            rtk: self.rtk,
            ppk: self.ppk,
            laz: self.laz,
            images: self.images,
            refine: self.refine,
        }
    }

    fn set_work_settings(&mut self, w: WorkSettings) {
        self.outdoor = w.outdoor;
        self.params = Some(w.params);
        self.rtk = w.rtk;
        self.ppk = w.ppk;
        self.laz = w.laz;
        self.images = w.images;
        self.refine = w.refine;
    }

    /// Whether the parameters differ from their preset.
    fn params_modified(&self) -> bool {
        self.params
            .as_ref()
            .is_some_and(|p| *p != ProcessParams::preset(self.outdoor))
    }
}

/// Steps to run: from `first` to `last` (None: to the end).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Run {
    first: Step,
    last: Option<Step>,
}

enum Event {
    Progress(String, u64, u64),
    Log(String),
    Done(anyhow::Result<()>),
}

struct Running {
    rx: mpsc::Receiver<Event>,
    cancel: Arc<AtomicBool>,
    started: Instant,
    opt_progress: OptProgress,
    /// the run's log file in the work folder (`logs/`)
    log_file: Option<Arc<s20_pipeline::runlog::LogFile>>,
}

/// Shares the optimisation bar across its phases: PPK when it is computed (not when cached), then
/// loading, the pose graph on the GNSS fixes, the loops' registration and the iterations. The
/// iterations take most of the time (a 30-minute recording: loading 9 s, GNSS 6 s, iterations
/// 170 s), so they take most of the bar.
#[derive(Default)]
struct OptProgress {
    ppk: bool,
    fraction: f32,
    label: &'static str,
}

impl OptProgress {
    fn update(&mut self, stage: &str, done: u64, total: u64) -> f32 {
        let f = if total > 0 {
            (done as f32 / total as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // the optimisation's own phases: where each starts and how much of the bar it takes
        let (label, start, width) = match stage {
            "ppk decode" | "ppk forward" | "ppk backward" => {
                self.ppk = true;
                let (label, overall) = match stage {
                    "ppk decode" => (
                        tl(
                            "PPK測位（データの読み込み）",
                            "PPK positioning (reading the data)",
                        ),
                        0.0,
                    ),
                    "ppk forward" => (
                        tl("PPK測位（前方解）", "PPK positioning (forward solution)"),
                        f * 0.4,
                    ),
                    _ => (
                        tl("PPK測位（後方解）", "PPK positioning (backward solution)"),
                        0.4 + f * 0.4,
                    ),
                };
                self.label = label;
                self.fraction = self.fraction.max(overall.min(0.99));
                return self.fraction;
            }
            "opt load" => (tl("データの読み込み", "Reading the data"), 0.0, 0.05),
            "opt gnss" => (tl("GNSS に合わせる", "Fitting to GNSS"), 0.05, 0.05),
            "opt loops" => (tl("ループの位置合わせ", "Aligning the loops"), 0.10, 0.05),
            _ => (tl("軌跡の最適化", "Trajectory optimization"), 0.15, 0.85),
        };
        self.label = label;
        let x = start + f * width;
        let overall = if self.ppk { 0.8 + x * 0.2 } else { x };
        // Extra iterations and saving may follow the nominal iteration count.
        // Only successful completion of the step represents 100%.
        self.fraction = self.fraction.max(overall.min(0.99));
        self.fraction
    }
}

/// Everything that changes the trajectory geometry or colours; the camera is deliberately absent.
#[derive(PartialEq)]
struct TrajectoryKey {
    scene_revision: u64,
    show_traj: bool,
    show_rtk: bool,
    show_edit_spans: bool,
    metric: usize,
    bad: u64,
    pixels: u32,
    edits: Edits,
    placing: Option<Placing>,
    placing_hover: Option<f64>,
    selected_edit: Option<EditRef>,
    marker: Option<(f64, DVec3)>,
    /// the sessions hidden, sorted
    hidden: Vec<String>,
}

type WorstPlaces = Arc<[(f64, f64)]>;
type WorstCache = ((u64, usize, u64), WorstPlaces);

#[derive(Clone, Copy, PartialEq)]
enum StepState {
    Pending,
    Done,
    /// done, but older than an earlier step's result (or the corrections)
    Stale,
    Running(f32),
    Failed,
}

pub struct App {
    s: Settings,
    job: Option<Running>,
    steps: HashMap<Step, StepState>,
    /// why the stale steps are stale
    stale_why: HashMap<Step, s20_pipeline::Stale>,
    /// the settings the step states were last computed with (they change the stale states)
    steps_key: Option<(Option<ProcessParams>, bool, bool)>,
    /// the processing settings as last read from or written to the open work folder
    work_settings_saved: Option<WorkSettings>,
    current: Option<Step>,
    log: Vec<String>,
    log_view: crate::log_view::LogView,
    /// outcomes of actions shown over the view for a few seconds (also in the log)
    notices: crate::notice::Notices,
    error: Option<String>,
    scene: Scene,
    scene_revision: u64,
    trajectory_cache: Option<(TrajectoryKey, Arc<[Line]>)>,
    worst_cache: Option<WorstCache>,
    loading: Option<Loading>,
    /// chunks drawn last frame, by draw id (for picking), and their points
    drawn: Vec<Arc<NodePoints>>,
    /// by octree node drawn: its points' spacings for the adaptive size, with the identity of
    /// the nodes drawn under it they were computed for
    point_spacings: HashMap<usize, (u64, Arc<[f32]>)>,
    drawn_points: usize,
    camera: Camera,
    renderer: Option<PointRenderer>,
    framed: bool,
    /// animated retarget: camera at the start, new orbit centre, start time
    /// animated retarget: camera at the start, new orbit centre, new distance, start time
    flight: Option<(Camera, DVec3, Option<f64>, Instant)>,
    /// a time picked on the trajectory (sensor clock) and its position
    marker: Option<(f64, DVec3)>,
    /// corrections of the work folder
    edits: Edits,
    /// undo and redo of the corrections
    edit_history: history::EditHistory,
    /// the saved revisions of the corrections, oldest first
    revisions: Vec<s20_pipeline::EditsRevision>,
    revisions_open: bool,
    /// the revision a first click on delete armed
    revision_delete: Option<String>,
    /// the raw data window
    raw_view: raw_view::RawView,
    /// a correction being placed on the trajectory, and the time under the pointer meanwhile
    placing: Option<Placing>,
    placing_hover: Option<f64>,
    /// a correction's end being dragged along the trajectory: (correction, second end)
    dragging_end: Option<(EditRef, bool)>,
    /// the correction picked on the trajectory or in the list
    selected_edit: Option<EditRef>,
    /// sessions hidden in the viewer (by name)
    hidden: HashSet<String>,
    /// the session of the time slider
    sel_session: usize,
    /// adding matched points: Some(None) waits for the first point, Some(Some(p)) for the second
    /// (the first: its session and the point as stored)
    matching: Option<Option<(usize, PointAt)>>,
    /// the matched point selected; the one being dragged (match, second point); the shown point
    /// under the pointer while adding or dragging, with its session
    selected_match: Option<usize>,
    dragging_match: Option<(usize, bool)>,
    match_hover: Option<(DVec3, Option<usize>)>,
    /// the session of each chunk drawn last frame (for picking)
    drawn_session: Vec<Option<usize>>,
    /// camera images shown (left, right): file and texture; requested file and decoder channel
    img_shown: [Option<(PathBuf, egui::TextureHandle)>; 2],
    img_want: [Option<PathBuf>; 2],
    img_busy: [bool; 2],
    img_rx: Option<mpsc::Receiver<(usize, PathBuf, Option<egui::ColorImage>)>>,
    img_tx: Option<mpsc::Sender<(usize, PathBuf, Option<egui::ColorImage>)>>,
    pixels_per_point: f32,
    /// the current preview has been drawn completely (all nodes uploaded)
    preview_drawn: bool,
    /// the parameter window is open; a preset waiting for confirmation (it replaces changed
    /// values); the outcome of the last JSON export or import (ok, message)
    params_open: bool,
    /// the preview to go back to when the debug layers are left
    /// the heights coloured over when set by hand (else the scene's)
    height_manual: Option<[f64; 2]>,
    /// the heights the points are cut below and above, by the slider on the view (none: not cut)
    height_clip: [Option<f64>; 2],
    processing_settings_open: bool,
    help_open: bool,
    /// the window with the version and where to report problems
    about_open: bool,
    /// the license window, and whether it shows the third-party licenses (else the NOTICE)
    licenses_open: bool,
    licenses_third_party: bool,
    confirm_preset: Option<bool>,
    /// a run that would start by redoing a finished step, waiting for confirmation
    confirm_run: Option<Run>,
    params_msg: Option<(bool, String)>,
    /// the new work folder dialog
    new_work: Option<NewWork>,
    /// a session to be replaced by another recording (its raw folder), waiting for confirmation
    confirm_replace: Option<(usize, PathBuf)>,
    /// a session to be taken out, waiting for confirmation
    confirm_remove: Option<usize>,
    /// a session being renamed, and the name typed
    rename_session: Option<(usize, String)>,
    /// closing the window while processing, waiting for confirmation; and once confirmed
    confirm_exit: bool,
    exit_confirmed: bool,
    /// the window's place kept over a restart
    window_place: crate::window_place::WindowPlace,
    /// the device calibrations, and their window when open
    device: s20_pipeline::device::Store,
    device_calib: Option<device_calib::CalibWindow>,
    smoke: Option<Smoke>,
}

/// The new work folder dialog: the raw folder it lists, its name and the folder it goes in.
struct NewWork {
    raw: String,
    name: String,
    location: String,
    /// the name was typed (else it follows the raw folder)
    named: bool,
}

impl NewWork {
    fn new(s: &Settings, raw: Option<&Path>) -> Self {
        let mut w = NewWork {
            raw: String::new(),
            name: String::new(),
            location: s
                .last_location
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            named: false,
        };
        if let Some(r) = raw {
            w.set_raw(r);
        }
        w
    }

    /// A chosen raw folder; the location, when there is none yet, is the raw folder's.
    fn set_raw(&mut self, raw: &Path) {
        self.raw = raw.display().to_string();
        if self.location.trim().is_empty()
            && let Some(p) = raw.parent()
        {
            self.location = p.display().to_string();
        }
        self.follow_raw();
    }

    /// The name from the raw folder, unless one was typed. Beside the raw folder it is
    /// `<raw>_s20`, so the two do not collide.
    fn follow_raw(&mut self) {
        let raw = Path::new(self.raw.trim());
        if !self.named
            && let Some(n) = raw.file_name()
        {
            let n = n.to_string_lossy();
            self.name = if raw.parent() == Some(Path::new(self.location.trim())) {
                format!("{n}_s20")
            } else {
                n.into_owned()
            };
        }
    }

    fn path(&self) -> PathBuf {
        Path::new(self.location.trim()).join(self.name.trim())
    }
}

/// A folder's name to show, else its path.
fn folder_label(p: &Path) -> String {
    p.file_name().map_or_else(
        || p.display().to_string(),
        |n| n.to_string_lossy().to_string(),
    )
}

/// Automated check: optionally run steps, then save a screenshot once the preview is shown, and quit.
pub struct Smoke {
    pub screenshot: PathBuf,
    pub work: Option<String>,
    pub raw: Option<String>,
    pub start: Option<Step>,
    /// preview points at most (else the default)
    pub points: Option<usize>,
    /// what to show (else the default)
    pub view: Option<PreviewSource>,
    pub scan_trajectory: Option<TrajectorySource>,
    /// adaptive point size on or off (else the default)
    pub adaptive: Option<bool>,
    /// camera distance factor after framing
    pub zoom: Option<f64>,
    /// trajectory colour metric (0 time, else metric + 1), and fly to its worst place
    pub metric: Option<usize>,
    pub goto_worst: bool,
    /// mark this time (seconds from the start) once the preview is there
    pub time: Option<f64>,
    /// the parameter window open, with its sections expanded
    pub params: bool,
    pub ppk: bool,
    /// the interface's language (else the default)
    pub lang: Option<Lang>,
    went: bool,
    started: bool,
    requested: bool,
    t0: Instant,
}

impl Smoke {
    pub fn new(
        screenshot: PathBuf,
        work: Option<String>,
        raw: Option<String>,
        start: Option<&str>,
    ) -> Self {
        Smoke {
            screenshot,
            work,
            raw,
            start: start.and_then(step_of),
            points: None,
            view: None,
            scan_trajectory: None,
            adaptive: None,
            zoom: None,
            metric: None,
            goto_worst: false,
            time: None,
            params: false,
            ppk: false,
            lang: None,
            went: false,
            started: false,
            requested: false,
            t0: Instant::now(),
        }
    }
}

fn step_label(s: Step) -> &'static str {
    match s {
        Step::Extract => tl(
            "抽出（bag の索引・IMU・GNSS）",
            "Extraction (bag index, IMU, GNSS)",
        ),
        Step::Lio => tl("オドメトリ（LIO）", "Odometry (LIO)"),
        Step::Opt => tl("軌跡の最適化", "Trajectory optimization"),
        Step::Cloud => tl("点群生成", "Point cloud"),
        Step::Export => tl("軌跡・画像姿勢の出力", "Trajectory and image pose export"),
        Step::Refine => tl("画像の姿勢合わせ", "Image pose refinement"),
        Step::Colorize => tl("着色", "Colorization"),
        Step::Images => tl(
            "画像の出力（PointClouds Studio 形式・COLMAP）",
            "Image export (PointClouds Studio format, COLMAP)",
        ),
    }
}

fn step_of(key: &str) -> Option<Step> {
    Step::ALL.into_iter().find(|s| s.key() == key)
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    for path in [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            fonts
                .font_data
                .insert("Japanese".into(), egui::FontData::from_owned(bytes).into());
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .push("Japanese".into());
            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .push("Japanese".into());
            break;
        }
    }
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    ctx.set_fonts(fonts);
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, smoke: Option<Smoke>) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        install_fonts(&cc.egui_ctx);
        let rs = cc.wgpu_render_state.as_ref().expect("wgpu renderer");
        let mut s: Settings = if smoke.is_some() {
            Settings::default()
        } else {
            cc.storage
                .and_then(|st| eframe::get_value(st, SETTINGS_KEY))
                .unwrap_or_default()
        };
        if let Some(p) = &mut s.params {
            p.update_old_defaults();
        }
        if let Some(l) = smoke.as_ref().and_then(|sm| sm.lang) {
            s.lang = Some(l);
        }
        crate::lang::set(s.lang.unwrap_or_else(Lang::of_system));
        if let Some(sm) = &smoke {
            s.ppk = sm.ppk;
            if let Some(w) = &sm.work {
                s.work = w.clone();
            }
            if let Some(n) = sm.points {
                s.preview_points = n;
            }
            if let Some(v) = sm.view {
                s.view = v;
            }
            if let Some(v) = sm.scan_trajectory {
                s.scan_trajectory = v;
            }
            if let Some(a) = sm.adaptive {
                s.adaptive = a;
                s.adaptive_scans = a;
            }
            if let Some(m) = sm.metric {
                s.traj_metric = m;
            }
        }
        // `--raw`: the work folder of the command line lists that raw folder
        if let Some(sm) = &smoke
            && let (Some(w), Some(r)) = (&sm.work, &sm.raw)
            && RunConfig::project_sessions(Path::new(w)).is_none()
        {
            let _ = RunConfig::with_sessions(vec![r.into()], w.into()).save_sessions();
        }
        // the start screen chooses the work folder (the recent ones stay listed)
        if smoke.is_none() {
            s.work.clear();
        }
        if !s.work.trim().is_empty() {
            let work = PathBuf::from(s.work.trim());
            if RunConfig::open_work(&work).is_ok() {
                s.remember(&work);
            } else {
                s.work.clear();
            }
        }
        let mut app = App {
            s,
            job: None,
            steps: HashMap::new(),
            stale_why: HashMap::new(),
            steps_key: None,
            work_settings_saved: None,
            current: None,
            log: Vec::new(),
            log_view: crate::log_view::LogView::default(),
            notices: Default::default(),
            error: None,
            scene: Scene::default(),
            scene_revision: 0,
            trajectory_cache: None,
            worst_cache: None,
            loading: None,
            point_spacings: HashMap::new(),
            drawn: Vec::new(),
            drawn_points: 0,
            camera: Camera::default(),
            renderer: Some(PointRenderer::new(rs.device.clone(), rs.queue.clone())),
            framed: false,
            flight: None,
            marker: None,
            edits: Edits::default(),
            edit_history: Default::default(),
            revisions: Vec::new(),
            revisions_open: false,
            revision_delete: None,
            raw_view: Default::default(),
            placing: None,
            placing_hover: None,
            dragging_end: None,
            selected_edit: None,
            hidden: HashSet::new(),
            sel_session: 0,
            matching: None,
            selected_match: None,
            dragging_match: None,
            match_hover: None,
            drawn_session: Vec::new(),
            img_shown: [None, None],
            img_want: [None, None],
            img_busy: [false; 2],
            img_rx: None,
            img_tx: None,
            pixels_per_point: 1.0,
            preview_drawn: false,
            params_open: false,
            height_manual: None,
            height_clip: [None; 2],
            processing_settings_open: false,
            help_open: false,
            about_open: false,
            licenses_open: false,
            licenses_third_party: false,
            confirm_preset: None,
            confirm_run: None,
            params_msg: None,
            new_work: None,
            confirm_replace: None,
            confirm_remove: None,
            rename_session: None,
            confirm_exit: false,
            exit_confirmed: false,
            window_place: crate::window_place::WindowPlace::load(cc.storage),
            device: device_calib::store(),
            device_calib: None,
            smoke,
        };
        app.params_open = app.smoke.as_ref().is_some_and(|s| s.params);
        app.load_work_settings();
        app.reload_edits();
        app.refresh_steps();
        app.load_scene(&cc.egui_ctx);
        if let Some(log) = crate::panics::take_last_crash() {
            let log = log.display();
            app.s.show_log = true;
            app.notify(
                Level::Error,
                tr!(
                    "前回は内部エラー（panic）で終了しました。記録: {log}",
                    "The last run ended by an internal error (panic). Record: {log}"
                ),
            );
        }
        app
    }

    /// The run of the open work folder, its sessions as listed (maybe none).
    fn project(&self) -> Option<RunConfig> {
        if self.s.work.trim().is_empty() {
            return None;
        }
        let work = PathBuf::from(self.s.work.trim());
        let mut c = RunConfig::of_sessions(RunConfig::project_sessions(&work)?, work);
        c.params = self
            .s
            .params
            .clone()
            .unwrap_or_else(|| ProcessParams::preset(self.s.outdoor));
        c.rtk = self.s.rtk;
        c.ppk = self.s.ppk;
        c.color = true;
        c.laz = self.s.laz;
        c.images = self.s.images;
        c.refine = self.s.refine;
        if !c.sessions.is_empty() {
            c.device_calib = self.device.find(&c.calib());
        }
        Some(c)
    }

    /// The run of the open work folder, when it has a session.
    fn config(&self) -> Option<RunConfig> {
        self.project().filter(|c| !c.sessions.is_empty())
    }

    /// Marks the steps whose results are on disk.
    fn refresh_steps(&mut self) {
        let cfg = self.config();
        self.stale_why.clear();
        for s in Step::ALL {
            let why = cfg
                .as_ref()
                .filter(|c| c.done(s))
                .and_then(|c| c.staleness(s));
            let st = match &cfg {
                Some(_) if why.is_some() => StepState::Stale,
                Some(c) if c.done(s) => StepState::Done,
                _ => StepState::Pending,
            };
            if let Some(w) = why {
                self.stale_why.insert(s, w);
            }
            self.steps.insert(s, st);
        }
        self.steps_key = Some((self.s.params.clone(), self.s.rtk, self.s.ppk));
    }

    /// Recomputes the step states when the settings they depend on changed (not while running).
    fn refresh_steps_on_settings(&mut self, ctx: &egui::Context) {
        if self.job.is_none()
            && self.steps_key.as_ref() != Some(&(self.s.params.clone(), self.s.rtk, self.s.ppk))
        {
            let source_changed = self.steps_key.as_ref().is_some_and(|k| {
                k.2 != self.s.ppk
                    || k.0.as_ref().map(|p| &p.ppk) != self.s.params.as_ref().map(|p| &p.ppk)
            });
            self.refresh_steps();
            if source_changed {
                self.load_scene(ctx);
            }
        }
    }

    fn load_scene(&mut self, ctx: &egui::Context) {
        let Some(cfg) = self.config() else { return };
        // Only the preview of a finished step is shown: while a step runs, the one shown stays;
        // else the choice goes to the nearest that can be shown.
        if self.preview_missing(self.preview_choice()).is_some() {
            if self.job.is_some() {
                return;
            }
            if let Some(p) = self.nearest_preview() {
                self.set_preview(p);
            }
        }
        // Cancel before starting another build; no queue of full previews can accumulate.
        self.loading = None;
        let (view, stride) = (self.s.view, self.s.scan_stride);
        let trajectory = self.s.scan_trajectory;
        // while a step runs it may be writing the output cloud: its octree is not made then
        let convert = self.job.is_none();
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        std::thread::spawn(move || {
            let loaded = crate::panics::catch(|| {
                scene::load(&cfg, view, trajectory, stride, convert, &worker_cancel)
            });
            let _ = tx.send(loaded.map_err(anyhow::Error::from).and_then(|r| r));
            ctx.request_repaint();
        });
        self.loading = Some(Loading { rx, cancel });
    }

    fn start(&mut self, ctx: &egui::Context, first: Step) {
        self.start_job(ctx, Run { first, last: None });
    }

    /// The run asked for by a step button: from step `s` to the end, or only `s` (`only`); it
    /// starts at the first earlier step that is not done (missing, stale or failed) instead.
    fn plan(&self, s: Step, only: bool) -> Run {
        let first = Step::ALL
            .into_iter()
            .filter(|&e| e < s)
            .find(|e| self.steps.get(e) != Some(&StepState::Done))
            .unwrap_or(s);
        Run {
            first,
            last: only.then_some(s),
        }
    }

    /// Starts a run, after a confirmation when it would begin by redoing a finished step.
    fn request_run(&mut self, ctx: &egui::Context, run: Run) {
        if self.steps.get(&run.first) == Some(&StepState::Done) {
            self.confirm_run = Some(run);
        } else {
            self.start_job(ctx, run);
        }
    }

    fn confirm_run_modal(&mut self, ctx: &egui::Context) {
        let Some(run) = self.confirm_run else {
            return;
        };
        let what = match run.last {
            Some(l) if l == run.first => tr!(
                "「{}」をやり直します。",
                "\"{}\" will be redone.",
                step_label(run.first)
            ),
            Some(l) => tr!(
                "「{}」から「{}」までをやり直します。",
                "\"{}\" through \"{}\" will be redone.",
                step_label(run.first),
                step_label(l)
            ),
            None => tr!(
                "「{}」から最後までをやり直します。",
                "\"{}\" through the end will be redone.",
                step_label(run.first)
            ),
        };
        let mut close = false;
        let r = egui::Modal::new(egui::Id::new("confirm_run")).show(ctx, |ui| {
            ui.label(tr!(
                "{}  完了済みの工程です。",
                "{}  This step has already finished.",
                icon::WARNING
            ));
            ui.label(what);
            ui.label(tl(
                "今ある結果は上書きされます。",
                "The existing results will be overwritten.",
            ));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button(tl("やり直す", "Redo")).clicked() {
                    self.start_job(ctx, run);
                    close = true;
                }
                if ui.button(tl("キャンセル", "Cancel")).clicked() {
                    close = true;
                }
            });
        });
        if close || r.should_close() {
            self.confirm_run = None;
        }
    }

    fn start_job(&mut self, ctx: &egui::Context, run: Run) {
        let Run { first, last } = run;
        if self.device_calib_running() {
            self.notify(
                Level::Info,
                tl("機体の校正の実行中は処理を始められません（終わってから実行してください）", "Processing can't start while the device calibration is running (run it after it finishes)"),
            );
            return;
        }
        // the corrections the optimisation runs with stay as a revision
        if first <= Step::Opt && last.is_none_or(|l| l >= Step::Opt) {
            self.save_revision(true);
        }
        let Some(cfg) = self.config() else { return };
        let what = match last {
            Some(l) if l == first => tr!("{} だけ実行", "Run {} only", step_label(first)),
            Some(l) => tr!(
                "{} から {} まで実行",
                "Run {} through {}",
                step_label(first),
                step_label(l)
            ),
            None => tr!("{} から実行", "Run from {}", step_label(first)),
        };
        // the log kept in the work folder as well
        let log_file = match s20_pipeline::runlog::LogFile::create(&cfg.log_dir(), cfg.path_mask())
        {
            Ok(f) => {
                let preset = if self.s.outdoor {
                    tl("屋外", "Outdoor")
                } else {
                    tl("屋内", "Indoor")
                };
                for l in s20_pipeline::runlog::header(
                    &cfg,
                    &tr!(
                        "{what}（GUI、プリセット: {preset}）",
                        "{what} (GUI, preset: {preset})"
                    ),
                ) {
                    f.line(&l);
                }
                Some(Arc::new(f))
            }
            Err(e) => {
                self.log_line(tr!(
                    "ログファイルを作れませんでした: {e:#}",
                    "Could not create the log file: {e:#}"
                ));
                None
            }
        };
        let (tx, rx) = mpsc::channel();
        let (tp, tl) = (tx.clone(), tx.clone());
        let (cp, cl) = (ctx.clone(), ctx.clone());
        let t0 = Instant::now();
        let lf = log_file.clone();
        let job = Job::new(
            move |stage, done, total| {
                let _ = tp.send(Event::Progress(stage.to_string(), done, total));
                cp.request_repaint();
            },
            move |s| {
                let mut line = format!("[{:7.1}s] {s}", t0.elapsed().as_secs_f64());
                if let Some(f) = &lf {
                    line = f.line(&line);
                }
                let _ = tl.send(Event::Log(line));
                cl.request_repaint();
            },
        );
        let cancel = job.cancel.clone();
        let ctx2 = ctx.clone();
        std::thread::spawn(move || {
            let r = crate::panics::catch(|| match last {
                Some(l) => cfg.run_range(first, l, &job),
                None => cfg.run_from(first, &job),
            })
            .unwrap_or_else(|p| {
                Err(anyhow::anyhow!(tr!(
                    "処理中に内部エラー（panic）が起きました: {}",
                    "An internal error (panic) occurred during processing: {}",
                    p.detail
                )))
            });
            let _ = tx.send(Event::Done(r));
            ctx2.request_repaint();
        });
        for s in Step::ALL {
            if s >= first && last.is_none_or(|l| s <= l) {
                self.steps.insert(s, StepState::Pending);
            }
        }
        self.error = None;
        self.log.push(format!("---- {what}"));
        if let Some(f) = &log_file {
            self.log_line(tr!("ログファイル: {}", "Log file: {}", f.path.display()));
        }
        self.job = Some(Running {
            rx,
            cancel,
            started: Instant::now(),
            opt_progress: OptProgress::default(),
            log_file,
        });
    }

    fn poll(&mut self, ctx: &egui::Context) {
        let mut finished = None;
        let mut preview_ready = false;
        if let Some(job) = &mut self.job {
            while let Ok(e) = job.rx.try_recv() {
                match e {
                    Event::Progress(stage, done, total) => {
                        // the phases of the optimisation (with PPK) report under their own names
                        let step = if stage.starts_with("ppk ") || stage.starts_with("opt ") {
                            Some(Step::Opt)
                        } else {
                            step_of(&stage)
                        };
                        if let Some(s) = step {
                            // steps before the reporting one are finished
                            if self.current != Some(s) {
                                if let Some(prev) = self.current {
                                    self.steps.insert(prev, StepState::Done);
                                }
                                self.current = Some(s);
                            }
                            let f = if s == Step::Opt {
                                job.opt_progress.update(&stage, done, total)
                            } else if total > 0 {
                                done as f32 / total as f32
                            } else {
                                0.0
                            };
                            self.steps.insert(s, StepState::Running(f));
                        }
                    }
                    Event::Log(l) => {
                        // "== step" opens a step
                        if let Some(k) = l.split("== ").nth(1).and_then(step_of) {
                            if let Some(prev) = self.current {
                                self.steps.insert(prev, StepState::Done);
                                preview_ready |= matches!(prev, Step::Lio | Step::Opt);
                            }
                            self.current = Some(k);
                            self.steps.insert(k, StepState::Running(0.0));
                        }
                        self.log.push(l);
                    }
                    Event::Done(r) => finished = Some(r),
                }
            }
        }
        if preview_ready && finished.is_none() && self.loading.is_none() {
            self.load_scene(ctx);
        }
        if let Some(r) = finished {
            let secs = self
                .job
                .as_ref()
                .map_or(0.0, |j| j.started.elapsed().as_secs_f64());
            if let Some(f) = self.job.as_ref().and_then(|j| j.log_file.as_ref()) {
                f.line(&match &r {
                    Ok(()) => tr!("---- 完了（{secs:.0} 秒）", "---- Done ({secs:.0} s)"),
                    Err(e) if is_cancelled(e) => {
                        tl("---- 中断しました", "---- Cancelled").to_string()
                    }
                    Err(e) => tr!("---- エラー: {e:#}", "---- Error: {e:#}"),
                });
            }
            self.job = None;
            match r {
                Ok(()) => {
                    self.log
                        .push(tr!("---- 完了（{secs:.0} 秒）", "---- Done ({secs:.0} s)"));
                    self.notices.push(
                        Level::Info,
                        tr!(
                            "処理が終わりました（{secs:.0} 秒）",
                            "Processing finished ({secs:.0} s)"
                        ),
                    );
                    self.refresh_steps();
                }
                Err(e) if is_cancelled(&e) => {
                    self.log
                        .push(tl("---- 中断しました", "---- Cancelled").into());
                    self.notices.push(
                        Level::Info,
                        tl("処理を中断しました", "Processing cancelled").into(),
                    );
                    if let Some(c) = self.current {
                        self.steps.insert(c, StepState::Pending);
                    }
                }
                Err(e) => {
                    if let Some(c) = self.current {
                        self.steps.insert(c, StepState::Failed);
                    }
                    let e = self.path_mask().apply(&format!("{e:#}"));
                    self.s.show_log = true;
                    self.log.push(tr!("---- エラー: {e}", "---- Error: {e}"));
                    self.notices.push(
                        Level::Error,
                        tr!("処理が失敗しました: {e}", "Processing failed: {e}"),
                    );
                    self.error = Some(e);
                }
            }
            self.current = None;
            self.load_scene(ctx);
        }
        if let Some(rx) = &self.loading
            && let Ok(r) = rx.rx.try_recv()
        {
            self.loading = None;
            match r {
                Ok(s) => {
                    let first = self.scene.bounds.is_none();
                    self.replace_scene(s);
                    self.scene_revision += 1;
                    self.reload_edits();
                    self.preview_drawn = false;
                    if first || !self.framed {
                        self.frame_all();
                    }
                }
                Err(e) => {
                    self.s.show_log = true;
                    self.notify(
                        Level::Error,
                        tr!(
                            "プレビューを読めませんでした: {e:#}",
                            "Could not read the preview: {e:#}"
                        ),
                    );
                }
            }
        }
    }

    fn frame_all(&mut self) {
        let c = self.scene.corners();
        if !c.is_empty() {
            self.camera.frame(&c);
            if let Some(z) = self.smoke.as_ref().and_then(|s| s.zoom) {
                self.camera.distance *= z;
                self.camera.pitch = 0.35;
            }
            self.framed = true;
        }
    }

    /// Keep the local view at the same recording time when changing trajectory frames.
    /// The spacings of the octree nodes' points drawn this frame, for the adaptive size (see
    /// `scene::Octree::point_spacings`): kept while the nodes drawn under each stay the same.
    fn octree_spacings(&mut self, alloc: &[(usize, usize)]) {
        use std::hash::{Hash, Hasher};
        let Some(tree) = self.scene.octree.clone().filter(|_| self.adaptive()) else {
            self.point_spacings.clear();
            return;
        };
        let nodes = &self.scene.nodes;
        let mut drawn = vec![false; nodes.len()];
        for &(i, _) in alloc {
            drawn[i] = nodes[i].total == 0 || !nodes[i].chunks.is_empty();
        }
        let spacings: Vec<f64> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                n.chunks
                    .first()
                    .filter(|_| drawn[i])
                    .map_or(tree.spacing(i), |chunk| tree.adaptive_spacing(i, chunk))
            })
            .collect();
        // children come after their parents
        let mut ids: HashMap<usize, u64> = HashMap::new();
        for &(i, _) in alloc.iter().rev() {
            if !drawn[i] {
                continue;
            }
            let mut h = std::collections::hash_map::DefaultHasher::new();
            i.hash(&mut h);
            for c in tree.children[i].into_iter().flatten() {
                if let Some(id) = ids.get(&c) {
                    (c, id).hash(&mut h);
                }
            }
            ids.insert(i, h.finish());
        }
        self.point_spacings.retain(|i, _| ids.contains_key(i));
        for (&i, &id) in &ids {
            let Some(chunk) = nodes[i].chunks.first() else {
                continue;
            };
            if self.point_spacings.get(&i).is_some_and(|e| e.0 == id) {
                continue;
            }
            self.point_spacings
                .insert(i, (id, tree.point_spacings(i, chunk, &drawn, &spacings)));
        }
    }

    fn replace_scene(&mut self, scene: Scene) {
        self.point_spacings.clear();
        if self.scene.trajectory_source != scene.trajectory_source {
            let target = DVec3::from(self.camera.target);
            let time = self.marker.map(|m| m.0).or_else(|| {
                self.scene
                    .traj
                    .iter()
                    .filter(|(t, _)| self.session_visible(self.scene.session_at(*t)))
                    .min_by(|a, b| {
                        a.1.distance_squared(target)
                            .total_cmp(&b.1.distance_squared(target))
                    })
                    .map(|p| p.0)
            });
            if let Some(t) = time
                && let (Some((old_r, old_p)), Some((new_r, new_p))) =
                    (self.scene.pose_at(t), scene.pose_at(t))
            {
                let rotation = new_r * old_r.inverse();
                self.camera.target = (new_p + rotation * (target - old_p)).to_array();
                let direction = rotation * (self.camera.eye() - DVec3::from(self.camera.target));
                self.camera.yaw = direction.y.atan2(direction.x);
                self.camera.pitch = (direction.z / direction.length())
                    .clamp(-1.0, 1.0)
                    .asin()
                    .clamp(-crate::camera::MAX_PITCH, crate::camera::MAX_PITCH);
                self.flight = None;
            }
        }
        self.marker = self
            .marker
            .and_then(|(t, _)| scene.position_at(t).map(|p| (t, p)));
        self.scene = scene;
    }

    /// Opens a chosen or dropped folder: a work folder, or a raw folder in the new work folder
    /// dialog.
    fn open_folder(&mut self, ctx: &egui::Context, dir: PathBuf) {
        match RunConfig::folder_kind(&dir) {
            FolderKind::Raw => self.new_work = Some(NewWork::new(&self.s, Some(&dir))),
            FolderKind::Work => match RunConfig::open_work(&dir) {
                Ok(_) => self.switch_work(ctx, &dir),
                Err(e) => self.notify(
                    Level::Error,
                    tr!(
                        "作業フォルダーを開けませんでした: {e:#}",
                        "Could not open the work folder: {e:#}"
                    ),
                ),
            },
            FolderKind::Other => self.notify(
                Level::Warn,
                tr!(
                    "作業フォルダーでも生データフォルダー（all_*.bag がある）でもありません: {}",
                    "Neither a work folder nor a raw data folder (with all_*.bag): {}",
                    dir.display()
                ),
            ),
        }
    }

    /// Shows work folder `work` (its raw folders listed): its steps, corrections and preview.
    fn switch_work(&mut self, ctx: &egui::Context, work: &Path) {
        self.s.work = work.display().to_string();
        self.s.remember(work);
        self.load_work_settings();
        self.clear_view();
        self.reload_edits();
        self.error = None;
        self.refresh_steps();
        self.load_scene(ctx);
        self.log.push(tr!(
            "作業フォルダーを開きました: {}",
            "Opened the work folder: {}",
            work.file_name().unwrap_or_default().to_string_lossy()
        ));
    }

    /// Takes the processing settings of the open work folder. One without them keeps the current
    /// ones (those used last), which [`Self::save_work_settings`] then saves in it.
    fn load_work_settings(&mut self) {
        self.work_settings_saved = None;
        let work = PathBuf::from(self.s.work.trim());
        if self.s.work.trim().is_empty() {
            return;
        }
        match WorkSettings::load(&work) {
            Some(Ok(w)) => {
                self.s.set_work_settings(w.clone());
                self.work_settings_saved = Some(w);
            }
            Some(Err(e)) => {
                // the file is left as it is until the settings are changed
                self.work_settings_saved = Some(self.s.work_settings());
                self.notify(
                    Level::Error,
                    tr!(
                        "処理設定を読めませんでした（今の設定を使います）: {e:#}",
                        "Could not read the processing settings (using the current ones): {e:#}"
                    ),
                );
            }
            // a new work folder: the settings used last, with today's defaults
            None => {
                if let Some(p) = &mut self.s.params {
                    p.update_old_defaults();
                }
            }
        }
    }

    /// Saves the processing settings in the open work folder when they changed.
    fn save_work_settings(&mut self) {
        let now = self.s.work_settings();
        if self.work_settings_saved.as_ref() == Some(&now) {
            return;
        }
        let work = PathBuf::from(self.s.work.trim());
        if self.s.work.trim().is_empty() || !work.join("project.json").is_file() {
            return;
        }
        if let Err(e) = now.save(&work) {
            self.notify(
                Level::Error,
                tr!(
                    "処理設定を保存できませんでした: {e:#}",
                    "Could not save the processing settings: {e:#}"
                ),
            );
        }
        self.work_settings_saved = Some(now);
    }

    /// Forgets the preview and what was being marked on it.
    fn clear_view(&mut self) {
        self.loading = None;
        self.scene = Scene::default();
        self.scene_revision += 1;
        self.trajectory_cache = None;
        self.worst_cache = None;
        self.drawn.clear();
        self.drawn_points = 0;
        self.drawn_session.clear();
        self.framed = false;
        self.flight = None;
        self.marker = None;
        self.placing = None;
        self.placing_hover = None;
        self.dragging_end = None;
        self.selected_edit = None;
        self.matching = None;
        self.selected_match = None;
        self.dragging_match = None;
        self.match_hover = None;
        self.hidden.clear();
        self.sel_session = 0;
        self.height_manual = None;
        self.height_clip = [None; 2];
    }

    /// A folder dropped on the window is opened (a work folder) or starts a new one (a raw folder).
    fn dropped_folder(&mut self, ctx: &egui::Context) {
        let dir = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .find(|p| p.is_dir())
        });
        let Some(dir) = dir else { return };
        if self.job.is_some() {
            self.notify(
                Level::Warn,
                tl(
                    "実行中は作業フォルダーを切り替えられません",
                    "The work folder can't be switched while processing",
                ),
            );
        } else {
            self.open_folder(ctx, dir);
        }
    }

    /// The new work folder dialog: the raw folder, the name and where it goes.
    fn new_work_modal(&mut self, ctx: &egui::Context) {
        let Some(w) = &mut self.new_work else {
            return;
        };
        let (mut create, mut close) = (false, false);
        let r = egui::Modal::new(egui::Id::new("new_work")).show(ctx, |ui| {
            ui.set_max_width(560.0);
            ui.heading(tr!(
                "{}  新規作業フォルダー",
                "{}  New work folder",
                icon::FOLDER_PLUS
            ));
            ui.add_space(4.0);
            egui::Grid::new("new_work_grid")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label(tl("生データ", "Raw data"));
                    ui.horizontal(|ui| {
                        let edit = egui::TextEdit::singleline(&mut w.raw)
                            .hint_text(tl("S20 の記録のフォルダー", "Folder of the S20 recording"))
                            .desired_width(380.0);
                        if ui.add(edit).changed() {
                            w.follow_raw();
                        }
                        if ui
                            .button(icon::FOLDER_OPEN)
                            .on_hover_text(tl("フォルダーを選ぶ", "Choose a folder"))
                            .clicked()
                            && let Some(p) = rfd::FileDialog::new()
                                .set_title(tl("生データフォルダー", "Raw data folder"))
                                .pick_folder()
                        {
                            w.set_raw(&p);
                        }
                    });
                    ui.end_row();
                    ui.label(tl("名前", "Name"));
                    if ui
                        .add(egui::TextEdit::singleline(&mut w.name).desired_width(380.0))
                        .changed()
                    {
                        w.named = true;
                    }
                    ui.end_row();
                    ui.label(tl("保存場所", "Location"));
                    ui.horizontal(|ui| {
                        if ui
                            .add(egui::TextEdit::singleline(&mut w.location).desired_width(380.0))
                            .changed()
                        {
                            w.follow_raw();
                        }
                        if ui
                            .button(icon::FOLDER_OPEN)
                            .on_hover_text(tl("フォルダーを選ぶ", "Choose a folder"))
                            .clicked()
                        {
                            let mut d = rfd::FileDialog::new().set_title(tl(
                                "作業フォルダーを置く場所",
                                "Where to put the work folder",
                            ));
                            let at = Path::new(w.location.trim());
                            if at.is_dir() {
                                d = d.set_directory(at);
                            }
                            if let Some(p) = d.pick_folder() {
                                w.location = p.display().to_string();
                                w.follow_raw();
                            }
                        }
                    });
                    ui.end_row();
                });
            let path = w.path();
            let taken =
                path.exists() && std::fs::read_dir(&path).map_or(true, |mut d| d.next().is_some());
            let problem = if w.raw.trim().is_empty() {
                Some(tl(
                    "生データフォルダーを選んでください",
                    "Choose the raw data folder",
                ))
            } else if RunConfig::folder_kind(Path::new(w.raw.trim())) != FolderKind::Raw {
                Some(tl(
                    "生データフォルダーに all_*.bag がありません",
                    "The raw data folder has no all_*.bag",
                ))
            } else if w.name.trim().is_empty() || w.location.trim().is_empty() {
                Some(tl(
                    "名前と保存場所を入れてください",
                    "Enter a name and a location",
                ))
            } else if taken {
                Some(tl(
                    "同じ名前のフォルダーがあります",
                    "A folder with the same name exists",
                ))
            } else {
                None
            };
            ui.add_space(4.0);
            ui.weak(tr!(
                "作成されるフォルダー: {}",
                "Folder to create: {}",
                path.display()
            ));
            if let Some(p) = problem {
                ui.colored_label(egui::Color32::from_rgb(240, 160, 60), p);
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(problem.is_none(), egui::Button::new(tl("作成", "Create")))
                    .clicked()
                {
                    create = true;
                }
                if ui.button(tl("キャンセル", "Cancel")).clicked() {
                    close = true;
                }
            });
        });
        if create {
            let raw = PathBuf::from(w.raw.trim());
            let (work, location) = (w.path(), PathBuf::from(w.location.trim()));
            match RunConfig::create_work(&raw, &work) {
                Ok(()) => {
                    self.new_work = None;
                    self.s.last_location = Some(location);
                    self.switch_work(ctx, &work);
                }
                Err(e) => self.notify(
                    Level::Error,
                    tr!(
                        "作業フォルダーを作れませんでした: {e:#}",
                        "Could not create the work folder: {e:#}"
                    ),
                ),
            }
        } else if close || r.should_close() {
            self.new_work = None;
        }
    }

    /// The indoor / outdoor preset buttons; a preset replaces changed values only after confirmation.
    fn preset_row(&mut self, ui: &mut egui::Ui) {
        let modified = self.s.params_modified();
        ui.horizontal(|ui| {
            ui.label(tl("プリセット", "Preset"));
            for (outdoor, name, tip) in [
                (false, tl("屋内", "Indoor"), tl("部屋・廊下・階段など狭い場所向け：LiDAR の点を細かく使い、近くの点も使います", "For tight spaces such as rooms, corridors and stairs: uses the LiDAR points finely, including near ones")),
                (true, tl("屋外", "Outdoor"), tl("神社・公園・街路など広い場所向け：LiDAR の点を粗めに間引いて、遠くまで速く処理します", "For wide spaces such as shrines, parks and streets: thins the LiDAR points coarsely and processes far quickly")),
            ] {
                let selected = self.s.outdoor == outdoor && !modified;
                let r = ui
                    .selectable_label(selected, name)
                    .on_hover_text(tr!("{tip}
詳細パラメータをこのプリセットの値にします", "{tip}\nSets the detailed parameters to this preset's values"));
                if r.clicked() && !selected {
                    if modified {
                        self.confirm_preset = Some(outdoor);
                    } else {
                        self.apply_preset(outdoor);
                    }
                }
            }
            if modified {
                ui.colored_label(crate::params_view::CHANGED, tl("（変更あり）", "(modified)"))
                    .on_hover_text(tl("詳細パラメータがプリセットから変更されています", "The detailed parameters differ from the preset"));
            }
        });
    }

    fn apply_preset(&mut self, outdoor: bool) {
        self.s.outdoor = outdoor;
        self.s.params = Some(ProcessParams::preset(outdoor));
    }

    fn confirm_preset_modal(&mut self, ctx: &egui::Context) {
        let Some(outdoor) = self.confirm_preset else {
            return;
        };
        let name = if outdoor {
            tl("屋外", "Outdoor")
        } else {
            tl("屋内", "Indoor")
        };
        let mut close = false;
        let r = egui::Modal::new(egui::Id::new("confirm_preset")).show(ctx, |ui| {
            ui.label(tr!(
                "変更した詳細パラメータを破棄して、{name}プリセットにしますか？",
                "Discard the changed detailed parameters and use the {name} preset?"
            ));
            ui.label(tl(
                "（必要なら先に「JSON 書き出し」で保存してください）",
                "(If needed, save them first with \"Export JSON\")",
            ));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui
                    .button(tr!("{name}プリセットにする", "Use the {name} preset"))
                    .clicked()
                {
                    self.apply_preset(outdoor);
                    close = true;
                }
                if ui.button(tl("キャンセル", "Cancel")).clicked() {
                    close = true;
                }
            });
        });
        if close || r.should_close() {
            self.confirm_preset = None;
        }
    }

    /// The parameter editor, with the presets and the JSON export and import.
    fn params_window(&mut self, ui: &mut egui::Ui) {
        let idle = self.job.is_none();
        if !idle {
            ui.label(tl(
                "実行中は変更できません",
                "Can't be changed while processing",
            ));
        }
        ui.add_enabled_ui(idle, |ui| {
            self.preset_row(ui);
            ui.horizontal(|ui| {
                if ui
                    .button(tr!("{}  JSON 書き出し…", "{}  Export JSON…", icon::EXPORT))
                    .clicked()
                {
                    self.export_params();
                }
                if ui
                    .button(tr!("{}  JSON 読み込み…", "{}  Import JSON…", icon::DOWNLOAD_SIMPLE))
                    .clicked()
                {
                    self.import_params();
                }
            });
            if let Some((ok, m)) = &self.params_msg {
                let col = if *ok {
                    egui::Color32::from_rgb(110, 200, 120)
                } else {
                    egui::Color32::from_rgb(240, 110, 100)
                };
                ui.colored_label(col, m);
            }
            ui.label(
                egui::RichText::new(
                    tl("黄色の項目はプリセットから変更した値です（項目に重ねると説明とプリセット値）。\
                     変更は次に実行する工程から使われます。", "Yellow items are values changed from the preset (hover an item for its explanation and preset value). Changes are used from the next step that runs."),
                )
                .weak(),
            );
            ui.separator();
            let preset = ProcessParams::preset(self.s.outdoor);
            let expand = self.smoke.as_ref().is_some_and(|s| s.params);
            let ppk = self.s.ppk;
            crate::params_view::show(ui, self.s.params(), &preset, expand, ppk);
        });
    }

    /// File dialog for parameter files, starting in the work folder.
    fn params_dialog(&self) -> rfd::FileDialog {
        let d = rfd::FileDialog::new().add_filter("JSON", &["json"]);
        let work = PathBuf::from(self.s.work.trim());
        if !self.s.work.trim().is_empty() && work.is_dir() {
            d.set_directory(work)
        } else {
            d
        }
    }

    fn export_params(&mut self) {
        let Some(path) = self
            .params_dialog()
            .set_file_name("s20_params.json")
            .save_file()
        else {
            return;
        };
        let json = self.s.params().to_json();
        self.params_msg = Some(match std::fs::write(&path, json) {
            Ok(()) => (
                true,
                tr!("書き出しました: {}", "Exported: {}", path.display()),
            ),
            Err(e) => (
                false,
                tr!("書き出せませんでした: {e}", "Could not export: {e}"),
            ),
        });
    }

    fn import_params(&mut self) {
        let Some(path) = self.params_dialog().pick_file() else {
            return;
        };
        let read = std::fs::read_to_string(&path)
            .map_err(anyhow::Error::from)
            .and_then(|s| ProcessParams::from_json(&s));
        self.params_msg = Some(match read {
            Ok(p) => {
                self.s.params = Some(p);
                (
                    true,
                    tr!("読み込みました: {}", "Imported: {}", path.display()),
                )
            }
            Err(e) => (
                false,
                tr!("読み込めませんでした: {e:#}", "Could not import: {e:#}"),
            ),
        });
    }

    /// The sessions of the work folder (raw folders of the same place, the first the reference):
    /// shown or not, their steps and links; adding, taking out, pointing at another raw folder,
    /// and choosing the reference.
    fn sessions_ui(&mut self, ui: &mut egui::Ui) {
        let Some(cfg) = self.project() else { return };
        let ctx = ui.ctx().clone();
        let idle = self.job.is_none();
        let n = cfg.sessions.len();
        let mut change: Option<Vec<s20_pipeline::SessionCfg>> = None;
        let mut relink = None;
        let mut raw = None;
        match n {
            0 => {
                ui.colored_label(
                    egui::Color32::from_rgb(240, 160, 60),
                    tl(
                        "生データが登録されていません。セッションを追加してください",
                        "No raw data registered. Add a session",
                    ),
                );
            }
            1 => {}
            _ => {
                ui.label(
                    egui::RichText::new(tl(
                        "同じ場所を分けて撮ったもの。先頭が基準",
                        "Recordings of the same place taken separately. The first is the reference",
                    ))
                    .weak(),
                );
            }
        }
        // one row a session: its name, then a column a step (done or not), how it is placed
        // (several sessions) and its menu, the columns titled above
        // the English titles are wider
        let cell = egui::vec2(tl(36.0, 48.0), ui.spacing().interact_size.y);
        let menu_w = 28.0;
        let header = |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(menu_w);
                    let mut titles = vec![
                        ("LIO", tl("オドメトリ（LIO）", "Odometry (LIO)")),
                        (tl("抽出", "Import"), tl("データ取り込み", "Data import")),
                    ];
                    if n > 1 {
                        titles.insert(
                            0,
                            (
                                tl("配置", "Place"),
                                tl(
                                    "軌跡の最適化での置かれ方",
                                    "How it's placed in the trajectory optimization",
                                ),
                            ),
                        );
                    }
                    for (title, tip) in titles {
                        ui.allocate_ui_with_layout(
                            cell,
                            egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                            |ui| {
                                ui.label(egui::RichText::new(title).weak())
                                    .on_hover_text(tip)
                            },
                        );
                    }
                });
            });
        };
        if n > 0 {
            header(ui);
        }
        for (k, sc) in cfg.sessions.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.menu_button(icon::DOTS_THREE, |ui| {
                        if ui
                            .button(tr!(
                                "{}  生データを見る…",
                                "{}  View raw data…",
                                icon::DATABASE
                            ))
                            .on_hover_text(tl(
                                "bag のトピックとメッセージ、ファイル、GNSS（読み取りだけ）",
                                "The bag's topics and messages, files, GNSS (read only)",
                            ))
                            .clicked()
                        {
                            raw = Some(k);
                        }
                        if ui
                            .add_enabled(
                                idle,
                                egui::Button::new(tr!(
                                    "{}  参照先を変える…",
                                    "{}  Change the source…",
                                    icon::FOLDER_OPEN
                                )),
                            )
                            .on_hover_text(tr!("今: {}", "Now: {}", sc.raw.display()))
                            .clicked()
                        {
                            relink = Some(k);
                        }
                        if ui
                            .add_enabled(
                                idle && !self.processing(),
                                egui::Button::new(tr!(
                                    "{}  名前を変える…",
                                    "{}  Rename…",
                                    icon::PENCIL_SIMPLE
                                )),
                            )
                            .on_hover_text(tl(
                                "作業フォルダーでのセッションの名前（生データフォルダーの名前は変わりません）",
                                "The session's name in the work folder (the raw data folder keeps its name)",
                            ))
                            .clicked()
                        {
                            self.rename_session = Some((k, sc.name.clone()));
                        }
                        if k > 0
                            && ui
                                .add_enabled(
                                    idle,
                                    egui::Button::new(tr!(
                                        "{}  基準にする",
                                        "{}  Make reference",
                                        icon::ANCHOR
                                    )),
                                )
                                .on_hover_text(tl(
                                    "このセッションを先頭（基準）にする",
                                    "Make this session the first (reference)",
                                ))
                                .clicked()
                        {
                            let mut v = cfg.sessions.clone();
                            let s = v.remove(k);
                            v.insert(0, s);
                            change = Some(v);
                        }
                        ui.separator();
                        if ui
                            .add_enabled(
                                idle,
                                egui::Button::new(
                                    egui::RichText::new(tr!(
                                        "{}  セッションを外す…",
                                        "{}  Remove session…",
                                        icon::TRASH
                                    ))
                                    .color(egui::Color32::from_rgb(240, 110, 100)),
                                ),
                            )
                            .clicked()
                        {
                            self.confirm_remove = Some(k);
                        }
                    })
                    .response
                    .on_hover_text(tl("このセッションの操作", "Actions on this session"));
                    let centered =
                        egui::Layout::centered_and_justified(egui::Direction::LeftToRight);
                    if n > 1 {
                        let (sym, text, col, tip) = self.session_status(k, &sc.name);
                        ui.allocate_ui_with_layout(cell, centered, |ui| {
                            ui.colored_label(col, sym)
                                .on_hover_text(format!("{text}: {tip}"))
                        });
                    }
                    for step in [Step::Lio, Step::Extract] {
                        let (sym, col, tip) = if cfg.session_done(step, k) {
                            (
                                icon::CHECK,
                                egui::Color32::from_rgb(110, 200, 120),
                                tl("済み", "Done"),
                            )
                        } else {
                            (icon::MINUS, egui::Color32::GRAY, tl("まだ", "Not yet"))
                        };
                        ui.allocate_ui_with_layout(cell, centered, |ui| {
                            ui.colored_label(col, sym)
                                .on_hover_text(format!("{}: {tip}", step_label(step)))
                        });
                    }
                    // the name in the width left, wrapped: all of it shows (sessions are few)
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        if n > 1 {
                            let c = Self::session_color(k);
                            ui.label(
                                egui::RichText::new(Self::session_tag(k))
                                    .strong()
                                    .color(egui::Color32::from_rgb(c[0], c[1], c[2])),
                            )
                            .on_hover_text(tl(
                                "ビュー上のセッションの名前と色",
                                "The session's name and color in the view",
                            ));
                            let mut vis = !self.hidden.contains(&sc.name);
                            if ui
                                .checkbox(&mut vis, "")
                                .on_hover_text(tl("表示する", "Show"))
                                .changed()
                            {
                                if vis {
                                    self.hidden.remove(&sc.name);
                                } else {
                                    self.hidden.insert(sc.name.clone());
                                }
                            }
                        }
                        // in pieces ending at "_": a long name wraps between its date, time and
                        // place, a piece going whole to the next row unless wider than a row
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                            let row = ui.available_width();
                            for part in sc.name.split_inclusive('_') {
                                let w = egui::WidgetText::from(part)
                                    .into_galley(
                                        ui,
                                        Some(egui::TextWrapMode::Extend),
                                        f32::INFINITY,
                                        egui::TextStyle::Body,
                                    )
                                    .size()
                                    .x;
                                let label = egui::Label::new(part);
                                ui.add(if w <= row {
                                    label.extend()
                                } else {
                                    label.wrap()
                                });
                            }
                        })
                        .response
                        .on_hover_text(format!(
                            "{}\n{}",
                            sc.name,
                            sc.raw.display()
                        ));
                    });
                });
            });
        }
        if ui
            .add_enabled(
                idle,
                egui::Button::new(tr!("{}  セッションを追加…", "{}  Add session…", icon::PLUS)),
            )
            .on_hover_text(
                tl("同じ場所を別に撮った生データフォルダーを足します。抽出と LIO はセッションごとに、軌跡の最適化から後は全セッションで一緒に行います", "Adds a raw data folder recorded separately at the same place. Extraction and LIO run per session; from the trajectory optimization on, all sessions are processed together"),
            )
            .clicked()
            && let Some(raw) = rfd::FileDialog::new()
                .set_title(tl("生データフォルダー", "Raw data folder"))
                .pick_folder()
        {
            if RunConfig::folder_kind(&raw) != FolderKind::Raw {
                self.notify(Level::Warn, tr!(
                    "生データフォルダー（all_*.bag がある）ではありません: {}", "Not a raw data folder (with all_*.bag): {}",
                    raw.display()
                ));
            } else if cfg.sessions.iter().any(|s| s.raw == raw) {
                self.notify(Level::Warn, tl("そのフォルダーはもうセッションにあります", "That folder is already a session"));
            } else {
                let mut v = cfg.sessions.clone();
                v.push(cfg.new_session(raw));
                change = Some(v);
            }
        }
        if let Some(k) = raw {
            self.open_raw_view(k);
        }
        if let Some(k) = relink {
            let mut d = rfd::FileDialog::new().set_title(tr!(
                "「{}」の生データフォルダー",
                "Raw data folder of \"{}\"",
                cfg.sessions[k].name
            ));
            if let Some(p) = cfg.sessions[k].raw.parent().filter(|p| p.is_dir()) {
                d = d.set_directory(p);
            }
            if let Some(raw) = d.pick_folder() {
                self.relink_session(&ctx, &cfg, k, raw);
            }
        }
        if let Some(v) = change {
            self.set_sessions(&ctx, &cfg, v);
        }
    }

    /// Points session `k` at raw folder `raw`: the same recording moved there keeps its results;
    /// another one replaces the session, after confirmation.
    fn relink_session(&mut self, ctx: &egui::Context, cfg: &RunConfig, k: usize, raw: PathBuf) {
        if RunConfig::folder_kind(&raw) != FolderKind::Raw {
            self.notify(
                Level::Warn,
                tr!(
                    "生データフォルダー（all_*.bag がある）ではありません: {}",
                    "Not a raw data folder (with all_*.bag): {}",
                    raw.display()
                ),
            );
        } else if cfg
            .sessions
            .iter()
            .enumerate()
            .any(|(j, s)| j != k && s.raw == raw)
        {
            self.notify(
                Level::Warn,
                tl(
                    "そのフォルダーはもうセッションにあります",
                    "That folder is already a session",
                ),
            );
        } else if cfg.same_recording(k, &raw) {
            let mut c = cfg.clone();
            let name = c.sessions[k].name.clone();
            match c.move_raw(k, raw.clone()) {
                Ok(()) => self.notify(Level::Info, tr!(
                    "「{name}」の参照先を {} に変えました（同じ記録なので結果はそのまま使えます）", "Changed the source of \"{name}\" to {} (the same recording, so the results stay usable)",
                    raw.display()
                )),
                Err(e) => self
                    .notify(Level::Error, tr!("参照先を変えられませんでした: {e:#}", "Could not change the source: {e:#}")),
            }
            self.refresh_steps();
            self.load_scene(ctx);
        } else {
            self.confirm_replace = Some((k, raw));
        }
    }

    /// Confirms replacing a session by another recording ([`App::relink_session`]).
    fn confirm_replace_modal(&mut self, ctx: &egui::Context) {
        let Some((k, raw)) = self.confirm_replace.clone() else {
            return;
        };
        let Some(cfg) = self.project().filter(|c| k < c.sessions.len()) else {
            self.confirm_replace = None;
            return;
        };
        let (mut replace, mut close) = (false, false);
        let r = egui::Modal::new(egui::Id::new("confirm_replace")).show(ctx, |ui| {
            ui.set_max_width(520.0);
            ui.label(tr!(
                "{}  {} は「{}」とは別の記録です（bag が違います）。", "{}  {} is a different recording from \"{}\" (the bag differs).",
                icon::WARNING,
                raw.display(),
                cfg.sessions[k].name
            ));
            ui.label(tl("差し替えると、このセッションは抽出からやり直しになり、このセッションへの修正は消えます。今の結果は消さずに残します。", "Replacing it redoes this session from extraction, and the corrections to this session are lost. The current results are kept."));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button(tl("差し替える", "Replace")).clicked() {
                    replace = true;
                }
                if ui.button(tl("キャンセル", "Cancel")).clicked() {
                    close = true;
                }
            });
        });
        if replace {
            self.confirm_replace = None;
            let mut others = cfg.clone();
            others.sessions.remove(k);
            let mut v = cfg.sessions.clone();
            v[k] = others.new_session(raw);
            self.set_sessions(ctx, &cfg, v);
        } else if close || r.should_close() {
            self.confirm_replace = None;
        }
    }

    fn confirm_remove_modal(&mut self, ctx: &egui::Context) {
        let Some(k) = self.confirm_remove else {
            return;
        };
        let Some(cfg) = self.project().filter(|c| k < c.sessions.len()) else {
            self.confirm_remove = None;
            return;
        };
        let (mut remove, mut close) = (false, false);
        let r = egui::Modal::new(egui::Id::new("confirm_remove")).show(ctx, |ui| {
            ui.set_max_width(520.0);
            ui.label(tr!(
                "{}  セッション「{}」を外しますか？", "{}  Remove the session \"{}\"?",
                icon::WARNING,
                cfg.sessions[k].name
            ));
            ui.label(tl("このセッションへの修正は消え、修正の「元に戻す」の履歴も消えます。生データと結果は消しません（同じ生データを足し直せば結果を使えます）。", "The corrections to this session and their undo history are lost. The raw data and results are not deleted (adding the same raw data again reuses the results)."));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button(tl("外す", "Remove")).clicked() {
                    remove = true;
                }
                if ui.button(tl("キャンセル", "Cancel")).clicked() {
                    close = true;
                }
            });
        });
        if remove {
            self.confirm_remove = None;
            let mut v = cfg.sessions.clone();
            v.remove(k);
            self.set_sessions(ctx, &cfg, v);
        } else if close || r.should_close() {
            self.confirm_remove = None;
        }
    }

    /// Asks a session's new name, checked as it is typed.
    fn rename_session_modal(&mut self, ctx: &egui::Context) {
        let Some((k, mut text)) = self.rename_session.take() else {
            return;
        };
        let Some(cfg) = self.project().filter(|c| k < c.sessions.len()) else {
            return;
        };
        let old = cfg.sessions[k].name.clone();
        let problem = |t: &str| {
            let t = t.trim();
            (t != old).then(|| cfg.session_name_problem(k, t)).flatten()
        };
        let (mut rename, mut close) = (false, false);
        let r = egui::Modal::new(egui::Id::new("rename_session")).show(ctx, |ui| {
            ui.set_max_width(520.0);
            ui.label(tr!(
                "{}  セッション「{old}」の名前",
                "{}  Name of the session \"{old}\"",
                icon::PENCIL_SIMPLE
            ));
            let edit = ui.add(egui::TextEdit::singleline(&mut text).desired_width(f32::INFINITY));
            if ui.memory(|m| m.focused().is_none()) {
                edit.request_focus();
            }
            let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let why = problem(&text);
            if let Some(p) = why {
                ui.colored_label(
                    egui::Color32::from_rgb(240, 160, 60),
                    match p {
                        NameProblem::Invalid => tl(
                            "フォルダー名に使えない文字（\\ / : * ? \" < > | と @）や、空白・ピリオドで終わる名前、CON などの名前は使えません",
                            "Not usable: characters a folder name cannot have (\\ / : * ? \" < > |, and @), a name ending in a space or a period, or a name such as CON",
                        ),
                        NameProblem::Number => tl(
                            "数字だけの名前は使えません（セッションは番号でも指定できるため）",
                            "A name of digits only is not usable (sessions can also be given by number)",
                        ),
                        NameProblem::Taken => tl(
                            "ほかのセッションがその名前です",
                            "Another session has that name",
                        ),
                        NameProblem::FolderInUse => tl(
                            "作業フォルダーの sessions に、外したセッションの結果がその名前で残っています",
                            "The work folder's sessions folder keeps the results of a removed session by that name",
                        ),
                    },
                );
            }
            ui.label(
                egui::RichText::new(tl(
                    "作業フォルダーの中のセッションのフォルダーと、修正の中の名前を変えます。結果はそのまま使えます。生データフォルダーの名前は変わりません。",
                    "Renames the session's folder in the work folder and its name in the corrections. The results stay usable. The raw data folder keeps its name.",
                ))
                .weak(),
            );
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(why.is_none(), egui::Button::new(tl("変える", "Rename")))
                    .clicked()
                    || (enter && why.is_none())
                {
                    rename = true;
                }
                if ui.button(tl("キャンセル", "Cancel")).clicked() {
                    close = true;
                }
            });
        });
        if rename {
            self.apply_rename_session(ctx, &cfg, k, text.trim());
        } else if !(close || r.should_close()) {
            self.rename_session = Some((k, text));
        }
    }

    /// Names session `k` as `name` ([`RunConfig::rename_session`]), and so the view's state.
    fn apply_rename_session(&mut self, ctx: &egui::Context, cfg: &RunConfig, k: usize, name: &str) {
        let old = cfg.sessions[k].name.clone();
        if old == name {
            return;
        }
        // the device calibration reads the sessions' folders too
        if self.processing() {
            self.notify(
                Level::Warn,
                tl(
                    "処理中は名前を変えられません（機体の校正も含みます）",
                    "Cannot rename while processing (including the device calibration)",
                ),
            );
            return;
        }
        let mut c = cfg.clone();
        match c.rename_session(k, name) {
            Ok(()) => {
                self.edit_history.rename_session(&old, name);
                if self.hidden.remove(&old) {
                    self.hidden.insert(name.to_string());
                }
                self.log.push(tr!(
                    "セッション「{old}」を「{name}」にしました",
                    "Renamed the session \"{old}\" to \"{name}\""
                ));
            }
            Err(e) => self.notify(
                Level::Error,
                tr!(
                    "名前を変えられませんでした: {e:#}",
                    "Could not rename it: {e:#}"
                ),
            ),
        }
        self.reload_edits();
        self.refresh_steps();
        self.load_scene(ctx);
    }

    /// Whether something is being processed that closing the window would stop.
    fn processing(&self) -> bool {
        self.job.is_some() || self.device_calib_running()
    }

    /// Holds the window open when it is closed (the title bar's button, Alt+F4, the menu's Exit)
    /// while processing, until confirmed.
    fn hold_close(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.viewport().close_requested())
            && !self.exit_confirmed
            && self.smoke.is_none()
            && self.processing()
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.confirm_exit = true;
        }
    }

    fn confirm_exit_modal(&mut self, ctx: &egui::Context) {
        if !self.confirm_exit {
            return;
        }
        let (mut exit, mut close) = (false, false);
        let r = egui::Modal::new(egui::Id::new("confirm_exit")).show(ctx, |ui| {
            ui.set_max_width(520.0);
            ui.label(tr!(
                "{}  処理中です。終了しますか？",
                "{}  Processing is running. Exit?",
                icon::WARNING
            ));
            ui.label(tl(
                "終了すると実行中の処理は中断されます（終わった工程の結果は残ります）。",
                "Exiting cancels the processing (the results of the finished steps are kept).",
            ));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button(tl("中断して終了", "Cancel and exit")).clicked() {
                    exit = true;
                }
                if ui.button(tl("キャンセル", "Cancel")).clicked() {
                    close = true;
                }
            });
        });
        if exit {
            self.confirm_exit = false;
            self.exit_confirmed = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if close || r.should_close() {
            self.confirm_exit = false;
        }
    }

    /// How session `k` is placed: the reference, placed by the optimisation, linked, or not.
    fn session_status(
        &self,
        k: usize,
        name: &str,
    ) -> (&'static str, String, egui::Color32, String) {
        let ok = egui::Color32::from_rgb(110, 200, 120);
        if k == 0 {
            return (
                icon::ANCHOR,
                tl("基準", "Reference").into(),
                ok,
                tl(
                    "他のセッションはこのセッションに合わせて置きます",
                    "The other sessions are placed to fit this session",
                )
                .into(),
            );
        }
        if self.scene.source == Some(PreviewSource::Scans)
            && self.scene.trajectory_source == TrajectorySource::Lio
        {
            return (
                icon::ARROWS_LEFT_RIGHT,
                tl("LIO（横並び）", "LIO (side by side)").into(),
                egui::Color32::LIGHT_GRAY,
                tl(
                    "各セッションのLIOを独立した座標系で横に並べています",
                    "Each session's LIO is laid out side by side in its own coordinates",
                )
                .into(),
            );
        }
        if self.scene.source == Some(PreviewSource::Scans)
            && self.scene.trajectory_source == TrajectorySource::Device
        {
            return (
                icon::ARROWS_LEFT_RIGHT,
                tl("機体（横並び）", "Device (side by side)").into(),
                egui::Color32::LIGHT_GRAY,
                tl("各セッションの機体の軌跡を独立した座標系で横に並べています", "Each session's device trajectory is laid out side by side in its own coordinates").into(),
            );
        }
        let links = self
            .edits
            .loops
            .iter()
            .filter(|(a, b)| a.session != b.session)
            .flat_map(|(a, b)| [&a.session, &b.session])
            .chain(
                self.edits
                    .matches
                    .iter()
                    .flat_map(|(a, b)| [&a.session, &b.session]),
            )
            .filter(|s| s.as_str() == name)
            .count();
        let placed = self
            .scene
            .sessions
            .iter()
            .any(|s| s.name == name && s.placed);
        let gnss = self.rtk_used();
        match (placed, links) {
            (true, _) => (
                icon::CHECK,
                tl("配置済み", "Placed").into(),
                ok,
                tl("軌跡の最適化で共通の座標に置かれています", "Placed in common coordinates by the trajectory optimization").into(),
            ),
            (false, 0) if !gnss => (
                icon::WARNING,
                tl("未接続", "Not connected").into(),
                egui::Color32::from_rgb(240, 160, 60),
                tl("GNSS がないときは、ループ（同じ場所を通った時刻）か点の対応で、配置済みのセッションにつないでください", "Without GNSS, connect it to a placed session with a loop (a time it passed the same place) or point matches").into(),
            ),
            (false, n) => (
                icon::LINK,
                tr!("つながり {n}", "Links {n}"),
                egui::Color32::LIGHT_GRAY,
                tl("軌跡の最適化でこのつながりから置きます（GNSS があれば GNSS で）", "The trajectory optimization places it from these links (with GNSS, from GNSS)").into(),
            ),
        }
    }

    /// Saves a new session list, keeping the corrections meaningful: a correction on the first
    /// session (stored without its name) is named, and those on a session taken out are dropped.
    fn set_sessions(
        &mut self,
        ctx: &egui::Context,
        cfg: &RunConfig,
        v: Vec<s20_pipeline::SessionCfg>,
    ) {
        if let Some(first) = cfg.sessions.first() {
            Self::name_first(&mut self.edits, &first.name);
        }
        let names: HashSet<&str> = v.iter().map(|s| s.name.as_str()).collect();
        let keep = |a: &str| a.is_empty() || names.contains(a);
        let count = |e: &Edits| {
            e.loops.len() + e.matches.len() + e.rtk_exclude.len() + e.heading_hold.len()
        };
        let before = count(&self.edits);
        self.edits
            .loops
            .retain(|(a, b)| keep(&a.session) && keep(&b.session));
        self.edits
            .matches
            .retain(|(a, b)| keep(&a.session) && keep(&b.session));
        self.edits
            .rtk_exclude
            .retain(|(a, b)| keep(&a.session) && keep(&b.session));
        self.edits
            .heading_hold
            .retain(|(a, b)| keep(&a.session) && keep(&b.session));
        let dropped = before - count(&self.edits);
        let c = RunConfig::of_sessions(v, cfg.work.clone());
        match c.save_sessions().and_then(|()| c.save_edits(&self.edits)) {
            Ok(()) => {
                // the corrections now name other sessions: no undo past the change
                self.edit_history.reset(self.edits.clone());
                if c.sessions.is_empty() {
                    self.clear_view();
                }
                self.log.push(tr!(
                    "セッション: {}{}",
                    "Sessions: {}{}",
                    c.sessions
                        .iter()
                        .map(|s| s.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                    if dropped > 0 {
                        tr!(
                            "（外したセッションの修正 {dropped} 件を消しました）",
                            " (deleted {dropped} corrections of the removed sessions)"
                        )
                    } else {
                        String::new()
                    }
                ));
            }
            Err(e) => self.notify(
                Level::Error,
                tr!(
                    "セッションを変えられませんでした: {e:#}",
                    "Could not change the sessions: {e:#}"
                ),
            ),
        }
        self.refresh_steps();
        self.load_scene(ctx);
    }

    /// Names the corrections stored on the first session without its name.
    fn name_first(e: &mut Edits, first: &str) {
        let fix = |s: &mut String| {
            if s.is_empty() {
                *s = first.to_string();
            }
        };
        for (a, b) in e
            .loops
            .iter_mut()
            .chain(e.rtk_exclude.iter_mut())
            .chain(e.heading_hold.iter_mut())
        {
            fix(&mut a.session);
            fix(&mut b.session);
        }
        for (a, b) in &mut e.matches {
            fix(&mut a.session);
            fix(&mut b.session);
        }
    }

    /// Writes `text` to the log and shows it over the view for a few seconds.
    fn notify(&mut self, level: Level, text: impl Into<String>) {
        let text = self.path_mask().apply(&text.into());
        self.log.push(text.clone());
        self.notices.push(level, text);
    }

    /// The paths of the log as in the log files: relative to the work folder, the raw folders
    /// by session, the rest of the user's folder as `~`.
    fn path_mask(&self) -> PathMask {
        let work = self.s.work.trim();
        match self.project() {
            Some(c) => c.path_mask(),
            None if !work.is_empty() => PathMask::home().with(Path::new(work), ""),
            None => PathMask::home(),
        }
    }

    /// Adds `s` to the log with its paths masked as in the log files.
    fn log_line(&mut self, s: impl Into<String>) {
        let s = self.path_mask().apply(&s.into());
        self.log.push(s);
    }

    fn log_panel(&mut self, ui: &mut egui::Ui) {
        self.log_view.show(ui, &self.log);
    }

    /// Whether the trajectory was optimised with RTK (it is then in the frame of the fixes).
    fn rtk_used(&self) -> bool {
        self.scene.rtk_used
    }

    fn multi(&self) -> bool {
        self.scene.sessions.len() > 1
    }

    /// The name session `k` is stored under in the corrections (none for one session: the first).
    fn session_key(&self, k: usize) -> String {
        if self.multi() {
            self.scene.sessions[k].name.clone()
        } else {
            String::new()
        }
    }

    /// The session a stored name refers to (empty: the first; a number: by index).
    fn session_index(&self, name: &str) -> Option<usize> {
        if name.is_empty() {
            return Some(0);
        }
        self.scene
            .sessions
            .iter()
            .position(|s| s.name == name)
            .or_else(|| name.parse().ok())
    }

    /// The start of the session of time `t` (the time shown as 0 on the slider).
    fn session_start(&self, t: f64) -> f64 {
        match self.scene.session_at(t) {
            Some(k) if self.multi() => self.scene.sessions[k].start,
            _ => self.scene.t_origin,
        }
    }

    /// Time `t` as stored: seconds from the first scan of its session.
    fn at_of(&self, t: f64) -> At {
        let k = self.scene.session_at(t).unwrap_or(0);
        At::on(&self.session_key(k), t - self.session_start(t))
    }

    /// A stored time on the common axis (negative: from the end of its session).
    fn abs_of(&self, a: &At) -> Option<f64> {
        let (ts, te) = (self.scene.traj.first()?.0, self.scene.traj.last()?.0);
        let k = self.session_index(&a.session)?;
        let (start, end) = match self.scene.sessions.get(k) {
            Some(s) if self.multi() => (s.start, s.end),
            _ if k == 0 => (self.scene.t_origin, te),
            _ => return None,
        };
        Some(if a.t < 0.0 {
            end + a.t
        } else {
            (start + a.t).max(ts)
        })
    }

    /// Label of a stored time: start, end or seconds, with the session when there are several.
    fn at_label(&self, a: &At) -> String {
        let t = Self::loop_label(a.t);
        match self.session_index(&a.session) {
            Some(k) if self.multi() => format!("{} {t}", self.scene.sessions[k].name),
            _ => t,
        }
    }

    /// Like `at_label`, with the short session name (for the view).
    fn at_short_label(&self, a: &At) -> String {
        let t = Self::loop_label(a.t);
        match self.session_index(&a.session) {
            Some(k) if self.multi() => format!("{} {t}", Self::session_tag(k)),
            _ => t,
        }
    }

    /// `at_short_label` after `prefix`, laid out for the view: the short session name in the
    /// session's colour, the rest in `color`.
    fn at_short_galley(
        &self,
        painter: &egui::Painter,
        prefix: &str,
        a: &At,
        color: egui::Color32,
    ) -> std::sync::Arc<egui::Galley> {
        let font = egui::FontId::proportional(13.0);
        let mut job = egui::text::LayoutJob::default();
        let mut push = |s: &str, c: egui::Color32| {
            job.append(s, 0.0, egui::TextFormat::simple(font.clone(), c));
        };
        push(prefix, color);
        if let Some(k) = self.session_index(&a.session).filter(|_| self.multi()) {
            let [r, g, b] = Self::session_color(k);
            push(&Self::session_tag(k), egui::Color32::from_rgb(r, g, b));
            push(" ", color);
        }
        push(&Self::loop_label(a.t), color);
        painter.layout_job(job)
    }

    /// The short name of session `k` on the view and beside its full name: S1, S2, …
    fn session_tag(k: usize) -> String {
        format!("S{}", k + 1)
    }

    /// The colour of session `k`.
    fn session_color(k: usize) -> [u8; 3] {
        const C: [[u8; 3]; 6] = [
            [255, 196, 60],
            [80, 200, 255],
            [255, 110, 200],
            [140, 230, 110],
            [190, 150, 255],
            [255, 140, 90],
        ];
        C[k % C.len()]
    }

    fn session_visible(&self, k: Option<usize>) -> bool {
        k.is_none_or(|k| {
            self.scene
                .sessions
                .get(k)
                .is_none_or(|s| !self.hidden.contains(&s.name))
        })
    }

    /// In an interval of `kind`, or in the one being placed.
    fn in_interval(&self, kind: EditKind, t: f64) -> bool {
        self.edit_list(kind).iter().any(|(a, b)| {
            self.abs_of(a)
                .zip(self.abs_of(b))
                .is_some_and(|(a, b)| t >= a && t < b)
        }) || self
            .placing_interval(kind)
            .is_some_and(|(a, b)| t >= a && t < b)
    }

    /// Whether the trajectory at `t` takes the colour of an interval of `kind`: those placed
    /// unless hidden in the display settings, and the one being placed.
    fn span_colored(&self, kind: EditKind, t: f64) -> bool {
        if self.s.show_edit_spans {
            self.in_interval(kind, t)
        } else {
            self.placing_interval(kind)
                .is_some_and(|(a, b)| t >= a && t < b)
        }
    }

    fn excluded(&self, t: f64) -> bool {
        self.in_interval(EditKind::Exclude, t)
    }

    /// Colour of a trajectory sample at time `t`: by time, or by the selected quality metric.
    fn traj_color(&self, t: f64, t0: f64, t1: f64) -> [u8; 3] {
        if self.span_colored(EditKind::Hold, t) {
            return HELD;
        }
        if self.rtk_used() && self.span_colored(EditKind::Exclude, t) {
            return EXCLUDED;
        }
        match (self.s.traj_metric, &self.scene.quality) {
            (m, Some(qy)) if m > 0 => {
                let met = &METRICS[m - 1];
                q::color(q::badness(met, self.s.metric_bad[m - 1], qy.at(m - 1, t)))
            }
            _ if self.multi() => Self::session_color(self.scene.session_at(t).unwrap_or(0)),
            _ => turbo(((t - t0) / (t1 - t0)) as f32),
        }
    }

    /// Reuses the trajectory until its data, display settings or edit overlays change.
    fn traj_lines(&mut self, pixels: f32) -> Arc<[Line]> {
        let key = TrajectoryKey {
            scene_revision: self.scene_revision,
            show_traj: self.s.show_traj,
            show_rtk: self.s.show_rtk,
            show_edit_spans: self.s.show_edit_spans,
            metric: self.s.traj_metric,
            bad: self
                .s
                .traj_metric
                .checked_sub(1)
                .map_or(0, |m| self.s.metric_bad[m].to_bits()),
            pixels: pixels.to_bits(),
            edits: self.edits.clone(),
            placing: self.placing,
            placing_hover: self.placing_hover,
            selected_edit: self.selected_edit,
            marker: self.marker,
            hidden: {
                let mut h: Vec<String> = self.hidden.iter().cloned().collect();
                h.sort();
                h
            },
        };
        if let Some((old, lines)) = &self.trajectory_cache
            && *old == key
        {
            return lines.clone();
        }
        let lines: Arc<[Line]> = self.build_traj_lines(pixels).into();
        self.trajectory_cache = Some((key, lines.clone()));
        lines
    }

    /// Trajectory as coloured segments, and a cross at the picked time.
    fn build_traj_lines(&self, pixels: f32) -> Vec<Line> {
        let t = &self.scene.traj;
        if t.len() < 2 || !self.s.show_traj {
            return Vec::new();
        }
        let step = (t.len() / 20_000).max(1);
        let (t0, t1) = (t[0].0, t.last().unwrap().0.max(t[0].0 + 1e-9));
        let mut out = Vec::new();
        // RTK fixes: white used, orange rejected by the validity check, purple excluded
        if self.rtk_used() && self.s.show_rtk {
            let step = (self.scene.rtk.len() / 2500).max(1);
            for (t, p, ok) in self.scene.rtk.iter().step_by(step) {
                let c = if self.excluded(*t) {
                    EXCLUDED
                } else if *ok {
                    [235, 235, 235]
                } else {
                    [255, 150, 40]
                };
                out.push(Line {
                    a: *p - DVec3::Z * 0.15,
                    b: *p + DVec3::Z * 0.15,
                    color: [c[0], c[1], c[2], 255],
                    width: 1.5 * pixels,
                });
            }
        }
        let selected = self
            .selected_edit
            .filter(|(k, _)| k.interval())
            .and_then(|(k, i)| self.edit_list(k).get(i))
            .and_then(|(a, b)| self.abs_of(a).zip(self.abs_of(b)));
        let mut i = 0;
        while i + step < t.len() {
            let (a, b) = (t[i], t[i + step]);
            let c = self.traj_color(a.0, t0, t1);
            let (sa, sb) = (self.scene.session_at(a.0), self.scene.session_at(b.0));
            if (b.1 - a.1).length() < 5.0 && sa == sb && self.session_visible(sa) {
                let wide = selected.is_some_and(|(x, y)| a.0 >= x && a.0 < y);
                out.push(Line {
                    a: a.1,
                    b: b.1,
                    color: [c[0], c[1], c[2], 255],
                    width: if wide { 6.0 } else { 3.0 } * pixels,
                });
            }
            i += step;
        }
        for (a, b) in &self.edits.loops {
            // not to or from a hidden session
            let shown = |x: &At| {
                self.abs_of(x)
                    .filter(|&t| self.session_visible(self.scene.session_at(t)))
                    .and_then(|t| self.scene.position_at(t))
            };
            if let (Some(pa), Some(pb)) = (shown(a), shown(b)) {
                let lift = DVec3::Z * 0.3;
                for (x, y) in [(pa, pa + lift), (pa + lift, pb + lift), (pb + lift, pb)] {
                    out.push(Line {
                        a: x,
                        b: y,
                        color: [LOOP[0], LOOP[1], LOOP[2], 255],
                        width: 3.0 * pixels,
                    });
                }
            }
        }
        // matched points: the line between the two (their markers are drawn on the view), not to
        // or from a hidden session
        for (a, b) in &self.edits.matches {
            let shown = |p: &PointAt| {
                self.session_index(&p.session)
                    .filter(|&k| self.session_visible(Some(k)))
                    .and_then(|_| self.match_world(p))
            };
            if let (Some(pa), Some(pb)) = (shown(a), shown(b)) {
                out.push(Line {
                    a: pa,
                    b: pb,
                    color: [MATCH[0], MATCH[1], MATCH[2], 255],
                    width: 1.5 * pixels,
                });
            }
        }
        if let Some((_, p)) = self.marker {
            for d in [DVec3::X, DVec3::Y, DVec3::Z] {
                out.push(Line {
                    a: p - d * 0.6,
                    b: p + d * 0.6,
                    color: [255, 255, 255, 255],
                    width: 3.0 * pixels,
                });
            }
        }
        out
    }

    /// A matched point where it is shown (the pose of its session at its time).
    fn match_world(&self, p: &PointAt) -> Option<DVec3> {
        let t = self.abs_of(&At::on(&p.session, p.t?))?;
        let (q, pos) = self.scene.pose_at(t)?;
        Some(q * DVec3::from(p.p) + pos)
    }

    fn worst_places(&mut self, m: usize, bad: f64) -> WorstPlaces {
        let key = (self.scene_revision, m, bad.to_bits());
        if let Some((old, places)) = &self.worst_cache
            && *old == key
        {
            return places.clone();
        }
        let places: WorstPlaces = self
            .scene
            .quality
            .as_ref()
            .map_or_else(Vec::new, |qy| q::worst(qy, m, bad, 5.0, 8))
            .into();
        self.worst_cache = Some((key, places.clone()));
        places
    }

    /// Trajectory colour choice, legend and the worst places of the selected metric.
    fn quality_ui(&mut self, ui: &mut egui::Ui) {
        let has_q = self.scene.quality.is_some();
        let label = |m: usize| {
            if m == 0 {
                tl("時刻", "Time")
            } else {
                METRICS[m - 1].label()
            }
        };
        ui.horizontal(|ui| {
            ui.label(tl("色分け:", "Color by:"));
            ui.add_enabled_ui(has_q, |ui| {
                egui::ComboBox::from_id_salt("traj_metric")
                    .selected_text(label(if has_q { self.s.traj_metric } else { 0 }))
                    .show_ui(ui, |ui| {
                        for m in 0..=METRICS.len() {
                            let r = ui.selectable_value(&mut self.s.traj_metric, m, label(m));
                            if m > 0 {
                                r.on_hover_text(METRICS[m - 1].help());
                            }
                        }
                    });
            })
            .response
            .on_disabled_hover_text(tl(
                "軌跡の最適化の結果（品質ファイル）があると選べる",
                "Available with the trajectory optimization's results (quality file)",
            ));
        });
        let m = self.s.traj_metric;
        if m == 0 && has_q {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(
                        tl("時刻で色分けしています。面のずれや IMU との不整合などの指標を選ぶと、\
                         問題のありそうな区間を一覧にします", "Colored by time. Choosing a metric such as plane offsets or disagreement with the IMU lists the sections that look problematic"),
                    )
                    .weak(),
                )
                .wrap(),
            );
        }
        if m == 0 || !has_q {
            return;
        }
        let met = &METRICS[m - 1];
        let bad = &mut self.s.metric_bad[m - 1];
        ui.horizontal(|ui| {
            let g = q::color(0.0);
            let y = q::color(0.5);
            let r = q::color(1.0);
            let c = |c: [u8; 3]| egui::Color32::from_rgb(c[0], c[1], c[2]);
            ui.colored_label(c(g), format!("● {}", met.good));
            ui.colored_label(c(y), "●");
            ui.colored_label(c(r), "●");
            let range = if met.bad > met.good {
                met.good * 1.2..=met.bad * 4.0
            } else {
                0.0..=met.good * 0.95
            };
            ui.add(
                egui::DragValue::new(bad)
                    .range(range)
                    .speed(met.bad.abs() * 0.01)
                    .max_decimals(2),
            );
            ui.label(met.unit());
        })
        .response
        .on_hover_text(tr!(
            "{}\n緑が {}、赤が右の値（変更できる）。灰色は値なし",
            "{}\nGreen is {}, red is the value on the right (editable). Gray: no value",
            met.help(),
            met.good
        ));
        let bad = *bad;
        let worst = self.worst_places(m - 1, bad);
        if worst.is_empty() {
            ui.label(tl(
                "問題のありそうな区間はありません",
                "No sections look problematic",
            ));
            return;
        }
        ui.label(tl(
            "問題のありそうな区間（クリックで移動）:",
            "Sections that look problematic (click to go):",
        ));
        let mut go = None;
        egui::Grid::new("worst")
            .num_columns(2)
            .spacing([12.0, 2.0])
            .show(ui, |ui| {
                for (t, v) in worst.iter() {
                    let c = q::color(q::badness(met, bad, *v));
                    let text = egui::RichText::new(self.at_label(&self.at_of(*t)))
                        .color(egui::Color32::from_rgb(c[0], c[1], c[2]));
                    if ui.button(text).clicked() {
                        go = Some(*t);
                    }
                    ui.label(format!("{v:.2} {}", met.unit()));
                    ui.end_row();
                }
            });
        if let Some(t) = go {
            self.go_to_time(t);
        }
    }

    /// The trajectory tab: its quality (to find where to look), then the corrections placed on it.
    fn trajectory_ui(&mut self, ui: &mut egui::Ui) {
        if self.scene.traj.is_empty() {
            return;
        }
        workspace::section(ui, tl("軌跡の色", "Trajectory colors"));
        self.quality_ui(ui);
        if self.scene.quality.is_none() {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(
                        tl("軌跡の最適化の品質情報があると、面のずれや IMU との不整合などで色分けし、\
                         問題のありそうな区間を一覧にします", "With the trajectory optimization's quality information, the trajectory is colored by plane offsets, disagreement with the IMU and so on, and the sections that look problematic are listed"),
                    )
                    .weak(),
                )
                .wrap(),
            );
        }
        workspace::section(ui, tl("修正", "Corrections"));
        self.edit_actions_ui(ui);
        let ctx = ui.ctx().clone();
        self.traj_edits_ui(ui, &ctx);
    }

    /// Time slider over the trajectory: marks the time, the camera may follow, images show it.
    fn timeline(&mut self, ui: &mut egui::Ui) {
        // the marked time's session, else the one chosen
        if let Some((t, _)) = self.marker
            && let Some(k) = self.scene.session_at(t)
        {
            self.sel_session = k;
        }
        let k = self
            .sel_session
            .min(self.scene.sessions.len().saturating_sub(1));
        let (mut ts, mut te) = (self.scene.traj[0].0, self.scene.traj.last().unwrap().0);
        let mut t0 = self.scene.t_origin;
        if self.multi() {
            let s = &self.scene.sessions[k];
            (ts, te, t0) = (s.start.max(ts), s.end.min(te), s.start);
        }
        let mut rel = self.marker.map_or(ts - t0, |m| m.0 - t0);
        let mut moved = false;
        ui.horizontal(|ui| {
            if self.multi() {
                let mut sel = k;
                egui::ComboBox::from_id_salt("timeline_session")
                    .selected_text(format!(
                        "{} {}",
                        Self::session_tag(k),
                        self.scene.sessions[k].name
                    ))
                    .width(160.0)
                    .show_ui(ui, |ui| {
                        for (i, s) in self.scene.sessions.iter().enumerate() {
                            ui.selectable_value(
                                &mut sel,
                                i,
                                format!("{} {}", Self::session_tag(i), s.name),
                            );
                        }
                    });
                if sel != k {
                    self.sel_session = sel;
                    let s = &self.scene.sessions[sel];
                    if let Some(p) = self.scene.position_at(s.start) {
                        self.marker = Some((s.start.max(self.scene.traj[0].0), p));
                    }
                    return;
                }
            }
            ui.label(tl("時刻", "Time")).on_hover_text(tl(
                "Ctrl+クリックで、軌跡上のその場所の時刻を選べる",
                "Ctrl+click picks the time of that place on the trajectory",
            ));
            for (label, step, tip) in [
                (icon::CARET_DOUBLE_LEFT, -1.0, tl("1 秒戻る", "Back 1 s")),
                (icon::CARET_LEFT, -0.1, tl("0.1 秒戻る", "Back 0.1 s")),
                (icon::CARET_RIGHT, 0.1, tl("0.1 秒進む", "Forward 0.1 s")),
                (icon::CARET_DOUBLE_RIGHT, 1.0, tl("1 秒進む", "Forward 1 s")),
            ] {
                if ui.button(label).on_hover_text(tip).clicked() {
                    rel += step;
                    moved = true;
                }
            }
            // left / right: 0.1 s, with Shift 1 s (when no widget has the keyboard)
            if ui.ctx().memory(|m| m.focused().is_none()) {
                let (l, r, sh) = ui.ctx().input(|i| {
                    (
                        i.key_pressed(egui::Key::ArrowLeft),
                        i.key_pressed(egui::Key::ArrowRight),
                        i.modifiers.shift,
                    )
                });
                let step = if sh { 1.0 } else { 0.1 };
                if l {
                    rel -= step;
                    moved = true;
                }
                if r {
                    rel += step;
                    moved = true;
                }
            }
            // the controls on the right, then the bar over the width left
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if self.marker.is_some()
                    && ui
                        .button(icon::X)
                        .on_hover_text(tl("時刻の印を消す", "Clear the time mark"))
                        .clicked()
                {
                    self.marker = None;
                }
                let mut images_visible = self.s.show_images
                    && self.s.inspector_open
                    && self.s.inspector_tab == InspectorTab::Images;
                if ui
                    .checkbox(&mut images_visible, tl("カメラ画像", "Camera images"))
                    .changed()
                {
                    if images_visible {
                        self.open_inspector(InspectorTab::Images);
                    } else {
                        self.s.show_images = false;
                        self.s.inspector_open = false;
                    }
                }
                ui.checkbox(&mut self.s.follow, tl("カメラ追従", "Follow camera"));
                ui.label(tr!("/ {:.1} 秒", "/ {:.1} s", te - t0));
                let before = rel;
                let r = ui.add(
                    egui::DragValue::new(&mut rel)
                        .range((ts - t0)..=(te - t0))
                        .speed(0.1)
                        .fixed_decimals(1),
                );
                // only a real change by the user moves the mark (and the camera with it): a value
                // rounded for display would report a change on every frame
                moved |= r.changed() && (rel - before).abs() > 1e-6;
                moved |= self.timeline_bar(ui, &mut rel, ts, te, t0);
            });
        });
        if moved {
            let t = (t0 + rel).clamp(ts, te);
            if let Some(p) = self.scene.position_at(t) {
                self.marker = Some((t, p));
                if self.s.follow {
                    self.flight = None;
                    self.camera.target = p.to_array();
                }
            }
        }
    }

    /// The time bar: the trajectory's colours along it (by time, quality metric or session, and
    /// the intervals marked), the worst places of the metric above it, the loop ends below it and
    /// the selected time. A click or drag selects a time (`rel`, from `t0`).
    fn timeline_bar(
        &mut self,
        ui: &mut egui::Ui,
        rel: &mut f64,
        ts: f64,
        te: f64,
        t0: f64,
    ) -> bool {
        let size = egui::vec2(ui.available_width().max(100.0), 24.0);
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        let span = (te - ts).max(1e-9);
        let x_of = |t: f64| rect.left() + ((t - ts) / span).clamp(0.0, 1.0) as f32 * rect.width();
        let t_of = |x: f32| ts + ((x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64 * span;
        let mut moved = false;
        if let Some(p) = response.interact_pointer_pos()
            && (response.clicked() || response.dragged())
        {
            *rel = t_of(p.x) - t0;
            moved = true;
        }
        let worst = match (self.s.traj_metric, &self.scene.quality) {
            (m, Some(_)) if m > 0 => {
                let bad = self.s.metric_bad[m - 1];
                Some((self.worst_places(m - 1, bad), &METRICS[m - 1], bad))
            }
            _ => None,
        };
        let painter = ui.painter_at(rect.expand(2.0));
        let visuals = ui.visuals();
        let rail =
            egui::Rect::from_x_y_ranges(rect.x_range(), rect.top() + 6.0..=rect.top() + 16.0);
        painter.rect_filled(rail, 2.0, visuals.extreme_bg_color);
        // the trajectory's colour, in runs of the same colour (2 px a sample)
        let (tr0, tr1) = (
            self.scene.traj[0].0,
            self.scene
                .traj
                .last()
                .unwrap()
                .0
                .max(self.scene.traj[0].0 + 1e-9),
        );
        let fill = |x0: f32, x1: f32, c: [u8; 3]| {
            let r = egui::Rect::from_x_y_ranges(x0..=x1, rail.y_range());
            painter.rect_filled(r, 0.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
        };
        let mut run: Option<(f32, [u8; 3])> = None;
        let mut x = rail.left();
        while x < rail.right() {
            let c = self.traj_color(t_of(x + 1.0), tr0, tr1);
            match run {
                Some((_, rc)) if rc == c => {}
                _ => {
                    if let Some((x0, rc)) = run {
                        fill(x0, x, rc);
                    }
                    run = Some((x, c));
                }
            }
            x += 2.0;
        }
        if let Some((x0, rc)) = run {
            fill(x0, rail.right(), rc);
        }
        // the worst places: triangles pointing down at the rail
        if let Some((places, met, bad)) = &worst {
            for (t, v) in places.iter() {
                let c = q::color(q::badness(met, *bad, *v));
                let x = x_of(*t);
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        egui::pos2(x - 4.0, rect.top()),
                        egui::pos2(x + 4.0, rect.top()),
                        egui::pos2(x, rail.top()),
                    ],
                    egui::Color32::from_rgb(c[0], c[1], c[2]),
                    egui::Stroke::new(1.0, egui::Color32::BLACK),
                ));
            }
        }
        // the loop ends: dots under the rail
        let cyan = EditKind::Loop.color();
        for (a, b) in &self.edits.loops {
            for t in [a, b].into_iter().filter_map(|x| self.abs_of(x)) {
                if t >= ts && t <= te {
                    painter.circle_filled(egui::pos2(x_of(t), rail.bottom() + 4.0), 2.5, cyan);
                }
            }
        }
        // the first end of the correction being placed
        if let Some(p) = self.placing
            && let Some(a) = p.first.filter(|t| (ts..=te).contains(t))
        {
            let x = x_of(a);
            painter.line_segment(
                [
                    egui::pos2(x, rail.top() - 3.0),
                    egui::pos2(x, rail.bottom() + 3.0),
                ],
                egui::Stroke::new(2.0, p.kind.color()),
            );
        }
        painter.rect_stroke(
            rail,
            2.0,
            visuals.widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Outside,
        );
        // the selected time
        if self.marker.is_some() {
            let x = x_of(t0 + *rel);
            let stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
            painter.line_segment(
                [
                    egui::pos2(x, rect.top() + 2.0),
                    egui::pos2(x, rect.bottom() - 2.0),
                ],
                stroke,
            );
            painter.circle(
                egui::pos2(x, rail.center().y),
                4.5,
                egui::Color32::WHITE,
                egui::Stroke::new(1.0, egui::Color32::BLACK),
            );
        }
        // under the pointer: its time, and the worst place there if any
        if let Some(p) = response.hover_pos() {
            painter.line_segment(
                [egui::pos2(p.x, rail.top()), egui::pos2(p.x, rail.bottom())],
                egui::Stroke::new(1.0, egui::Color32::from_white_alpha(140)),
            );
            let mut tip = tr!("{:.1} 秒", "{:.1} s", t_of(p.x) - t0);
            if let Some((places, met, _)) = &worst
                && let Some((_, v)) = places
                    .iter()
                    .find(|(w, _)| (x_of(*w) - p.x).abs() <= 5.0 && p.y < rail.top())
            {
                tip += &tr!(
                    "\n問題のありそうな区間: {v:.2} {}",
                    "\nProblematic section: {v:.2} {}",
                    met.unit()
                );
            }
            response.on_hover_text_at_pointer(tip);
        }
        moved
    }

    /// Requests the decode of the images nearest to the marked time.
    fn poll_images(&mut self, ctx: &egui::Context) {
        if self.img_rx.is_none() {
            let (tx, rx) = mpsc::channel();
            self.img_tx = Some(tx);
            self.img_rx = Some(rx);
        }
        while let Some(Ok((side, path, img))) = self.img_rx.as_ref().map(|r| r.try_recv()) {
            self.img_busy[side] = false;
            if let Some(img) = img {
                let tex = ctx.load_texture(format!("cam{side}"), img, egui::TextureOptions::LINEAR);
                self.img_shown[side] = Some((path, tex));
            }
        }
        let Some((t, _)) = self.marker else { return };
        for side in 0..2 {
            let Some(st) = crate::images::nearest(&self.scene.stamps, side == 0, t) else {
                continue;
            };
            self.img_want[side] = Some(st.file.clone());
            let shown = self.img_shown[side].as_ref().map(|s| &s.0);
            if self.img_busy[side] || shown == Some(&st.file) {
                continue;
            }
            self.img_busy[side] = true;
            let (tx, file, ctx) = (self.img_tx.clone().unwrap(), st.file.clone(), ctx.clone());
            std::thread::spawn(move || {
                // a panic must still answer: the side stays busy until it does
                let img = crate::panics::catch(|| crate::images::decode_small(&file, 4))
                    .ok()
                    .flatten();
                let _ = tx.send((side, file, img));
                ctx.request_repaint();
            });
        }
        if self.img_busy.iter().any(|b| *b) {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }

    fn images_panel(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_width();
        ui.set_min_width(full);
        if full < 480.0 {
            for side in 0..2 {
                self.camera_image_ui(ui, side, full);
                ui.add_space(8.0);
            }
        } else {
            let w = ((full - ui.spacing().item_spacing.x) / 2.0).floor();
            ui.horizontal_top(|ui| {
                for side in 0..2 {
                    ui.vertical(|ui| {
                        ui.set_width(w);
                        self.camera_image_ui(ui, side, w);
                    });
                }
            });
        }
    }

    fn camera_image_ui(&self, ui: &mut egui::Ui, side: usize, w: f32) {
        let name = if side == 0 {
            tl("左カメラ", "Left camera")
        } else {
            tl("右カメラ", "Right camera")
        };
        let when = self
            .scene
            .stamps
            .iter()
            .find(|s| Some(&s.file) == self.img_want[side].as_ref())
            .map(|s| s.t - self.session_start(s.t));
        ui.label(match when {
            Some(t) => tr!("{name}  {t:.1} 秒", "{name}  {t:.1} s"),
            None => name.to_string(),
        });
        match &self.img_shown[side] {
            Some((_, tex)) => {
                let sz = tex.size_vec2();
                ui.image((tex.id(), egui::vec2(w, w * sz.y / sz.x)));
            }
            None => {
                ui.spinner();
            }
        }
    }

    fn go_to_time(&mut self, t: f64) {
        let Some(p) = self.scene.position_at(t) else {
            return;
        };
        self.marker = Some((t, p));
        let dist = self.camera.distance.min(12.0);
        self.flight = Some((self.camera, p, Some(dist), Instant::now()));
    }

    fn viewport(&mut self, ui: &mut egui::Ui, rs: &eframe::egui_wgpu::RenderState) {
        let size = ui.available_size().max(egui::vec2(1., 1.));
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        let ctx = ui.ctx().clone();
        let (edit_busy, _) = self.traj_edit_input(&ctx, &response);
        let loop_busy = edit_busy | self.match_input(&ctx, &response);
        if !loop_busy
            && (response.dragged_by(egui::PointerButton::Primary)
                || response.dragged_by(egui::PointerButton::Middle))
        {
            self.camera.orbit(ctx.input(|i| i.pointer.delta()));
        }
        if response.dragged_by(egui::PointerButton::Secondary) {
            self.camera
                .pan(ctx.input(|i| i.pointer.delta()), rect.height());
        }
        if response.hovered() {
            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.camera.zoom(scroll);
            }
        }
        // Ctrl+click: the time of the trajectory point nearest to the pointer on screen
        if response.clicked() && ctx.input(|i| i.modifiers.command) {
            self.pick_time(&response);
        }
        // double click: orbit around the point under the pointer (as in Workbench)
        if (response.double_clicked() || response.double_clicked_by(egui::PointerButton::Middle))
            && let Some(p) = self.pick(&response)
        {
            self.flight = Some((self.camera, p, None, Instant::now()));
        }
        self.advance_flight(&ctx);
        self.camera.aspect = (size.x / size.y) as f64;
        let pixels = ctx.pixels_per_point();
        self.pixels_per_point = pixels;
        let lines = self.traj_lines(pixels);
        self.drawn.clear();
        self.drawn_session.clear();
        // each visible node draws a prefix of its shuffled points, by its size on screen
        let alloc = if self.s.show_points {
            self.scene.allocate(
                &self.camera,
                (size.y * pixels) as f64,
                self.s.preview_points,
                &|n| self.session_visible(n.session) && self.time_visible(n),
            )
        } else {
            Vec::new()
        };
        if let Some(stream) = &mut self.scene.streaming {
            let repaint = ctx.clone();
            let memory_budget = stream.view_budget(&alloc);
            stream.update(&mut self.scene.nodes, &alloc, memory_budget, move || {
                repaint.request_repaint()
            });
            if stream.pending() {
                ctx.request_repaint_after(std::time::Duration::from_millis(16));
            }
        }
        let coloring = self.coloring();
        self.octree_spacings(&alloc);
        let mut nodes: Vec<DrawNode> = Vec::new();
        self.drawn_points = 0;
        for (ni, count) in alloc {
            let count = count.min(self.scene.nodes[ni].chunks.iter().map(|c| c.len()).sum());
            let mut left = count;
            for c in &self.scene.nodes[ni].chunks {
                if left == 0 {
                    break;
                }
                let k = left.min(c.len());
                // by session: the session's colour, drawn shaded instead of mixed in
                let tint = match (coloring, self.scene.nodes[ni].session) {
                    (Coloring::Tint, Some(s)) => {
                        let c = Self::session_color(s).map(|v| v as f32 / 255.0);
                        [c[0], c[1], c[2], 0.0]
                    }
                    _ => [0.0; 4],
                };
                // an octree node's points by the detail drawn where they are
                let spacings = self.point_spacings.get(&ni).map(|s| &s.1);
                nodes.push(DrawNode {
                    id: self.drawn.len(),
                    points: c,
                    world: DMat4::IDENTITY,
                    tint,
                    marks: None,
                    spacings,
                    limit: k as u32,
                    spacing: if spacings.is_some() {
                        0.0
                    } else {
                        self.scene.nodes[ni].spacing(count, self.scene.nn_cap)
                    },
                });
                self.drawn.push(c.clone());
                self.drawn_session.push(self.scene.nodes[ni].session);
                self.drawn_points += k;
                left -= k;
            }
        }
        let mut renderer = self.renderer.take().expect("renderer");
        renderer.set_point_limit(self.s.preview_points * 2);
        let id = renderer.draw(
            rs,
            &self.camera,
            [(size.x * pixels) as u32, (size.y * pixels) as u32],
            &DrawOptions {
                point_size: self.s.point_size * pixels,
                // Potree's size 1 matches the default fixed size of 2 (as Workbench)
                adaptive: self.adaptive().then_some(self.s.point_size / 2.0),
                min_size: if self.scene.octree.is_some() {
                    2.0
                } else {
                    pixels
                },
                max_size: if self.scene.octree.is_some() {
                    50.0
                } else {
                    24.0
                },
                edl: Edl {
                    radius: 1.4 * pixels,
                    strength: if self.s.edl { 0.4 } else { 0.0 },
                },
                highlight_box: None,
                coloring,
                height_clip: self.height_clip(),
                nodes: &nodes,
                lines: &lines,
                lines_on_top: self.s.traj_on_top,
            },
        );
        self.preview_drawn = !renderer.pending();
        if renderer.pending() {
            ctx.request_repaint();
        }
        self.renderer = Some(renderer);
        let uv = egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(1., 1.));
        ui.painter().image(id, rect, uv, egui::Color32::WHITE);
        self.draw_view_marks(ui, rect);
        self.rtk_hover(ui, &response, rect);
        self.draw_axes(ui, rect);
        self.height_clip_ui(ui, rect);
        self.draw_pivot(ui, &response);
        if self.scene.bounds.is_none() && self.loading.is_none() {
            let hint = if self.config().is_none() {
                tl(
                    "作業フォルダーを開くか新規作成してください（フォルダーをドロップしても開けます）",
                    "Open or create a work folder (dropping a folder opens it too)",
                )
            } else {
                tl(
                    "「未完了・未反映の工程から実行」を押してください",
                    "Press \"Run from unfinished or unapplied steps\"",
                )
            };
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                hint,
                egui::FontId::proportional(15.0),
                egui::Color32::from_gray(170),
            );
        }
    }

    /// The shown point under the pointer, as last drawn (frontmost, else the nearest within a few pixels).
    fn pick(&self, response: &egui::Response) -> Option<DVec3> {
        self.pick_session(response).map(|p| p.0)
    }

    /// As [`App::pick`], with the session of the point (scans only).
    fn pick_session(&self, response: &egui::Response) -> Option<(DVec3, Option<usize>)> {
        self.pick_at(response.rect, response.interact_pointer_pos()?)
    }

    /// The shown point at `pos` in the viewport `rect`, with its session (as [`App::pick_session`]).
    fn pick_at(&self, rect: egui::Rect, pos: egui::Pos2) -> Option<(DVec3, Option<usize>)> {
        let click = [
            ((pos.x - rect.left()) / rect.width()) as f64,
            ((pos.y - rect.top()) / rect.height()) as f64,
        ];
        let reach = ((self.s.point_size / 2.0 + 6.0) * self.pixels_per_point).ceil() as u32;
        let (node, index) = self.renderer.as_ref()?.pick(click, reach)?;
        let n = self.drawn.get(node)?;
        (index < n.len()).then(|| {
            (
                n.position(index),
                self.drawn_session.get(node).copied().flatten(),
            )
        })
    }

    /// A shown point of session `k` as stored: in the body of the session's nearest trajectory
    /// sample, at its time.
    fn point_at(&self, k: usize, p: DVec3) -> Option<PointAt> {
        let i = (0..self.scene.traj.len())
            .filter(|&i| self.scene.session_at(self.scene.traj[i].0) == Some(k))
            .min_by(|&a, &b| {
                (self.scene.traj[a].1 - p)
                    .length_squared()
                    .total_cmp(&(self.scene.traj[b].1 - p).length_squared())
            })?;
        let (t, pos) = self.scene.traj[i];
        let q = self.scene.rot.get(i).copied().unwrap_or_default();
        let x = q.inverse() * (p - pos);
        let at = self.at_of(t);
        Some(PointAt {
            session: at.session,
            t: Some(at.t),
            p: x.to_array(),
        })
    }

    /// Marks the trajectory sample drawn nearest to the pointer (within 20 pixels).
    fn pick_time(&mut self, response: &egui::Response) {
        if let Some(pos) = response.interact_pointer_pos()
            && let Some(hit) = self.nearest_sample(response.rect, pos)
        {
            self.marker = Some(hit);
        }
    }

    /// Screen position of a world point in the viewport `rect`.
    fn to_screen(&self, rect: egui::Rect, p: DVec3) -> Option<egui::Pos2> {
        let uv = self.camera.project(p)?;
        Some(egui::pos2(
            rect.left() + uv[0] as f32 * rect.width(),
            rect.top() + uv[1] as f32 * rect.height(),
        ))
    }

    /// The trajectory sample drawn nearest to `pos` (within 20 pixels): (time, position).
    fn nearest_sample(&self, rect: egui::Rect, pos: egui::Pos2) -> Option<(f64, DVec3)> {
        let mut best: Option<(f32, f64, DVec3)> = None;
        for &(t, p) in &self.scene.traj {
            if !self.session_visible(self.scene.session_at(t)) {
                continue;
            }
            let Some(sp) = self.to_screen(rect, p) else {
                continue;
            };
            let d = sp.distance(pos);
            if d < 20.0 && best.is_none_or(|b| d < b.0) {
                best = Some((d, t, p));
            }
        }
        best.map(|(_, t, p)| (t, p))
    }

    /// Time of a loop end of the first session as stored (seconds from the first scan; negative:
    /// from the end).
    #[cfg(test)]
    fn loop_time(&self, x: f64) -> Option<f64> {
        self.abs_of(&At::new(x))
    }

    /// Label of a loop end as stored.
    fn loop_label(x: f64) -> String {
        if x < 0.0 {
            tl("終点", "end").into()
        } else if x <= 1.0 {
            tl("始点", "start").into()
        } else {
            tr!("{x:.1} 秒", "{x:.1} s")
        }
    }

    /// The GNSS fix under the pointer (of those drawn, within 6 pixels): ringed, with what it is and
    /// why it is used or not.
    fn rtk_hover(&self, ui: &egui::Ui, response: &egui::Response, rect: egui::Rect) {
        if !(self.s.show_traj && self.s.show_rtk && self.rtk_used())
            || self.placing.is_some()
            || self.dragging_end.is_some()
            || ui.ctx().input(|i| i.pointer.any_down())
        {
            return;
        }
        let Some(pos) = response.hover_pos() else {
            return;
        };
        // the fixes drawn: as `build_traj_lines` thins them
        let step = (self.scene.rtk.len() / 2500).max(1);
        let Some((i, sp)) = self
            .scene
            .rtk
            .iter()
            .enumerate()
            .step_by(step)
            .filter_map(|(i, (_, p, _))| Some((i, self.to_screen(rect, *p)?)))
            .map(|(i, sp)| (sp.distance(pos), i, sp))
            .filter(|h| h.0 < 6.0)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, i, sp)| (i, sp))
        else {
            return;
        };
        let (t, _, used) = self.scene.rtk[i];
        let color = if used {
            egui::Color32::WHITE
        } else {
            egui::Color32::from_rgb(255, 150, 40)
        };
        ui.painter_at(rect)
            .circle_stroke(sp, 6.0, egui::Stroke::new(2.0, color));
        let info = self.scene.rtk_info.get(i).cloned().unwrap_or_default();
        response.clone().on_hover_ui_at_pointer(|ui| {
            ui.label(egui::RichText::new(self.at_label(&self.at_of(t))).strong());
            ui.label(info);
        });
    }

    /// Markers and time labels on the corrections' ends, the session names, the points of a
    /// correction being placed, and what to click next. The labels on the view use the short session names
    /// (the full one shows under the pointer) and are moved off one another, a label with no free
    /// place being left out (its marker stays).
    fn draw_view_marks(&self, ui: &egui::Ui, rect: egui::Rect) {
        let painter = ui.painter().with_clip_rect(rect);
        let pointer = ui.ctx().input(|i| i.pointer.hover_pos());
        let mut labels = ViewLabels::default();
        // the markers first: the labels keep off them
        let handles = self.draw_edit_markers(&painter, rect, &mut labels);
        // what to click or press next, always shown at the top, one under another
        let mut row = 0.0;
        let mut top = || {
            row += 1.0;
            rect.center_top() + egui::vec2(0.0, 8.0 + (row - 1.0) * 26.0)
        };
        if let Some((text, color)) = self.edit_hint() {
            labels.add(&painter, top(), text, None, color, true, true);
        }
        if let Some(text) = self.match_hint() {
            let white = egui::Color32::from_rgb(MATCH[0], MATCH[1], MATCH[2]);
            labels.add(&painter, top(), text, None, white, true, true);
        }
        self.draw_match_marks(&painter, rect, &mut labels);
        // the names of the sessions at their first sample
        if self.multi() && self.s.show_traj {
            for (k, s) in self.scene.sessions.iter().enumerate() {
                if !self.session_visible(Some(k)) {
                    continue;
                }
                if let Some(p) = self.scene.position_at(s.start)
                    && let Some(sp) = self.to_screen(rect, p)
                {
                    let c = Self::session_color(k);
                    let unplaced = if s.placed {
                        ""
                    } else {
                        tl("（未配置）", " (not placed)")
                    };
                    labels.add(
                        &painter,
                        sp,
                        format!("{}{unplaced}", Self::session_tag(k)),
                        Some(format!("{} {}{unplaced}", Self::session_tag(k), s.name)),
                        egui::Color32::from_rgb(c[0], c[1], c[2]),
                        false,
                        false,
                    );
                }
            }
        }
        self.draw_edit_labels(&painter, handles, &mut labels);
        labels.finish(&painter, pointer);
    }

    fn advance_flight(&mut self, ctx: &egui::Context) {
        let Some((from, to, dist, start)) = self.flight else {
            return;
        };
        let s = (start.elapsed().as_secs_f64() / 0.25).min(1.0);
        let eased = s * s * (3.0 - 2.0 * s);
        let target = DVec3::from(from.target).lerp(to, eased);
        self.camera = match dist {
            // keep the view direction, move and zoom
            Some(d) => Camera {
                target: target.to_array(),
                distance: from.distance + (d - from.distance) * eased,
                ..from
            },
            None => from.looking_at(target),
        };
        if s >= 1.0 {
            self.flight = None;
        } else {
            ctx.request_repaint();
        }
    }

    /// Marks the orbit centre while it matters.
    fn draw_pivot(&self, ui: &egui::Ui, response: &egui::Response) {
        let orbiting = response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Middle);
        if !(orbiting || self.flight.is_some()) {
            return;
        }
        let c = response.rect.center();
        ui.painter()
            .circle_stroke(c, 6., egui::Stroke::new(3., egui::Color32::BLACK));
        ui.painter()
            .circle_stroke(c, 6., egui::Stroke::new(1.5, egui::Color32::WHITE));
    }

    /// A small world-axis triad in the corner.
    fn draw_axes(&self, ui: &egui::Ui, rect: egui::Rect) {
        let painter = ui.painter().with_clip_rect(rect);
        let origin = rect.right_bottom() - egui::vec2(50., 50.);
        let (right, up, _) = self.camera.basis();
        for (axis, name, color) in [
            (DVec3::X, "X", egui::Color32::from_rgb(230, 80, 80)),
            (DVec3::Y, "Y", egui::Color32::from_rgb(90, 200, 90)),
            (DVec3::Z, "Z", egui::Color32::from_rgb(90, 140, 240)),
        ] {
            let end = origin + egui::vec2(axis.dot(right) as f32, -axis.dot(up) as f32) * 30.;
            painter.line_segment([origin, end], egui::Stroke::new(2.0, color));
            painter.text(
                end,
                egui::Align2::CENTER_CENTER,
                name,
                egui::FontId::proportional(12.),
                color,
            );
        }
    }
}

/// Trajectory and RTK fixes inside an RTK exclusion.
const EXCLUDED: [u8; 3] = [190, 90, 230];
/// trajectory in a heading hold interval
const HELD: [u8; 3] = [40, 210, 230];
/// a loop's markers and the line between its ends
const LOOP: [u8; 3] = [255, 190, 50];

/// Turbo colour map (sRGB bytes), as the point shader uses for heights.
/// Labels drawn on the view, kept off one another.
#[derive(Default)]
pub(super) struct ViewLabels {
    /// the space taken by each label drawn, with its full text and colour if it has one
    placed: Vec<(egui::Rect, Option<(String, egui::Color32)>)>,
}

impl ViewLabels {
    /// Draws `text` above `at` (below if `below`), else at the first free place further up or
    /// down, else not at all; `force` draws it where asked. `full` shows under the pointer.
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        painter: &egui::Painter,
        at: egui::Pos2,
        text: String,
        full: Option<String>,
        color: egui::Color32,
        below: bool,
        force: bool,
    ) {
        let galley = painter.layout_no_wrap(text, egui::FontId::proportional(13.0), color);
        self.add_galley(painter, at, galley, full, color, below, force);
    }

    /// Like `add`, with the text laid out already (in several colours).
    #[allow(clippy::too_many_arguments)]
    fn add_galley(
        &mut self,
        painter: &egui::Painter,
        at: egui::Pos2,
        galley: std::sync::Arc<egui::Galley>,
        full: Option<String>,
        color: egui::Color32,
        below: bool,
        force: bool,
    ) {
        let size = galley.size();
        let spot = |below: bool, k: f32| {
            let dy = if below {
                12.0 + k * (size.y + 8.0)
            } else {
                -12.0 - size.y - k * (size.y + 8.0)
            };
            egui::Rect::from_min_size(at + egui::vec2(-size.x / 2.0, dy), size).expand(3.0)
        };
        let free = |r: &egui::Rect| !self.placed.iter().any(|(p, _)| p.intersects(*r));
        let spot = if force {
            Some(spot(below, 0.0))
        } else {
            (0..3)
                .flat_map(|k| [spot(below, k as f32), spot(!below, k as f32)])
                .find(free)
        };
        let Some(r) = spot else { return };
        painter.rect_filled(r, 3.0, egui::Color32::from_black_alpha(190));
        painter.galley(r.min + egui::vec2(3.0, 3.0), galley, color);
        self.placed.push((r, full.map(|f| (f, color))));
    }

    /// Keeps the labels off `r` (a marker).
    fn occupy(&mut self, r: egui::Rect) {
        self.placed.push((r, None));
    }

    /// Shows the full text of the label under the pointer, over the others.
    fn finish(self, painter: &egui::Painter, pointer: Option<egui::Pos2>) {
        let Some(p) = pointer else { return };
        let Some((r, Some((full, color)))) = self
            .placed
            .into_iter()
            .rev()
            .find(|(r, full)| full.is_some() && r.contains(p))
        else {
            return;
        };
        let galley = painter.layout_no_wrap(full, egui::FontId::proportional(13.0), color);
        let min = r.center_bottom() + egui::vec2(-galley.size().x / 2.0, 6.0);
        let back = egui::Rect::from_min_size(min, galley.size()).expand(3.0);
        painter.rect_filled(back, 3.0, egui::Color32::from_black_alpha(230));
        painter.galley(min, galley, color);
    }
}

/// The part of the Turbo map used: its ends (near black and dark red) hardly show on the dark
/// background (contrast 1.4), 0.12 (blue) and 0.88 (red) do (2.7 and 2.3). As in `points.wgsl`.
const TURBO_RANGE: [f32; 2] = [0.12, 0.88];

fn turbo(t: f32) -> [u8; 3] {
    let t = TURBO_RANGE[0] + t.clamp(0.0, 1.0) * (TURBO_RANGE[1] - TURBO_RANGE[0]);
    let r = 0.13572138
        + t * (4.615_392_7
            + t * (-42.660_324 + t * (132.131_09 + t * (-152.942_4 + t * 59.286_38))));
    let g = 0.09140261
        + t * (2.194_188_4
            + t * (4.842_966_6 + t * (-14.185_034 + t * (4.277_298_5 + t * 2.829_566))));
    let b = 0.106_673_3
        + t * (12.641_946
            + t * (-60.582_047 + t * (110.362_77 + t * (-89.903_11 + t * 27.348_25))));
    [r, g, b].map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        crate::panics::crash_to(self.s.work.trim(), || self.path_mask());
        self.window_place.track(&ctx);
        self.hold_close(&ctx);
        self.poll(&ctx);
        self.save_work_settings();
        self.refresh_steps_on_settings(&ctx);
        self.smoke_frame(&ctx);
        self.dropped_folder(&ctx);
        self.edit_shortcuts(&ctx);
        self.workspace_panels(ui);
        // the view's area, for the notices
        let view = ui.available_rect_before_wrap();
        let rs = frame.wgpu_render_state().expect("wgpu");
        if self.project().is_some() {
            egui::CentralPanel::no_frame().show(ui, |ui| self.viewport(ui, rs));
        } else {
            egui::CentralPanel::default().show(ui, |ui| self.welcome(ui));
        }
        if self.params_open {
            let mut open = true;
            egui::Window::new(tl("詳細パラメータ", "Detailed parameters"))
                .open(&mut open)
                .default_width(480.0)
                .default_height(640.0)
                .vscroll(true)
                .show(&ctx, |ui| self.params_window(ui));
            self.params_open = open;
        }
        self.workspace_windows(&ctx);
        self.revisions_window(&ctx);
        self.raw_window(&ctx);
        self.device_calib_window(&ctx);
        self.confirm_preset_modal(&ctx);
        self.confirm_run_modal(&ctx);
        self.new_work_modal(&ctx);
        self.confirm_replace_modal(&ctx);
        self.confirm_remove_modal(&ctx);
        self.rename_session_modal(&ctx);
        self.confirm_exit_modal(&ctx);
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let rect = ctx.content_rect();
            let p = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("drop"),
            ));
            p.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(170));
            p.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                tl(
                    "作業フォルダーを離すと開きます（生データフォルダーなら新規作成）",
                    "Drop the work folder to open it (a raw data folder creates a new one)",
                ),
                egui::FontId::proportional(18.0),
                egui::Color32::WHITE,
            );
        }
        self.notices.show(&ctx, view);
        if self.job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if self.smoke.is_some() {
            return;
        }
        eframe::set_value(storage, SETTINGS_KEY, &self.s);
        self.window_place.save(storage);
    }

    fn on_exit(&mut self) {
        if let Some(j) = &self.job {
            j.cancel.store(true, Ordering::Relaxed);
        }
        self.cancel_device_calib();
    }
}

impl App {
    fn smoke_frame(&mut self, ctx: &egui::Context) {
        let Some(sm) = &mut self.smoke else { return };
        if let Some(st) = sm.start
            && !sm.started
        {
            sm.started = true;
            self.start(ctx, st);
            return;
        }
        // once the preview is there: fly to the worst place of the metric
        if sm.goto_worst && !sm.went && self.loading.is_none() && self.scene.bounds.is_some() {
            sm.went = true;
            let m = self.s.traj_metric;
            if m > 0
                && let Some(qy) = &self.scene.quality
                && let Some(&(t, _)) = q::worst(qy, m - 1, self.s.metric_bad[m - 1], 5.0, 1).first()
            {
                self.go_to_time(t);
                if let Some(s) = &mut self.smoke {
                    s.t0 = Instant::now();
                }
            }
        }
        if let Some(rel) = self.smoke.as_ref().and_then(|s| s.time)
            && self.loading.is_none()
            && self.scene.bounds.is_some()
            && self.marker.is_none()
        {
            let t = self.scene.t_origin + rel;
            if let Some(p) = self.scene.position_at(t) {
                self.marker = Some((t, p));
            }
        }
        let sm = self.smoke.as_mut().unwrap();
        for event in ctx.input(|i| i.events.clone()) {
            if let egui::Event::Screenshot { image, .. } = event {
                let bytes: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                if let Err(e) = image::save_buffer(
                    &sm.screenshot,
                    &bytes,
                    image.size[0] as u32,
                    image.size[1] as u32,
                    image::ColorType::Rgba8,
                ) {
                    eprintln!("screenshot failed: {e}");
                }
                for l in &self.log {
                    eprintln!("{l}");
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
        }
        let ready = self.job.is_none()
            && self.loading.is_none()
            && self.preview_drawn
            && self.scene.streaming.as_ref().is_none_or(|s| !s.pending())
            && self.flight.is_none();
        if !sm.requested && ready && sm.t0.elapsed().as_secs_f64() > 2.0 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            sm.requested = true;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switching_trajectory_keeps_time_scale_and_local_view() {
        let mut app = app();
        app.marker = Some((115.0, DVec3::new(1.5, 0.0, 0.0)));
        app.camera.target = [3.0, 4.0, 2.0];
        let camera = app.camera;
        let rotation = glam::DQuat::from_rotation_z(0.8);
        let shift = DVec3::new(100.0, -30.0, 7.0);
        let transformed = Scene {
            trajectory_source: TrajectorySource::Lio,
            traj: app
                .scene
                .traj
                .iter()
                .map(|(t, p)| (*t, rotation * *p + shift))
                .collect(),
            rot: vec![rotation; app.scene.traj.len()],
            ..Default::default()
        };
        app.replace_scene(transformed);
        assert_eq!(app.marker.unwrap().0, 115.0);
        assert!(
            app.marker
                .unwrap()
                .1
                .distance(rotation * DVec3::new(1.5, 0.0, 0.0) + shift)
                < 1e-9
        );
        assert!(
            DVec3::from(app.camera.target).distance(rotation * DVec3::from(camera.target) + shift)
                < 1e-9
        );
        assert!(app.camera.eye().distance(rotation * camera.eye() + shift) < 1e-9);
        assert_eq!(app.camera.distance, camera.distance);
        assert_eq!(app.camera.ortho, camera.ortho);
        assert!(!app.rtk_used());
        assert!(app.scene.quality.is_none());
    }

    #[test]
    fn switching_without_marker_uses_view_and_reload_keeps_camera() {
        let mut app = app();
        let camera = app.camera;
        let shift = DVec3::new(100.0, 20.0, 0.0);
        let scene = Scene {
            trajectory_source: TrajectorySource::Lio,
            traj: app
                .scene
                .traj
                .iter()
                .map(|(t, p)| (*t, *p + shift))
                .collect(),
            ..Default::default()
        };
        app.replace_scene(scene);
        assert!(app.marker.is_none());
        assert!(DVec3::from(app.camera.target).distance(DVec3::from(camera.target) + shift) < 1e-9);
        let camera = app.camera;
        app.replace_scene(Scene {
            trajectory_source: TrajectorySource::Lio,
            ..Default::default()
        });
        assert_eq!(app.camera, camera);
    }

    #[test]
    fn old_settings_default_to_optimized_and_lio_selection_is_saved() {
        let mut settings: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(settings.scan_trajectory, TrajectorySource::Optimized);
        settings.scan_trajectory = TrajectorySource::Lio;
        let saved = serde_json::to_string(&settings).unwrap();
        let restored: Settings = serde_json::from_str(&saved).unwrap();
        assert_eq!(restored.scan_trajectory, TrajectorySource::Lio);
    }

    #[test]
    fn new_work_name_has_the_suffix_only_beside_the_raw_folder() {
        let settings: Settings = serde_json::from_str("{}").unwrap();
        let raw = Path::new("data").join("rec01");
        let mut w = NewWork::new(&settings, Some(&raw));
        assert_eq!(w.location, "data");
        assert_eq!(w.name, "rec01_s20");
        w.location = "out".into();
        w.follow_raw();
        assert_eq!(w.name, "rec01");
        w.location = "data".into();
        w.follow_raw();
        assert_eq!(w.name, "rec01_s20");
        w.named = true;
        w.name = "mine".into();
        w.location = "out".into();
        w.follow_raw();
        assert_eq!(w.name, "mine");
    }

    #[test]
    fn legacy_memory_limit_does_not_cap_the_saved_draw_budget() {
        let settings: Settings =
            serde_json::from_str(r#"{"preview_points":10000000,"loaded_points":500000}"#).unwrap();
        assert_eq!(settings.preview_points, 10_000_000);
        let saved = serde_json::to_value(&settings).unwrap();
        assert!(saved.get("loaded_points").is_none());
    }

    #[test]
    fn the_adaptive_size_is_kept_apart_for_the_scans_and_off_by_default() {
        // settings saved before the scans had their own keep the output cloud's choice
        let settings: Settings = serde_json::from_str(r#"{"adaptive":true}"#).unwrap();
        assert!(settings.adaptive && !settings.adaptive_scans);
        let mut a = app();
        a.s.adaptive = true;
        a.scene.source = Some(PreviewSource::Scans);
        assert!(!a.adaptive());
        a.s.adaptive_scans = true;
        a.s.adaptive = false;
        assert!(a.adaptive());
        a.scene.source = Some(PreviewSource::Colored);
        assert!(!a.adaptive());
    }

    pub(super) fn app() -> App {
        App {
            s: Settings::default(),
            job: None,
            steps: HashMap::new(),
            stale_why: HashMap::new(),
            steps_key: None,
            work_settings_saved: None,
            current: None,
            log: Vec::new(),
            log_view: Default::default(),
            notices: Default::default(),
            error: None,
            scene: Scene {
                traj: (0..120)
                    .map(|i| (100.0 + i as f64, DVec3::new(i as f64 * 0.1, 0.0, 0.0)))
                    .collect(),
                quality: Some(q::Quality {
                    t: (0..120).map(|i| 100.0 + i as f64).collect(),
                    v: (0..120)
                        .map(|i| [10.0 + i as f64, 0.3, 2.0, 30.0, 30.0, 2.0, 4.0])
                        .collect(),
                }),
                rtk_used: true,
                t_origin: 100.0,
                rtk: vec![(110.0, DVec3::X, true)],
                ..Default::default()
            },
            scene_revision: 0,
            trajectory_cache: None,
            worst_cache: None,
            loading: None,
            point_spacings: HashMap::new(),
            drawn: Vec::new(),
            drawn_points: 0,
            camera: Camera::default(),
            renderer: None,
            framed: false,
            flight: None,
            marker: None,
            edits: Edits::default(),
            edit_history: Default::default(),
            revisions: Vec::new(),
            revisions_open: false,
            revision_delete: None,
            raw_view: Default::default(),
            placing: None,
            placing_hover: None,
            dragging_end: None,
            selected_edit: None,
            hidden: HashSet::new(),
            sel_session: 0,
            matching: None,
            selected_match: None,
            dragging_match: None,
            match_hover: None,
            drawn_session: Vec::new(),
            img_shown: [None, None],
            img_want: [None, None],
            img_busy: [false; 2],
            img_rx: None,
            img_tx: None,
            pixels_per_point: 1.0,
            preview_drawn: false,
            params_open: false,
            height_manual: None,
            height_clip: [None; 2],
            processing_settings_open: false,
            help_open: false,
            about_open: false,
            licenses_open: false,
            licenses_third_party: false,
            confirm_preset: None,
            confirm_run: None,
            params_msg: None,
            new_work: None,
            confirm_replace: None,
            confirm_remove: None,
            rename_session: None,
            confirm_exit: false,
            exit_confirmed: false,
            window_place: Default::default(),
            device: device_calib::store(),
            device_calib: None,
            smoke: None,
        }
    }

    #[test]
    fn trajectory_cache_follows_edits_and_display_changes_but_not_camera_changes() {
        let mut app = app();
        let first = app.traj_lines(1.0);
        assert!(first.as_ref() == app.build_traj_lines(1.0));
        app.camera.yaw += 0.5;
        app.camera.target[0] += 10.0;
        assert!(Arc::ptr_eq(&first, &app.traj_lines(1.0)));
        let changes: &[fn(&mut App)] = &[
            |a| a.marker = Some((115.0, DVec3::Y)),
            |a| a.edits.loops.push((At::new(0.0), At::new(-1.0))),
            |a| a.edits.rtk_exclude.push((At::new(10.0), At::new(20.0))),
            |a| {
                a.placing = Some(Placing {
                    kind: EditKind::Exclude,
                    first: Some(100.0),
                });
                a.placing_hover = Some(105.0);
            },
            |a| a.edits.heading_hold.push((At::new(30.0), At::new(40.0))),
            |a| a.selected_edit = Some((EditKind::Hold, 0)),
            |a| a.s.traj_metric = 1,
            |a| a.s.metric_bad[0] = 100.0,
            |a| a.s.show_rtk = false,
            |a| a.s.show_edit_spans = false,
            |a| a.s.show_traj = false,
            |a| a.s.show_traj = true,
            |a| {
                a.hidden.insert("b".into());
            },
            |a| {
                a.hidden.remove("b");
            },
            |a| {
                a.scene.traj[0].1 = DVec3::Z;
                a.scene_revision += 1;
            },
        ];
        for change in changes {
            let old = app.traj_lines(1.0);
            change(&mut app);
            let new = app.traj_lines(1.0);
            assert!(!Arc::ptr_eq(&old, &new));
            assert!(new.as_ref() == app.build_traj_lines(1.0));
            assert!(Arc::ptr_eq(&new, &app.traj_lines(1.0)));
        }
        let old = app.traj_lines(1.0);
        let new = app.traj_lines(2.0);
        assert!(!Arc::ptr_eq(&old, &new));
        assert!(new.as_ref() == app.build_traj_lines(2.0));
    }

    #[test]
    fn worst_places_cache_follows_threshold_metric_and_scene_reload() {
        let mut app = app();
        for (m, bad) in [(0, 25.0), (0, 25.0), (0, 100.0), (1, 0.4)] {
            let actual = app.worst_places(m, bad);
            let expected = q::worst(app.scene.quality.as_ref().unwrap(), m, bad, 5.0, 8);
            assert_eq!(actual.as_ref(), expected);
            assert!(Arc::ptr_eq(&actual, &app.worst_places(m, bad)));
        }
        let before = app.worst_places(0, 25.0);
        app.scene.quality = None;
        app.scene_revision += 1;
        let after = app.worst_places(0, 25.0);
        assert!(!Arc::ptr_eq(&before, &after));
        assert!(after.is_empty());
    }

    #[test]
    fn step_buttons_start_at_the_first_unfinished_step_and_confirm_redos() {
        let mut app = app();
        for (s, st) in [
            (Step::Extract, StepState::Done),
            (Step::Lio, StepState::Stale),
            (Step::Opt, StepState::Done),
            (Step::Cloud, StepState::Pending),
            (Step::Export, StepState::Done),
            (Step::Colorize, StepState::Done),
        ] {
            app.steps.insert(s, st);
        }
        let run = |first, last| Run { first, last };
        // an earlier stale step: the run starts there
        assert_eq!(
            app.plan(Step::Cloud, true),
            run(Step::Lio, Some(Step::Cloud))
        );
        assert_eq!(app.plan(Step::Colorize, false), run(Step::Lio, None));
        // nothing unfinished before: the step itself
        assert_eq!(app.plan(Step::Lio, true), run(Step::Lio, Some(Step::Lio)));
        assert_eq!(app.plan(Step::Extract, false), run(Step::Extract, None));
        // redoing a finished step waits for a confirmation
        let ctx = egui::Context::default();
        app.request_run(&ctx, app.plan(Step::Extract, true));
        assert_eq!(
            app.confirm_run,
            Some(run(Step::Extract, Some(Step::Extract)))
        );
        app.confirm_run = None;
        app.request_run(&ctx, app.plan(Step::Cloud, true));
        assert_eq!(app.confirm_run, None);
    }

    #[test]
    fn optimisation_bar_does_not_reset_after_ppk_or_finish_before_extra_iterations() {
        let mut app = app();
        app.s.ppk = true;
        let (tx, rx) = mpsc::channel();
        app.job = Some(Running {
            rx,
            cancel: Arc::new(AtomicBool::new(false)),
            started: Instant::now(),
            opt_progress: OptProgress::default(),
            log_file: None,
        });
        let ctx = egui::Context::default();
        tx.send(Event::Log("[0.0s] == opt".into())).unwrap();
        let mut previous = 0.0;
        for (stage, done, total) in [
            ("ppk forward", 0, 1000),
            ("ppk forward", 500, 1000),
            ("ppk backward", 0, 1000),
            ("ppk backward", 1000, 1000),
            ("opt load", 0, 0),
            ("opt gnss", 0, 8),
            ("opt gnss", 7, 8),
            ("opt", 0, 10),
            ("opt", 5, 10),
            ("opt", 10, 10),
            ("opt", 20, 10),
        ] {
            tx.send(Event::Progress(stage.into(), done, total)).unwrap();
            app.poll(&ctx);
            assert_eq!(app.current, Some(Step::Opt));
            let Some(StepState::Running(f)) = app.steps.get(&Step::Opt) else {
                panic!("optimisation must remain running");
            };
            assert!(*f >= previous && *f < 1.0, "progress {previous} -> {f}");
            if stage.starts_with("opt") {
                assert!(*f >= 0.8, "PPK completed, but the bar reset");
            }
            if stage == "opt" {
                assert_eq!(app.job.as_ref().unwrap().opt_progress.label, "軌跡の最適化");
            }
            previous = *f;
        }
    }

    #[test]
    fn the_gnss_pose_graph_leaves_most_of_the_bar_to_the_iterations() {
        let mut p = OptProgress::default();
        assert_eq!(p.update("opt load", 0, 0), 0.0);
        // the pose graph on the GNSS fixes is quick: it nearly done is still a tenth of the bar
        let f = p.update("opt gnss", 7, 8);
        assert!((0.05..=0.1).contains(&f), "{f}");
        assert_eq!(p.label, "GNSS に合わせる");
        // the loops, then the iterations from where those left the bar
        assert!((p.update("opt loops", 1, 2) - 0.125).abs() < 1e-6);
        assert!((p.update("opt", 0, 26) - 0.15).abs() < 1e-6);
        assert!((p.update("opt", 13, 26) - 0.575).abs() < 1e-6);
        assert_eq!(p.label, "軌跡の最適化");
        // without loops the iterations start at their place all the same
        let mut p = OptProgress::default();
        p.update("opt gnss", 8, 8);
        assert!((p.update("opt", 0, 26) - 0.15).abs() < 1e-6);
    }

    #[test]
    fn cached_ppk_and_runs_without_ppk_use_the_full_optimisation_bar() {
        for ppk in [false, true] {
            let mut app = app();
            app.s.ppk = ppk;
            let (tx, rx) = mpsc::channel();
            app.job = Some(Running {
                rx,
                cancel: Arc::new(AtomicBool::new(false)),
                started: Instant::now(),
                opt_progress: OptProgress::default(),
                log_file: None,
            });
            // Cache reuse emits no PPK solver progress, even with PPK selected.
            tx.send(Event::Log("[0.0s] == opt".into())).unwrap();
            tx.send(Event::Progress("opt load".into(), 0, 0)).unwrap();
            app.poll(&egui::Context::default());
            assert!(matches!(
                app.steps.get(&Step::Opt),
                Some(StepState::Running(0.0))
            ));
            tx.send(Event::Progress("opt".into(), 5, 10)).unwrap();
            app.poll(&egui::Context::default());
            let Some(StepState::Running(f)) = app.steps.get(&Step::Opt) else {
                panic!("optimisation must be running");
            };
            assert!((f - 0.575).abs() < 1e-6, "{f}");
        }
    }

    #[test]
    fn a_dragged_matched_point_moves_only_onto_its_own_session() {
        let mut app = app();
        app.scene.sessions = ["a", "b"]
            .iter()
            .zip([(100.0, 159.0), (160.0, 219.0)])
            .map(|(name, (start, end))| scene::SceneSession {
                name: (*name).into(),
                start,
                end,
                placed: true,
            })
            .collect();
        let p0 = app.point_at(0, DVec3::new(1.0, 0.0, 0.0)).unwrap();
        let p1 = app.point_at(1, DVec3::new(2.0, 0.0, 0.0)).unwrap();
        app.edits.matches.push((p0, p1));
        let moved = DVec3::new(3.0, 1.0, 0.5);
        app.match_hover = Some((moved, Some(1)));
        app.drop_match(0, true);
        assert!((app.match_world(&app.edits.matches[0].1).unwrap() - moved).length() < 1e-9);
        // onto a point of the other session: unchanged
        let before = app.edits.matches[0].clone();
        app.match_hover = Some((DVec3::new(9.0, 0.0, 0.0), Some(0)));
        app.drop_match(0, true);
        app.match_hover = None;
        app.drop_match(0, false);
        assert_eq!(app.edits.matches[0], before);
    }

    #[test]
    fn times_and_points_on_sessions() {
        let mut app = app();
        // two sessions: 100..160 s and 160..220 s, the second turned by 90 degrees
        app.scene.sessions = vec![
            scene::SceneSession {
                name: "a".into(),
                start: 100.0,
                end: 159.0,
                placed: true,
            },
            scene::SceneSession {
                name: "b".into(),
                start: 160.0,
                end: 219.0,
                placed: false,
            },
        ];
        app.scene.rot = (0..120)
            .map(|i| {
                if i < 60 {
                    glam::DQuat::IDENTITY
                } else {
                    glam::DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2)
                }
            })
            .collect();
        let at = app.at_of(175.0);
        assert_eq!((at.session.as_str(), at.t), ("b", 15.0));
        assert_eq!(app.abs_of(&at), Some(175.0));
        assert_eq!(app.abs_of(&At::on("a", -1.0)), Some(158.0));
        assert_eq!(app.at_label(&At::on("b", -0.5)), "b 終点");
        // a picked point comes back where it was shown
        let p = DVec3::new(8.3, 1.5, 0.7);
        let pa = app.point_at(1, p).unwrap();
        assert_eq!(pa.session, "b");
        assert!((app.match_world(&pa).unwrap() - p).length() < 1e-9);
        // the corrections of the first session get its name before the order changes
        let mut e = Edits::default();
        e.loops.push((At::new(1.0), At::on("b", 2.0)));
        App::name_first(&mut e, "a");
        assert_eq!(e.loops[0].0.session, "a");
    }

    #[test]
    fn loop_ends_have_times_and_labels() {
        let app = app();
        // the fixture's trajectory runs from 100 s to 219 s, its origin is 100 s
        assert_eq!(app.loop_time(0.5), Some(100.5));
        assert_eq!(app.loop_time(-0.5), Some(218.5));
        assert_eq!(app.loop_time(15.0), Some(115.0));
        assert_eq!(App::loop_label(0.5), "始点");
        assert_eq!(App::loop_label(-0.5), "終点");
        assert_eq!(App::loop_label(1206.25), "1206.2 秒");
    }

    #[test]
    fn an_interval_end_picks_the_pass_nearest_in_time_to_the_other_end() {
        use super::traj_edit::choose_pass;
        // the trajectory goes by twice: samples 10-12 (at 10-12 s) and 80-81 (at 80-81 s)
        let cands = [
            (10, 9.0, 10.0),
            (11, 3.0, 11.0),
            (12, 7.0, 12.0),
            (80, 2.0, 80.0),
            (81, 6.0, 81.0),
        ];
        // without a preference the nearest on screen, with one the pass nearest in time, and
        // on that pass the sample nearest on screen
        assert_eq!(choose_pass(&cands, None), Some(80));
        assert_eq!(choose_pass(&cands, Some(5.0)), Some(11));
        assert_eq!(choose_pass(&cands, Some(70.0)), Some(80));
        assert_eq!(choose_pass(&[], Some(5.0)), None);
    }

    #[test]
    fn corrections_are_placed_by_two_clicks_and_removed_when_selected() {
        let mut a = app();
        // an interval: too short is refused and waits for another end
        a.toggle_placing(EditKind::Hold);
        a.place_at(130.0);
        a.place_at(130.5);
        assert!(a.edits.heading_hold.is_empty());
        assert_eq!(a.notices.texts()[0].0, Level::Warn);
        // the ends in either order; the new one is selected
        a.place_at(120.0);
        assert_eq!(a.edits.heading_hold.len(), 1);
        let (x, y) = &a.edits.heading_hold[0];
        assert_eq!((a.abs_of(x), a.abs_of(y)), (Some(120.0), Some(130.0)));
        assert_eq!(a.placing, None);
        assert_eq!(a.selected_edit, Some((EditKind::Hold, 0)));
        // while one is placed, the trajectory shows it
        a.toggle_placing(EditKind::Exclude);
        a.place_at(150.0);
        a.placing_hover = Some(140.0);
        assert_eq!(a.placing_interval(EditKind::Exclude), Some((140.0, 150.0)));
        assert!(a.excluded(145.0) && !a.excluded(155.0));
        // a second press of the button stops placing
        a.toggle_placing(EditKind::Exclude);
        assert_eq!(a.placing, None);
        // a loop needs two times apart
        a.toggle_placing(EditKind::Loop);
        a.place_at(110.0);
        a.place_at(111.0);
        assert!(a.edits.loops.is_empty());
        a.place_at(200.0);
        assert_eq!(a.edits.loops.len(), 1);
        // Delete removes the selected one, and undo brings it back
        a.selected_edit = Some((EditKind::Hold, 0));
        a.remove_selected_edit();
        assert!(a.edits.heading_hold.is_empty());
        assert_eq!(a.selected_edit, None);
        a.undo_edits();
        assert_eq!(a.edits.heading_hold.len(), 1);
    }

    #[test]
    fn the_intervals_colours_can_be_hidden_but_not_the_one_being_placed() {
        let mut a = app();
        a.edits.heading_hold.push((At::new(10.0), At::new(20.0)));
        assert_eq!(a.traj_color(115.0, 100.0, 219.0), HELD);
        a.s.show_edit_spans = false;
        assert_ne!(a.traj_color(115.0, 100.0, 219.0), HELD);
        a.toggle_placing(EditKind::Hold);
        a.place_at(150.0);
        a.placing_hover = Some(160.0);
        assert_eq!(a.traj_color(155.0, 100.0, 219.0), HELD);
        assert_ne!(a.traj_color(115.0, 100.0, 219.0), HELD);
    }

    #[test]
    fn applying_the_corrections_starts_from_the_odometry_when_it_is_out_of_date() {
        let mut a = app();
        a.steps.insert(Step::Extract, StepState::Done);
        a.steps.insert(Step::Lio, StepState::Done);
        a.steps.insert(Step::Opt, StepState::Stale);
        assert_eq!(a.apply_run().first, Step::Opt);
        a.steps.insert(Step::Lio, StepState::Stale);
        let run = a.apply_run();
        assert_eq!((run.first, run.last), (Step::Lio, Some(Step::Opt)));
    }

    #[test]
    fn outcomes_of_actions_show_as_notices_and_stay_in_the_log() {
        let mut a = app();
        a.undo_edits();
        assert_eq!(
            a.notices.texts(),
            [(Level::Info, "元に戻す修正はありません")]
        );
        // the same message twice is shown once
        a.notify(Level::Warn, "区間が短すぎます");
        a.notify(Level::Warn, "区間が短すぎます");
        assert_eq!(a.notices.texts().len(), 2);
        assert_eq!(a.log.last().map(String::as_str), Some("区間が短すぎます"));
        // only the newest few
        for k in 0..10 {
            a.notify(Level::Error, format!("失敗 {k}"));
        }
        assert_eq!(a.notices.texts().last(), Some(&(Level::Error, "失敗 9")));
        assert!(a.notices.texts().len() <= 4);
    }
}
