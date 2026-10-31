//! The raw data of a session: its files, the topics of its bags and their messages (read only).
use super::*;
use crate::lang::tl;
use crate::rosmsg::{self, Schema};
use s20_io::bag::{Bag, MsgRef};
use s20_pipeline::ppk::inspect::{self as gnss, Fix, Positioning, RawEpoch, RawSummary};
use std::collections::BTreeMap;

/// A bag opened for viewing, with its messages by topic.
pub(super) struct RawBag {
    bag: Bag,
    topics: Vec<RawTopic>,
    /// the first record time (ns), the origin of the times shown
    start_ns: u64,
    /// sensor clock minus record time (s), from the IMU (none in a bag without it)
    sensor_offset: Option<f64>,
}

struct RawTopic {
    name: String,
    msg_type: String,
    definition: String,
    schema: Schema,
    /// by record time
    messages: Vec<MsgRef>,
    /// the longest interval between two messages (s)
    max_gap: f64,
}

impl RawBag {
    fn open(path: &Path) -> anyhow::Result<Self> {
        let bag = Bag::open(path)?;
        let mut kinds = BTreeMap::<String, (String, String)>::new();
        for c in bag.connections.values() {
            kinds
                .entry(c.topic.clone())
                .or_insert_with(|| (c.msg_type.clone(), c.message_definition.clone()));
        }
        let topics: Vec<RawTopic> = kinds
            .into_iter()
            .map(|(name, (msg_type, definition))| {
                let messages = bag.topic_messages(&name);
                let max_gap = messages
                    .windows(2)
                    .map(|w| (w[1].time_ns - w[0].time_ns) as f64 * 1e-9)
                    .fold(0.0, f64::max);
                RawTopic {
                    schema: Schema::new(&msg_type, &definition),
                    name,
                    msg_type,
                    definition,
                    messages,
                    max_gap,
                }
            })
            .collect();
        let start_ns = bag.messages.iter().map(|m| m.time_ns).min().unwrap_or(0);
        // as the extraction moves the info bag to the sensor clock: the median of IMU header stamp
        // minus record time
        let mut d: Vec<f64> = bag
            .topic_messages("/livox/imu")
            .iter()
            .step_by(50)
            .filter_map(|m| Some(s20_io::msgs::imu(bag.data(m)).ok()?[0] - m.time_ns as f64 * 1e-9))
            .collect();
        d.sort_by(f64::total_cmp);
        let sensor_offset = d.get(d.len() / 2).copied();
        Ok(Self {
            bag,
            topics,
            start_ns,
            sensor_offset,
        })
    }

    /// Seconds from the bag's first record.
    fn seconds(&self, ns: u64) -> f64 {
        ns.saturating_sub(self.start_ns) as f64 * 1e-9
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
enum RawTab {
    #[default]
    Files,
    Topics,
    Messages,
    Gnss,
}

/// The GNSS raw files of a session (the rover's and the base's) and its position solutions.
pub(super) struct GnssFiles {
    rover: Option<RawSummary>,
    base: Option<RawSummary>,
    /// the PPK result, else the receiver's real-time RTK; why none
    positioning: Result<Positioning, String>,
}

/// The position solutions of session `k`: its PPK result when there is one, else the receiver's
/// real-time RTK as extracted (on the device clock, back to UTC by the synchronisation).
fn positioning(
    cfg: &RunConfig,
    k: usize,
    raw: &Path,
    job: &s20_io::job::Job,
) -> Result<Positioning, String> {
    if let Ok(r) = s20_pipeline::ppk::read(&cfg.ppk_file_of(k)) {
        // a FIX told apart by the directions that fixed it, and whether the optimisation uses it
        let directions = r.directions();
        return Ok(Positioning {
            source: "PPK".into(),
            states: r
                .solutions
                .iter()
                .zip(directions)
                .filter_map(|(s, d)| Some((s.utc, Fix::of_ppk(s, d, &r.params)?)))
                .collect(),
        });
    }
    let (rows, n, cols) =
        s20_io::npy::read_f64(&cfg.session_data(k).join("gnss.npy")).map_err(|_| {
            tl(
                "PPK も、機体の RTK の取り込みもまだありません（データ取り込みで読みます）",
                "Neither PPK nor the device's RTK has been imported yet (data import reads them)",
            )
            .to_string()
        })?;
    let clock = gnss::SyncClock::read(raw, job).map_err(|e| {
        tr!(
            "時刻の同期を読めません: {e:#}",
            "Can't read the clock sync: {e:#}"
        )
    })?;
    let mut states: Vec<(f64, Fix)> = (0..n)
        .filter_map(|i| {
            let r = &rows[i * cols..(i + 1) * cols];
            Some((clock.utc(r[0])?, Fix::of_receiver(r[1] as u8)?))
        })
        .collect();
    states.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(Positioning {
        source: tl("機体の RTK", "Device RTK").into(),
        states,
    })
}

/// The colours of the solution states in the charts, as [`Fix::ALL`].
const FIX_COLORS: [egui::Color32; Fix::ALL.len()] = [
    egui::Color32::from_rgb(110, 200, 120),
    egui::Color32::from_rgb(190, 215, 90),
    egui::Color32::from_rgb(235, 95, 120),
    egui::Color32::from_rgb(240, 160, 60),
    egui::Color32::from_rgb(90, 160, 255),
    egui::Color32::from_rgb(150, 150, 150),
];

/// The GNSS raw files of a session being read: the session, the result to come, and the job.
type GnssLoading = (
    usize,
    mpsc::Receiver<Result<GnssFiles, String>>,
    s20_io::job::Job,
);

#[derive(Default)]
pub(super) struct RawView {
    pub(super) open: bool,
    session: usize,
    /// the bag chosen, by its path
    file: Option<PathBuf>,
    tab: RawTab,
    loaded: Option<(PathBuf, Arc<RawBag>)>,
    loading: Option<(PathBuf, mpsc::Receiver<Result<RawBag, String>>)>,
    error: Option<String>,
    topic: usize,
    row: Option<usize>,
    /// a session's clock offset from its recording bag, for its other bags
    offset: Option<(usize, f64)>,
    /// the GNSS raw files of a session, read
    gnss: Option<(usize, Result<Arc<GnssFiles>, String>)>,
    /// and being read, with its job (dropped when the window closes)
    gnss_loading: Option<GnssLoading>,
}

/// The files of a raw folder (with those in its subfolders), as (path from it, bytes).
fn raw_files(raw: &Path) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let mut dirs = vec![raw.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let path = e.path();
            if path.is_dir() {
                dirs.push(path);
            } else if out.len() < 500 {
                let name = path
                    .strip_prefix(raw)
                    .unwrap_or(&path)
                    .display()
                    .to_string();
                out.push((name, e.metadata().map_or(0, |m| m.len())));
            }
        }
    }
    out.sort();
    out
}

/// The bags of a raw folder: the recording (`all_*.bag`) and the device information (`info/`).
fn raw_bags(raw: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = [raw.to_path_buf(), raw.join("info")]
        .iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flat_map(|rd| rd.flatten().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "bag"))
        .collect();
    out.sort();
    out
}

/// The files the processing reads: (pattern, what it is for).
const KEY_FILES: [(&str, [&str; 2]); 6] = [
    (
        "all_*.bag",
        [
            "LiDAR・IMU・画像・GNSS の記録",
            "LiDAR, IMU, image and GNSS recording",
        ],
    ),
    (
        "info/info_*.bag",
        [
            "機器の情報（時刻の同期、温度）",
            "Device information (clock sync, temperature)",
        ],
    ),
    (
        "info/calibration.yaml",
        ["LiDAR・カメラの校正値", "LiDAR and camera calibration"],
    ),
    (
        "ROVER_PPKRAW.bin",
        ["機体の GNSS 生データ（PPK）", "Device GNSS raw data (PPK)"],
    ),
    (
        "BASE.BIN",
        ["基準局のデータ（PPK）", "Base station data (PPK)"],
    ),
    (
        "frame_pose.txt",
        ["機体が推定した軌跡", "Trajectory estimated by the device"],
    ),
];

fn matches(pattern: &str, name: &str) -> bool {
    let name = name.replace('\\', "/");
    match pattern.split_once('*') {
        Some((a, b)) => {
            name.len() >= a.len() + b.len()
                && name[..a.len()].eq_ignore_ascii_case(a)
                && name[name.len() - b.len()..].eq_ignore_ascii_case(b)
        }
        None => name.eq_ignore_ascii_case(pattern),
    }
}

impl App {
    /// Shows the raw data window of session `k`.
    pub(super) fn open_raw_view(&mut self, k: usize) {
        self.raw_view.open = true;
        if self.raw_view.session != k {
            self.raw_view.session = k;
            self.raw_view.file = None;
        }
    }

    /// The time on the common axis of a message recorded at `record` (s), on session `k` whose
    /// sensor clock is `offset` ahead of the record times, when it falls in the session.
    fn raw_axis_time(&self, k: usize, record: f64, offset: f64) -> Option<f64> {
        let cfg = self.config()?;
        let shift = s20_io::dataset::sessions(&cfg.data())
            .ok()
            .and_then(|d| d.get(k).map(|s| s.shift))
            .unwrap_or(0.0);
        let t = record + offset + shift;
        let (start, end) = match self.scene.sessions.get(k) {
            Some(s) if self.multi() => (s.start, s.end),
            _ => (self.scene.traj.first()?.0, self.scene.traj.last()?.0),
        };
        (t >= start - 1.0 && t <= end + 1.0).then_some(t)
    }

    pub(super) fn raw_window(&mut self, ctx: &egui::Context) {
        if !self.raw_view.open {
            return;
        }
        let Some(cfg) = self.config() else {
            self.raw_view.open = false;
            return;
        };
        let k = self.raw_view.session.min(cfg.sessions.len() - 1);
        let raw = cfg.sessions[k].raw.clone();
        // the bag chosen: by default the recording
        let bags = raw_bags(&raw);
        if self
            .raw_view
            .file
            .as_ref()
            .is_none_or(|f| !bags.contains(f))
        {
            self.raw_view.file = bags.first().cloned();
        }
        self.poll_raw_bag(ctx);
        let mut open = true;
        egui::Window::new(tr!("{} 生データ", "{} Raw data", icon::DATABASE))
            .open(&mut open)
            .default_size([980.0, 620.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if cfg.sessions.len() > 1 {
                        let mut sel = k;
                        egui::ComboBox::from_id_salt("raw_session")
                            .selected_text(format!(
                                "{} {}",
                                Self::session_tag(k),
                                cfg.sessions[k].name
                            ))
                            .show_ui(ui, |ui| {
                                for (i, s) in cfg.sessions.iter().enumerate() {
                                    ui.selectable_value(
                                        &mut sel,
                                        i,
                                        format!("{} {}", Self::session_tag(i), s.name),
                                    );
                                }
                            });
                        if sel != k {
                            self.open_raw_view(sel);
                        }
                    }
                    ui.add(
                        egui::Label::new(egui::RichText::new(raw.display().to_string()).weak())
                            .truncate(),
                    )
                    .on_hover_text(tl(
                        "読み取りだけで、生データには書き込みません",
                        "Read only; nothing is written to the raw data",
                    ));
                });
                ui.horizontal(|ui| {
                    for (tab, text) in [
                        (RawTab::Files, tr!("{} ファイル", "{} Files", icon::FOLDER)),
                        (
                            RawTab::Topics,
                            tr!("{} トピック", "{} Topics", icon::LIST_BULLETS),
                        ),
                        (
                            RawTab::Messages,
                            tr!("{} メッセージ", "{} Messages", icon::LIST_MAGNIFYING_GLASS),
                        ),
                        (
                            RawTab::Gnss,
                            format!("{} GNSS", icon::GLOBE_HEMISPHERE_EAST),
                        ),
                    ] {
                        ui.selectable_value(&mut self.raw_view.tab, tab, text);
                    }
                    if matches!(self.raw_view.tab, RawTab::Topics | RawTab::Messages)
                        && !bags.is_empty()
                    {
                        ui.separator();
                        ui.label("bag:");
                        let name =
                            |p: &PathBuf| p.strip_prefix(&raw).unwrap_or(p).display().to_string();
                        let mut file = self.raw_view.file.clone();
                        egui::ComboBox::from_id_salt("raw_bag")
                            .width(260.0)
                            .selected_text(file.as_ref().map(name).unwrap_or_default())
                            .show_ui(ui, |ui| {
                                for b in &bags {
                                    ui.selectable_value(&mut file, Some(b.clone()), name(b));
                                }
                            });
                        if file != self.raw_view.file {
                            self.raw_view.file = file;
                            self.raw_view.topic = 0;
                            self.raw_view.row = None;
                        }
                    }
                });
                ui.separator();
                match self.raw_view.tab {
                    RawTab::Files => raw_files_ui(ui, &raw),
                    RawTab::Gnss => self.gnss_ui(ui, k, &raw),
                    _ => {
                        let Some(bag) = self.raw_bag(ctx) else {
                            if let Some(e) = &self.raw_view.error {
                                ui.colored_label(
                                    egui::Color32::LIGHT_RED,
                                    tr!("bag を読めません: {e}", "Can't read the bag: {e}"),
                                );
                            } else if self.raw_view.loading.is_some() {
                                ui.horizontal(|ui| {
                                    ui.spinner();
                                    ui.label(tl(
                                        "bag の索引を読み込んでいます",
                                        "Reading the bag index",
                                    ));
                                });
                            } else {
                                ui.weak(tl("bag がありません", "No bag"));
                            }
                            return;
                        };
                        if self.raw_view.tab == RawTab::Topics {
                            if let Some(i) = topics_ui(ui, &bag, self.raw_view.topic) {
                                self.raw_view.topic = i;
                                self.raw_view.row = None;
                                self.raw_view.tab = RawTab::Messages;
                            }
                        } else {
                            self.messages_ui(ui, &bag, k);
                        }
                    }
                }
            });
        self.raw_view.open = open;
        if !open && let Some((_, _, job)) = self.raw_view.gnss_loading.take() {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// The GNSS raw files of session `k`: read on a thread the first time.
    fn gnss_ui(&mut self, ui: &mut egui::Ui, k: usize, raw: &Path) {
        if let Some((s, rx, _)) = &self.raw_view.gnss_loading
            && let Ok(r) = rx.try_recv()
        {
            self.raw_view.gnss = Some((*s, r.map(Arc::new)));
            self.raw_view.gnss_loading = None;
        }
        let loaded = self
            .raw_view
            .gnss
            .as_ref()
            .filter(|g| g.0 == k)
            .map(|g| g.1.clone());
        let Some(result) = loaded else {
            if self.raw_view.gnss_loading.as_ref().is_none_or(|l| l.0 != k) {
                if let Some((_, _, job)) = self.raw_view.gnss_loading.take() {
                    job.cancel.store(true, Ordering::Relaxed);
                }
                let (tx, rx) = mpsc::channel();
                let job = s20_io::job::Job::new(|_, _, _| {}, |_| {});
                let (raw, run, ctx) = (raw.to_path_buf(), job.clone(), ui.ctx().clone());
                let cfg = self.config();
                std::thread::spawn(move || {
                    let result = crate::panics::catch(|| {
                        gnss::inspect(&raw, &run)
                            .map(|(rover, base)| GnssFiles {
                                rover,
                                base,
                                positioning: cfg
                                    .map_or(Err(String::new()), |c| positioning(&c, k, &raw, &run)),
                            })
                            .map_err(|e| format!("{e:#}"))
                    });
                    let _ = tx.send(result.map_err(|p| p.to_string()).and_then(|r| r));
                    ctx.request_repaint();
                });
                self.raw_view.gnss_loading = Some((k, rx, job));
            }
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(tl(
                    "ROVER_PPKRAW.bin と BASE.BIN を読んでいます（RTKLIB で復号）",
                    "Reading ROVER_PPKRAW.bin and BASE.BIN (decoded with RTKLIB)",
                ));
            });
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(200));
            return;
        };
        let files = match result {
            Ok(f) => f,
            Err(e) => {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    tr!("読めません: {e}", "Can't read it: {e}"),
                );
                return;
            }
        };
        let (rover, base) = (&files.rover, &files.base);
        // one time axis for both
        let range = [rover, base]
            .into_iter()
            .flatten()
            .flat_map(|s| [s.epochs.first(), s.epochs.last()])
            .flatten()
            .map(|e| e.utc)
            .fold(None, |r: Option<[f64; 2]>, t| {
                Some(r.map_or([t, t], |[a, b]| [a.min(t), b.max(t)]))
            });
        egui::ScrollArea::vertical()
            .id_salt("raw_gnss")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (title, file, summary, rover) in [
                    (
                        tl("ローバー（機体）", "Rover (device)"),
                        "ROVER_PPKRAW.bin",
                        rover,
                        true,
                    ),
                    (tl("基準局", "Base station"), "BASE.BIN", base, false),
                ] {
                    ui.add_space(4.0);
                    ui.strong(format!("{title}  {file}"));
                    match summary {
                        Some(s) => gnss_file_ui(ui, s, rover.then_some(&files.positioning), range),
                        None => {
                            ui.colored_label(
                                egui::Color32::from_rgb(230, 190, 70),
                                tr!(
                                    "{} ありません（PPK はできません）",
                                    "{} missing (PPK not possible)",
                                    icon::WARNING
                                ),
                            );
                        }
                    }
                    ui.separator();
                }
            });
    }

    /// The bag chosen when it is loaded, else starts loading it.
    fn raw_bag(&mut self, ctx: &egui::Context) -> Option<Arc<RawBag>> {
        let file = self.raw_view.file.clone()?;
        if let Some((p, b)) = &self.raw_view.loaded
            && *p == file
        {
            return Some(b.clone());
        }
        if self
            .raw_view
            .loading
            .as_ref()
            .is_none_or(|(p, _)| *p != file)
        {
            let (tx, rx) = mpsc::channel();
            let path = file.clone();
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                let bag =
                    crate::panics::catch(|| RawBag::open(&path).map_err(|e| format!("{e:#}")));
                let _ = tx.send(bag.map_err(|p| p.to_string()).and_then(|r| r));
                ctx.request_repaint();
            });
            self.raw_view.loading = Some((file, rx));
            self.raw_view.error = None;
        }
        None
    }

    fn poll_raw_bag(&mut self, ctx: &egui::Context) {
        let Some((path, rx)) = &self.raw_view.loading else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(bag)) => {
                if let Some(o) = bag.sensor_offset {
                    self.raw_view.offset = Some((self.raw_view.session, o));
                }
                // the previous bag's map is released
                self.raw_view.loaded = Some((path.clone(), Arc::new(bag)));
                self.raw_view.loading = None;
                self.raw_view.topic = self.raw_view.topic.min(
                    self.raw_view
                        .loaded
                        .as_ref()
                        .map_or(0, |(_, b)| b.topics.len().saturating_sub(1)),
                );
            }
            Ok(Err(e)) => {
                self.raw_view.error = Some(e);
                self.raw_view.loading = None;
            }
            Err(mpsc::TryRecvError::Empty) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(100))
            }
            Err(mpsc::TryRecvError::Disconnected) => self.raw_view.loading = None,
        }
    }

    /// The messages of the topic chosen, and the one selected: its fields and definition.
    fn messages_ui(&mut self, ui: &mut egui::Ui, bag: &RawBag, k: usize) {
        let Some(t) = bag.topics.get(self.raw_view.topic) else {
            ui.weak(tl("トピックがありません", "No topics"));
            return;
        };
        let mut topic = self.raw_view.topic;
        egui::ComboBox::from_id_salt("raw_topic")
            .width(320.0)
            .selected_text(&t.name)
            .show_ui(ui, |ui| {
                for (i, t) in bag.topics.iter().enumerate() {
                    ui.selectable_value(
                        &mut topic,
                        i,
                        format!("{}  ({})", t.name, t.messages.len()),
                    );
                }
            });
        if topic != self.raw_view.topic {
            self.raw_view.topic = topic;
            self.raw_view.row = None;
            return;
        }
        ui.weak(tr!(
            "{} · {} 件 · 時刻は bag の最初の記録からの秒",
            "{} · {} messages · times in seconds from the bag's first record",
            t.msg_type,
            t.messages.len()
        ));
        let width = ui.available_width();
        let height = ui.available_height();
        let mut clicked = None;
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(width * 0.5);
                ui.set_height(height);
                egui::ScrollArea::both()
                    .id_salt("raw_rows")
                    .auto_shrink([false, false])
                    .show_rows(ui, 18.0, t.messages.len(), |ui, range| {
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                        for row in range {
                            let m = &t.messages[row];
                            let summary = t
                                .schema
                                .decode(bag.bag.data(m))
                                .map(|v| rosmsg::brief(&v, &t.msg_type))
                                .unwrap_or_else(|e| e.to_string());
                            let text =
                                format!("{row:>7} {:>10.3}  {summary}", bag.seconds(m.time_ns));
                            if ui
                                .selectable_label(
                                    self.raw_view.row == Some(row),
                                    egui::RichText::new(text).monospace(),
                                )
                                .clicked()
                            {
                                clicked = Some(row);
                            }
                        }
                    });
            });
            ui.separator();
            ui.vertical(|ui| {
                egui::ScrollArea::vertical()
                    .id_salt("raw_detail")
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.message_detail_ui(ui, bag, t, k));
            });
        });
        if let Some(row) = clicked {
            self.raw_view.row = Some(row);
        }
    }

    fn message_detail_ui(&mut self, ui: &mut egui::Ui, bag: &RawBag, t: &RawTopic, k: usize) {
        let Some(m) = self.raw_view.row.and_then(|r| t.messages.get(r)) else {
            ui.weak(tl(
                "左の表で行を選ぶと、内容を表示します",
                "Select a row in the table on the left to show its content",
            ));
            definition_ui(ui, t);
            return;
        };
        ui.strong(tr!(
            "#{} · {:.3} 秒 · {} バイト",
            "#{} · {:.3} s · {} bytes",
            self.raw_view.row.unwrap_or(0),
            bag.seconds(m.time_ns),
            m.len
        ));
        match t.schema.decode(bag.bag.data(m)) {
            Ok(v) => {
                if let Some(stamp) = rosmsg::stamp(&v) {
                    ui.weak(tr!("ヘッダーの時刻 {stamp:.6}", "Header time {stamp:.6}"));
                }
                // by the record time on the sensor clock: the header stamps of some topics are on
                // another clock (the GNSS's are UTC)
                let offset =
                    bag.sensor_offset
                        .or(self.raw_view.offset.filter(|o| o.0 == k).map(|o| o.1));
                let at = offset.and_then(|o| self.raw_axis_time(k, m.time_ns as f64 * 1e-9, o));
                let why = if offset.is_none() {
                    tl(
                        "時計を合わせられません（このセッションの記録 bag（all_*.bag）を一度開くと合わせられます）",
                        "The clocks can't be matched (opening this session's recording bag (all_*.bag) once matches them)",
                    )
                } else {
                    tl(
                        "軌跡の時間の範囲外です",
                        "Outside the trajectory's time range",
                    )
                };
                if ui
                    .add_enabled(
                        at.is_some(),
                        egui::Button::new(tr!(
                            "{} この時刻へ",
                            "{} Go to this time",
                            icon::CROSSHAIR
                        )),
                    )
                    .on_hover_text(tl(
                        "時刻バーの印を、このメッセージが記録された時刻に動かす",
                        "Move the time bar's mark to the time this message was recorded",
                    ))
                    .on_disabled_hover_text(why)
                    .clicked()
                    && let Some(t) = at
                {
                    self.go_to_time(t);
                }
                let text = serde_json::to_string_pretty(&v).unwrap_or_default();
                ui.add(
                    egui::Label::new(egui::RichText::new(text).monospace().size(12.0))
                        .wrap_mode(egui::TextWrapMode::Extend),
                );
            }
            Err(e) => {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    tr!("読めません: {e:#}", "Can't read it: {e:#}"),
                );
            }
        }
        definition_ui(ui, t);
    }
}

fn definition_ui(ui: &mut egui::Ui, t: &RawTopic) {
    egui::CollapsingHeader::new(tl("メッセージの定義", "Message definition"))
        .id_salt(("raw_definition", &t.name))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(&t.definition).monospace().size(12.0));
        });
}

fn raw_files_ui(ui: &mut egui::Ui, raw: &Path) {
    let files = raw_files(raw);
    ui.strong(tl("処理に使うファイル", "Files used in processing"));
    egui::Grid::new("raw_key_files")
        .striped(true)
        .show(ui, |ui| {
            for (pattern, what) in KEY_FILES {
                let found: Vec<&(String, u64)> =
                    files.iter().filter(|(n, _)| matches(pattern, n)).collect();
                if found.is_empty() {
                    ui.colored_label(egui::Color32::from_rgb(230, 190, 70), icon::WARNING);
                } else {
                    ui.colored_label(egui::Color32::from_rgb(110, 200, 120), icon::CHECK);
                }
                ui.monospace(pattern);
                ui.label(tl(what[0], what[1]));
                ui.label(if found.is_empty() {
                    tl("ありません", "None").to_string()
                } else {
                    format_bytes(found.iter().map(|f| f.1).sum())
                });
                ui.end_row();
            }
        });
    ui.add_space(8.0);
    ui.strong(tr!("すべてのファイル（{}）", "All files ({})", files.len()));
    egui::ScrollArea::vertical()
        .id_salt("raw_files")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::Grid::new("raw_files_grid")
                .striped(true)
                .show(ui, |ui| {
                    for (name, bytes) in &files {
                        ui.monospace(name);
                        ui.label(format_bytes(*bytes));
                        ui.end_row();
                    }
                });
        });
}

/// The topics of a bag; returns the one clicked.
fn topics_ui(ui: &mut egui::Ui, bag: &RawBag, current: usize) -> Option<usize> {
    let mut chosen = None;
    ui.weak(tl("時刻は bag の最初の記録からの秒。最大間隔は、隣り合うメッセージの記録時刻の差の最大（欠けている区間の目安）", "Times in seconds from the bag's first record. Max gap is the largest difference between neighbouring messages' record times (a hint of missing sections)"));
    egui::ScrollArea::both()
        .id_salt("raw_topics")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::Grid::new("raw_topics_grid")
                .striped(true)
                .show(ui, |ui| {
                    for title in [
                        tl("トピック", "Topic"),
                        tl("型", "Type"),
                        tl("件数", "Count"),
                        tl("開始", "Start"),
                        tl("終了", "End"),
                        tl("平均 Hz", "Mean Hz"),
                        tl("最大間隔", "Max gap"),
                    ] {
                        ui.strong(title);
                    }
                    ui.end_row();
                    for (i, t) in bag.topics.iter().enumerate() {
                        if ui
                            .selectable_label(i == current, &t.name)
                            .on_hover_text(tl(
                                "クリックでメッセージを表示",
                                "Click to show the messages",
                            ))
                            .clicked()
                        {
                            chosen = Some(i);
                        }
                        ui.label(&t.msg_type);
                        ui.monospace(t.messages.len().to_string());
                        let (first, last) = (
                            t.messages.first().map_or(0, |m| m.time_ns),
                            t.messages.last().map_or(0, |m| m.time_ns),
                        );
                        ui.monospace(format!("{:.3}", bag.seconds(first)));
                        ui.monospace(format!("{:.3}", bag.seconds(last)));
                        let span = (last - first) as f64 * 1e-9;
                        let hz = (span > 0.0).then(|| (t.messages.len() - 1) as f64 / span);
                        ui.monospace(hz.map_or("—".into(), |h| format!("{h:.2}")));
                        // a gap ten times the usual interval stands out
                        let long = hz.is_some_and(|h| t.max_gap > 10.0 / h && t.max_gap > 0.5);
                        let gap =
                            egui::RichText::new(tr!("{:.3} 秒", "{:.3} s", t.max_gap)).monospace();
                        if long {
                            ui.label(gap.color(egui::Color32::from_rgb(230, 190, 70)))
                                .on_hover_text(tl(
                                    "ふだんの間隔の 10 倍を超える間隔があります",
                                    "There are gaps over 10 times the usual interval",
                                ));
                        } else {
                            ui.label(gap);
                        }
                        ui.end_row();
                    }
                });
        });
    chosen
}

/// Local time of a UTC time (s since 1970).
fn local_time(utc: f64) -> String {
    chrono::DateTime::from_timestamp(utc.floor() as i64, 0)
        .map(|d| {
            d.with_timezone(&chrono::Local)
                .format("%H:%M:%S")
                .to_string()
        })
        .unwrap_or_default()
}

/// The colours of the satellite systems in the charts.
const SYSTEM_COLORS: [egui::Color32; 6] = [
    egui::Color32::from_rgb(90, 160, 255),
    egui::Color32::from_rgb(240, 110, 100),
    egui::Color32::from_rgb(110, 200, 120),
    egui::Color32::from_rgb(230, 190, 70),
    egui::Color32::from_rgb(190, 150, 255),
    egui::Color32::GRAY,
];

/// A GNSS raw file: the rover's with the position solutions, the base's without.
fn gnss_file_ui(
    ui: &mut egui::Ui,
    s: &RawSummary,
    positioning: Option<&Result<Positioning, String>>,
    range: Option<[f64; 2]>,
) {
    let rover = positioning.is_some();
    let yellow = egui::Color32::from_rgb(230, 190, 70);
    egui::Grid::new(("gnss_grid", rover))
        .num_columns(2)
        .spacing([16.0, 2.0])
        .show(ui, |ui| {
            ui.label(tl("大きさ", "Size"));
            ui.label(format_bytes(s.bytes));
            ui.end_row();
            ui.label(tl("フレーム", "Frames"));
            let text = tr!("{} 件", "{}", s.frames);
            if s.bad > 0 || s.errors > 0 {
                ui.label(egui::RichText::new(tr!(
                    "{text}（壊れた・途切れたもの {}、復号できないもの {}）", "{text} (broken or cut off {}, undecodable {})",
                    s.bad, s.errors
                )).color(yellow))
                .on_hover_text(tl("CRC が合わないフレームは飛ばし、次の同期から読み直します。少数なら問題ありません", "Frames whose CRC doesn't match are skipped and reading resumes from the next sync. A few are no problem"));
            } else {
                ui.label(text);
            }
            ui.end_row();
            if let (Some(first), Some(last)) = (s.epochs.first(), s.epochs.last()) {
                ui.label(tl("観測", "Observations"));
                let rate = s.interval().filter(|&d| d > 0.0).map_or(String::new(), |d| format!("、{:.0} Hz", 1.0 / d));
                ui.label(tr!(
                    "{} 〜 {}（{:.0} 秒、{} エポック{rate}）", "{} to {} ({:.0} s, {} epochs{rate})",
                    local_time(first.utc),
                    local_time(last.utc),
                    last.utc - first.utc,
                    s.epochs.len()
                ));
                ui.end_row();
                ui.label(tl("最大の途切れ", "Largest gap"));
                let gap = s.max_gap();
                let long = s.interval().is_some_and(|d| gap > (5.0 * d).max(2.0));
                let text = egui::RichText::new(tr!("{gap:.1} 秒", "{gap:.1} s"));
                ui.label(if long { text.color(yellow) } else { text });
                ui.end_row();
                let n = s.epochs.len() as f64;
                let mean = |f: &dyn Fn(&RawEpoch) -> f64| s.epochs.iter().map(f).sum::<f64>() / n;
                ui.label(tl("平均の衛星数", "Mean satellites"));
                ui.label(tr!(
                    "{:.1}（{}）", "{:.1} ({})",
                    mean(&|e| e.total() as f64),
                    gnss::SYSTEMS
                        .iter()
                        .enumerate()
                        .map(|(k, name)| format!("{} {:.1}", crate::lang::of(name), mean(&|e| e.sats[k] as f64)))
                        .filter(|t| !t.ends_with(" 0.0"))
                        .collect::<Vec<_>>()
                        .join(tl("・", ", "))
                ));
                ui.end_row();
                ui.label(tl("平均の信号強度", "Mean signal strength"));
                ui.label(format!("{:.1} dB-Hz", mean(&|e| e.snr as f64)));
                ui.end_row();
            }
            if let Some(p) = positioning {
                ui.label(tl("測位の状態", "Solution status"));
                match p {
                    Ok(p) if !p.states.is_empty() => {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(tr!("{}：", "{}: ", p.source));
                            for (f, share) in Fix::ALL.iter().zip(p.shares()) {
                                if share > 0.0 {
                                    ui.colored_label(
                                        FIX_COLORS[*f as usize],
                                        format!("{} {:.1}%", crate::lang::of(f.label()), share * 100.0),
                                    )
                                    .on_hover_text(crate::lang::of(f.hint()));
                                }
                            }
                        });
                    }
                    Ok(p) => {
                        ui.weak(tr!("{}：解がありません", "{}: no solutions", p.source));
                    }
                    Err(e) => {
                        ui.weak(e);
                    }
                }
                ui.end_row();
            }
            if let Some((_, llh)) = s.position {
                ui.label(tl("基準局の位置", "Base station position"));
                ui.label(tr!(
                    "北緯 {:.8}°  東経 {:.8}°  楕円体高 {:.3} m", "Lat {:.8}°  Lon {:.8}°  Ellipsoidal height {:.3} m",
                    llh[0], llh[1], llh[2]
                ));
                ui.end_row();
            }
        });
    if let Some(range) = range
        && !s.epochs.is_empty()
    {
        let positioning = positioning
            .and_then(|p| p.as_ref().ok())
            .filter(|p| !p.states.is_empty());
        gnss_chart(ui, s, positioning, range, rover);
    }
    egui::CollapsingHeader::new(tr!(
        "メッセージの種類（{}）",
        "Message types ({})",
        s.messages.len()
    ))
    .id_salt(("gnss_messages", rover))
    .show(ui, |ui| {
        egui::Grid::new(("gnss_messages_grid", rover))
            .striped(true)
            .show(ui, |ui| {
                for (id, n) in &s.messages {
                    ui.monospace(id.to_string());
                    ui.label(crate::lang::of(gnss::message_name(rover, *id)));
                    ui.monospace(n.to_string());
                    ui.end_row();
                }
            });
    });
}

/// The epochs falling in a pixel column of the GNSS chart: sums of their values.
#[derive(Clone, Default)]
struct Column {
    sats: [f32; 6],
    snr: f32,
    slips: u32,
    epochs: u32,
    /// the first and last epoch (UTC s)
    first: f64,
    last: f64,
}

/// Over time (on the range shared by the rover and the base): the satellites by system, the mean
/// signal strength, the slips as a strip (darker for more) and, for the rover, the state of the
/// position solutions, with the gaps shaded; the values under the pointer.
fn gnss_chart(
    ui: &mut egui::Ui,
    s: &RawSummary,
    positioning: Option<&Positioning>,
    [t0, t1]: [f64; 2],
    rover: bool,
) {
    const SLIP: egui::Color32 = egui::Color32::from_rgb(255, 90, 220);
    let snr_color = ui.visuals().text_color().gamma_multiply(0.8);
    let width = ui.available_width().min(900.0);
    let height = if positioning.is_some() { 210.0 } else { 190.0 };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let inner = rect.shrink2(egui::vec2(0.0, 6.0));
    let left = rect.left() + 40.0;
    let right = rect.right() - 8.0;
    // bands: satellites, signal strength, slips (and the solutions)
    let sats_band = egui::Rect::from_x_y_ranges(left..=right, inner.top()..=inner.top() + 100.0);
    let snr_band = egui::Rect::from_x_y_ranges(
        left..=right,
        sats_band.bottom() + 10.0..=sats_band.bottom() + 55.0,
    );
    let slip_band = egui::Rect::from_x_y_ranges(
        left..=right,
        snr_band.bottom() + 8.0..=snr_band.bottom() + 20.0,
    );
    let fix_band =
        egui::Rect::from_x_y_ranges(left..=right, slip_band.bottom() + 6.0..=inner.bottom());
    let bottom = if positioning.is_some() {
        fix_band.bottom()
    } else {
        slip_band.bottom()
    };
    let span = (t1 - t0).max(1.0);
    let x = |t: f64| left + ((t - t0) / span) as f32 * (right - left);
    let weak = ui.visuals().weak_text_color();
    let grid = weak.gamma_multiply(0.3);
    let label = |y: f32, text: String, color: egui::Color32| {
        painter.text(
            egui::pos2(rect.left() + 4.0, y),
            egui::Align2::LEFT_CENTER,
            text,
            egui::FontId::proportional(12.0),
            color,
        );
    };
    let max_sats = s
        .epochs
        .iter()
        .map(|e| e.sats.iter().copied().max().unwrap_or(0))
        .max()
        .unwrap_or(1)
        .max(10) as f32;
    let y_sats = |n: f32| sats_band.bottom() - n / max_sats * sats_band.height();
    for n in (0..=max_sats as u16).step_by(5) {
        let y = y_sats(n as f32);
        painter.line_segment([egui::pos2(left, y), egui::pos2(right, y)], (1.0, grid));
        label(y, n.to_string(), weak);
    }
    let y_snr =
        |snr: f32| snr_band.bottom() - ((snr - 20.0) / 35.0).clamp(0.0, 1.0) * snr_band.height();
    for db in [20.0, 35.0, 50.0] {
        let y = y_snr(db);
        painter.line_segment([egui::pos2(left, y), egui::pos2(right, y)], (1.0, grid));
        label(y, format!("{db:.0}"), snr_color);
    }
    label(slip_band.center().y, tl("スリップ", "Slips").into(), SLIP);
    if positioning.is_some() {
        label(fix_band.center().y, tl("測位", "Fix").into(), weak);
    }
    // gaps: five usual intervals (at least 2 s) without an epoch
    let usual = s.interval().unwrap_or(1.0);
    let gap = (5.0 * usual).max(2.0);
    for w in s.epochs.windows(2) {
        if w[1].utc - w[0].utc > gap {
            painter.rect_filled(
                egui::Rect::from_x_y_ranges(x(w[0].utc)..=x(w[1].utc), sats_band.top()..=bottom),
                0.0,
                egui::Color32::from_rgba_unmultiplied(230, 190, 70, 40),
            );
        }
    }
    // per pixel column: the mean of each value, the slips summed; a gap breaks the lines
    let columns = (right - left).max(1.0) as usize;
    let mut cols: Vec<Option<Column>> = vec![None; columns];
    for e in &s.epochs {
        let c = (((e.utc - t0) / span) * columns as f64).clamp(0.0, columns as f64 - 1.0) as usize;
        let v = cols[c].get_or_insert(Column {
            first: e.utc,
            ..Default::default()
        });
        for k in 0..6 {
            v.sats[k] += e.sats[k] as f32;
        }
        v.snr += e.snr;
        v.slips += e.slips as u32;
        v.epochs += 1;
        v.last = e.utc;
    }
    let column_x = |c: usize| left + (c as f32 + 0.5) / columns as f32 * (right - left);
    // a line of the column means, broken where the epochs have a gap
    let polyline =
        |value: &dyn Fn(&Column) -> f32, y: &dyn Fn(f32) -> f32, color: egui::Color32| {
            let mut points = Vec::new();
            let mut last = f64::NEG_INFINITY;
            for (c, v) in cols.iter().enumerate() {
                let Some(v) = v else { continue };
                if v.first - last > gap && points.len() > 1 {
                    painter.add(egui::Shape::line(std::mem::take(&mut points), (1.2, color)));
                }
                if v.first - last > gap {
                    points.clear();
                }
                points.push(egui::pos2(column_x(c), y(value(v) / v.epochs as f32)));
                last = v.last;
            }
            if points.len() > 1 {
                painter.add(egui::Shape::line(points, (1.2, color)));
            }
        };
    for (k, &color) in SYSTEM_COLORS.iter().enumerate() {
        if s.epochs.iter().any(|e| e.sats[k] > 0) {
            polyline(&|v| v.sats[k], &y_sats, color);
        }
    }
    polyline(&|v| v.snr, &y_snr, snr_color);
    // slips per epoch, as a share of the satellites tracked
    let tracked =
        s.epochs.iter().map(|e| e.total() as f32).sum::<f32>() / s.epochs.len().max(1) as f32;
    for (c, v) in cols.iter().enumerate() {
        if let Some(v) = v
            && v.slips > 0
        {
            let share = (v.slips as f32 / v.epochs as f32 / tracked.max(1.0)).clamp(0.0, 1.0);
            let alpha = (60.0 + 195.0 * share.sqrt()) as u8;
            painter.rect_filled(
                egui::Rect::from_x_y_ranges(
                    column_x(c) - 0.5..=column_x(c) + 0.5,
                    slip_band.y_range(),
                ),
                0.0,
                egui::Color32::from_rgba_unmultiplied(SLIP.r(), SLIP.g(), SLIP.b(), alpha),
            );
        }
    }
    // the solutions: per pixel column, the state most of them have
    if let Some(p) = positioning {
        let mut counts = vec![[0u32; Fix::ALL.len()]; columns];
        for &(t, f) in &p.states {
            if t < t0 || t > t1 {
                continue;
            }
            let c = (((t - t0) / span) * columns as f64).clamp(0.0, columns as f64 - 1.0) as usize;
            counts[c][f as usize] += 1;
        }
        for (c, n) in counts.iter().enumerate() {
            if let Some(f) = (0..Fix::ALL.len())
                .filter(|&f| n[f] > 0)
                .max_by_key(|&f| (n[f], std::cmp::Reverse(f)))
            {
                painter.rect_filled(
                    egui::Rect::from_x_y_ranges(
                        column_x(c) - 0.5..=column_x(c) + 0.5,
                        fix_band.y_range(),
                    ),
                    0.0,
                    FIX_COLORS[f],
                );
            }
        }
    }
    if let Some(pos) = response.hover_pos()
        && pos.x >= left
        && pos.x <= right
    {
        let t = t0 + ((pos.x - left) / (right - left)) as f64 * span;
        let i = s
            .epochs
            .partition_point(|e| e.utc < t)
            .min(s.epochs.len() - 1);
        let e = &s.epochs[i];
        painter.line_segment(
            [
                egui::pos2(x(e.utc), sats_band.top()),
                egui::pos2(x(e.utc), bottom),
            ],
            (1.0, ui.visuals().text_color()),
        );
        response.on_hover_ui_at_pointer(|ui| {
            ui.label(local_time(e.utc));
            for (k, name) in gnss::SYSTEMS.iter().enumerate() {
                if e.sats[k] > 0 {
                    ui.colored_label(
                        SYSTEM_COLORS[k],
                        format!("{} {}", crate::lang::of(name), e.sats[k]),
                    );
                }
            }
            ui.label(tr!(
                "信号強度 {:.1} dB-Hz・信号 {}",
                "Signal strength {:.1} dB-Hz, signals {}",
                e.snr,
                e.signals
            ));
            ui.colored_label(
                SLIP,
                tr!("スリップした衛星 {}", "Slipped satellites {}", e.slips),
            );
            if let Some(p) = positioning {
                match p.at(e.utc) {
                    Some(f) => ui.colored_label(
                        FIX_COLORS[f as usize],
                        tr!("{}（{}）", "{} ({})", crate::lang::of(f.label()), p.source),
                    ),
                    None => ui.weak(tr!("解なし（{}）", "No solution ({})", p.source)),
                };
            }
        });
    }
    // the legend
    ui.horizontal_wrapped(|ui| {
        for (k, name) in gnss::SYSTEMS.iter().enumerate() {
            if s.epochs.iter().any(|e| e.sats[k] > 0) {
                ui.colored_label(SYSTEM_COLORS[k], format!("━ {}", crate::lang::of(name)));
            }
        }
        ui.colored_label(
            snr_color,
            tl(
                "━ 平均の信号強度（dB-Hz）",
                "━ Mean signal strength (dB-Hz)",
            ),
        );
        ui.colored_label(
            SLIP,
            tl("▮ スリップ（濃いほど多い）", "▮ Slips (darker: more)"),
        );
        ui.colored_label(
            egui::Color32::from_rgb(230, 190, 70),
            tl("■ 途切れ", "■ Gaps"),
        );
        if let Some(p) = positioning {
            ui.weak(tr!("測位（{}）:", "Solutions ({}):", p.source));
            // the states there are (the PPK ones apart only with PPK)
            for f in Fix::ALL {
                if p.states.iter().any(|s| s.1 == f) {
                    ui.colored_label(
                        FIX_COLORS[f as usize],
                        format!("■ {}", crate::lang::of(f.label())),
                    )
                    .on_hover_text(crate::lang::of(f.hint()));
                }
            }
        }
        if !rover {
            ui.weak(tl(
                "（横軸はローバーと共通）",
                "(same time axis as the rover)",
            ));
        }
    });
}

fn format_bytes(b: u64) -> String {
    match b {
        b if b >= 1 << 30 => format!("{:.2} GB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / (1u64 << 20) as f64),
        b if b >= 1 << 10 => format!("{:.1} KB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_files_match_their_patterns_whatever_the_case_and_separator() {
        assert!(matches("all_*.bag", "all_2026-10-03-13-08-34.bag"));
        assert!(!matches("all_*.bag", "info\\info_2026.bag"));
        assert!(matches("info/info_*.bag", "info\\info_2026-10-03.bag"));
        assert!(matches("BASE.BIN", "base.bin"));
        assert!(!matches("BASE.BIN", "BASE.BIN.bak"));
        assert_eq!(format_bytes(3 << 20), "3.0 MB");
    }
}

#[cfg(test)]
impl App {
    /// Opens session `k`'s recording in the raw data window and waits for it: its topics, or a
    /// GNSS message selected.
    pub(super) fn raw_test_open(&mut self, ctx: &egui::Context, k: usize, message: bool) {
        self.open_raw_view(k);
        self.raw_view.tab = RawTab::Topics;
        let cfg = self.config().unwrap();
        self.raw_view.file = raw_bags(&cfg.sessions[k].raw).first().cloned();
        let deadline = Instant::now() + std::time::Duration::from_secs(300);
        while self.raw_bag(ctx).is_none() {
            assert!(Instant::now() < deadline, "bag not loaded");
            std::thread::sleep(std::time::Duration::from_millis(20));
            self.poll_raw_bag(ctx);
            assert!(self.raw_view.error.is_none(), "{:?}", self.raw_view.error);
        }
        if message {
            let bag = self.raw_bag(ctx).unwrap();
            self.raw_view.topic = bag
                .topics
                .iter()
                .position(|t| t.name == "/rtk_agent/pvtsln")
                .unwrap();
            self.raw_view.row = Some(100);
            self.raw_view.tab = RawTab::Messages;
        }
    }
}

#[cfg(test)]
impl App {
    /// Opens session `k`'s GNSS raw files in the raw data window, read here.
    pub(super) fn raw_test_gnss(&mut self, k: usize) {
        self.open_raw_view(k);
        self.raw_view.tab = RawTab::Gnss;
        let raw = self.config().unwrap().sessions[k].raw.clone();
        let job = s20_io::job::Job::new(|_, _, _| {}, |_| {});
        let (rover, base) = gnss::inspect(&raw, &job).unwrap();
        let positioning = positioning(&self.config().unwrap(), k, &raw, &job);
        assert!(
            positioning.as_ref().is_ok_and(|p| !p.states.is_empty()),
            "{:?}",
            positioning.err()
        );
        self.raw_view.gnss = Some((
            k,
            Ok(Arc::new(GnssFiles {
                rover,
                base,
                positioning,
            })),
        ));
    }
}

#[cfg(test)]
mod real_data {
    use super::*;

    #[test]
    #[ignore = "reads the local multi_park recording; prints how long a bag takes to open"]
    fn open_the_recording_bag() {
        let work = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/multi_park");
        let sessions = RunConfig::project_sessions(&work).unwrap();
        for s in sessions {
            for path in raw_bags(&s.raw) {
                let started = Instant::now();
                let bag = RawBag::open(&path).unwrap();
                let n: usize = bag.topics.iter().map(|t| t.messages.len()).sum();
                eprintln!(
                    "{}: {} topics, {n} messages, {:.2} s",
                    path.display(),
                    bag.topics.len(),
                    started.elapsed().as_secs_f64()
                );
                for t in &bag.topics {
                    let m = &t.messages[t.messages.len() / 2];
                    let v = t.schema.decode(bag.bag.data(m));
                    eprintln!(
                        "  {} {} {} max gap {:.3} s: {}",
                        t.name,
                        t.msg_type,
                        t.messages.len(),
                        t.max_gap,
                        v.map(|v| rosmsg::brief(&v, &t.msg_type))
                            .unwrap_or_else(|e| format!("ERR {e:#}"))
                    );
                }
            }
        }
    }
}
