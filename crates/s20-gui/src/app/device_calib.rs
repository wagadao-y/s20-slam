//! The device's camera calibration: chosen work folders of one device, their cameras solved
//! together in the background (`s20_pipeline::device::calibrate`), kept by the maker calibration's
//! digest and used for that device's recordings from then on.
use super::*;
use crate::lang::tl;
use s20_pipeline::device::{Info, Store, calibrate, unusable};
use s20_pipeline::runlog::PathMask;

/// The folder the device calibrations are kept in (beside the GUI settings).
pub(super) fn store() -> Store {
    let base = if cfg!(test) {
        std::env::temp_dir().join(format!("s20_device_calibrations_{}", std::process::id()))
    } else {
        eframe::storage_dir("S20 SLAM")
            .unwrap_or_else(std::env::temp_dir)
            .join("device_calibrations")
    };
    Store { dir: base }
}

enum CalibEvent {
    Progress(f32),
    Log(String),
    Done(anyhow::Result<(String, serde_json::Value)>),
}

struct CalibJob {
    rx: mpsc::Receiver<CalibEvent>,
    cancel: Arc<AtomicBool>,
    progress: f32,
    last: String,
    /// the maker calibration and the work folders it is made from
    maker: PathBuf,
    works: Vec<PathBuf>,
    /// the log file beside the device calibrations (`logs/`)
    log_file: Option<Arc<s20_pipeline::runlog::LogFile>>,
}

#[derive(Default)]
pub(super) struct CalibWindow {
    /// work folders offered, and whether chosen
    picks: Vec<(PathBuf, bool)>,
    job: Option<CalibJob>,
    message: Option<(bool, String)>,
}

/// A work folder's run, as the GUI would make it (its own settings), if it is a work folder.
fn work_config(work: &Path) -> Option<RunConfig> {
    let mut c = RunConfig::of_sessions(RunConfig::project_sessions(work)?, work.to_path_buf());
    if let Some(Ok(w)) = WorkSettings::load(work) {
        c.laz = w.laz;
    }
    Some(c)
}

impl App {
    /// Whether the device calibration is running.
    pub(super) fn device_calib_running(&self) -> bool {
        self.device_calib.as_ref().is_some_and(|w| w.job.is_some())
    }

    pub(super) fn cancel_device_calib(&self) {
        if let Some(job) = self.device_calib.as_ref().and_then(|w| w.job.as_ref()) {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    pub(super) fn open_device_calib(&mut self) {
        let current = self.config().map(|c| c.calib());
        let digest = current.as_deref().map(s20_pipeline::file_digest);
        let mut picks: Vec<(PathBuf, bool)> = Vec::new();
        for w in
            std::iter::once(PathBuf::from(self.s.work.trim())).chain(self.s.recent.iter().cloned())
        {
            if w.as_os_str().is_empty() || picks.iter().any(|(p, _)| *p == w) {
                continue;
            }
            let same = work_config(&w).is_some_and(|c| {
                unusable(&c).is_none() && Some(s20_pipeline::file_digest(&c.calib())) == digest
            });
            picks.push((w, same));
        }
        self.device_calib = Some(CalibWindow {
            picks,
            ..Default::default()
        });
    }

    fn poll_device_calib(&mut self) {
        let Some(win) = &mut self.device_calib else {
            return;
        };
        let Some(job) = &mut win.job else { return };
        let mut done = None;
        while let Ok(e) = job.rx.try_recv() {
            match e {
                CalibEvent::Progress(f) => job.progress = f,
                CalibEvent::Log(l) => {
                    job.last = l.clone();
                    self.log.push(l);
                }
                CalibEvent::Done(r) => done = Some(r),
            }
        }
        let Some(r) = done else { return };
        let job = win.job.take().expect("running");
        if let Some(f) = &job.log_file {
            f.line(&match &r {
                Ok(_) => tl("---- 完了", "---- Done").to_string(),
                Err(e) if is_cancelled(e) => tl("---- 中断しました", "---- Cancelled").to_string(),
                Err(e) => tr!("---- エラー: {e:#}", "---- Error: {e:#}"),
            });
        }
        win.message = Some(match r {
            Ok((yaml, report)) => {
                let info = Info {
                    made_at: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |d| d.as_secs()),
                    recordings: job.works.clone(),
                    report,
                };
                match self.device.save(&job.maker, &yaml, &info) {
                    Ok(f) => {
                        self.log.push(PathMask::home().apply(&tr!(
                            "機体の校正を保存しました: {}",
                            "Saved the device calibration: {}",
                            f.display()
                        )));
                        // the processed recordings of this device among the recent work folders
                        let digest = s20_pipeline::file_digest(&job.maker);
                        let mut works: Vec<PathBuf> =
                            std::iter::once(PathBuf::from(self.s.work.trim()))
                                .chain(self.s.recent.iter().cloned())
                                .filter(|w| !w.as_os_str().is_empty())
                                .collect();
                        works.dedup();
                        let n = works
                            .iter()
                            .filter_map(|w| work_config(w))
                            .filter(|c| {
                                !c.sessions.is_empty()
                                    && c.done(Step::Export)
                                    && s20_pipeline::file_digest(&c.calib()) == digest
                            })
                            .count();
                        let text = if n > 0 {
                            tr!(
                                "保存しました。この機体の処理済みの撮影 {n} 件（最近開いた作業フォルダー）は、\
                                 「軌跡・画像姿勢の出力」からやり直すとこの校正が使われます",
                                "Saved. The {n} processed recordings of this device (in the recent work folders) use this calibration when redone from \"Trajectory and image pose export\""
                            )
                        } else {
                            tl("保存しました。この機体の撮影では、これから「軌跡・画像姿勢の出力」から\
                             やり直すと、この校正が使われます", "Saved. Recordings of this device use this calibration when redone from \"Trajectory and image pose export\"")
                                .to_string()
                        };
                        self.notices.push(Level::Info, text.clone());
                        (true, text)
                    }
                    Err(e) => (
                        false,
                        tr!("保存できませんでした: {e:#}", "Could not save: {e:#}"),
                    ),
                }
            }
            Err(e) if is_cancelled(&e) => (false, tl("中止しました", "Cancelled").to_string()),
            Err(e) => (
                false,
                tr!("求められませんでした: {e:#}", "Could not compute it: {e:#}"),
            ),
        });
        self.refresh_steps();
    }

    fn start_device_calib(&mut self, ctx: &egui::Context) {
        let Some(win) = &mut self.device_calib else {
            return;
        };
        let works: Vec<PathBuf> = win
            .picks
            .iter()
            .filter(|(_, on)| *on)
            .map(|(p, _)| p.clone())
            .collect();
        let Some(cfgs) = works
            .iter()
            .map(|w| work_config(w))
            .collect::<Option<Vec<_>>>()
        else {
            win.message = Some((
                false,
                tl(
                    "作業フォルダーとして開けないものがあります",
                    "Some of them can't be opened as work folders",
                )
                .to_string(),
            ));
            return;
        };
        let maker = cfgs[0].calib();
        // the recordings by the names of their work and raw folders
        let mask = cfgs.iter().fold(PathMask::home(), |m, c| {
            let name = c.work.file_name().unwrap_or_default().to_string_lossy();
            c.sessions
                .iter()
                .fold(m.with(&c.work, &format!("<work:{name}>")), |m, s| {
                    m.with(&s.raw, &format!("<raw:{}>", s.name))
                })
        });
        let log_file = s20_pipeline::runlog::LogFile::create(&self.device.dir.join("logs"), mask)
            .ok()
            .map(Arc::new);
        if let Some(f) = &log_file {
            f.line(&tr!(
                "s20 {}  機体の校正",
                "s20 {}  device calibration",
                env!("CARGO_PKG_VERSION")
            ));
            f.line(&tr!("開始: {}", "Start: {}", s20_pipeline::runlog::stamp()));
            f.line(&tr!(
                "メーカーの校正: {} ({})",
                "Vendor calibration: {} ({})",
                maker.display(),
                s20_pipeline::file_digest(&maker)
            ));
            for w in &works {
                f.line(&tr!("撮影: {}", "Recording: {}", w.display()));
            }
            f.line("");
        }
        let (tx, rx) = mpsc::channel();
        let (tp, tl) = (tx.clone(), tx.clone());
        let (cp, cl) = (ctx.clone(), ctx.clone());
        let t0 = Instant::now();
        let lf = log_file.clone();
        let job = Job::new(
            move |_, done, total| {
                if total > 0 {
                    let _ = tp.send(CalibEvent::Progress(done as f32 / total as f32));
                    cp.request_repaint();
                }
            },
            move |s| {
                let mut line = format!("[{:7.1}s] {s}", t0.elapsed().as_secs_f64());
                if let Some(f) = &lf {
                    line = f.line(&line);
                }
                let _ = tl.send(CalibEvent::Log(line));
                cl.request_repaint();
            },
        );
        let cancel = job.cancel.clone();
        let ctx2 = ctx.clone();
        std::thread::spawn(move || {
            let r =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| calibrate(&cfgs, &job)))
                    .unwrap_or_else(|_| {
                        Err(anyhow::anyhow!(crate::lang::tl(
                            "処理中に内部エラー（panic）が起きました",
                            "An internal error (panic) occurred during processing"
                        )))
                    });
            let _ = tx.send(CalibEvent::Done(r));
            ctx2.request_repaint();
        });
        self.log.push(tr!(
            "---- 機体の校正（{} 撮影）",
            "---- Device calibration ({} recordings)",
            works.len()
        ));
        if let Some(f) = &log_file {
            self.log.push(PathMask::home().apply(&tr!(
                "ログファイル: {}",
                "Log file: {}",
                f.path.display()
            )));
        }
        win.message = None;
        win.job = Some(CalibJob {
            rx,
            cancel,
            progress: 0.0,
            last: String::new(),
            maker,
            works,
            log_file,
        });
    }

    pub(super) fn device_calib_window(&mut self, ctx: &egui::Context) {
        self.poll_device_calib();
        if self.device_calib.is_none() {
            return;
        }
        let current = self.config().map(|c| c.calib());
        let digest = current.as_deref().map(s20_pipeline::file_digest);
        let info = current.as_deref().and_then(|m| self.device.info(m));
        let mut open = true;
        let mut start = false;
        let mut remove = false;
        let mut add = None;
        let pipeline_idle = self.job.is_none();
        egui::Window::new(tr!("{} 機体の校正", "{} Device calibration", icon::CAMERA))
            .open(&mut open)
            .default_width(560.0)
            .show(ctx, |ui| {
                let win = self.device_calib.as_mut().expect("open");
                ui.add(
                    egui::Label::new(
                        tl("同じ機体の処理済みの撮影をいくつか選び、カメラの取り付け・魚眼・主点を画像ごとの姿勢と\
                         いっしょに求めます。屋外と、近くを写す屋内の撮影を両方含めてください（取り付け位置の\
                         数 mm は近くを写す撮影でしか決まりません）。1 撮影あたり 2〜4 分かかります。", "Choose some processed recordings of the same device; the camera mounting, fisheye and principal point are solved together with each image's pose. Include both outdoor and indoor recordings that see things up close (the last few mm of the mounting are only determined by close-up recordings). Each recording takes 2-4 minutes."),
                    )
                    .wrap(),
                );
                ui.add_space(4.0);
                match (&current, &info) {
                    (Some(_), Some(i)) => {
                        ui.label(tr!(
                            "{} この機体の校正: あり（{} 撮影から）", "{} This device's calibration: yes (from {} recordings)",
                            icon::CHECK_CIRCLE,
                            i.recordings.len()
                        ));
                    }
                    (Some(_), None) => {
                        ui.label(tr!(
                            "{} この機体の校正: なし（撮影ごとに、その撮影からカメラを求めています）", "{} This device's calibration: none (the camera is solved from each recording itself)",
                            icon::CIRCLE
                        ));
                    }
                    (None, _) => {
                        ui.label(tl("作業フォルダーを開くと、その機体の校正の有無が表示されます", "Opening a work folder shows whether its device has a calibration"));
                    }
                }
                ui.separator();
                let running = win.job.is_some();
                egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                    for (w, on) in &mut win.picks {
                        let cfg = work_config(w);
                        let why = match &cfg {
                            None => Some(tl("作業フォルダーではありません", "Not a work folder").to_string()),
                            Some(c) => unusable(c).map(|w| crate::lang::of(w).to_string()).or_else(|| {
                                (digest.is_some() && Some(s20_pipeline::file_digest(&c.calib())) != digest)
                                    .then(|| tl("開いている撮影と別の機体です", "A different device from the open recording").to_string())
                            }),
                        };
                        ui.horizontal(|ui| {
                            ui.add_enabled_ui(!running && why.is_none(), |ui| ui.checkbox(on, ""));
                            let name = w.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                            ui.label(name).on_hover_text(w.display().to_string());
                            if let Some(why) = &why {
                                *on = false;
                                ui.label(egui::RichText::new(why).weak());
                            }
                        });
                    }
                });
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!running, egui::Button::new(tr!("{} 作業フォルダーを追加…", "{} Add work folder…", icon::FOLDER_PLUS)))
                        .clicked()
                        && let Some(p) = rfd::FileDialog::new().set_title(tl("処理済みの作業フォルダー", "Processed work folder")).pick_folder()
                    {
                        add = Some(p);
                    }
                });
                ui.separator();
                let chosen = win.picks.iter().filter(|(_, on)| *on).count();
                if let Some(job) = &win.job {
                    ui.add(egui::ProgressBar::new(job.progress).show_percentage());
                    ui.label(egui::RichText::new(&job.last).weak());
                    if ui.button(tl("中止", "Cancel")).clicked() {
                        job.cancel.store(true, Ordering::Relaxed);
                    }
                } else {
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                chosen > 0 && pipeline_idle,
                                egui::Button::new(tr!("{} {chosen} 撮影から校正を求める", "{} Compute the calibration from {chosen} recordings", icon::PLAY)),
                            )
                            .on_disabled_hover_text(tl("撮影を選んでください（処理の実行中は使えません）", "Choose recordings (not available while processing)"))
                            .clicked()
                        {
                            start = true;
                        }
                        if ui
                            .button(tr!("{} 保存先を開く", "{} Open the save location", icon::FOLDER_OPEN))
                            .on_hover_text(tr!("機体の校正（.yaml）とログ（logs）の保存先: {}", "Where the device calibrations (.yaml) and logs (logs) are saved: {}", self.device.dir.display()))
                            .clicked()
                        {
                            let _ = std::fs::create_dir_all(&self.device.dir);
                            let _ = std::process::Command::new("explorer").arg(&self.device.dir).spawn();
                        }
                        if info.is_some()
                            && ui
                                .add_enabled(pipeline_idle, egui::Button::new(tr!("{} この機体の校正を削除", "{} Delete this device's calibration", icon::TRASH)))
                                .clicked()
                        {
                            remove = true;
                        }
                    });
                    if chosen == 1 {
                        ui.label(
                            egui::RichText::new(tl("1 撮影だけでも求められますが、撮影ごとに少し揺れます。数撮影を推奨します", "One recording is enough, but the result varies a little between recordings. Several are recommended"))
                                .weak(),
                        );
                    }
                }
                if let Some((ok, m)) = &win.message {
                    let col = if *ok { egui::Color32::from_rgb(110, 200, 120) } else { egui::Color32::from_rgb(240, 110, 100) };
                    ui.add(egui::Label::new(egui::RichText::new(m).color(col)).wrap());
                }
            });
        if let Some(p) = add
            && let Some(win) = &mut self.device_calib
            && !win.picks.iter().any(|(w, _)| *w == p)
        {
            win.picks.push((p, true));
        }
        if start {
            self.start_device_calib(ctx);
        }
        if remove && let Some(m) = &current {
            let r = self.device.remove(m);
            if let Some(win) = &mut self.device_calib {
                win.message = Some(match r {
                    Ok(()) => (
                        true,
                        tl(
                            "削除しました。撮影ごとにその撮影からカメラを求める処理に戻ります",
                            "Deleted. The camera is solved from each recording itself again",
                        )
                        .to_string(),
                    ),
                    Err(e) => (
                        false,
                        tr!("削除できませんでした: {e:#}", "Could not delete: {e:#}"),
                    ),
                });
            }
            self.refresh_steps();
        }
        if !open {
            self.cancel_device_calib();
            self.device_calib = None;
        }
    }
}
