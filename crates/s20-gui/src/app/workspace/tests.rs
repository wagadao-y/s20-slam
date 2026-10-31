use super::*;

fn ui_frame(
    ctx: &egui::Context,
    app: &mut App,
    size: egui::Vec2,
    events: Vec<egui::Event>,
) -> (egui::FullOutput, egui::Rect) {
    let mut viewport = egui::Rect::NOTHING;
    let output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            events,
            ..Default::default()
        },
        |ui| {
            app.workspace_panels(ui);
            egui::CentralPanel::no_frame().show(ui, |ui| {
                viewport = ui.max_rect();
            });
            app.workspace_windows(ui.ctx());
        },
    );
    (output, viewport)
}

fn has_text(output: &egui::FullOutput, text: &str) -> bool {
    !texts(output, text).is_empty()
}

fn text_rect(output: &egui::FullOutput, text: &str) -> egui::Rect {
    texts(output, text)
        .into_iter()
        .min_by(|a, b| a.top().total_cmp(&b.top()))
        .unwrap_or_else(|| panic!("UI text missing: {text}"))
}

fn texts(output: &egui::FullOutput, text: &str) -> Vec<egui::Rect> {
    fn find(shape: &egui::Shape, text: &str, found: &mut Vec<egui::Rect>) {
        match shape {
            egui::Shape::Text(t) if t.galley.text() == text => {
                found.push(t.galley.rect.translate(t.pos.to_vec2()))
            }
            egui::Shape::Vec(shapes) => {
                for s in shapes {
                    find(s, text, found);
                }
            }
            _ => {}
        }
    }
    let mut found = Vec::new();
    for s in &output.shapes {
        find(&s.shape, text, &mut found);
    }
    found
}

fn click(pos: egui::Pos2) -> Vec<egui::Event> {
    vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Default::default(),
        },
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        },
    ]
}

/// A work folder for a test, removed after it.
struct TempWork(PathBuf);
impl Drop for TempWork {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// An app with a work folder open (no sessions yet): the workspace, not the start screen.
fn opened(name: &str) -> (App, TempWork) {
    let work = std::env::temp_dir().join(format!("s20_gui_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    RunConfig::with_sessions(Vec::new(), work.clone())
        .save_sessions()
        .unwrap();
    let mut app = crate::app::tests::app();
    app.s.work = work.display().to_string();
    (app, TempWork(work))
}

fn context() -> egui::Context {
    let ctx = egui::Context::default();
    install_fonts(&ctx);
    ctx.style_mut_of(egui::Theme::Dark, |s| s.animation_time = 0.0);
    ctx
}

#[test]
fn default_workspace_gives_the_view_space_and_log_can_be_opened_and_closed() {
    let ctx = context();
    let (mut app, _work) = opened("t0");
    let size = egui::vec2(1360.0, 860.0);
    ui_frame(&ctx, &mut app, size, vec![])
        .0
        .drop_without_applying_deltas();
    let (out, initial) = ui_frame(&ctx, &mut app, size, vec![]);
    assert!(initial.width() >= 1000.0, "{initial:?}");
    assert!(initial.height() >= 600.0, "{initial:?}");
    for text in ["ファイル", "処理", "表示", "ツール", "ヘルプ", "工程"] {
        assert!(text_rect(&out, text).is_positive());
    }
    let pos = text_rect(&out, &format!("{} ログ (0)", icon::TERMINAL_WINDOW)).center();
    out.drop_without_applying_deltas();
    ui_frame(&ctx, &mut app, size, click(pos))
        .0
        .drop_without_applying_deltas();
    assert!(app.s.show_log);
    let (out, with_log) = ui_frame(&ctx, &mut app, size, vec![]);
    assert!(initial.height() - with_log.height() > 100.0);
    let pos = text_rect(&out, "閉じる").center();
    out.drop_without_applying_deltas();
    ui_frame(&ctx, &mut app, size, click(pos))
        .0
        .drop_without_applying_deltas();
    assert!(!app.s.show_log);
    let (out, restored) = ui_frame(&ctx, &mut app, size, vec![]);
    assert!((restored.height() - initial.height()).abs() < 2.0);
    out.drop_without_applying_deltas();
}

#[test]
fn contextual_tools_do_not_change_the_preview_and_fit_the_minimum_window() {
    let ctx = context();
    let (mut app, _work) = opened("t1");
    app.s.view = PreviewSource::Scans;
    let size = egui::vec2(900.0, 600.0);
    ui_frame(&ctx, &mut app, size, vec![])
        .0
        .drop_without_applying_deltas();
    let (out, wide) = ui_frame(&ctx, &mut app, size, vec![]);
    let pos = text_rect(&out, &InspectorTab::Trajectory.text()).center();
    out.drop_without_applying_deltas();
    ui_frame(&ctx, &mut app, size, click(pos))
        .0
        .drop_without_applying_deltas();
    assert!(app.s.inspector_open);
    assert_eq!(app.s.inspector_tab, InspectorTab::Trajectory);
    assert_eq!(app.s.view, PreviewSource::Scans);
    let (out, narrow) = ui_frame(&ctx, &mut app, size, vec![]);
    assert!(
        narrow.width() >= 240.0 && narrow.height() >= 300.0,
        "{narrow:?}"
    );
    // the inspector takes at least its minimum width
    assert!(
        wide.width() - narrow.width() >= 289.0,
        "{wide:?} {narrow:?}"
    );
    let pos = text_rect(&out, &InspectorTab::Trajectory.text()).center();
    out.drop_without_applying_deltas();
    ui_frame(&ctx, &mut app, size, click(pos))
        .0
        .drop_without_applying_deltas();
    assert!(!app.s.inspector_open);
}

#[test]
fn processing_menu_opens_settings_without_starting_a_job_and_layout_preferences_round_trip() {
    let ctx = context();
    let mut app = crate::app::tests::app();
    let size = egui::vec2(1360.0, 860.0);
    ui_frame(&ctx, &mut app, size, vec![])
        .0
        .drop_without_applying_deltas();
    let (out, _) = ui_frame(&ctx, &mut app, size, vec![]);
    let pos = text_rect(&out, "処理").center();
    out.drop_without_applying_deltas();
    ui_frame(&ctx, &mut app, size, click(pos))
        .0
        .drop_without_applying_deltas();
    let (out, _) = ui_frame(&ctx, &mut app, size, vec![]);
    let pos = text_rect(&out, &format!("{} 処理設定…", icon::SLIDERS_HORIZONTAL)).center();
    out.drop_without_applying_deltas();
    ui_frame(&ctx, &mut app, size, click(pos))
        .0
        .drop_without_applying_deltas();
    assert!(app.processing_settings_open);
    assert!(app.job.is_none());
    app.s.show_project = false;
    app.s.show_log = true;
    app.open_inspector(InspectorTab::Images);
    let settings: Settings = serde_json::from_str(&serde_json::to_string(&app.s).unwrap()).unwrap();
    assert!(!settings.show_project && settings.show_log && settings.inspector_open);
    assert_eq!(settings.inspector_tab, InspectorTab::Images);
}

#[test]
fn corrections_are_added_from_the_trajectory_tab_by_kind() {
    let ctx = context();
    let (mut app, _work) = opened("t2");
    app.marker = Some((115.0, DVec3::new(1.5, 0.0, 0.0)));
    app.open_inspector(InspectorTab::Trajectory);
    let size = egui::vec2(1360.0, 860.0);
    ui_frame(&ctx, &mut app, size, vec![])
        .0
        .drop_without_applying_deltas();
    let (out, viewport) = ui_frame(&ctx, &mut app, size, vec![]);
    // no buttons that start from the selected time any more
    assert!(texts(&out, "ループ: A").is_empty());
    // one add button a kind, the first for the GNSS exclusions (GNSS is used here)
    let add = format!("{} 追加", icon::PLUS);
    assert_eq!(texts(&out, &add).len(), 3);
    let action = text_rect(&out, &add);
    assert!(action.left() >= viewport.right());
    let pos = action.center();
    out.drop_without_applying_deltas();
    ui_frame(&ctx, &mut app, size, click(pos))
        .0
        .drop_without_applying_deltas();
    assert_eq!(
        app.placing,
        Some(super::traj_edit::Placing {
            kind: super::traj_edit::EditKind::Exclude,
            first: None
        })
    );
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    struct WakeThread(std::thread::Thread);
    impl std::task::Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

fn snapshot(
    rs: &eframe::egui_wgpu::RenderState,
    ctx: &egui::Context,
    output: egui::FullOutput,
    size: [u32; 2],
    path: &Path,
) {
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = rs.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("workspace screenshot"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let jobs = ctx.tessellate(output.shapes.clone(), output.pixels_per_point);
    let screen = eframe::egui_wgpu::ScreenDescriptor {
        size_in_pixels: size,
        pixels_per_point: output.pixels_per_point,
    };
    let mut encoder = rs.device.create_command_encoder(&Default::default());
    let buffers =
        rs.renderer
            .write()
            .update_buffers(&rs.device, &rs.queue, &mut encoder, &jobs, &screen);
    {
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("workspace UI"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            })
            .forget_lifetime();
        rs.renderer.read().render(&mut pass, &jobs, &screen);
    }
    let row = (size[0] * 4).div_ceil(256) * 256;
    let buffer = rs.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (row * size[1]) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
    );
    rs.queue
        .submit(buffers.into_iter().chain([encoder.finish()]));
    buffer.map_async(wgpu::MapMode::Read, .., |_| {});
    rs.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buffer.get_mapped_range(..).unwrap();
    let bytes: Vec<u8> = data
        .chunks(row as usize)
        .flat_map(|line| line[..size[0] as usize * 4].iter().copied())
        .collect();
    image::save_buffer(path, &bytes, size[0], size[1], image::ColorType::Rgba8).unwrap();
    output.drop_without_applying_deltas();
}

#[test]
#[ignore = "requires a GPU and the local multi_park work folder; writes UI screenshots to out"]
fn render_workspace_screenshots() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let work = root.join("out/multi_park").canonicalize().unwrap();
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = if cfg!(windows) {
        wgpu::Backends::DX12
    } else {
        wgpu::Backends::PRIMARY
    };
    eprintln!("creating GPU instance");
    let instance = wgpu::Instance::new(desc);
    let adapter =
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    eprintln!("GPU ready");
    let renderer = eframe::egui_wgpu::Renderer::new(
        &device,
        wgpu::TextureFormat::Rgba8Unorm,
        Default::default(),
    );
    let rs = eframe::egui_wgpu::RenderState {
        instance,
        available_adapters: vec![adapter.clone()],
        adapter,
        device,
        queue,
        renderer: Arc::new(egui::epaint::mutex::RwLock::new(renderer)),
        target_format: wgpu::TextureFormat::Rgba8Unorm,
        surface_config: eframe::egui_wgpu::SurfaceConfig::LOW_LATENCY,
    };
    for (name, size, tab, settings) in [
        ("default", [1360, 860], None, false),
        ("display", [1360, 860], Some(InspectorTab::Display), false),
        ("compact", [900, 600], Some(InspectorTab::Display), false),
        (
            "trajectory",
            [1360, 860],
            Some(InspectorTab::Trajectory),
            false,
        ),
        ("images", [1360, 860], Some(InspectorTab::Images), false),
        ("settings", [1360, 860], None, true),
        ("folded", [1360, 860], None, false),
        (
            "color-height",
            [1360, 860],
            Some(InspectorTab::Display),
            false,
        ),
        (
            "color-session",
            [1360, 860],
            Some(InspectorTab::Display),
            false,
        ),
        (
            "output-reflectivity",
            [1360, 860],
            Some(InspectorTab::Display),
            false,
        ),
        (
            "output-rgb",
            [1360, 860],
            Some(InspectorTab::Display),
            false,
        ),
        (
            "output-close",
            [1360, 860],
            Some(InspectorTab::Display),
            false,
        ),
        (
            "output-uncolored",
            [1360, 860],
            Some(InspectorTab::Display),
            false,
        ),
        (
            "color-height-clipped",
            [1360, 860],
            Some(InspectorTab::Display),
            false,
        ),
        (
            "revisions",
            [1360, 860],
            Some(InspectorTab::Trajectory),
            false,
        ),
        ("start", [1360, 860], None, false),
        (
            "scan-around",
            [1360, 860],
            Some(InspectorTab::Display),
            false,
        ),
        ("raw-topics", [1360, 860], None, false),
        ("raw-messages", [1360, 860], None, false),
        ("raw-gnss", [1360, 860], None, false),
    ] {
        // S20_SCREENSHOTS=output: only the shots whose names have that in them
        if let Ok(only) = std::env::var("S20_SCREENSHOTS")
            && !name.contains(&only)
        {
            continue;
        }
        let ctx = context();
        let mut app = crate::app::tests::app();
        app.s.work = work.display().to_string();
        app.s.view = if name == "output-uncolored" {
            PreviewSource::Uncolored
        } else if name.starts_with("output") {
            PreviewSource::Colored
        } else {
            PreviewSource::Scans
        };
        app.s.point_color_scans = match name {
            "color-height" | "color-height-clipped" => PointColor::Height,
            "color-session" => PointColor::Session,
            _ => PointColor::Reflectivity,
        };
        if name == "output-reflectivity" {
            app.s.point_color_output = PointColor::Reflectivity;
        }
        app.s.preview_points = 500_000;
        if name == "scan-around" {
            app.s.scan_range = ScanRange::Around;
            app.s.scan_window = 5.0;
        }
        if name == "start" {
            // nothing open: the start screen with the recent folders
            app.s.work.clear();
            app.s.recent = vec![
                work.clone(),
                root.join("out/room2").canonicalize().unwrap(),
                root.join("out/no_such_folder"),
            ];
        }
        eprintln!("loading {name}");
        if let Some(cfg) = app.config() {
            app.scene = scene::load(
                &cfg,
                app.s.view,
                TrajectorySource::Optimized,
                4,
                true,
                &AtomicBool::new(false),
            )
            .unwrap();
        }
        eprintln!("scene loaded {name}");
        app.renderer = Some(PointRenderer::new(rs.device.clone(), rs.queue.clone()));
        app.refresh_steps();
        app.frame_all();
        if let Some(tab) = tab {
            app.open_inspector(tab);
        }
        app.processing_settings_open = settings;
        if name == "output-close" {
            // the detail: 15 m from the middle of the trajectory
            let (_, p) = app.scene.traj[app.scene.traj.len() / 2];
            app.camera.target = p.to_array();
            app.camera.distance = 15.0;
            app.s.preview_points = 3_000_000;
        }
        if name == "revisions" {
            // in memory only: the work folder is not written
            let one = Edits {
                loops: vec![(At::new(10.0), At::new(900.0))],
                ..Default::default()
            };
            let revision =
                |id: &str, name: &str, by_run, edits: &Edits| s20_pipeline::EditsRevision {
                    id: id.into(),
                    name: name.into(),
                    saved_at: 1_792_000_000,
                    by_run,
                    edits: edits.clone(),
                };
            app.revisions = vec![
                revision("1", "保存 1", false, &Edits::default()),
                revision("2", "最適化の実行時", true, &one),
            ];
            app.edits = one.clone();
            app.edits.rtk_exclude.push((At::new(100.0), At::new(120.0)));
            app.revisions_open = true;
        }
        if name == "color-height-clipped"
            && let Some([lo, hi]) = app.scene.z_range
        {
            // the top half cut off, by the slider on the view
            app.height_clip[1] = Some((lo + hi) / 2.0);
        }
        if name == "folded" {
            app.s.show_project = false;
            app.error = Some("軌跡の最適化に失敗しました".into());
        }
        if matches!(tab, Some(InspectorTab::Trajectory | InspectorTab::Images))
            || name == "scan-around"
        {
            let (t, p) = app.scene.traj[app.scene.traj.len() / 2];
            app.marker = Some((t, p));
        }
        if name == "raw-gnss" {
            app.raw_test_gnss(1);
        } else if name.starts_with("raw") {
            app.raw_test_open(&ctx, 1, name == "raw-messages");
        }
        for pass in 0..200 {
            let output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(size[0] as f32, size[1] as f32),
                    )),
                    ..Default::default()
                },
                |ui| {
                    app.workspace_panels(ui);
                    if app.project().is_some() {
                        egui::CentralPanel::no_frame().show(ui, |ui| app.viewport(ui, &rs));
                    } else {
                        egui::CentralPanel::default().show(ui, |ui| app.welcome(ui));
                    }
                    app.workspace_windows(ui.ctx());
                    app.revisions_window(ui.ctx());
                    app.raw_window(ui.ctx());
                },
            );
            for (id, deltas) in &output.textures_delta.set {
                for delta in deltas {
                    rs.renderer
                        .write()
                        .update_texture(&rs.device, &rs.queue, *id, delta);
                }
            }
            let settled = pass > 4
                && app.preview_drawn
                && app.scene.streaming.as_ref().is_none_or(|s| s.idle());
            if settled || pass == 199 {
                eprintln!("snapshot {name} pass {pass} drawn {}", app.drawn_points);
                snapshot(
                    &rs,
                    &ctx,
                    output,
                    size,
                    &root.join(format!("out/ui-{name}.png")),
                );
                break;
            }
            output.drop_without_applying_deltas();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

#[test]
fn run_menu_names_the_range_that_will_run_and_the_shown_result_tells_when_it_is_old() {
    let mut app = crate::app::tests::app();
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
    let label = |s, only| run_label(s, app.plan(s, only));
    assert_eq!(label(Step::Lio, true), "この工程だけ実行");
    assert_eq!(label(Step::Lio, false), "ここから最後まで実行");
    assert_eq!(
        label(Step::Cloud, true),
        "「オドメトリ（LIO）」からこの工程まで実行"
    );
    assert_eq!(
        label(Step::Cloud, false),
        "「オドメトリ（LIO）」から最後まで実行"
    );
    assert_eq!(run_choices(Step::Images), &[true]);

    app.stale_why.insert(Step::Lio, s20_pipeline::Stale::Params);
    app.scene.source = None;
    assert_eq!(app.shown_stale(), None);
    // the optimised scans depend on the LIO
    app.scene.source = Some(PreviewSource::Scans);
    app.scene.trajectory_source = TrajectorySource::Optimized;
    assert_eq!(
        app.shown_stale(),
        Some((Step::Lio, Some(s20_pipeline::Stale::Params)))
    );
    app.steps.insert(Step::Lio, StepState::Done);
    assert_eq!(app.shown_stale(), None);
    // an old trajectory/image pose output does not make the cloud old
    app.steps.insert(Step::Export, StepState::Stale);
    app.scene.source = Some(PreviewSource::Colored);
    assert_eq!(app.shown_stale(), None);
}

#[test]
fn view_labels_move_off_one_another() {
    let ctx = context();
    let mut placed = Vec::new();
    let mut output = ctx.run_ui(Default::default(), |ui| {
        let painter = ui.painter().clone();
        let mut labels = ViewLabels::default();
        let at = egui::pos2(300.0, 300.0);
        for k in 0..3 {
            let text = format!("S{k} 始点");
            labels.add(&painter, at, text, None, egui::Color32::WHITE, false, false);
        }
        placed = labels.placed.iter().map(|p| p.0).collect();
    });
    output.textures_delta.clear();
    assert_eq!(placed.len(), 3);
    for (i, a) in placed.iter().enumerate() {
        for b in &placed[i + 1..] {
            assert!(!a.intersects(*b), "{a:?} {b:?}");
        }
    }
}

#[test]
fn one_preview_choice_sets_the_cloud_and_trajectory_and_names_what_is_shown() {
    let mut app = crate::app::tests::app();
    for p in Preview::ALL {
        app.set_preview(p);
        assert_eq!(app.preview_choice(), p);
    }
    // the output keeps the scans' trajectory for the next time
    app.set_preview(Preview::ScansLio);
    app.set_preview(Preview::Output);
    assert_eq!(app.s.scan_trajectory, TrajectorySource::Lio);
    // settings saved before the single choice read the same
    app.s.view = PreviewSource::Scans;
    app.s.scan_trajectory = TrajectorySource::Optimized;
    assert_eq!(app.preview_choice(), Preview::ScansOpt);
    // shown as loaded
    app.scene.source = Some(PreviewSource::Scans);
    app.scene.trajectory_source = TrajectorySource::Device;
    assert_eq!(app.source_label(), "スキャン点群（機体）");
    app.scene.source = Some(PreviewSource::Colored);
    assert_eq!(app.source_label(), "出力点群（着色済み）");
    // the uncoloured cloud chosen: no colour, said so
    app.set_preview(Preview::OutputUncolored);
    app.scene.source = Some(PreviewSource::Uncolored);
    assert!(
        app.point_color_unavailable(PointColor::Rgb)
            .unwrap()
            .contains("着色済み")
    );
}

#[test]
fn only_previews_of_finished_steps_are_chosen_and_the_nearest_replaces_one_that_is_not() {
    let work = std::env::temp_dir().join(format!("s20_gui_preview_done_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    let _keep = TempWork(work.clone());
    let cfg = RunConfig::with_sessions(vec!["raw0".into()], work.clone());
    cfg.save_sessions().unwrap();
    let mut app = crate::app::tests::app();
    app.s.work = work.display().to_string();
    app.set_preview(Preview::Output);
    assert_eq!(app.preview_missing(Preview::Output), Some(Step::Colorize));
    assert_eq!(app.nearest_preview(), None);
    std::fs::create_dir_all(cfg.runs()).unwrap();
    std::fs::create_dir_all(cfg.output()).unwrap();
    std::fs::write(cfg.opt_file(), "").unwrap();
    std::fs::write(cfg.uncolored(), "").unwrap();
    // the uncoloured cloud alone does not make the coloured one
    assert_eq!(app.preview_missing(Preview::Output), Some(Step::Colorize));
    assert_eq!(app.nearest_preview(), Some(Preview::OutputUncolored));
    // a cloud generation that did not finish
    std::fs::write(
        cfg.record_file(Step::Cloud),
        r#"{"done": false, "params": null}"#,
    )
    .unwrap();
    assert_eq!(
        app.preview_missing(Preview::OutputUncolored),
        Some(Step::Cloud)
    );
    assert_eq!(app.nearest_preview(), Some(Preview::ScansOpt));
    app.load_scene(&context());
    assert_eq!(app.preview_choice(), Preview::ScansOpt);
}

#[test]
fn project_panel_folds_to_a_strip_that_opens_it_again() {
    let ctx = context();
    let (mut app, _work) = opened("t3");
    let size = egui::vec2(1360.0, 860.0);
    ui_frame(&ctx, &mut app, size, vec![])
        .0
        .drop_without_applying_deltas();
    let (out, open) = ui_frame(&ctx, &mut app, size, vec![]);
    let pos = text_rect(&out, icon::CARET_DOUBLE_LEFT).center();
    out.drop_without_applying_deltas();
    ui_frame(&ctx, &mut app, size, click(pos))
        .0
        .drop_without_applying_deltas();
    assert!(!app.s.show_project);
    let (out, folded) = ui_frame(&ctx, &mut app, size, vec![]);
    out.drop_without_applying_deltas();
    // the strip stays, narrow
    assert!(folded.left() > 10.0 && folded.left() < 40.0, "{folded:?}");
    assert!(folded.width() - open.width() > 200.0);
    ui_frame(&ctx, &mut app, size, click(egui::pos2(10.0, 300.0)))
        .0
        .drop_without_applying_deltas();
    assert!(app.s.show_project);
}

#[test]
fn saved_quality_and_edits_tabs_open_the_trajectory_tab() {
    for old in ["\"Quality\"", "\"Edits\""] {
        let tab: InspectorTab = serde_json::from_str(old).unwrap();
        assert_eq!(tab, InspectorTab::Trajectory);
    }
    assert_eq!(
        serde_json::to_string(&InspectorTab::Trajectory).unwrap(),
        "\"Edits\""
    );
}

#[test]
fn point_colors_are_kept_per_cloud_and_fall_back_where_they_cannot_show() {
    let mut app = crate::app::tests::app();
    app.scene.source = Some(PreviewSource::Scans);
    // the scans have no image colours: grey by reflectivity
    app.s.point_color_scans = PointColor::Rgb;
    assert_eq!(app.point_color(), PointColor::Reflectivity);
    assert_eq!(app.coloring(), Coloring::Reflectivity);
    app.s.point_color_scans = PointColor::Plain;
    assert_eq!(app.coloring(), Coloring::Plain);
    // the output cloud keeps its own choice; uncoloured, it falls back to the height
    app.scene.source = Some(PreviewSource::Colored);
    assert_eq!(app.point_color(), PointColor::Rgb);
    app.scene.source = Some(PreviewSource::Uncolored);
    app.scene.z_range = Some([0.0, 5.0]);
    assert_eq!(app.coloring(), Coloring::Height([0.0, 5.0]));
    // no sessions in the output cloud
    app.s.point_color_output = PointColor::Session;
    assert!(app.point_color_unavailable(PointColor::Session).is_some());
}

#[test]
fn ctrl_z_undoes_a_correction_ctrl_s_saves_it_and_a_run_keeps_its_corrections() {
    let work = std::env::temp_dir().join(format!("s20_gui_undo_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    let cfg = RunConfig::with_sessions(vec![PathBuf::from("raw/a")], work.clone());
    cfg.save_sessions().unwrap();
    let ctx = context();
    let mut app = crate::app::tests::app();
    app.s.work = work.display().to_string();
    app.reload_edits();
    assert!(!app.edits_unsaved());
    // a loop added: written at once, unsaved, undoable
    app.edits.loops.push((At::new(1.0), At::new(9.0)));
    app.save_edits();
    assert_eq!(cfg.edits().loops.len(), 1);
    assert!(app.edits_unsaved());
    let key = |key, modifiers: egui::Modifiers| {
        vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }]
    };
    let size = egui::vec2(1360.0, 860.0);
    let frame = |app: &mut App, events| {
        ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
                events,
                ..Default::default()
            },
            |ui| app.edit_shortcuts(ui.ctx()),
        )
        .drop_without_applying_deltas();
    };
    frame(&mut app, key(egui::Key::Z, egui::Modifiers::COMMAND));
    assert!(app.edits.loops.is_empty());
    assert!(cfg.edits().loops.is_empty());
    frame(&mut app, key(egui::Key::Y, egui::Modifiers::COMMAND));
    assert_eq!(cfg.edits().loops.len(), 1);
    // Ctrl+Shift+Z redoes too, without undoing first
    frame(&mut app, key(egui::Key::Z, egui::Modifiers::COMMAND));
    frame(
        &mut app,
        key(
            egui::Key::Z,
            egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
        ),
    );
    assert_eq!(app.edits.loops.len(), 1);
    // saved: a revision, nothing unsaved; saving again adds none
    frame(&mut app, key(egui::Key::S, egui::Modifiers::COMMAND));
    assert!(!app.edits_unsaved());
    frame(&mut app, key(egui::Key::S, egui::Modifiers::COMMAND));
    assert_eq!(cfg.revisions().len(), 1);
    assert_eq!(cfg.revisions()[0].name, "保存 1");
    // a run of the optimisation keeps changed corrections, and only changed ones
    app.edits.loops.clear();
    app.save_edits();
    app.save_revision(true);
    app.save_revision(true);
    let revisions = cfg.revisions();
    assert_eq!(revisions.len(), 2);
    assert!(revisions[1].by_run && revisions[1].edits.loops.is_empty());
    // the corrections changed elsewhere: the history starts over
    cfg.save_edits(&Edits::default()).unwrap();
    app.edits.loops.push((At::new(2.0), At::new(8.0)));
    app.save_edits();
    std::fs::write(cfg.edits_file(), "{}").unwrap();
    app.reload_edits();
    assert!(!app.edit_history.can_undo());
    std::fs::remove_dir_all(&work).unwrap();
}

#[test]
fn the_start_screen_opens_a_recent_work_folder_and_closing_goes_back_to_it() {
    let ctx = context();
    let (_, work) = opened("start");
    let mut app = crate::app::tests::app();
    let missing = work.0.with_file_name("s20_gui_missing_folder");
    app.s.recent = vec![work.0.clone(), missing];
    let size = egui::vec2(1360.0, 860.0);
    let frame = |app: &mut App, events| {
        ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
                events,
                ..Default::default()
            },
            |ui| {
                app.workspace_panels(ui);
                egui::CentralPanel::default().show(ui, |ui| app.welcome(ui));
            },
        )
    };
    frame(&mut app, vec![]).drop_without_applying_deltas();
    let out = frame(&mut app, vec![]);
    // no tools without a work folder
    for text in ["プレビュー:", "工程"] {
        assert!(!has_text(&out, text), "{text} shown");
    }
    let name = format!("{}  {}", icon::FOLDER, folder_label(&work.0));
    let pos = text_rect(&out, &name).center();
    out.drop_without_applying_deltas();
    frame(&mut app, click(pos)).drop_without_applying_deltas();
    assert_eq!(PathBuf::from(&app.s.work), work.0);
    assert!(app.project().is_some());
    app.close_work();
    assert!(app.project().is_none());
    assert!(app.s.work.is_empty());
    // the recent folders stay to choose from
    assert_eq!(app.s.recent[0], work.0);
}

#[test]
fn scans_are_drawn_around_or_up_to_the_marked_time() {
    let mut app = crate::app::tests::app();
    let node = |t0: f64| scene::LodNode {
        center: DVec3::ZERO,
        radius: 1.0,
        sample_nn: 0.01,
        total: 1,
        chunks: Vec::new(),
        session: None,
        time: Some([t0, t0 + 1.0]),
    };
    let cloud = || scene::LodNode {
        time: None,
        ..node(0.0)
    };
    let shown = |app: &App| -> Vec<bool> {
        [node(0.0), node(10.0), node(20.0), cloud()]
            .iter()
            .map(|n| app.time_visible(n))
            .collect()
    };
    app.s.scan_range = ScanRange::Around;
    app.s.scan_window = 2.0;
    // no time marked: all
    assert_eq!(shown(&app), [true; 4]);
    app.marker = Some((12.5, DVec3::ZERO));
    // [10, 11] reaches 12.5 - 2; the finished cloud has no times
    assert_eq!(shown(&app), [false, true, false, true]);
    app.s.scan_range = ScanRange::UpTo;
    assert_eq!(shown(&app), [true, true, false, true]);
    app.s.scan_range = ScanRange::All;
    assert_eq!(shown(&app), [true; 4]);
}

#[test]
fn processing_settings_are_kept_by_work_folder_and_a_new_one_starts_from_the_last() {
    let (mut app, outdoors) = opened("settings_out");
    app.s.set_work_settings(WorkSettings {
        outdoor: true,
        params: ProcessParams::preset(true),
        rtk: false,
        ppk: true,
        laz: true,
        images: true,
        refine: true,
    });
    app.save_work_settings();
    let saved = WorkSettings::load(&outdoors.0).unwrap().unwrap();
    assert!(saved.outdoor && saved.ppk && saved.images);

    // a work folder without settings takes those used last, and keeps its own once changed
    let (_, indoors) = opened("settings_in");
    app.s.work = indoors.0.display().to_string();
    app.load_work_settings();
    assert!(app.s.outdoor && app.s.ppk);
    app.s.set_work_settings(WorkSettings::default());
    app.save_work_settings();

    app.s.work = outdoors.0.display().to_string();
    app.load_work_settings();
    assert!(app.s.outdoor && app.s.ppk && !app.s.rtk);
    app.s.work = indoors.0.display().to_string();
    app.load_work_settings();
    assert!(!app.s.outdoor && !app.s.ppk && !app.s.rtk);
    assert_eq!(app.s.params, Some(ProcessParams::preset(false)));
}

#[test]
fn the_gnss_is_chosen_in_the_project_panel_and_the_rest_in_the_settings_window() {
    let ctx = context();
    let (mut app, _work) = opened("quick_settings");
    app.s.set_gnss_mode(GnssMode::Rtk);
    let size = egui::vec2(1360.0, 860.0);
    ui_frame(&ctx, &mut app, size, vec![])
        .0
        .drop_without_applying_deltas();
    let (out, _) = ui_frame(&ctx, &mut app, size, vec![]);
    assert!(has_text(&out, GnssMode::Rtk.label()));
    let pos = text_rect(
        &out,
        &format!("{}  その他の設定…", icon::SLIDERS_HORIZONTAL),
    )
    .center();
    out.drop_without_applying_deltas();
    ui_frame(&ctx, &mut app, size, click(pos))
        .0
        .drop_without_applying_deltas();
    assert!(app.processing_settings_open);
}

#[test]
fn height_clip_slider_cuts_from_its_ends_and_lets_go_at_them() {
    let ctx = context();
    let mut app = crate::app::tests::app();
    app.scene.z_range = Some([-2.0, 8.0]);
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800., 600.));
    // top 24, bottom 500: 47.6 pixels a metre
    let (x, top, bottom) = (rect.right() - 22., 24., 500.);
    let frame = |app: &mut App, events: Vec<egui::Event>| {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(rect),
                events,
                ..Default::default()
            },
            |ui| app.height_clip_ui(ui, rect),
        );
        output.textures_delta.clear();
        output
    };
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: Default::default(),
    };
    let drag = |app: &mut App, from: egui::Pos2, to: egui::Pos2| {
        frame(app, vec![egui::Event::PointerMoved(from)]);
        frame(app, vec![button(from, true)]);
        frame(app, vec![egui::Event::PointerMoved(from.lerp(to, 0.5))]);
        frame(app, vec![egui::Event::PointerMoved(to)]);
        frame(app, vec![button(to, false)])
    };
    assert_eq!(app.height_clip(), [f64::NEG_INFINITY, f64::INFINITY]);
    // the top handle (above the height it cuts at) down to 3 m: the points above are cut, the
    // height shown by the handle
    let y3 = bottom - 5. / 10. * (bottom - top);
    let output = drag(&mut app, egui::pos2(x, top - 4.), egui::pos2(x, y3 - 4.));
    let [low, high] = app.height_clip();
    assert_eq!(low, f64::NEG_INFINITY);
    assert!((high - 3.0).abs() < 0.05, "{high}");
    assert!(has_text(&output, &format!("{high:.2} m")));
    assert!(app.height_kept(2.9) && !app.height_kept(3.1));
    // the bottom handle stops short of the top one
    drag(&mut app, egui::pos2(x, bottom + 4.), egui::pos2(x, top));
    let [low, high] = app.height_clip();
    assert!((high - low - 0.05).abs() < 1e-9, "{low} {high}");
    // just under where they meet, the bottom handle; back past its end: not cut there any more
    drag(
        &mut app,
        egui::pos2(x, y3 + 6.),
        egui::pos2(x, bottom + 20.),
    );
    assert!((app.height_clip()[1] - high).abs() < 1e-9);
    assert_eq!(app.height_clip()[0], f64::NEG_INFINITY);
    // a range that no longer reaches a cut (as after a reload) cuts nothing
    app.scene.z_range = Some([-2.0, 2.5]);
    assert_eq!(app.height_clip(), [f64::NEG_INFINITY, f64::INFINITY]);
}
