// A GUI program: Windows opens no console window for it.
#![windows_subsystem = "windows"]

#[macro_use]
mod lang;
mod app;
mod camera;
mod images;
mod log_view;
mod notice;
mod panics;
mod params_view;
mod quality;
mod render;
mod rosmsg;
mod scene;
mod stream;
mod window_place;

/// Smoke tests report on stderr; borrow the terminal's console.
#[cfg(windows)]
fn attach_parent_console() {
    const ATTACH_PARENT_PROCESS: u32 = u32::MAX;
    unsafe extern "system" {
        fn AttachConsole(process: u32) -> i32;
    }
    // SAFETY: plain Win32 call without pointers.
    unsafe {
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

fn main() -> eframe::Result<()> {
    panics::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    // Exercise exactly the viewer's streaming and point budget without creating a window.
    if let Some(work) = arg("--preview-bench") {
        #[cfg(windows)]
        attach_parent_console();
        let result = preview_bench(
            &work,
            arg("--preview-source").as_deref().unwrap_or("colored"),
            arg("--preview-trajectory")
                .as_deref()
                .unwrap_or("optimized"),
            arg("--preview-budget")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1_000_000),
        );
        match result {
            Ok(report) => {
                let json = serde_json::to_string_pretty(&report).unwrap();
                if let Some(path) = arg("--preview-report") {
                    std::fs::write(path, &json).expect("write preview benchmark report");
                }
                eprintln!("{json}");
            }
            Err(e) => {
                eprintln!("preview benchmark failed: {e:#}");
                std::process::exit(1);
            }
        }
        return Ok(());
    }
    let smoke = arg("--smoke-screenshot").map(|p| {
        let mut sm = app::Smoke::new(
            p.into(),
            arg("--work"),
            arg("--raw"),
            arg("--smoke-start").as_deref(),
        );
        sm.points = arg("--smoke-points").and_then(|v| v.parse().ok());
        sm.adaptive = arg("--smoke-adaptive").map(|v| v != "0");
        sm.zoom = arg("--smoke-zoom").and_then(|v| v.parse().ok());
        sm.metric = arg("--smoke-metric").and_then(|v| v.parse().ok());
        sm.goto_worst = args.iter().any(|a| a == "--smoke-goto-worst");
        sm.time = arg("--smoke-time").and_then(|v| v.parse().ok());
        sm.params = args.iter().any(|a| a == "--smoke-params");
        sm.ppk = args.iter().any(|a| a == "--smoke-ppk");
        sm.lang = arg("--smoke-lang").map(|v| {
            if v == "en" {
                lang::Lang::En
            } else {
                lang::Lang::Ja
            }
        });
        sm.view = arg("--smoke-view").map(|v| {
            if v == "scans" {
                app::PreviewSource::Scans
            } else if v == "uncolored" {
                app::PreviewSource::Uncolored
            } else {
                app::PreviewSource::Colored
            }
        });
        sm.scan_trajectory = arg("--smoke-trajectory").map(|v| {
            if v == "device" {
                scene::TrajectorySource::Device
            } else if v == "lio" {
                scene::TrajectorySource::Lio
            } else {
                scene::TrajectorySource::Optimized
            }
        });
        sm
    });
    #[cfg(windows)]
    if smoke.is_some() {
        attach_parent_console();
    }
    let mut gpu = eframe::egui_wgpu::WgpuConfiguration::default();
    // Use the Windows graphics API without probing Vulkan drivers on startup.
    if cfg!(target_os = "windows")
        && let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut gpu.wgpu_setup
    {
        setup.instance_descriptor.backends = wgpu::Backends::DX12.with_env();
    }
    eframe::run_native(
        "S20 SLAM",
        eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            wgpu_options: gpu,
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([1360., 860.])
                .with_min_inner_size([900., 560.]),
            persist_window: true,
            ..Default::default()
        },
        Box::new(move |cc| Ok(Box::new(app::App::new(cc, smoke)))),
    )
}

fn preview_bench(
    work: &str,
    source: &str,
    trajectory: &str,
    budget: usize,
) -> anyhow::Result<serde_json::Value> {
    use anyhow::{Context, ensure};
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};
    let work = std::path::PathBuf::from(work);
    let cfg =
        s20_pipeline::RunConfig::of_sessions(s20_pipeline::RunConfig::open_work(&work)?, work);
    let source = match source {
        "scans" => scene::PreviewSource::Scans,
        "uncolored" => scene::PreviewSource::Uncolored,
        _ => scene::PreviewSource::Colored,
    };
    let trajectory = if trajectory == "device" {
        scene::TrajectorySource::Device
    } else if trajectory == "lio" {
        scene::TrajectorySource::Lio
    } else {
        scene::TrajectorySource::Optimized
    };
    let budget = budget.max(stream::OVERVIEW_POINTS);
    let started = Instant::now();
    let mut scene = scene::load(&cfg, source, trajectory, 4, true, &AtomicBool::new(false))?;
    let load_seconds = started.elapsed().as_secs_f64();
    let total = scene.points();
    let mut camera = camera::Camera::default();
    camera.frame(&scene.corners());
    let mut views = Vec::new();
    for step in 0..4 {
        if step > 0 && !scene.traj.is_empty() {
            let k = (scene.traj.len() - 1) * step / 3;
            camera.target = scene.traj[k].1.to_array();
            camera.distance = 20.0;
        }
        let started = Instant::now();
        let allocation = scene.allocate(&camera, 900.0, budget, &|_| true);
        let stream = scene
            .streaming
            .as_mut()
            .context("preview is not streamed")?;
        let memory_budget = stream.view_budget(&allocation);
        loop {
            stream.update(&mut scene.nodes, &allocation, memory_budget, || {});
            ensure!(stream.error.is_none(), "{:?}", stream.error);
            if !stream.pending() {
                break;
            }
            ensure!(
                started.elapsed() < Duration::from_secs(120),
                "preview reader timeout"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let resident: usize = scene
            .nodes
            .iter()
            .flat_map(|n| &n.chunks)
            .map(|c| c.len())
            .sum();
        ensure!(
            resident <= memory_budget,
            "memory budget exceeded: {resident} > {memory_budget}"
        );
        views.push(serde_json::json!({ "view": step, "resident": resident,
            "seconds": started.elapsed().as_secs_f64(),
            "drawn_nodes": allocation.len(),
            "memory_budget": memory_budget,
            "drawn_points": allocation.iter().map(|(_, n)| n).sum::<usize>(),
            "loaded_nodes": scene.nodes.iter().filter(|n| !n.chunks.is_empty()).count() }));
    }
    Ok(
        serde_json::json!({ "total_points": total, "budget": budget, "load_seconds": load_seconds,
        "nodes": scene.nodes.len(), "views": views }),
    )
}
