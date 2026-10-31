//! The window's place kept over a restart, with its size from before it was maximized.
//!
//! eframe stores the window as it is on closing: a window closed maximized comes back with the
//! maximized size as its normal size, so leaving maximized keeps it filling the screen. This keeps
//! the last place and size seen not maximized and writes them over eframe's (which it writes just
//! before `App::save`).
use eframe::egui;
use serde::{Deserialize, Serialize};

/// eframe's key of the window
const KEY: &str = "window";

/// The fields of `egui_winit::WindowSettings` (private there), which eframe reads back.
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
#[serde(default)]
struct Stored {
    /// position of the window content in physical pixels
    inner_position_pixels: Option<egui::Pos2>,
    /// position of the window frame in physical pixels
    outer_position_pixels: Option<egui::Pos2>,
    fullscreen: bool,
    maximized: bool,
    /// size of the window content in points
    inner_size_points: Option<egui::Vec2>,
}

#[derive(Default)]
pub struct WindowPlace {
    /// the place last seen not maximized, minimized or full screen
    normal: Stored,
    maximized: bool,
    fullscreen: bool,
}

impl WindowPlace {
    pub fn load(storage: Option<&dyn eframe::Storage>) -> Self {
        let normal: Stored = storage
            .and_then(|s| eframe::get_value(s, KEY))
            .unwrap_or_default();
        Self {
            maximized: normal.maximized,
            fullscreen: normal.fullscreen,
            normal,
        }
    }

    /// Called every frame: the state of the window, and its place while it is a normal one.
    pub fn track(&mut self, ctx: &egui::Context) {
        let ppp = ctx.pixels_per_point();
        ctx.input(|i| {
            let v = i.viewport();
            if v.minimized == Some(true) {
                return;
            }
            self.maximized = v.maximized.unwrap_or(self.maximized);
            self.fullscreen = v.fullscreen.unwrap_or(self.fullscreen);
            if self.maximized || self.fullscreen {
                return;
            }
            if let (Some(inner), Some(outer)) = (v.inner_rect, v.outer_rect) {
                let px = |p: egui::Pos2| egui::pos2(p.x * ppp, p.y * ppp);
                self.normal.inner_position_pixels = Some(px(inner.min));
                self.normal.outer_position_pixels = Some(px(outer.min));
                self.normal.inner_size_points = Some(inner.size());
            }
        });
    }

    pub fn save(&self, storage: &mut dyn eframe::Storage) {
        if self.normal.inner_size_points.is_none() {
            return;
        }
        let stored = Stored {
            maximized: self.maximized,
            fullscreen: self.fullscreen,
            ..self.normal
        };
        eframe::set_value(storage, KEY, &stored);
    }
}
