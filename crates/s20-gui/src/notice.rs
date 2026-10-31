//! Short notices over the view: the outcome of an action, which the log alone would hide.
use crate::lang::tl;
use eframe::egui;
use egui_phosphor::regular as icon;
use std::time::{Duration, Instant};

/// At most this many are shown; a new one pushes the oldest out.
const MAX_SHOWN: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    /// done as asked
    Info,
    /// not done: the input was not right
    Warn,
    /// not done: something failed
    Error,
}

impl Level {
    fn icon(self) -> &'static str {
        match self {
            Level::Info => icon::CHECK_CIRCLE,
            Level::Warn => icon::WARNING,
            Level::Error => icon::WARNING_CIRCLE,
        }
    }

    fn color(self) -> egui::Color32 {
        match self {
            Level::Info => egui::Color32::from_rgb(110, 200, 120),
            Level::Warn => egui::Color32::from_rgb(230, 190, 70),
            Level::Error => egui::Color32::from_rgb(240, 110, 100),
        }
    }

    /// Errors stay longer: they may need reading.
    fn lifetime(self) -> Duration {
        Duration::from_secs(match self {
            Level::Info => 4,
            Level::Warn => 6,
            Level::Error => 12,
        })
    }
}

struct Notice {
    id: u64,
    level: Level,
    text: String,
    shown: Instant,
}

#[derive(Default)]
pub struct Notices {
    items: Vec<Notice>,
    next: u64,
}

impl Notices {
    pub fn push(&mut self, level: Level, text: String) {
        // the same message again (a button pressed twice): shown once, from now
        self.items.retain(|n| n.text != text);
        self.items.push(Notice {
            id: self.next,
            level,
            text,
            shown: Instant::now(),
        });
        self.next += 1;
        if self.items.len() > MAX_SHOWN {
            self.items.remove(0);
        }
    }

    #[cfg(test)]
    pub fn texts(&self) -> Vec<(Level, &str)> {
        self.items
            .iter()
            .map(|n| (n.level, n.text.as_str()))
            .collect()
    }

    /// Draws the notices stacked up from the bottom right corner of `area`, newest at the bottom.
    /// A notice under the pointer stays; a click closes it.
    pub fn show(&mut self, ctx: &egui::Context, area: egui::Rect) {
        let now = Instant::now();
        let pointer = ctx.input(|i| i.pointer.hover_pos());
        // above the axes in the corner of the view
        let mut bottom = area.max.y - 64.0;
        let mut close = None;
        let mut next_expiry: Option<Duration> = None;
        for n in self.items.iter_mut().rev() {
            let r = egui::Area::new(egui::Id::new(("notice", n.id)))
                .order(egui::Order::Foreground)
                .pivot(egui::Align2::RIGHT_BOTTOM)
                .fixed_pos(egui::pos2(area.max.x - 12.0, bottom))
                .interactable(true)
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style())
                        .stroke(egui::Stroke::new(1.0, n.level.color().gamma_multiply(0.6)))
                        .show(ui, |ui| {
                            ui.set_max_width(420.0);
                            ui.horizontal(|ui| {
                                ui.colored_label(n.level.color(), n.level.icon());
                                ui.add(egui::Label::new(n.text.as_str()).wrap());
                            });
                        })
                })
                .response;
            if r.hovered() || pointer.is_some_and(|p| r.rect.contains(p)) {
                // read at leisure: the time starts again when the pointer leaves
                n.shown = now;
            }
            if r.interact(egui::Sense::click())
                .on_hover_text(tl(
                    "クリックで閉じる（処理ログにも残っています）",
                    "Click to close (it stays in the processing log)",
                ))
                .clicked()
            {
                close = Some(n.id);
            }
            bottom = r.rect.min.y - 6.0;
            let left = n.level.lifetime().saturating_sub(now - n.shown);
            next_expiry = Some(next_expiry.map_or(left, |e| e.min(left)));
        }
        self.items
            .retain(|n| Some(n.id) != close && now - n.shown < n.level.lifetime());
        if let Some(d) = next_expiry {
            ctx.request_repaint_after(d.max(Duration::from_millis(50)));
        }
    }
}
