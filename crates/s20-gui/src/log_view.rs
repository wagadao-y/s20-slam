//! Virtualised log labels with their original wrapping and variable heights.
use eframe::egui;
use std::{ops::Range, sync::Arc};

const MAX_LOGS: usize = 2000;

#[derive(Default)]
pub struct LogView {
    first: usize,
    format: Option<(f32, f32, egui::TextWrapMode)>,
    labels: Vec<Arc<egui::Galley>>,
    /// Top of each label, with one final entry for the bottom including spacing.
    offsets: Vec<f32>,
    spacing: f32,
}

impl LogView {
    /// Logs are appended by the app. Only new labels or a changed layout need text shaping.
    fn sync(&mut self, ui: &egui::Ui, log: &[String]) {
        let first = log.len().saturating_sub(MAX_LOGS);
        let format = (
            ui.available_width(),
            ui.ctx().pixels_per_point(),
            ui.wrap_mode(),
        );
        let mut changed = false;
        if self.format != Some(format)
            || first < self.first
            || first > self.first + self.labels.len()
            || log.len() < self.first + self.labels.len()
        {
            self.labels.clear();
            self.first = first;
            self.format = Some(format);
            changed = true;
        }
        if first > self.first {
            self.labels.drain(..first - self.first);
            self.first = first;
            changed = true;
        }
        for l in &log[self.first + self.labels.len()..] {
            self.labels.push(
                egui::WidgetText::from(egui::RichText::new(l).monospace().size(11.5)).into_galley(
                    ui,
                    None,
                    format.0,
                    egui::FontSelection::Default,
                ),
            );
            changed = true;
        }
        let spacing = ui.spacing().item_spacing.y;
        if changed || self.offsets.is_empty() || self.spacing != spacing {
            self.spacing = spacing;
            self.offsets.clear();
            self.offsets.push(0.0);
            for label in &self.labels {
                self.offsets
                    .push(self.offsets.last().unwrap() + label.size().y + spacing);
            }
        }
    }

    fn visible(&self, viewport: egui::Rect) -> Range<usize> {
        let first = self
            .offsets
            .partition_point(|&y| y <= viewport.min.y)
            .saturating_sub(1)
            .min(self.labels.len());
        let end = self
            .offsets
            .partition_point(|&y| y < viewport.max.y)
            .min(self.labels.len());
        first..end.max(first)
    }

    pub fn show(&mut self, ui: &mut egui::Ui, log: &[String]) {
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show_viewport(ui, |ui, viewport| {
                self.sync(ui, log);
                let height = (self.offsets.last().copied().unwrap_or(0.0) - self.spacing).max(0.0);
                ui.set_height(height);
                // egui drops a text selection when a label holding one of its ends is not laid
                // out, so all labels are shown while there is a selection, as before virtualising.
                let range = if ui
                    .ctx()
                    .plugin::<egui::text_selection::LabelSelectionState>()
                    .lock()
                    .has_selection()
                {
                    0..self.labels.len()
                } else {
                    self.visible(viewport)
                };
                if range.is_empty() {
                    return;
                }
                let rect = egui::Rect::from_x_y_ranges(
                    ui.max_rect().x_range(),
                    ui.max_rect().top() + self.offsets[range.start]
                        ..=ui.max_rect().top() + self.offsets[range.end],
                );
                ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                    ui.skip_ahead_auto_ids(range.start);
                    for label in &self.labels[range] {
                        ui.label(label.clone());
                    }
                });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_wrapping_and_reuses_labels_when_appending_or_resizing() {
        let ctx = egui::Context::default();
        let mut view = LogView::default();
        let mut log: Vec<String> = (0..MAX_LOGS)
            .map(|i| format!("{i}: a long optimisation log entry with several words to wrap"))
            .collect();
        let mut run = |width: f32, log: &[String], check: &mut dyn FnMut(&LogView)| {
            ctx.run_ui(egui::RawInput::default(), |ui| {
                ui.set_width(width);
                view.sync(ui, log);
                check(&view);
            })
            .drop_without_applying_deltas();
        };
        let mut first = None;
        let mut second = None;
        run(180.0, &log, &mut |v| {
            assert_eq!(v.labels.len(), MAX_LOGS);
            assert!(v.labels[0].rows.len() > 1);
            first = Some(v.labels[0].clone());
            second = Some(v.labels[1].clone());
        });
        run(180.0, &log, &mut |v| {
            assert!(Arc::ptr_eq(first.as_ref().unwrap(), &v.labels[0]))
        });
        log.push("new\nmultiline log".into());
        run(180.0, &log, &mut |v| {
            assert_eq!(v.first, 1);
            assert_eq!(v.labels.len(), MAX_LOGS);
            assert!(Arc::ptr_eq(second.as_ref().unwrap(), &v.labels[0]));
            assert_eq!(v.labels.last().unwrap().text(), log.last().unwrap());
        });
        run(600.0, &log, &mut |v| {
            assert!(!Arc::ptr_eq(second.as_ref().unwrap(), &v.labels[0]));
            assert_eq!(v.labels[0].rows.len(), 1);
        });
        run(600.0, &[], &mut |v| {
            assert!(v.labels.is_empty());
            assert!(
                v.visible(egui::Rect::from_min_max(
                    egui::Pos2::ZERO,
                    egui::pos2(100., 100.)
                ))
                .is_empty()
            );
        });
    }

    #[test]
    fn virtualised_logs_paint_like_the_original_labels() {
        let mut log: Vec<String> = (0..2000)
            .map(|i| {
                if i % 7 == 0 {
                    format!(
                        "{i}: {}\nsecond line",
                        "a long optimisation log entry ".repeat(3)
                    )
                } else {
                    format!("{i}: short entry")
                }
            })
            .collect();
        let old_ctx = egui::Context::default();
        let new_ctx = egui::Context::default();
        let mut view = LogView::default();
        for width in [240.0, 600.0] {
            // Check initial display, bottom following and a user scroll back into history.
            for frame in 0..6 {
                if frame == 4 {
                    log.push("appended while viewing history\nsecond line".into());
                }
                let input = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width, 160.0),
                    )),
                    events: if frame == 3 {
                        vec![
                            egui::Event::PointerMoved(egui::pos2(80.0, 80.0)),
                            egui::Event::MouseWheel {
                                unit: egui::MouseWheelUnit::Point,
                                phase: egui::TouchPhase::Move,
                                delta: egui::vec2(0.0, 400.0),
                                modifiers: Default::default(),
                            },
                        ]
                    } else {
                        Vec::new()
                    },
                    ..Default::default()
                };
                let old = old_ctx.run_ui(input.clone(), |ui| {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            for l in &log[log.len().saturating_sub(MAX_LOGS)..] {
                                ui.label(egui::RichText::new(l).monospace().size(11.5));
                            }
                        });
                });
                let new = new_ctx.run_ui(input, |ui| view.show(ui, &log));
                let text = |out: &egui::FullOutput| {
                    out.shapes
                        .iter()
                        .filter_map(|s| {
                            if let egui::epaint::Shape::Text(t) = &s.shape {
                                Some((t.pos, t.galley.text().to_owned(), t.galley.size()))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                };
                let old_text = text(&old);
                let new_text = text(&new);
                assert_eq!(
                    old_text.len(),
                    new_text.len(),
                    "width {width}, frame {frame}"
                );
                for (a, b) in old_text.iter().zip(&new_text) {
                    assert_eq!(a.1, b.1);
                    assert!((a.0 - b.0).length() < 0.1, "{a:?} != {b:?}");
                    assert_eq!(a.2, b.2);
                }
                old.drop_without_applying_deltas();
                new.drop_without_applying_deltas();
            }
        }
    }
}
