//! Undo and redo of the trajectory corrections, and their saved revisions.
//!
//! Every change of the corrections is written to the work folder at once (`s20_edits.json`, the
//! working state the optimisation uses); it becomes a revision only when the user saves (Ctrl+S)
//! or when the optimisation runs with it. Undo and redo replace the working state with an earlier
//! one, kept in memory for this session.
use super::*;
use crate::lang::tl;
use s20_pipeline::EditsRevision;

/// The corrections before (undo) and after (redo) the working state.
#[derive(Default)]
pub(super) struct EditHistory {
    undo: Vec<Edits>,
    redo: Vec<Edits>,
    /// the corrections as last written to the work folder
    written: Edits,
}

impl EditHistory {
    const LIMIT: usize = 200;

    /// Starts over from corrections read from the work folder (no undo past them).
    pub(super) fn reset(&mut self, written: Edits) {
        self.undo.clear();
        self.redo.clear();
        self.written = written;
    }

    /// Records corrections about to be written: those written before become undoable. False
    /// when nothing changed.
    pub(super) fn record(&mut self, now: &Edits) -> bool {
        if *now == self.written {
            return false;
        }
        if self.undo.len() == Self::LIMIT {
            self.undo.remove(0);
        }
        self.undo
            .push(std::mem::replace(&mut self.written, now.clone()));
        self.redo.clear();
        true
    }

    /// Names session `old` as `new` in every state, as the work folder's were.
    pub(super) fn rename_session(&mut self, old: &str, new: &str) {
        for e in self
            .undo
            .iter_mut()
            .chain(self.redo.iter_mut())
            .chain(std::iter::once(&mut self.written))
        {
            e.rename_session(old, new);
        }
    }

    pub(super) fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub(super) fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// The corrections to go back to; the written ones become redoable.
    fn undo(&mut self) -> Option<Edits> {
        let state = self.undo.pop()?;
        self.redo
            .push(std::mem::replace(&mut self.written, state.clone()));
        Some(state)
    }

    /// The corrections to go forward to; the written ones become undoable.
    fn redo(&mut self) -> Option<Edits> {
        let state = self.redo.pop()?;
        self.undo
            .push(std::mem::replace(&mut self.written, state.clone()));
        Some(state)
    }
}

/// "ループ 2・対応 1・IMU 1・GNSS 除外 1", or "修正なし".
fn summary(e: &Edits) -> String {
    let parts: Vec<String> = [
        (tl("ループ", "Loops"), e.loops.len()),
        (tl("対応", "Matches"), e.matches.len()),
        ("IMU", e.heading_hold.len()),
        (tl("GNSS 除外", "GNSS exclusions"), e.rtk_exclude.len()),
    ]
    .into_iter()
    .filter(|(_, n)| *n > 0)
    .map(|(name, n)| format!("{name} {n}"))
    .collect();
    if parts.is_empty() {
        tl("修正なし", "No corrections").into()
    } else {
        parts.join(tl("・", ", "))
    }
}

/// When a revision was saved, in local time.
fn saved_time(r: &EditsRevision) -> String {
    chrono::DateTime::from_timestamp(r.saved_at as i64, 0)
        .map(|d| {
            d.with_timezone(&chrono::Local)
                .format("%m/%d %H:%M")
                .to_string()
        })
        .unwrap_or_default()
}

impl App {
    /// Writes the corrections after a change, which becomes undoable.
    pub(super) fn save_edits(&mut self) {
        if self.edit_history.record(&self.edits) {
            self.write_edits();
        }
    }

    /// Writes the working corrections and refreshes what depends on them.
    fn write_edits(&mut self) {
        if let Some(c) = self.config()
            && let Err(e) = c.save_edits(&self.edits)
        {
            self.notify(
                Level::Error,
                tr!(
                    "修正内容を保存できませんでした: {e:#}",
                    "Could not save the corrections: {e:#}"
                ),
            );
        }
        if self.job.is_none() {
            self.refresh_steps();
        }
    }

    /// Takes the corrections from the work folder: when they differ from those written here
    /// (another program changed them, or another folder was opened), the history starts over.
    pub(super) fn reload_edits(&mut self) {
        self.edits = self.config().map(|c| c.edits()).unwrap_or_default();
        if self.edits != self.edit_history.written {
            self.edit_history.reset(self.edits.clone());
        }
        self.revisions = self.config().map(|c| c.revisions()).unwrap_or_default();
    }

    pub(super) fn undo_edits(&mut self) {
        if let Some(e) = self.edit_history.undo() {
            self.edits = e;
            self.write_edits();
            self.notify(
                Level::Info,
                tl("修正を元に戻しました", "Undid the correction"),
            );
        } else {
            self.notices.push(
                Level::Info,
                tl("元に戻す修正はありません", "Nothing to undo").into(),
            );
        }
    }

    pub(super) fn redo_edits(&mut self) {
        if let Some(e) = self.edit_history.redo() {
            self.edits = e;
            self.write_edits();
            self.notify(
                Level::Info,
                tl("修正をやり直しました", "Redid the correction"),
            );
        } else {
            self.notices.push(
                Level::Info,
                tl("やり直す修正はありません", "Nothing to redo").into(),
            );
        }
    }

    /// Whether the working corrections differ from the last revision (none: from no corrections).
    pub(super) fn edits_unsaved(&self) -> bool {
        let saved = self.revisions.last().map(|r| &r.edits);
        saved.map_or(self.edits != Edits::default(), |s| *s != self.edits)
    }

    /// Saves the working corrections as a revision (by the user, or as the optimisation runs with
    /// them); nothing when they are those of the last revision.
    pub(super) fn save_revision(&mut self, by_run: bool) {
        if self.config().is_none() || (!by_run && !self.edits_unsaved()) {
            return;
        }
        if by_run && self.revisions.last().is_some_and(|r| r.edits == self.edits) {
            return;
        }
        let name = if by_run {
            tl("最適化の実行時", "At optimization run").to_string()
        } else {
            let n = self.revisions.iter().filter(|r| !r.by_run).count() + 1;
            tr!("保存 {n}", "Save {n}")
        };
        let Some(c) = self.config() else { return };
        match c.save_revision(&name, by_run, &self.edits) {
            Ok(r) => {
                if !by_run {
                    self.notify(
                        Level::Info,
                        tr!(
                            "修正を保存しました（{}）",
                            "Saved the corrections ({})",
                            r.name
                        ),
                    );
                }
                self.revisions.push(r);
            }
            Err(e) => self.notify(
                Level::Error,
                tr!(
                    "修正を保存できませんでした: {e:#}",
                    "Could not save the corrections: {e:#}"
                ),
            ),
        }
    }

    /// Ctrl+Z, Ctrl+Y (Ctrl+Shift+Z) and Ctrl+S, unless a text field has the keyboard.
    pub(super) fn edit_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() || self.config().is_none() {
            return;
        }
        let (save, redo, undo) = ctx.input_mut(|i| {
            let save = i.consume_shortcut(&SAVE);
            let redo = i.consume_shortcut(&REDO_SHIFT) || i.consume_shortcut(&REDO);
            let undo = i.consume_shortcut(&UNDO);
            (save, redo, undo)
        });
        if save {
            self.save_revision(false);
        }
        if redo {
            self.redo_edits();
        } else if undo {
            self.undo_edits();
        }
    }

    /// The edit menu: undo, redo, save and the revisions.
    pub(super) fn edit_menu(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let open = self.config().is_some();
        let item = |ui: &mut egui::Ui, on: bool, text: String, key: &egui::KeyboardShortcut| {
            ui.add_enabled(
                on,
                egui::Button::new(text).shortcut_text(ctx.format_shortcut(key)),
            )
            .clicked()
        };
        if item(
            ui,
            self.edit_history.can_undo(),
            tr!("{} 元に戻す", "{} Undo", icon::ARROW_COUNTER_CLOCKWISE),
            &UNDO,
        ) {
            ui.close();
            self.undo_edits();
        }
        if item(
            ui,
            self.edit_history.can_redo(),
            tr!("{} やり直す", "{} Redo", icon::ARROW_CLOCKWISE),
            &REDO,
        ) {
            ui.close();
            self.redo_edits();
        }
        ui.separator();
        if item(
            ui,
            open && self.edits_unsaved(),
            tr!("{} 修正を保存", "{} Save corrections", icon::FLOPPY_DISK),
            &SAVE,
        ) {
            ui.close();
            self.save_revision(false);
        }
        if ui
            .add_enabled(
                open,
                egui::Button::new(tr!(
                    "{} 修正の履歴…",
                    "{} Correction history…",
                    icon::CLOCK_COUNTER_CLOCKWISE
                )),
            )
            .clicked()
        {
            ui.close();
            self.revisions_open = true;
        }
    }

    /// Undo, redo, save and the revisions, above the corrections made.
    pub(super) fn edit_actions_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    self.edit_history.can_undo(),
                    egui::Button::new(icon::ARROW_COUNTER_CLOCKWISE),
                )
                .on_hover_text(tl("元に戻す（Ctrl+Z）", "Undo (Ctrl+Z)"))
                .clicked()
            {
                self.undo_edits();
            }
            if ui
                .add_enabled(
                    self.edit_history.can_redo(),
                    egui::Button::new(icon::ARROW_CLOCKWISE),
                )
                .on_hover_text(tl("やり直す（Ctrl+Y）", "Redo (Ctrl+Y)"))
                .clicked()
            {
                self.redo_edits();
            }
            if ui
                .add_enabled(
                    self.edits_unsaved(),
                    egui::Button::new(tr!("{} 保存", "{} Save", icon::FLOPPY_DISK)),
                )
                .on_hover_text(tl(
                    "いまの修正をリビジョンとして残す（Ctrl+S）",
                    "Keep the current corrections as a revision (Ctrl+S)",
                ))
                .clicked()
            {
                self.save_revision(false);
            }
            if ui
                .button(tr!(
                    "{} 履歴…",
                    "{} History…",
                    icon::CLOCK_COUNTER_CLOCKWISE
                ))
                .on_hover_text(tl(
                    "保存した修正の一覧。前の内容に戻せる",
                    "The saved corrections. Earlier contents can be restored",
                ))
                .clicked()
            {
                self.revisions_open = true;
            }
        });
        if self.edits_unsaved() {
            ui.label(
                egui::RichText::new(tl("● 未保存の修正があります", "● Unsaved corrections"))
                    .color(egui::Color32::from_rgb(230, 190, 70)),
            )
            .on_hover_text(
                tl("修正はすぐ作業フォルダーに書かれ、軌跡の最適化はこの内容を使います。\
                 保存すると、あとで戻せるリビジョンとして残ります", "Corrections are written to the work folder at once, and the trajectory optimization uses them. Saving keeps them as a revision you can go back to"),
            );
        }
    }

    /// The revisions, newest first: rename, go back to one (undoable) or delete it.
    pub(super) fn revisions_window(&mut self, ctx: &egui::Context) {
        if !self.revisions_open {
            return;
        }
        let mut open = true;
        let mut restore = None;
        let mut delete = None;
        let mut rename = None;
        egui::Window::new(tl("修正の履歴", "Correction history"))
            .open(&mut open)
            .default_width(480.0)
            .default_height(260.0)
            .vscroll(true)
            .show(ctx, |ui| {
                if self.revisions.is_empty() {
                    ui.weak(
                        tl("保存したリビジョンはまだありません。Ctrl+S（編集→修正を保存）で保存でき、\
                         軌跡の最適化を実行したときにも自動で残ります", "No revisions saved yet. Save with Ctrl+S (Edit → Save corrections); one is also kept automatically when the trajectory optimization runs"),
                    );
                    return;
                }
                egui::Grid::new("revisions")
                    .num_columns(4)
                    .spacing([10.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        for r in self.revisions.iter().rev() {
                            let mut name = r.name.clone();
                            // a fixed width: a grid cell would shrink the field to its first letters
                            let edit =
                                ui.add_sized([150.0, 20.0], egui::TextEdit::singleline(&mut name));
                            if edit.changed() {
                                rename = Some((r.id.clone(), name));
                            }
                            ui.vertical(|ui| {
                                ui.weak(saved_time(r));
                                if r.by_run {
                                    ui.weak(tl("最適化の実行時", "At optimization run"));
                                }
                            });
                            let current = r.edits == self.edits;
                            ui.label(summary(&r.edits)).on_hover_text(if current {
                                tl("いまの修正と同じ内容", "Same as the current corrections")
                            } else {
                                ""
                            });
                            ui.horizontal(|ui| {
                                if current {
                                    ui.weak(tl("● 現在", "● Current"));
                                } else if ui
                                    .button(tl("戻す", "Restore"))
                                    .on_hover_text(tl("修正をこの内容にする（元に戻すで取り消せる）", "Make the corrections these (Undo reverts it)"))
                                    .clicked()
                                {
                                    restore = Some(r.edits.clone());
                                }
                                let armed = self.revision_delete.as_deref() == Some(r.id.as_str());
                                let text = if armed {
                                    tr!("{} もう一度で削除", "{} Click again to delete", icon::TRASH)
                                } else {
                                    icon::TRASH.to_string()
                                };
                                if ui
                                    .button(text)
                                    .on_hover_text(tl("このリビジョンを削除", "Delete this revision"))
                                    .clicked()
                                {
                                    if armed {
                                        delete = Some(r.id.clone());
                                    } else {
                                        self.revision_delete = Some(r.id.clone());
                                    }
                                }
                            });
                            ui.end_row();
                        }
                    });
            });
        self.revisions_open = open;
        let Some(c) = self.config() else { return };
        if let Some((id, name)) = rename {
            match c.rename_revision(&id, &name) {
                Ok(()) => {
                    if let Some(r) = self.revisions.iter_mut().find(|r| r.id == id) {
                        r.name = name;
                    }
                }
                Err(e) => self.notify(
                    Level::Error,
                    tr!(
                        "名前を変えられませんでした: {e:#}",
                        "Could not rename it: {e:#}"
                    ),
                ),
            }
        }
        if let Some(id) = delete {
            self.revision_delete = None;
            match c.delete_revision(&id) {
                Ok(()) => self.revisions.retain(|r| r.id != id),
                Err(e) => self.notify(
                    Level::Error,
                    tr!("削除できませんでした: {e:#}", "Could not delete: {e:#}"),
                ),
            }
        }
        if let Some(e) = restore {
            self.edits = e;
            self.save_edits();
            self.notify(
                Level::Info,
                tl(
                    "修正を保存した内容に戻しました",
                    "Restored the corrections to the saved contents",
                ),
            );
        }
    }
}

const UNDO: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
const REDO: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);
const REDO_SHIFT: egui::KeyboardShortcut = egui::KeyboardShortcut::new(
    egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
    egui::Key::Z,
);
const SAVE: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);

#[cfg(test)]
mod tests {
    use super::*;

    fn edits(n: usize) -> Edits {
        Edits {
            loops: (0..n)
                .map(|k| (At::new(k as f64), At::new(k as f64 + 5.0)))
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn undo_and_redo_walk_the_written_corrections_and_a_new_change_drops_the_redo() {
        let mut h = EditHistory::default();
        assert!(!h.record(&Edits::default()));
        assert!(h.record(&edits(1)));
        assert!(h.record(&edits(2)));
        assert_eq!(h.undo(), Some(edits(1)));
        assert_eq!(h.undo(), Some(Edits::default()));
        assert_eq!(h.undo(), None);
        assert_eq!(h.redo(), Some(edits(1)));
        assert!(h.can_redo());
        // a new change after an undo: no redo past it
        assert!(h.record(&edits(3)));
        assert!(!h.can_redo());
        assert_eq!(h.undo(), Some(edits(1)));
        // the limit keeps the latest
        h.reset(Edits::default());
        for k in 1..=EditHistory::LIMIT + 5 {
            h.record(&edits(k));
        }
        let mut n = 0;
        while h.undo().is_some() {
            n += 1;
        }
        assert_eq!(n, EditHistory::LIMIT);
    }

    #[test]
    fn a_renamed_session_is_renamed_in_every_state() {
        let on = |s: &str, n: usize| Edits {
            loops: vec![(At::on(s, n as f64), At::on("b", 1.0))],
            ..Default::default()
        };
        let mut h = EditHistory::default();
        h.record(&on("a", 1));
        h.record(&on("a", 2));
        h.record(&on("a", 3));
        h.undo();
        h.rename_session("a", "park");
        assert_eq!(h.written, on("park", 2));
        assert_eq!(h.undo(), Some(on("park", 1)));
        assert_eq!(h.redo(), Some(on("park", 2)));
        assert_eq!(h.redo(), Some(on("park", 3)));
    }

    #[test]
    fn summaries_count_the_corrections() {
        assert_eq!(summary(&Edits::default()), "修正なし");
        let mut e = edits(2);
        e.rtk_exclude.push((At::new(1.0), At::new(2.0)));
        assert_eq!(summary(&e), "ループ 2・GNSS 除外 1");
    }
}
