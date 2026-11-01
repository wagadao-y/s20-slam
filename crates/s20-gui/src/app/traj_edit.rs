//! The corrections placed on the trajectory in the view: intervals without GNSS, intervals where
//! the heading is left to the IMU, and loops (two times at the same place). Each is placed by two
//! clicks on the trajectory, its ends are dragged along it, a click picks it and Delete removes it.
//! The trajectory tab lists them by kind, and one button applies them.

use super::*;
use crate::lang::tl;

/// What a correction on the trajectory is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum EditKind {
    Exclude,
    Hold,
    Loop,
}

impl EditKind {
    pub(super) const ALL: [EditKind; 3] = [EditKind::Exclude, EditKind::Hold, EditKind::Loop];

    /// The colour it takes on the trajectory and the time bar, and of its markers.
    pub(super) fn color(self) -> egui::Color32 {
        let c = match self {
            EditKind::Exclude => EXCLUDED,
            EditKind::Hold => HELD,
            EditKind::Loop => LOOP,
        };
        egui::Color32::from_rgb(c[0], c[1], c[2])
    }

    fn title(self) -> &'static str {
        match self {
            EditKind::Exclude => tl("GNSS を使わない区間", "Sections without GNSS"),
            EditKind::Hold => tl(
                "方位を IMU に任せる区間",
                "Sections with heading held to the IMU",
            ),
            EditKind::Loop => tl("ループ", "Loops"),
        }
    }

    fn help(self) -> &'static str {
        match self {
            EditKind::Exclude => tl(
                "この区間の RTK / PPK の測位を軌跡の最適化で使いません。建物や木の陰で測位が飛ぶところに使います",
                "The RTK / PPK positions in this section are not used in the trajectory optimization. For places where the positions jump in the shade of buildings or trees",
            ),
            EditKind::Hold => tl(
                "この区間では LIO が方位（鉛直軸まわりの回転）とジャイロのバイアスを IMU のまま保ちます。\
                 丸い丘や特徴のない斜面など、LiDAR が方位を引きずる場所に使います。反映は LIO からやり直します",
                "In this section LIO keeps the heading (rotation about the vertical) and the gyro bias as the IMU gives them. For places where the LiDAR drags the heading, such as round hills or featureless slopes. Applied by redoing from LIO",
            ),
            EditKind::Loop => tl(
                "同じ場所を通った 2 つの時刻です。軌跡の最適化が 2 時刻の位置をそろえ、ずれを軌跡全体に配ります",
                "Two times that passed the same place. The trajectory optimization brings the two times' positions together and spreads the offset over the whole trajectory",
            ),
        }
    }

    /// An interval (else a pair of times at the same place).
    pub(super) fn interval(self) -> bool {
        self != EditKind::Loop
    }

    /// The shortest interval, or the least time between the two ends of a loop (seconds).
    fn min_span(self) -> f64 {
        if self.interval() { 1.0 } else { 2.0 }
    }
}

/// A correction: its kind and place in its list.
pub(super) type EditRef = (EditKind, usize);

/// A correction being placed: the time of its first click, once made.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) struct Placing {
    pub kind: EditKind,
    pub first: Option<f64>,
}

/// Picks the pass of the trajectory under the pointer: `cands` are the samples near it (index,
/// distance on screen, time) in index order. Samples close in index are one pass; of each pass the
/// nearest on screen stands for it, and the pass nearest in time to `prefer` wins (else the nearest
/// on screen): where the trajectory goes by the same place again, an interval's end stays on the
/// pass of its other end and a dragged end on its own pass.
pub(super) fn choose_pass(cands: &[(usize, f32, f64)], prefer: Option<f64>) -> Option<usize> {
    let mut passes: Vec<(usize, f32, f64)> = Vec::new();
    let mut last = None;
    for &(i, d, t) in cands {
        let same = last.is_some_and(|l: usize| i <= l + 5);
        match passes.last_mut() {
            Some(p) if same => {
                if d < p.1 {
                    *p = (i, d, t);
                }
            }
            _ => passes.push((i, d, t)),
        }
        last = Some(i);
    }
    let best = match prefer {
        Some(pt) => passes
            .iter()
            .min_by(|a, b| (a.2 - pt).abs().total_cmp(&(b.2 - pt).abs())),
        None => passes.iter().min_by(|a, b| a.1.total_cmp(&b.1)),
    };
    best.map(|p| p.0)
}

impl App {
    pub(super) fn edit_list(&self, kind: EditKind) -> &Vec<(At, At)> {
        match kind {
            EditKind::Exclude => &self.edits.rtk_exclude,
            EditKind::Hold => &self.edits.heading_hold,
            EditKind::Loop => &self.edits.loops,
        }
    }

    fn edit_list_mut(&mut self, kind: EditKind) -> &mut Vec<(At, At)> {
        match kind {
            EditKind::Exclude => &mut self.edits.rtk_exclude,
            EditKind::Hold => &mut self.edits.heading_hold,
            EditKind::Loop => &mut self.edits.loops,
        }
    }

    /// The kinds shown: GNSS exclusions only when GNSS is used.
    fn edit_kinds(&self) -> impl Iterator<Item = EditKind> + use<> {
        let rtk = self.rtk_used();
        EditKind::ALL
            .into_iter()
            .filter(move |k| *k != EditKind::Exclude || rtk)
    }

    /// The interval being placed, from its first click to the pointer.
    pub(super) fn placing_interval(&self, kind: EditKind) -> Option<(f64, f64)> {
        let p = self.placing.filter(|p| p.kind == kind && kind.interval())?;
        let (a, b) = (p.first?, self.placing_hover?);
        Some((a.min(b), a.max(b)))
    }

    /// The sample of the trajectory under `pos` (within 20 pixels), on the pass nearest in time to
    /// `prefer` where it goes by more than once.
    pub(super) fn traj_at(
        &self,
        rect: egui::Rect,
        pos: egui::Pos2,
        prefer: Option<f64>,
    ) -> Option<(f64, DVec3)> {
        let traj = &self.scene.traj;
        let cands: Vec<(usize, f32, f64)> = traj
            .iter()
            .enumerate()
            .filter(|(_, (t, _))| self.session_visible(self.scene.session_at(*t)))
            .filter_map(|(i, &(t, p))| {
                let d = self.to_screen(rect, p)?.distance(pos);
                (d < 20.0).then_some((i, d, t))
            })
            .collect();
        choose_pass(&cands, prefer).map(|i| traj[i])
    }

    /// The ends of the corrections on screen: (correction, second end, position).
    pub(super) fn edit_handles(&self, rect: egui::Rect) -> Vec<(EditRef, bool, egui::Pos2)> {
        let mut out = Vec::new();
        for kind in self.edit_kinds() {
            for (i, (a, b)) in self.edit_list(kind).iter().enumerate() {
                for (second, x) in [(false, a), (true, b)] {
                    if let Some(t) = self.abs_of(x)
                        && self.session_visible(self.scene.session_at(t))
                        && let Some(p) = self.scene.position_at(t)
                        && let Some(sp) = self.to_screen(rect, p)
                        && rect.contains(sp)
                    {
                        out.push(((kind, i), second, sp));
                    }
                }
            }
        }
        out
    }

    /// The end under `pos` (within 9 pixels), the nearest.
    pub(super) fn handle_at(&self, rect: egui::Rect, pos: egui::Pos2) -> Option<(EditRef, bool)> {
        self.edit_handles(rect)
            .into_iter()
            .map(|(e, second, sp)| (sp.distance(pos), e, second))
            .filter(|h| h.0 < 9.0)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, e, second)| (e, second))
    }

    /// The interval at time `t`, GNSS exclusions after heading holds (as they are drawn).
    fn interval_at(&self, t: f64) -> Option<EditRef> {
        [EditKind::Hold, EditKind::Exclude]
            .into_iter()
            .filter(|k| self.edit_kinds().any(|x| x == *k))
            .find_map(|kind| {
                self.edit_list(kind)
                    .iter()
                    .position(|(a, b)| {
                        self.abs_of(a)
                            .zip(self.abs_of(b))
                            .is_some_and(|(a, b)| t >= a && t < b)
                    })
                    .map(|i| (kind, i))
            })
    }

    /// Starts placing a correction of `kind` (a second press of its button stops).
    pub(super) fn toggle_placing(&mut self, kind: EditKind) {
        self.placing = match self.placing {
            Some(p) if p.kind == kind => None,
            _ => Some(Placing { kind, first: None }),
        };
        self.placing_hover = None;
        self.selected_edit = None;
        self.selected_match = None;
        self.matching = None;
    }

    /// A click on the trajectory at `t` while placing: the first end, or the second, which adds
    /// the correction.
    pub(super) fn place_at(&mut self, t: f64) {
        let Some(p) = self.placing else { return };
        let Some(a) = p.first else {
            self.placing = Some(Placing {
                first: Some(t),
                ..p
            });
            return;
        };
        if (t - a).abs() < p.kind.min_span() {
            let text = if p.kind.interval() {
                tl(
                    "区間が短すぎます（1 秒以上にしてください）",
                    "The section is too short (make it 1 s or longer)",
                )
            } else {
                tl(
                    "ループの 2 点が近すぎます（同じ場所を通った別の時刻を選んでください）",
                    "The loop's two points are too close (choose another time that passed the same place)",
                )
            };
            self.notify(Level::Warn, text);
            return;
        }
        let ends = (self.at_of(a.min(t)), self.at_of(a.max(t)));
        let list = self.edit_list_mut(p.kind);
        list.push(ends);
        let i = list.len() - 1;
        self.placing = None;
        self.placing_hover = None;
        self.selected_edit = Some((p.kind, i));
        self.save_edits();
    }

    /// Removes the selected correction.
    pub(super) fn remove_selected_edit(&mut self) {
        let Some((kind, i)) = self.selected_edit.take() else {
            return;
        };
        if i < self.edit_list(kind).len() {
            self.edit_list_mut(kind).remove(i);
            self.save_edits();
            self.notify(
                Level::Info,
                tr!(
                    "{}を削除しました（Ctrl+Z で元に戻せます）",
                    "Deleted the {} (Ctrl+Z undoes it)",
                    kind.title()
                ),
            );
        }
    }

    /// Placing, dragging the ends, picking a correction by a click, and Delete. Returns whether
    /// the pointer is busy dragging an end (the camera then stays) and whether a click picked a
    /// correction (it is then not a pick of a point).
    pub(super) fn traj_edit_input(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
    ) -> (bool, bool) {
        let rect = response.rect;
        let pointer = ctx.input(|i| i.pointer.hover_pos());
        let free_keys = ctx.memory(|m| m.focused().is_none());
        // start dragging an end where the press began
        if self.placing.is_none()
            && self.s.show_traj
            && response.drag_started_by(egui::PointerButton::Primary)
            && let Some(press) = ctx.input(|i| i.pointer.press_origin())
        {
            self.dragging_end = self.handle_at(rect, press);
            if let Some((e, _)) = self.dragging_end {
                self.selected_edit = Some(e);
                self.selected_match = None;
            }
        }
        if let Some(((kind, i), second)) = self.dragging_end {
            let now = self
                .edit_list(kind)
                .get(i)
                .and_then(|(a, b)| self.abs_of(if second { b } else { a }));
            if let Some(pos) = pointer
                && let Some((t, _)) = self.traj_at(rect, pos, now)
            {
                let at = self.at_of(t);
                if let Some(e) = self.edit_list_mut(kind).get_mut(i) {
                    if second {
                        e.1 = at;
                    } else {
                        e.0 = at;
                    }
                }
            }
            if !ctx.input(|i| i.pointer.primary_down()) {
                self.dragging_end = None;
                // an end dragged past the other: the interval keeps its start first
                if let Some((a, b)) = self.edit_list(kind).get(i).cloned()
                    && let (Some(ta), Some(tb)) = (self.abs_of(&a), self.abs_of(&b))
                    && ta > tb
                {
                    self.edit_list_mut(kind)[i] = (b, a);
                }
                self.save_edits();
            }
            ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
            return (true, false);
        }
        if let Some(p) = self.placing {
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.placing = None;
                self.placing_hover = None;
                return (false, false);
            }
            // an interval's second end on the pass of its first; a loop's on any pass
            let prefer = p.first.filter(|_| p.kind.interval());
            self.placing_hover = pointer
                .filter(|_| response.hovered())
                .and_then(|pos| self.traj_at(rect, pos, prefer))
                .map(|(t, _)| t);
            let mut placed = false;
            if response.clicked()
                && !ctx.input(|i| i.modifiers.command)
                && let Some(pos) = response.interact_pointer_pos()
                && let Some((t, _)) = self.traj_at(rect, pos, prefer)
            {
                self.place_at(t);
                placed = true;
            }
            if response.hovered() {
                ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
            }
            return (false, placed);
        }
        if free_keys
            && self.selected_edit.is_some()
            && ctx
                .input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
        {
            self.remove_selected_edit();
        }
        if free_keys && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.selected_edit = None;
        }
        let mut picked = false;
        if self.s.show_traj
            && response.clicked()
            && !ctx.input(|i| i.modifiers.command)
            && let Some(pos) = response.interact_pointer_pos()
        {
            let hit = self.handle_at(rect, pos).map(|(e, _)| e).or_else(|| {
                self.traj_at(rect, pos, None)
                    .filter(|(_, p)| {
                        self.to_screen(rect, *p)
                            .is_some_and(|sp| sp.distance(pos) < 8.0)
                    })
                    .and_then(|(t, _)| self.interval_at(t))
            });
            picked = hit.is_some();
            if picked {
                self.selected_match = None;
            }
            self.selected_edit = hit;
        }
        if response.hovered()
            && self.s.show_traj
            && let Some(pos) = pointer
            && self.handle_at(rect, pos).is_some()
        {
            ctx.set_cursor_icon(egui::CursorIcon::Grab);
        }
        (false, picked)
    }

    /// Label of the two ends of a correction, with the short session names (S1, S2, …) if `short`.
    fn edit_label(&self, kind: EditKind, a: &At, b: &At, short: bool) -> String {
        let l = |x: &At| {
            if short {
                self.at_short_label(x)
            } else {
                self.at_label(x)
            }
        };
        if kind.interval() {
            format!("{} 〜 {}", l(a), l(b))
        } else {
            format!("{} ⇄ {}", l(a), l(b))
        }
    }

    /// What to click next while placing, and the correction selected; for the top of the view.
    pub(super) fn edit_hint(&self) -> Option<(String, egui::Color32)> {
        if let Some(p) = self.placing {
            let first = p.first.map(|t| self.at_label(&self.at_of(t)));
            let text = match (p.kind.interval(), first) {
                (true, None) => tr!(
                    "{}：始点を軌跡でクリック",
                    "{}: click the start on the trajectory",
                    p.kind.title()
                ),
                (true, Some(f)) => tr!(
                    "{}：終点を軌跡でクリック（始点 {f}）",
                    "{}: click the end on the trajectory (start {f})",
                    p.kind.title()
                ),
                (false, None) => tl(
                    "ループの1点目：軌跡をクリック（同じ場所を通った一方の時刻）",
                    "First point of the loop: click the trajectory (one time it passed the place)",
                )
                .to_string(),
                (false, Some(f)) => {
                    tr!(
                        "ループの2点目：同じ場所を通ったもう一方の軌跡をクリック（1点目 {f}）",
                        "Second point of the loop: click the trajectory where it passed the same place again (first point {f})"
                    )
                }
            };
            return Some((
                tr!("{text}　　Esc で取り消し", "{text}    Esc to cancel"),
                p.kind.color(),
            ));
        }
        let (kind, i) = self.selected_edit?;
        let (a, b) = self.edit_list(kind).get(i)?;
        Some((
            tr!(
                "{}: {}　　端をドラッグで調整、Delete で削除",
                "{}: {}    drag the ends to adjust, Delete to delete",
                kind.title(),
                self.edit_label(kind, a, b, false)
            ),
            kind.color(),
        ))
    }

    /// The ends of the corrections, the point placed first and the one under the pointer while
    /// placing, with their labels: the loops' ends always, an interval's when it is selected.
    pub(super) fn draw_edit_markers(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        labels: &mut ViewLabels,
    ) -> Vec<(EditRef, bool, egui::Pos2)> {
        let pointer = painter.ctx().input(|i| i.pointer.hover_pos());
        let handles = if self.s.show_traj {
            self.edit_handles(rect)
        } else {
            Vec::new()
        };
        let hover = pointer
            .filter(|_| self.placing.is_none())
            .and_then(|p| self.handle_at(rect, p));
        for &(e, second, sp) in &handles {
            let active = self.dragging_end == Some((e, second)) || hover == Some((e, second));
            let selected = self.selected_edit == Some(e);
            let r = if active { 8.0 } else { 6.0 };
            let outline = if selected {
                egui::Stroke::new(2.5, egui::Color32::WHITE)
            } else {
                egui::Stroke::new(2.0, egui::Color32::BLACK)
            };
            painter.circle(sp, r, e.0.color(), outline);
            labels.occupy(egui::Rect::from_center_size(sp, egui::Vec2::splat(2.0 * r)));
        }
        if let Some(p) = self.placing {
            for (t, what) in [
                (p.first, tl("1点目", "first point")),
                (self.placing_hover, ""),
            ] {
                if let Some(t) = t
                    && let Some(pos) = self.scene.position_at(t)
                    && let Some(sp) = self.to_screen(rect, pos)
                {
                    let r = if what.is_empty() { 5.0 } else { 7.0 };
                    painter.circle(
                        sp,
                        r,
                        p.kind.color(),
                        egui::Stroke::new(2.0, egui::Color32::BLACK),
                    );
                    labels.occupy(egui::Rect::from_center_size(sp, egui::Vec2::splat(2.0 * r)));
                    let at = self.at_of(t);
                    let what = if what.is_empty() {
                        what
                    } else {
                        tl("1点目 ", "first point ")
                    };
                    let (galley, color) = self.end_label(painter, what, p.kind, &at);
                    labels.add_galley(
                        painter,
                        sp,
                        galley,
                        Some(format!("{what}{}", self.at_label(&at))),
                        color,
                        false,
                        false,
                    );
                }
            }
        }
        handles
    }

    /// The labels of the ends drawn by `draw_edit_markers`, after the other labels.
    pub(super) fn draw_edit_labels(
        &self,
        painter: &egui::Painter,
        handles: Vec<(EditRef, bool, egui::Pos2)>,
        labels: &mut ViewLabels,
    ) {
        for ((kind, i), second, sp) in handles {
            if kind.interval() && self.selected_edit != Some((kind, i)) {
                continue;
            }
            let Some((a, b)) = self.edit_list(kind).get(i) else {
                continue;
            };
            let x = if second { b } else { a };
            // the two ends may coincide (start = end): one label above, one below
            let full = self.multi().then(|| self.at_label(x));
            let (galley, color) = self.end_label(painter, "", kind, x);
            labels.add_galley(painter, sp, galley, full, color, second, false);
        }
    }

    /// The view's label of an end of a correction, after `what`, and its colour: a loop's in the
    /// colour of its session when there are several (its marker shows it is a loop), an
    /// interval's in the colour of its kind with the short session name in the session's.
    fn end_label(
        &self,
        painter: &egui::Painter,
        what: &str,
        kind: EditKind,
        at: &At,
    ) -> (std::sync::Arc<egui::Galley>, egui::Color32) {
        let session = self.session_index(&at.session).filter(|_| self.multi());
        match session {
            Some(k) if !kind.interval() => {
                let [r, g, b] = Self::session_color(k);
                let color = egui::Color32::from_rgb(r, g, b);
                let text = format!("{what}{}", self.at_short_label(at));
                let galley = painter.layout_no_wrap(text, egui::FontId::proportional(13.0), color);
                (galley, color)
            }
            _ => (
                self.at_short_galley(painter, what, at, kind.color()),
                kind.color(),
            ),
        }
    }

    /// The trajectory tab's corrections: by kind, each with its button to place one and its list,
    /// then the button that applies them.
    pub(super) fn traj_edits_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let kinds: Vec<EditKind> = self.edit_kinds().collect();
        for kind in kinds {
            ui.add_space(4.0);
            self.edit_kind_ui(ui, kind);
        }
        if self.multi() {
            ui.add_space(4.0);
            self.match_edits_ui(ui, ctx);
        }
        ui.add_space(8.0);
        self.apply_edits_ui(ui, ctx);
    }

    fn edit_kind_ui(&mut self, ui: &mut egui::Ui, kind: EditKind) {
        let n = self.edit_list(kind).len();
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("■").color(kind.color()));
            ui.label(egui::RichText::new(tr!("{}（{n}）", "{} ({n})", kind.title())).strong())
                .on_hover_text(kind.help());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let placing = self.placing.is_some_and(|p| p.kind == kind);
                let text = if placing {
                    tr!("{} 追加中…", "{} Adding…", icon::PLUS)
                } else {
                    tr!("{} 追加", "{} Add", icon::PLUS)
                };
                let tip = if kind.interval() {
                    tl("ビューで軌跡の始点と終点をクリックします（Esc で取り消し）", "Click the start and end on the trajectory in the view (Esc to cancel)")
                } else {
                    tl("ビューで、同じ場所を通った 2 つの時刻の軌跡をクリックします（Esc で取り消し）", "Click the trajectory in the view at two times that passed the same place (Esc to cancel)")
                };
                if ui
                    .add(egui::Button::new(text).selected(placing))
                    .on_hover_text(tip)
                    .clicked()
                {
                    self.toggle_placing(kind);
                }
                if kind == EditKind::Loop
                    && ui
                        .add(egui::Button::new(tl("始点＝終点", "Start = end")))
                        .on_hover_text(if self.multi() {
                            tl("時刻バーで選んでいるセッションの最初と最後が同じ場所", "The first and last of the session chosen on the time bar are the same place")
                        } else {
                            tl("最初と最後が同じ場所（同じ位置に戻って撮影を終えたとき）", "The first and last are the same place (when the recording ended back at the same position)")
                        })
                        .clicked()
                {
                    let k = self.sel_session.min(self.scene.sessions.len().max(1) - 1);
                    let key = self.session_key(k);
                    self.edits
                        .loops
                        .push((At::on(&key, 0.5), At::on(&key, -0.5)));
                    self.selected_edit = Some((kind, self.edits.loops.len() - 1));
                    self.save_edits();
                }
            });
        });
        let mut go = None;
        let mut remove = None;
        let mut select = None;
        let mut retime: Option<(usize, bool, f64)> = None;
        for (i, (a, b)) in self.edit_list(kind).iter().enumerate() {
            let selected = self.selected_edit == Some((kind, i));
            // short session names in the list, the full ones on hover
            let mut tip = String::new();
            if self.multi() {
                tip = self.edit_label(kind, a, b, false) + "\n";
            }
            tip += tl(
                "選ぶと軌跡上で強調し、端の時刻を直せます",
                "Selecting it highlights it on the trajectory and lets you fix the times of its ends",
            );
            ui.horizontal(|ui| {
                ui.add_space(14.0);
                if ui
                    .selectable_label(selected, self.edit_label(kind, a, b, true))
                    .on_hover_text(tip)
                    .clicked()
                {
                    select = Some(if selected { None } else { Some((kind, i)) });
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button(icon::TRASH)
                        .on_hover_text(tl("削除", "Delete"))
                        .clicked()
                    {
                        remove = Some(i);
                    }
                    if ui
                        .button(icon::CROSSHAIR)
                        .on_hover_text(tl("この時刻へ移動", "Go to this time"))
                        .clicked()
                    {
                        go = self.abs_of(a).map(|t| (i, t));
                    }
                });
            });
            // the selected one: its ends' times, to set exactly
            if selected {
                ui.horizontal(|ui| {
                    ui.add_space(28.0);
                    for (second, x) in [(false, a), (true, b)] {
                        if second {
                            ui.label(if kind.interval() { "〜" } else { "⇄" });
                        }
                        let Some(t) = self.abs_of(x) else { continue };
                        let start = self.session_start(t);
                        let mut rel = t - start;
                        let r = ui.add(
                            egui::DragValue::new(&mut rel)
                                .speed(0.1)
                                .fixed_decimals(1)
                                .suffix(tl(" 秒", " s")),
                        );
                        if r.changed() {
                            retime = Some((i, second, start + rel));
                        }
                    }
                });
            }
        }
        if let Some(s) = select {
            self.selected_edit = s;
            self.selected_match = None;
            self.placing = None;
        }
        if let Some((i, second, t)) = retime {
            let (ts, te) = (self.scene.traj[0].0, self.scene.traj.last().unwrap().0);
            let at = self.at_of(t.clamp(ts, te));
            let e = &mut self.edit_list_mut(kind)[i];
            if second {
                e.1 = at;
            } else {
                e.0 = at;
            }
            self.save_edits();
        }
        if let Some((i, t)) = go {
            self.selected_edit = Some((kind, i));
            self.selected_match = None;
            self.placing = None;
            self.go_to_time(t);
        }
        if let Some(i) = remove {
            self.selected_edit = Some((kind, i));
            self.remove_selected_edit();
        }
    }

    /// The run that applies the corrections: from the odometry when it is out of date (the
    /// heading holds changed), else the optimisation alone.
    pub(super) fn apply_run(&self) -> Run {
        if self.steps.get(&Step::Lio) == Some(&StepState::Stale) {
            Run {
                first: self.plan(Step::Lio, true).first,
                last: Some(Step::Opt),
            }
        } else {
            self.plan(Step::Opt, true)
        }
    }

    fn apply_edits_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let lio = self.steps.get(&Step::Lio) == Some(&StepState::Stale);
        let opt = self.steps.get(&Step::Opt) == Some(&StepState::Stale);
        let amber = egui::Color32::from_rgb(230, 190, 70);
        if lio {
            ui.colored_label(
                amber,
                tr!(
                    "{} 方位の区間などが LIO に未反映",
                    "{} Heading sections etc. not yet applied in LIO",
                    icon::ARROW_CLOCKWISE
                ),
            );
        } else if opt {
            ui.colored_label(
                amber,
                tr!(
                    "{} 修正が軌跡に反映されていません",
                    "{} Corrections not yet applied to the trajectory",
                    icon::ARROW_CLOCKWISE
                ),
            );
        }
        let text = if lio {
            tr!(
                "{}  修正を反映する（LIO から）",
                "{}  Apply the corrections (from LIO)",
                icon::ARROW_CLOCKWISE
            )
        } else {
            tr!(
                "{}  修正を反映する",
                "{}  Apply the corrections",
                icon::ARROW_CLOCKWISE
            )
        };
        let tip = if lio {
            tl(
                "LIO と軌跡の最適化をやり直す（点群と着色は後で）",
                "Redo LIO and the trajectory optimization (point cloud and colorization later)",
            )
        } else {
            tl(
                "軌跡の最適化をやり直す（点群と着色は後で）",
                "Redo the trajectory optimization (point cloud and colorization later)",
            )
        };
        if ui
            .add_enabled(
                self.job.is_none(),
                egui::Button::new(text).selected(lio || opt),
            )
            .on_hover_text(tip)
            .clicked()
        {
            // a deliberate redo with the corrections: no confirmation
            let run = self.apply_run();
            self.start_job(ctx, run);
        }
    }
}
