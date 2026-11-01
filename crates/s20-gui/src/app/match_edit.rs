//! Matched points: the same feature picked in two sessions, to place a session whose trajectory
//! does not meet the others. Added by two clicks on the scans (the point under the pointer shown
//! before the click), shown as numbered markers (P1, P2, …), picked by a click, moved by dragging
//! a marker onto another point of its session, removed by Delete. The trajectory tab lists them by
//! pair of sessions.

use super::*;
use crate::lang::tl;

/// The colour of the matched points' markers and the lines between them: none of the sessions'
/// or the corrections'.
pub(super) const MATCH: [u8; 3] = [245, 245, 245];

fn match_color() -> egui::Color32 {
    egui::Color32::from_rgb(MATCH[0], MATCH[1], MATCH[2])
}

/// The sessions of a match's two points, the lower first.
type SessionPair = (Option<usize>, Option<usize>);

/// The name of match `i` on the view and in the list: P1, P2, …
fn match_tag(i: usize) -> String {
    format!("P{}", i + 1)
}

impl App {
    /// The session of a matched point, if it is one of those loaded.
    fn match_session(&self, p: &PointAt) -> Option<usize> {
        self.session_index(&p.session)
    }

    /// The matched points on screen, of the sessions shown: (match, second point, position).
    fn match_handles(&self, rect: egui::Rect) -> Vec<(usize, bool, egui::Pos2)> {
        let mut out = Vec::new();
        for (i, (a, b)) in self.edits.matches.iter().enumerate() {
            for (second, x) in [(false, a), (true, b)] {
                if let Some(k) = self.match_session(x)
                    && self.session_visible(Some(k))
                    && let Some(w) = self.match_world(x)
                    && let Some(sp) = self.to_screen(rect, w)
                    && rect.contains(sp)
                {
                    out.push((i, second, sp));
                }
            }
        }
        out
    }

    /// The matched point under `pos` (within 9 pixels), the nearest.
    fn match_handle_at(&self, rect: egui::Rect, pos: egui::Pos2) -> Option<(usize, bool)> {
        self.match_handles(rect)
            .into_iter()
            .map(|(i, second, sp)| (sp.distance(pos), i, second))
            .filter(|h| h.0 < 9.0)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, i, second)| (i, second))
    }

    /// Selects match `i` (or none); a correction on the trajectory is then not selected.
    fn select_match(&mut self, i: Option<usize>) {
        self.selected_match = i;
        if i.is_some() {
            self.selected_edit = None;
            self.placing = None;
        }
    }

    fn remove_match(&mut self, i: usize) {
        if i >= self.edits.matches.len() {
            return;
        }
        self.edits.matches.remove(i);
        self.selected_match = None;
        self.save_edits();
        self.notify(
            Level::Info,
            tl(
                "点の対応を削除しました（Ctrl+Z で元に戻せます）",
                "Deleted the point match (Ctrl+Z undoes it)",
            ),
        );
    }

    /// Adding matched points, dragging one onto another point of its session, picking one by a
    /// click, and Delete. Returns whether the pointer is busy with it (the camera then stays).
    pub(super) fn match_input(&mut self, ctx: &egui::Context, response: &egui::Response) -> bool {
        let rect = response.rect;
        let pointer = ctx.input(|i| i.pointer.hover_pos());
        let free_keys = ctx.memory(|m| m.focused().is_none());
        // a match taken away by an undo
        if self
            .selected_match
            .is_some_and(|i| i >= self.edits.matches.len())
        {
            self.selected_match = None;
        }
        self.match_hover = None;
        if let Some(first) = self.matching.clone() {
            return self.match_add_input(ctx, response, first);
        }
        // start dragging a point where the press began
        if self.placing.is_none()
            && self.dragging_end.is_none()
            && response.drag_started_by(egui::PointerButton::Primary)
            && let Some(press) = ctx.input(|i| i.pointer.press_origin())
        {
            self.dragging_match = self.match_handle_at(rect, press);
            if let Some((i, _)) = self.dragging_match {
                self.select_match(Some(i));
            }
        }
        if let Some((i, second)) = self.dragging_match {
            self.match_hover = pointer.and_then(|pos| self.pick_at(rect, pos));
            if !ctx.input(|i| i.pointer.primary_down()) {
                self.dragging_match = None;
                self.drop_match(i, second);
                self.match_hover = None;
            }
            ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
            return true;
        }
        if let Some(i) = self.selected_match
            && free_keys
        {
            if ctx
                .input(|k| k.key_pressed(egui::Key::Delete) || k.key_pressed(egui::Key::Backspace))
            {
                self.remove_match(i);
            } else if ctx.input(|k| k.key_pressed(egui::Key::Escape)) {
                self.selected_match = None;
            }
        }
        if response.clicked()
            && !ctx.input(|i| i.modifiers.command)
            && let Some(pos) = response.interact_pointer_pos()
        {
            // a click elsewhere drops the selection
            let hit = self.match_handle_at(rect, pos).map(|h| h.0);
            if hit.is_some() || self.selected_match.is_some() {
                self.select_match(hit);
            }
        }
        if response.hovered()
            && let Some(pos) = pointer
            && self.match_handle_at(rect, pos).is_some()
        {
            ctx.set_cursor_icon(egui::CursorIcon::Grab);
        }
        false
    }

    /// A dragged point let go: onto the point under the pointer if it is of the same session.
    pub(super) fn drop_match(&mut self, i: usize, second: bool) {
        let Some((a, b)) = self.edits.matches.get(i) else {
            return;
        };
        let old = if second { b } else { a };
        let want = self.match_session(old);
        match self.match_hover {
            Some((p, Some(k))) if Some(k) == want => {
                if let Some(pa) = self.point_at(k, p) {
                    let m = &mut self.edits.matches[i];
                    if second {
                        m.1 = pa;
                    } else {
                        m.0 = pa;
                    }
                    self.save_edits();
                }
            }
            _ => {
                let tag = want.map(Self::session_tag).unwrap_or_default();
                self.notify(
                    Level::Warn,
                    tr!(
                        "{tag} の点の上で離してください（点の対応は元のままです）",
                        "Let go over a point of {tag} (the match is unchanged)"
                    ),
                );
            }
        }
    }

    /// Adding matched points: the point under the pointer previewed, a click takes it.
    fn match_add_input(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        first: Option<(usize, PointAt)>,
    ) -> bool {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.matching = None;
            return false;
        }
        if response.hovered() {
            ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
            if !ctx.input(|i| i.pointer.any_down())
                && let Some(pos) = response.hover_pos()
            {
                self.match_hover = self.pick_at(response.rect, pos);
            }
        }
        if !(response.clicked() && !ctx.input(|i| i.modifiers.command)) {
            return false;
        }
        let Some((p, session)) = self.pick_session(response) else {
            return true;
        };
        let Some(k) = session else {
            self.notify(
                Level::Warn,
                tl(
                    "点の対応は「スキャン」の表示で、セッションの点をクリックしてください",
                    "For point matches, click a session's points in the \"Scans\" view",
                ),
            );
            return true;
        };
        let Some(pa) = self.point_at(k, p) else {
            return true;
        };
        match first {
            None => self.matching = Some(Some((k, pa))),
            Some((k0, _)) if k0 == k => self.notify(
                Level::Warn,
                tl(
                    "2 点目は別のセッションの、同じ特徴をクリックしてください",
                    "Click the same feature in a different session for the second point",
                ),
            ),
            Some((_, p0)) => {
                self.edits.matches.push((p0, pa));
                self.save_edits();
                // ready for the next pair
                self.matching = Some(None);
            }
        }
        true
    }

    /// What to do next with matched points, for the top of the view.
    pub(super) fn match_hint(&self) -> Option<String> {
        if let Some(first) = &self.matching {
            let text = if first.is_none() {
                tl(
                    "点の対応の1点目：あるセッションの特徴（柱の角、窓枠など）をクリック",
                    "First point of the match: click a feature (a pillar's corner, a window frame…) in one session",
                )
            } else {
                tl(
                    "点の対応の2点目：別のセッションで同じ特徴をクリック",
                    "Second point of the match: click the same feature in another session",
                )
            };
            return Some(tr!("{text}　　Esc で終わる", "{text}    Esc to finish"));
        }
        if let Some((i, second)) = self.dragging_match {
            let (a, b) = self.edits.matches.get(i)?;
            let k = self.match_session(if second { b } else { a })?;
            return Some(tr!(
                "{} の同じ特徴の点の上で離します",
                "Let go over the same feature in {}",
                Self::session_tag(k)
            ));
        }
        let i = self.selected_match?;
        Some(tr!(
            "点の対応 {}: 点をドラッグで付け直し、Delete で削除",
            "Point match {}: drag a point to pick it again, Delete to delete",
            match_tag(i)
        ))
    }

    /// The matched points' markers, the first point of a match being added, and the point under
    /// the pointer while adding or dragging, with their labels.
    pub(super) fn draw_match_marks(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        labels: &mut ViewLabels,
    ) {
        let pointer = painter.ctx().input(|i| i.pointer.hover_pos());
        let hover = pointer
            .filter(|_| self.matching.is_none() && self.dragging_match.is_none())
            .and_then(|p| self.match_handle_at(rect, p));
        let mut marks = Vec::new();
        for (i, second, sp) in self.match_handles(rect) {
            let (a, b) = &self.edits.matches[i];
            let x = if second { b } else { a };
            let active = self.dragging_match == Some((i, second)) || hover == Some((i, second));
            marks.push((i, x.clone(), sp, active));
        }
        // the first point of the match being added
        if let Some(Some((_, p0))) = &self.matching
            && let Some(w) = self.match_world(p0)
            && let Some(sp) = self.to_screen(rect, w)
        {
            marks.push((self.edits.matches.len(), p0.clone(), sp, true));
        }
        for (i, _, sp, active) in &marks {
            self.match_marker(painter, *sp, *active, self.selected_match == Some(*i));
            labels.occupy(egui::Rect::from_center_size(*sp, egui::Vec2::splat(16.0)));
        }
        // the point a click (or letting go) would take
        if let Some((p, session)) = self.match_hover
            && let Some(sp) = self.to_screen(rect, p)
        {
            let (text, full, ok) = self.hover_label(p, session);
            let color = match session.filter(|_| ok) {
                Some(k) => {
                    let [r, g, b] = Self::session_color(k);
                    egui::Color32::from_rgb(r, g, b)
                }
                None => egui::Color32::from_gray(150),
            };
            painter.circle_stroke(sp, 7.0, egui::Stroke::new(2.0, egui::Color32::BLACK));
            painter.circle_stroke(sp, 6.0, egui::Stroke::new(1.5, color));
            painter.circle_filled(sp, 1.5, color);
            labels.add(painter, sp, text, Some(full), color, true, true);
        }
        for (i, x, sp, _) in marks {
            let Some(k) = self.match_session(&x) else {
                continue;
            };
            let [r, g, b] = Self::session_color(k);
            let name = &self.scene.sessions[k].name;
            labels.add(
                painter,
                sp,
                format!("{} {}", match_tag(i), Self::session_tag(k)),
                Some(format!("{} {name}", match_tag(i))),
                egui::Color32::from_rgb(r, g, b),
                false,
                false,
            );
        }
    }

    /// A matched point's marker: a square, larger under the pointer, ringed when selected.
    fn match_marker(&self, painter: &egui::Painter, sp: egui::Pos2, active: bool, selected: bool) {
        let r = if active { 6.0 } else { 4.5 };
        let square = egui::Rect::from_center_size(sp, egui::Vec2::splat(2.0 * r));
        painter.rect(
            square,
            1.0,
            match_color(),
            egui::Stroke::new(2.0, egui::Color32::BLACK),
            egui::StrokeKind::Outside,
        );
        if selected {
            let ring = square.expand(4.0);
            painter.rect_stroke(
                ring,
                2.0,
                egui::Stroke::new(2.0, match_color()),
                egui::StrokeKind::Middle,
            );
        }
    }

    /// The label of the point under the pointer: its session and how far it is (a point behind
    /// the one meant is further), and whether it can be taken.
    fn hover_label(&self, p: DVec3, session: Option<usize>) -> (String, String, bool) {
        let d = (p - self.camera.eye()).length();
        let Some(k) = session else {
            let t = tl("セッションの点ではありません", "Not a session's point");
            return (t.into(), t.into(), false);
        };
        let tag = Self::session_tag(k);
        let name = &self.scene.sessions[k].name;
        let text = tr!("{tag}  {d:.1} m 先", "{tag}  {d:.1} m away");
        // adding: the second point on another session; dragging: on the point's own
        let wrong = match (&self.matching, self.dragging_match) {
            (Some(Some((k0, _))), _) => (*k0 == k).then(|| {
                tl(
                    "1点目と同じセッションです",
                    "The same session as the first point",
                )
            }),
            (_, Some((i, second))) => {
                let want = self
                    .edits
                    .matches
                    .get(i)
                    .and_then(|(a, b)| self.match_session(if second { b } else { a }));
                (want != Some(k))
                    .then(|| tl("別のセッションの点です", "A point of another session"))
            }
            _ => None,
        };
        match wrong {
            Some(w) => (format!("{text}（{w}）"), format!("{name}: {w}"), false),
            None => (text, format!("{name}  {d:.1} m"), true),
        }
    }

    /// The trajectory tab's matched points: the button that adds them, then the matches by pair
    /// of sessions, with how many and how far apart their points are now.
    pub(super) fn match_edits_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let n = self.edits.matches.len();
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("■").color(match_color()));
            ui.label(egui::RichText::new(tr!("点の対応（{n}）", "Point matches ({n})")).strong())
                .on_hover_text(tl(
                    "2 つのセッションに写っている同じ特徴（柱の角、窓枠など）です。軌跡が交わらない\
                     セッション（別の階など）をつなぐときに使います",
                    "The same feature (a pillar's corner, a window frame…) seen in two sessions. Used to connect sessions whose trajectories don't cross (different floors, etc.)",
                ));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let on = self.matching.is_some();
                let text = if on {
                    tr!("{} 追加中…", "{} Adding…", icon::PLUS)
                } else {
                    tr!("{} 追加", "{} Add", icon::PLUS)
                };
                if ui
                    .add(egui::Button::new(text).selected(on))
                    .on_hover_text(tl(
                        "スキャン表示で、2 つのセッションの同じ特徴を順にクリックします。クリックで\
                         拾う点は、ポインタの位置にセッション名と距離で示します（Esc で終わる）",
                        "In the scan view, click the same feature in two sessions in turn. The point a click takes is shown at the pointer with its session and distance (Esc to finish)",
                    ))
                    .clicked()
                {
                    self.matching = if on { None } else { Some(None) };
                    if !on {
                        self.selected_match = None;
                        self.selected_edit = None;
                        self.placing = None;
                        if self.s.view != PreviewSource::Scans {
                            self.s.view = PreviewSource::Scans;
                            self.load_scene(ctx);
                        }
                    }
                }
            });
        });
        // by pair of sessions, in the order of the first match of each
        let mut groups: Vec<(SessionPair, Vec<usize>)> = Vec::new();
        for (i, (a, b)) in self.edits.matches.iter().enumerate() {
            let (ka, kb) = (self.match_session(a), self.match_session(b));
            let key = if kb < ka { (kb, ka) } else { (ka, kb) };
            match groups.iter_mut().find(|g| g.0 == key) {
                Some(g) => g.1.push(i),
                None => groups.push((key, vec![i])),
            }
        }
        let tag = |k: Option<usize>| match k {
            Some(k) => {
                let [r, g, b] = Self::session_color(k);
                egui::RichText::new(Self::session_tag(k)).color(egui::Color32::from_rgb(r, g, b))
            }
            None => egui::RichText::new("?"),
        };
        let mut go = None;
        let mut remove = None;
        let mut select = None;
        for ((ka, kb), list) in &groups {
            ui.horizontal(|ui| {
                ui.add_space(14.0);
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.label(tag(*ka));
                ui.label("＝");
                ui.label(tag(*kb));
                ui.label(tr!("：{} 組", ": {} pairs", list.len()));
                if list.len() < 2 {
                    ui.colored_label(
                        egui::Color32::from_rgb(230, 190, 70),
                        tl("（2 組以上を推奨）", "(two or more recommended)"),
                    )
                    .on_hover_text(tl(
                        "1 組だけだと、向きは点の周りの形だけで探します。離れた場所でもう 1 組以上足すと確実です",
                        "With one pair the heading is found from the shape around the point alone. Another pair or more, apart from it, makes it reliable",
                    ));
                }
            });
            for &i in list {
                let (a, b) = &self.edits.matches[i];
                let selected = self.selected_match == Some(i);
                let gap = self
                    .match_world(a)
                    .zip(self.match_world(b))
                    .map(|(x, y)| (x - y).length());
                let text = match gap {
                    Some(g) => tr!("{}　ずれ {g:.2} m", "{}  {g:.2} m apart", match_tag(i)),
                    None => match_tag(i),
                };
                ui.horizontal(|ui| {
                    ui.add_space(28.0);
                    if ui
                        .selectable_label(selected, text)
                        .on_hover_text(tl(
                            "選ぶとビューで強調します。ビューで点をドラッグすると、同じセッションの別の点に付け直せます。\
                             ずれは今の軌跡での 2 点の距離で、最適化の後に小さくなっていれば合っています",
                            "Selecting it highlights it in the view. Dragging a point in the view picks another point of the same session. The gap is the two points' distance on the current trajectory: small after the optimization when they match",
                        ))
                        .clicked()
                    {
                        select = Some(if selected { None } else { Some(i) });
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
                            .on_hover_text(tl("この点へ移動", "Go to this point"))
                            .clicked()
                        {
                            go = self.match_world(a).map(|p| (i, p));
                        }
                    });
                });
            }
        }
        if let Some(s) = select {
            self.select_match(s);
        }
        if let Some((i, p)) = go {
            self.select_match(Some(i));
            let dist = self.camera.distance.min(12.0);
            self.flight = Some((self.camera, p, Some(dist), Instant::now()));
        }
        if let Some(i) = remove {
            self.remove_match(i);
        }
    }
}
