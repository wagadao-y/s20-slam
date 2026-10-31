//! The start screen shown while no work folder is open.
use super::*;
use crate::lang::tl;

impl App {
    pub(super) fn welcome(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let idle = self.job.is_none();
        let mut open = None;
        ui.vertical_centered(|ui| {
            ui.add_space((ui.available_height() * 0.15).max(16.0));
            ui.label(
                egui::RichText::new(format!("{} S20 SLAM", icon::MAP_TRIFOLD))
                    .size(30.0)
                    .strong(),
            );
            ui.label(
                egui::RichText::new(tl("S20 の生データから、軌跡と点群を作ります", "Makes trajectories and point clouds from S20 raw data"))
                    .size(15.0)
                    .weak(),
            );
            ui.add_space(20.0);
            ui.horizontal(|ui| {
                // the two buttons centred
                let width = 2.0 * 220.0 + ui.spacing().item_spacing.x;
                ui.add_space(((ui.available_width() - width) / 2.0).max(0.0));
                let big = |text: String| {
                    egui::Button::new(egui::RichText::new(text).size(16.0))
                        .min_size(egui::vec2(220.0, 44.0))
                };
                if ui
                    .add_enabled(
                        idle,
                        big(tr!("{}  新規作業フォルダー…", "{}  New work folder…", icon::FOLDER_PLUS)),
                    )
                    .on_hover_text(tl("生データフォルダーを選んで、新しい作業フォルダーを作る", "Choose a raw data folder and create a new work folder"))
                    .clicked()
                {
                    self.new_work = Some(NewWork::new(&self.s, None));
                }
                if ui
                    .add_enabled(idle, big(tr!("{}  開く…", "{}  Open…", icon::FOLDER_OPEN)))
                    .on_hover_text(tl("作業フォルダーを開く", "Open a work folder"))
                    .clicked()
                {
                    self.choose_work(&ctx);
                }
            });
            ui.add_space(24.0);
            if !self.s.recent.is_empty() {
                ui.label(egui::RichText::new(tl("最近の作業フォルダー", "Recent work folders")).strong());
                ui.add_space(4.0);
                for path in self.s.recent.clone() {
                    let exists = path.join("project.json").is_file();
                    let button =
                        egui::Button::new(format!("{}  {}", icon::FOLDER, folder_label(&path)))
                            .min_size(egui::vec2(460.0, 0.0))
                            .right_text(
                                // without the extended-length prefix of a canonical path
                                egui::RichText::new(
                                    path.display()
                                        .to_string()
                                        .trim_start_matches(r"\\?\")
                                        .to_string(),
                                )
                                .weak(),
                            );
                    if ui
                        .add_enabled(idle && exists, button)
                        .on_disabled_hover_text(tl("作業フォルダーが見つかりません", "Work folder not found"))
                        .clicked()
                    {
                        open = Some(path);
                    }
                }
                ui.add_space(4.0);
                if ui
                    .button(tr!("{} 履歴をクリア", "{} Clear history", icon::TRASH))
                    .on_hover_text(
                        tl("最近の作業フォルダーの一覧を空にします（フォルダーは消えません）", "Empties the list of recent work folders (the folders are not deleted)"),
                    )
                    .clicked()
                {
                    self.s.recent.clear();
                }
                ui.add_space(24.0);
            }
            ui.weak(tr!(
                "{}  作業フォルダーか生データフォルダーをウィンドウにドロップしても始められます\
                 （生データなら新規作成）", "{}  You can also start by dropping a work folder or a raw data folder on the window (a raw data folder creates a new one)",
                icon::FILE_ARROW_DOWN
            ));
        });
        if let Some(path) = open {
            self.open_folder(&ctx, path);
        }
    }

    /// Back to the start screen.
    pub(super) fn close_work(&mut self) {
        self.clear_view();
        self.s.work.clear();
        self.error = None;
        self.reload_edits();
        self.refresh_steps();
        self.log
            .push(tl("作業フォルダーを閉じました", "Closed the work folder").into());
    }
}
