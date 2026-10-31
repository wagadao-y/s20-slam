//! Editor of the processing parameters: the parameters drawn as their JSON tree (numbers, switches,
//! lists, nested groups) with Japanese labels, and the values that differ from the preset marked.
//! Walking the serialised tree keeps every parameter editable without a widget per field.

use crate::lang::tl;
use eframe::egui;
use egui_phosphor::regular as icon;
use s20_pipeline::ProcessParams;
use serde_json::{Value, json};

/// Colour of the values that differ from the preset.
pub const CHANGED: egui::Color32 = egui::Color32::from_rgb(230, 190, 70);

/// Number fields that may also be unset (`None`).
const NULLABLE: &[&str] = &["rtk_c"];

/// Draws the editor. `p` is replaced when a value changed and the result is still valid.
/// `expand`: the sections start open.
pub fn show(
    ui: &mut egui::Ui,
    p: &mut ProcessParams,
    preset: &ProcessParams,
    expand: bool,
    ppk: bool,
) {
    let mut v = serde_json::to_value(&*p).expect("parameters serialise");
    let base = serde_json::to_value(preset).expect("parameters serialise");
    let mut changed = false;
    if let Value::Object(m) = &mut v {
        for (key, child) in m.iter_mut() {
            if key == "ppk" && !ppk {
                continue;
            }
            let b = base.get(key);
            let title = header(section_title(key), Some(&*child) != b);
            egui::CollapsingHeader::new(title)
                .id_salt(key)
                .default_open(expand)
                .show(ui, |ui| changed |= object(ui, key, key, child, b));
        }
    }
    if changed && let Ok(n) = serde_json::from_value::<ProcessParams>(v) {
        *p = n;
    }
}

fn header(label: &str, changed: bool) -> egui::RichText {
    let t = egui::RichText::new(label).strong();
    if changed { t.color(CHANGED) } else { t }
}

/// Groups are drawn as collapsible sections below the plain values.
fn is_group(v: &Value) -> bool {
    match v {
        Value::Object(_) => true,
        Value::Array(a) => a.first().is_some_and(Value::is_object),
        _ => false,
    }
}

fn object(
    ui: &mut egui::Ui,
    section: &str,
    path: &str,
    v: &mut Value,
    base: Option<&Value>,
) -> bool {
    let Value::Object(m) = v else {
        return false;
    };
    let mut changed = false;
    egui::Grid::new(path)
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            for (k, c) in m.iter_mut().filter(|(_, c)| !is_group(c)) {
                let b = base.and_then(|b| b.get(k));
                let (label, hint) = info(section, k);
                let mut text = egui::RichText::new(label);
                if Some(&*c) != b {
                    text = text.color(CHANGED);
                }
                let mut tip = String::new();
                if !hint.is_empty() {
                    tip.push_str(hint);
                    tip.push_str("\n\n");
                }
                tip.push_str(&tr!("JSON キー: {k}", "JSON key: {k}"));
                if let Some(b) = b {
                    tip.push_str(&tr!(
                        "\nプリセット値: {}",
                        "\nPreset value: {}",
                        show_value(b)
                    ));
                }
                ui.label(text).on_hover_text(tip);
                changed |= leaf(ui, k, c, b);
                ui.end_row();
            }
        });
    for (k, c) in m.iter_mut().filter(|(_, c)| is_group(c)) {
        let b = base.and_then(|b| b.get(k));
        let id = format!("{path}.{k}");
        let title = header(info(section, k).0, Some(&*c) != b);
        egui::CollapsingHeader::new(title)
            .id_salt(&id)
            .show(ui, |ui| match c {
                Value::Array(items) => changed |= list(ui, section, &id, items, b),
                _ => changed |= object(ui, section, &id, c, b),
            });
    }
    changed
}

/// A list of groups (the optimisation stages): each one, and adding or removing one.
fn list(
    ui: &mut egui::Ui,
    section: &str,
    path: &str,
    items: &mut Vec<Value>,
    base: Option<&Value>,
) -> bool {
    let mut changed = false;
    let mut remove = None;
    let n = items.len();
    for (i, item) in items.iter_mut().enumerate() {
        let b = base.and_then(|b| b.get(i));
        let id = format!("{path}[{i}]");
        ui.horizontal(|ui| {
            let title = header(&tr!("段階 {}", "Stage {}", i + 1), Some(&*item) != b);
            egui::CollapsingHeader::new(title)
                .id_salt(&id)
                .show(ui, |ui| changed |= object(ui, section, &id, item, b));
            if n > 1
                && ui
                    .button(icon::TRASH)
                    .on_hover_text(tl("この段階を削除", "Delete this stage"))
                    .clicked()
            {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        items.remove(i);
        changed = true;
    }
    if let Some(last) = items.last().cloned()
        && ui
            .button(tr!("{}  段階を追加", "{}  Add stage", icon::PLUS))
            .on_hover_text(tl(
                "最後の段階を複製して追加します",
                "Duplicates the last stage and adds it",
            ))
            .clicked()
    {
        items.push(last);
        changed = true;
    }
    changed
}

fn leaf(ui: &mut egui::Ui, key: &str, v: &mut Value, base: Option<&Value>) -> bool {
    match v {
        Value::Bool(x) => ui.checkbox(x, "").changed(),
        Value::Number(_) if NULLABLE.contains(&key) => {
            let mut changed = false;
            ui.horizontal(|ui| {
                changed |= number(ui, v);
                let mut on = true;
                if ui.checkbox(&mut on, tl("指定", "Set")).changed() {
                    *v = Value::Null;
                    changed = true;
                }
            });
            changed
        }
        Value::Number(_) => number(ui, v),
        Value::Null => {
            let mut on = false;
            let r = ui
                .checkbox(&mut on, tl("未指定", "Unset"))
                .on_hover_text(tl("チェックすると値を指定できます", "Check to set a value"));
            if r.changed() {
                *v = base
                    .filter(|b| b.is_number())
                    .cloned()
                    .unwrap_or(json!(0.1));
                return true;
            }
            false
        }
        Value::Array(a) => {
            let mut changed = false;
            ui.horizontal_wrapped(|ui| {
                for x in a.iter_mut() {
                    changed |= number(ui, x);
                }
                // only the plane levels have a free length; the others are fixed vectors
                if key == "levels" {
                    if ui
                        .button(icon::PLUS)
                        .on_hover_text(tl("細かい段を追加", "Add a finer level"))
                        .clicked()
                        && let Some(l) = a.last().and_then(Value::as_f64)
                    {
                        a.push(json!(l / 2.0));
                        changed = true;
                    }
                    if a.len() > 1
                        && ui
                            .button(icon::MINUS)
                            .on_hover_text(tl("最後の段を削除", "Delete the last level"))
                            .clicked()
                    {
                        a.pop();
                        changed = true;
                    }
                }
            });
            changed
        }
        other => {
            ui.label(other.to_string());
            false
        }
    }
}

fn number(ui: &mut egui::Ui, v: &mut Value) -> bool {
    if let Some(mut x) = v.as_u64() {
        let speed = (x as f64 * 0.01).max(0.2);
        let r = ui.add(egui::DragValue::new(&mut x).speed(speed));
        if r.changed() {
            *v = json!(x);
        }
        return r.changed();
    }
    if let Some(mut x) = v.as_i64() {
        let speed = (x.abs() as f64 * 0.01).max(0.2);
        let r = ui.add(egui::DragValue::new(&mut x).speed(speed));
        if r.changed() {
            *v = json!(x);
        }
        return r.changed();
    }
    let Some(mut x) = v.as_f64() else {
        return false;
    };
    let speed = x.abs().max(1e-3) * 0.01;
    let r = ui.add(
        egui::DragValue::new(&mut x)
            .speed(speed)
            .custom_formatter(|x, _| round_sig(x).to_string())
            .custom_parser(|s| s.trim().parse().ok()),
    );
    if r.changed() {
        *v = json!(round_sig(x));
    }
    r.changed()
}

/// Six significant digits: dragging (and f32 fields) would otherwise show long tails.
fn round_sig(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    format!("{x:.5e}").parse().unwrap_or(x)
}

fn show_value(v: &Value) -> String {
    match v {
        Value::Number(n) => n
            .as_f64()
            .map_or_else(|| n.to_string(), |x| round_sig(x).to_string()),
        Value::Null => tl("未指定", "Unset").into(),
        Value::Array(a) if a.iter().all(Value::is_number) => {
            let s: Vec<String> = a.iter().map(show_value).collect();
            format!("[{}]", s.join(", "))
        }
        Value::Bool(true) => tl("オン", "On").into(),
        Value::Bool(false) => tl("オフ", "Off").into(),
        _ => tl("（グループ）", "(group)").into(),
    }
}

fn section_title(key: &str) -> &str {
    match key {
        "lio" => tl(
            "LIO（逐次の自己位置推定）",
            "LIO (incremental localization)",
        ),
        "opt" => tl("軌跡の最適化", "Trajectory optimization"),
        "rtk" => "RTK",
        "ppk" => tl("PPK 測位", "PPK positioning"),
        "cloud" => tl("点群生成", "Point cloud"),
        "color" => tl("色付け", "Colorization"),
        k => k,
    }
}

/// Label and explanation of a parameter (the key itself when unknown).
fn info<'a>(section: &str, key: &'a str) -> (&'a str, &'static str) {
    let found = LABELS
        .iter()
        .find(|(s, k, _, _)| (*s == section || s.is_empty()) && *k == key);
    match found {
        Some((_, _, l, h)) => (tl(l[0], l[1]), tl(h[0], h[1])),
        None => (key, ""),
    }
}

/// (section or "" for any, key, label, explanation)
const LABELS: &[(&str, &str, [&str; 2], [&str; 2])] = &[
    // shared
    (
        "",
        "rmin",
        ["最小距離 (m)", "Min range (m)"],
        [
            "センサーからこれより近い点は使いません",
            "Points closer to the sensor than this are not used",
        ],
    ),
    (
        "",
        "rmax",
        ["最大距離 (m)", "Max range (m)"],
        [
            "センサーからこれより遠い点は使いません",
            "Points farther from the sensor than this are not used",
        ],
    ),
    (
        "",
        "lidar_imu_t",
        ["LiDAR→IMU の並進 (m)", "LiDAR→IMU translation (m)"],
        [
            "機材の校正値です。通常は変更しません",
            "Calibration value of the hardware. Normally not changed",
        ],
    ),
    (
        "",
        "imu_shift",
        ["IMU 時刻の補正 (s)", "IMU time correction (s)"],
        [
            "IMU の時刻からこの値を引きます",
            "This value is subtracted from the IMU times",
        ],
    ),
    ("", "ng", ["ジャイロの雑音", "Gyro noise"], ["", ""]),
    ("", "na", ["加速度の雑音", "Accelerometer noise"], ["", ""]),
    (
        "",
        "nbg",
        ["ジャイロバイアスの変動", "Gyro bias random walk"],
        ["", ""],
    ),
    (
        "",
        "nba",
        ["加速度バイアスの変動", "Accelerometer bias random walk"],
        ["", ""],
    ),
    // LIO
    (
        "lio",
        "static_sec",
        ["開始時の静止時間 (s)", "Initial still time (s)"],
        [
            "最初のこの時間は静止しているとみなし、重力の向きとバイアスを初期化します",
            "The first this many seconds are taken as still, to initialize the gravity direction and biases",
        ],
    ),
    (
        "lio",
        "down",
        ["スキャンの間引き (m)", "Scan thinning (m)"],
        [
            "位置合わせに使う点のボクセルの大きさ",
            "Voxel size of the points used for registration",
        ],
    ),
    (
        "lio",
        "map_vox",
        ["地図のボクセル (m)", "Map voxel (m)"],
        ["", ""],
    ),
    (
        "lio",
        "max_corr",
        ["対応点の最大距離 (m)", "Max correspondence distance (m)"],
        ["", ""],
    ),
    (
        "lio",
        "lru",
        ["地図ボクセルの保持数", "Map voxels kept"],
        [
            "これを超えると古いボクセルから捨てます",
            "Beyond this the oldest voxels are dropped",
        ],
    ),
    (
        "lio",
        "iters",
        ["反復回数", "Iterations"],
        [
            "1 スキャンあたりの位置合わせの反復回数",
            "Registration iterations per scan",
        ],
    ),
    (
        "lio",
        "min_inlier",
        ["最小一致率", "Min inlier ratio"],
        [
            "一致率がこれ未満のスキャンは位置合わせの結果を使いません",
            "The registration of scans with an inlier ratio below this is not used",
        ],
    ),
    (
        "lio",
        "lidar_scale",
        ["LiDAR の重み", "LiDAR weight"],
        [
            "IMU に対する LiDAR の拘束の重み",
            "Weight of the LiDAR constraints relative to the IMU",
        ],
    ),
    (
        "lio",
        "num_neighbors",
        ["共分散の近傍点数", "Neighbours for covariances"],
        ["", ""],
    ),
    (
        "lio",
        "search7",
        ["隣のボクセルも探す", "Search the neighbouring voxels"],
        [
            "対応点を、点の入るボクセルと面で接する 6 つのボクセルからも探します。既定はオフ",
            "Also looks for correspondences in the 6 voxels sharing a face with the point's voxel. Off by default",
        ],
    ),
    (
        "lio",
        "huber",
        ["位置合わせの頑健化の閾値", "Robust registration threshold"],
        [
            "残差（マハラノビス距離）がこれを超える点の重みを下げます。0 で無効",
            "Lowers the weight of points whose residual (Mahalanobis distance) exceeds this. 0 disables it",
        ],
    ),
    (
        "lio",
        "grav",
        ["重力で傾きを補正", "Correct the tilt with gravity"],
        [
            "加速度から求めた鉛直の向きで、スキャンごとに傾き（ロール・ピッチ）を補正します。下の重力の観測の項目はこれが有効なときだけ使います",
            "Corrects each scan's tilt (roll, pitch) with the vertical found from the accelerations. The gravity observation items below are only used when this is on",
        ],
    ),
    (
        "lio",
        "grav_sd",
        [
            "重力の観測の標準偏差（度）",
            "Gravity observation standard deviation (deg)",
        ],
        [
            "スキャンごとの鉛直の向きの観測の確からしさ。小さいほど強く効きます",
            "Confidence of each scan's vertical observation. Smaller acts more strongly",
        ],
    ),
    (
        "lio",
        "grav_win",
        ["重力の観測の窓（秒）", "Gravity observation window (s)"],
        [
            "鉛直の向きを求めるのに加速度を積分する長さ",
            "Length over which the accelerations are integrated to find the vertical",
        ],
    ),
    (
        "lio",
        "tilt_lidar",
        ["傾きへの LiDAR の重み", "LiDAR weight on the tilt"],
        [
            "重力で補正するとき、LiDAR の位置合わせが傾きを決める強さに掛ける係数。1 で変更なし。小さくすると傾きを IMU と重力に任せます",
            "With the gravity correction, a factor on how strongly the LiDAR registration decides the tilt. 1 leaves it unchanged. Smaller leaves the tilt to the IMU and gravity",
        ],
    ),
    (
        "lio",
        "grav_nba",
        [
            "重力で補正するときの加速度計のバイアスの変動",
            "Accelerometer bias random walk with gravity correction",
        ],
        [
            "重力で補正するとき、上の「加速度計のバイアスの変動」の代わりに使います。鉛直の向きだけでは水平のバイアスと傾きを区別できないため、バイアスはほぼ一定にします",
            "Used instead of the accelerometer bias random walk above with the gravity correction. The vertical alone can't tell the horizontal bias from the tilt, so the bias is kept nearly constant",
        ],
    ),
    (
        "lio",
        "ba0",
        [
            "加速度計のバイアスの初期値（m/s²）",
            "Initial accelerometer bias (m/s²)",
        ],
        [
            "機体座標系（x, y, z）。静止した開始時の重力の向きからも差し引きます。既定値はこの S20 を PPK で最適化したときの推定値。機体を替えたときは、PPK で最適化したセッションのログに出る推定値を入れてください",
            "In the device frame (x, y, z). Also subtracted from the gravity direction at the still start. The default is the estimate from optimizing this S20 with PPK. For another device, enter the estimate printed in the log of a session optimized with PPK",
        ],
    ),
    (
        "lio",
        "gate_nis",
        [
            "IMU との食い違いの検査（NIS）",
            "Disagreement check with the IMU (NIS)",
        ],
        [
            "位置合わせの補正が IMU の予測から見てこの値（NIS）を超えたら、予測の近くで対応点を探し直します。誤った対応で位置がずるっとずれるのを防ぎます。0 で無効",
            "When the registration's correction exceeds this value (NIS) as seen from the IMU prediction, correspondences are searched again near the prediction. Prevents the position slipping with wrong correspondences. 0 disables it",
        ],
    ),
    (
        "lio",
        "gate_corr",
        [
            "探し直しの対応点の最大距離 (m)",
            "Max correspondence distance of the re-search (m)",
        ],
        [
            "上の検査で探し直すときの対応点の最大距離。探し直しも検査を通らなければ、最初の結果を使います",
            "Max correspondence distance when searching again after the check above. If the re-search also fails the check, the first result is used",
        ],
    ),
    // unified optimisation
    (
        "opt",
        "every",
        ["状態を置く間隔（スキャン数）", "State interval (scans)"],
        [
            "何スキャンごとに状態（ノット）を置くか。大きいほど速く、粗くなります",
            "Every how many scans a state (knot) is placed. Larger is faster and coarser",
        ],
    ),
    ("opt", "noise", ["IMU の雑音", "IMU noise"], ["", ""]),
    (
        "opt",
        "sigma_pt",
        ["点の雑音 (m)", "Point noise (m)"],
        ["", ""],
    ),
    (
        "opt",
        "lidar_w",
        ["LiDAR 情報の倍率", "LiDAR information factor"],
        [
            "点同士の相関を考えた、LiDAR の情報の追加の倍率",
            "Additional factor on the LiDAR information, accounting for the correlation between points",
        ],
    ),
    (
        "opt",
        "max_gap",
        [
            "IMU 拘束を切る間隔 (s)",
            "Gap that cuts IMU constraints (s)",
        ],
        [
            "ノットの間がこれより空くと IMU の拘束を入れません",
            "No IMU constraint is added between knots further apart than this",
        ],
    ),
    (
        "opt",
        "stages",
        ["段階（粗→細）", "Stages (coarse → fine)"],
        ["上から順に実行します", "Run from the top down"],
    ),
    ("opt", "iters", ["反復回数", "Iterations"], ["", ""]),
    ("opt", "planes", ["面の抽出", "Plane extraction"], ["", ""]),
    (
        "opt",
        "levels",
        [
            "ボクセルの大きさ（粗→細, m）",
            "Voxel sizes (coarse → fine, m)",
        ],
        ["", ""],
    ),
    (
        "opt",
        "min_pts",
        ["面の最小点数", "Min points per plane"],
        ["", ""],
    ),
    (
        "opt",
        "max_std",
        ["面の厚さの上限 (m)", "Max plane thickness (m)"],
        ["", ""],
    ),
    (
        "opt",
        "ratio",
        ["平面度の閾値", "Planarity threshold"],
        [
            "小さいほど平らなものだけを面とみなします",
            "Smaller takes only flatter things as planes",
        ],
    ),
    (
        "opt",
        "c_cauchy",
        ["Cauchy の尺度 (m)", "Cauchy scale (m)"],
        [
            "これより大きい残差の点を弱く扱います",
            "Points with larger residuals than this are treated weakly",
        ],
    ),
    (
        "opt",
        "couple",
        ["面の重心を結合", "Couple the plane centroids"],
        ["", ""],
    ),
    (
        "opt",
        "rtk_c",
        ["RTK の Cauchy の尺度 (m)", "RTK Cauchy scale (m)"],
        [
            "未指定なら RTK の設定値を使います",
            "Unset uses the RTK setting",
        ],
    ),
    (
        "opt",
        "odometry",
        ["姿勢グラフの段階", "Pose graph stage"],
        [
            "面を使わず、オドメトリで形を保つ段階",
            "A stage that keeps the shape with odometry, without planes",
        ],
    ),
    (
        "opt",
        "lambda",
        ["LM の初期減衰", "LM initial damping"],
        ["", ""],
    ),
    ("opt", "cg_tol", ["CG の収束判定", "CG tolerance"], ["", ""]),
    (
        "opt",
        "loop_cg_tol",
        [
            "CG の収束判定（ループあり、面の段階）",
            "CG tolerance (with loops, plane stages)",
        ],
        [
            "ループがあるときの面の段階の収束判定です。ループの補正を軌跡全体に広げる姿勢グラフの段階は、常に厳密に解きます",
            "Tolerance of the plane stages when there are loops. The pose graph stage, which spreads the loop corrections over the whole trajectory, is always solved exactly",
        ],
    ),
    (
        "opt",
        "cg_max",
        ["CG の最大反復", "CG max iterations"],
        ["", ""],
    ),
    ("opt", "lm_tries", ["LM の試行回数", "LM tries"], ["", ""]),
    (
        "opt",
        "loop_pre_iters",
        ["ループ前段の反復回数", "Loop pre-stage iterations"],
        [
            "ループがあるとき、最初に姿勢グラフで補正を軌跡全体に広げる反復",
            "With loops, the iterations that first spread the correction over the whole trajectory with the pose graph",
        ],
    ),
    (
        "opt",
        "extra_iters",
        [
            "最終段の追加反復（上限）",
            "Extra iterations of the last stage (max)",
        ],
        [
            "収束するまで最終段をこの回数まで追加で回します",
            "The last stage runs up to this many extra times until it converges",
        ],
    ),
    (
        "opt",
        "conv_mm",
        ["収束判定：並進 (mm)", "Convergence: translation (mm)"],
        ["", ""],
    ),
    (
        "opt",
        "conv_deg",
        ["収束判定：回転 (deg)", "Convergence: rotation (deg)"],
        ["", ""],
    ),
    (
        "opt",
        "loop_reg",
        ["ループの位置合わせ", "Aligning the loops"],
        [
            "指定したループの 2 時刻の周りの局所地図を位置合わせします",
            "Registers the local maps around the two times of each given loop",
        ],
    ),
    (
        "opt",
        "half_s",
        ["局所地図の時間幅 ± (s)", "Local map time span ± (s)"],
        [
            "ループの各時刻の前後この秒数のスキャンで局所地図を作ります。植生が多く構造物が少ない場所では長め（10 s 程度）にすると一致しやすくなります",
            "The local map is built from the scans this many seconds before and after each loop time. Where there is much vegetation and few structures, longer (about 10 s) matches more easily",
        ],
    ),
    (
        "opt",
        "radius",
        ["局所地図の半径 (m)", "Local map radius (m)"],
        ["", ""],
    ),
    (
        "opt",
        "min_inliers",
        ["採用する一致率の下限", "Min inlier ratio accepted"],
        ["", ""],
    ),
    (
        "opt",
        "max_spread",
        [
            "推定のばらつきの上限 (m)",
            "Max spread of the estimates (m)",
        ],
        [
            "初期値を変えた推定どうしがこれ以上ずれると、あいまいとして採用しません",
            "If estimates from different initial values differ by more than this, it's ambiguous and not accepted",
        ],
    ),
    (
        "opt",
        "sig_r_deg",
        [
            "ループ拘束の標準偏差：回転 (deg)",
            "Loop constraint standard deviation: rotation (deg)",
        ],
        ["", ""],
    ),
    (
        "opt",
        "sig_t",
        [
            "ループ拘束の標準偏差：並進 (m)",
            "Loop constraint standard deviation: translation (m)",
        ],
        ["", ""],
    ),
    (
        "opt",
        "remeasure",
        ["ループを測り直す", "Remeasure the loops"],
        [
            "面を使う段階ごとに、その時点の軌跡で局所地図を作り直してループを測り直します。最初の測定は LIO の軌跡の歪みで偏るためです",
            "At each plane stage, the local maps are rebuilt with the trajectory at that point and the loops remeasured, since the first measurement is biased by the LIO trajectory's distortion",
        ],
    ),
    // RTK
    (
        "rtk",
        "sig_h",
        ["水平の標準偏差 (m)", "Horizontal standard deviation (m)"],
        ["", ""],
    ),
    (
        "ppk",
        "interval",
        ["測位間隔 (s)", "Solution interval (s)"],
        [
            "0.05～10 秒、0.05 秒刻み。間引く場合も全フレームを復号して位相の途切れを追跡します。",
            "0.05 to 10 s in 0.05 s steps. When thinned, all frames are still decoded to track phase breaks.",
        ],
    ),
    (
        "ppk",
        "elevation_deg",
        ["衛星の最低仰角 (度)", "Satellite elevation mask (deg)"],
        ["5～45 度", "5 to 45 deg"],
    ),
    (
        "ppk",
        "phase_snr_error",
        [
            "信号強度による位相誤差 (m)",
            "Phase error from signal strength (m)",
        ],
        [
            "52 dB-Hz を基準に、弱い信号の拘束を弱めます。0 で無効。整数確定の比の閾値は変わりません。",
            "Weakens the constraints of weak signals relative to 52 dB-Hz. 0 disables it. The ambiguity ratio threshold is unchanged.",
        ],
    ),
    (
        "ppk",
        "prefer_base_l2c",
        [
            "基準局の GPS L2C を優先",
            "Prefer the base station's GPS L2C",
        ],
        [
            "ローバーが L2S のみを追尾する衛星で、基準局の L2X を使います。受信機の組合せによって前方解・後方解の整合性が悪化するため、比較検証用の任意設定です。",
            "Uses the base station's L2X for satellites the rover tracks on L2S only. Some receiver combinations worsen the agreement of the forward and backward solutions, so this is an optional setting for comparison.",
        ],
    ),
    (
        "ppk",
        "estimate_troposphere",
        [
            "基線の対流圏差を推定",
            "Estimate the baseline's tropospheric difference",
        ],
        [
            "Saastamoinen モデルに加えて遅延差を推定します。",
            "Estimates the delay difference on top of the Saastamoinen model.",
        ],
    ),
    (
        "ppk",
        "frequencies",
        ["使用する周波数の数", "Number of frequencies"],
        [
            "1～3。電離層差の推定には 2 以上が必要です。",
            "1 to 3. Estimating the ionospheric difference needs 2 or more.",
        ],
    ),
    ("ppk", "gps", ["GPS", "GPS"], ["", ""]),
    (
        "ppk",
        "glonass",
        ["GLONASS", "GLONASS"],
        [
            "異種受信機のため整数確定には使わず、FLOAT 測位に使います。",
            "Not used for ambiguity fixing because the receivers differ; used for FLOAT solutions.",
        ],
    ),
    ("ppk", "galileo", ["Galileo", "Galileo"], ["", ""]),
    ("ppk", "beidou", ["BeiDou", "BeiDou"], ["", ""]),
    (
        "ppk",
        "qzss",
        ["QZSS（みちびき）", "QZSS (Michibiki)"],
        ["", ""],
    ),
    (
        "ppk",
        "estimate_ionosphere",
        [
            "基線の電離層差を推定",
            "Estimate the baseline's ionospheric difference",
        ],
        [
            "基準局との距離が長い場合に有効です。",
            "Useful when the base station is far.",
        ],
    ),
    (
        "ppk",
        "float_sigma_h",
        [
            "FLOAT の水平標準偏差の下限 (m)",
            "Min FLOAT horizontal standard deviation (m)",
        ],
        [
            "整数未確定の誤差を考慮し、RTKLIB の推定値が小さくても拘束を弱めます。",
            "Accounts for unresolved ambiguities by weakening the constraint even when RTKLIB's estimate is small.",
        ],
    ),
    (
        "ppk",
        "float_sigma_v",
        [
            "FLOAT の鉛直標準偏差の下限 (m)",
            "Min FLOAT vertical standard deviation (m)",
        ],
        ["", ""],
    ),
    (
        "ppk",
        "ar_ratio",
        ["整数アンビギュイティ比の閾値", "Ambiguity ratio threshold"],
        ["2 以上。既定値 3", "2 or more. Default 3"],
    ),
    (
        "ppk",
        "fix_check_sd_h",
        [
            "前方解・後方解の照合：水平標準偏差の下限 (m)",
            "Forward/backward check: min horizontal standard deviation (m)",
        ],
        [
            "前方解と後方解の FIX 位置を 4σ で照合するときの σ の下限。FIX の推定誤差は数 mm と過小なため、cm 程度の差は許容し、整数の誤確定による dm 級の食い違いだけを FLOAT に落とします。0～0.1 m、0 で RTKLIB の推定値のみ。既定値 0.01",
            "Lower bound on σ when the forward and backward FIX positions are compared at 4σ. FIX error estimates of a few mm are too small, so cm-level differences are tolerated and only dm-level disagreements from wrong ambiguity fixes drop to FLOAT. 0 to 0.1 m; 0 uses RTKLIB's estimate only. Default 0.01",
        ],
    ),
    (
        "ppk",
        "fix_check_sd_v",
        [
            "前方解・後方解の照合：鉛直標準偏差の下限 (m)",
            "Forward/backward check: min vertical standard deviation (m)",
        ],
        ["0～0.2 m、既定値 0.02", "0 to 0.2 m, default 0.02"],
    ),
    (
        "ppk",
        "float_check_sd_h",
        [
            "片方向 FIX の照合：FLOAT 側の水平標準偏差の下限 (m)",
            "One-sided FIX check: min horizontal standard deviation of the FLOAT side (m)",
        ],
        [
            "片方だけ FIX した解をもう片方の FLOAT 解と 4σ で照合するとき、FLOAT 側に加える σ の下限。収束中や基線が長いときの FLOAT は推定誤差が数 mm でも実際は dm ずれるため、その程度の差では FIX を捨てず、それを超える食い違いだけ標準偏差を大きくします（既定値で 4σ は水平の各軸 約 40 cm・鉛直 約 80 cm）。0～0.5 m、0 で FIX 同士と同じ照合。既定値 0.10",
            "Lower bound on the σ added to the FLOAT side when a FIX from one direction is compared with the other direction's FLOAT at 4σ. A converging or long-baseline FLOAT may be off by dm even with an estimated error of a few mm, so such differences don't discard the FIX; only larger disagreements enlarge its standard deviation (with the defaults 4σ is about 40 cm per horizontal axis and 80 cm vertically). 0 to 0.5 m; 0 checks like two FIXes. Default 0.10",
        ],
    ),
    (
        "ppk",
        "float_check_sd_v",
        [
            "片方向 FIX の照合：FLOAT 側の鉛直標準偏差の下限 (m)",
            "One-sided FIX check: min vertical standard deviation of the FLOAT side (m)",
        ],
        ["0～1 m、既定値 0.20", "0 to 1 m, default 0.20"],
    ),
    (
        "ppk",
        "use_float",
        [
            "FLOAT も品質条件で採用",
            "Also accept FLOAT by the quality conditions",
        ],
        [
            "既定は FIX のみ。FLOAT も推定誤差の上限を満たした点だけを使います。",
            "By default FIX only. FLOAT positions are also used if they meet the error limit.",
        ],
    ),
    (
        "ppk",
        "max_sd_h",
        [
            "採用する水平標準偏差の上限 (m)",
            "Max horizontal standard deviation accepted (m)",
        ],
        [
            "東・北の標準偏差の二乗和平方根",
            "Root sum square of the east and north standard deviations",
        ],
    ),
    (
        "ppk",
        "max_sd_v",
        [
            "採用する鉛直標準偏差の上限 (m)",
            "Max vertical standard deviation accepted (m)",
        ],
        ["", ""],
    ),
    (
        "rtk",
        "sig_v",
        ["鉛直の標準偏差 (m)", "Vertical standard deviation (m)"],
        ["", ""],
    ),
    (
        "rtk",
        "c",
        ["Huber の閾値 (m)", "Huber threshold (m)"],
        [
            "これを超える残差でも一定の力で引き寄せます。ドリフトした軌跡から離れた正しい測位を捨てないためです。誤 FIX は事前の検出で除きます。",
            "Residuals beyond this still pull with a constant force, so correct positions away from a drifted trajectory aren't discarded. Wrong FIXes are removed by the detection beforehand.",
        ],
    ),
    (
        "rtk",
        "check",
        [
            "誤 FIX の検出（LIO との整合）",
            "Wrong FIX detection (agreement with LIO)",
        ],
        [
            "測位と LIO の差が連続する測位の間で段差になった区間を誤 FIX として除きます。",
            "Sections where the difference between the positions and LIO steps between consecutive positions are removed as wrong FIXes.",
        ],
    ),
    (
        "rtk",
        "enabled",
        ["誤 FIX を検出する", "Detect wrong FIXes"],
        ["", ""],
    ),
    (
        "rtk",
        "step_h",
        [
            "誤 FIX とみなす水平の段差 (m)",
            "Horizontal step taken as a wrong FIX (m)",
        ],
        [
            "連続する測位の間の段差。これより小さい誤差は重みの頑健化と点群の拘束に任せます。既定値 0.15",
            "Step between consecutive positions. Smaller errors are left to the robust weights and the point cloud constraints. Default 0.15",
        ],
    ),
    (
        "rtk",
        "step_v",
        [
            "誤 FIX とみなす鉛直の段差 (m)",
            "Vertical step taken as a wrong FIX (m)",
        ],
        ["既定値 0.30", "Default 0.30"],
    ),
    (
        "rtk",
        "lio_tol",
        [
            "LIO の乱れとみなす位置と速度の食い違い (m)",
            "Position and velocity disagreement taken as LIO disturbance (m)",
        ],
        [
            "LIO がこれ以上乱れた時刻の近くの段差は、測位ではなく LIO の跳びとみなします。既定値 0.03",
            "Steps near times when LIO is disturbed by more than this are taken as LIO jumps, not position errors. Default 0.03",
        ],
    ),
    (
        "rtk",
        "max_gap",
        [
            "誤 FIX の区間を広げる欠測の長さ (s)",
            "Gap length that extends wrong FIX sections (s)",
        ],
        [
            "誤 FIX と判定した区間から、これ以下の欠測をはさんで同じずれが続く区間も除きます。既定値 10",
            "Sections where the same offset continues across gaps no longer than this from a section judged as wrong FIX are removed too. Default 10",
        ],
    ),
    (
        "rtk",
        "lever",
        [
            "アンテナの位置（IMU 座標, m）",
            "Antenna position (IMU frame, m)",
        ],
        [
            "機材の校正値です。通常は変更しません",
            "Calibration value of the hardware. Normally not changed",
        ],
    ),
    // cloud
    (
        "cloud",
        "trim_still",
        [
            "始点・終点で外す範囲 (m)",
            "Range trimmed at the start and end (m)",
        ],
        [
            "機材を置いた位置からこの距離だけ動き出すまでと、最後に置く位置のこの距離内に戻ってからのスキャンを点群に使いません（後ろに立つ撮影者が映り込むため）。0 で使います",
            "Scans until the device moves this far from where it was set down, and after it returns within this distance of where it's set down at the end, are not used in the point cloud (the operator standing behind shows up). 0 uses them",
        ],
    ),
    (
        "cloud",
        "max_range",
        ["読み込む最大距離 (m)", "Max range read (m)"],
        [
            "これより遠くで測った点は使いません",
            "Points measured farther than this are not used",
        ],
    ),
    (
        "cloud",
        "min_range",
        ["出力する最小距離 (m)", "Min range written (m)"],
        [
            "これより近くで測った点は出力しません（面の推定には使います）",
            "Points measured closer than this are not written (they are used to estimate planes)",
        ],
    ),
    (
        "cloud",
        "support_min_range",
        ["面に使う最小距離 (m)", "Min range for planes (m)"],
        [
            "これより近くで測った点は面の推定にも使いません",
            "Points measured closer than this are not used for planes either",
        ],
    ),
    (
        "cloud",
        "moment_cell",
        ["面の集計セル (m)", "Plane moment cell (m)"],
        [
            "全観測の点数・和・2 次モーメントをこの大きさのセルで集計し、面を当てます",
            "The counts, sums and second moments of all observations are accumulated in cells of this size, and planes fitted",
        ],
    ),
    (
        "cloud",
        "min_support",
        ["面の最小観測数", "Min observations per plane"],
        ["", ""],
    ),
    (
        "cloud",
        "fallback_scales",
        ["小半径での再試行の回数", "Retries at a smaller radius"],
        [
            "面が当たらない・外れる点は半径を半分にして当て直します",
            "Points where no plane fits or that fall off are refitted at half the radius",
        ],
    ),
    (
        "cloud",
        "fast_plane",
        [
            "平らな所は候補探しを省く（高速化）",
            "Skip the candidate search on flat areas (faster)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "consensus_seeds",
        ["面の候補の数", "Plane candidates"],
        ["", ""],
    ),
    (
        "cloud",
        "consensus_seed_radius",
        [
            "面の候補を探す半径 (m)",
            "Plane candidate search radius (m)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "anchor_grid",
        ["面を共有する格子 (m)", "Plane sharing grid (m)"],
        [
            "この格子と視線の 8 方向ごとに面を 1 回だけ当て、その中の点で共有します。0 で点ごと",
            "A plane is fitted once per cell of this grid and each of 8 view directions, and shared by the points in it. 0 fits per point",
        ],
    ),
    (
        "cloud",
        "support_min_obs",
        [
            "孤立点：裏付けに要る観測数",
            "Isolated points: observations needed as support",
        ],
        [
            "測距の比の半径（下限あり）の中にほかの観測がこれより少ない点を出力しません。0 で使いません",
            "Points with fewer other observations than this within a range-proportional radius (with a lower bound) are not written. 0 disables it",
        ],
    ),
    (
        "cloud",
        "support_radius",
        [
            "孤立点：裏付けの半径の下限 (m)",
            "Isolated points: min support radius (m)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "support_range",
        [
            "孤立点：裏付けの半径（測距に対する比）",
            "Isolated points: support radius (ratio to the range)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "free_ratio",
        ["通り抜け／観測の比", "Pass-through / observation ratio"],
        [
            "ボクセルを素通りしたスキャンが、点を測ったスキャンのこの倍より多ければ、その点は空き（鏡像・動いた物・もや）として出力しません。0 で使いません",
            "If more scans passed through a voxel than this times the scans that measured a point there, the point is free space (mirror image, moved object, haze) and not written. 0 disables it",
        ],
    ),
    (
        "cloud",
        "free_min_pass",
        ["通り抜けの最小スキャン数", "Min pass-through scans"],
        ["", ""],
    ),
    (
        "cloud",
        "free_voxel",
        ["通り抜けのボクセル (m)", "Pass-through voxel (m)"],
        ["", ""],
    ),
    (
        "cloud",
        "free_margin",
        [
            "当たる手前で止める距離 (m)",
            "Stop distance before the hit (m)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "free_support",
        [
            "空きの観測を面の推定に使わない",
            "Don't use free-space observations for planes",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "free_pass_radius",
        [
            "観測の重心からこの距離を通るレーザーだけ数える (m)",
            "Count only lasers passing within this distance of the observations' centroid (m)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "free_pass_far",
        [
            "ただし先へ抜けるなら数える距離 (m)",
            "But count them up to this distance if they go beyond (m)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "free_protect_direction_spread",
        [
            "細い物の保護：向きのばらつき",
            "Thin object protection: direction spread",
        ],
        [
            "いろいろな向きから多くのスキャンで測られたボクセルは消しません（懸垂スタンドの横棒など）。0 で使いません",
            "Voxels measured by many scans from various directions are not removed (the bars of a hanging stand, etc.). 0 disables it",
        ],
    ),
    (
        "cloud",
        "free_protect_direction_scans",
        ["細い物の保護：スキャン数", "Thin object protection: scans"],
        ["", ""],
    ),
    (
        "cloud",
        "free_fine_voxel",
        [
            "細かい格子でも数える：ボクセル (m)",
            "Also count on a fine grid: voxel (m)",
        ],
        [
            "この大きさのボクセルでも保護なしで数え、どちらかで空きなら消します（細い物の周りに混ざった点）。0 で使いません",
            "Also counts in voxels of this size without the protection, and removes points that are free in either (points mixed around thin objects). 0 disables it",
        ],
    ),
    (
        "cloud",
        "free_fine_ratio",
        [
            "細かい格子：通り抜け／観測の比",
            "Fine grid: pass-through / observation ratio",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "transient_angle_deg",
        [
            "通過物：見通す角度 (deg)",
            "Passing objects: view angle (deg)",
        ],
        [
            "面にならない点を、スキャンごとに点の方向のこの角度の中のレーザーで調べ、少しの間だけ測られて多くのスキャンに見通された点（通った車や人）を出力しません。0 で使いません",
            "Points that don't form planes are checked per scan with the lasers within this angle of the point's direction; points measured only briefly and seen through by many scans (passing cars or people) are not written. 0 disables it",
        ],
    ),
    (
        "cloud",
        "transient_far",
        [
            "通過物：この先まで抜けたら通り抜け (m)",
            "Passing objects: passes through if beyond this (m)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "transient_hit",
        [
            "通過物：この距離以内で当たれば観測 (m)",
            "Passing objects: observed if hit within this distance (m)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "transient_max_hits",
        [
            "通過物：観測したスキャン数の上限",
            "Passing objects: max scans observing it",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "transient_clean_share",
        [
            "通過物：手前で当たらず抜けたスキャンの割合の下限",
            "Passing objects: min share of scans passing without a hit in front",
        ],
        [
            "浅い角度で見る地面は、点のすぐ下のレーザーがいつも手前に当たるので消えません",
            "Ground seen at a shallow angle isn't removed, since the lasers just below the point always hit in front",
        ],
    ),
    (
        "cloud",
        "transient_hit_span_s",
        [
            "通過物：観測したスキャンの時間の幅 (s)",
            "Passing objects: time span of the observing scans (s)",
        ],
        [
            "数分おきに少しずつ測られる草の先などは消えません",
            "Grass tips measured a little every few minutes and the like aren't removed",
        ],
    ),
    (
        "cloud",
        "transient_ratio",
        [
            "通過物：通り抜け／観測の比",
            "Passing objects: pass-through / observation ratio",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "transient_min_pass",
        [
            "通過物：通り抜けの最小スキャン数",
            "Passing objects: min pass-through scans",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "transient_max_range",
        [
            "通過物：数えるスキャンの距離 (m)",
            "Passing objects: range of the scans counted (m)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "reject_sigma",
        ["外れの判定：σ の倍率", "Outlier test: σ factor"],
        [
            "面からこの倍 × max(面の σ, 測距の見込み誤差) を超えた点は外れとして当て直し、合わなければ動かさずに残します。この距離までは誤差として面へ寄せます",
            "Points farther from the plane than this times max(plane σ, expected range error) are refitted as outliers, and left unmoved if nothing fits. Up to this distance they're moved onto the plane as error",
        ],
    ),
    (
        "cloud",
        "mls_taper_start",
        [
            "投影を弱め始める距離 (m)",
            "Distance where projection starts to weaken (m)",
        ],
        ["", ""],
    ),
    (
        "cloud",
        "mls_taper_end",
        [
            "投影を 0 にする距離 (m)",
            "Distance where projection reaches 0 (m)",
        ],
        [
            "遠くで測った点は外れの判定の距離まで広がります",
            "Points measured far away spread up to the outlier test's distance",
        ],
    ),
    (
        "cloud",
        "sample_leaf",
        ["代表点の格子 (m)", "Representative point grid (m)"],
        [
            "格子ごとに中心に最も近い実測点を 1 つ選びます。0 で全観測",
            "Per cell the measured point nearest the center is chosen. 0 keeps all observations",
        ],
    ),
    (
        "cloud",
        "post_leaf",
        ["仕上げの間引き (m)", "Final thinning (m)"],
        [
            "面へ寄せた後の位置で、この格子に 1 点だけ残します。0 で使いません",
            "Keeps one point per cell of this grid, at the positions after moving onto the planes. 0 disables it",
        ],
    ),
    (
        "cloud",
        "mls_radius",
        ["面の半径 (m)", "Plane radius (m)"],
        ["", ""],
    ),
    (
        "cloud",
        "mls_strength",
        ["投影の強さ", "Projection strength"],
        ["", ""],
    ),
    (
        "cloud",
        "plane_band",
        ["面の候補の幅 (m)", "Plane candidate band (m)"],
        ["", ""],
    ),
    (
        "cloud",
        "min_plane_fraction",
        ["必要な面支持率", "Required plane support"],
        ["", ""],
    ),
    (
        "cloud",
        "max_plane_ratio",
        ["平面性 λ0 / λ1 の上限", "Max planarity λ0 / λ1"],
        ["", ""],
    ),
    (
        "cloud",
        "max_surface_sigma",
        ["面の σ の上限 (m)", "Max plane σ (m)"],
        ["", ""],
    ),
    (
        "cloud",
        "tile_size",
        ["タイルの大きさ (m)", "Tile size (m)"],
        [
            "点群をこの大きさの正方形ごとに作り、メモリを抑えます（結果は同じ）。0 で一度に作ります",
            "The point cloud is built per square of this size to save memory (same result). 0 builds it at once",
        ],
    ),
    (
        "cloud",
        "tile_obs",
        ["タイルの観測数の上限", "Max observations per tile"],
        [
            "これより多いタイル（余白を含む）は 4 つに分けます。小さいほどメモリが少なく、時間がかかります（2000 万で約 8 GB）",
            "Tiles (with margins) with more than this are split in 4. Smaller uses less memory and takes longer (about 8 GB at 20 million)",
        ],
    ),
    // colour
    (
        "color",
        "views",
        ["1 点に混ぜる画像の数", "Images blended per point"],
        ["", ""],
    ),
    (
        "color",
        "keep",
        [
            "1 点に残す候補の画像の数",
            "Candidate images kept per point",
        ],
        [
            "この中から外れた画像を除いてから混ぜます",
            "Outlying images among these are removed before blending",
        ],
    ),
    (
        "color",
        "max_theta_deg",
        ["画角の上限 (deg)", "Max view angle (deg)"],
        [
            "光軸からこれより外れた画像は、ほかに写る画像がない点にだけ使います",
            "Images where the point is farther than this from the optical axis are only used for points no other image sees",
        ],
    ),
    (
        "color",
        "soft_theta_deg",
        [
            "画角の点数を下げ始める角度 (deg)",
            "Angle where the view angle score starts to drop (deg)",
        ],
        ["", ""],
    ),
    (
        "color",
        "theta_slope",
        ["画角の点数の下げ方", "View angle score slope"],
        ["", ""],
    ),
    (
        "color",
        "turn_weight",
        ["回転の速さの点数の下げ方", "Turn rate score drop"],
        [
            "機体が 30 deg/s で回っているときに下げる点数（ぶれ）",
            "Score dropped when the device turns at 30 deg/s (blur)",
        ],
    ),
    (
        "color",
        "outlier",
        ["外れの判定：色の差", "Outlier test: color difference"],
        [
            "近くで 2 番目に暗い画像との色の差（RGB の差の合計）がこの値のとき、点数を ln 2 × 下の値だけ下げます（映り込み・白飛び）。0 で使いません",
            "When the color difference (sum of RGB differences) from the second darkest nearby image is this value, the score drops by ln 2 × the value below (reflections, blown highlights). 0 disables it",
        ],
    ),
    (
        "color",
        "outlier_penalty",
        ["外れの判定：点数の下げ幅", "Outlier test: score drop"],
        ["", ""],
    ),
    (
        "color",
        "cell",
        [
            "外れの判定と露出補正のセル (m)",
            "Outlier test and exposure cell (m)",
        ],
        ["", ""],
    ),
    (
        "color",
        "temperature",
        ["混ぜ方の温度", "Blending temperature"],
        [
            "点数の差がこの値で重みが 1/e になります",
            "A score difference of this value makes the weight 1/e",
        ],
    ),
    (
        "color",
        "band_temperature",
        [
            "大まかな色を混ぜる温度",
            "Coarse color blending temperature",
        ],
        [
            "0 より大きいと、細部は上の画像から、セルの大まかな色は候補の画像すべてからこの温度で混ぜます（画像ごとの色味の段差を消します）。0 で使いません",
            "Above 0, details come from the images above and the cell's coarse color is blended from all candidate images at this temperature (removes color steps between images). 0 disables it",
        ],
    ),
    (
        "color",
        "dark_window",
        [
            "外れの基準を探す点数の幅",
            "Score window for the outlier reference",
        ],
        ["", ""],
    ),
    (
        "color",
        "outlier_quantile",
        [
            "外れの基準の明るさの位置",
            "Brightness position of the outlier reference",
        ],
        [
            "近い画像を暗い順に並べて、この割合の位置の画像を基準にします（0.5 で中央値：一部の画像の影も白飛びも外れになります）。0 で 2 番目に暗い画像",
            "Nearby images are sorted from dark to bright, and the image at this fraction is the reference (0.5 is the median: shadows in some images and blown highlights are both outliers). 0 uses the second darkest image",
        ],
    ),
    (
        "color",
        "fill",
        [
            "白飛びを周りの色で埋める半径 (m)",
            "Radius to fill blown highlights with surrounding colors (m)",
        ],
        [
            "1 枚にしか写らず周りより明るい点は、周りの色にします。0 で使いません",
            "Points seen in only one image and brighter than their surroundings take the surrounding color. 0 disables it",
        ],
    ),
    (
        "color",
        "gain",
        ["画像ごとの露出補正", "Per-image exposure correction"],
        ["", ""],
    ),
    (
        "color",
        "gain_window",
        ["露出補正の時間窓 (s)", "Exposure correction window (s)"],
        [
            "前後の画像の利得の平均で割って、隣の画像との段差と左右のカメラの差だけを直します（日向・木陰の明るさは写真のまま）。0 で全画像にそろえます",
            "Divides by the mean gain of the images before and after, correcting only steps between neighbouring images and the difference between the left and right cameras (sun and shade stay as photographed). 0 matches all images",
        ],
    ),
    (
        "color",
        "max_range",
        ["着色する最大距離 (m)", "Max colorization range (m)"],
        [
            "カメラからこれより遠い点はその画像で色を付けません",
            "Points farther from the camera than this are not colored by that image",
        ],
    ),
    (
        "color",
        "spacing",
        ["点の間隔 (m)", "Point spacing (m)"],
        [
            "隠れの判定で周りのセルをどこまで見るかに使います",
            "Used for how far the occlusion test looks at surrounding cells",
        ],
    ),
    (
        "color",
        "tile_memory",
        ["使うメモリの割合", "Share of memory used"],
        [
            "着色を始めるときの空きメモリのうち、この割合に収まるよう点群を正方形に分けます（結果は同じ）。小さいほどメモリが少なく、時間がかかります。0 で分けずに一度に行います",
            "The point cloud is split into squares to fit this share of the free memory when colorization starts (same result). Smaller uses less memory and takes longer. 0 does it at once without splitting",
        ],
    ),
    (
        "color",
        "graze",
        [
            "浅く見る面の隠れを許す",
            "Allow occlusion on grazing surfaces",
        ],
        [
            "遠くの地面のように浅い角度で見る面は、z バッファの隣のセルの距離の傾きの分だけ隠れの判定を許します",
            "Surfaces seen at a shallow angle, like distant ground, get occlusion tolerance by the slope of the distances of neighbouring z-buffer cells",
        ],
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every parameter key has a label (a new field shows up as its raw key until it gets one).
    #[test]
    fn every_parameter_has_a_label() {
        fn walk(section: &str, v: &Value, missing: &mut Vec<String>) {
            match v {
                Value::Object(m) => {
                    for (k, c) in m {
                        if info(section, k).0 == k {
                            missing.push(format!("{section}.{k}"));
                        }
                        walk(section, c, missing);
                    }
                }
                Value::Array(a) => a.iter().for_each(|c| walk(section, c, missing)),
                _ => {}
            }
        }
        let v = serde_json::to_value(ProcessParams::default()).unwrap();
        let mut missing = Vec::new();
        for (s, c) in v.as_object().unwrap() {
            walk(s, c, &mut missing);
        }
        assert!(missing.is_empty(), "no label: {missing:?}");
    }

    #[test]
    fn rounding_keeps_six_digits() {
        assert_eq!(round_sig(0.15000000002), 0.15);
        assert_eq!(round_sig(1e-6), 1e-6);
        assert_eq!(round_sig(0.3f32 as f64), 0.3);
        assert_eq!(round_sig(100000.0), 100000.0);
    }
}
