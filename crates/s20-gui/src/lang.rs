//! The interface's language: Japanese or English, chosen from the menu bar. Each text is written
//! with its English beside it: `tl("ja", "en")`, `tr!("ja {x}", "en {x}")` in place of `format!`.
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lang {
    Ja,
    En,
}

impl Lang {
    pub const ALL: [Self; 2] = [Self::En, Self::Ja];

    /// The language's own name, as in the menu.
    pub fn name(self) -> &'static str {
        match self {
            Self::En => "English",
            Self::Ja => "日本語",
        }
    }

    /// Japanese when Windows shows Japanese, else English (until one is chosen in the menu).
    pub fn of_system() -> Self {
        if system_japanese() {
            Self::Ja
        } else {
            Self::En
        }
    }
}

/// Whether the user's display language is Japanese.
#[cfg(windows)]
fn system_japanese() -> bool {
    const LANG_JAPANESE: u16 = 0x11;
    unsafe extern "system" {
        fn GetUserDefaultUILanguage() -> u16;
    }
    // SAFETY: plain Win32 call without arguments.
    let id = unsafe { GetUserDefaultUILanguage() };
    // the primary language is the low 10 bits of the language id
    id & 0x3ff == LANG_JAPANESE
}

#[cfg(not(windows))]
fn system_japanese() -> bool {
    ["LC_ALL", "LC_MESSAGES", "LANG"]
        .iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
        .is_some_and(|v| v.starts_with("ja"))
}

/// Read from the worker threads too (the messages they make), so not kept in the app state.
static ENGLISH: AtomicBool = AtomicBool::new(false);

pub fn set(lang: Lang) {
    ENGLISH.store(lang == Lang::En, Ordering::Relaxed);
}

pub fn english() -> bool {
    ENGLISH.load(Ordering::Relaxed)
}

/// The language shown.
pub fn current() -> Lang {
    if english() { Lang::En } else { Lang::Ja }
}

/// The text in the chosen language.
pub fn tl<T>(ja: T, en: T) -> T {
    if english() { en } else { ja }
}

/// `format!` with the Japanese and the English format strings.
macro_rules! tr {
    ($ja:literal, $en:literal $(, $arg:expr)* $(,)?) => {
        if $crate::lang::english() {
            format!($en $(, $arg)*)
        } else {
            format!($ja $(, $arg)*)
        }
    };
}

/// The English of the texts the other crates give in Japanese, when English is chosen (else, and
/// for texts not listed, the text itself).
pub fn of(ja: &'static str) -> &'static str {
    if !english() {
        return ja;
    }
    match ja {
        // s20_pipeline::ppk::inspect::Fix
        "片方向 FIX" => "One-sided FIX",
        "FIX（不採用）" => "FIX (rejected)",
        "単独" => "Single",
        "整数の不確定性が解けた解（PPK では前方解と後方解の両方で FIX）。軌跡の最適化で使う" => {
            "Solutions with the integer ambiguities resolved (with PPK, FIX in both the forward and \
             backward solutions). Used in the trajectory optimization"
        }
        "前方解か後方解の片方だけで FIX。もう片方の解と近いので、軌跡の最適化で使う" => {
            "FIX in only one of the forward and backward solutions. Close to the other solution, so \
             used in the trajectory optimization"
        }
        "FIX だが軌跡の最適化で使わない解。ほとんどは片方向だけの FIX で、もう片方の解と離れていたもの\
         （どちらが正しいか決められないため）。標準偏差が上限を超えたものも含む" => {
            "FIX solutions not used in the trajectory optimization. Mostly one-sided FIXes far from \
             the other solution (which one is right can't be decided). Includes those whose standard \
             deviation exceeds the limit"
        }
        "整数の不確定性が解けていない解。軌跡の最適化では使わない（設定による）" => {
            "Solutions with unresolved integer ambiguities. Not used in the trajectory optimization \
             (depending on the settings)"
        }
        "コード差分の解" => "Code differential solutions",
        "単独測位の解" => "Single point solutions",
        // inspect::SYSTEMS
        "その他" => "Other",
        // inspect::message_name
        "OBSVM 観測" => "OBSVM observations",
        "GPSEPH 軌道" => "GPSEPH ephemeris",
        "GLOEPH 軌道" => "GLOEPH ephemeris",
        "BDSEPH 軌道" => "BDSEPH ephemeris",
        "GALEPH 軌道" => "GALEPH ephemeris",
        "QZSSEPH 軌道" => "QZSSEPH ephemeris",
        "IRNSSEPH 軌道" => "IRNSSEPH ephemeris",
        "BD3EPH 軌道" => "BD3EPH ephemeris",
        "基準局の位置" => "Base station position",
        "基準局の位置（アンテナ高つき）" => {
            "Base station position (with antenna height)"
        }
        "アンテナ・受信機の情報" => "Antenna and receiver information",
        "GPS 軌道" => "GPS ephemeris",
        "GLONASS 軌道" => "GLONASS ephemeris",
        "BeiDou 軌道" => "BeiDou ephemeris",
        "QZSS 軌道" => "QZSS ephemeris",
        "Galileo 軌道" => "Galileo ephemeris",
        "GPS 観測（MSM）" => "GPS observations (MSM)",
        "GLONASS 観測（MSM）" => "GLONASS observations (MSM)",
        "Galileo 観測（MSM）" => "Galileo observations (MSM)",
        "SBAS 観測（MSM）" => "SBAS observations (MSM)",
        "QZSS 観測（MSM）" => "QZSS observations (MSM)",
        "BeiDou 観測（MSM）" => "BeiDou observations (MSM)",
        "GLONASS コードと位相のずれ" => "GLONASS code-phase biases",
        // s20_pipeline::device::unusable
        "セッションがありません" => "No sessions",
        "生データにカメラの校正（info/calibration.yaml）がありません" => {
            "The raw data has no camera calibration (info/calibration.yaml)"
        }
        "点群生成まで済んでいません" => "Not processed through the point cloud",
        other => other,
    }
}
