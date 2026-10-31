//! Trajectory colouring by the per-knot quality of the optimisation (`opt_quality.txt`).

use crate::lang::tl;
use anyhow::Result;
use std::path::Path;

/// Per-knot quality rows: t and the metrics in the order of [`METRICS`] (NaN where not defined).
pub struct Quality {
    pub t: Vec<f64>,
    pub v: Vec<[f64; 7]>,
}

impl Quality {
    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let mut q = Quality {
            t: Vec::new(),
            v: Vec::new(),
        };
        for l in s
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let x: Vec<f64> = l
                .split_whitespace()
                .map(|w| w.parse().unwrap_or(f64::NAN))
                .collect();
            if x.len() >= 8 {
                q.t.push(x[0]);
                q.v.push(std::array::from_fn(|i| x[i + 1]));
            }
        }
        Ok(q)
    }

    /// Value of metric `m` for the knot interval holding time `t`.
    pub fn at(&self, m: usize, t: f64) -> f64 {
        let i = self.t.partition_point(|&s| s <= t);
        if i == 0 {
            return f64::NAN;
        }
        self.v[i - 1][m]
    }
}

pub struct Metric {
    /// texts in Japanese and English
    label: [&'static str; 2],
    unit: [&'static str; 2],
    help: [&'static str; 2],
    /// value shown green and value shown red (red may be below green: low is bad)
    pub good: f64,
    pub bad: f64,
}

impl Metric {
    pub fn label(&self) -> &'static str {
        tl(self.label[0], self.label[1])
    }

    pub fn unit(&self) -> &'static str {
        tl(self.unit[0], self.unit[1])
    }

    pub fn help(&self) -> &'static str {
        tl(self.help[0], self.help[1])
    }
}

pub const METRICS: [Metric; 7] = [
    Metric {
        label: ["面のずれ（RMS）", "Plane offset (RMS)"],
        unit: ["mm", "mm"],
        help: [
            "その区間の点の、面からの距離の RMS。大きいと点群が二重化・ぼやけている可能性",
            "RMS of the section's points' distances from their planes. Large values may mean a doubled or blurred point cloud",
        ],
        good: 8.0,
        bad: 25.0,
    },
    Metric {
        label: ["面に乗った点の割合", "Share of points on planes"],
        unit: ["", ""],
        help: [
            "その区間の点のうち面に乗った割合。低いと手がかりが少ない（狭い・雑然・植生）",
            "Share of the section's points on planes. Low means few cues (narrow, cluttered, vegetation)",
        ],
        good: 0.8,
        bad: 0.4,
    },
    Metric {
        label: ["IMU との不整合", "Disagreement with the IMU"],
        unit: ["χ²/自由度", "χ²/DoF"],
        help: [
            "IMU の事前積分の残差。大きいと LiDAR と IMU の動きが合っていない",
            "Residual of the IMU preintegration. Large means the LiDAR and IMU motions don't agree",
        ],
        good: 1.0,
        bad: 8.0,
    },
    Metric {
        label: ["GNSS 残差（水平）", "GNSS residual (horizontal)"],
        unit: ["mm", "mm"],
        help: [
            "選択した RTK / PPK 解と軌跡（アンテナ位置）の水平の差",
            "Horizontal difference between the chosen RTK / PPK solution and the trajectory (antenna position)",
        ],
        good: 10.0,
        bad: 50.0,
    },
    Metric {
        label: ["GNSS 残差（鉛直）", "GNSS residual (vertical)"],
        unit: ["mm", "mm"],
        help: [
            "選択した RTK / PPK 解と軌跡（アンテナ位置）の鉛直の差",
            "Vertical difference between the chosen RTK / PPK solution and the trajectory (antenna position)",
        ],
        good: 20.0,
        bad: 80.0,
    },
    Metric {
        label: ["補正量（回転）", "Correction (rotation)"],
        unit: ["°/秒", "°/s"],
        help: [
            "1 秒間の回転が、オドメトリからどれだけ直されたか。大きいとオドメトリが跳んでいた",
            "How much the rotation over 1 s was corrected from the odometry. Large means the odometry jumped",
        ],
        good: 0.2,
        bad: 2.0,
    },
    Metric {
        label: ["補正量（移動）", "Correction (translation)"],
        unit: ["cm/秒", "cm/s"],
        help: [
            "1 秒間の移動が、オドメトリからどれだけ直されたか",
            "How much the translation over 1 s was corrected from the odometry",
        ],
        good: 1.0,
        bad: 10.0,
    },
];

/// 0 at `good`, 1 at `bad` (clamped); NaN stays NaN.
pub fn badness(m: &Metric, bad: f64, v: f64) -> f64 {
    ((v - m.good) / (bad - m.good)).clamp(0.0, 1.0)
}

/// Green -> yellow -> red; grey for NaN.
pub fn color(u: f64) -> [u8; 3] {
    if !u.is_finite() {
        return [120, 120, 120];
    }
    let u = u.clamp(0.0, 1.0) as f32;
    let (g, y, r) = (
        [60.0, 200.0, 90.0],
        [235.0, 210.0, 60.0],
        [230.0, 60.0, 50.0],
    );
    let mix =
        |a: [f32; 3], b: [f32; 3], s: f32| [0, 1, 2].map(|i| (a[i] + (b[i] - a[i]) * s) as u8);
    if u < 0.5 {
        mix(g, y, u * 2.0)
    } else {
        mix(y, r, (u - 0.5) * 2.0)
    }
}

/// The worst places for metric `m`: values averaged over one second, local peaks at least `gap`
/// seconds apart, beyond the green value, worst first. Returns (time, value).
pub fn worst(q: &Quality, m: usize, bad: f64, gap: f64, n: usize) -> Vec<(f64, f64)> {
    let met = &METRICS[m];
    let k = q.t.len();
    let half = 5;
    let smooth: Vec<f64> = (0..k)
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half + 1).min(k));
            let v: Vec<f64> = q.v[a..b]
                .iter()
                .map(|r| r[m])
                .filter(|x| x.is_finite())
                .collect();
            if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        })
        .collect();
    let mut idx: Vec<usize> = (0..k)
        .filter(|&i| smooth[i].is_finite() && badness(met, bad, smooth[i]) > 0.0)
        .collect();
    idx.sort_by(|&a, &b| {
        badness(met, bad, smooth[b])
            .total_cmp(&badness(met, bad, smooth[a]))
            .then(
                (smooth[b] - met.good)
                    .abs()
                    .total_cmp(&(smooth[a] - met.good).abs()),
            )
    });
    let mut out: Vec<(f64, f64)> = Vec::new();
    for i in idx {
        if out.iter().all(|&(t, _)| (t - q.t[i]).abs() > gap) {
            out.push((q.t[i], smooth[i]));
            if out.len() == n {
                break;
            }
        }
    }
    out
}
