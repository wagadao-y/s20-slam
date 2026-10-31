//! Several recordings of one device refined together: every image's pose, and optionally the two
//! cameras' changes shared by all of them (`refine::Cameras`), with checks on the held-out
//! observations, each recording also checked with the cameras solved from the others.

use crate::anchors::Pose;
use crate::pairs::Obs;
use crate::refine::{Cameras, Correction, Prepared, Settings, apply, refine_camera, residual_with};
use crate::run::{Collected, Kind, median, side};
use serde_json::{Value, json};

/// All recordings in one index space: recording `r` holds images `offset[r]..offset[r + 1]`.
pub struct Joint {
    pub names: Vec<String>,
    rec: Vec<Collected>,
    offset: Vec<usize>,
    poses: Vec<Pose>,
    obs: Vec<Obs>,
    owner: Vec<usize>,
    prepared: Vec<Prepared>,
    held: Vec<bool>,
    kinds: Vec<Kind>,
}

impl Joint {
    pub fn new(recordings: Vec<(String, Collected)>) -> Self {
        let mut j = Joint {
            names: Vec::new(),
            rec: Vec::new(),
            offset: vec![0],
            poses: Vec::new(),
            obs: Vec::new(),
            owner: Vec::new(),
            prepared: Vec::new(),
            held: Vec::new(),
            kinds: Vec::new(),
        };
        for (r, (name, mut c)) in recordings.into_iter().enumerate() {
            let base = j.poses.len();
            j.poses.extend(c.poses.poses.iter().copied());
            j.offset.push(j.poses.len());
            for o in &c.obs {
                let mut o = o.clone();
                o.a += base;
                o.b += base;
                j.obs.push(o);
                j.owner.push(r);
            }
            j.prepared.append(&mut c.prepared);
            j.held.extend_from_slice(&c.held);
            j.kinds.extend_from_slice(&c.kinds);
            j.names.push(name);
            j.rec.push(c);
        }
        j
    }

    pub fn len(&self) -> usize {
        self.rec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rec.is_empty()
    }

    /// Refine the recordings of `subset` on their training observations; the cameras are solved
    /// as asked, else kept at `camera`.
    pub fn solve(
        &self,
        subset: &[usize],
        fit_camera: bool,
        fit_mount: bool,
        camera: &mut Cameras,
    ) -> Vec<Correction> {
        let use_obs: Vec<bool> = self
            .held
            .iter()
            .zip(&self.owner)
            .map(|(h, r)| !h && subset.contains(r))
            .collect();
        let chain: Vec<(usize, usize)> = subset
            .iter()
            .flat_map(|&r| {
                self.rec[r]
                    .chain
                    .iter()
                    .map(move |&(i, j)| (i + self.offset[r], j + self.offset[r]))
            })
            .collect();
        let fit = fit_camera || fit_mount;
        let s = Settings {
            fit_camera,
            fit_mount,
            iterations: if fit { 25 } else { 12 },
            ..Default::default()
        };
        refine_camera(
            &self.poses,
            &self.obs,
            &self.prepared,
            &use_obs,
            &chain,
            &s,
            camera,
        )
    }

    /// Median held-out residuals of recording `r` by kind of pair (pixel-equivalents).
    pub fn check(&self, r: usize, c: &[Correction], camera: &Cameras) -> Value {
        let mut out = serde_json::Map::new();
        for (kind, label) in [
            (Kind::Temporal, "temporal"),
            (Kind::Simultaneous, "simultaneous"),
            (Kind::Cross, "cross"),
        ] {
            let v: Vec<f64> = (0..self.obs.len())
                .filter(|&k| self.owner[k] == r && self.held[k] && self.kinds[k] == kind)
                .map(|k| {
                    let o = &self.obs[k];
                    let e = residual_with(
                        o,
                        &self.prepared[k],
                        &apply(&self.poses[o.a], &c[o.a]),
                        &apply(&self.poses[o.b], &c[o.b]),
                        camera,
                    );
                    e[0].hypot(e[1])
                })
                .collect();
            out.insert(label.into(), json!(median(v)));
        }
        Value::Object(out)
    }

    /// Median per-image rotation (degrees) and the mean in each camera's own frame: what is left
    /// to the images, and what a camera change could still take.
    pub fn image_rotations(&self, subset: &[usize], c: &[Correction]) -> Value {
        let mut sum = [[0f64; 3]; 2];
        let mut count = [0usize; 2];
        let mut size = Vec::new();
        for &r in subset {
            for &i in &self.rec[r].used {
                let g = i + self.offset[r];
                let s = side(&self.rec[r].poses.names[i]);
                let w = c[g].fixed_rows::<3>(0).into_owned();
                let v = self.poses[g].r.transpose() * w;
                for k in 0..3 {
                    sum[s][k] += v[k];
                }
                count[s] += 1;
                size.push(w.norm().to_degrees());
            }
        }
        json!({
            "median_deg": median(size),
            "mean_in_camera_deg": (0..2).map(|s| sum[s].map(|x| (x / count[s].max(1) as f64).to_degrees())).collect::<Vec<_>>(),
        })
    }
}

/// Mount rotations in degrees and offsets in millimetres, with the raw values.
pub fn camera_json(c: &Cameras) -> Value {
    let mount: Vec<Value> = c
        .mount
        .iter()
        .map(|m| {
            json!({
                "rotation_deg": [m[0].to_degrees(), m[1].to_degrees(), m[2].to_degrees()],
                "rotation_norm_deg": (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt().to_degrees(),
                "offset_mm": [m[3] * 1e3, m[4] * 1e3, m[5] * 1e3],
            })
        })
        .collect();
    json!({"intrinsic": c.intrinsic, "mount": mount, "raw_mount": c.mount})
}

/// The cameras solved from all recordings, and a report: each recording with the present cameras,
/// with the solved ones, and (`folds`) with those solved from the other recordings and from itself
/// alone.
pub fn compare(
    j: &Joint,
    fit_camera: bool,
    fit_mount: bool,
    folds: bool,
    log: &dyn Fn(&str),
) -> (Cameras, Value) {
    let all: Vec<usize> = (0..j.len()).collect();
    let zero = Cameras::default();
    let mut report = serde_json::Map::new();
    report.insert("recordings".into(), json!(j.names));
    report.insert(
        "solved".into(),
        json!({"intrinsic": fit_camera, "mount": fit_mount}),
    );
    // the recordings are independent with the cameras kept
    let base = j.solve(&all, false, false, &mut zero.clone());
    report.insert("present".into(), json!({
        "images": j.image_rotations(&all, &base), "checks": (0..j.len()).map(|r| j.check(r, &base, &zero)).collect::<Vec<_>>(),
    }));
    log("refined with the present cameras");
    let mut camera = zero;
    let c = j.solve(&all, fit_camera, fit_mount, &mut camera);
    report.insert(
        "joint".into(),
        json!({
            "camera": camera_json(&camera), "images": j.image_rotations(&all, &c),
            "checks": (0..j.len()).map(|r| j.check(r, &c, &camera)).collect::<Vec<_>>(),
        }),
    );
    log(&format!("cameras solved from all {} recordings", j.len()));
    if folds && j.len() > 1 {
        let mut out = Vec::new();
        for r in 0..j.len() {
            let others: Vec<usize> = (0..j.len()).filter(|&q| q != r).collect();
            let mut from_others = zero;
            j.solve(&others, fit_camera, fit_mount, &mut from_others);
            let c = j.solve(&[r], false, false, &mut from_others.clone());
            let mut own = zero;
            let c_own = j.solve(&[r], fit_camera, fit_mount, &mut own);
            out.push(json!({
                "held_out": j.names[r], "camera_from_others": camera_json(&from_others), "check": j.check(r, &c, &from_others),
                "camera_alone": camera_json(&own), "check_alone": j.check(r, &c_own, &own),
            }));
            log(&format!(
                "checked {} with the cameras of the others",
                j.names[r]
            ));
        }
        report.insert("folds".into(), json!(out));
    }
    (camera, Value::Object(report))
}
