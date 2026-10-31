//! RTK fixes in the chosen local frame (ellipsoidal height = hgt + undulation),
//! with the plausibility check of the Python version (`gnss.rtk_valid`).

use crate::geodesy::{CoordinateFrame, valid_llh};

/// Antenna offset relative to the LiDAR (calibration RTK_offset).
pub const RTK_OFFSET: [f64; 3] = [-0.0621, 0.0, -0.0108];
pub const FIX: f64 = 50.0;

#[derive(Clone, Copy, Debug)]
pub struct Fix {
    pub t: f64,
    /// Local east/north/up position (metres).
    pub p: [f64; 3],
    /// PPK: east/north/up standard deviations (m). Receiver RTK retains its
    /// lat/lon/height deviations; legacy RTK weighting uses configured sigmas.
    pub sd: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct Validity {
    pub max_dh: f64,
    pub max_dv: f64,
    pub sv_frac: f64,
}

impl Default for Validity {
    fn default() -> Self {
        Validity {
            max_dh: 3.0,
            max_dv: 5.0,
            sv_frac: 0.6,
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// FIX epochs that are plausible: close to the single-point solution and with a normal satellite count.
/// Catches false fixes (e.g. after losing most satellites the "fixed" height can be off by 10 m).
pub fn rtk_valid(g: &[[f64; 14]], v: &Validity) -> Vec<bool> {
    let fix: Vec<bool> = g.iter().map(|r| r[1] == FIX).collect();
    let svs: Vec<f64> = g
        .iter()
        .zip(&fix)
        .filter(|(_, f)| **f)
        .map(|(r, _)| r[12])
        .collect();
    let svmed = if svs.is_empty() {
        median(g.iter().map(|r| r[12]).collect())
    } else {
        median(svs)
    };
    g.iter()
        .zip(&fix)
        .map(|(r, &f)| {
            if !f {
                return false;
            }
            let best = [r[2], r[3], r[4] + r[8]];
            let psr = [r[9], r[10], r[11] + r[8]];
            if !valid_llh(best) || !valid_llh(psr) {
                return false;
            }
            let frame = CoordinateFrame::local_enu(best).unwrap();
            let delta = frame.position(psr);
            let dh = delta[0].hypot(delta[1]);
            let dv = (r[4] - r[11]).abs();
            dh < v.max_dh && dv < v.max_dv && r[12] >= v.sv_frac * svmed
        })
        .collect()
}

/// Valid fixes in `frame`.
pub fn load_fixes(g: &[[f64; 14]], frame: &CoordinateFrame, v: &Validity) -> Vec<Fix> {
    let ok = rtk_valid(g, v);
    g.iter()
        .zip(ok)
        .filter(|(_, o)| *o)
        .map(|(r, _)| Fix {
            t: r[0],
            p: frame.position([r[2], r[3], r[4] + r[8]]),
            sd: [r[5], r[6], r[7]],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receiver_height_and_geoid_recover_ellipsoidal_height_outside_japan() {
        let mut r = [0.; 14];
        r[1] = FIX;
        r[2] = -33.9;
        r[3] = 151.2;
        r[4] = 10.;
        r[8] = 25.;
        r[9] = r[2];
        r[10] = r[3];
        r[11] = r[4];
        r[12] = 20.;
        let frame = CoordinateFrame::local_enu([r[2], r[3], 35.]).unwrap();
        let fixes = load_fixes(&[r], &frame, &Validity::default());
        assert_eq!(fixes.len(), 1);
        assert_eq!(fixes[0].p, [0.; 3]);
        let mut bad = r;
        bad[9] += 0.01;
        assert!(!rtk_valid(&[bad], &Validity::default())[0]);
        bad[9] = f64::NAN;
        assert!(!rtk_valid(&[bad], &Validity::default())[0]);
    }
}
