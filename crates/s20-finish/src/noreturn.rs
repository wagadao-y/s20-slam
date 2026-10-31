//! The directions of the shots that came back from nothing (the sky, water, beyond the range),
//! for the passing-object test ([`crate::transient`]). Ported from the author's earlier
//! prototype (slam-experiment).
//!
//! The MID-360 records every shot; one without a return has coordinates 0 and no direction. Each
//! of the 4 lasers fires every 19.8 us, about 1.25 deg on from its last shot, the elevation
//! changing by a few hundredths of a degree, so within a gap of up to [`MAX_GAP`] shots the
//! direction is a cubic Hermite curve in azimuth and elevation between the returned shots on
//! either side, their per-shot rates as slopes (hidden returned shots: 0.27 deg q90 at 80 shots).
//! Longer gaps are left out. The sequence does not repeat from scan to scan, so a long gap cannot
//! be filled from another scan.
//!
//! Each shot keeps its reach, the farther range of the returned shots bounding its gap: it only
//! shows that nothing was there up to that far. Indoors 11-16% of the shots come back from
//! nothing in every direction (glass, dark patches of a ceiling, things too close, the operator),
//! with the ceiling on either side; counted as rays without an end they took the upper parts of
//! rooms.

/// Gaps of a laser longer than this many shots are left out: 100 deg of azimuth.
pub const MAX_GAP: usize = 80;

/// The shots of one scan without a return in gaps of up to [`MAX_GAP`] shots of their laser:
/// (index in the scan, unit direction in the LiDAR frame, the farther range of the returned shots
/// bounding the gap).
pub fn directions(xyz: &[[f64; 3]], line: &[u8]) -> Vec<(usize, [f64; 3], f64)> {
    assert_eq!(xyz.len(), line.len());
    let mut out = Vec::new();
    for l in 0..4u8 {
        let idx: Vec<usize> = (0..xyz.len()).filter(|&i| line[i] == l).collect();
        // The returned shots of this laser: position in `idx`, azimuth (unwrapped), elevation.
        let mut ret: Vec<(usize, f64, f64)> = Vec::new();
        let mut range: Vec<f64> = Vec::new();
        for (k, &i) in idx.iter().enumerate() {
            let p = xyz[i];
            let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if n == 0. || !n.is_finite() {
                continue;
            }
            let mut az = p[1].atan2(p[0]);
            if let Some(&(_, last, _)) = ret.last() {
                az += ((last - az) / std::f64::consts::TAU).round() * std::f64::consts::TAU;
            }
            ret.push((k, az, (p[2] / n).clamp(-1., 1.).asin()));
            range.push(n);
        }
        // A gap between returned shots g and g+1, with a returned shot before g and after g+1 for
        // the slopes.
        for g in 1..ret.len().saturating_sub(2) {
            let (a, b) = (ret[g], ret[g + 1]);
            let gap = b.0 - a.0 - 1;
            if gap == 0 || gap > MAX_GAP {
                continue;
            }
            let (p, n) = (ret[g - 1], ret[g + 2]);
            let rate = |x: (usize, f64, f64), y: (usize, f64, f64)| {
                let d = (y.0 - x.0) as f64;
                ((y.1 - x.1) / d, (y.2 - x.2) / d)
            };
            let (ma, mb) = (rate(p, a), rate(b, n));
            let h = (b.0 - a.0) as f64;
            let reach = range[g].max(range[g + 1]);
            for j in 1..=gap {
                let s = j as f64 / h;
                let (h00, h10, h01, h11) = (
                    2. * s.powi(3) - 3. * s * s + 1.,
                    s.powi(3) - 2. * s * s + s,
                    -2. * s.powi(3) + 3. * s * s,
                    s.powi(3) - s * s,
                );
                let az = h00 * a.1 + h10 * h * ma.0 + h01 * b.1 + h11 * h * mb.0;
                let el = h00 * a.2 + h10 * h * ma.1 + h01 * b.2 + h11 * h * mb.1;
                out.push((
                    idx[a.0 + j],
                    [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()],
                    reach,
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shot(az: f64, el: f64, r: f64) -> [f64; 3] {
        [
            r * el.cos() * az.cos(),
            r * el.cos() * az.sin(),
            r * el.sin(),
        ]
    }

    #[test]
    fn a_gap_is_filled_along_the_laser_and_a_long_one_left_out() {
        // One laser sweeping 1.25 deg a shot across the -180/180 seam, elevation drifting.
        let n = 400;
        let truth: Vec<[f64; 3]> = (0..n)
            .map(|k| {
                shot(
                    (150. + 1.25 * k as f64).to_radians(),
                    (10. + 0.03 * k as f64).to_radians(),
                    1.,
                )
            })
            .collect();
        let mut xyz: Vec<[f64; 3]> = truth.iter().map(|u| u.map(|x| x * 20.)).collect();
        for x in &mut xyz[30..90] {
            *x = [0.; 3]; // 60 shots: filled
        }
        for x in &mut xyz[200..300] {
            *x = [0.; 3]; // 100 shots: left out
        }
        let line = vec![0u8; n];
        let d = directions(&xyz, &line);
        assert_eq!(d.len(), 60);
        for (i, u, reach) in d {
            assert!((30..90).contains(&i));
            assert!((reach - 20.).abs() < 1e-9);
            let c: f64 = (0..3).map(|k| u[k] * truth[i][k]).sum();
            assert!(c.clamp(-1., 1.).acos().to_degrees() < 0.05, "shot {i}");
        }
    }
}
