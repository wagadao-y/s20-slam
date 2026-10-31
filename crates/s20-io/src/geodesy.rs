//! Local GNSS coordinates. Transverse Mercator is retained for reading legacy results only.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

/// Coordinates of the trajectory and its exported points (metres).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CoordinateFrame {
    /// WGS84 geodetic latitude/longitude (degrees), ellipsoidal height (metres).
    LocalEnu { origin_llh: [f64; 3] },
    /// GNSS was not used; the first session's SLAM frame defines the coordinates.
    LocalSlam,
    /// Old internal coordinates were EPSG:6678 minus this offset.
    LegacyProjected { origin: [f64; 3] },
}

impl CoordinateFrame {
    pub fn local_enu(origin_llh: [f64; 3]) -> Result<Self> {
        ensure!(valid_llh(origin_llh), "invalid GNSS origin: {origin_llh:?}");
        Ok(Self::LocalEnu { origin_llh })
    }

    /// LLH to local east/north/up through WGS84 ECEF, without a geoid model.
    pub fn position(&self, llh: [f64; 3]) -> [f64; 3] {
        match *self {
            Self::LocalEnu { origin_llh } => {
                let p = ecef(llh);
                let o = ecef(origin_llh);
                let d: [f64; 3] = std::array::from_fn(|i| p[i] - o[i]);
                let (s, c) = origin_llh[0].to_radians().sin_cos();
                let (sl, cl) = origin_llh[1].to_radians().sin_cos();
                [
                    -sl * d[0] + cl * d[1],
                    -s * cl * d[0] - s * sl * d[1] + c * d[2],
                    c * cl * d[0] + c * sl * d[1] + s * d[2],
                ]
            }
            Self::LegacyProjected { origin } => {
                let (e, n) = TransverseMercator::epsg6678().forward(llh[0], llh[1]);
                [e - origin[0], n - origin[1], llh[2] - origin[2]]
            }
            Self::LocalSlam => panic!("a SLAM frame has no GNSS transform"),
        }
    }

    pub fn output_offset(&self) -> [f64; 3] {
        match *self {
            Self::LegacyProjected { origin } => origin,
            _ => [0.; 3],
        }
    }

    pub fn read(path: &Path) -> Result<Self> {
        let frame: Self = serde_json::from_slice(&fs::read(path)?)?;
        if let Self::LocalEnu { origin_llh } = frame {
            ensure!(
                valid_llh(origin_llh),
                "invalid GNSS origin in {}",
                path.display()
            );
        }
        Ok(frame)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

pub fn valid_llh(p: [f64; 3]) -> bool {
    p.iter().all(|x| x.is_finite()) && p[0].abs() <= 90. && p[1].abs() <= 180.
}

fn ecef(llh: [f64; 3]) -> [f64; 3] {
    let f = 1. / 298.257223563;
    let e2 = f * (2. - f);
    let (s, c) = llh[0].to_radians().sin_cos();
    let (sl, cl) = llh[1].to_radians().sin_cos();
    let n = 6378137. / (1. - e2 * s * s).sqrt();
    [
        (n + llh[2]) * c * cl,
        (n + llh[2]) * c * sl,
        (n * (1. - e2) + llh[2]) * s,
    ]
}

#[derive(Clone, Copy, Debug)]
pub struct TransverseMercator {
    lat0: f64,
    lon0: f64,
    k0: f64,
    a_hat: f64,
    alpha: [f64; 6],
    e: f64,
    xi0: f64,
}

const GRS80_A: f64 = 6378137.0;
const GRS80_F: f64 = 1.0 / 298.257222101;

impl TransverseMercator {
    pub fn new(lat0_deg: f64, lon0_deg: f64, k0: f64) -> Self {
        let n = GRS80_F / (2.0 - GRS80_F);
        let (n2, n3) = (n * n, n * n * n);
        let (n4, n5, n6) = (n3 * n, n3 * n2, n3 * n3);
        let a_hat = GRS80_A / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0 + n6 / 256.0);
        let alpha = [
            n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0 + 41.0 * n4 / 180.0 - 127.0 * n5 / 288.0
                + 7891.0 * n6 / 37800.0,
            13.0 * n2 / 48.0 - 3.0 * n3 / 5.0 + 557.0 * n4 / 1440.0 + 281.0 * n5 / 630.0
                - 1983433.0 * n6 / 1935360.0,
            61.0 * n3 / 240.0 - 103.0 * n4 / 140.0
                + 15061.0 * n5 / 26880.0
                + 167603.0 * n6 / 181440.0,
            49561.0 * n4 / 161280.0 - 179.0 * n5 / 168.0 + 6601661.0 * n6 / 7257600.0,
            34729.0 * n5 / 80640.0 - 3418889.0 * n6 / 1995840.0,
            212378941.0 * n6 / 319334400.0,
        ];
        let e = 2.0 * n.sqrt() / (1.0 + n);
        let mut tm = TransverseMercator {
            lat0: lat0_deg.to_radians(),
            lon0: lon0_deg.to_radians(),
            k0,
            a_hat,
            alpha,
            e,
            xi0: 0.0,
        };
        tm.xi0 = tm.xi_eta(tm.lat0, tm.lon0).0;
        tm
    }

    /// EPSG:6678 (JGD2011 / Japan Plane Rectangular CS X).
    pub fn epsg6678() -> Self {
        Self::new(40.0, 140.0 + 50.0 / 60.0, 0.9999)
    }

    fn xi_eta(&self, lat: f64, lon: f64) -> (f64, f64) {
        let s = lat.sin();
        let t = (s.atanh() - self.e * (self.e * s).atanh()).sinh();
        let dl = lon - self.lon0;
        let xi_p = t.atan2(dl.cos());
        let eta_p = (dl.sin() / (1.0 + t * t).sqrt()).atanh();
        let (mut xi, mut eta) = (xi_p, eta_p);
        for (j, a) in self.alpha.iter().enumerate() {
            let k = 2.0 * (j + 1) as f64;
            xi += a * (k * xi_p).sin() * (k * eta_p).cosh();
            eta += a * (k * xi_p).cos() * (k * eta_p).sinh();
        }
        (xi, eta)
    }

    /// (easting, northing) in metres from latitude/longitude in degrees.
    pub fn forward(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        let (xi, eta) = self.xi_eta(lat_deg.to_radians(), lon_deg.to_radians());
        (
            self.k0 * self.a_hat * eta,
            self.k0 * self.a_hat * (xi - self.xi0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enu_axes_have_metric_scale_and_earth_curvature() {
        let frame = CoordinateFrame::local_enu([0., 0., 0.]).unwrap();
        assert_eq!(frame.position([0., 0., 0.]), [0.; 3]);
        // At the equator the ECEF axes give independent analytic checks.
        let east = frame.position([0., 90., 0.]);
        assert!((east[0] - 6378137.).abs() < 1e-8);
        assert!(east[1].abs() < 1e-8);
        assert!((east[2] + 6378137.).abs() < 1e-8);
        let north = frame.position([90., 0., 0.]);
        let b = 6378137. * (1. - 1. / 298.257223563);
        assert!((north[1] - b).abs() < 1e-8);
        assert!((north[2] + 6378137.).abs() < 1e-8);
        let metre = frame.position([0., (1.0_f64 / 6378137.).to_degrees(), 0.]);
        assert!((metre[0] - 1.).abs() < 1e-10);
    }

    #[test]
    fn enu_is_local_in_both_hemispheres_and_across_dateline() {
        for origin in [[35.0, 139.0, 45.], [-33.9, 151.2, 12.], [51.5, -0.1, 30.]] {
            let frame = CoordinateFrame::local_enu(origin).unwrap();
            let up = frame.position([origin[0], origin[1], origin[2] + 10.]);
            assert!(up[0].abs() < 1e-8 && up[1].abs() < 1e-8);
            assert!((up[2] - 10.).abs() < 1e-8);
            assert!(frame.position([origin[0], origin[1] + 0.00001, origin[2]])[0] > 0.);
            assert!(frame.position([origin[0] + 0.00001, origin[1], origin[2]])[1] > 0.);
            assert_eq!(frame.output_offset(), [0.; 3]);
        }
        let frame = CoordinateFrame::local_enu([0., 179.99999, 0.]).unwrap();
        let across = frame.position([0., -179.99999, 0.]);
        assert!((across[0] - 2.226389816).abs() < 1e-6);
        assert!(CoordinateFrame::local_enu([91., 0., 0.]).is_err());
        assert!(CoordinateFrame::local_enu([0., f64::NAN, 0.]).is_err());
    }

    #[test]
    fn matches_pyproj() {
        let tm = TransverseMercator::epsg6678();
        // pyproj (EPSG:4326 -> EPSG:6678, always_xy)
        let cases = [
            (39.7, 141.15, 27156.66378241117, -33258.25752533557),
            (41.5, 139.9, -77922.39161173443, 166977.53909698204),
            (37.0, 142.5, 148343.7727183963, -331686.08386926365),
        ];
        for (lat, lon, x, y) in cases {
            let (e, n) = tm.forward(lat, lon);
            assert!(
                (e - x).abs() < 1e-4 && (n - y).abs() < 1e-4,
                "{lat} {lon}: {e} {n} vs {x} {y}"
            );
        }
    }
}
