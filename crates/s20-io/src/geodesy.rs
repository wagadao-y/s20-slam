//! Local GNSS coordinates. Transverse Mercator is retained for reading legacy results only.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

/// Coordinates of the trajectory and its exported points (meters).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CoordinateFrame {
    /// WGS84 geodetic latitude/longitude (degrees), ellipsoidal height (meters).
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

    /// A PDAL pipeline that puts `cloud` (a file name relative to where PDAL runs) of this frame
    /// into a coordinate reference system: east/north/up to WGS84 latitude/longitude/ellipsoidal
    /// height (PROJ's topocentric conversion reversed, as [`CoordinateFrame::position`]), then
    /// `filters.reprojection` to its `out_srs`, the one value to change: `target` (one kept from an
    /// earlier pipeline), else [`default_epsg`] of the origin; a point it cannot transform stops
    /// PDAL (with PROJ_ONLY_BEST_DEFAULT=YES also a transformation whose grid is missing, which
    /// PROJ else replaces silently by a coarser one, such as geoid heights left out). Written as
    /// `<stem>_crs.<ext>` without the extra dimensions (the normals: they are not rotated, and
    /// QGIS 3.44 does not open LAS 1.4 files with extra bytes; `--writers.las.extra_dims=all`
    /// keeps them). Only for a local ENU frame.
    pub fn pdal_pipeline(&self, cloud: &str, target: Option<&str>) -> Option<String> {
        let Self::LocalEnu {
            origin_llh: [lat, lon, h],
        } = *self
        else {
            return None;
        };
        let (stem, ext) = cloud.rsplit_once('.').unwrap_or((cloud, "laz"));
        // written by hand rather than through serde_json, which would sort each stage's keys
        // ("type" last): people read and edit this file
        let quote = |s: &str| serde_json::Value::from(s).to_string();
        let op = format!(
            "+proj=pipeline +step +inv +proj=topocentric +ellps=WGS84 +lat_0={lat} +lon_0={lon} \
             +h_0={h} +step +inv +proj=cart +ellps=WGS84 \
             +step +proj=unitconvert +xy_in=rad +xy_out=deg"
        );
        Some(format!(
            r#"[
  {{ "type": "readers.las", "filename": {input} }},
  {{
    "type": "filters.projpipeline",
    "coord_op": "{op}",
    "out_srs": "EPSG:4979"
  }},
  {{ "type": "filters.reprojection", "out_srs": {target}, "error_on_failure": true }},
  {{
    "type": "writers.las",
    "filename": {output},
    "minor_version": 4,
    "scale_x": 0.0001, "scale_y": 0.0001, "scale_z": 0.0001,
    "offset_x": "auto", "offset_y": "auto", "offset_z": "auto"
  }}
]
"#,
            input = quote(cloud),
            target = quote(&target.map_or_else(
                || format!("EPSG:{}", default_epsg(lat, lon)),
                str::to_string
            )),
            output = quote(&format!("{stem}_crs.{ext}")),
        ))
    }
}

/// The file [`CoordinateFrame::pdal_pipeline`] is written to beside the outputs.
pub const PDAL_PIPELINE_FILE: &str = "pdal_to_crs.json";

/// The `out_srs` of a pipeline [`CoordinateFrame::pdal_pipeline`] wrote, to keep it when the
/// pipeline is written again (none for text that is not such a pipeline).
pub fn pdal_pipeline_target(text: &str) -> Option<String> {
    let stages: serde_json::Value = serde_json::from_str(text).ok()?;
    stages
        .as_array()?
        .iter()
        .find(|s| s["type"] == "filters.reprojection")?["out_srs"]
        .as_str()
        .map(str::to_string)
}

/// The default target of [`CoordinateFrame::pdal_pipeline`] at a latitude/longitude (degrees):
/// the EPSG code of the WGS84 UTM zone (326zz north, 327zz south; the zones' exceptions around
/// Norway and Svalbard not taken), beyond UTM's latitudes (84°N, 80°S) the polar
/// stereographic UPS (5041 north, 5042 south, easting and northing).
pub fn default_epsg(lat: f64, lon: f64) -> u32 {
    if lat > 84. {
        return 5041;
    }
    if lat < -80. {
        return 5042;
    }
    let zone = (((lon + 180.) / 6.).floor() as i64).rem_euclid(60) as u32 + 1;
    if lat >= 0. {
        32600 + zone
    } else {
        32700 + zone
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

    /// (easting, northing) in meters from latitude/longitude in degrees.
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
        let meter = frame.position([0., (1.0_f64 / 6378137.).to_degrees(), 0.]);
        assert!((meter[0] - 1.).abs() < 1e-10);
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
    fn default_targets_utm_by_hemisphere_and_dateline_ups_at_the_poles() {
        assert_eq!(default_epsg(35.0, 139.0), 32654);
        assert_eq!(default_epsg(-33.9, 151.2), 32756);
        assert_eq!(default_epsg(51.5, -0.1), 32630);
        assert_eq!(default_epsg(0., -180.), 32601);
        assert_eq!(default_epsg(0., 179.99), 32660);
        assert_eq!(default_epsg(0., 180.), 32601);
        assert_eq!(default_epsg(84., 10.), 32632);
        assert_eq!(default_epsg(84.5, 10.), 5041);
        assert_eq!(default_epsg(-80., 10.), 32732);
        assert_eq!(default_epsg(-80.5, 10.), 5042);
    }

    #[test]
    fn pdal_pipeline_carries_the_origin_exactly_and_names_the_output() {
        let origin = [35.68123456789012, 139.76712345678901, 40.123456789012345];
        let frame = CoordinateFrame::local_enu(origin).unwrap();
        let text = frame
            .pdal_pipeline("pointcloud_colorized.laz", None)
            .unwrap();
        let p: serde_json::Value = serde_json::from_str(&text).unwrap();
        let stages = p.as_array().unwrap();
        let types: Vec<_> = stages.iter().map(|s| s["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            [
                "readers.las",
                "filters.projpipeline",
                "filters.reprojection",
                "writers.las"
            ]
        );
        assert_eq!(stages[0]["filename"], "pointcloud_colorized.laz");
        // the origin as written by coordinate_frame.json: the shortest text giving the same f64
        let op = stages[1]["coord_op"].as_str().unwrap();
        let value = |key: &str| -> f64 {
            let at = op.find(key).unwrap() + key.len();
            op[at..].split_whitespace().next().unwrap().parse().unwrap()
        };
        assert_eq!([value("+lat_0="), value("+lon_0="), value("+h_0=")], origin);
        assert!(op.starts_with("+proj=pipeline +step +inv +proj=topocentric +ellps=WGS84"));
        assert_eq!(stages[1]["out_srs"], "EPSG:4979");
        assert_eq!(stages[2]["out_srs"], "EPSG:32654");
        assert_eq!(stages[2]["error_on_failure"], true);
        assert_eq!(stages[3]["filename"], "pointcloud_colorized_crs.laz");
        // the normals left out by default (QGIS 3.44 opens no LAS 1.4 file with extra bytes)
        assert!(stages[3].get("extra_dims").is_none());
        assert_eq!(pdal_pipeline_target(&text).as_deref(), Some("EPSG:32654"));
        // a target kept from an earlier pipeline, quoted as JSON
        let kept = frame
            .pdal_pipeline("cloud.las", Some("EPSG:6678+6695"))
            .unwrap();
        assert!(kept.contains("cloud_crs.las"));
        assert_eq!(
            pdal_pipeline_target(&kept).as_deref(),
            Some("EPSG:6678+6695")
        );
        let odd = frame.pdal_pipeline("a.laz", Some("say \"x\"")).unwrap();
        assert_eq!(pdal_pipeline_target(&odd).as_deref(), Some("say \"x\""));
        assert_eq!(pdal_pipeline_target("not json"), None);
        assert_eq!(pdal_pipeline_target("[{\"type\": \"readers.las\"}]"), None);
        assert_eq!(
            CoordinateFrame::LocalSlam.pdal_pipeline("a.laz", None),
            None
        );
        let legacy = CoordinateFrame::LegacyProjected { origin: [0.; 3] };
        assert_eq!(legacy.pdal_pipeline("a.laz", None), None);
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
