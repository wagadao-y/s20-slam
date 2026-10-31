//! The S20 polynomial fisheye: r(theta) = theta + k2 theta^2 + ... + k7 theta^7, then the affine
//! A11, A12, A22, u0, v0 to pixels (as `s20_finish::color` projects).

use s20_io::calib::FisheyeIntrinsics;

#[derive(Clone, Debug)]
pub struct Fisheye {
    pub k: [f64; 6],
    pub a11: f64,
    pub a12: f64,
    pub a22: f64,
    pub u0: f64,
    pub v0: f64,
}

impl From<&FisheyeIntrinsics> for Fisheye {
    fn from(c: &FisheyeIntrinsics) -> Self {
        Self {
            k: c.k,
            a11: c.a11,
            a12: c.a12,
            a22: c.a22,
            u0: c.u0,
            v0: c.v0,
        }
    }
}

impl Fisheye {
    /// r(theta) before the affine.
    pub fn radius(&self, th: f64) -> f64 {
        let k = &self.k;
        th * (1. + th * (k[0] + th * (k[1] + th * (k[2] + th * (k[3] + th * (k[4] + th * k[5]))))))
    }

    /// Pixel of a camera-frame direction (any length, z forward).
    pub fn project(&self, x: [f64; 3]) -> [f64; 2] {
        let rho = x[0].hypot(x[1]);
        let th = rho.atan2(x[2]);
        let rr = self.radius(th);
        let (c, s) = if rho > 1e-15 {
            (x[0] / rho, x[1] / rho)
        } else {
            (1., 0.)
        };
        [
            self.a11 * rr * c + self.a12 * rr * s + self.u0,
            self.a22 * rr * s + self.v0,
        ]
    }

    /// Unit camera-frame ray of a pixel (theta up to 85 degrees, by bisection as the polynomial may
    /// bend near the rim).
    pub fn ray(&self, uv: [f64; 2]) -> [f64; 3] {
        let y = (uv[1] - self.v0) / self.a22;
        let x = (uv[0] - self.u0 - self.a12 * y) / self.a11;
        let r = x.hypot(y);
        let (mut lo, mut hi) = (0., 85f64.to_radians());
        for _ in 0..55 {
            let mid = 0.5 * (lo + hi);
            if self.radius(mid) < r {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let th = 0.5 * (lo + hi);
        let s = th.sin() / r.max(1e-15);
        [x * s, y * s, th.cos()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_invert_the_projection_across_the_field() {
        let cam = Fisheye {
            k: [-2.3e-3, 2.67e-2, -3.07e-2, 4.48e-2, -2.79e-2, 7.26e-3],
            a11: 1481.9,
            a12: 0.245,
            a22: 1483.4,
            u0: 1732.3,
            v0: 2308.3,
        };
        for th in [0.01f64, 0.3, 0.8, 1.2, 1.38] {
            for ph in [0.1f64, 2.0, 4.0] {
                let d = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
                let r = cam.ray(cam.project(d));
                assert!((0..3).all(|i| (r[i] - d[i]).abs() < 1e-9), "{d:?} {r:?}");
            }
        }
    }
}
