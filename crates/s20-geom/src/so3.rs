//! SO(3) / SE(3) helpers on nalgebra types (rotation matrices, rotation vectors, quaternions).

use nalgebra::{Matrix3, Quaternion, Rotation3, UnitQuaternion, Vector3};

pub type M3 = Matrix3<f64>;
pub type V3 = Vector3<f64>;

#[inline]
pub fn skew(w: &V3) -> M3 {
    M3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0)
}

/// Rodrigues formula (same small-angle handling as the Python version).
#[inline]
pub fn exp(w: &V3) -> M3 {
    let th2 = w.norm_squared();
    let k = skew(w);
    if th2 < 1e-20 {
        return M3::identity() + k;
    }
    let th = th2.sqrt();
    M3::identity() + (th.sin() / th) * k + ((1.0 - th.cos()) / th2) * (k * k)
}

/// Rotation vector of a rotation matrix. Robust to rounding (no acos of values slightly above 1).
#[inline]
pub fn log(r: &M3) -> V3 {
    let tr = r.trace();
    let vee = V3::new(
        r[(2, 1)] - r[(1, 2)],
        r[(0, 2)] - r[(2, 0)],
        r[(1, 0)] - r[(0, 1)],
    );
    if tr > 3.0 - 1e-6 {
        // small angle: sin(th) ~ th, th^2 ~ 3 - tr
        return 0.5 * vee * (1.0 + (3.0 - tr) / 6.0);
    }
    if tr > -0.9 {
        let c = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0);
        let s = 0.5 * vee.norm();
        let th = s.atan2(c);
        return vee * (0.5 * th / s);
    }
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r));
    q.scaled_axis()
}

/// Quaternion [qx, qy, qz, qw] of a rotation matrix.
pub fn to_quat(r: &M3) -> [f64; 4] {
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r));
    let mut c = q.coords; // (x, y, z, w)
    // scipy convention: w >= 0
    if c.w < 0.0 {
        c = -c;
    }
    [c.x, c.y, c.z, c.w]
}

pub fn from_quat(q: [f64; 4]) -> M3 {
    let u = UnitQuaternion::from_quaternion(Quaternion::new(q[3], q[0], q[1], q[2]));
    *u.to_rotation_matrix().matrix()
}

/// Re-orthonormalise a rotation matrix that accumulated rounding error.
pub fn normalize(r: &M3) -> M3 {
    from_quat(to_quat(r))
}

/// Rotation that maps unit vector `a` onto unit vector `b` (minimal rotation).
pub fn align_vectors(a: &V3, b: &V3) -> M3 {
    let a = a.normalize();
    let b = b.normalize();
    match UnitQuaternion::rotation_between(&a, &b) {
        Some(q) => *q.to_rotation_matrix().matrix(),
        None => {
            // opposite vectors: rotate pi about any axis orthogonal to a
            let ax = if a.x.abs() < 0.9 { V3::x() } else { V3::y() };
            let n = a.cross(&ax).normalize();
            exp(&(n * std::f64::consts::PI))
        }
    }
}

/// SE(3) exponential of [omega, v] (small_gicp `se3_exp`): returns (R, t).
pub fn se3_exp(a: &[f64; 6]) -> (M3, V3) {
    let w = V3::new(a[0], a[1], a[2]);
    let v = V3::new(a[3], a[4], a[5]);
    let r = exp(&w);
    let th2 = w.norm_squared();
    if th2 < 1e-20 {
        return (r, r * v);
    }
    let th = th2.sqrt();
    let o = skew(&w);
    let vm =
        M3::identity() + ((1.0 - th.cos()) / th2) * o + ((th - th.sin()) / (th2 * th)) * (o * o);
    (r, vm * v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_log_roundtrip() {
        let w = V3::new(0.3, -0.2, 0.9);
        let r = exp(&w);
        assert!((log(&r) - w).norm() < 1e-12);
        let q = to_quat(&r);
        assert!((from_quat(q) - r).norm() < 1e-12);
    }

    #[test]
    fn log_ranges() {
        for w in [
            V3::new(1e-9, -2e-9, 3e-9),
            V3::new(1e-4, 2e-4, -1e-4),
            V3::new(0.5, -1.0, 0.7),
            V3::new(3.0, 0.5, -0.2) * (3.1 / 3.0496),
        ] {
            let r = exp(&w);
            assert!(
                (log(&r) - w).norm() < 1e-9 * (1.0 + w.norm()),
                "{w} {}",
                log(&r)
            );
        }
        // slightly non-orthonormal input must not produce NaN
        let r = M3::identity() * (1.0 + 1e-12);
        assert!(log(&r).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn align() {
        let a = V3::new(0.1, 0.2, 0.97);
        let r = align_vectors(&a, &V3::z());
        assert!((r * a.normalize() - V3::z()).norm() < 1e-12);
    }
}
