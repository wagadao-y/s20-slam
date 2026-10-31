//! Sampled trajectory (world_T_imu) with interpolation: linear in position, slerp in rotation.

use crate::so3::{self, M3, V3};
use nalgebra::{Quaternion, UnitQuaternion};

#[derive(Clone, Default)]
pub struct Traj {
    pub t: Vec<f64>,
    pub p: Vec<V3>,
    pub q: Vec<UnitQuaternion<f64>>,
}

impl Traj {
    /// From rows `t x y z qx qy qz qw` (sorted by time).
    pub fn from_rows(rows: &[[f64; 8]]) -> Self {
        let mut tr = Traj::default();
        for r in rows {
            if let Some(&l) = tr.t.last()
                && r[0] <= l
            {
                continue;
            }
            tr.t.push(r[0]);
            tr.p.push(V3::new(r[1], r[2], r[3]));
            tr.q.push(UnitQuaternion::from_quaternion(Quaternion::new(
                r[7], r[4], r[5], r[6],
            )));
        }
        tr
    }

    pub fn len(&self) -> usize {
        self.t.len()
    }
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }
    pub fn start(&self) -> f64 {
        self.t[0]
    }
    pub fn end(&self) -> f64 {
        *self.t.last().unwrap()
    }

    /// Pose at time t (clamped to the ends).
    pub fn at(&self, t: f64) -> (M3, V3) {
        let n = self.t.len();
        let i = self.t.partition_point(|&s| s <= t);
        if i == 0 {
            return (*self.q[0].to_rotation_matrix().matrix(), self.p[0]);
        }
        if i >= n {
            return (*self.q[n - 1].to_rotation_matrix().matrix(), self.p[n - 1]);
        }
        let (a, b) = (i - 1, i);
        let u = (t - self.t[a]) / (self.t[b] - self.t[a]);
        let p = self.p[a] * (1.0 - u) + self.p[b] * u;
        let q = self.q[a].slerp(&self.q[b], u);
        (*q.to_rotation_matrix().matrix(), p)
    }

    /// The time between the samples around t (0 at a sample, infinite outside the trajectory).
    pub fn gap_at(&self, t: f64) -> f64 {
        let i = self.t.partition_point(|&s| s < t);
        match (i.checked_sub(1), self.t.get(i)) {
            (_, Some(&b)) if b == t => 0.0,
            (Some(a), Some(&b)) => b - self.t[a],
            _ => f64::INFINITY,
        }
    }

    /// Rotation matrices of the samples (for fast per-point interpolation inside a sample interval).
    pub fn rot(&self, i: usize) -> M3 {
        *self.q[i].to_rotation_matrix().matrix()
    }

    /// Relative rotation vector between samples i and i+1.
    pub fn delta(&self, i: usize) -> V3 {
        so3::log(&(self.rot(i).transpose() * self.rot(i + 1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_is_the_interval_around_a_time() {
        let row = |t| [t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let tr = Traj::from_rows(&[row(0.0), row(0.1), row(2.0)]);
        assert_eq!(tr.gap_at(0.05), 0.1);
        assert_eq!(tr.gap_at(0.1), 0.0);
        assert_eq!(tr.gap_at(1.0), 1.9);
        assert_eq!(tr.gap_at(-0.1), f64::INFINITY);
        assert_eq!(tr.gap_at(2.1), f64::INFINITY);
    }
}
