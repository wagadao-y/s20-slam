//! Orbit camera and its mouse navigation, adapted from Geemil Workbench.
//! Z is up. Middle or left drag orbits, right drag pans, the wheel zooms.

use eframe::egui;
use glam::{DMat4, DVec3, DVec4};

pub const MAX_PITCH: f64 = std::f64::consts::FRAC_PI_2 - 1e-3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub target: [f64; 3],
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub aspect: f64,
    pub fov: f64,
    /// Parallel projection; the visible height stays what perspective shows at `target`.
    pub ortho: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: [0.; 3],
            yaw: -1.2,
            pitch: 0.65,
            distance: 10.,
            aspect: 1.,
            fov: std::f64::consts::FRAC_PI_4,
            ortho: false,
        }
    }
}

impl Camera {
    pub fn eye(&self) -> DVec3 {
        DVec3::from(self.target)
            + DVec3::new(
                self.yaw.cos() * self.pitch.cos(),
                self.yaw.sin() * self.pitch.cos(),
                self.pitch.sin(),
            ) * self.distance
    }
    pub fn relative_matrix(&self) -> DMat4 {
        self.projection() * self.relative_view()
    }
    /// Right-handed with a 0..1 depth range, as wgpu expects.
    pub fn projection(&self) -> DMat4 {
        if self.ortho {
            let h = self.half_height();
            let w = h * self.aspect;
            let reach = (self.distance * 1000.).max(100.);
            glam::dcamera::rh::proj::directx::orthographic(-w, w, -h, h, -reach, reach)
        } else {
            glam::dcamera::rh::proj::directx::perspective(
                self.fov,
                self.aspect,
                (self.distance * 1e-5).max(0.0001),
                (self.distance * 1000.).max(100.),
            )
        }
    }
    pub fn half_height(&self) -> f64 {
        self.distance * (self.fov * 0.5).tan()
    }
    pub fn relative_view(&self) -> DMat4 {
        glam::dcamera::rh::view::look_at_mat4(
            self.eye() - DVec3::from(self.target),
            DVec3::ZERO,
            DVec3::Z,
        )
    }
    /// View depth of a world point `p` as `row.dot(p.extend(1))`.
    pub fn depth_row(&self) -> DVec4 {
        let view = self.relative_view() * DMat4::from_translation(-DVec3::from(self.target));
        -view.row(2)
    }

    /// Normalized viewport coordinates (origin top left) of a point in the view volume.
    pub fn project(&self, p: DVec3) -> Option<[f64; 2]> {
        let m = self.relative_matrix() * DMat4::from_translation(-DVec3::from(self.target));
        let clip = m * p.extend(1.);
        if clip.w <= 0. || clip.z < 0. || clip.z > clip.w {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some([(ndc.x + 1.) * 0.5, (1. - ndc.y) * 0.5])
    }

    /// (right, up, forward) of the view.
    pub fn basis(&self) -> (DVec3, DVec3, DVec3) {
        let forward = -DVec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
        );
        let right = DVec3::new(-self.yaw.sin(), self.yaw.cos(), 0.);
        let up = right.cross(forward);
        (right, up, forward)
    }

    pub fn orbit(&mut self, delta: egui::Vec2) {
        self.yaw -= delta.x as f64 * 0.007;
        self.pitch = (self.pitch + delta.y as f64 * 0.007).clamp(-MAX_PITCH, MAX_PITCH);
    }

    pub fn pan(&mut self, delta: egui::Vec2, height: f32) {
        let (right, up, _) = self.basis();
        self.target = (DVec3::from(self.target)
            + (right * (-delta.x as f64) + up * delta.y as f64) * self.distance
                / height.max(1.) as f64)
            .to_array();
    }

    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (-scroll as f64 * 0.003).exp()).clamp(0.001, 1e10);
    }

    /// The same eye position, looking at and orbiting around `target`.
    pub fn looking_at(&self, target: DVec3) -> Self {
        if self.ortho {
            return Self {
                target: target.to_array(),
                ..*self
            };
        }
        let offset = self.eye() - target;
        let distance = offset.length();
        if distance <= f64::EPSILON {
            return *self;
        }
        Self {
            target: target.to_array(),
            yaw: offset.y.atan2(offset.x),
            pitch: (offset.z / distance).asin().clamp(-MAX_PITCH, MAX_PITCH),
            distance,
            ..*self
        }
    }

    /// Fits points (e.g. bounding box corners) keeping the view direction.
    pub fn frame(&mut self, corners: &[DVec3]) {
        if corners.is_empty() {
            return;
        }
        let target = corners.iter().copied().sum::<DVec3>() / corners.len() as f64;
        let (right, up, forward) = self.basis();
        let vertical = (self.fov * 0.5).tan();
        let horizontal = vertical * self.aspect;
        let mut distance: f64 = 0.001;
        for corner in corners {
            let q = corner - target;
            let needed = (q.dot(right).abs() / horizontal).max(q.dot(up).abs() / vertical) * 1.08;
            distance = distance.max(if self.ortho {
                needed
            } else {
                (needed - q.dot(forward)).max(0.001 - q.dot(forward))
            });
        }
        self.target = target.to_array();
        self.distance = distance;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_target_to_centre_and_right_to_the_right() {
        for ortho in [false, true] {
            let c = Camera {
                target: [3.0, -2.0, 1.0],
                aspect: 1.5,
                ortho,
                ..Default::default()
            };
            let uv = c.project(DVec3::from(c.target)).unwrap();
            assert!((uv[0] - 0.5).abs() < 1e-9 && (uv[1] - 0.5).abs() < 1e-9);
            let (right, up, _) = c.basis();
            let r = c.project(DVec3::from(c.target) + right).unwrap();
            assert!(r[0] > 0.5 && (r[1] - 0.5).abs() < 1e-9);
            let u = c.project(DVec3::from(c.target) + up).unwrap();
            assert!(u[1] < 0.5);
        }
    }
}
