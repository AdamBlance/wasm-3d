use cgmath::{InnerSpace, Matrix, Matrix4, Point3, SquareMatrix, Vector3, Vector4};
use winit::keyboard::KeyCode;
use cgmath::prelude::*;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    position: [f32; 4]
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            position: Vector4::zero().into(),
        }
    }
    pub fn update_uniform(&mut self, camera: &mut Camera) {
        self.view_proj = camera.world_to_clip_matrix().into();
        self.position = camera.position.extend(1.0).into();
    }
}

#[derive(Debug)]
pub struct Camera {
    pub up: Vector3<f32>,
    pub lookat: Vector3<f32>,
    pub position: Vector3<f32>,
    pub horizontal_fov: f32,
    pub aspect: f32,
    pub near_clip: f32,
    pub far_clip: f32,
}

fn from_rows(r0: [f32; 4], r1: [f32; 4], r2: [f32; 4], r3: [f32; 4]) -> Matrix4<f32> {
    Matrix4::from_cols(r0.into(), r1.into(), r2.into(), r3.into()).transpose()
}

impl Camera {
    fn world_to_clip_matrix(&mut self) -> Matrix4<f32> {
        self.projection_matrix() * self.world_to_view_matrix()
    }

    fn world_to_view_matrix(&self) -> Matrix4<f32> {
        /*
            This is the product of two transforms:
                1. The matrix that translates the camera to the origin
                2. The inverse* of the matrix whose columns are the camera's basis vectors
                   in world space, which rotates the camera to point down the Z+ axis.
            *The inverse, in the case of orthogonal matrices, is the same as the transpose.
        */
        let camera_positive_z = (self.lookat - self.position).normalize();
        let camera_positive_x = camera_positive_z.cross(self.up).normalize();
        let camera_positive_y = camera_positive_x.cross(camera_positive_z);
        #[rustfmt::skip]
        from_rows(
            [camera_positive_x.x, camera_positive_x.y, camera_positive_x.z, -self.position.dot(camera_positive_x)],
            [camera_positive_y.x, camera_positive_y.y, camera_positive_y.z, -self.position.dot(camera_positive_y)],
            [camera_positive_z.x, camera_positive_z.y, camera_positive_z.z, -self.position.dot(camera_positive_z)],
            [0.0,                 0.0,                 0.0,                 1.0                                  ]
        )
    }

    fn projection_matrix(&self) -> Matrix4<f32> {
        #[rustfmt::skip]
        from_rows(
            [1.0 / (self.aspect * ((self.horizontal_fov / 2.0).tan())), 0.0,                                       0.0,                                              0.0                                                                 ],
            [0.0,                                                       1.0 / ((self.horizontal_fov / 2.0).tan()), 0.0,                                              0.0                                                                 ],
            [0.0,                                                       0.0,                                       self.far_clip / (self.far_clip - self.near_clip), (-self.far_clip * self.near_clip) / (self.far_clip - self.near_clip)],
            [0.0,                                                       0.0,                                       1.0,                                              0.0                                                                 ]
        )
    }
}

#[derive(Debug)]
pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    pub fn handle_key(&mut self, code: KeyCode, is_pressed: bool) -> bool {
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.is_forward_pressed = is_pressed;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.is_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.is_backward_pressed = is_pressed;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.is_right_pressed = is_pressed;
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.lookat - camera.position;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when the camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.position += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.position -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = camera.lookat - camera.position;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and the eye so
            // that it doesn't change. The eye, therefore, still
            // lies on the circle made by the target and eye.
            camera.position = camera.lookat - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.position = camera.lookat - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}
