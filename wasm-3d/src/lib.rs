#![feature(stmt_expr_attributes)]

mod texture;
mod vertex;
mod camera;
mod state;
mod app;
mod model;
mod resources;
mod light;

use std::sync::Arc;

use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::Window, platform::web::WindowAttributesExtWebSys
};
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use crate::app::App;

pub fn run() -> anyhow::Result<()> {
    console_log::init_with_level(log::Level::Info).unwrap_throw();

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        &event_loop,
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
