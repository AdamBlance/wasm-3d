use std::sync::Arc;
use wasm_bindgen::{JsCast, UnwrapThrowExt};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::web::WindowAttributesExtWebSys;
use winit::window::Window;
use crate::state::State;

pub struct App {
    /*
        State creation has to happen asynchronously because it needs to go away and negotiate with
        the OS, GPU etc.
        When the event loop first starts, it calls ApplicationHandler::resumed. This is where we
        have to create the State (and the Window for the State code to draw on).
        The ApplicationHandler methods are not async, so we need to give the State creation work to
        a thread.
        When the thread finishes creating the State, it needs to hand it back to the App in the
        main thread. It does this by sending the State as an event to the event loop with the proxy.
        ApplicationHandler::user_event then receives this event and populates the App::state field
        with the new State.
        The proxy is optional so that we can give ownership (with .take()) to the async thread.
     */
    event_loop_proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

impl App {
    pub fn new(event_loop: &EventLoop<State>) -> Self {
        Self {
            state: None,
            event_loop_proxy: Some(event_loop.create_proxy()),
        }
    }
}

impl ApplicationHandler<State> for App {
    // Called when the window first loads (only?)
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let document = web_sys::window().unwrap_throw().document().unwrap_throw();
        let canvas = document.create_element("canvas").unwrap_throw();
        canvas.set_id("wgpu-canvas");
        canvas.set_attribute("style", "width: 100%; height: 100%;").unwrap_throw();
        document.body().unwrap_throw().append_child(&canvas).unwrap_throw();

        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_canvas(Some(canvas.unchecked_into()))
            ).unwrap()
        );

        let owned_proxy = self.event_loop_proxy.take().unwrap();
        let async_state_creation = async move {
            owned_proxy.send_event(
                State::new(window)
                    .await
                    .expect("Something went wrong with window creation")
            ).unwrap()
        };
        // Create State asynchronously
        wasm_bindgen_futures::spawn_local(async_state_creation);
    }

    // State created! Store it in App.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: State) {
        let mut initial_state = event;

        // Create RedrawRequested event so we can start drawing!
        initial_state.window.request_redraw();

        // We need to tell State the size of the HTML canvas/window so it can configure the WebGPU
        // surface for the first time. This also sets is_surface_configured so that State knows
        // it can safely start rendering to the surface.
        let size = initial_state.window.inner_size();
        initial_state.resize(size.width, size.height);

        // Finally set State in App
        self.state = Some(initial_state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Return if State isn't ready yet
        let state = match &mut self.state {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => state.render().unwrap_throw(),
            WindowEvent::CursorMoved { position, .. } => {
                let height = state.window.inner_size().height;
                state.blueness = (position.y / height as f64 ).clamp(0.0, 1.0);
            }
            _ => {}
        }
    }
}