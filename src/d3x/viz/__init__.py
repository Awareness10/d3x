"""D3X visualization module - real-time 3D orbital viewer.

Usage:
    from d3x import World, step_rk4
    from d3x.viz import Viewer

    world = World()
    world.add_body(pos=(0, 0, 0), vel=(0, 0, 0), mass=1.989e30)
    world.add_body(pos=(1.496e11, 0, 0), vel=(0, 29780, 0), mass=5.972e24)

    viewer = Viewer(world)
    while viewer.running:
        step_rk4(world, 3600)
        viewer.update()
"""

import time
from typing import TYPE_CHECKING

import glfw
import numpy as np

from .camera import Camera
from .renderer import Renderer
from .trail import TrailBuffer
from .window import Window

if TYPE_CHECKING:
    from d3x import World

__all__ = ["Viewer"]


class Viewer:
    """Real-time 3D viewer for orbital simulations.

    Args:
        world: The simulation World to visualize
        width: Window width in pixels
        height: Window height in pixels
        title: Window title
        trail_length: Number of position samples to keep for trails
        auto_scale: Automatically scale view to fit simulation
        show_fps: Display FPS and sim time in window title
    """

    def __init__(
        self,
        world: "World",
        width: int = 1280,
        height: int = 720,
        title: str = "D3X Orbital Viewer",
        trail_length: int = 500,
        auto_scale: bool = True,
        show_fps: bool = False,
    ):
        self.world = world
        self.auto_scale = auto_scale
        self._paused = False
        self._scale = 1.0
        self._base_title = title
        self._show_fps = show_fps
        self._selected_body: int | None = None  # Currently selected/tracked body

        # Create window
        self.window = Window(width, height, title)
        # Camera looking down at XY orbital plane from above
        self.camera = Camera(distance=12.0, pitch=1.0, yaw=0.3)
        self.camera.aspect = width / height
        self.window.camera = self.camera

        # Create renderer
        self.renderer = Renderer(self.window.ctx, max_bodies=64)
        self.window.on_resize(self._on_resize)

        # Create trail buffer
        self.trail_buffer = TrailBuffer(max_bodies=64, trail_length=trail_length)

        # Register callbacks
        self.window.on_key(self._on_key)
        self.window.on_click(self._on_click)

        # Timing
        self._last_time = time.perf_counter()
        self._frame_count = 0
        self._fps = 0.0
        self._scale_initialized = False

    def _on_resize(self, width: int, height: int) -> None:
        self.renderer.resize(width, height)
        self.camera.aspect = width / max(height, 1)

    def _on_click(self, x: float, y: float, button: int) -> None:
        """Handle mouse click for body selection."""
        if button != 0:  # Left click only
            return

        # Pick body under cursor
        picked = self._pick_body(x, y)
        if picked is not None:
            self._selected_body = picked
        else:
            # Click on empty space deselects
            self._selected_body = None

    def _pick_body(self, screen_x: float, screen_y: float) -> int | None:
        """Find which body (if any) is under the screen coordinates."""
        if self.world.count == 0:
            return None

        width, height = self.window.width, self.window.height

        # Normalized device coordinates (-1 to 1)
        ndc_x = (2.0 * screen_x / width) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / height)  # Flip Y

        view_proj = self.camera.view_projection_matrix()

        best_idx = None
        best_dist = float("inf")
        pick_radius = 0.05  # NDC radius for picking tolerance

        for i in range(self.world.count):
            # Get body position in world space
            pos = np.array(
                [
                    self.world.px[i] * self._scale,
                    self.world.py_[i] * self._scale,
                    self.world.pz[i] * self._scale,
                    1.0,
                ],
                dtype=np.float32,
            )

            # Project to clip space
            clip = view_proj @ pos
            if clip[3] <= 0:  # Behind camera
                continue

            # To NDC
            ndc = clip[:2] / clip[3]

            # Distance in NDC space
            dist = np.sqrt((ndc[0] - ndc_x) ** 2 + (ndc[1] - ndc_y) ** 2)

            if dist < pick_radius and dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    @property
    def selected_body(self) -> int | None:
        """Index of currently selected body, or None."""
        return self._selected_body

    @selected_body.setter
    def selected_body(self, idx: int | None) -> None:
        """Set selected body by index."""
        if idx is None or 0 <= idx < self.world.count:
            self._selected_body = idx

    def _on_key(self, key: int) -> None:
        if key == glfw.KEY_ESCAPE:
            self.window.should_close = True
        elif key == glfw.KEY_R:
            self.camera.reset()
            self._scale_initialized = False  # Recalculate scale on reset
        elif key == glfw.KEY_G:
            self.renderer.show_grid = not self.renderer.show_grid
        elif key == glfw.KEY_A:
            self.renderer.show_axes = not self.renderer.show_axes
        elif key == glfw.KEY_SPACE:
            self._paused = not self._paused
        elif key == glfw.KEY_C:
            self.trail_buffer.clear()
        # Select body 1-9
        elif glfw.KEY_1 <= key <= glfw.KEY_9:
            body_idx = key - glfw.KEY_1
            if body_idx < self.world.count:
                self._selected_body = body_idx
        # Deselect with 0 or Backspace
        elif key in (glfw.KEY_0, glfw.KEY_BACKSPACE):
            self._selected_body = None

    def _update_scale(self) -> None:
        """Calculate scale factor to fit simulation in view."""
        if not self.auto_scale or self.world.count == 0:
            return

        # Find maximum extent
        px = np.array(self.world.px)
        py = np.array(self.world.py_)
        pz = np.array(self.world.pz)

        max_extent = max(
            np.abs(px).max() if len(px) > 0 else 1.0,
            np.abs(py).max() if len(py) > 0 else 1.0,
            np.abs(pz).max() if len(pz) > 0 else 1.0,
            1.0,
        )

        # Scale to fit in view (normalized to ~4 units radius)
        self._scale = 4.0 / max_extent

    @property
    def running(self) -> bool:
        """Check if viewer is still running."""
        return not self.window.should_close

    @property
    def paused(self) -> bool:
        """Check if simulation is paused."""
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = value

    @property
    def fps(self) -> float:
        """Current frames per second."""
        return self._fps

    def update(self) -> None:
        """Update display - call once per simulation step."""
        # 1. Process window events
        self.window.poll_events()

        if not self.running:
            return

        # 2. Logic & Scaling
        if not self._scale_initialized:
            self._update_scale()
            self._scale_initialized = True

        self.trail_buffer.push(self.world)

        # 3. Camera Tracking
        if self._selected_body is not None and self._selected_body < self.world.count:
            target_pos = np.array(
                [
                    self.world.px[self._selected_body] * self._scale,
                    self.world.py_[self._selected_body] * self._scale,
                    self.world.pz[self._selected_body] * self._scale,
                ],
                dtype=np.float32,
            )
            self.camera.target += (target_pos - self.camera.target) * 0.15

        self.camera.update()

        # 4. Prepare & Update GPU Buffers
        n = self.world.count
        if n > 0:
            positions = np.column_stack(
                [
                    self.world.px[:n],
                    self.world.py_[:n],
                    self.world.pz[:n],
                ]
            )
            self.renderer.update_bodies(
                positions,
                np.array(self.world.mass[:n]),
                scale=self._scale,
                selected=self._selected_body,
            )
            self.renderer.update_trails(self.trail_buffer, n, scale=self._scale)

        # 5. Render Scene
        self.renderer.render(self.camera)

        # 6. Overlay & FPS logic
        self._frame_count += 1
        now = time.perf_counter()
        dt = now - self._last_time

        # Update FPS calculation every 250ms
        if dt >= 0.25:
            self._fps = self._frame_count / dt
            self._frame_count = 0
            self._last_time = now

        # TODO
        # if self._show_fps:

        # 7. Finalize frame
        self.window.swap_buffers()

    def close(self) -> None:
        """Close viewer and release resources."""
        self.renderer.release()
        self.window.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
