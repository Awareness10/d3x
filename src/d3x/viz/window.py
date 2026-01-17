"""GLFW window management for cross-platform visualization."""

import glfw
import moderngl

from .camera import Camera


class Window:
    """GLFW window with input handling."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        title: str = "D3X Orbital Viewer",
        vsync: bool = True,
    ):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # OpenGL context hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 4)  # MSAA

        self.handle = glfw.create_window(width, height, title, None, None)
        if not self.handle:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.handle)
        glfw.swap_interval(1 if vsync else 0)

        # Create ModernGL context
        self.ctx = moderngl.create_context()

        # Input state
        self._mouse_pos = (0.0, 0.0)
        self._mouse_pressed = {
            glfw.MOUSE_BUTTON_LEFT: False,
            glfw.MOUSE_BUTTON_MIDDLE: False,
            glfw.MOUSE_BUTTON_RIGHT: False,
        }
        self._keys_pressed: set[int] = set()
        self._ctrl_held = False

        # Callbacks
        self._on_key: list = []
        self._on_resize: list = []
        self._on_click: list = []

        # Register GLFW callbacks
        glfw.set_cursor_pos_callback(self.handle, self._cursor_callback)
        glfw.set_mouse_button_callback(self.handle, self._mouse_button_callback)
        glfw.set_scroll_callback(self.handle, self._scroll_callback)
        glfw.set_key_callback(self.handle, self._key_callback)
        glfw.set_framebuffer_size_callback(self.handle, self._resize_callback)

        # Camera reference (set externally)
        self.camera: Camera | None = None

        self.width = width
        self.height = height

    def _cursor_callback(self, window, x: float, y: float) -> None:
        dx = x - self._mouse_pos[0]
        dy = y - self._mouse_pos[1]
        self._mouse_pos = (x, y)

        if self.camera is None:
            return

        left_down = self._mouse_pressed[glfw.MOUSE_BUTTON_LEFT]
        middle_down = self._mouse_pressed[glfw.MOUSE_BUTTON_MIDDLE]
        right_down = self._mouse_pressed[glfw.MOUSE_BUTTON_RIGHT]

        # Middle drag or Right drag = pan
        if middle_down or right_down:
            self.camera.pan(dx, dy)
        # Left = orbit (Shift+Left = pan alternative)
        elif left_down:
            if self._ctrl_held:
                self.camera.pan(dx, dy)
            else:
                self.camera.orbit(dx, dy)

    def _mouse_button_callback(self, window, button: int, action: int, mods: int) -> None:
        was_pressed = self._mouse_pressed.get(button, False)
        if button in self._mouse_pressed:
            self._mouse_pressed[button] = action == glfw.PRESS

        # Fire click on release (if wasn't a drag)
        if action == glfw.RELEASE and was_pressed:
            for callback in self._on_click:
                callback(self._mouse_pos[0], self._mouse_pos[1], button)

    def _scroll_callback(self, window, xoffset: float, yoffset: float) -> None:
        if self.camera is not None:
            self.camera.zoom(yoffset)

    def _key_callback(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        # Track Ctrl key state
        if key in (glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL):
            self._ctrl_held = action != glfw.RELEASE

        if action == glfw.PRESS:
            self._keys_pressed.add(key)
            for callback in self._on_key:
                callback(key)
        elif action == glfw.RELEASE:
            self._keys_pressed.discard(key)

    def _resize_callback(self, window, width: int, height: int) -> None:
        self.width = width
        self.height = height
        if self.camera is not None:
            self.camera.aspect = width / max(height, 1)
        for callback in self._on_resize:
            callback(width, height)

    def on_key(self, callback) -> None:
        """Register key press callback."""
        self._on_key.append(callback)

    def on_resize(self, callback) -> None:
        """Register resize callback."""
        self._on_resize.append(callback)

    def on_click(self, callback) -> None:
        """Register click callback (x, y, button)."""
        self._on_click.append(callback)

    def is_key_pressed(self, key: int) -> bool:
        """Check if key is currently pressed."""
        return key in self._keys_pressed

    @property
    def should_close(self) -> bool:
        """Check if window should close."""
        return glfw.window_should_close(self.handle)

    @should_close.setter
    def should_close(self, value: bool) -> None:
        glfw.set_window_should_close(self.handle, value)

    def poll_events(self) -> None:
        """Process pending events."""
        glfw.poll_events()

    def swap_buffers(self) -> None:
        """Swap front and back buffers."""
        glfw.swap_buffers(self.handle)

    def close(self) -> None:
        """Close window and terminate GLFW."""
        glfw.destroy_window(self.handle)
        glfw.terminate()

    @property
    def framebuffer_size(self) -> tuple[int, int]:
        """Get actual framebuffer size (for HiDPI)."""
        return glfw.get_framebuffer_size(self.handle)
