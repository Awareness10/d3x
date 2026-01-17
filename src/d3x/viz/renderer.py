"""ModernGL renderer for orbital visualization."""

import moderngl
import numpy as np

from .camera import Camera
from .shaders import (
    AXIS_FRAGMENT,
    AXIS_VERTEX,
    BODY_FRAGMENT,
    BODY_VERTEX,
    GIZMO_FRAGMENT,
    GIZMO_VERTEX,
    GRID_FRAGMENT,
    GRID_VERTEX,
    TRAIL_FRAGMENT,
    TRAIL_VERTEX,
)
from .trail import TrailBuffer

# Professional dark color scheme
BACKGROUND_COLOR = (0.02, 0.02, 0.03, 1.0)  # Deep space black
GRID_COLOR = (0.3, 0.3, 0.4, 1.0)  # Visible grid

# Muted body colors (avoiding neon)
BODY_COLORS = [
    (0.35, 0.55, 0.85),  # Body 0 - blue (Earth-like)
    (0.70, 0.70, 0.72),  # Body 1 - gray (Moon-like)
    (0.85, 0.65, 0.45),  # Body 2 - orange (Mars-like)
    (0.95, 0.85, 0.45),  # Body 3 - yellow (Sun/Venus)
    (0.75, 0.55, 0.45),  # Body 4 - rust (Mercury)
    (0.50, 0.75, 0.55),  # Body 5 - green
    (0.80, 0.60, 0.75),  # Body 6 - purple
    (0.45, 0.70, 0.70),  # Body 7 - teal
]


class Renderer:
    """ModernGL renderer managing GPU resources and draw calls."""

    def __init__(self, ctx: moderngl.Context, max_bodies: int = 64):
        self.ctx = ctx
        self.max_bodies = max_bodies

        # Enable required features
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # Compile shader programs
        self.body_prog = ctx.program(
            vertex_shader=BODY_VERTEX,
            fragment_shader=BODY_FRAGMENT,
        )
        self.trail_prog = ctx.program(
            vertex_shader=TRAIL_VERTEX,
            fragment_shader=TRAIL_FRAGMENT,
        )
        self.grid_prog = ctx.program(
            vertex_shader=GRID_VERTEX,
            fragment_shader=GRID_FRAGMENT,
        )
        self.axis_prog = ctx.program(
            vertex_shader=AXIS_VERTEX,
            fragment_shader=AXIS_FRAGMENT,
        )
        self.gizmo_prog = ctx.program(
            vertex_shader=GIZMO_VERTEX,
            fragment_shader=GIZMO_FRAGMENT,
        )

        # Create buffers
        self._init_body_buffers()
        self._init_grid_buffers()
        self._init_axis_buffers()
        self._init_gizmo_buffers()

        # Trail buffers (created dynamically)
        self.trail_vbo: moderngl.Buffer | None = None
        self.trail_age_vbo: moderngl.Buffer | None = None
        self.trail_color_vbo: moderngl.Buffer | None = None
        self.trail_vao: moderngl.VertexArray | None = None

        self.show_grid = True
        self.show_axes = False  # Hidden by default - toggle with 'A' key
        self.show_gizmo = True  # Corner orientation gizmo

    def _init_body_buffers(self) -> None:
        """Create GPU buffers for body rendering."""
        # Pre-allocate for max bodies
        self.body_pos_vbo = self.ctx.buffer(reserve=self.max_bodies * 3 * 4)
        self.body_color_vbo = self.ctx.buffer(reserve=self.max_bodies * 3 * 4)
        self.body_size_vbo = self.ctx.buffer(reserve=self.max_bodies * 4)

        self.body_vao = self.ctx.vertex_array(
            self.body_prog,
            [
                (self.body_pos_vbo, "3f", "in_position"),
                (self.body_color_vbo, "3f", "in_color"),
                (self.body_size_vbo, "f", "in_size"),
            ],
        )

    def _init_grid_buffers(self) -> None:
        """Create grid geometry buffers (will be updated dynamically)."""
        # Pre-allocate buffer for adaptive grid (max ~200 lines = 400 verts)
        self.grid_vbo = self.ctx.buffer(reserve=400 * 3 * 4)
        self.grid_vao = self.ctx.vertex_array(
            self.grid_prog,
            [(self.grid_vbo, "3f", "in_position")],
        )
        self.grid_vertex_count = 0
        self.grid_extent = 1.0
        self._last_grid_scale = -1.0

    def _update_grid(self, camera_distance: float) -> None:
        """Update grid to appropriate scale for current zoom level."""
        # Calculate grid spacing based on camera distance
        # We want ~10-20 grid lines visible, so spacing ~ distance / 10
        raw_spacing = camera_distance / 8.0

        # Snap to nice round numbers (1, 2, 5, 10, 20, 50, etc.)
        magnitude = 10 ** np.floor(np.log10(raw_spacing))
        normalized = raw_spacing / magnitude
        if normalized < 1.5:
            nice_spacing = 1.0
        elif normalized < 3.5:
            nice_spacing = 2.0
        elif normalized < 7.5:
            nice_spacing = 5.0
        else:
            nice_spacing = 10.0
        grid_step = nice_spacing * magnitude

        # Skip update if scale hasn't changed significantly
        if abs(grid_step - self._last_grid_scale) / max(grid_step, 0.001) < 0.01:
            return
        self._last_grid_scale = grid_step

        # Generate grid lines on XY plane (z=0) - the orbital plane
        lines = []
        grid_count = 10  # Lines in each direction from center
        extent = grid_count * grid_step

        # Grid lines on XY plane (z=0)
        for i in range(-grid_count, grid_count + 1):
            pos = i * grid_step
            # Lines parallel to Y axis (constant X)
            lines.extend([pos, -extent, 0])
            lines.extend([pos, extent, 0])
            # Lines parallel to X axis (constant Y)
            lines.extend([-extent, pos, 0])
            lines.extend([extent, pos, 0])

        vertices = np.array(lines, dtype=np.float32)
        self.grid_vbo.write(vertices.tobytes())
        self.grid_vertex_count = len(lines) // 3
        self.grid_extent = extent

    def _init_axis_buffers(self) -> None:
        """Create axis indicator geometry (updated dynamically with grid)."""
        self.axis_vbo = self.ctx.buffer(reserve=6 * 6 * 4)  # 6 verts, 6 floats each
        self.axis_vao = self.ctx.vertex_array(
            self.axis_prog,
            [(self.axis_vbo, "3f 3f", "in_position", "in_color")],
        )
        self._update_axes(1.0)

    def _update_axes(self, scale: float) -> None:
        """Update axis position and length to match grid scale."""
        axis_len = scale * 2.0  # Axes span 2 grid units
        # Position axes at corner of grid on XY plane
        corner = -scale * 8.0  # Offset to grid corner
        vertices = np.array(
            [
                # X axis (red) - horizontal right
                corner,
                corner,
                0,
                1,
                0.4,
                0.4,
                corner + axis_len,
                corner,
                0,
                1,
                0.4,
                0.4,
                # Y axis (green) - horizontal up in orbital plane
                corner,
                corner,
                0,
                0.4,
                1,
                0.4,
                corner,
                corner + axis_len,
                0,
                0.4,
                1,
                0.4,
                # Z axis (blue) - vertical (out of plane)
                corner,
                corner,
                0,
                0.4,
                0.4,
                1,
                corner,
                corner,
                axis_len,
                0.4,
                0.4,
                1,
            ],
            dtype=np.float32,
        )
        self.axis_vbo.write(vertices.tobytes())

    def _init_gizmo_buffers(self) -> None:
        """Create corner orientation gizmo (XYZ axes indicator)."""
        # Simple XYZ axis lines from origin - each axis is a line with color
        vertices = np.array(
            [
                # X axis (red)
                0.0,
                0.0,
                0.0,
                1.0,
                0.3,
                0.3,
                1.0,
                0.0,
                0.0,
                1.0,
                0.3,
                0.3,
                # Y axis (green)
                0.0,
                0.0,
                0.0,
                0.3,
                1.0,
                0.3,
                0.0,
                1.0,
                0.0,
                0.3,
                1.0,
                0.3,
                # Z axis (blue)
                0.0,
                0.0,
                0.0,
                0.3,
                0.5,
                1.0,
                0.0,
                0.0,
                1.0,
                0.3,
                0.5,
                1.0,
            ],
            dtype=np.float32,
        )
        self.gizmo_vbo = self.ctx.buffer(vertices.tobytes())
        self.gizmo_vao = self.ctx.vertex_array(
            self.gizmo_prog,
            [(self.gizmo_vbo, "3f 3f", "in_position", "in_color")],
        )

    def update_bodies(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        scale: float = 1.0,
        selected: int | None = None,
    ) -> None:
        """Update body positions and sizes on GPU.

        Args:
            positions: (n, 3) array of XYZ positions
            masses: (n,) array of masses
            scale: Scale factor for visualization
            selected: Index of selected body (highlighted)
        """
        n = len(positions)
        if n == 0:
            return

        # Scale positions for visualization
        scaled_pos = (positions * scale).astype(np.float32)
        self.body_pos_vbo.write(scaled_pos.tobytes())

        # Colors (cycle through palette, brighten selected)
        colors = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            base_color = BODY_COLORS[i % len(BODY_COLORS)]
            if i == selected:
                # Brighten selected body
                colors[i] = np.clip(np.array(base_color) * 1.5, 0.0, 1.0)
            else:
                colors[i] = base_color
        self.body_color_vbo.write(colors.tobytes())

        # Sizes based on log(mass) - normalized, view-relative (not world-scaled)
        log_masses = np.log10(np.maximum(masses, 1.0))
        min_log = log_masses.min()
        max_log = log_masses.max()
        if max_log > min_log:
            normalized = (log_masses - min_log) / (max_log - min_log)
        else:
            normalized = np.ones_like(log_masses) * 0.5

        # Map to visual size range (0.1 to 0.4 in normalized view space)
        sizes = (0.1 + normalized * 0.3).astype(np.float32)
        # Make selected body slightly larger
        if selected is not None and selected < n:
            sizes[selected] *= 1.2
        self.body_size_vbo.write(sizes.tobytes())

        self.body_count = n

    def update_trails(
        self,
        trail_buffer: TrailBuffer,
        num_bodies: int,
        scale: float = 1.0,
    ) -> None:
        """Update trail geometry on GPU."""
        vertices, ages = trail_buffer.get_all_trails_interleaved(num_bodies)

        if len(vertices) == 0:
            self.trail_vertex_count = 0
            return

        # Scale positions
        vertices = (vertices * scale).astype(np.float32)

        # Create colors matching bodies
        points_per_body = len(ages) // num_bodies if num_bodies > 0 else 0
        colors = np.zeros((len(vertices), 3), dtype=np.float32)
        for b in range(num_bodies):
            start = b * points_per_body
            end = start + points_per_body
            colors[start:end] = BODY_COLORS[b % len(BODY_COLORS)]

        # Recreate buffers if needed (or write)
        byte_size = vertices.nbytes
        if self.trail_vbo is None or self.trail_vbo.size < byte_size:
            if self.trail_vbo:
                self.trail_vbo.release()
                self.trail_age_vbo.release()
                self.trail_color_vbo.release()

            self.trail_vbo = self.ctx.buffer(vertices.tobytes())
            self.trail_age_vbo = self.ctx.buffer(ages.tobytes())
            self.trail_color_vbo = self.ctx.buffer(colors.tobytes())

            self.trail_vao = self.ctx.vertex_array(
                self.trail_prog,
                [
                    (self.trail_vbo, "3f", "in_position"),
                    (self.trail_age_vbo, "f", "in_age"),
                    (self.trail_color_vbo, "3f", "in_color"),
                ],
            )
        else:
            self.trail_vbo.write(vertices.tobytes())
            self.trail_age_vbo.write(ages.tobytes())
            self.trail_color_vbo.write(colors.tobytes())

        self.trail_vertex_count = len(vertices)
        self.trail_points_per_body = points_per_body
        self.trail_num_bodies = num_bodies

    def render(self, camera: Camera) -> None:
        """Render frame."""
        self.ctx.clear(*BACKGROUND_COLOR)

        view_proj = camera.view_projection_matrix()

        # Update adaptive grid and axes based on zoom
        old_scale = self._last_grid_scale
        self._update_grid(camera.distance)
        if self._last_grid_scale != old_scale:
            self._update_axes(self._last_grid_scale)

        # Grid (render first, behind everything)
        if self.show_grid and self.grid_vertex_count > 0:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.grid_prog["view_proj"].write(view_proj.tobytes())
            self.grid_prog["grid_color"].value = GRID_COLOR
            self.grid_vao.render(moderngl.LINES, vertices=self.grid_vertex_count)
            self.ctx.enable(moderngl.DEPTH_TEST)

        # Axes
        if self.show_axes:
            self.axis_prog["view_proj"].write(view_proj.tobytes())
            self.axis_vao.render(moderngl.LINES, vertices=6)

        # Trails
        if hasattr(self, "trail_vertex_count") and self.trail_vertex_count > 0:
            self.trail_prog["view_proj"].write(view_proj.tobytes())
            # Render each body's trail as separate line strip
            for b in range(self.trail_num_bodies):
                start = b * self.trail_points_per_body
                self.trail_vao.render(
                    moderngl.LINE_STRIP,
                    first=start,
                    vertices=self.trail_points_per_body,
                )

        # Bodies
        if hasattr(self, "body_count") and self.body_count > 0:
            self.body_prog["view_proj"].write(view_proj.tobytes())
            self.body_prog["viewport_height"].value = float(self.ctx.viewport[3])
            self.body_vao.render(moderngl.POINTS, vertices=self.body_count)

        # Corner orientation gizmo (rendered last, on top, no depth test)
        if self.show_gizmo:
            self.ctx.disable(moderngl.DEPTH_TEST)
            rotation = camera.rotation_matrix()
            self.gizmo_prog["rotation"].write(rotation.tobytes())
            # Position in bottom-left corner (NDC: -1 to 1)
            self.gizmo_prog["offset"].value = (-0.8, -0.75)
            self.gizmo_prog["scale"].value = 0.12
            self.gizmo_vao.render(moderngl.LINES, vertices=6)
            self.ctx.enable(moderngl.DEPTH_TEST)

    def resize(self, width: int, height: int) -> None:
        """Handle window resize."""
        self.ctx.viewport = (0, 0, width, height)

    def release(self) -> None:
        """Release GPU resources."""
        self.body_vao.release()
        self.body_pos_vbo.release()
        self.body_color_vbo.release()
        self.body_size_vbo.release()
        self.grid_vao.release()
        self.grid_vbo.release()
        self.axis_vao.release()
        self.axis_vbo.release()
        self.gizmo_vao.release()
        self.gizmo_vbo.release()
        if self.trail_vbo:
            self.trail_vbo.release()
            self.trail_age_vbo.release()
            self.trail_color_vbo.release()
