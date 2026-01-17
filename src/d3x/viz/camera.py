"""Arcball orbit camera for 3D visualization."""

import glm
import numpy as np


class Camera:
    """Orbit camera with spherical coordinates around a target point."""

    __slots__ = (
        "target",
        "distance",
        "yaw",
        "pitch",
        "fov",
        "near",
        "far",
        "aspect",
        "_smoothing",
        "_target_distance",
        "_target_yaw",
        "_target_pitch",
    )

    def __init__(
        self,
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance: float = 12.0,
        yaw: float = 0.3,
        pitch: float = 1.0,  # ~57 degrees - looking down at orbital plane
        fov: float = 45.0,
    ):
        self.target = np.array(target, dtype=np.float32)
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.fov = fov
        self.near = 0.01
        self.far = 100000.0
        self.aspect = 16.0 / 9.0

        # Smoothing targets
        self._smoothing = 0.35  # Responsive feel
        self._target_distance = distance
        self._target_yaw = yaw
        self._target_pitch = pitch

    def orbit(self, dx: float, dy: float) -> None:
        """Rotate camera around target (mouse drag)."""
        sensitivity = 0.005
        self._target_yaw -= dx * sensitivity
        self._target_pitch -= dy * sensitivity  # Natural: drag up = look up
        # Allow full 360° but avoid exact poles (gimbal lock)
        self._target_pitch = np.clip(self._target_pitch, -1.55, 1.55)  # ~±89°

    def zoom(self, delta: float) -> None:
        """Zoom in/out (scroll wheel)."""
        # Multiplicative zoom for consistent feel at all scales
        factor = 1.0 - delta * 0.15
        self._target_distance *= factor
        self._target_distance = np.clip(self._target_distance, 0.5, 1000.0)

    def pan(self, dx: float, dy: float) -> None:
        """Pan target in view plane (middle mouse or ctrl+left drag)."""
        # Scale sensitivity with distance for consistent feel
        sensitivity = self.distance * 0.0015

        # Calculate right vector in world space (perpendicular to view)
        right = np.array([np.cos(self.yaw), 0.0, -np.sin(self.yaw)], dtype=np.float32)

        # Up vector considers pitch for proper view-plane panning
        forward = np.array(
            [
                -np.cos(self.pitch) * np.sin(self.yaw),
                -np.sin(self.pitch),
                -np.cos(self.pitch) * np.cos(self.yaw),
            ],
            dtype=np.float32,
        )
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        self.target -= right * dx * sensitivity
        self.target -= up * dy * sensitivity  # Natural: drag up = pan up

    def focus(self, position: np.ndarray, distance: float | None = None) -> None:
        """Move camera to focus on a point."""
        self.target = np.array(position, dtype=np.float32)
        if distance is not None:
            self._target_distance = distance

    def reset(self) -> None:
        """Reset to default view."""
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._target_distance = 12.0
        self._target_yaw = 0.3
        self._target_pitch = 1.0

    def update(self) -> None:
        """Smooth interpolation of camera parameters."""
        self.distance += (self._target_distance - self.distance) * self._smoothing
        self.yaw += (self._target_yaw - self.yaw) * self._smoothing
        self.pitch += (self._target_pitch - self.pitch) * self._smoothing

    @property
    def position(self) -> np.ndarray:
        """Camera position in world space."""
        x = self.distance * np.cos(self.pitch) * np.sin(self.yaw)
        y = self.distance * np.sin(self.pitch)
        z = self.distance * np.cos(self.pitch) * np.cos(self.yaw)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        """Compute view matrix."""
        eye = self.position
        target = glm.vec3(*self.target)
        up = glm.vec3(0.0, 1.0, 0.0)
        view = glm.lookAt(glm.vec3(*eye), target, up)
        return np.array(view, dtype=np.float32)

    def projection_matrix(self) -> np.ndarray:
        """Compute perspective projection matrix."""
        proj = glm.perspective(glm.radians(self.fov), self.aspect, self.near, self.far)
        return np.array(proj, dtype=np.float32)

    def view_projection_matrix(self) -> np.ndarray:
        """Combined view-projection matrix."""
        proj = glm.perspective(glm.radians(self.fov), self.aspect, self.near, self.far)
        eye = self.position
        view = glm.lookAt(glm.vec3(*eye), glm.vec3(*self.target), glm.vec3(0.0, 1.0, 0.0))
        return np.array(proj * view, dtype=np.float32)

    def rotation_matrix(self) -> np.ndarray:
        """Rotation-only view matrix (for orientation gizmo)."""
        # Camera looks from position toward target - extract rotation only
        eye = glm.vec3(
            np.cos(self.pitch) * np.sin(self.yaw),
            np.sin(self.pitch),
            np.cos(self.pitch) * np.cos(self.yaw),
        )
        view = glm.lookAt(eye, glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
        return np.array(view, dtype=np.float32)
