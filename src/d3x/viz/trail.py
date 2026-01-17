"""SoA ring buffer for position history trails."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from d3x import World


class TrailBuffer:
    """Fixed-size ring buffer storing position history for all bodies.

    SoA layout for efficient GPU upload - positions stored as separate
    x, y, z arrays with shape (max_bodies, trail_length).
    """

    __slots__ = ("x", "y", "z", "head", "length", "max_bodies", "filled")

    def __init__(self, max_bodies: int, trail_length: int):
        # SoA layout - contiguous per component
        self.x = np.zeros((max_bodies, trail_length), dtype=np.float32)
        self.y = np.zeros((max_bodies, trail_length), dtype=np.float32)
        self.z = np.zeros((max_bodies, trail_length), dtype=np.float32)

        self.head = 0
        self.length = trail_length
        self.max_bodies = max_bodies
        self.filled = 0  # How many slots have been written

    def push(self, world: "World") -> None:
        """Record current positions from World into ring buffer."""
        n = min(world.count, self.max_bodies)

        # Direct copy from World's SoA arrays (numpy views, no allocation)
        self.x[:n, self.head] = world.px[:n]
        self.y[:n, self.head] = world.py_[:n]
        self.z[:n, self.head] = world.pz[:n]

        self.head = (self.head + 1) % self.length
        self.filled = min(self.filled + 1, self.length)

    def clear(self) -> None:
        """Clear all trail data."""
        self.x.fill(0.0)
        self.y.fill(0.0)
        self.z.fill(0.0)
        self.head = 0
        self.filled = 0

    def get_trail_vertices(self, body_idx: int, count: int) -> np.ndarray:
        """Get trail vertices for a single body as interleaved XYZ.

        Returns positions from oldest to newest, shape (n, 3).
        """
        n = min(count, self.filled, self.length)
        if n == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Indices from oldest to newest
        if self.filled < self.length:
            indices = np.arange(n)
        else:
            start = (self.head - n) % self.length
            if start + n <= self.length:
                indices = np.arange(start, start + n)
            else:
                indices = np.concatenate(
                    [np.arange(start, self.length), np.arange(0, (start + n) % self.length)]
                )

        # Interleave for GPU vertex attribute
        vertices = np.column_stack(
            [self.x[body_idx, indices], self.y[body_idx, indices], self.z[body_idx, indices]]
        )
        return vertices

    def get_all_trails_interleaved(self, num_bodies: int) -> tuple[np.ndarray, np.ndarray]:
        """Get all trail vertices and ages for GPU upload.

        Returns:
            vertices: shape (num_bodies * filled, 3) - XYZ positions
            ages: shape (num_bodies * filled,) - age values 0.0 (newest) to 1.0 (oldest)
        """
        n = min(self.filled, self.length)
        if n == 0 or num_bodies == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)

        num_bodies = min(num_bodies, self.max_bodies)

        # Calculate ordered indices (oldest to newest)
        if self.filled < self.length:
            indices = np.arange(n)
        else:
            start = self.head  # Oldest is at head (about to be overwritten)
            indices = np.roll(np.arange(self.length), -start)

        # Pre-allocate output
        total_verts = num_bodies * n
        vertices = np.empty((total_verts, 3), dtype=np.float32)
        ages = np.empty(total_verts, dtype=np.float32)

        # Age array (same for all bodies)
        body_ages = np.linspace(1.0, 0.0, n, dtype=np.float32)

        # Fill per-body (contiguous in output for line strip rendering)
        for b in range(num_bodies):
            start_idx = b * n
            end_idx = start_idx + n
            vertices[start_idx:end_idx, 0] = self.x[b, indices]
            vertices[start_idx:end_idx, 1] = self.y[b, indices]
            vertices[start_idx:end_idx, 2] = self.z[b, indices]
            ages[start_idx:end_idx] = body_ages

        return vertices, ages
