"""D3X - Data-oriented orbital mechanics simulation framework."""

from typing import TYPE_CHECKING

from d3x._core import (
    StepResult,
    # Types
    Vec3,
    World,
    # Gravity
    compute_gravity,
    # Constants
    constants,
    step_dopri54,
    step_leapfrog,
    # Integrators
    step_rk4,
)

if TYPE_CHECKING:
    from d3x._core import Vec3Like

__version__ = "0.1.0"

__all__ = [
    "Vec3",
    "Vec3Like",
    "StepResult",
    "World",
    "constants",
    "compute_gravity",
    "step_rk4",
    "step_dopri54",
    "step_leapfrog",
]
