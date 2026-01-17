"""
D3X orbital mechanics simulation core
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

from . import constants

__all__ = [
    "StepResult",
    "Vec3",
    "Vec3Like",
    "World",
    "compute_gravity",
    "constants",
    "step_dopri54",
    "step_leapfrog",
    "step_rk4",
]

class StepResult:
    def __repr__(self) -> str: ...
    @property
    def dt_next(self) -> float: ...
    @property
    def dt_used(self) -> float: ...
    @property
    def error_estimate(self) -> float: ...

class Vec3:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, x: typing.SupportsFloat, y: typing.SupportsFloat, z: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(self, arg0: tuple) -> None: ...
    def __repr__(self) -> str: ...
    def magnitude(self) -> float: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def z(self) -> float: ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat) -> None: ...

# Type alias for Vec3-like inputs (Vec3 or tuple of 3 floats)
# Reflects pybind11's implicit conversion from tuple to Vec3
Vec3Like = Vec3 | tuple[float, float, float]

class World:
    def __init__(self) -> None: ...
    def add_body(self, pos: Vec3Like, vel: Vec3Like, mass: typing.SupportsFloat) -> int:
        """
        Add a body with position [m], velocity [m/s], and mass [kg]. Returns body index.
        """
    def angular_momentum(self) -> Vec3:
        """
        Total angular momentum vector [kg·m²/s]
        """
    def clear(self) -> None:
        """
        Remove all bodies and reset time
        """
    def kinetic_energy(self) -> float:
        """
        Total kinetic energy [J]
        """
    def potential_energy(self) -> float:
        """
        Total gravitational potential energy [J]
        """
    def reserve(self, n: typing.SupportsInt) -> None:
        """
        Pre-allocate memory for n bodies
        """
    def total_energy(self) -> float:
        """
        Total mechanical energy [J]
        """
    @property
    def count(self) -> int:
        """
        Number of bodies in the simulation
        """
    @property
    def mass(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Masses [kg] (numpy view)
        """
    @property
    def px(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Position x-components [m] (numpy view)
        """
    @property
    def py_(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Position y-components [m] (numpy view, named py_ to avoid collision)
        """
    @property
    def pz(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Position z-components [m] (numpy view)
        """
    @property
    def time(self) -> float:
        """
        Current simulation time [s]
        """
    @time.setter
    def time(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def vx(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Velocity x-components [m/s] (numpy view)
        """
    @property
    def vy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Velocity y-components [m/s] (numpy view)
        """
    @property
    def vz(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Velocity z-components [m/s] (numpy view)
        """

@typing.overload
def compute_gravity(world: World) -> None:
    """
    Compute gravitational accelerations for all bodies
    """

@typing.overload
def compute_gravity(world: World, softening: typing.SupportsFloat) -> None:
    """
    Compute gravitational accelerations with softening parameter
    """

def step_dopri54(
    world: World, dt: typing.SupportsFloat, tol: typing.SupportsFloat = 1e-09
) -> StepResult:
    """
    Advance simulation using adaptive Dormand-Prince 5(4) method
    """

def step_leapfrog(world: World, dt: typing.SupportsFloat) -> None:
    """
    Advance simulation using symplectic leapfrog (requires pre-computed accelerations)
    """

def step_rk4(world: World, dt: typing.SupportsFloat) -> None:
    """
    Advance simulation by dt seconds using 4th-order Runge-Kutta
    """
