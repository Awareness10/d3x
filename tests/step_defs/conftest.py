"""Shared fixtures and configuration for BDD tests."""

import numpy as np
import pytest

import d3x


@pytest.fixture
def world():
    """Create a fresh World instance."""
    return d3x.World()


@pytest.fixture
def constants():
    """Access to physical constants."""
    return d3x.constants


@pytest.fixture
def earth_moon_system(world):
    """
    Earth-Moon system for orbital tests.

    Returns world with Earth at origin and Moon in circular orbit.
    """
    # Earth at origin
    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=d3x.constants.M_EARTH)

    # Moon in circular orbit
    r_moon = 384400e3  # 384,400 km
    v_moon = np.sqrt(d3x.constants.G * d3x.constants.M_EARTH / r_moon)
    world.add_body(pos=(r_moon, 0.0, 0.0), vel=(0.0, v_moon, 0.0), mass=d3x.constants.M_MOON)

    return world


@pytest.fixture
def sun_earth_system(world):
    """
    Sun-Earth system for orbital tests.

    Returns world with Sun at origin and Earth in circular orbit.
    """
    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=d3x.constants.M_SUN)

    r_earth = d3x.constants.AU
    v_earth = np.sqrt(d3x.constants.MU_SUN / r_earth)
    world.add_body(pos=(r_earth, 0.0, 0.0), vel=(0.0, v_earth, 0.0), mass=d3x.constants.M_EARTH)

    return world


def orbital_period(r, M):
    """Calculate orbital period for circular orbit."""
    return 2 * np.pi * np.sqrt(r**3 / (d3x.constants.G * M))
