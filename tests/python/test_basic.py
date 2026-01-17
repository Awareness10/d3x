"""Basic functionality tests for d3x."""

import numpy as np
import pytest

import d3x


def test_world_creation():
    """Test World can be created and has correct initial state."""
    world = d3x.World()
    assert world.count == 0
    assert world.time == 0.0


def test_add_body():
    """Test adding bodies to the world."""
    world = d3x.World()

    idx = world.add_body(pos=(1.0, 2.0, 3.0), vel=(4.0, 5.0, 6.0), mass=100.0)

    assert idx == 0
    assert world.count == 1
    assert world.px[0] == pytest.approx(1.0)
    assert world.py_[0] == pytest.approx(2.0)
    assert world.pz[0] == pytest.approx(3.0)
    assert world.mass[0] == pytest.approx(100.0)


def test_numpy_views():
    """Test that position/velocity arrays are numpy views (zero-copy)."""
    world = d3x.World()
    world.add_body(pos=(1.0, 0.0, 0.0), vel=(0.0, 1.0, 0.0), mass=1.0)
    world.add_body(pos=(2.0, 0.0, 0.0), vel=(0.0, 2.0, 0.0), mass=2.0)

    px = world.px
    assert isinstance(px, np.ndarray)
    assert px.dtype == np.float64
    assert len(px) == 2
    assert px[0] == pytest.approx(1.0)
    assert px[1] == pytest.approx(2.0)


def test_constants():
    """Test that physical constants are exposed correctly."""
    assert d3x.constants.G == pytest.approx(6.67430e-11)
    assert d3x.constants.AU == pytest.approx(1.495978707e11)
    assert d3x.constants.M_SUN == pytest.approx(1.98892e30)
    assert d3x.constants.M_EARTH == pytest.approx(5.97217e24)


def test_rk4_integration():
    """Test RK4 integrator runs without error."""
    world = d3x.World()

    # Sun and Earth
    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=d3x.constants.M_SUN)
    world.add_body(
        pos=(d3x.constants.AU, 0.0, 0.0),
        vel=(0.0, 29780.0, 0.0),  # ~Earth orbital velocity
        mass=d3x.constants.M_EARTH,
    )

    initial_energy = world.total_energy()

    # Simulate 1 day
    dt = 3600.0  # 1 hour
    for _ in range(24):
        d3x.step_rk4(world, dt)

    assert world.time == pytest.approx(24 * 3600.0)

    # Energy should be approximately conserved
    final_energy = world.total_energy()
    relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
    assert relative_error < 1e-6


def test_dopri54_adaptive():
    """Test adaptive integrator adjusts step size."""
    world = d3x.World()

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1e12)
    world.add_body(pos=(1000.0, 0.0, 0.0), vel=(0.0, 300.0, 0.0), mass=1.0)

    dt = 1.0
    total_time = 0.0
    step_sizes = []

    while total_time < 100.0:
        result = d3x.step_dopri54(world, dt, tol=1e-8)

        if result.dt_used > 0:
            total_time += result.dt_used
            step_sizes.append(result.dt_used)

        dt = result.dt_next

    # Should have taken multiple steps
    assert len(step_sizes) > 1
    # Step sizes should vary
    assert max(step_sizes) != min(step_sizes)
