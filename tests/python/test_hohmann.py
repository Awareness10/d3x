"""Hohmann transfer validation tests."""

import numpy as np
import pytest

import d3x


def test_hohmann_transfer_geometry():
    """
    Validate Hohmann transfer reaches target orbit.

    A Hohmann transfer from circular orbit r1 to r2 involves:
    1. Burn at periapsis to enter elliptical transfer orbit
    2. Coast along transfer orbit (half period)
    3. Burn at apoapsis to circularize (not simulated here)

    We verify the spacecraft reaches the target orbital radius.
    """
    world = d3x.World()

    # Central body (Sun-like)
    M = d3x.constants.M_SUN
    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)

    # Starting orbit (Earth-like, 1 AU)
    r1 = d3x.constants.AU
    v1 = np.sqrt(d3x.constants.MU_SUN / r1)  # Circular velocity

    # Target orbit (Mars-like, 1.524 AU)
    r2 = 1.524 * d3x.constants.AU

    # Hohmann transfer velocity at periapsis
    # v_transfer = sqrt(mu * (2/r1 - 2/(r1+r2)))
    a_transfer = (r1 + r2) / 2  # Semi-major axis of transfer orbit
    v_transfer = np.sqrt(d3x.constants.MU_SUN * (2 / r1 - 1 / a_transfer))
    delta_v = v_transfer - v1

    # Spacecraft with transfer velocity
    world.add_body(
        pos=(r1, 0.0, 0.0),
        vel=(0.0, v1 + delta_v, 0.0),
        mass=1000.0,  # 1 ton spacecraft
    )

    # Transfer time (half the transfer orbit period)
    # T_transfer = pi * sqrt(a³/mu)
    transfer_time = np.pi * np.sqrt(a_transfer**3 / d3x.constants.MU_SUN)

    # Simulate the transfer
    dt = transfer_time / 1000  # 1000 steps
    for _ in range(1000):
        d3x.step_rk4(world, dt)

    # Check arrival distance
    final_x = world.px[1]
    final_y = world.py_[1]
    final_r = np.sqrt(final_x**2 + final_y**2)

    # Should be at Mars orbit (within 0.5%)
    relative_error = abs(final_r - r2) / r2
    assert relative_error < 0.005, f"Final radius {final_r:.3e} vs expected {r2:.3e}"


def test_hohmann_transfer_time():
    """Validate Hohmann transfer takes the expected time."""
    world = d3x.World()

    M = 1e20  # Arbitrary central mass for faster simulation
    mu = d3x.constants.G * M

    r1 = 1e6  # 1000 km starting orbit
    r2 = 2e6  # 2000 km target orbit

    a_transfer = (r1 + r2) / 2
    v_transfer = np.sqrt(mu * (2 / r1 - 1 / a_transfer))

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world.add_body(pos=(r1, 0.0, 0.0), vel=(0.0, v_transfer, 0.0), mass=1.0)

    # Expected transfer time
    expected_time = np.pi * np.sqrt(a_transfer**3 / mu)

    # Simulate with adaptive integrator for accuracy
    dt = expected_time / 100
    total_time = 0.0

    while total_time < expected_time:
        result = d3x.step_dopri54(world, dt, tol=1e-10)
        if result.dt_used > 0:
            total_time += result.dt_used
        dt = min(result.dt_next, expected_time - total_time + 1)

    # Check we're at apoapsis (maximum distance)
    final_r = np.sqrt(world.px[1] ** 2 + world.py_[1] ** 2)

    # At apoapsis, radial velocity should be ~0
    radial_vel = (world.px[1] * world.vx[1] + world.py_[1] * world.vy[1]) / final_r
    assert abs(radial_vel) < 1.0, f"Radial velocity at apoapsis: {radial_vel}"

    # Should be at target radius
    assert final_r == pytest.approx(r2, rel=0.01)


def test_orbital_energy_in_transfer():
    """Verify orbital energy is conserved during coast phase."""
    world = d3x.World()

    M = 1e20
    mu = d3x.constants.G * M

    r1 = 1e6
    r2 = 2e6

    a_transfer = (r1 + r2) / 2
    v_transfer = np.sqrt(mu * (2 / r1 - 1 / a_transfer))

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world.add_body(pos=(r1, 0.0, 0.0), vel=(0.0, v_transfer, 0.0), mass=1.0)

    initial_energy = world.total_energy()

    # Simulate transfer
    transfer_time = np.pi * np.sqrt(a_transfer**3 / mu)
    dt = transfer_time / 500

    for _ in range(500):
        d3x.step_rk4(world, dt)

    final_energy = world.total_energy()

    # Energy should be conserved to high precision
    relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
    assert relative_error < 1e-6, f"Energy drift: {relative_error:.2e}"
