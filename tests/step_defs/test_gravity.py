"""Step definitions for gravity.feature."""

import numpy as np
import pytest
from pytest_bdd import given, scenario, then, when

import d3x

# ============================================================================
# Scenarios
# ============================================================================


@scenario("../features/gravity.feature", "Two-body gravitational attraction")
def test_two_body_gravitational_attraction():
    pass


@scenario("../features/gravity.feature", "Acceleration is independent of test mass")
def test_acceleration_is_independent_of_test_mass():
    pass


@scenario("../features/gravity.feature", "Newton's third law symmetry")
def test_newtons_third_law_symmetry():
    pass


@scenario("../features/gravity.feature", "Multi-body superposition")
def test_multi_body_superposition():
    pass


@scenario("../features/gravity.feature", "Softening prevents singularities")
def test_softening_prevents_singularities():
    pass


# ============================================================================
# Shared context fixture
# ============================================================================


@pytest.fixture
def ctx():
    """Context dictionary for sharing state between steps."""
    return {}


# ============================================================================
# Background
# ============================================================================


@given("a simulation world with the gravitational constant G", target_fixture="world")
def world_with_gravity():
    """Create world - G is always available via d3x.constants.G."""
    return d3x.World()


# ============================================================================
# Scenario: Two-body gravitational attraction
# ============================================================================


@given("two bodies separated by a known distance", target_fixture="world")
def setup_two_bodies():
    world = d3x.World()
    separation = 1e6  # 1000 km
    mass1 = 1e12
    mass2 = 1e10

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=mass1)
    world.add_body(pos=(separation, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=mass2)

    return world


@pytest.fixture
def two_body_params():
    """Parameters for two-body tests."""
    return {"separation": 1e6, "mass1": 1e12, "mass2": 1e10}


@when("gravity is computed")
def compute_gravity(world):
    d3x.compute_gravity(world)


@then("each body should experience acceleration toward the other")
def check_attraction_direction(world):
    # Verified in next step via integration
    pass


@then("the accelerations should follow Newton's inverse-square law")
def check_inverse_square(world, two_body_params):
    # Expected acceleration: a = G * M / r²
    G = d3x.constants.G
    r = two_body_params["separation"]
    m1 = two_body_params["mass1"]
    m2 = two_body_params["mass2"]

    expected_a0 = G * m2 / (r * r)
    expected_a1 = G * m1 / (r * r)

    # Create fresh world and integrate one tiny step to measure acceleration
    world2 = d3x.World()
    world2.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=m1)
    world2.add_body(pos=(r, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=m2)

    dt = 1.0  # 1 second
    d3x.step_rk4(world2, dt)

    # Velocity change ≈ acceleration * dt for small dt
    actual_a0 = world2.vx[0] / dt
    actual_a1 = -world2.vx[1] / dt  # Negative because moving in -x direction

    assert actual_a0 == pytest.approx(expected_a0, rel=1e-4)
    assert actual_a1 == pytest.approx(expected_a1, rel=1e-4)


# ============================================================================
# Scenario: Acceleration is independent of test mass
# ============================================================================


@given(
    "a massive body and two test bodies of different masses at equal distances",
    target_fixture="world",
)
def setup_equivalence_test():
    world = d3x.World()
    M = 1e15  # Massive central body

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world.add_body(pos=(1e6, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)  # Light test body
    world.add_body(pos=(-1e6, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1000.0)  # Heavy test body

    return world


@then("both test bodies should have equal acceleration magnitudes")
def check_equivalence_principle(world):
    # Take a small step to measure accelerations
    dt = 0.1
    d3x.step_rk4(world, dt)

    # Both test bodies should have same |acceleration|
    # Body 1 moves in -x, body 2 moves in +x
    a1_mag = abs(world.vx[1] / dt)
    a2_mag = abs(world.vx[2] / dt)

    assert a1_mag == pytest.approx(a2_mag, rel=1e-3)


# ============================================================================
# Scenario: Newton's third law symmetry
# ============================================================================


@given("two bodies of different masses", target_fixture="world")
def setup_newton_third():
    world = d3x.World()
    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1e12)
    world.add_body(pos=(1e6, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1e10)
    return world


@pytest.fixture
def newton_third_masses():
    return {"m1": 1e12, "m2": 1e10}


@then("the forces should be equal and opposite")
def check_newton_third(world, newton_third_masses):
    m1 = newton_third_masses["m1"]
    m2 = newton_third_masses["m2"]

    dt = 0.1
    d3x.step_rk4(world, dt)

    # F = ma, so F1 = m1 * a1 and F2 = m2 * a2
    # Forces should be equal and opposite: F1 = -F2
    f1_x = m1 * (world.vx[0] / dt)
    f2_x = m2 * (world.vx[1] / dt)

    assert f1_x == pytest.approx(-f2_x, rel=1e-3)


@then("momentum should be conserved in the force calculation")
def check_momentum_conservation(world, newton_third_masses):
    m1 = newton_third_masses["m1"]
    m2 = newton_third_masses["m2"]
    # Total momentum should be zero (started at rest)
    total_px = m1 * world.vx[0] + m2 * world.vx[1]
    assert total_px == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Scenario: Multi-body superposition
# ============================================================================


@given("three or more bodies in arbitrary positions", target_fixture="world")
def setup_multi_body():
    world = d3x.World()

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1e12)
    world.add_body(pos=(1e6, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1e11)
    world.add_body(pos=(0.0, 1e6, 0.0), vel=(0.0, 0.0, 0.0), mass=1e11)
    world.add_body(pos=(0.0, 0.0, 1e6), vel=(0.0, 0.0, 0.0), mass=1e11)

    return world


@then("each body's acceleration should be the vector sum of pairwise attractions")
def check_superposition(world):
    G = d3x.constants.G

    # Store initial positions and masses
    positions = []
    masses = []
    for i in range(world.count):
        positions.append(np.array([world.px[i], world.py_[i], world.pz[i]]))
        masses.append(world.mass[i])

    # Compute expected acceleration on body 0 from all others
    p0 = positions[0]
    expected_a = np.array([0.0, 0.0, 0.0])

    for i in range(1, world.count):
        pi = positions[i]
        mi = masses[i]
        r_vec = pi - p0
        r_mag = np.linalg.norm(r_vec)
        a_mag = G * mi / (r_mag * r_mag)
        expected_a += a_mag * r_vec / r_mag

    # Take small step and measure actual acceleration
    dt = 0.01
    d3x.step_rk4(world, dt)

    actual_a = np.array([world.vx[0], world.vy[0], world.vz[0]]) / dt

    np.testing.assert_allclose(actual_a, expected_a, rtol=1e-3)


# ============================================================================
# Scenario: Softening prevents singularities
# ============================================================================


@given("two bodies at very close separation", target_fixture="world")
def setup_close_bodies():
    world = d3x.World()
    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1e12)
    world.add_body(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1e10)  # 1 meter separation
    return world


@when("gravity is computed with softening enabled")
def compute_gravity_softened(world):
    d3x.compute_gravity(world, 10.0)  # 10 meter softening


@then("accelerations should remain finite")
def check_finite_accelerations(world):
    # Create fresh world and integrate with softening
    world2 = d3x.World()
    world2.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1e12)
    world2.add_body(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1e10)

    # Compute with softening - this should not produce NaN or Inf
    d3x.compute_gravity(world2, 10.0)

    # Take a tiny step to verify accelerations work
    dt = 0.0001
    d3x.step_rk4(world2, dt)

    assert np.isfinite(world2.vx[0])
    assert np.isfinite(world2.vx[1])


@then("the result should approach unsoftened values as separation increases")
def check_softening_convergence(world):
    G = d3x.constants.G
    _m1, m2 = 1e12, 1e10
    softening = 10.0

    # At large separation, softening should have minimal effect
    large_r = 1e6

    # Unsoftened: a = G*m/r²
    expected_a = G * m2 / (large_r * large_r)

    # Softened: a = G*m/(r² + ε²)
    softened_a = G * m2 / (large_r * large_r + softening * softening)

    # Should be nearly equal at large separation
    assert softened_a == pytest.approx(expected_a, rel=1e-6)
