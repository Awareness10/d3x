"""Step definitions for conservation.feature."""

import numpy as np
import pytest
from pytest_bdd import given, scenario, then, when

import d3x

# ============================================================================
# Scenarios
# ============================================================================


@scenario("../features/conservation.feature", "Total energy is preserved during integration")
def test_total_energy_is_preserved_during_integration():
    pass


@scenario("../features/conservation.feature", "Energy decomposes into kinetic and potential")
def test_energy_decomposes_into_kinetic_and_potential():
    pass


@scenario("../features/conservation.feature", "Kinetic energy depends on velocities")
def test_kinetic_energy_depends_on_velocities():
    pass


@scenario("../features/conservation.feature", "Potential energy depends on separations")
def test_potential_energy_depends_on_separations():
    pass


@scenario("../features/conservation.feature", "Angular momentum is preserved during integration")
def test_angular_momentum_is_preserved_during_integration():
    pass


@scenario("../features/conservation.feature", "Angular momentum is a vector quantity")
def test_angular_momentum_is_a_vector_quantity():
    pass


@scenario("../features/conservation.feature", "Symplectic integrator bounds energy drift")
def test_symplectic_integrator_bounds_energy_drift():
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


@given("an isolated gravitational system with no external forces", target_fixture="world")
def isolated_system():
    """Create an isolated gravitational system."""
    return d3x.World()


# ============================================================================
# Energy Conservation Scenarios
# ============================================================================


@given("a bound two-body system with known initial energy", target_fixture="world")
def bound_two_body(ctx):
    world = d3x.World()

    M = 1e15
    r = 1e6
    v = np.sqrt(d3x.constants.G * M / r)

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)

    ctx["initial_energy"] = world.total_energy()
    ctx["central_mass"] = M
    ctx["orbital_radius"] = r

    return world


@when("I integrate for a significant time period")
def integrate_significant_time(world, ctx):
    period = (
        2 * np.pi * np.sqrt(ctx["orbital_radius"] ** 3 / (d3x.constants.G * ctx["central_mass"]))
    )
    n_steps = 500
    dt = period / n_steps

    for _ in range(n_steps):
        d3x.step_rk4(world, dt)


@then("the total energy should remain close to the initial value")
def check_energy_conserved(world, ctx):
    final_energy = world.total_energy()
    relative_error = abs(final_energy - ctx["initial_energy"]) / abs(ctx["initial_energy"])

    assert relative_error < 1e-4  # Less than 0.01% error


@given("a system of bodies with positions and velocities", target_fixture="world")
def multi_body_system(ctx):
    world = d3x.World()

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1e12)
    world.add_body(pos=(1e6, 0.0, 0.0), vel=(0.0, 100.0, 0.0), mass=1e6)
    world.add_body(pos=(0.0, 1e6, 0.0), vel=(100.0, 0.0, 0.0), mass=1e6)

    return world


@when("I compute the total energy")
def compute_total_energy(world, ctx):
    ctx["total"] = world.total_energy()
    ctx["kinetic"] = world.kinetic_energy()
    ctx["potential"] = world.potential_energy()


@then("it should equal kinetic energy plus gravitational potential energy")
def check_energy_decomposition(ctx):
    assert ctx["total"] == pytest.approx(ctx["kinetic"] + ctx["potential"])


@given("bodies with known masses and velocities", target_fixture="world")
def known_velocity_bodies(ctx):
    world = d3x.World()

    # Body 1: mass=2, velocity=(3,0,0) -> KE = 0.5 * 2 * 9 = 9
    world.add_body(pos=(0.0, 0.0, 0.0), vel=(3.0, 0.0, 0.0), mass=2.0)

    # Body 2: mass=4, velocity=(0,5,0) -> KE = 0.5 * 4 * 25 = 50
    world.add_body(pos=(1e10, 0.0, 0.0), vel=(0.0, 5.0, 0.0), mass=4.0)

    ctx["expected_ke"] = 9.0 + 50.0

    return world


@when("I compute kinetic energy")
def compute_kinetic(world, ctx):
    ctx["actual_ke"] = world.kinetic_energy()


@then("it should follow the half-mv-squared formula for each body")
def check_kinetic_formula(ctx):
    assert ctx["actual_ke"] == pytest.approx(ctx["expected_ke"])


@given("bodies with known masses and positions", target_fixture="world")
def known_position_bodies(ctx):
    world = d3x.World()

    m1 = 1e10
    m2 = 2e10
    r = 1e6

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=m1)
    world.add_body(pos=(r, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=m2)

    # Expected PE = -G * m1 * m2 / r
    ctx["expected_pe"] = -d3x.constants.G * m1 * m2 / r

    return world


@when("I compute potential energy")
def compute_potential(world, ctx):
    ctx["actual_pe"] = world.potential_energy()


@then("it should be negative and depend on pairwise inverse distances")
def check_potential_formula(ctx):
    assert ctx["actual_pe"] < 0
    assert ctx["actual_pe"] == pytest.approx(ctx["expected_pe"])


# ============================================================================
# Angular Momentum Conservation Scenarios
# ============================================================================


@given("a system with known initial angular momentum", target_fixture="world")
def angular_momentum_system(ctx):
    world = d3x.World()

    M = 1e15
    r = 1e6
    v = np.sqrt(d3x.constants.G * M / r)

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)

    L = world.angular_momentum()
    ctx["initial_L"] = np.array([L.x, L.y, L.z])
    ctx["central_mass"] = M
    ctx["orbital_radius"] = r

    return world


@when("I integrate for many orbital periods")
def integrate_many_periods(world, ctx):
    period = (
        2 * np.pi * np.sqrt(ctx["orbital_radius"] ** 3 / (d3x.constants.G * ctx["central_mass"]))
    )
    n_orbits = 5
    n_steps_per_orbit = 100
    dt = period / n_steps_per_orbit

    for _ in range(n_orbits * n_steps_per_orbit):
        d3x.step_rk4(world, dt)


@then("the angular momentum vector should remain close to the initial value")
def check_angular_momentum_conserved(world, ctx):
    L = world.angular_momentum()
    final_L = np.array([L.x, L.y, L.z])

    # Check each component
    relative_error = np.linalg.norm(final_L - ctx["initial_L"]) / np.linalg.norm(ctx["initial_L"])

    assert relative_error < 1e-4  # Less than 0.01% error


@given("an orbital system in a specific plane", target_fixture="world")
def planar_orbit(ctx):
    world = d3x.World()

    # Orbit in x-y plane
    M = 1e15
    r = 1e6
    v = np.sqrt(d3x.constants.G * M / r)

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)

    return world


@when("I compute angular momentum")
def compute_angular_momentum(world, ctx):
    ctx["L"] = world.angular_momentum()


@then("it should be perpendicular to the orbital plane")
def check_angular_momentum_direction(ctx):
    L = ctx["L"]

    # For x-y plane orbit, L should be along z-axis
    L_mag = np.sqrt(L.x**2 + L.y**2 + L.z**2)

    # x and y components should be negligible
    assert abs(L.x) / L_mag < 1e-10
    assert abs(L.y) / L_mag < 1e-10

    # z component should be the full magnitude
    assert abs(L.z) == pytest.approx(L_mag)


# ============================================================================
# Integrator Comparison Scenario
# ============================================================================


@given("identical initial conditions", target_fixture="comparison_setup")
def identical_conditions():
    """Store initial conditions for comparison."""
    return {
        "M": 1e15,
        "r": 1e6,
    }


@when("I compare RK4 and leapfrog over many orbits")
def compare_integrators(comparison_setup, ctx):
    setup = comparison_setup
    M = setup["M"]
    r = setup["r"]
    v = np.sqrt(d3x.constants.G * M / r)

    period = 2 * np.pi * np.sqrt(r**3 / (d3x.constants.G * M))
    n_orbits = 20
    n_steps_per_orbit = 100
    dt = period / n_steps_per_orbit

    # RK4 integration
    world_rk4 = d3x.World()
    world_rk4.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world_rk4.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)
    E0_rk4 = world_rk4.total_energy()

    rk4_errors = []
    for _ in range(n_orbits * n_steps_per_orbit):
        d3x.step_rk4(world_rk4, dt)
        error = abs(world_rk4.total_energy() - E0_rk4) / abs(E0_rk4)
        rk4_errors.append(error)

    # Leapfrog integration
    world_lf = d3x.World()
    world_lf.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world_lf.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)
    E0_lf = world_lf.total_energy()

    d3x.compute_gravity(world_lf)  # Required for leapfrog

    lf_errors = []
    for _ in range(n_orbits * n_steps_per_orbit):
        d3x.step_leapfrog(world_lf, dt)
        error = abs(world_lf.total_energy() - E0_lf) / abs(E0_lf)
        lf_errors.append(error)

    ctx["rk4_errors"] = rk4_errors
    ctx["lf_errors"] = lf_errors


@then("leapfrog energy error should remain bounded")
def check_leapfrog_bounded(ctx):
    # Leapfrog error should stay bounded
    max_error = max(ctx["lf_errors"])
    assert max_error < 0.01  # Less than 1%

    # Error should not grow systematically
    early = ctx["lf_errors"][: len(ctx["lf_errors"]) // 4]
    late = ctx["lf_errors"][-len(ctx["lf_errors"]) // 4 :]

    # Late errors should be similar magnitude to early errors
    assert np.mean(late) < np.mean(early) * 5


@then("RK4 energy error may grow over very long integrations")
def check_rk4_behavior(ctx):
    # RK4 can have growing error, but should still be reasonable
    max_error = max(ctx["rk4_errors"])
    assert max_error < 0.1  # Less than 10% over 20 orbits

    # We just verify the test completed - RK4 may or may not show drift
    # depending on step size and number of orbits
    assert len(ctx["rk4_errors"]) > 0
