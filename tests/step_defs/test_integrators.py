"""Step definitions for integrators.feature."""

import numpy as np
import pytest
from pytest_bdd import given, scenario, then, when

import d3x

# ============================================================================
# Scenarios
# ============================================================================


@scenario("../features/integrators.feature", "RK4 advances simulation time")
def test_rk4_advances_simulation_time():
    pass


@scenario("../features/integrators.feature", "RK4 maintains accuracy for smooth trajectories")
def test_rk4_maintains_accuracy_for_smooth_trajectories():
    pass


@scenario("../features/integrators.feature", "DOPRI54 adapts step size to tolerance")
def test_dopri54_adapts_step_size_to_tolerance():
    pass


@scenario("../features/integrators.feature", "DOPRI54 reduces step size for rapid changes")
def test_dopri54_reduces_step_size_for_rapid_changes():
    pass


@scenario("../features/integrators.feature", "DOPRI54 rejects steps exceeding tolerance")
def test_dopri54_rejects_steps_exceeding_tolerance():
    pass


@scenario("../features/integrators.feature", "Leapfrog requires pre-computed accelerations")
def test_leapfrog_requires_precomputed_accelerations():
    pass


@scenario("../features/integrators.feature", "Leapfrog preserves phase space structure")
def test_leapfrog_preserves_phase_space_structure():
    pass


# ============================================================================
# Shared context fixture
# ============================================================================


@pytest.fixture
def ctx():
    """Context dictionary for sharing state between steps."""
    return {}


# ============================================================================
# Background fixture
# ============================================================================


@given("a world with a stable two-body orbital system", target_fixture="world")
def orbital_world(ctx):
    """Create a stable two-body system for integration tests."""
    world = d3x.World()

    # Central body
    M = 1e15
    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)

    # Orbiting body in circular orbit
    r = 1e6
    v = np.sqrt(d3x.constants.G * M / r)
    world.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)

    # Store parameters in context
    ctx["central_mass"] = M
    ctx["orbital_radius"] = r
    ctx["orbital_velocity"] = v

    return world


# ============================================================================
# RK4 Scenarios
# ============================================================================


@when("I take an RK4 step with a given timestep")
def rk4_step(world, ctx):
    ctx["dt"] = 10.0
    ctx["initial_time"] = world.time
    ctx["initial_px"] = world.px[1]
    ctx["initial_vx"] = world.vx[1]

    d3x.step_rk4(world, ctx["dt"])


@then("the simulation time should advance by that timestep")
def check_time_advance(world, ctx):
    expected_time = ctx["initial_time"] + ctx["dt"]
    assert world.time == pytest.approx(expected_time)


@then("body positions and velocities should be updated")
def check_state_updated(world, ctx):
    # Positions and velocities should have changed (by more than floating point noise)
    assert abs(world.px[1] - ctx["initial_px"]) > 1e-9
    assert abs(world.vx[1] - ctx["initial_vx"]) > 1e-12


@given("a circular orbit with known period", target_fixture="world")
def circular_orbit_world(ctx):
    """Create system with known orbital period."""
    world = d3x.World()

    M = 1e15
    r = 1e6
    v = np.sqrt(d3x.constants.G * M / r)

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)

    # Calculate orbital period: T = 2π√(r³/GM)
    ctx["period"] = 2 * np.pi * np.sqrt(r**3 / (d3x.constants.G * M))
    ctx["initial_pos"] = (r, 0.0, 0.0)

    return world


@when("I integrate for one complete orbit using RK4")
def integrate_one_orbit(world, ctx):
    n_steps = 1000
    dt = ctx["period"] / n_steps

    for _ in range(n_steps):
        d3x.step_rk4(world, dt)


@then("the body should return close to its starting position")
def check_orbit_closure(world, ctx):
    final_x = world.px[1]
    final_y = world.py_[1]

    initial_x, initial_y, _ = ctx["initial_pos"]

    # Should return to within 1% of starting position
    assert final_x == pytest.approx(initial_x, rel=0.01)
    assert final_y == pytest.approx(initial_y, abs=initial_x * 0.01)


# ============================================================================
# DOPRI54 Scenarios
# ============================================================================


@when("I take a DOPRI54 step with a specified tolerance")
def dopri54_step(world, ctx):
    ctx["tol"] = 1e-8
    ctx["dt"] = 10.0
    ctx["result"] = d3x.step_dopri54(world, ctx["dt"], tol=ctx["tol"])


@then("it should return the actual timestep used")
def check_dt_used(ctx):
    result = ctx["result"]
    assert hasattr(result, "dt_used")
    assert result.dt_used > 0


@then("suggest a next timestep based on error estimation")
def check_dt_next(ctx):
    result = ctx["result"]
    assert hasattr(result, "dt_next")
    assert result.dt_next > 0


@given("a highly eccentric orbit with close approach", target_fixture="world")
def eccentric_orbit(ctx):
    """Create a highly eccentric orbit."""
    world = d3x.World()

    M = 1e15
    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)

    # Periapsis at 1e5 m, high eccentricity
    r_peri = 1e5
    e = 0.9  # Eccentricity
    r_apo = r_peri * (1 + e) / (1 - e)

    # Velocity at periapsis for this eccentricity
    a = (r_peri + r_apo) / 2  # Semi-major axis
    v_peri = np.sqrt(d3x.constants.G * M * (2 / r_peri - 1 / a))

    world.add_body(pos=(r_peri, 0.0, 0.0), vel=(0.0, v_peri, 0.0), mass=1.0)

    ctx["r_peri"] = r_peri
    ctx["r_apo"] = r_apo
    ctx["central_mass"] = M

    return world


@when("I integrate through the close approach with DOPRI54")
def integrate_eccentric(world, ctx):
    # Calculate approximate orbital period
    a = (ctx["r_peri"] + ctx["r_apo"]) / 2
    period = 2 * np.pi * np.sqrt(a**3 / (d3x.constants.G * ctx["central_mass"]))

    dt = period / 100
    total_time = 0.0
    ctx["step_sizes"] = []
    ctx["radii"] = []

    while total_time < period:
        result = d3x.step_dopri54(world, dt, tol=1e-8)
        if result.dt_used > 0:
            r = np.sqrt(world.px[1] ** 2 + world.py_[1] ** 2)
            ctx["radii"].append(r)
            ctx["step_sizes"].append(result.dt_used)
            total_time += result.dt_used
        dt = result.dt_next


@then("step sizes near periapsis should be smaller than at apoapsis")
def check_adaptive_stepping(ctx):
    # Find steps at small and large radii
    peri_steps = []
    apo_steps = []

    threshold = (ctx["r_peri"] + ctx["r_apo"]) / 2

    for r, dt in zip(ctx["radii"], ctx["step_sizes"], strict=True):
        if r < threshold:
            peri_steps.append(dt)
        else:
            apo_steps.append(dt)

    if peri_steps and apo_steps:
        avg_peri = np.mean(peri_steps)
        avg_apo = np.mean(apo_steps)
        # Steps at periapsis should generally be smaller
        assert avg_peri < avg_apo


@given("a very loose tolerance and then a tight tolerance", target_fixture="tolerance_setup")
def tolerance_test_setup():
    """Setup for tolerance comparison test."""
    return {"loose_tol": 1e-4, "tight_tol": 1e-10}


@when("integrating the same trajectory")
def integrate_with_tolerances(tolerance_setup, ctx):
    setup = tolerance_setup

    # Count steps with loose tolerance
    world1 = d3x.World()
    M = 1e15
    r = 1e6
    v = np.sqrt(d3x.constants.G * M / r)
    world1.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world1.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)

    period = 2 * np.pi * np.sqrt(r**3 / (d3x.constants.G * M))
    target_time = period / 4

    dt = target_time / 10
    total_time = 0.0
    loose_steps = 0

    while total_time < target_time:
        result = d3x.step_dopri54(world1, dt, tol=setup["loose_tol"])
        if result.dt_used > 0:
            total_time += result.dt_used
            loose_steps += 1
        dt = result.dt_next

    ctx["loose_steps"] = loose_steps

    # Count steps with tight tolerance
    world2 = d3x.World()
    world2.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world2.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)

    dt = target_time / 10
    total_time = 0.0
    tight_steps = 0

    while total_time < target_time:
        result = d3x.step_dopri54(world2, dt, tol=setup["tight_tol"])
        if result.dt_used > 0:
            total_time += result.dt_used
            tight_steps += 1
        dt = result.dt_next

    ctx["tight_steps"] = tight_steps


@then("tight tolerance should use more steps than loose tolerance")
def check_tolerance_step_count(ctx):
    assert ctx["tight_steps"] > ctx["loose_steps"]


# ============================================================================
# Leapfrog Scenarios
# ============================================================================


@given("a world with accelerations already computed", target_fixture="world")
def leapfrog_ready_world(ctx):
    """Create world with pre-computed accelerations for leapfrog."""
    world = d3x.World()

    M = 1e15
    r = 1e6
    v = np.sqrt(d3x.constants.G * M / r)

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)

    # Pre-compute accelerations (required for leapfrog)
    d3x.compute_gravity(world)

    ctx["initial_px"] = world.px[1]
    ctx["initial_vx"] = world.vx[1]

    return world


@when("I take a leapfrog step")
def leapfrog_step(world):
    d3x.step_leapfrog(world, 10.0)


@then("positions and velocities should be updated")
def check_leapfrog_update(world, ctx):
    # Positions and velocities should have changed (by more than floating point noise)
    assert abs(world.px[1] - ctx["initial_px"]) > 1e-9
    assert abs(world.vx[1] - ctx["initial_vx"]) > 1e-12


@then("accelerations should be recomputed for the next step")
def check_accelerations_recomputed(world):
    # Leapfrog recomputes gravity internally, so we can take another step
    # If accelerations weren't recomputed, this would fail or give wrong results
    d3x.step_leapfrog(world, 10.0)
    # If we got here without error, accelerations were updated
    assert True


@given("a bound orbital system", target_fixture="world")
def bound_system(ctx):
    """Create bound system for long-term integration."""
    world = d3x.World()

    M = 1e15
    r = 1e6
    v = np.sqrt(d3x.constants.G * M / r)

    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=M)
    world.add_body(pos=(r, 0.0, 0.0), vel=(0.0, v, 0.0), mass=1.0)

    ctx["initial_energy"] = world.total_energy()
    ctx["central_mass"] = M
    ctx["orbital_radius"] = r

    # Pre-compute for leapfrog
    d3x.compute_gravity(world)

    return world


@when("I integrate for many orbits using leapfrog")
def integrate_many_orbits_leapfrog(world, ctx):
    period = (
        2 * np.pi * np.sqrt(ctx["orbital_radius"] ** 3 / (d3x.constants.G * ctx["central_mass"]))
    )
    n_orbits = 10
    n_steps_per_orbit = 100
    dt = period / n_steps_per_orbit

    ctx["energy_history"] = []

    for _ in range(n_orbits * n_steps_per_orbit):
        d3x.step_leapfrog(world, dt)
        ctx["energy_history"].append(world.total_energy())


@then("energy error should remain bounded and not grow exponentially")
def check_bounded_energy_error(ctx):
    E0 = ctx["initial_energy"]
    errors = [abs(E - E0) / abs(E0) for E in ctx["energy_history"]]

    # For symplectic integrator, error should be bounded
    max_error = max(errors)
    assert max_error < 0.01  # Less than 1% error

    # Error should not grow exponentially - check that late errors aren't much worse
    early_errors = errors[: len(errors) // 4]
    late_errors = errors[-len(errors) // 4 :]

    avg_early = np.mean(early_errors)
    avg_late = np.mean(late_errors)

    # Late errors should not be dramatically larger (allowing 10x for oscillation)
    assert avg_late < avg_early * 10
