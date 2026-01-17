"""Step definitions for world.feature."""

import numpy as np
import pytest
from pytest_bdd import given, parsers, scenario, then, when

import d3x

# ============================================================================
# Scenarios
# ============================================================================


@scenario("../features/world.feature", "Creating an empty world")
def test_creating_an_empty_world():
    pass


@scenario("../features/world.feature", "Adding a single body")
def test_adding_a_single_body():
    pass


@scenario("../features/world.feature", "Adding multiple bodies")
def test_adding_multiple_bodies():
    pass


@scenario("../features/world.feature", "Accessing positions as arrays")
def test_accessing_positions_as_arrays():
    pass


@scenario("../features/world.feature", "Zero-copy numpy integration")
def test_zero_copy_numpy_integration():
    pass


@scenario("../features/world.feature", "Clearing the world")
def test_clearing_the_world():
    pass


# ============================================================================
# Shared state container
# ============================================================================


@pytest.fixture
def ctx():
    """Context dictionary for sharing state between steps."""
    return {}


# ============================================================================
# Background
# ============================================================================


@given("an empty simulation world", target_fixture="world")
def empty_world():
    """Create and return an empty World."""
    return d3x.World()


# ============================================================================
# Scenario: Creating an empty world
# ============================================================================


@then("the world should have no bodies")
def check_no_bodies(world):
    assert world.count == 0


@then("the simulation time should be zero")
def check_time_zero(world):
    assert world.time == 0.0


# ============================================================================
# Scenario: Adding a single body
# ============================================================================


@when("I add a body at the origin with zero velocity and unit mass")
def add_origin_body(world):
    world.add_body(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)


@then("the world should have one body")
def check_one_body(world):
    assert world.count == 1


@then("the body should be at the origin")
def check_at_origin(world):
    assert world.px[0] == pytest.approx(0.0)
    assert world.py_[0] == pytest.approx(0.0)
    assert world.pz[0] == pytest.approx(0.0)


# ============================================================================
# Scenario: Adding multiple bodies
# ============================================================================


@when(parsers.parse("I add a body with position [{x:g}, {y:g}, {z:g}] and mass {mass:g}"))
def add_body_at_position(world, x, y, z, mass):
    world.add_body(pos=(x, y, z), vel=(0.0, 0.0, 0.0), mass=mass)


@then(parsers.parse("the world should have {n:d} bodies"))
def check_body_count(world, n):
    assert world.count == n


# ============================================================================
# Scenario: Accessing positions as arrays
# ============================================================================


@when("I add several bodies to the world")
def add_several_bodies(world):
    world.add_body(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
    world.add_body(pos=(0.0, 2.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0)
    world.add_body(pos=(0.0, 0.0, 3.0), vel=(0.0, 0.0, 0.0), mass=3.0)


@then("positions should be accessible as separate x, y, z arrays")
def check_array_access(world):
    px = world.px
    py = world.py_
    pz = world.pz

    assert isinstance(px, np.ndarray)
    assert isinstance(py, np.ndarray)
    assert isinstance(pz, np.ndarray)


@then("each array should have length equal to the body count")
def check_array_lengths(world):
    assert len(world.px) == world.count
    assert len(world.py_) == world.count
    assert len(world.pz) == world.count


# ============================================================================
# Scenario: Zero-copy numpy integration
# ============================================================================


@when("I add a body and modify its position through the numpy view")
def modify_through_numpy(world):
    world.add_body(pos=(1.0, 2.0, 3.0), vel=(0.0, 0.0, 0.0), mass=1.0)
    # Modify position through numpy view
    world.px[0] = 99.0


@then("the world's internal state should reflect the change")
def check_numpy_modification(world):
    assert world.px[0] == pytest.approx(99.0)


# ============================================================================
# Scenario: Clearing the world
# ============================================================================


@given("a world with multiple bodies", target_fixture="world")
def world_with_bodies():
    world = d3x.World()
    world.add_body(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
    world.add_body(pos=(0.0, 1.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
    world.time = 100.0
    return world


@when("I clear the world")
def clear_world(world):
    world.clear()
