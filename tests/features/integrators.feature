Feature: Time Integration Methods
  Integrators advance the simulation forward in time by solving
  the equations of motion with different numerical methods.

  Background:
    Given a world with a stable two-body orbital system

  # RK4 - Fixed Step
  Scenario: RK4 advances simulation time
    When I take an RK4 step with a given timestep
    Then the simulation time should advance by that timestep
    And body positions and velocities should be updated

  Scenario: RK4 maintains accuracy for smooth trajectories
    Given a circular orbit with known period
    When I integrate for one complete orbit using RK4
    Then the body should return close to its starting position

  # Dormand-Prince 5(4) - Adaptive
  Scenario: DOPRI54 adapts step size to tolerance
    When I take a DOPRI54 step with a specified tolerance
    Then it should return the actual timestep used
    And suggest a next timestep based on error estimation

  Scenario: DOPRI54 reduces step size for rapid changes
    Given a highly eccentric orbit with close approach
    When I integrate through the close approach with DOPRI54
    Then step sizes near periapsis should be smaller than at apoapsis

  Scenario: DOPRI54 rejects steps exceeding tolerance
    Given a very loose tolerance and then a tight tolerance
    When integrating the same trajectory
    Then tight tolerance should use more steps than loose tolerance

  # Leapfrog - Symplectic
  Scenario: Leapfrog requires pre-computed accelerations
    Given a world with accelerations already computed
    When I take a leapfrog step
    Then positions and velocities should be updated
    And accelerations should be recomputed for the next step

  Scenario: Leapfrog preserves phase space structure
    Given a bound orbital system
    When I integrate for many orbits using leapfrog
    Then energy error should remain bounded and not grow exponentially
