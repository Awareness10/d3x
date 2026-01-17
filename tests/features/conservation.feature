Feature: Conservation Laws
  A physically correct simulation must conserve energy and angular
  momentum in isolated systems where no external forces act.

  Background:
    Given an isolated gravitational system with no external forces

  # Energy Conservation
  Scenario: Total energy is preserved during integration
    Given a bound two-body system with known initial energy
    When I integrate for a significant time period
    Then the total energy should remain close to the initial value

  Scenario: Energy decomposes into kinetic and potential
    Given a system of bodies with positions and velocities
    When I compute the total energy
    Then it should equal kinetic energy plus gravitational potential energy

  Scenario: Kinetic energy depends on velocities
    Given bodies with known masses and velocities
    When I compute kinetic energy
    Then it should follow the half-mv-squared formula for each body

  Scenario: Potential energy depends on separations
    Given bodies with known masses and positions
    When I compute potential energy
    Then it should be negative and depend on pairwise inverse distances

  # Angular Momentum Conservation
  Scenario: Angular momentum is preserved during integration
    Given a system with known initial angular momentum
    When I integrate for many orbital periods
    Then the angular momentum vector should remain close to the initial value

  Scenario: Angular momentum is a vector quantity
    Given an orbital system in a specific plane
    When I compute angular momentum
    Then it should be perpendicular to the orbital plane

  # Integrator Comparison
  Scenario: Symplectic integrator bounds energy drift
    Given identical initial conditions
    When I compare RK4 and leapfrog over many orbits
    Then leapfrog energy error should remain bounded
    And RK4 energy error may grow over very long integrations
