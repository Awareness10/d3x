Feature: World Container
  The World is a Struct-of-Arrays container that holds celestial bodies
  with positions, velocities, and masses in contiguous memory.

  Background:
    Given an empty simulation world

  Scenario: Creating an empty world
    Then the world should have no bodies
    And the simulation time should be zero

  Scenario: Adding a single body
    When I add a body at the origin with zero velocity and unit mass
    Then the world should have one body
    And the body should be at the origin

  Scenario: Adding multiple bodies
    When I add a body with position [1, 0, 0] and mass 1000
    And I add a body with position [0, 1, 0] and mass 500
    Then the world should have 2 bodies

  Scenario: Accessing positions as arrays
    When I add several bodies to the world
    Then positions should be accessible as separate x, y, z arrays
    And each array should have length equal to the body count

  Scenario: Zero-copy numpy integration
    When I add a body and modify its position through the numpy view
    Then the world's internal state should reflect the change

  Scenario: Clearing the world
    Given a world with multiple bodies
    When I clear the world
    Then the world should have no bodies
    And the simulation time should be zero
