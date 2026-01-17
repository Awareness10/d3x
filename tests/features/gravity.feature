Feature: Gravity Computation
  The gravity solver computes gravitational accelerations between all
  body pairs using Newton's law of universal gravitation.

  Background:
    Given a simulation world with the gravitational constant G

  Scenario: Two-body gravitational attraction
    Given two bodies separated by a known distance
    When gravity is computed
    Then each body should experience acceleration toward the other
    And the accelerations should follow Newton's inverse-square law

  Scenario: Acceleration is independent of test mass
    Given a massive body and two test bodies of different masses at equal distances
    When gravity is computed
    Then both test bodies should have equal acceleration magnitudes

  Scenario: Newton's third law symmetry
    Given two bodies of different masses
    When gravity is computed
    Then the forces should be equal and opposite
    And momentum should be conserved in the force calculation

  Scenario: Multi-body superposition
    Given three or more bodies in arbitrary positions
    When gravity is computed
    Then each body's acceleration should be the vector sum of pairwise attractions

  Scenario: Softening prevents singularities
    Given two bodies at very close separation
    When gravity is computed with softening enabled
    Then accelerations should remain finite
    And the result should approach unsoftened values as separation increases
