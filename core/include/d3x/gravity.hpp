#pragma once

#include <d3x/world.hpp>

namespace d3x {

// Compute gravitational accelerations for all bodies
// Uses O(n²) direct summation with Newton's 3rd law optimization
void compute_gravity(World& w);

// Compute with softening to prevent singularities at close approach
void compute_gravity(World& w, real softening);

}  // namespace d3x
