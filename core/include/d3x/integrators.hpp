#pragma once

#include <d3x/world.hpp>
#include <d3x/gravity.hpp>

namespace d3x {

// Fixed-step 4th order Runge-Kutta
// Good general-purpose integrator for orbital mechanics
void step_rk4(World& w, real dt);

// Adaptive Dormand-Prince 5(4) integrator
// Embedded error estimation for automatic step size control
StepResult step_dopri54(World& w, real dt, real tol = 1e-9);

// Symplectic leapfrog (Verlet) - good energy conservation
// Requires accelerations to be pre-computed
void step_leapfrog(World& w, real dt);

}  // namespace d3x
