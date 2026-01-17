#pragma once

#include <cstddef>
#include <cmath>

namespace d3x {

// Precision alias - single point to switch float/double
using real = double;

// Physical constants (SI units)
namespace constants {
    constexpr real G = 6.67430e-11;      // Gravitational constant [m³/(kg·s²)]
    constexpr real AU = 1.495978707e11;  // Astronomical unit [m]
    constexpr real DAY = 86400.0;        // Seconds per day

    // Solar system masses [kg]
    constexpr real M_SUN = 1.98892e30;
    constexpr real M_EARTH = 5.97217e24;
    constexpr real M_MOON = 7.342e22;
    constexpr real M_MARS = 6.4171e23;

    // Standard gravitational parameters [m³/s²]
    constexpr real MU_SUN = G * M_SUN;
    constexpr real MU_EARTH = G * M_EARTH;
}

// Simple 3D vector for API convenience
struct Vec3 {
    real x = 0.0;
    real y = 0.0;
    real z = 0.0;

    constexpr Vec3() = default;
    constexpr Vec3(real x_, real y_, real z_) : x(x_), y(y_), z(z_) {}

    real magnitude() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    real magnitude_squared() const {
        return x*x + y*y + z*z;
    }
};

// Integrator step result for adaptive methods
struct StepResult {
    real dt_used;
    real dt_next;
    real error_estimate;
};

}  // namespace d3x
