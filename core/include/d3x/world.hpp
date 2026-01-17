#pragma once

#include <d3x/types.hpp>
#include <vector>
#include <cstddef>

namespace d3x {

// SoA container for celestial bodies
// Flat arrays enable cache-efficient iteration and SIMD
struct World {
    // Position components [m]
    std::vector<real> px;
    std::vector<real> py;
    std::vector<real> pz;

    // Velocity components [m/s]
    std::vector<real> vx;
    std::vector<real> vy;
    std::vector<real> vz;

    // Acceleration scratch buffers [m/s²]
    std::vector<real> ax;
    std::vector<real> ay;
    std::vector<real> az;

    // Mass [kg]
    std::vector<real> mass;

    // Body count
    std::size_t count = 0;

    // Simulation time [s]
    real time = 0.0;

    // Pre-allocate memory for n bodies
    void reserve(std::size_t n);

    // Add a body, returns its index
    std::size_t add_body(Vec3 pos, Vec3 vel, real m);

    // Clear all bodies
    void clear();

    // Compute total kinetic energy [J]
    real kinetic_energy() const;

    // Compute total potential energy [J]
    real potential_energy() const;

    // Compute total mechanical energy [J]
    real total_energy() const;

    // Compute total angular momentum magnitude [kg·m²/s]
    Vec3 angular_momentum() const;
};

}  // namespace d3x
