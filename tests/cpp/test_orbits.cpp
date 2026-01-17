#include <doctest/doctest.h>
#include <d3x/types.hpp>
#include <d3x/world.hpp>
#include <d3x/gravity.hpp>
#include <d3x/integrators.hpp>
#include <cmath>

using namespace d3x;

TEST_CASE("World add and clear bodies") {
    World w;

    CHECK(w.count == 0);

    auto idx = w.add_body({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, 100.0);
    CHECK(idx == 0);
    CHECK(w.count == 1);
    CHECK(w.px[0] == doctest::Approx(1.0));
    CHECK(w.py[0] == doctest::Approx(2.0));
    CHECK(w.pz[0] == doctest::Approx(3.0));
    CHECK(w.mass[0] == doctest::Approx(100.0));

    w.clear();
    CHECK(w.count == 0);
}

TEST_CASE("Two-body energy conservation") {
    World w;

    // Earth-Moon system (simplified, Earth at origin)
    constexpr real earth_mass = 5.972e24;
    constexpr real moon_mass = 7.342e22;
    constexpr real moon_distance = 384400e3;  // meters
    constexpr real moon_velocity = 1022.0;    // m/s

    w.add_body({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, earth_mass);
    w.add_body({moon_distance, 0.0, 0.0}, {0.0, moon_velocity, 0.0}, moon_mass);

    real initial_energy = w.total_energy();

    // Simulate ~1 day with RK4
    constexpr real dt = 60.0;  // 1 minute steps
    constexpr int steps = 24 * 60;  // 1 day

    for (int i = 0; i < steps; ++i) {
        step_rk4(w, dt);
    }

    real final_energy = w.total_energy();

    // Energy should be conserved within 0.01%
    real relative_error = std::abs(final_energy - initial_energy) / std::abs(initial_energy);
    CHECK(relative_error < 1e-4);
}

TEST_CASE("Angular momentum conservation") {
    World w;

    // Simple two-body
    w.add_body({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 1.0e10);
    w.add_body({1000.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, 1.0);

    Vec3 initial_L = w.angular_momentum();

    for (int i = 0; i < 1000; ++i) {
        step_rk4(w, 0.1);
    }

    Vec3 final_L = w.angular_momentum();

    // Angular momentum magnitude should be conserved
    real initial_mag = initial_L.magnitude();
    real final_mag = final_L.magnitude();
    real relative_error = std::abs(final_mag - initial_mag) / initial_mag;
    CHECK(relative_error < 1e-6);
}

TEST_CASE("Circular orbit period") {
    World w;

    // Central body and orbiting test mass
    constexpr real M = 1.0e12;  // Central mass
    constexpr real r = 1000.0;  // Orbital radius
    constexpr real v = std::sqrt(constants::G * M / r);  // Circular velocity

    w.add_body({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, M);
    w.add_body({r, 0.0, 0.0}, {0.0, v, 0.0}, 1.0);

    // Theoretical period: T = 2*pi*sqrt(r³/(G*M))
    real expected_period = 2.0 * M_PI * std::sqrt(r * r * r / (constants::G * M));

    // Record initial position angle
    real initial_angle = std::atan2(w.py[1], w.px[1]);

    // Simulate one full orbit
    real dt = expected_period / 1000.0;
    for (int i = 0; i < 1000; ++i) {
        step_rk4(w, dt);
    }

    // Should return close to initial position
    real final_r = std::sqrt(w.px[1]*w.px[1] + w.py[1]*w.py[1]);
    CHECK(final_r == doctest::Approx(r).epsilon(0.01));

    real final_angle = std::atan2(w.py[1], w.px[1]);
    real angle_diff = std::abs(final_angle - initial_angle);
    if (angle_diff > M_PI) angle_diff = 2.0 * M_PI - angle_diff;
    CHECK(angle_diff < 0.05);  // Within ~3 degrees
}

TEST_CASE("Leapfrog symplectic properties") {
    World w;

    // Two-body problem
    w.add_body({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 1.0e10);
    w.add_body({1000.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, 1.0);

    // Initialize accelerations for leapfrog
    compute_gravity(w);

    real initial_energy = w.total_energy();

    // Symplectic integrators have bounded energy error
    for (int i = 0; i < 10000; ++i) {
        step_leapfrog(w, 0.01);
    }

    real final_energy = w.total_energy();

    // Leapfrog should maintain energy very well
    real relative_error = std::abs(final_energy - initial_energy) / std::abs(initial_energy);
    CHECK(relative_error < 1e-4);
}

TEST_CASE("Adaptive integrator step control") {
    World w;

    w.add_body({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 1.0e12);
    w.add_body({1000.0, 0.0, 0.0}, {0.0, 300.0, 0.0}, 1.0);

    real dt = 1.0;
    real total_time = 0.0;

    while (total_time < 100.0) {
        auto result = step_dopri54(w, dt, 1e-8);

        if (result.dt_used > 0) {
            // Step was accepted
            total_time += result.dt_used;
        }

        // Step size should adapt
        dt = result.dt_next;
        CHECK(dt > 0.0);
        CHECK(dt < 1000.0);  // Sanity check
    }

    CHECK(total_time >= 100.0);
}
