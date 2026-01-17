#include <d3x/integrators.hpp>
#include <cmath>
#include <algorithm>
#include <vector>

namespace d3x {

namespace {
    // Thread-local scratch buffers to avoid allocations in hot path
    thread_local std::vector<real> scratch_px, scratch_py, scratch_pz;
    thread_local std::vector<real> scratch_vx, scratch_vy, scratch_vz;
    thread_local std::vector<real> k1x, k1y, k1z, k1vx, k1vy, k1vz;
    thread_local std::vector<real> k2x, k2y, k2z, k2vx, k2vy, k2vz;
    thread_local std::vector<real> k3x, k3y, k3z, k3vx, k3vy, k3vz;
    thread_local std::vector<real> k4x, k4y, k4z, k4vx, k4vy, k4vz;

    void ensure_scratch(std::size_t n) {
        if (scratch_px.size() < n) {
            scratch_px.resize(n); scratch_py.resize(n); scratch_pz.resize(n);
            scratch_vx.resize(n); scratch_vy.resize(n); scratch_vz.resize(n);
            k1x.resize(n); k1y.resize(n); k1z.resize(n);
            k1vx.resize(n); k1vy.resize(n); k1vz.resize(n);
            k2x.resize(n); k2y.resize(n); k2z.resize(n);
            k2vx.resize(n); k2vy.resize(n); k2vz.resize(n);
            k3x.resize(n); k3y.resize(n); k3z.resize(n);
            k3vx.resize(n); k3vy.resize(n); k3vz.resize(n);
            k4x.resize(n); k4y.resize(n); k4z.resize(n);
            k4vx.resize(n); k4vy.resize(n); k4vz.resize(n);
        }
    }

    // Save current state to scratch
    void save_state(const World& w) {
        const std::size_t n = w.count;
        for (std::size_t i = 0; i < n; ++i) {
            scratch_px[i] = w.px[i]; scratch_py[i] = w.py[i]; scratch_pz[i] = w.pz[i];
            scratch_vx[i] = w.vx[i]; scratch_vy[i] = w.vy[i]; scratch_vz[i] = w.vz[i];
        }
    }

    // Restore state from scratch
    void restore_state(World& w) {
        const std::size_t n = w.count;
        for (std::size_t i = 0; i < n; ++i) {
            w.px[i] = scratch_px[i]; w.py[i] = scratch_py[i]; w.pz[i] = scratch_pz[i];
            w.vx[i] = scratch_vx[i]; w.vy[i] = scratch_vy[i]; w.vz[i] = scratch_vz[i];
        }
    }
}

void step_rk4(World& w, real dt) {
    const std::size_t n = w.count;
    ensure_scratch(n);
    save_state(w);

    // k1 = f(t, y) - derivatives at current state
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k1x[i] = w.vx[i];  k1y[i] = w.vy[i];  k1z[i] = w.vz[i];
        k1vx[i] = w.ax[i]; k1vy[i] = w.ay[i]; k1vz[i] = w.az[i];
    }

    // k2 = f(t + dt/2, y + dt/2 * k1)
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] = scratch_px[i] + 0.5 * dt * k1x[i];
        w.py[i] = scratch_py[i] + 0.5 * dt * k1y[i];
        w.pz[i] = scratch_pz[i] + 0.5 * dt * k1z[i];
        w.vx[i] = scratch_vx[i] + 0.5 * dt * k1vx[i];
        w.vy[i] = scratch_vy[i] + 0.5 * dt * k1vy[i];
        w.vz[i] = scratch_vz[i] + 0.5 * dt * k1vz[i];
    }
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k2x[i] = w.vx[i];  k2y[i] = w.vy[i];  k2z[i] = w.vz[i];
        k2vx[i] = w.ax[i]; k2vy[i] = w.ay[i]; k2vz[i] = w.az[i];
    }

    // k3 = f(t + dt/2, y + dt/2 * k2)
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] = scratch_px[i] + 0.5 * dt * k2x[i];
        w.py[i] = scratch_py[i] + 0.5 * dt * k2y[i];
        w.pz[i] = scratch_pz[i] + 0.5 * dt * k2z[i];
        w.vx[i] = scratch_vx[i] + 0.5 * dt * k2vx[i];
        w.vy[i] = scratch_vy[i] + 0.5 * dt * k2vy[i];
        w.vz[i] = scratch_vz[i] + 0.5 * dt * k2vz[i];
    }
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k3x[i] = w.vx[i];  k3y[i] = w.vy[i];  k3z[i] = w.vz[i];
        k3vx[i] = w.ax[i]; k3vy[i] = w.ay[i]; k3vz[i] = w.az[i];
    }

    // k4 = f(t + dt, y + dt * k3)
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] = scratch_px[i] + dt * k3x[i];
        w.py[i] = scratch_py[i] + dt * k3y[i];
        w.pz[i] = scratch_pz[i] + dt * k3z[i];
        w.vx[i] = scratch_vx[i] + dt * k3vx[i];
        w.vy[i] = scratch_vy[i] + dt * k3vy[i];
        w.vz[i] = scratch_vz[i] + dt * k3vz[i];
    }
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k4x[i] = w.vx[i];  k4y[i] = w.vy[i];  k4z[i] = w.vz[i];
        k4vx[i] = w.ax[i]; k4vy[i] = w.ay[i]; k4vz[i] = w.az[i];
    }

    // y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    const real dt6 = dt / 6.0;
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] = scratch_px[i] + dt6 * (k1x[i] + 2.0*k2x[i] + 2.0*k3x[i] + k4x[i]);
        w.py[i] = scratch_py[i] + dt6 * (k1y[i] + 2.0*k2y[i] + 2.0*k3y[i] + k4y[i]);
        w.pz[i] = scratch_pz[i] + dt6 * (k1z[i] + 2.0*k2z[i] + 2.0*k3z[i] + k4z[i]);
        w.vx[i] = scratch_vx[i] + dt6 * (k1vx[i] + 2.0*k2vx[i] + 2.0*k3vx[i] + k4vx[i]);
        w.vy[i] = scratch_vy[i] + dt6 * (k1vy[i] + 2.0*k2vy[i] + 2.0*k3vy[i] + k4vy[i]);
        w.vz[i] = scratch_vz[i] + dt6 * (k1vz[i] + 2.0*k2vz[i] + 2.0*k3vz[i] + k4vz[i]);
    }

    w.time += dt;
}

StepResult step_dopri54(World& w, real dt, real tol) {
    // Dormand-Prince 5(4) coefficients
    constexpr real c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 4.0/5.0, c5 = 8.0/9.0;

    constexpr real a21 = 1.0/5.0;
    constexpr real a31 = 3.0/40.0,     a32 = 9.0/40.0;
    constexpr real a41 = 44.0/45.0,    a42 = -56.0/15.0,    a43 = 32.0/9.0;
    constexpr real a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
    constexpr real a61 = 9017.0/3168.0,  a62 = -355.0/33.0,    a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
    constexpr real a71 = 35.0/384.0,     a73 = 500.0/1113.0,   a74 = 125.0/192.0,    a75 = -2187.0/6784.0, a76 = 11.0/84.0;

    // 5th order weights (for solution)
    constexpr real b1 = 35.0/384.0, b3 = 500.0/1113.0, b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;

    // 4th order weights (for error estimation)
    constexpr real e1 = 71.0/57600.0, e3 = -71.0/16695.0, e4 = 71.0/1920.0, e5 = -17253.0/339200.0, e6 = 22.0/525.0, e7 = -1.0/40.0;

    const std::size_t n = w.count;
    ensure_scratch(n);
    save_state(w);

    // We'll reuse k arrays for stages, need k5, k6, k7 too
    thread_local std::vector<real> k5x, k5y, k5z, k5vx, k5vy, k5vz;
    thread_local std::vector<real> k6x, k6y, k6z, k6vx, k6vy, k6vz;
    thread_local std::vector<real> k7x, k7y, k7z, k7vx, k7vy, k7vz;

    if (k5x.size() < n) {
        k5x.resize(n); k5y.resize(n); k5z.resize(n);
        k5vx.resize(n); k5vy.resize(n); k5vz.resize(n);
        k6x.resize(n); k6y.resize(n); k6z.resize(n);
        k6vx.resize(n); k6vy.resize(n); k6vz.resize(n);
        k7x.resize(n); k7y.resize(n); k7z.resize(n);
        k7vx.resize(n); k7vy.resize(n); k7vz.resize(n);
    }

    // Stage 1: k1 at current state
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k1x[i] = w.vx[i];  k1y[i] = w.vy[i];  k1z[i] = w.vz[i];
        k1vx[i] = w.ax[i]; k1vy[i] = w.ay[i]; k1vz[i] = w.az[i];
    }

    // Stage 2
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] = scratch_px[i] + dt * a21 * k1x[i];
        w.py[i] = scratch_py[i] + dt * a21 * k1y[i];
        w.pz[i] = scratch_pz[i] + dt * a21 * k1z[i];
        w.vx[i] = scratch_vx[i] + dt * a21 * k1vx[i];
        w.vy[i] = scratch_vy[i] + dt * a21 * k1vy[i];
        w.vz[i] = scratch_vz[i] + dt * a21 * k1vz[i];
    }
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k2x[i] = w.vx[i];  k2y[i] = w.vy[i];  k2z[i] = w.vz[i];
        k2vx[i] = w.ax[i]; k2vy[i] = w.ay[i]; k2vz[i] = w.az[i];
    }

    // Stage 3
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] = scratch_px[i] + dt * (a31*k1x[i] + a32*k2x[i]);
        w.py[i] = scratch_py[i] + dt * (a31*k1y[i] + a32*k2y[i]);
        w.pz[i] = scratch_pz[i] + dt * (a31*k1z[i] + a32*k2z[i]);
        w.vx[i] = scratch_vx[i] + dt * (a31*k1vx[i] + a32*k2vx[i]);
        w.vy[i] = scratch_vy[i] + dt * (a31*k1vy[i] + a32*k2vy[i]);
        w.vz[i] = scratch_vz[i] + dt * (a31*k1vz[i] + a32*k2vz[i]);
    }
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k3x[i] = w.vx[i];  k3y[i] = w.vy[i];  k3z[i] = w.vz[i];
        k3vx[i] = w.ax[i]; k3vy[i] = w.ay[i]; k3vz[i] = w.az[i];
    }

    // Stage 4
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] = scratch_px[i] + dt * (a41*k1x[i] + a42*k2x[i] + a43*k3x[i]);
        w.py[i] = scratch_py[i] + dt * (a41*k1y[i] + a42*k2y[i] + a43*k3y[i]);
        w.pz[i] = scratch_pz[i] + dt * (a41*k1z[i] + a42*k2z[i] + a43*k3z[i]);
        w.vx[i] = scratch_vx[i] + dt * (a41*k1vx[i] + a42*k2vx[i] + a43*k3vx[i]);
        w.vy[i] = scratch_vy[i] + dt * (a41*k1vy[i] + a42*k2vy[i] + a43*k3vy[i]);
        w.vz[i] = scratch_vz[i] + dt * (a41*k1vz[i] + a42*k2vz[i] + a43*k3vz[i]);
    }
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k4x[i] = w.vx[i];  k4y[i] = w.vy[i];  k4z[i] = w.vz[i];
        k4vx[i] = w.ax[i]; k4vy[i] = w.ay[i]; k4vz[i] = w.az[i];
    }

    // Stage 5
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] = scratch_px[i] + dt * (a51*k1x[i] + a52*k2x[i] + a53*k3x[i] + a54*k4x[i]);
        w.py[i] = scratch_py[i] + dt * (a51*k1y[i] + a52*k2y[i] + a53*k3y[i] + a54*k4y[i]);
        w.pz[i] = scratch_pz[i] + dt * (a51*k1z[i] + a52*k2z[i] + a53*k3z[i] + a54*k4z[i]);
        w.vx[i] = scratch_vx[i] + dt * (a51*k1vx[i] + a52*k2vx[i] + a53*k3vx[i] + a54*k4vx[i]);
        w.vy[i] = scratch_vy[i] + dt * (a51*k1vy[i] + a52*k2vy[i] + a53*k3vy[i] + a54*k4vy[i]);
        w.vz[i] = scratch_vz[i] + dt * (a51*k1vz[i] + a52*k2vz[i] + a53*k3vz[i] + a54*k4vz[i]);
    }
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k5x[i] = w.vx[i];  k5y[i] = w.vy[i];  k5z[i] = w.vz[i];
        k5vx[i] = w.ax[i]; k5vy[i] = w.ay[i]; k5vz[i] = w.az[i];
    }

    // Stage 6
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] = scratch_px[i] + dt * (a61*k1x[i] + a62*k2x[i] + a63*k3x[i] + a64*k4x[i] + a65*k5x[i]);
        w.py[i] = scratch_py[i] + dt * (a61*k1y[i] + a62*k2y[i] + a63*k3y[i] + a64*k4y[i] + a65*k5y[i]);
        w.pz[i] = scratch_pz[i] + dt * (a61*k1z[i] + a62*k2z[i] + a63*k3z[i] + a64*k4z[i] + a65*k5z[i]);
        w.vx[i] = scratch_vx[i] + dt * (a61*k1vx[i] + a62*k2vx[i] + a63*k3vx[i] + a64*k4vx[i] + a65*k5vx[i]);
        w.vy[i] = scratch_vy[i] + dt * (a61*k1vy[i] + a62*k2vy[i] + a63*k3vy[i] + a64*k4vy[i] + a65*k5vy[i]);
        w.vz[i] = scratch_vz[i] + dt * (a61*k1vz[i] + a62*k2vz[i] + a63*k3vz[i] + a64*k4vz[i] + a65*k5vz[i]);
    }
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k6x[i] = w.vx[i];  k6y[i] = w.vy[i];  k6z[i] = w.vz[i];
        k6vx[i] = w.ax[i]; k6vy[i] = w.ay[i]; k6vz[i] = w.az[i];
    }

    // Stage 7 (FSAL - first same as last, at the 5th order solution point)
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] = scratch_px[i] + dt * (a71*k1x[i] + a73*k3x[i] + a74*k4x[i] + a75*k5x[i] + a76*k6x[i]);
        w.py[i] = scratch_py[i] + dt * (a71*k1y[i] + a73*k3y[i] + a74*k4y[i] + a75*k5y[i] + a76*k6y[i]);
        w.pz[i] = scratch_pz[i] + dt * (a71*k1z[i] + a73*k3z[i] + a74*k4z[i] + a75*k5z[i] + a76*k6z[i]);
        w.vx[i] = scratch_vx[i] + dt * (a71*k1vx[i] + a73*k3vx[i] + a74*k4vx[i] + a75*k5vx[i] + a76*k6vx[i]);
        w.vy[i] = scratch_vy[i] + dt * (a71*k1vy[i] + a73*k3vy[i] + a74*k4vy[i] + a75*k5vy[i] + a76*k6vy[i]);
        w.vz[i] = scratch_vz[i] + dt * (a71*k1vz[i] + a73*k3vz[i] + a74*k4vz[i] + a75*k5vz[i] + a76*k6vz[i]);
    }
    compute_gravity(w);
    for (std::size_t i = 0; i < n; ++i) {
        k7x[i] = w.vx[i];  k7y[i] = w.vy[i];  k7z[i] = w.vz[i];
        k7vx[i] = w.ax[i]; k7vy[i] = w.ay[i]; k7vz[i] = w.az[i];
    }

    // Compute error estimate (difference between 5th and 4th order solutions)
    real max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        real err_px = dt * std::abs(e1*k1x[i] + e3*k3x[i] + e4*k4x[i] + e5*k5x[i] + e6*k6x[i] + e7*k7x[i]);
        real err_py = dt * std::abs(e1*k1y[i] + e3*k3y[i] + e4*k4y[i] + e5*k5y[i] + e6*k6y[i] + e7*k7y[i]);
        real err_pz = dt * std::abs(e1*k1z[i] + e3*k3z[i] + e4*k4z[i] + e5*k5z[i] + e6*k6z[i] + e7*k7z[i]);
        real err_vx = dt * std::abs(e1*k1vx[i] + e3*k3vx[i] + e4*k4vx[i] + e5*k5vx[i] + e6*k6vx[i] + e7*k7vx[i]);
        real err_vy = dt * std::abs(e1*k1vy[i] + e3*k3vy[i] + e4*k4vy[i] + e5*k5vy[i] + e6*k6vy[i] + e7*k7vy[i]);
        real err_vz = dt * std::abs(e1*k1vz[i] + e3*k3vz[i] + e4*k4vz[i] + e5*k5vz[i] + e6*k6vz[i] + e7*k7vz[i]);

        // Scale by current state magnitude for relative error
        real scale_p = std::max(1.0, std::sqrt(w.px[i]*w.px[i] + w.py[i]*w.py[i] + w.pz[i]*w.pz[i]));
        real scale_v = std::max(1.0, std::sqrt(w.vx[i]*w.vx[i] + w.vy[i]*w.vy[i] + w.vz[i]*w.vz[i]));

        max_err = std::max(max_err, (err_px + err_py + err_pz) / scale_p);
        max_err = std::max(max_err, (err_vx + err_vy + err_vz) / scale_v);
    }

    // Step size control
    constexpr real safety = 0.9;
    constexpr real min_scale = 0.2;
    constexpr real max_scale = 5.0;

    real scale = safety * std::pow(tol / (max_err + 1e-30), 0.2);
    scale = std::clamp(scale, min_scale, max_scale);

    real dt_next = dt * scale;

    // If error too large, reject step and retry with smaller dt
    if (max_err > tol) {
        restore_state(w);
        return StepResult{0.0, dt_next, max_err};
    }

    // Accept step
    w.time += dt;
    return StepResult{dt, dt_next, max_err};
}

void step_leapfrog(World& w, real dt) {
    const std::size_t n = w.count;

    // Kick-Drift-Kick (KDK) leapfrog
    // Half kick
    const real half_dt = 0.5 * dt;
    for (std::size_t i = 0; i < n; ++i) {
        w.vx[i] += half_dt * w.ax[i];
        w.vy[i] += half_dt * w.ay[i];
        w.vz[i] += half_dt * w.az[i];
    }

    // Full drift
    for (std::size_t i = 0; i < n; ++i) {
        w.px[i] += dt * w.vx[i];
        w.py[i] += dt * w.vy[i];
        w.pz[i] += dt * w.vz[i];
    }

    // Recompute accelerations at new positions
    compute_gravity(w);

    // Half kick
    for (std::size_t i = 0; i < n; ++i) {
        w.vx[i] += half_dt * w.ax[i];
        w.vy[i] += half_dt * w.ay[i];
        w.vz[i] += half_dt * w.az[i];
    }

    w.time += dt;
}

}  // namespace d3x
