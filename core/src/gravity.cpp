#include <d3x/gravity.hpp>
#include <cmath>
#include <algorithm>

namespace d3x {

void compute_gravity(World& w) {
    compute_gravity(w, 0.0);
}

void compute_gravity(World& w, real softening) {
    const std::size_t n = w.count;
    const real eps2 = softening * softening;

    // Zero accelerations
    std::fill(w.ax.begin(), w.ax.end(), 0.0);
    std::fill(w.ay.begin(), w.ay.end(), 0.0);
    std::fill(w.az.begin(), w.az.end(), 0.0);

    // O(n²) pairwise - exploit Newton's 3rd law (halves computation)
    for (std::size_t i = 0; i < n; ++i) {
        const real pxi = w.px[i];
        const real pyi = w.py[i];
        const real pzi = w.pz[i];
        const real mi = w.mass[i];

        real axi = 0.0, ayi = 0.0, azi = 0.0;

        for (std::size_t j = i + 1; j < n; ++j) {
            const real dx = w.px[j] - pxi;
            const real dy = w.py[j] - pyi;
            const real dz = w.pz[j] - pzi;

            const real dist2 = dx*dx + dy*dy + dz*dz + eps2;
            const real dist = std::sqrt(dist2);
            const real inv_dist3 = 1.0 / (dist2 * dist);

            // Force direction scaled by G / r³
            const real fx = constants::G * dx * inv_dist3;
            const real fy = constants::G * dy * inv_dist3;
            const real fz = constants::G * dz * inv_dist3;

            // a = F/m, but F = G*m1*m2/r², so a1 = G*m2/r² * r_hat
            const real mj = w.mass[j];

            axi += fx * mj;
            ayi += fy * mj;
            azi += fz * mj;

            // Newton's 3rd law: equal and opposite
            w.ax[j] -= fx * mi;
            w.ay[j] -= fy * mi;
            w.az[j] -= fz * mi;
        }

        w.ax[i] += axi;
        w.ay[i] += ayi;
        w.az[i] += azi;
    }
}

}  // namespace d3x
