#include <d3x/world.hpp>
#include <d3x/types.hpp>

namespace d3x {

void World::reserve(std::size_t n) {
    px.reserve(n); py.reserve(n); pz.reserve(n);
    vx.reserve(n); vy.reserve(n); vz.reserve(n);
    ax.reserve(n); ay.reserve(n); az.reserve(n);
    mass.reserve(n);
}

std::size_t World::add_body(Vec3 pos, Vec3 vel, real m) {
    px.push_back(pos.x); py.push_back(pos.y); pz.push_back(pos.z);
    vx.push_back(vel.x); vy.push_back(vel.y); vz.push_back(vel.z);
    ax.push_back(0.0);   ay.push_back(0.0);   az.push_back(0.0);
    mass.push_back(m);
    return count++;
}

void World::clear() {
    px.clear(); py.clear(); pz.clear();
    vx.clear(); vy.clear(); vz.clear();
    ax.clear(); ay.clear(); az.clear();
    mass.clear();
    count = 0;
    time = 0.0;
}

real World::kinetic_energy() const {
    real ke = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        real v2 = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
        ke += 0.5 * mass[i] * v2;
    }
    return ke;
}

real World::potential_energy() const {
    real pe = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = i + 1; j < count; ++j) {
            real dx = px[j] - px[i];
            real dy = py[j] - py[i];
            real dz = pz[j] - pz[i];
            real r = std::sqrt(dx*dx + dy*dy + dz*dz);
            pe -= constants::G * mass[i] * mass[j] / r;
        }
    }
    return pe;
}

real World::total_energy() const {
    return kinetic_energy() + potential_energy();
}

Vec3 World::angular_momentum() const {
    Vec3 L{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < count; ++i) {
        // L = r × p = r × (m * v)
        L.x += mass[i] * (py[i] * vz[i] - pz[i] * vy[i]);
        L.y += mass[i] * (pz[i] * vx[i] - px[i] * vz[i]);
        L.z += mass[i] * (px[i] * vy[i] - py[i] * vx[i]);
    }
    return L;
}

}  // namespace d3x
