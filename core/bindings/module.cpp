#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <d3x/types.hpp>
#include <d3x/world.hpp>
#include <d3x/gravity.hpp>
#include <d3x/integrators.hpp>

namespace py = pybind11;
using namespace d3x;

PYBIND11_MODULE(_core, m) {
    m.doc() = "D3X orbital mechanics simulation core";

    // Constants submodule
    auto constants = m.def_submodule("constants", "Physical constants (SI units)");
    constants.attr("G") = d3x::constants::G;
    constants.attr("AU") = d3x::constants::AU;
    constants.attr("DAY") = d3x::constants::DAY;
    constants.attr("M_SUN") = d3x::constants::M_SUN;
    constants.attr("M_EARTH") = d3x::constants::M_EARTH;
    constants.attr("M_MOON") = d3x::constants::M_MOON;
    constants.attr("M_MARS") = d3x::constants::M_MARS;
    constants.attr("MU_SUN") = d3x::constants::MU_SUN;
    constants.attr("MU_EARTH") = d3x::constants::MU_EARTH;

    // Vec3 for convenient tuple conversion
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<real, real, real>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def(py::init([](py::tuple t) {
            if (t.size() != 3) throw std::runtime_error("Vec3 requires 3 elements");
            return Vec3(t[0].cast<real>(), t[1].cast<real>(), t[2].cast<real>());
        }))
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("magnitude", &Vec3::magnitude)
        .def("__repr__", [](const Vec3& v) {
            return "Vec3(" + std::to_string(v.x) + ", " +
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

    // Implicit conversion from tuple to Vec3
    py::implicitly_convertible<py::tuple, Vec3>();

    // StepResult for adaptive integrators
    py::class_<StepResult>(m, "StepResult")
        .def_readonly("dt_used", &StepResult::dt_used)
        .def_readonly("dt_next", &StepResult::dt_next)
        .def_readonly("error_estimate", &StepResult::error_estimate)
        .def("__repr__", [](const StepResult& r) {
            return "StepResult(dt_used=" + std::to_string(r.dt_used) +
                   ", dt_next=" + std::to_string(r.dt_next) +
                   ", error=" + std::to_string(r.error_estimate) + ")";
        });

    // World - main simulation container
    py::class_<World>(m, "World")
        .def(py::init<>())
        .def("reserve", &World::reserve, py::arg("n"),
             "Pre-allocate memory for n bodies")
        .def("add_body", &World::add_body,
             py::arg("pos"), py::arg("vel"), py::arg("mass"),
             "Add a body with position [m], velocity [m/s], and mass [kg]. Returns body index.")
        .def("clear", &World::clear,
             "Remove all bodies and reset time")
        .def_readonly("count", &World::count,
             "Number of bodies in the simulation")
        .def_readwrite("time", &World::time,
             "Current simulation time [s]")

        // Zero-copy numpy array views
        .def_property_readonly("px", [](World& w) {
            return py::array_t<real>({w.count}, {sizeof(real)}, w.px.data(), py::cast(&w));
        }, "Position x-components [m] (numpy view)")
        .def_property_readonly("py_", [](World& w) {
            return py::array_t<real>({w.count}, {sizeof(real)}, w.py.data(), py::cast(&w));
        }, "Position y-components [m] (numpy view, named py_ to avoid collision)")
        .def_property_readonly("pz", [](World& w) {
            return py::array_t<real>({w.count}, {sizeof(real)}, w.pz.data(), py::cast(&w));
        }, "Position z-components [m] (numpy view)")
        .def_property_readonly("vx", [](World& w) {
            return py::array_t<real>({w.count}, {sizeof(real)}, w.vx.data(), py::cast(&w));
        }, "Velocity x-components [m/s] (numpy view)")
        .def_property_readonly("vy", [](World& w) {
            return py::array_t<real>({w.count}, {sizeof(real)}, w.vy.data(), py::cast(&w));
        }, "Velocity y-components [m/s] (numpy view)")
        .def_property_readonly("vz", [](World& w) {
            return py::array_t<real>({w.count}, {sizeof(real)}, w.vz.data(), py::cast(&w));
        }, "Velocity z-components [m/s] (numpy view)")
        .def_property_readonly("mass", [](World& w) {
            return py::array_t<real>({w.count}, {sizeof(real)}, w.mass.data(), py::cast(&w));
        }, "Masses [kg] (numpy view)")

        // Energy and momentum
        .def("kinetic_energy", &World::kinetic_energy,
             "Total kinetic energy [J]")
        .def("potential_energy", &World::potential_energy,
             "Total gravitational potential energy [J]")
        .def("total_energy", &World::total_energy,
             "Total mechanical energy [J]")
        .def("angular_momentum", &World::angular_momentum,
             "Total angular momentum vector [kg·m²/s]");

    // Gravity computation
    m.def("compute_gravity", py::overload_cast<World&>(&compute_gravity),
          py::arg("world"),
          "Compute gravitational accelerations for all bodies");
    m.def("compute_gravity", py::overload_cast<World&, real>(&compute_gravity),
          py::arg("world"), py::arg("softening"),
          "Compute gravitational accelerations with softening parameter");

    // Integrators
    m.def("step_rk4", &step_rk4,
          py::arg("world"), py::arg("dt"),
          "Advance simulation by dt seconds using 4th-order Runge-Kutta");
    m.def("step_dopri54", &step_dopri54,
          py::arg("world"), py::arg("dt"), py::arg("tol") = 1e-9,
          "Advance simulation using adaptive Dormand-Prince 5(4) method");
    m.def("step_leapfrog", &step_leapfrog,
          py::arg("world"), py::arg("dt"),
          "Advance simulation using symplectic leapfrog (requires pre-computed accelerations)");
}
