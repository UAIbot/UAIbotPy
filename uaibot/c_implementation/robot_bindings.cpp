// Keep Python/pybind headers before Eigen and project headers.
#include "dynamics_bindings_helpers.hpp"

#include "robot.hpp"

namespace py = pybind11;
namespace {
template <class Robot>
void bind_robot(py::module_& m, const char* name) {
  using Parameters = typename Robot::Parameters;
  using State = typename Robot::State;
  py::class_<Robot> cls(m, name);
  cls.def(py::init<Parameters>(), py::arg("parameters"))
      .def(py::init<Parameters, State>(), py::arg("parameters"),
           py::arg("state"))
      .def_readwrite("parameters", &Robot::parameters)
      .def_readwrite("state", &Robot::state)
      // Snapshots: holding a prior return value must not alias later steps.
      .def(
          "step",
          [](Robot& self, const Eigen::VectorXd& u) {
            return State(self.step(u));
          },
          py::arg("target_velocity"), py::call_guard<py::gil_scoped_release>())
      .def(
          "step_open_loop",
          [](Robot& self, const Eigen::VectorXd& u) {
            return State(self.step_open_loop(u));
          },
          py::arg("actuator_setpoints"),
          py::call_guard<py::gil_scoped_release>());
  dynamics_bindings::copyable(cls);
}
}  // namespace
void bind_dynamics_robots(py::module_& m) {
  bind_robot<dynamics::AUV>(m, "AUVDynamics");
  bind_robot<dynamics::UAV>(m, "UAVDynamics");
  bind_robot<dynamics::Manipulator>(m, "ManipulatorDynamics");
}
