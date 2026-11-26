// Keep Python/pybind headers before Eigen and project headers.
#include "dynamics_bindings_helpers.hpp"

#include "uav_physics.hpp"

namespace py = pybind11;
void bind_uav_dynamics(py::module_& m) {
  py::enum_<uav::VelocityControl>(m, "UAVVelocityControl")
      .value("WrenchProjection", uav::VelocityControl::WrenchProjection)
      .value("ThrustVectoring", uav::VelocityControl::ThrustVectoring);
  py::class_<uav::Parameters> parameters(m, "UAVParameters");
  parameters.def(py::init<>())
      .def_readwrite("dt", &uav::Parameters::dt)
      .def_readwrite("max_internal_dt", &uav::Parameters::max_internal_dt)
      .def_readwrite("velocity_gains", &uav::Parameters::velocity_gains)
      .def_readwrite("model_compensation", &uav::Parameters::model_compensation)
      .def_readwrite("measurement_noise", &uav::Parameters::measurement_noise)
      .def_readwrite("mass", &uav::Parameters::mass)
      .def_readwrite("gravity", &uav::Parameters::gravity)
      .def_readwrite("inertia_at_cg_body", &uav::Parameters::inertia_at_cg_body)
      .def_readwrite("center_of_gravity_body",
                     &uav::Parameters::center_of_gravity_body)
      .def_readwrite("aerodynamic_center_body",
                     &uav::Parameters::aerodynamic_center_body)
      .def_readwrite("air_density", &uav::Parameters::air_density)
      .def_readwrite("aerodynamic_reference_density",
                     &uav::Parameters::aerodynamic_reference_density)
      .def_readwrite("linear_drag_body_at_reference_density",
                     &uav::Parameters::linear_drag_body_at_reference_density)
      .def_readwrite("quadratic_drag_body_at_reference_density",
                     &uav::Parameters::quadratic_drag_body_at_reference_density)
      .def_readwrite("wind_world", &uav::Parameters::wind_world)
      .def_readwrite("external_wrench_world",
                     &uav::Parameters::external_wrench_world)
      .def_readwrite("actuation_matrix_body",
                     &uav::Parameters::actuation_matrix_body)
      .def_readwrite("actuator_min", &uav::Parameters::actuator_min)
      .def_readwrite("actuator_max", &uav::Parameters::actuator_max)
      .def_readwrite("actuator_time_constants",
                     &uav::Parameters::actuator_time_constants)
      .def_readwrite("allocation_weights", &uav::Parameters::allocation_weights)
      .def_readwrite("allocation_regularization",
                     &uav::Parameters::allocation_regularization)
      .def_readwrite("allocation_max_iterations",
                     &uav::Parameters::allocation_max_iterations)
      .def_readwrite("allocation_tolerance",
                     &uav::Parameters::allocation_tolerance)
      .def_readwrite("velocity_control", &uav::Parameters::velocity_control)
      .def_readwrite("attitude_gains", &uav::Parameters::attitude_gains)
      .def_readwrite("max_tilt_angle", &uav::Parameters::max_tilt_angle);
  dynamics_bindings::copyable(parameters);
  py::class_<uav::Diagnostics> diagnostics(m, "UAVDiagnostics");
  diagnostics.def(py::init<>())
      .def_readwrite("requested_wrench", &uav::Diagnostics::requested_wrench)
      .def_readwrite("allocated_wrench", &uav::Diagnostics::allocated_wrench)
      .def_readwrite("applied_wrench", &uav::Diagnostics::applied_wrench)
      .def_readwrite("allocation_residual",
                     &uav::Diagnostics::allocation_residual)
      .def_readwrite("actuator_setpoints",
                     &uav::Diagnostics::actuator_setpoints)
      .def_readwrite("saturated_actuators",
                     &uav::Diagnostics::saturated_actuators)
      .def_readwrite("allocation_iterations",
                     &uav::Diagnostics::allocation_iterations)
      .def_readwrite("allocation_converged",
                     &uav::Diagnostics::allocation_converged);
  dynamics_bindings::copyable(diagnostics);
  py::class_<uav::State, dynamics::RigidBodyState> state(m, "UAVState");
  state.def(py::init<>())
      .def_readwrite("actuator_outputs", &uav::State::actuator_outputs)
      .def_readwrite("diagnostics", &uav::State::diagnostics);
  dynamics_bindings::copyable(state);
  py::class_<uav::Rotor> rotor(m, "UAVRotor");
  rotor.def(py::init<>())
      .def_readwrite("position_body", &uav::Rotor::position_body)
      .def_readwrite("thrust_axis_body", &uav::Rotor::thrust_axis_body)
      .def_readwrite("reaction_moment_per_newton",
                     &uav::Rotor::reaction_moment_per_newton);
  dynamics_bindings::copyable(rotor);
  m.def("uav_step",
        py::overload_cast<const Eigen::VectorXd&, const uav::State&,
                          const uav::Parameters&>(&uav::step),
        py::arg("target_twist"), py::arg("state"),
        py::arg("p") = uav::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def("uav_step_stateless",
        py::overload_cast<const Eigen::VectorXd&, const Eigen::Matrix4d&,
                          const Eigen::VectorXd&, const uav::Parameters&>(
            &uav::step),
        py::arg("target_twist"), py::arg("pose"), py::arg("current_twist"),
        py::arg("p") = uav::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def(
      "uav_step_direct",
      py::overload_cast<const Eigen::VectorXd&, const Eigen::Matrix4d&,
                        const Eigen::VectorXd&, const Eigen::VectorXd&, double>(
          &uav::step),
      py::arg("target_twist"), py::arg("pose"), py::arg("current_twist"),
      py::arg("velocity_gains"), py::arg("dt"),
      py::call_guard<py::gil_scoped_release>());
  m.def("uav_step_open_loop", &uav::step_open_loop,
        py::arg("actuator_setpoints"), py::arg("state"),
        py::arg("p") = uav::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def("uav_step_wrench", &uav::step_wrench, py::arg("wrench_world"),
        py::arg("state"), py::arg("p") = uav::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def("uav_configure_actuators", &uav::configure_actuators, py::arg("p"),
        py::arg("matrix_body"), py::arg("lower"), py::arg("upper"),
        py::arg("time_constant") = 0.,
        py::call_guard<py::gil_scoped_release>());
  m.def("uav_rotor_allocation_matrix", &uav::rotor_allocation_matrix,
        py::arg("rotors"), py::call_guard<py::gil_scoped_release>());
  m.def("uav_allocation_matrix", &uav::allocation_matrix, py::arg("pose"),
        py::arg("p") = uav::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def("uav_total_mass_matrix", &uav::total_mass_matrix, py::arg("pose"),
        py::arg("p") = uav::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def("uav_inertial_acceleration", &uav::inertial_acceleration,
        py::arg("state"), py::arg("actuator_outputs"),
        py::arg("p") = uav::Parameters{},
        py::call_guard<py::gil_scoped_release>());
}
