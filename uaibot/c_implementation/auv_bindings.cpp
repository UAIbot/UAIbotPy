// Keep Python/pybind headers before Eigen and project headers.
#include "dynamics_bindings_helpers.hpp"

#include "auv_physics.hpp"

namespace py = pybind11;
void bind_auv_dynamics(py::module_& m) {
  py::class_<auv::Parameters> parameters(m, "AUVParameters");
  parameters.def(py::init<>())
      .def_readwrite("dt", &auv::Parameters::dt)
      .def_readwrite("max_internal_dt", &auv::Parameters::max_internal_dt)
      .def_readwrite("velocity_gains", &auv::Parameters::velocity_gains)
      .def_readwrite("model_compensation", &auv::Parameters::model_compensation)
      .def_readwrite("measurement_noise", &auv::Parameters::measurement_noise)
      .def_readwrite("mass", &auv::Parameters::mass)
      .def_readwrite("gravity", &auv::Parameters::gravity)
      .def_readwrite("water_density", &auv::Parameters::water_density)
      .def_readwrite("displaced_volume", &auv::Parameters::displaced_volume)
      .def_readwrite("inertia_at_cg_body", &auv::Parameters::inertia_at_cg_body)
      .def_readwrite("center_of_gravity_body",
                     &auv::Parameters::center_of_gravity_body)
      .def_readwrite("center_of_buoyancy_body",
                     &auv::Parameters::center_of_buoyancy_body)
      .def_readwrite("added_mass_body_at_reference_density",
                     &auv::Parameters::added_mass_body_at_reference_density)
      .def_readwrite("linear_damping_body_at_reference_density",
                     &auv::Parameters::linear_damping_body_at_reference_density)
      .def_readwrite(
          "quadratic_damping_body_at_reference_density",
          &auv::Parameters::quadratic_damping_body_at_reference_density)
      .def_readwrite("hydrodynamic_reference_density",
                     &auv::Parameters::hydrodynamic_reference_density)
      .def_readwrite("current_world", &auv::Parameters::current_world)
      .def_readwrite("external_wrench_world",
                     &auv::Parameters::external_wrench_world)
      .def_readwrite("thruster_positions_body",
                     &auv::Parameters::thruster_positions_body)
      .def_readwrite("positive_thrust_axes_body",
                     &auv::Parameters::positive_thrust_axes_body)
      .def_readwrite("thrust_max", &auv::Parameters::thrust_max)
      .def_readwrite("thrust_min", &auv::Parameters::thrust_min)
      .def_readwrite("reaction_moment_per_newton",
                     &auv::Parameters::reaction_moment_per_newton)
      .def_readwrite("allocation_weights", &auv::Parameters::allocation_weights)
      .def_readwrite("allocation_regularization",
                     &auv::Parameters::allocation_regularization)
      .def_readwrite("allocation_max_iterations",
                     &auv::Parameters::allocation_max_iterations)
      .def_readwrite("allocation_tolerance",
                     &auv::Parameters::allocation_tolerance)
      .def_readwrite("thruster_time_constant",
                     &auv::Parameters::thruster_time_constant);
  dynamics_bindings::copyable(parameters);
  py::class_<auv::Diagnostics> diagnostics(m, "AUVDiagnostics");
  diagnostics.def(py::init<>())
      .def_readwrite("requested_wrench", &auv::Diagnostics::requested_wrench)
      .def_readwrite("allocated_wrench", &auv::Diagnostics::allocated_wrench)
      .def_readwrite("applied_wrench", &auv::Diagnostics::applied_wrench)
      .def_readwrite("allocation_residual",
                     &auv::Diagnostics::allocation_residual)
      .def_readwrite("thrust_setpoints", &auv::Diagnostics::thrust_setpoints)
      .def_readwrite("saturated_thrusters",
                     &auv::Diagnostics::saturated_thrusters)
      .def_readwrite("allocation_iterations",
                     &auv::Diagnostics::allocation_iterations)
      .def_readwrite("allocation_converged",
                     &auv::Diagnostics::allocation_converged);
  dynamics_bindings::copyable(diagnostics);
  py::class_<auv::State, dynamics::RigidBodyState> state(m, "AUVState");
  state.def(py::init<>())
      .def_readwrite("thruster_forces", &auv::State::thruster_forces)
      .def_readwrite("diagnostics", &auv::State::diagnostics);
  dynamics_bindings::copyable(state);
  m.def("auv_step",
        py::overload_cast<const Eigen::VectorXd&, const auv::State&,
                          const auv::Parameters&>(&auv::step),
        py::arg("target_twist"), py::arg("state"),
        py::arg("p") = auv::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def("auv_step_stateless",
        py::overload_cast<const Eigen::VectorXd&, const Eigen::Matrix4d&,
                          const Eigen::VectorXd&, const auv::Parameters&>(
            &auv::step),
        py::arg("target_twist"), py::arg("pose"), py::arg("current_twist"),
        py::arg("p") = auv::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def(
      "auv_step_direct",
      py::overload_cast<const Eigen::VectorXd&, const Eigen::Matrix4d&,
                        const Eigen::VectorXd&, const Eigen::VectorXd&, double>(
          &auv::step),
      py::arg("target_twist"), py::arg("pose"), py::arg("current_twist"),
      py::arg("velocity_gains"), py::arg("dt"),
      py::call_guard<py::gil_scoped_release>());
  m.def("fossen_step",
        py::overload_cast<const Eigen::VectorXd&, const auv::State&,
                          const auv::Parameters&>(&auv::step),
        py::arg("target_twist"), py::arg("state"),
        py::arg("p") = auv::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def("fossen_step_stateless",
        py::overload_cast<const Eigen::VectorXd&, const Eigen::Matrix4d&,
                          const Eigen::VectorXd&, const auv::Parameters&>(
            &auv::step),
        py::arg("target_twist"), py::arg("pose"), py::arg("current_twist"),
        py::arg("p") = auv::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def(
      "fossen_step_direct",
      py::overload_cast<const Eigen::VectorXd&, const Eigen::Matrix4d&,
                        const Eigen::VectorXd&, const Eigen::VectorXd&, double>(
          &auv::step),
      py::arg("target_twist"), py::arg("pose"), py::arg("current_twist"),
      py::arg("velocity_gains"), py::arg("dt"),
      py::call_guard<py::gil_scoped_release>());
  m.def("auv_step_open_loop", &auv::step_open_loop, py::arg("thrust_setpoints"),
        py::arg("state"), py::arg("p") = auv::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def("auv_allocation_matrix", &auv::allocation_matrix, py::arg("pose"),
        py::arg("p") = auv::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def("auv_total_mass_matrix", &auv::total_mass_matrix, py::arg("pose"),
        py::arg("p") = auv::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.def("auv_inertial_acceleration", &auv::inertial_acceleration,
        py::arg("state"), py::arg("thruster_forces"),
        py::arg("p") = auv::Parameters{},
        py::call_guard<py::gil_scoped_release>());
  m.attr("FossenParameters") = m.attr("AUVParameters");
  m.attr("FossenState") = m.attr("AUVState");
  m.attr("FossenDiagnostics") = m.attr("AUVDiagnostics");
}
