// Keep Python/pybind headers before Eigen and project headers.
#include "dynamics_bindings_helpers.hpp"

#include "declarations.h"
#include "manipulator_dh.hpp"
#include "manipulator_physics.hpp"

namespace py = pybind11;
void bind_manipulator_dynamics(py::module_& m) {
  py::enum_<manipulator::JointType>(m, "ManipulatorJointType")
      .value("Revolute", manipulator::JointType::Revolute)
      .value("Prismatic", manipulator::JointType::Prismatic);
  py::enum_<manipulator::InertiaReference>(m, "ManipulatorInertiaReference")
      .value("CenterOfMass", manipulator::InertiaReference::CenterOfMass)
      .value("DHOrigin", manipulator::InertiaReference::DHOrigin);
  py::class_<manipulator::Link> link(m, "ManipulatorLink");
  link.def(py::init<>())
      .def_readwrite("joint_type", &manipulator::Link::joint_type)
      .def_readwrite("parent_to_joint", &manipulator::Link::parent_to_joint)
      .def_readwrite("axis_joint", &manipulator::Link::axis_joint)
      .def_readwrite("mass", &manipulator::Link::mass)
      .def_readwrite("center_of_mass_body",
                     &manipulator::Link::center_of_mass_body)
      .def_readwrite("inertia_at_com_body",
                     &manipulator::Link::inertia_at_com_body)
      .def_readwrite("viscous_friction", &manipulator::Link::viscous_friction)
      .def_readwrite("coulomb_friction", &manipulator::Link::coulomb_friction)
      .def_readwrite("friction_speed", &manipulator::Link::friction_speed)
      .def_readwrite("reflected_motor_inertia",
                     &manipulator::Link::reflected_motor_inertia);
  dynamics_bindings::copyable(link);
  py::class_<manipulator::JointMeasurementNoiseParameters> noise(
      m, "JointMeasurementNoiseParameters");
  noise.def(py::init<>())
      .def_readwrite("enabled",
                     &manipulator::JointMeasurementNoiseParameters::enabled)
      .def_readwrite(
          "white_variances",
          &manipulator::JointMeasurementNoiseParameters::white_variances)
      .def_readwrite(
          "colored_variances",
          &manipulator::JointMeasurementNoiseParameters::colored_variances)
      .def_readwrite(
          "correlation_times",
          &manipulator::JointMeasurementNoiseParameters::correlation_times)
      .def_readwrite(
          "sample_period",
          &manipulator::JointMeasurementNoiseParameters::sample_period)
      .def_readwrite("seed",
                     &manipulator::JointMeasurementNoiseParameters::seed)
      .def_readwrite("stationary_initialization",
                     &manipulator::JointMeasurementNoiseParameters::
                         stationary_initialization);
  dynamics_bindings::copyable(noise);
  py::class_<manipulator::Parameters> parameters(m, "ManipulatorParameters");
  parameters.def(py::init<>())
      .def_readwrite("links", &manipulator::Parameters::links)
      .def_readwrite("gravity_base", &manipulator::Parameters::gravity_base)
      .def_readwrite("dt", &manipulator::Parameters::dt)
      .def_readwrite("max_internal_dt",
                     &manipulator::Parameters::max_internal_dt)
      .def_readwrite("external_wrenches_body",
                     &manipulator::Parameters::external_wrenches_body)
      .def_readwrite("actuation_matrix",
                     &manipulator::Parameters::actuation_matrix)
      .def_readwrite("actuator_min", &manipulator::Parameters::actuator_min)
      .def_readwrite("actuator_max", &manipulator::Parameters::actuator_max)
      .def_readwrite("actuator_time_constants",
                     &manipulator::Parameters::actuator_time_constants)
      .def_readwrite("allocation_weights",
                     &manipulator::Parameters::allocation_weights)
      .def_readwrite("allocation_regularization",
                     &manipulator::Parameters::allocation_regularization)
      .def_readwrite("allocation_max_iterations",
                     &manipulator::Parameters::allocation_max_iterations)
      .def_readwrite("allocation_tolerance",
                     &manipulator::Parameters::allocation_tolerance)
      .def_readwrite("velocity_gains", &manipulator::Parameters::velocity_gains)
      .def_readwrite("model_compensation",
                     &manipulator::Parameters::model_compensation)
      .def_readwrite("measurement_noise",
                     &manipulator::Parameters::measurement_noise);
  dynamics_bindings::copyable(parameters);
  py::class_<manipulator::JointMeasurement> measurement(m, "JointMeasurement");
  measurement.def(py::init<>())
      .def_readwrite("q", &manipulator::JointMeasurement::q)
      .def_readwrite("qd", &manipulator::JointMeasurement::qd)
      .def_readwrite("sample_time", &manipulator::JointMeasurement::sample_time)
      .def_readwrite("valid", &manipulator::JointMeasurement::valid);
  dynamics_bindings::copyable(measurement);
  py::class_<manipulator::Diagnostics> diagnostics(m, "ManipulatorDiagnostics");
  diagnostics.def(py::init<>())
      .def_readwrite("requested_effort",
                     &manipulator::Diagnostics::requested_effort)
      .def_readwrite("allocated_effort",
                     &manipulator::Diagnostics::allocated_effort)
      .def_readwrite("applied_effort",
                     &manipulator::Diagnostics::applied_effort)
      .def_readwrite("allocation_residual",
                     &manipulator::Diagnostics::allocation_residual)
      .def_readwrite("actuator_setpoints",
                     &manipulator::Diagnostics::actuator_setpoints)
      .def_readwrite("saturated_actuators",
                     &manipulator::Diagnostics::saturated_actuators)
      .def_readwrite("allocation_iterations",
                     &manipulator::Diagnostics::allocation_iterations)
      .def_readwrite("allocation_converged",
                     &manipulator::Diagnostics::allocation_converged);
  dynamics_bindings::copyable(diagnostics);
  py::class_<manipulator::State> state(m, "ManipulatorState");
  state.def(py::init<>())
      .def_readwrite("q", &manipulator::State::q)
      .def_readwrite("qd", &manipulator::State::qd)
      .def_readwrite("actuator_outputs", &manipulator::State::actuator_outputs)
      .def_readwrite("time", &manipulator::State::time)
      .def_readwrite("measurement", &manipulator::State::measurement)
      .def_readwrite("diagnostics", &manipulator::State::diagnostics);
  dynamics_bindings::copyable(state);
  state.def("reset_measurement_noise", [](manipulator::State& self) {
    self.measurement_noise_state.reset();
    self.measurement = manipulator::JointMeasurement{};
  });
  m.def("manipulator_make_state", &manipulator::make_state, py::arg("p"),
        py::call_guard<py::gil_scoped_release>());
  m.def("manipulator_mass_matrix", &manipulator::mass_matrix, py::arg("q"),
        py::arg("p"), py::call_guard<py::gil_scoped_release>());
  m.def("manipulator_bias_forces", &manipulator::bias_forces, py::arg("q"),
        py::arg("qd"), py::arg("p"), py::call_guard<py::gil_scoped_release>());
  m.def("manipulator_inverse_dynamics", &manipulator::inverse_dynamics,
        py::arg("q"), py::arg("qd"), py::arg("qdd"), py::arg("p"),
        py::call_guard<py::gil_scoped_release>());
  m.def("manipulator_forward_kinematics", &manipulator::forward_kinematics,
        py::arg("q"), py::arg("p"), py::call_guard<py::gil_scoped_release>());
  m.def("manipulator_forward_dynamics", &manipulator::forward_dynamics,
        py::arg("state"), py::arg("outputs"), py::arg("p"),
        py::call_guard<py::gil_scoped_release>());
  m.def("manipulator_step_open_loop", &manipulator::step_open_loop,
        py::arg("actuator_setpoints"), py::arg("state"), py::arg("p"),
        py::call_guard<py::gil_scoped_release>());
  m.def("manipulator_step_effort", &manipulator::step_effort,
        py::arg("joint_effort"), py::arg("state"), py::arg("p"),
        py::call_guard<py::gil_scoped_release>());
  m.def("manipulator_step", &manipulator::step,
        py::arg("target_joint_velocity"), py::arg("state"), py::arg("p"),
        py::call_guard<py::gil_scoped_release>());
  m.def("manipulator_configure_actuators", &manipulator::configure_actuators,
        py::arg("p"), py::arg("matrix"), py::arg("lower"), py::arg("upper"),
        py::arg("time_constant") = 0.,
        py::call_guard<py::gil_scoped_release>());
  m.def("manipulator_parameters_from_dh", &manipulator::parameters_from_dh,
        py::arg("dh_table"), py::arg("masses"), py::arg("com_dh"),
        py::arg("inertias_dh"),
        py::arg("base_to_dh0") = Eigen::Matrix4d::Identity().eval(),
        py::arg("gravity_base") = Eigen::Vector3d(0., 0., -9.81),
        py::arg("inertia_reference") =
            manipulator::InertiaReference::CenterOfMass,
        py::call_guard<py::gil_scoped_release>());
  // Add the conversion to the existing kinematic type without changing its
  // layout.
  auto legacy = py::reinterpret_borrow<py::class_<Manipulator>>(
      m.attr("CPP_Manipulator"));
  legacy.def(
      "dynamics_parameters",
      [](const Manipulator& self, const Eigen::VectorXd& masses,
         const Eigen::MatrixXd& com_dh,
         const std::vector<Eigen::Matrix3d>& inertias_dh,
         const Eigen::Vector3d& gravity_base,
         manipulator::InertiaReference reference) {
        const int n = self.no_links;
        dynamics::detail::require(
            n > 0 && n <= 128 &&
                self.theta.size() == static_cast<std::size_t>(n) &&
                self.dh_d.size() == static_cast<std::size_t>(n) &&
                self.alpha.size() == static_cast<std::size_t>(n) &&
                self.dh_a.size() == static_cast<std::size_t>(n) &&
                self.joint_type.size() == static_cast<std::size_t>(n),
            "Manipulator: invalid legacy DH model");
        Eigen::MatrixXd table(n, 5);
        for (int i = 0; i < n; ++i)
          table.row(i) << self.theta[i], self.dh_d[i], self.alpha[i],
              self.dh_a[i], self.joint_type[i];
        return manipulator::parameters_from_dh(
            table, masses, com_dh, inertias_dh,
            self.htm_world_to_dh0.cast<double>(), gravity_base, reference);
      },
      py::arg("masses"), py::arg("com_dh"), py::arg("inertias_dh"),
      py::arg("gravity_base") = Eigen::Vector3d(0., 0., -9.81),
      py::arg("inertia_reference") = manipulator::InertiaReference::DHOrigin,
      py::call_guard<py::gil_scoped_release>());
}
