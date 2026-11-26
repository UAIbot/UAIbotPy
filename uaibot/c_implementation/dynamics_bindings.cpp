// Keep Python/pybind headers before Eigen and project headers.
#include "dynamics_bindings_helpers.hpp"

#include "dynamics.hpp"

namespace py = pybind11;
void bind_dynamics(py::module_& m) {
  py::class_<dynamics::MeasurementNoiseParameters> noise(
      m, "MeasurementNoiseParameters");
  noise.def(py::init<>())
      .def_readwrite("enabled", &dynamics::MeasurementNoiseParameters::enabled)
      .def_readwrite("white_variances",
                     &dynamics::MeasurementNoiseParameters::white_variances)
      .def_readwrite("colored_variances",
                     &dynamics::MeasurementNoiseParameters::colored_variances)
      .def_readwrite("correlation_times",
                     &dynamics::MeasurementNoiseParameters::correlation_times)
      .def_readwrite("sample_period",
                     &dynamics::MeasurementNoiseParameters::sample_period)
      .def_readwrite("seed", &dynamics::MeasurementNoiseParameters::seed)
      .def_readwrite(
          "stationary_initialization",
          &dynamics::MeasurementNoiseParameters::stationary_initialization);
  dynamics_bindings::copyable(noise);
  py::class_<dynamics::Measurement> measurement(m, "RigidBodyMeasurement");
  measurement.def(py::init<>())
      .def_readwrite("pose", &dynamics::Measurement::pose)
      .def_readwrite("twist", &dynamics::Measurement::twist)
      .def_readwrite("sample_time", &dynamics::Measurement::sample_time)
      .def_readwrite("valid", &dynamics::Measurement::valid);
  dynamics_bindings::copyable(measurement);
  py::class_<dynamics::RigidBodyState> state(m, "RigidBodyState");
  state.def(py::init<>())
      .def_readwrite("pose", &dynamics::RigidBodyState::pose)
      .def_readwrite("twist", &dynamics::RigidBodyState::twist)
      .def_readwrite("time", &dynamics::RigidBodyState::time)
      .def_readwrite("measurement", &dynamics::RigidBodyState::measurement);
  dynamics_bindings::copyable(state);
  state.def("reset_measurement_noise", [](dynamics::RigidBodyState& self) {
    self.measurement_noise_state.reset();
    self.measurement = dynamics::Measurement{};
  });
  m.attr("FossenMeasurement") = m.attr("RigidBodyMeasurement");
  m.attr("FossenMeasurementNoiseParameters") =
      m.attr("MeasurementNoiseParameters");
}
