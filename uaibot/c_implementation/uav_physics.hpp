#pragma once
#include "dynamics.hpp"
// Configurable multirotor/VTOL rigid-body dynamics. C++17, Eigen >= 3.3.
// SI units. World z up; body x forward, y left, z up. See README.md.
// Pose and geometric twist are INERTIAL, at the chosen vehicle origin.
// Measurement-noise algorithm retained from the supplied BlueROV2 reference.

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uav {
using dynamics::Mat6;
using dynamics::Measurement;
using dynamics::MeasurementNoiseParameters;
using dynamics::six;
using dynamics::skew;
using dynamics::Vec12;
using dynamics::Vec6;
using AllocationMatrix = Eigen::Matrix<double, 6, Eigen::Dynamic>;

// The plant accepts ANY constant 6-by-N body-frame actuation matrix.
// WrenchProjection: generic instantaneous wrench/velocity projection.
// ThrustVectoring: cascaded multirotor velocity/attitude controller, +body-z
// thrust only, pose origin at CG, world yaw-rate command. Roll/pitch rate
// targets must be zero because tilt is used to control translation.
enum class VelocityControl { WrenchProjection, ThrustVectoring };

struct Parameters {
  double dt = 0.01;
  double max_internal_dt = 0.002;
  // INERTIAL velocity error -> acceleration, gains in 1/s.
  Eigen::VectorXd velocity_gains = six(2., 2., 2., 8., 8., 6.);
  bool model_compensation = true;  // gravity, drag, rigid-body coupling
  MeasurementNoiseParameters measurement_noise;

  // Illustrative 1.5-kg vehicle; replace with identified/measured properties.
  double mass = 1.5;      // kg
  double gravity = 9.81;  // m/s^2, world -z
  Eigen::Matrix3d inertia_at_cg_body =
      Eigen::Vector3d(0.03, 0.03, 0.05).asDiagonal();  // kg m^2
  Eigen::Vector3d center_of_gravity_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d aerodynamic_center_body = Eigen::Vector3d::Zero();
  double air_density = 1.225;  // kg/m^3
  double aerodynamic_reference_density = 1.225;
  // Drag at aerodynamic_center_body: -(linear +
  // quadratic*abs(relative))*relative. Translation: N/(m/s), N/(m/s)^2.
  // Rotation: Nm/(rad/s), Nm/(rad/s)^2. Density-scaled empirical coefficients;
  // quadratic values already include the 0.5*rho_ref*Cd*A factor. These
  // defaults are NOT calibrated airframe data.
  Vec6 linear_drag_body_at_reference_density =
      six(.08, .08, .12, .001, .001, .002);
  Vec6 quadratic_drag_body_at_reference_density =
      six(.06, .06, .10, .002, .002, .004);
  Eigen::Vector3d wind_world =
      Eigen::Vector3d::Zero();  // uniform, constant m/s
  // About vehicle origin, in INERTIAL axes; excluded from controller
  // compensation.
  Vec6 external_wrench_world = Vec6::Zero();

  // w_body = B * actuator_outputs, rows [Fx,Fy,Fz,Tx,Ty,Tz], about pose origin.
  // B is CONSTANT in BODY axes; the simulator rotates it into world axes.
  // Default: ideal fully actuated force/torque channels (N and Nm).
  // Rotor columns typically map thrust in N to force and torque (lever arms in
  // m).
  AllocationMatrix actuation_matrix_body = Mat6::Identity();
  Eigen::VectorXd actuator_min = six(-30., -30., -30., -3., -3., -3.);
  Eigen::VectorXd actuator_max = six(30., 30., 30., 3., 3., 3.);
  // First-order lag in OUTPUT (e.g. thrust), seconds per channel. Zero = ideal.
  Eigen::VectorXd actuator_time_constants = Eigen::VectorXd::Zero(6);
  Vec6 allocation_weights = six(1., 1., 1., 4., 4., 4.);  // INERTIAL components
  double allocation_regularization = 1.e-8;
  int allocation_max_iterations = 200;
  double allocation_tolerance = 1.e-9;

  VelocityControl velocity_control = VelocityControl::WrenchProjection;
  // ThrustVectoring only: shortest-quaternion attitude error -> angular
  // acceleration, in 1/s^2. Angular entries of velocity_gains provide rate
  // damping.
  Eigen::Vector3d attitude_gains = Eigen::Vector3d(36., 36., 16.);
  double max_tilt_angle = 0.70;  // radians, (0, pi/2)
};

struct Diagnostics {
  // World-axis wrenches, about vehicle origin. Requested is the last command;
  // allocated/applied use the RETURNED true pose. Residual includes saturation,
  // underactuation and any measured/true attitude discrepancy.
  Vec6 requested_wrench = Vec6::Zero();
  Vec6 allocated_wrench = Vec6::Zero();
  Vec6 applied_wrench = Vec6::Zero();
  Vec6 allocation_residual = Vec6::Zero();
  Eigen::VectorXd actuator_setpoints;
  int saturated_actuators = 0;
  int allocation_iterations = 0;
  bool allocation_converged = true;
};

namespace detail {
using dynamics::detail::clock_tolerance;
using dynamics::detail::Derivative;
using dynamics::detail::exact_measurement;
using dynamics::detail::InternalState;
using dynamics::detail::MeasurementNoiseMemory;
using dynamics::detail::orientation_map;
using dynamics::detail::require;
using dynamics::detail::same_noise_settings;
using dynamics::detail::sample_if_due;
using dynamics::detail::shifted;
using dynamics::detail::to_internal;
using dynamics::detail::transport;
using dynamics::detail::validate_pose;
}  // namespace detail

// Shared rigid-body pose/twist, sensor clock and random memory live in the
// base. Retain the entire State between calls for actuator and sensor
// continuity.
struct State : dynamics::RigidBodyState {
  Eigen::VectorXd actuator_outputs;  // empty initializes to zero outputs
  Diagnostics diagnostics;
};

namespace detail {
inline void validate_state(const State& s) {
  validate_pose(s.pose);
  require(std::isfinite(s.time) && s.time >= 0., "invalid state time");
  require(s.twist.size() == 6 && s.twist.allFinite(),
          "actual twist must have six finite entries");
  require(s.actuator_outputs.allFinite(), "nonfinite actuator state");
}
struct Model {
  Mat6 mass = Mat6::Zero();
  Mat6 inverse_mass = Mat6::Zero();
  AllocationMatrix allocation;
  Vec6 linear_drag, quadratic_drag;
  double max_step = 0.;
};

inline Model prepare(const Parameters& p) {
  require(std::isfinite(p.dt) && p.dt > 0., "dt must be positive and finite");
  require(std::isfinite(p.max_internal_dt) && p.max_internal_dt > 0.,
          "max_internal_dt must be positive and finite");
  require(std::ceil(p.dt / p.max_internal_dt) <= 1.e6,
          "too many integration steps");
  if (p.measurement_noise.enabled) {
    const auto& n = p.measurement_noise;
    require(n.white_variances.allFinite() &&
                (n.white_variances.array() >= 0.).all() &&
                n.colored_variances.allFinite() &&
                (n.colored_variances.array() >= 0.).all(),
            "noise variances must be finite and nonnegative");
    require(n.correlation_times.allFinite() &&
                (n.correlation_times.array() > 0.).all(),
            "noise correlation times must be finite and positive");
    require(std::isfinite(n.sample_period) && n.sample_period >= 0.,
            "noise sample_period must be finite and nonnegative");
    require(n.sample_period == 0. || std::ceil(p.dt / n.sample_period) <= 1.e6,
            "too many measurement updates per call");
  }
  require(p.velocity_gains.size() == 6 && p.velocity_gains.allFinite() &&
              (p.velocity_gains.array() >= 0.).all(),
          "invalid velocity gains");
  require(std::isfinite(p.mass) && p.mass > 0.,
          "mass must be positive and finite");
  require(std::isfinite(p.gravity) && p.gravity >= 0., "invalid gravity");
  require(std::isfinite(p.air_density) && p.air_density >= 0. &&
              std::isfinite(p.aerodynamic_reference_density) &&
              p.aerodynamic_reference_density > 0.,
          "invalid air/reference density");
  require(p.center_of_gravity_body.allFinite() &&
              p.aerodynamic_center_body.allFinite() &&
              p.wind_world.allFinite() && p.external_wrench_world.allFinite(),
          "nonfinite physical vector");
  require(p.inertia_at_cg_body.allFinite() &&
              p.inertia_at_cg_body.isApprox(p.inertia_at_cg_body.transpose(),
                                            1.e-12),
          "CG inertia must be symmetric");
  Eigen::LLT<Eigen::Matrix3d> inertia_llt(p.inertia_at_cg_body);
  require(inertia_llt.info() == Eigen::Success,
          "CG inertia must be positive definite");
  // A physical inertia also obeys the principal-moment triangle inequalities.
  const Eigen::Matrix3d second_moment =
      .5 * p.inertia_at_cg_body.trace() * Eigen::Matrix3d::Identity() -
      p.inertia_at_cg_body;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> inertia_eig(second_moment);
  require(inertia_eig.info() == Eigen::Success &&
              inertia_eig.eigenvalues().minCoeff() >=
                  -1.e-10 * p.inertia_at_cg_body.norm(),
          "CG inertia violates principal-moment triangle inequalities");
  require(p.linear_drag_body_at_reference_density.allFinite() &&
              p.quadratic_drag_body_at_reference_density.allFinite() &&
              (p.linear_drag_body_at_reference_density.array() >= 0.).all() &&
              (p.quadratic_drag_body_at_reference_density.array() >= 0.).all(),
          "drag coefficients must be finite and nonnegative");
  const Eigen::Index n = p.actuation_matrix_body.cols();
  require(n > 0 && n <= 256 && p.actuation_matrix_body.allFinite(),
          "actuation matrix must have 1..256 finite columns");
  require(p.actuator_min.size() == n && p.actuator_max.size() == n &&
              p.actuator_time_constants.size() == n,
          "actuator parameter dimensions must match B");
  require(p.actuator_min.allFinite() && p.actuator_max.allFinite() &&
              (p.actuator_min.array() <= 0.).all() &&
              (p.actuator_max.array() >= 0.).all(),
          "finite actuator bounds must contain zero");
  require(p.actuator_time_constants.allFinite() &&
              (p.actuator_time_constants.array() >= 0.).all(),
          "invalid actuator time constants");
  require(p.allocation_weights.allFinite() &&
              (p.allocation_weights.array() > 0.).all(),
          "allocation weights must be positive and finite");
  require(std::isfinite(p.allocation_regularization) &&
              p.allocation_regularization > 0. &&
              std::isfinite(p.allocation_tolerance) &&
              p.allocation_tolerance > 0. && p.allocation_max_iterations > 0,
          "invalid allocation settings");
  require(p.velocity_control == VelocityControl::WrenchProjection ||
              p.velocity_control == VelocityControl::ThrustVectoring,
          "invalid velocity control mode");
  require(p.attitude_gains.allFinite() &&
              (p.attitude_gains.array() >= 0.).all() &&
              std::isfinite(p.max_tilt_angle) && p.max_tilt_angle > 0. &&
              p.max_tilt_angle < 1.5707963267948966,
          "invalid attitude controller settings");
  if (p.velocity_control == VelocityControl::ThrustVectoring) {
    require(p.center_of_gravity_body.norm() < 1.e-12,
            "ThrustVectoring requires the pose origin at CG");
    require(p.actuation_matrix_body.topRows<2>().norm() < 1.e-12 &&
                p.actuation_matrix_body.row(2).norm() > 1.e-12,
            "ThrustVectoring requires thrust parallel to body z");
    for (Eigen::Index j = 0; j < n; ++j) {
      const double z = p.actuation_matrix_body(2, j);
      require(
          z * p.actuator_min[j] >= -1.e-12 && z * p.actuator_max[j] >= -1.e-12,
          "ThrustVectoring requires nonnegative body-z thrust");
    }
  }
  Model m;
  const Eigen::Matrix3d sg = skew(p.center_of_gravity_body);
  m.mass.topLeftCorner<3, 3>() = p.mass * Eigen::Matrix3d::Identity();
  m.mass.topRightCorner<3, 3>() = -p.mass * sg;
  m.mass.bottomLeftCorner<3, 3>() = p.mass * sg;
  m.mass.bottomRightCorner<3, 3>() = p.inertia_at_cg_body - p.mass * sg * sg;
  Eigen::LLT<Mat6> mass_llt(m.mass);
  require(mass_llt.info() == Eigen::Success,
          "rigid-body mass matrix is not positive definite");
  m.inverse_mass = mass_llt.solve(Mat6::Identity());
  m.allocation = p.actuation_matrix_body;
  m.max_step = p.max_internal_dt;
  for (Eigen::Index j = 0; j < n; ++j)
    if (p.actuator_time_constants[j] > 0.)
      m.max_step = std::min(m.max_step, .25 * p.actuator_time_constants[j]);
  require(m.max_step > 0. && std::ceil(p.dt / m.max_step) <= 1.e6,
          "actuator lag requires too many integration steps; use a smaller dt "
          "or ideal actuators");
  const double density_scale = p.air_density / p.aerodynamic_reference_density;
  m.linear_drag = density_scale * p.linear_drag_body_at_reference_density;
  m.quadratic_drag = density_scale * p.quadratic_drag_body_at_reference_density;
  require(m.mass.allFinite() && m.inverse_mass.allFinite() &&
              m.linear_drag.allFinite() && m.quadratic_drag.allFinite(),
          "physical parameter scaling overflow");
  return m;
}

// Kirchhoff rigid-body coupling; nu.dot(coriolis_product(M, nu)) == 0.
inline Vec6 coriolis_product(const Mat6& mass, const Vec6& nu) {
  const Vec6 momentum = mass * nu;
  Vec6 c;
  c.head<3>() = nu.tail<3>().cross(momentum.head<3>());
  c.tail<3>() = nu.head<3>().cross(momentum.head<3>()) +
                nu.tail<3>().cross(momentum.tail<3>());
  return c;
}
struct Terms {
  Vec6 coriolis = Vec6::Zero();
  Vec6 aerodynamic = Vec6::Zero();  // APPLIED drag wrench
  Vec6 gravitational = Vec6::Zero();
};
inline Terms terms(const Eigen::Matrix3d& rotation, const Vec6& nu,
                   const Parameters& p, const Model& m) {
  Terms t;
  t.coriolis = coriolis_product(m.mass, nu);
  Vec6 relative = nu;
  relative.head<3>() += nu.tail<3>().cross(p.aerodynamic_center_body) -
                        rotation.transpose() * p.wind_world;
  t.aerodynamic = -((m.linear_drag.array() +
                     m.quadratic_drag.array() * relative.array().abs()) *
                    relative.array())
                       .matrix();
  t.aerodynamic.tail<3>() +=
      p.aerodynamic_center_body.cross(t.aerodynamic.head<3>());
  t.gravitational.head<3>() =
      rotation.transpose() * Eigen::Vector3d(0., 0., -p.mass * p.gravity);
  t.gravitational.tail<3>() =
      p.center_of_gravity_body.cross(t.gravitational.head<3>());
  return t;
}
inline Vec6 passive_wrench(const Terms& t) {
  return t.gravitational + t.aerodynamic - t.coriolis;
}
inline Eigen::VectorXd clamp_outputs(const Eigen::VectorXd& u,
                                     const Parameters& p) {
  return u.cwiseMax(p.actuator_min).cwiseMin(p.actuator_max);
}
inline Eigen::VectorXd initial_outputs(const State& state,
                                       const Parameters& p) {
  const Eigen::Index n = p.actuation_matrix_body.cols();
  if (state.actuator_outputs.size() == 0) return Eigen::VectorXd::Zero(n);
  require(state.actuator_outputs.size() == n,
          "actuator state dimension must match B");
  for (Eigen::Index j = 0; j < n; ++j)
    if (p.actuator_time_constants[j] > 0.)
      require(state.actuator_outputs[j] >= p.actuator_min[j] - 1.e-10 &&
                  state.actuator_outputs[j] <= p.actuator_max[j] + 1.e-10,
              "initial actuator output is outside configured limits");
  return clamp_outputs(state.actuator_outputs, p);
}

inline Eigen::VectorXd allocate(const Vec6& requested,
                                const Eigen::VectorXd& warm_start,
                                const Parameters& p,
                                const AllocationMatrix& allocation_world,
                                int& iterations, bool& converged) {
  return dynamics::detail::bounded_least_squares(
      allocation_world, requested, warm_start, p.actuator_min, p.actuator_max,
      p.allocation_weights, p.allocation_regularization,
      p.allocation_max_iterations, p.allocation_tolerance, iterations,
      converged);
}

inline Eigen::VectorXd actuator_output(const Eigen::VectorXd& initial,
                                       const Eigen::VectorXd& setpoint,
                                       double elapsed, const Parameters& p) {
  Eigen::VectorXd u = setpoint;
  for (Eigen::Index j = 0; j < u.size(); ++j)
    if (p.actuator_time_constants[j] > 0.)
      u[j] += std::exp(-elapsed / p.actuator_time_constants[j]) *
              (initial[j] - setpoint[j]);
  return u;
}
inline Derivative derivative(const InternalState& x,
                             const Eigen::VectorXd& outputs,
                             const Parameters& p, const Model& m) {
  const Eigen::Matrix3d r = x.q.normalized().toRotationMatrix();
  Derivative d;
  d.position = r * x.nu.head<3>();
  const Eigen::Quaterniond omega(0., x.nu[3], x.nu[4], x.nu[5]);
  d.quaternion = .5 * (x.q * omega).coeffs();
  d.nu = m.inverse_mass *
         (m.allocation * outputs + passive_wrench(terms(r, x.nu, p, m)) +
          orientation_map(r).transpose() * p.external_wrench_world);
  return d;
}
inline void rk4(InternalState& x, Eigen::VectorXd& outputs,
                const Eigen::VectorXd& setpoint, double h, const Parameters& p,
                const Model& m) {
  dynamics::detail::rk4_rigid_body(
      x, h, [&](const InternalState& stage, double elapsed) {
        return derivative(stage, actuator_output(outputs, setpoint, elapsed, p),
                          p, m);
      });
  outputs = clamp_outputs(actuator_output(outputs, setpoint, h, p), p);
}

inline Vec6 velocity_wrench(const Vec6& target, const Measurement& feedback,
                            const Parameters& p, const Model& m) {
  const Eigen::Matrix3d r = feedback.pose.topLeftCorner<3, 3>();
  const Mat6 t = orientation_map(r);
  const Vec6 nu = t.transpose() * feedback.twist;
  const Vec6 acceleration_world =
      p.velocity_gains.cwiseProduct(target - feedback.twist);
  Vec6 acceleration_body = t.transpose() * acceleration_world - transport(nu);
  const Terms passive = terms(r, nu, p, m);
  if (p.velocity_control == VelocityControl::ThrustVectoring) {
    Eigen::Vector3d force_world = p.mass * acceleration_world.head<3>();
    if (p.model_compensation)
      force_world -=
          r * (passive.gravitational.head<3>() + passive.aerodynamic.head<3>());
    const double horizontal = force_world.head<2>().norm();
    const double tilt =
        std::min(p.max_tilt_angle,
                 std::atan2(horizontal, std::max(0., force_world.z())));
    Eigen::Vector3d z_desired = Eigen::Vector3d::UnitZ();
    if (horizontal > 1.e-12) {
      z_desired.head<2>() = std::sin(tilt) * force_world.head<2>() / horizontal;
      z_desired.z() = std::cos(tilt);
    }
    // Velocity API specifies yaw RATE, so maintain measured heading and damp
    // toward the requested world-z rate. There is no hidden yaw-angle state.
    double heading = std::atan2(r(1, 0), r(0, 0));
    if (r.col(0).head<2>().norm() < 1.e-9)
      heading = std::atan2(-r(0, 1), r(1, 1));
    const Eigen::Vector3d heading_axis(std::cos(heading), std::sin(heading),
                                       0.);
    const Eigen::Vector3d y_desired =
        z_desired.cross(heading_axis).normalized();
    Eigen::Matrix3d desired;
    desired.col(0) = y_desired.cross(z_desired);
    desired.col(1) = y_desired;
    desired.col(2) = z_desired;
    Eigen::Quaterniond error(r.transpose() * desired);
    error.normalize();
    if (error.w() < 0.) error.coeffs() *= -1.;
    const Eigen::Vector3d rate_desired =
        r.transpose() * Eigen::Vector3d(0., 0., target[5]);
    acceleration_body.tail<3>() =
        p.attitude_gains.cwiseProduct(2. * error.vec()) +
        p.velocity_gains.tail<3>().cwiseProduct(rate_desired - nu.tail<3>());
    Vec6 requested_body = m.mass * acceleration_body;
    if (p.model_compensation) requested_body -= passive_wrench(passive);
    requested_body.head<3>() =
        Eigen::Vector3d(0., 0., std::max(0., force_world.dot(r.col(2))));
    return t * requested_body;
  }
  Vec6 requested_body = m.mass * acceleration_body;
  if (p.model_compensation) requested_body -= passive_wrench(passive);
  return t * requested_body;
}

enum class InputMode { Actuators, Wrench, Velocity };

inline State integrate(const Eigen::VectorXd& command, InputMode mode,
                       const State& state, const Parameters& p,
                       const Model& m) {
  State out =
      state;  // random stream is copied by value; input is never mutated
  auto x = to_internal(state);
  Eigen::VectorXd outputs = initial_outputs(state, p);
  Eigen::VectorXd setpoint = outputs;
  Vec6 requested_world = Vec6::Zero();
  int iterations = 0;
  bool converged = true;
  const double end = state.time + p.dt;
  require(std::isfinite(end) && end > state.time,
          "dt is outside simulation clock resolution");
  if (p.measurement_noise.enabled) {
    if (!out.measurement_noise_state ||
        !same_noise_settings(out.measurement_noise_state->configuration,
                             p.measurement_noise))
      out.measurement_noise_state.emplace(p.measurement_noise);
    auto& memory = *out.measurement_noise_state;
    const double tol = clock_tolerance(state.time);
    if (memory.initialized) {
      require(memory.last_sample_time <= state.time + tol &&
                  (p.measurement_noise.sample_period == 0. ||
                   memory.next_sample_time >= state.time - tol),
              "inconsistent measurement clock; reset noise state after "
              "changing time");
      require(out.measurement.valid,
              "missing cached measurement; reset noise state");
      validate_pose(out.measurement.pose);
      require(out.measurement.twist.size() == 6 &&
                  out.measurement.twist.allFinite() &&
                  std::isfinite(out.measurement.sample_time) &&
                  std::abs(out.measurement.sample_time -
                           memory.last_sample_time) <= tol,
              "invalid cached measurement; reset noise state");
    }
  } else {
    // Disabling noise discards sensor memory. Re-enabling starts a new epoch.
    out.measurement_noise_state.reset();
  }
  double time = state.time;
  int count = 0;
  while (time < end) {
    if (++count > 2000002)
      throw std::runtime_error("UAV: too many sensor/integration substeps");
    double h = std::min(m.max_step, end - time);
    if (p.measurement_noise.enabled) {
      auto& memory = *out.measurement_noise_state;
      sample_if_due(x, time, memory, out.measurement);
      if (p.measurement_noise.sample_period > 0.) {
        require(memory.next_sample_time > time,
                "measurement period is below clock resolution");
        h = std::min(h, memory.next_sample_time - time);
      }
    } else
      out.measurement = exact_measurement(x, time);
    if (mode == InputMode::Actuators) {
      setpoint = clamp_outputs(command, p);
      requested_world =
          orientation_map(x.q.toRotationMatrix()) * m.allocation * command;
    } else {
      const Eigen::Matrix3d feedback_rotation =
          out.measurement.pose.topLeftCorner<3, 3>();
      const Mat6 t = orientation_map(feedback_rotation);
      requested_world = mode == InputMode::Velocity
                            ? velocity_wrench(command, out.measurement, p, m)
                            : Vec6(command);
      require(requested_world.allFinite(),
              "nonfinite controller wrench; check gains and commands");
      // A world wrench is converted using the measured attitude when noise is
      // enabled, so feedback errors influence the controller, never the truth.
      setpoint = allocate(requested_world, outputs, p, t * m.allocation,
                          iterations, converged);
    }
    rk4(x, outputs, setpoint, h, p, m);
    const double next = time + h;
    require(next > time, "integration step is below clock resolution");
    time = next;
    if (end - time <= clock_tolerance(end)) time = end;
  }
  const Mat6 t = orientation_map(x.q.toRotationMatrix());
  if (mode == InputMode::Actuators)
    requested_world = t * m.allocation * command;
  out.pose = Eigen::Matrix4d::Identity();
  out.pose.topLeftCorner<3, 3>() = x.q.toRotationMatrix();
  out.pose.topRightCorner<3, 1>() = x.position;
  out.twist = t * x.nu;
  out.actuator_outputs = outputs;
  out.time = end;
  Diagnostics d;
  d.requested_wrench = requested_world;
  d.allocated_wrench = t * m.allocation * setpoint;
  d.applied_wrench = t * m.allocation * outputs;
  d.allocation_residual = d.requested_wrench - d.allocated_wrench;
  d.actuator_setpoints = setpoint;
  d.allocation_iterations = iterations;
  d.allocation_converged = converged;
  for (Eigen::Index j = 0; j < setpoint.size(); ++j)
    if (setpoint[j] <= p.actuator_min[j] + 1.e-7 ||
        setpoint[j] >= p.actuator_max[j] - 1.e-7)
      ++d.saturated_actuators;
  out.diagnostics = std::move(d);
  if (p.measurement_noise.enabled)
    sample_if_due(x, end, *out.measurement_noise_state, out.measurement);
  else
    out.measurement = exact_measurement(x, end);
  return out;
}
}  // namespace detail

// Resize all actuator parameters together when selecting a new model.
// Subsequent calls validate all physical/actuator settings before integrating.
inline void configure_actuators(Parameters& p,
                                const AllocationMatrix& matrix_body,
                                const Eigen::VectorXd& lower,
                                const Eigen::VectorXd& upper,
                                double time_constant = 0.) {
  detail::require(matrix_body.cols() > 0 &&
                      lower.size() == matrix_body.cols() &&
                      upper.size() == matrix_body.cols(),
                  "actuator dimensions must match B");
  detail::require(std::isfinite(time_constant) && time_constant >= 0.,
                  "invalid actuator lag");
  p.actuation_matrix_body = matrix_body;
  p.actuator_min = lower;
  p.actuator_max = upper;
  p.actuator_time_constants =
      Eigen::VectorXd::Constant(matrix_body.cols(), time_constant);
}

struct Rotor {
  Eigen::Vector3d position_body =
      Eigen::Vector3d::Zero();  // relative to pose origin, m
  Eigen::Vector3d thrust_axis_body = Eigen::Vector3d::UnitZ();  // unit vector
  double reaction_moment_per_newton = 0.;  // signed kQ/kT, metres
};
inline AllocationMatrix rotor_allocation_matrix(
    const std::vector<Rotor>& rotors) {
  detail::require(!rotors.empty() && rotors.size() <= 256,
                  "invalid rotor count");
  AllocationMatrix b(6, static_cast<Eigen::Index>(rotors.size()));
  for (Eigen::Index i = 0; i < b.cols(); ++i) {
    const auto& rotor = rotors[static_cast<std::size_t>(i)];
    detail::require(rotor.position_body.allFinite() &&
                        rotor.thrust_axis_body.allFinite() &&
                        std::abs(rotor.thrust_axis_body.norm() - 1.) < 1.e-8 &&
                        std::isfinite(rotor.reaction_moment_per_newton),
                    "invalid rotor geometry");
    b.col(i).head<3>() = rotor.thrust_axis_body;
    b.col(i).tail<3>() =
        rotor.position_body.cross(rotor.thrust_axis_body) +
        rotor.reaction_moment_per_newton * rotor.thrust_axis_body;
  }
  return b;
}

inline AllocationMatrix allocation_matrix(const Eigen::Matrix4d& pose,
                                          const Parameters& p = Parameters()) {
  detail::validate_pose(pose);
  return detail::orientation_map(pose.topLeftCorner<3, 3>()) *
         detail::prepare(p).allocation;
}
inline Mat6 total_mass_matrix(const Eigen::Matrix4d& pose,
                              const Parameters& p = Parameters()) {
  detail::validate_pose(pose);
  const Mat6 t = detail::orientation_map(pose.topLeftCorner<3, 3>());
  return t * detail::prepare(p).mass * t.transpose();
}
// Actual inertial acceleration for the specified instantaneous outputs,
// after hard saturation. This does not advance lag, time or sensor state.
inline Vec6 inertial_acceleration(const State& state,
                                  const Eigen::VectorXd& actuator_outputs,
                                  const Parameters& p = Parameters()) {
  detail::validate_state(state);
  const auto m = detail::prepare(p);
  detail::require(
      actuator_outputs.size() == m.allocation.cols() &&
          actuator_outputs.allFinite(),
      "output command dimension must match B and entries must be finite");
  const auto x = detail::to_internal(state);
  const auto d =
      detail::derivative(x, detail::clamp_outputs(actuator_outputs, p), p, m);
  return detail::orientation_map(x.q.toRotationMatrix()) *
         (d.nu + detail::transport(x.nu));
}

// Physical output SETPOINTS, e.g. rotor thrusts in N. NO velocity controller.
inline State step_open_loop(const Eigen::VectorXd& actuator_setpoints,
                            const State& state,
                            const Parameters& p = Parameters()) {
  detail::validate_state(state);
  const auto m = detail::prepare(p);
  detail::require(actuator_setpoints.size() == m.allocation.cols() &&
                      actuator_setpoints.allFinite(),
                  "setpoint dimension must match B and entries must be finite");
  return detail::integrate(actuator_setpoints, detail::InputMode::Actuators,
                           state, p, m);
}
// Requested INERTIAL actuator wrench about origin, held fixed over this call.
// The allocator handles unavailable directions and bounds; no passive-force
// cancellation.
inline State step_wrench(const Eigen::VectorXd& wrench_world,
                         const State& state,
                         const Parameters& p = Parameters()) {
  detail::validate_state(state);
  detail::require(wrench_world.size() == 6 && wrench_world.allFinite(),
                  "invalid world wrench");
  return detail::integrate(wrench_world, detail::InputMode::Wrench, state, p,
                           detail::prepare(p));
}
// Main velocity-command API, matching the reference's inertial twist
// convention. WrenchProjection cannot independently track velocities outside
// the attainable wrench space. ThrustVectoring uses roll/pitch to track
// translation and yaw rate.
inline State step(const Eigen::VectorXd& target_twist, const State& state,
                  const Parameters& p = Parameters()) {
  detail::validate_state(state);
  detail::require(target_twist.size() == 6 && target_twist.allFinite(),
                  "invalid target twist");
  if (p.velocity_control == VelocityControl::ThrustVectoring)
    detail::require(std::abs(target_twist[3]) < 1.e-12 &&
                        std::abs(target_twist[4]) < 1.e-12,
                    "ThrustVectoring accepts translation and world yaw rate; "
                    "roll/pitch rate targets must be zero");
  return detail::integrate(target_twist, detail::InputMode::Velocity, state, p,
                           detail::prepare(p));
}
// Stateless convenience API. Stateful lag/noise must use the State overload.
inline State step(const Eigen::VectorXd& target_twist,
                  const Eigen::Matrix4d& pose,
                  const Eigen::VectorXd& current_twist,
                  const Parameters& p = Parameters()) {
  detail::require((p.actuator_time_constants.array() == 0.).all(),
                  "actuator lag requires step(target, State, parameters)");
  detail::require(!p.measurement_noise.enabled,
                  "measurement noise requires persistent State");
  State state;
  state.pose = pose;
  state.twist = current_twist;
  return step(target_twist, state, p);
}
inline State step(const Eigen::VectorXd& target_twist,
                  const Eigen::Matrix4d& pose,
                  const Eigen::VectorXd& current_twist,
                  const Eigen::VectorXd& velocity_gains, double dt) {
  Parameters p;
  p.velocity_gains = velocity_gains;
  p.dt = dt;
  return step(target_twist, pose, current_twist, p);
}
}  // namespace uav
