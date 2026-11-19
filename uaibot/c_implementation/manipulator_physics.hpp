#pragma once
// Fixed-base serial-chain rigid-body dynamics. C++17, Eigen >= 3.3, SI units.
// All public vectors are in joint order. One 1-DoF joint per Link.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "dynamics.hpp"

namespace manipulator {
using Vec6 = dynamics::Vec6;
enum class JointType { Revolute, Prismatic };

struct Link {
  JointType joint_type = JointType::Revolute;
  // Fixed transform: joint frame at q=0 -> PARENT link frame (base for link 0).
  Eigen::Matrix4d parent_to_joint = Eigen::Matrix4d::Identity();
  // Motion axis in joint frame. Child link frame = joint frame *
  // joint_motion(q).
  Eigen::Vector3d axis_joint = Eigen::Vector3d::UnitZ();
  double mass = 1.;
  Eigen::Vector3d center_of_mass_body = Eigen::Vector3d::Zero();
  Eigen::Matrix3d inertia_at_com_body = .01 * Eigen::Matrix3d::Identity();
  // Generalized friction: viscous*qd + coulomb*tanh(qd/friction_speed).
  double viscous_friction = 0.;
  double coulomb_friction = 0.;
  double friction_speed = .01;
  // Already reflected to this joint, kg*m^2 for revolute / kg for prismatic.
  double reflected_motor_inertia = 0.;
};

struct JointMeasurementNoiseParameters {
  bool enabled = false;
  // Order [q(n), qd(n)]. Variances per sample, NOT standard deviations.
  // Empty vectors resolve to zero variances and 1-second correlation times.
  Eigen::VectorXd white_variances, colored_variances, correlation_times;
  double sample_period = .01;
  std::uint64_t seed = 42;
  bool stationary_initialization = true;
};
struct Parameters {
  std::vector<Link> links;
  Eigen::Vector3d gravity_base = Eigen::Vector3d(0., 0., -9.81);
  double dt = .01;
  double max_internal_dt = .001;
  // Optional applied external [force;torque] at each LINK FRAME ORIGIN,
  // expressed in that link's moving body axes. Empty means all zero.
  // Body-fixed values are held per API call; no hidden end-effector assumption.
  std::vector<Vec6> external_wrenches_body;
  // tau_joint = B*u. Empty B resolves to n-by-n identity. Bounds must include
  // 0.
  Eigen::MatrixXd actuation_matrix;
  Eigen::VectorXd actuator_min, actuator_max, actuator_time_constants;
  Eigen::VectorXd allocation_weights;
  double allocation_regularization = 1.e-8;
  int allocation_max_iterations = 300;
  double allocation_tolerance = 1.e-9;
  Eigen::VectorXd velocity_gains;  // empty = 6/s per joint
  bool model_compensation = true;
  JointMeasurementNoiseParameters measurement_noise;
};
struct JointMeasurement {
  Eigen::VectorXd q, qd;
  double sample_time = 0.;
  bool valid = false;
};
struct Diagnostics {
  Eigen::VectorXd requested_effort, allocated_effort, applied_effort,
      allocation_residual;
  Eigen::VectorXd actuator_setpoints;
  int saturated_actuators = 0, allocation_iterations = 0;
  bool allocation_converged = true;
};
namespace detail {
struct JointNoiseMemory {
  explicit JointNoiseMemory(const JointMeasurementNoiseParameters& p)
      : configuration(p),
        generator(p.seed),
        colored_error(Eigen::VectorXd::Zero(p.white_variances.size())) {}
  JointMeasurementNoiseParameters configuration;
  std::mt19937_64 generator;
  std::normal_distribution<double> normal{0., 1.};
  Eigen::VectorXd colored_error;
  double epoch = 0., last_sample_time = 0., next_sample_time = 0.;
  std::uint64_t sample_count = 0;
  bool initialized = false;
};
}  // namespace detail
struct State {
  // Both empty initializes at q=qd=0. Otherwise both must have n entries.
  Eigen::VectorXd q, qd;
  Eigen::VectorXd actuator_outputs;  // empty = zero outputs
  double time = 0.;
  JointMeasurement measurement;
  Diagnostics diagnostics;
  std::optional<detail::JointNoiseMemory> measurement_noise_state;
};

namespace detail {
using dynamics::detail::clock_tolerance;
using dynamics::detail::require;
struct Model {
  Eigen::Index n = 0, m = 0;
  Eigen::MatrixXd allocation;
  Eigen::VectorXd lower, upper, time_constants, weights, gains;
  JointMeasurementNoiseParameters noise;
  double max_step = 0.;
};
inline Eigen::VectorXd resolved(const Eigen::VectorXd& value, Eigen::Index n,
                                double fallback, const char* label) {
  if (value.size() == 0) return Eigen::VectorXd::Constant(n, fallback);
  require(value.size() == n && value.allFinite(),
          std::string("Manipulator: invalid ") + label);
  return value;
}
inline Model prepare(const Parameters& p) {
  Model model;
  model.n = static_cast<Eigen::Index>(p.links.size());
  require(model.n > 0 && model.n <= 128,
          "Manipulator: chain must have 1..128 links");
  require(std::isfinite(p.dt) && p.dt > 0. &&
              std::isfinite(p.max_internal_dt) && p.max_internal_dt > 0.,
          "Manipulator: invalid integration interval");
  require(p.gravity_base.allFinite(), "Manipulator: nonfinite gravity");
  require(p.external_wrenches_body.empty() ||
              p.external_wrenches_body.size() == p.links.size(),
          "Manipulator: external wrench count must equal link count");
  for (std::size_t i = 0; i < p.links.size(); ++i) {
    const auto& link = p.links[i];
    dynamics::detail::validate_pose(link.parent_to_joint);
    require(link.joint_type == JointType::Revolute ||
                link.joint_type == JointType::Prismatic,
            "Manipulator: invalid joint type");
    require(link.axis_joint.allFinite() &&
                std::abs(link.axis_joint.norm() - 1.) < 1.e-10,
            "Manipulator: joint axis must be a unit vector");
    require(std::isfinite(link.mass) && link.mass > 0. &&
                link.center_of_mass_body.allFinite(),
            "Manipulator: link mass must be positive and COM finite");
    require(link.inertia_at_com_body.allFinite() &&
                link.inertia_at_com_body.isApprox(
                    link.inertia_at_com_body.transpose(), 1.e-12),
            "Manipulator: inertia must be symmetric");
    Eigen::LLT<Eigen::Matrix3d> llt(link.inertia_at_com_body);
    require(llt.info() == Eigen::Success,
            "Manipulator: inertia must be positive definite");
    const Eigen::Matrix3d second =
        .5 * link.inertia_at_com_body.trace() * Eigen::Matrix3d::Identity() -
        link.inertia_at_com_body;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(second);
    require(eig.info() == Eigen::Success &&
                eig.eigenvalues().minCoeff() >=
                    -1.e-10 * link.inertia_at_com_body.norm(),
            "Manipulator: inertia violates triangle inequalities");
    require(
        std::isfinite(link.viscous_friction) && link.viscous_friction >= 0. &&
            std::isfinite(link.coulomb_friction) &&
            link.coulomb_friction >= 0. && std::isfinite(link.friction_speed) &&
            link.friction_speed > 0. &&
            std::isfinite(link.reflected_motor_inertia) &&
            link.reflected_motor_inertia >= 0.,
        "Manipulator: invalid friction/motor inertia");
    if (!p.external_wrenches_body.empty())
      require(p.external_wrenches_body[i].allFinite(),
              "Manipulator: nonfinite external wrench");
  }
  if (p.actuation_matrix.size() == 0)
    model.allocation = Eigen::MatrixXd::Identity(model.n, model.n);
  else {
    require(
        p.actuation_matrix.rows() == model.n && p.actuation_matrix.cols() > 0 &&
            p.actuation_matrix.cols() <= 256 && p.actuation_matrix.allFinite(),
        "Manipulator: B must be finite n-by-m with 1..256 actuators");
    model.allocation = p.actuation_matrix;
  }
  model.m = model.allocation.cols();
  model.lower = resolved(p.actuator_min, model.m, -100., "actuator_min");
  model.upper = resolved(p.actuator_max, model.m, 100., "actuator_max");
  model.time_constants = resolved(p.actuator_time_constants, model.m, 0.,
                                  "actuator_time_constants");
  model.weights =
      resolved(p.allocation_weights, model.n, 1., "allocation_weights");
  model.gains = resolved(p.velocity_gains, model.n, 6., "velocity_gains");
  require((model.lower.array() <= 0.).all() &&
              (model.upper.array() >= 0.).all() &&
              (model.time_constants.array() >= 0.).all() &&
              (model.weights.array() > 0.).all() &&
              (model.gains.array() >= 0.).all(),
          "Manipulator: invalid actuator limits/lag/weights/gains");
  require(std::isfinite(p.allocation_regularization) &&
              p.allocation_regularization > 0. &&
              std::isfinite(p.allocation_tolerance) &&
              p.allocation_tolerance > 0. && p.allocation_max_iterations > 0,
          "Manipulator: invalid allocator settings");
  model.max_step = p.max_internal_dt;
  for (Eigen::Index j = 0; j < model.m; ++j)
    if (model.time_constants[j] > 0.)
      model.max_step = std::min(model.max_step, .25 * model.time_constants[j]);
  require(model.max_step > 0. && std::ceil(p.dt / model.max_step) <= 1.e6,
          "Manipulator: too many integration steps");
  model.noise = p.measurement_noise;
  if (model.noise.enabled) {
    auto& noise = model.noise;
    noise.white_variances = resolved(noise.white_variances, 2 * model.n, 0.,
                                     "noise white_variances");
    noise.colored_variances = resolved(noise.colored_variances, 2 * model.n, 0.,
                                       "noise colored_variances");
    noise.correlation_times = resolved(noise.correlation_times, 2 * model.n, 1.,
                                       "noise correlation_times");
    require((noise.white_variances.array() >= 0.).all() &&
                (noise.colored_variances.array() >= 0.).all() &&
                (noise.correlation_times.array() > 0.).all() &&
                std::isfinite(noise.sample_period) && noise.sample_period >= 0.,
            "Manipulator: invalid noise settings");
    require(noise.sample_period == 0. ||
                std::ceil(p.dt / noise.sample_period) <= 1.e6,
            "Manipulator: too many sensor updates");
  }
  return model;
}
inline void validate_joint_vector(const Eigen::VectorXd& x, Eigen::Index n,
                                  const char* name) {
  require(x.size() == n && x.allFinite(),
          std::string("Manipulator: invalid ") + name);
}
inline Eigen::VectorXd packed_state(const State& state, const Model& model) {
  require(std::isfinite(state.time) && state.time >= 0.,
          "Manipulator: invalid simulation time");
  Eigen::VectorXd x = Eigen::VectorXd::Zero(2 * model.n);
  if (state.q.size() == 0 && state.qd.size() == 0) return x;
  validate_joint_vector(state.q, model.n, "q");
  validate_joint_vector(state.qd, model.n, "qd");
  x.head(model.n) = state.q;
  x.tail(model.n) = state.qd;
  return x;
}

struct Kinematics {
  std::vector<Eigen::Matrix4d> poses;
  std::vector<Eigen::Vector3d> axis, omega, alpha, acceleration;
};
inline Kinematics kinematics(const Eigen::VectorXd& q,
                             const Eigen::VectorXd& qd,
                             const Eigen::VectorXd& qdd, const Parameters& p,
                             bool gravity) {
  const Eigen::Index n = q.size();
  Kinematics k;
  k.poses.resize(n);
  k.axis.resize(n);
  k.omega.resize(n);
  k.alpha.resize(n);
  k.acceleration.resize(n);
  Eigen::Matrix4d parent = Eigen::Matrix4d::Identity();
  Eigen::Vector3d omega = Eigen::Vector3d::Zero(),
                  alpha = Eigen::Vector3d::Zero();
  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  if (gravity) acceleration = -p.gravity_base;
  for (Eigen::Index i = 0; i < n; ++i) {
    const auto& link = p.links[static_cast<std::size_t>(i)];
    const Eigen::Matrix4d joint = parent * link.parent_to_joint;
    const Eigen::Matrix3d joint_rotation = joint.topLeftCorner<3, 3>();
    const Eigen::Vector3d axis = joint_rotation * link.axis_joint;
    Eigen::Matrix4d child = joint;
    if (link.joint_type == JointType::Revolute)
      child.topLeftCorner<3, 3>() =
          joint_rotation *
          Eigen::AngleAxisd(q[i], link.axis_joint).toRotationMatrix();
    else
      child.topRightCorner<3, 1>() += axis * q[i];
    const Eigen::Vector3d offset =
        child.topRightCorner<3, 1>() - parent.topRightCorner<3, 1>();
    Eigen::Vector3d child_acceleration =
        acceleration + alpha.cross(offset) + omega.cross(omega.cross(offset));
    Eigen::Vector3d child_omega = omega, child_alpha = alpha;
    if (link.joint_type == JointType::Revolute) {
      child_omega += axis * qd[i];
      child_alpha += axis * qdd[i] + omega.cross(axis * qd[i]);
    } else
      child_acceleration += 2. * omega.cross(axis * qd[i]) + axis * qdd[i];
    k.poses[i] = child;
    k.axis[i] = axis;
    k.omega[i] = child_omega;
    k.alpha[i] = child_alpha;
    k.acceleration[i] = child_acceleration;
    parent = child;
    omega = child_omega;
    alpha = child_alpha;
    acceleration = child_acceleration;
  }
  return k;
}

// Recursive Newton-Euler in BASE axes: geometric accelerations and force/moment
// about each moving link origin. This is equivalent to spatial RNEA without
// mixing linear-first and angular-first spatial-vector conventions.
inline Eigen::VectorXd rnea(const Eigen::VectorXd& q, const Eigen::VectorXd& qd,
                            const Eigen::VectorXd& qdd, const Parameters& p,
                            bool gravity = true, bool friction = true,
                            bool external = true) {
  const Eigen::Index n = q.size();
  const auto k = kinematics(q, qd, qdd, p, gravity);
  std::vector<Eigen::Vector3d> forces(n), torques(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    const auto& link = p.links[static_cast<std::size_t>(i)];
    const Eigen::Matrix3d rotation = k.poses[i].topLeftCorner<3, 3>();
    const Eigen::Vector3d com = rotation * link.center_of_mass_body;
    const Eigen::Matrix3d inertia =
        rotation * link.inertia_at_com_body * rotation.transpose();
    forces[i] = link.mass * (k.acceleration[i] + k.alpha[i].cross(com) +
                             k.omega[i].cross(k.omega[i].cross(com)));
    torques[i] = inertia * k.alpha[i] + k.omega[i].cross(inertia * k.omega[i]) +
                 com.cross(forces[i]);
    if (external && !p.external_wrenches_body.empty()) {
      forces[i] -=
          rotation *
          p.external_wrenches_body[static_cast<std::size_t>(i)].head<3>();
      torques[i] -=
          rotation *
          p.external_wrenches_body[static_cast<std::size_t>(i)].tail<3>();
    }
  }
  Eigen::VectorXd effort(n);
  for (Eigen::Index i = n; i-- > 0;) {
    const auto& link = p.links[static_cast<std::size_t>(i)];
    effort[i] = k.axis[i].dot(
        link.joint_type == JointType::Revolute ? torques[i] : forces[i]);
    effort[i] += link.reflected_motor_inertia * qdd[i];
    if (friction)
      effort[i] +=
          link.viscous_friction * qd[i] +
          link.coulomb_friction * std::tanh(qd[i] / link.friction_speed);
    if (i > 0) {
      const Eigen::Vector3d offset = k.poses[i].topRightCorner<3, 1>() -
                                     k.poses[i - 1].topRightCorner<3, 1>();
      forces[i - 1] += forces[i];
      torques[i - 1] += torques[i] + offset.cross(forces[i]);
    }
  }
  require(effort.allFinite(), "Manipulator: inverse dynamics became nonfinite");
  return effort;
}
inline Eigen::MatrixXd mass(const Eigen::VectorXd& q, const Parameters& p) {
  const Eigen::Index n = q.size();
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(n);
  Eigen::MatrixXd m(n, n);
  for (Eigen::Index j = 0; j < n; ++j) {
    Eigen::VectorXd acceleration = zero;
    acceleration[j] = 1.;
    m.col(j) = rnea(q, zero, acceleration, p, false, false, false);
  }
  require(m.isApprox(m.transpose(), 1.e-10),
          "Manipulator: inconsistent mass matrix");
  return .5 * (m + m.transpose()).eval();
}
inline Eigen::VectorXd acceleration(const Eigen::VectorXd& q,
                                    const Eigen::VectorXd& qd,
                                    const Eigen::VectorXd& joint_effort,
                                    const Parameters& p) {
  const Eigen::MatrixXd m = mass(q, p);
  Eigen::LLT<Eigen::MatrixXd> factor(m);
  require(factor.info() == Eigen::Success,
          "Manipulator: singular/nonpositive joint-space inertia");
  const Eigen::VectorXd rhs =
      joint_effort - rnea(q, qd, Eigen::VectorXd::Zero(q.size()), p);
  const Eigen::VectorXd qdd = factor.solve(rhs);
  require(qdd.allFinite(), "Manipulator: forward dynamics became nonfinite");
  return qdd;
}
inline Eigen::VectorXd clamp(const Eigen::VectorXd& u, const Model& m) {
  return u.cwiseMax(m.lower).cwiseMin(m.upper);
}
inline Eigen::VectorXd initial_outputs(const State& state, const Model& m) {
  if (state.actuator_outputs.size() == 0) return Eigen::VectorXd::Zero(m.m);
  validate_joint_vector(state.actuator_outputs, m.m, "actuator state");
  for (Eigen::Index j = 0; j < m.m; ++j)
    if (m.time_constants[j] > 0.)
      require(state.actuator_outputs[j] >= m.lower[j] - 1.e-10 &&
                  state.actuator_outputs[j] <= m.upper[j] + 1.e-10,
              "Manipulator: actuator state outside bounds");
  return clamp(state.actuator_outputs, m);
}
inline Eigen::VectorXd actuator_output(const Eigen::VectorXd& initial,
                                       const Eigen::VectorXd& setpoint,
                                       double elapsed, const Model& m) {
  Eigen::VectorXd output = setpoint;
  for (Eigen::Index j = 0; j < m.m; ++j)
    if (m.time_constants[j] > 0.)
      output[j] +=
          std::exp(-elapsed / m.time_constants[j]) * (initial[j] - setpoint[j]);
  return output;
}

inline bool same_noise(const JointMeasurementNoiseParameters& a,
                       const JointMeasurementNoiseParameters& b) {
  return a.seed == b.seed && a.sample_period == b.sample_period &&
         a.stationary_initialization == b.stationary_initialization &&
         a.white_variances.size() == b.white_variances.size() &&
         (a.white_variances.array() == b.white_variances.array()).all() &&
         (a.colored_variances.array() == b.colored_variances.array()).all() &&
         (a.correlation_times.array() == b.correlation_times.array()).all();
}
inline JointMeasurement exact_measurement(const Eigen::VectorXd& x,
                                          double time) {
  const Eigen::Index n = x.size() / 2;
  return {x.head(n), x.tail(n), time, true};
}
inline void sample_if_due(const Eigen::VectorXd& x, double time,
                          JointNoiseMemory& memory,
                          JointMeasurement& measurement) {
  const auto& noise = memory.configuration;
  const double tol = clock_tolerance(time);
  if (memory.initialized) {
    if (noise.sample_period > 0. && time < memory.next_sample_time - tol)
      return;
    if (noise.sample_period == 0. && time <= memory.last_sample_time + tol)
      return;
    const double elapsed = time - memory.last_sample_time;
    require(elapsed >= 0., "Manipulator: sensor clock moved backward");
    dynamics::detail::advance_colored_error(
        memory.colored_error, noise.colored_variances, noise.correlation_times,
        elapsed, memory.normal, memory.generator);
  } else {
    memory.epoch = time;
    dynamics::detail::initialize_colored_error(
        memory.colored_error, noise.colored_variances,
        noise.stationary_initialization, memory.normal, memory.generator);
    memory.initialized = true;
  }
  Eigen::VectorXd error = memory.colored_error;
  dynamics::detail::add_white_error(error, noise.white_variances, memory.normal,
                                    memory.generator);
  measurement = exact_measurement((x + error).eval(), time);
  require(measurement.q.allFinite() && measurement.qd.allFinite(),
          "Manipulator: nonfinite measurement");
  memory.last_sample_time = time;
  ++memory.sample_count;
  memory.next_sample_time =
      memory.epoch + double(memory.sample_count) * noise.sample_period;
}

inline Eigen::VectorXd allocate(const Eigen::VectorXd& requested,
                                const Eigen::VectorXd& warm,
                                const Parameters& p, const Model& m,
                                int& iterations, bool& converged) {
  return dynamics::detail::bounded_least_squares(
      m.allocation, requested, warm, m.lower, m.upper, m.weights,
      p.allocation_regularization, p.allocation_max_iterations,
      p.allocation_tolerance, iterations, converged);
}

enum class InputMode { Actuator, Effort, Velocity };
inline State integrate(const Eigen::VectorXd& command, InputMode mode,
                       const State& state, const Parameters& p,
                       const Model& m) {
  Eigen::VectorXd x = packed_state(state, m);
  Eigen::VectorXd outputs = initial_outputs(state, m), setpoint = outputs;
  State out = state;
  const double end = state.time + p.dt;
  require(std::isfinite(end) && end > state.time,
          "Manipulator: dt below clock resolution");
  if (m.noise.enabled) {
    if (!out.measurement_noise_state ||
        !same_noise(out.measurement_noise_state->configuration, m.noise))
      out.measurement_noise_state.emplace(m.noise);
    const auto& memory = *out.measurement_noise_state;
    if (memory.initialized) {
      const double tol = clock_tolerance(state.time);
      require(memory.last_sample_time <= state.time + tol &&
                  (m.noise.sample_period == 0. ||
                   memory.next_sample_time >= state.time - tol),
              "Manipulator: inconsistent sensor clock; reset noise state after "
              "changing time");
      require(
          out.measurement.valid && out.measurement.q.size() == m.n &&
              out.measurement.qd.size() == m.n &&
              out.measurement.q.allFinite() && out.measurement.qd.allFinite() &&
              std::isfinite(out.measurement.sample_time) &&
              std::abs(out.measurement.sample_time - memory.last_sample_time) <=
                  tol,
          "Manipulator: invalid cached measurement");
    }
  } else
    out.measurement_noise_state.reset();
  Eigen::VectorXd requested = Eigen::VectorXd::Zero(m.n);
  int iterations = 0;
  bool converged = true;
  double time = state.time;
  int count = 0;
  while (time < end) {
    require(++count <= 2000002,
            "Manipulator: too many sensor/integration substeps");
    double h = std::min(m.max_step, end - time);
    if (m.noise.enabled) {
      auto& memory = *out.measurement_noise_state;
      sample_if_due(x, time, memory, out.measurement);
      if (m.noise.sample_period > 0.) {
        require(memory.next_sample_time > time,
                "Manipulator: sensor period below clock resolution");
        h = std::min(h, memory.next_sample_time - time);
      }
    } else
      out.measurement = exact_measurement(x, time);
    if (mode == InputMode::Actuator) {
      setpoint = clamp(command, m);
      requested = m.allocation * command;
    } else {
      requested = command;
      if (mode == InputMode::Velocity) {
        const auto& y = out.measurement;
        requested = mass(y.q, p) * m.gains.cwiseProduct(command - y.qd);
        if (p.model_compensation)
          requested +=
              rnea(y.q, y.qd, Eigen::VectorXd::Zero(m.n), p, true, true, false);
      }
      require(requested.allFinite(), "Manipulator: nonfinite requested effort");
      setpoint = allocate(requested, outputs, p, m, iterations, converged);
    }
    const Eigen::VectorXd start_outputs = outputs;
    x = dynamics::detail::rk4_euclidean(
        x, h, [&](const Eigen::VectorXd& stage, double elapsed) {
          Eigen::VectorXd derivative(2 * m.n);
          derivative.head(m.n) = stage.tail(m.n);
          derivative.tail(m.n) =
              acceleration(stage.head(m.n), stage.tail(m.n),
                           m.allocation * actuator_output(start_outputs,
                                                          setpoint, elapsed, m),
                           p);
          return derivative;
        });
    outputs = clamp(actuator_output(start_outputs, setpoint, h, m), m);
    require(x.allFinite(),
            "Manipulator: integration became nonfinite; reduce timestep");
    const double next = time + h;
    require(next > time,
            "Manipulator: integration step below clock resolution");
    time = next;
    if (end - time <= clock_tolerance(end)) time = end;
  }
  out.q = x.head(m.n);
  out.qd = x.tail(m.n);
  out.time = end;
  out.actuator_outputs = outputs;
  out.diagnostics.requested_effort = requested;
  out.diagnostics.allocated_effort = m.allocation * setpoint;
  out.diagnostics.applied_effort = m.allocation * outputs;
  out.diagnostics.allocation_residual =
      requested - out.diagnostics.allocated_effort;
  out.diagnostics.actuator_setpoints = setpoint;
  out.diagnostics.allocation_iterations = iterations;
  out.diagnostics.allocation_converged = converged;
  out.diagnostics.saturated_actuators = 0;
  for (Eigen::Index j = 0; j < m.m; ++j)
    if (setpoint[j] <= m.lower[j] + 1.e-7 || setpoint[j] >= m.upper[j] - 1.e-7)
      ++out.diagnostics.saturated_actuators;
  if (m.noise.enabled)
    sample_if_due(x, end, *out.measurement_noise_state, out.measurement);
  else
    out.measurement = exact_measurement(x, end);
  return out;
}
}  // namespace detail

inline State make_state(const Parameters& p) {
  const auto m = detail::prepare(p);
  State state;
  state.q = state.qd = Eigen::VectorXd::Zero(m.n);
  state.actuator_outputs = Eigen::VectorXd::Zero(m.m);
  return state;
}
inline void configure_actuators(Parameters& p, const Eigen::MatrixXd& matrix,
                                const Eigen::VectorXd& lower,
                                const Eigen::VectorXd& upper,
                                double time_constant = 0.) {
  detail::require(matrix.rows() == static_cast<Eigen::Index>(p.links.size()) &&
                      matrix.cols() > 0 && lower.size() == matrix.cols() &&
                      upper.size() == matrix.cols(),
                  "Manipulator: actuator dimension mismatch");
  detail::require(std::isfinite(time_constant) && time_constant >= 0.,
                  "Manipulator: invalid lag");
  p.actuation_matrix = matrix;
  p.actuator_min = lower;
  p.actuator_max = upper;
  p.actuator_time_constants =
      Eigen::VectorXd::Constant(matrix.cols(), time_constant);
}
inline Eigen::MatrixXd mass_matrix(const Eigen::VectorXd& q,
                                   const Parameters& p) {
  const auto m = detail::prepare(p);
  detail::validate_joint_vector(q, m.n, "q");
  return detail::mass(q, p);
}
// Required APPLIED joint effort, including friction/gravity, minus supplied
// external-body-wrench contributions. No actuator saturation or allocation.
inline Eigen::VectorXd inverse_dynamics(const Eigen::VectorXd& q,
                                        const Eigen::VectorXd& qd,
                                        const Eigen::VectorXd& qdd,
                                        const Parameters& p) {
  const auto m = detail::prepare(p);
  detail::validate_joint_vector(q, m.n, "q");
  detail::validate_joint_vector(qd, m.n, "qd");
  detail::validate_joint_vector(qdd, m.n, "qdd");
  return detail::rnea(q, qd, qdd, p);
}
inline Eigen::VectorXd bias_forces(const Eigen::VectorXd& q,
                                   const Eigen::VectorXd& qd,
                                   const Parameters& p) {
  return inverse_dynamics(q, qd, Eigen::VectorXd::Zero(q.size()), p);
}
inline std::vector<Eigen::Matrix4d> forward_kinematics(const Eigen::VectorXd& q,
                                                       const Parameters& p) {
  const auto m = detail::prepare(p);
  detail::validate_joint_vector(q, m.n, "q");
  const auto zero = Eigen::VectorXd::Zero(m.n).eval();
  return detail::kinematics(q, zero, zero, p, false).poses;
}
// Actual qdd for instantaneous motor outputs, after hard saturation; no
// lag/time advance.
inline Eigen::VectorXd forward_dynamics(const State& state,
                                        const Eigen::VectorXd& outputs,
                                        const Parameters& p) {
  const auto m = detail::prepare(p);
  const auto x = detail::packed_state(state, m);
  detail::validate_joint_vector(outputs, m.m, "outputs");
  return detail::acceleration(x.head(m.n), x.tail(m.n),
                              m.allocation * detail::clamp(outputs, m), p);
}
inline State step_open_loop(const Eigen::VectorXd& actuator_setpoints,
                            const State& state, const Parameters& p) {
  const auto m = detail::prepare(p);
  detail::validate_joint_vector(actuator_setpoints, m.m, "setpoints");
  return detail::integrate(actuator_setpoints, detail::InputMode::Actuator,
                           state, p, m);
}
inline State step_effort(const Eigen::VectorXd& joint_effort,
                         const State& state, const Parameters& p) {
  const auto m = detail::prepare(p);
  detail::validate_joint_vector(joint_effort, m.n, "joint effort");
  return detail::integrate(joint_effort, detail::InputMode::Effort, state, p,
                           m);
}
inline State step(const Eigen::VectorXd& target_joint_velocity,
                  const State& state, const Parameters& p) {
  const auto m = detail::prepare(p);
  detail::validate_joint_vector(target_joint_velocity, m.n, "velocity target");
  return detail::integrate(target_joint_velocity, detail::InputMode::Velocity,
                           state, p, m);
}
}  // namespace manipulator
