#pragma once
// Shared numerical integration, SE(3) state and sampled measurement noise.
// C++17, Eigen >= 3.3. Model-specific equations remain in their own headers.
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
#include <vector>

namespace dynamics {
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

inline Vec6 six(double a, double b, double c, double d, double e, double f) {
  Vec6 v;
  v << a, b, c, d, e, f;
  return v;
}

inline Eigen::Matrix3d skew(const Eigen::Vector3d& x) {
  Eigen::Matrix3d s;
  s << 0., -x.z(), x.y(), x.z(), 0., -x.x(), -x.y(), x.x(), 0.;
  return s;
}

struct MeasurementNoiseParameters {
  bool enabled = false;
  // Variances, NOT standard deviations or spectral noise densities.
  // Order: [position(3), rotation-vector error(3), linear velocity(3),
  //         angular velocity(3)]. Position/rotation errors use inertial axes;
  // velocity errors use aligned sensor/body axes. All outputs are inertial.
  // Units: [m^2, rad^2, (m/s)^2, (rad/s)^2], three entries per group.
  // These defaults are illustrative aided-estimator assumptions, not a
  // calibrated vehicle sensor model. See README.md.
  Vec12 white_variances = [] {
    Vec12 v;
    v << 4.e-4, 4.e-4, 9.e-6, 4.e-6, 4.e-6, 25.e-6, 25.e-6, 25.e-6, 25.e-6,
        1.e-6, 1.e-6, 1.e-6;
    return v;
  }();
  Vec12 colored_variances = [] {
    Vec12 v;
    v << 2.5e-3, 2.5e-3, 1.e-4, 25.e-6, 25.e-6, 4.e-4, 1.e-4, 1.e-4, 1.e-4,
        9.e-6, 9.e-6, 9.e-6;
    return v;
  }();
  Vec12 correlation_times = [] {
    Vec12 t;
    t << 10., 10., 10., 1., 1., 5., .2, .2, .2, 30., 30., 30.;
    return t;
  }();  // seconds, one per colored error channel
  double sample_period = 0.01;  // seconds; 0 = sample at every internal step
  std::uint64_t seed = 42;
  bool stationary_initialization = true;
};
struct Measurement {
  // Noisy estimated pose and geometric twist, both in INERTIAL coordinates.
  // Held between sensor updates; sample_time is the measurement timestamp.
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  Eigen::VectorXd twist = Eigen::VectorXd::Zero(6);
  double sample_time = 0.;
  bool valid = false;
};

namespace detail {
struct MeasurementNoiseMemory {
  explicit MeasurementNoiseMemory(const MeasurementNoiseParameters& p)
      : configuration(p), generator(p.seed) {}
  MeasurementNoiseParameters configuration;
  std::mt19937_64 generator;
  std::normal_distribution<double> normal{0., 1.};
  Vec12 colored_error = Vec12::Zero();
  double epoch = 0., last_sample_time = 0., next_sample_time = 0.;
  std::uint64_t sample_count = 0;
  bool initialized = false;
};
}  // namespace detail

// Common physical state for free-floating AUV/UAV bodies. Joint coordinates
// have their own state because their configuration space is not SE(3).
struct RigidBodyState {
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  Eigen::VectorXd twist = Eigen::VectorXd::Zero(6);
  double time = 0.;
  Measurement measurement;
  std::optional<detail::MeasurementNoiseMemory> measurement_noise_state;
};

namespace detail {
inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("Dynamics: " + message);
}
inline void validate_pose(const Eigen::Matrix4d& pose) {
  require(pose.allFinite(), "nonfinite pose");
  require((pose.row(3) - Eigen::RowVector4d(0., 0., 0., 1.)).norm() < 1.e-12,
          "pose must have homogeneous bottom row [0 0 0 1]");
  const Eigen::Matrix3d r = pose.topLeftCorner<3, 3>();
  require((r.transpose() * r - Eigen::Matrix3d::Identity()).norm() < 1.e-6 &&
              std::abs(r.determinant() - 1.) < 1.e-6,
          "rotation must be a proper 3x3 rotation matrix");
}
// Rotation of two geometric vectors at the SAME point, not an SE(3) Adjoint.
inline Mat6 orientation_map(const Eigen::Matrix3d& rotation) {
  Mat6 t = Mat6::Zero();
  t.topLeftCorner<3, 3>() = rotation;
  t.bottomRightCorner<3, 3>() = rotation;
  return t;
}
inline Vec6 transport(const Vec6& nu) {
  Vec6 c = Vec6::Zero();
  c.head<3>() = nu.tail<3>().cross(nu.head<3>());
  return c;
}
struct InternalState {
  Eigen::Vector3d position;
  Eigen::Quaterniond q;
  Vec6 nu;
};
inline InternalState to_internal(const RigidBodyState& state) {
  const Eigen::Matrix3d r = state.pose.topLeftCorner<3, 3>();
  return {state.pose.topRightCorner<3, 1>(), Eigen::Quaterniond(r).normalized(),
          orientation_map(r).transpose() * state.twist};
}
struct Derivative {
  Eigen::Vector3d position;
  Eigen::Vector4d quaternion;  // Eigen order: x,y,z,w
  Vec6 nu;
};
inline InternalState shifted(const InternalState& x, const Derivative& d,
                             double h) {
  InternalState y = x;
  y.position += h * d.position;
  y.q.coeffs() += h * d.quaternion;
  y.nu += h * d.nu;
  return y;
}

// These generic channel updates are shared by rigid-pose and joint encoders.
// Random draws and floating-point evaluation order match the original sensor.
// Settings/shape validation is performed once by the model at API entry.
template <class ErrorVector, class VarianceVector, class TimeVector,
          class Normal, class Generator>
inline void advance_colored_error(ErrorVector& error,
                                  const VarianceVector& variances,
                                  const TimeVector& correlation_times,
                                  double elapsed, Normal& normal,
                                  Generator& generator) {
  require(error.size() == variances.size() &&
              error.size() == correlation_times.size(),
          "colored-noise dimensions must agree");
  require(std::isfinite(elapsed) && elapsed >= 0.,
          "invalid colored-noise elapsed time");
  for (Eigen::Index j = 0; j < error.size(); ++j) {
    const double r = elapsed / correlation_times[j];
    const double a = std::exp(-r);
    const double innovation_std =
        std::sqrt(variances[j]) * std::sqrt(-std::expm1(-2. * r));
    error[j] = a * error[j] + innovation_std * normal(generator);
  }
}
template <class ErrorVector, class VarianceVector, class Normal,
          class Generator>
inline void initialize_colored_error(ErrorVector& error,
                                     const VarianceVector& variances,
                                     bool stationary, Normal& normal,
                                     Generator& generator) {
  require(error.size() == variances.size(),
          "colored-noise dimensions must agree");
  for (Eigen::Index j = 0; j < error.size(); ++j)
    error[j] = stationary ? std::sqrt(variances[j]) * normal(generator) : 0.;
}
template <class ErrorVector, class VarianceVector, class Normal,
          class Generator>
inline void add_white_error(ErrorVector& error, const VarianceVector& variances,
                            Normal& normal, Generator& generator) {
  require(error.size() == variances.size(),
          "white-noise dimensions must agree");
  for (Eigen::Index j = 0; j < error.size(); ++j)
    error[j] += std::sqrt(variances[j]) * normal(generator);
}

// Nonautonomous callback f(state, elapsed_in_substep). Actuator forces may be
// evaluated analytically at stage times without integrating artificial states.
template <class Function>
inline Eigen::VectorXd rk4_euclidean(const Eigen::VectorXd& x, double h,
                                     Function&& f) {
  const Eigen::VectorXd k1 = f(x, 0.);
  require(k1.size() == x.size(), "RK4 derivative dimension mismatch");
  const Eigen::VectorXd k2 = f((x + .5 * h * k1).eval(), .5 * h);
  require(k2.size() == x.size(), "RK4 derivative dimension mismatch");
  const Eigen::VectorXd k3 = f((x + .5 * h * k2).eval(), .5 * h);
  require(k3.size() == x.size(), "RK4 derivative dimension mismatch");
  const Eigen::VectorXd k4 = f((x + h * k3).eval(), h);
  require(k4.size() == x.size(), "RK4 derivative dimension mismatch");
  Eigen::VectorXd out = x + (h / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
  if (!out.allFinite())
    throw std::runtime_error(
        "Dynamics: integration became nonfinite; reduce step size");
  return out;
}
template <class Function>
inline void rk4_rigid_body(InternalState& x, double h, Function&& f) {
  const auto k1 = f(x, 0.);
  const auto k2 = f(shifted(x, k1, .5 * h), .5 * h);
  const auto k3 = f(shifted(x, k2, .5 * h), .5 * h);
  const auto k4 = f(shifted(x, k3, h), h);
  x.position += (h / 6.) * (k1.position + 2. * k2.position + 2. * k3.position +
                            k4.position);
  x.q.coeffs() += (h / 6.) * (k1.quaternion + 2. * k2.quaternion +
                              2. * k3.quaternion + k4.quaternion);
  x.q.normalize();
  x.nu += (h / 6.) * (k1.nu + 2. * k2.nu + 2. * k3.nu + k4.nu);
  if (!x.position.allFinite() || !x.nu.allFinite() || !x.q.coeffs().allFinite())
    throw std::runtime_error(
        "Dynamics: integration became nonfinite; check parameters and reduce "
        "step size");
}
inline Measurement exact_measurement(const InternalState& x, double time) {
  Measurement y;
  y.pose.topLeftCorner<3, 3>() = x.q.toRotationMatrix();
  y.pose.topRightCorner<3, 1>() = x.position;
  y.twist = orientation_map(x.q.toRotationMatrix()) * x.nu;
  y.sample_time = time;
  y.valid = true;
  return y;
}
inline bool same_noise_settings(const MeasurementNoiseParameters& a,
                                const MeasurementNoiseParameters& b) {
  return a.seed == b.seed && a.sample_period == b.sample_period &&
         a.stationary_initialization == b.stationary_initialization &&
         (a.white_variances.array() == b.white_variances.array()).all() &&
         (a.colored_variances.array() == b.colored_variances.array()).all() &&
         (a.correlation_times.array() == b.correlation_times.array()).all();
}

inline double clock_tolerance(double time) {
  return 32. * std::numeric_limits<double>::epsilon() *
         std::max(1., std::abs(time));
}

// Update an estimated-state sensor at its own clock. Its sample is held between
// updates. Position errors do not enter the physical integrator. Rotation
// errors act on SO(3); the shared rotation perturbation couples attitude and
// twist.
inline void sample_if_due(const InternalState& truth, double time,
                          MeasurementNoiseMemory& memory, Measurement& y) {
  const auto& n = memory.configuration;
  const double tol = clock_tolerance(time);
  if (memory.initialized) {
    if (n.sample_period > 0. && time < memory.next_sample_time - tol) return;
    if (n.sample_period == 0. && time <= memory.last_sample_time + tol) return;
    const double elapsed = time - memory.last_sample_time;
    require(elapsed >= 0.,
            "measurement clock moved backwards; reset noise state");
    advance_colored_error(memory.colored_error, n.colored_variances,
                          n.correlation_times, elapsed, memory.normal,
                          memory.generator);
  } else {
    memory.epoch = time;
    initialize_colored_error(memory.colored_error, n.colored_variances,
                             n.stationary_initialization, memory.normal,
                             memory.generator);
    memory.initialized = true;
  }
  Vec12 e = memory.colored_error;
  add_white_error(e, n.white_variances, memory.normal, memory.generator);
  const Eigen::Vector3d theta = e.segment<3>(3);
  const double angle = theta.norm();
  Eigen::Matrix3d error_rotation = Eigen::Matrix3d::Identity();
  if (angle > 0.)
    error_rotation = Eigen::AngleAxisd(angle, theta / angle).toRotationMatrix();
  y = exact_measurement(truth, time);
  y.pose.topRightCorner<3, 1>() += e.head<3>();
  y.pose.topLeftCorner<3, 3>() = error_rotation * truth.q.toRotationMatrix();
  const Eigen::Matrix3d measured_rotation = y.pose.topLeftCorner<3, 3>();
  // Body/sensor errors rotate into the inertial frame through the SAME
  // measured attitude, creating attitude/velocity cross-correlations.
  y.twist.head<3>() =
      measured_rotation * (truth.nu.head<3>() + e.segment<3>(6));
  y.twist.tail<3>() = measured_rotation * (truth.nu.tail<3>() + e.tail<3>());
  require(y.pose.allFinite() && y.twist.allFinite(),
          "nonfinite measurement; check noise variances");
  memory.last_sample_time = time;
  ++memory.sample_count;
  memory.next_sample_time =
      memory.epoch + double(memory.sample_count) * n.sample_period;
}

// Weighted, box-constrained linear allocation shared by all model families.
inline Eigen::VectorXd bounded_least_squares(
    const Eigen::MatrixXd& matrix, const Eigen::VectorXd& requested,
    const Eigen::VectorXd& warm_start, const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper, const Eigen::VectorXd& weights,
    double regularization, int max_iterations, double tolerance_setting,
    int& iterations, bool& converged) {
  const Eigen::Index n = matrix.cols();
  require(n > 0 && matrix.rows() > 0 && matrix.rows() == requested.size() &&
              lower.size() == n && upper.size() == n &&
              warm_start.size() == n && weights.size() == requested.size(),
          "allocation dimensions must agree");
  require(matrix.allFinite() && requested.allFinite() &&
              warm_start.allFinite() && lower.allFinite() &&
              upper.allFinite() && weights.allFinite() &&
              (lower.array() <= upper.array()).all() &&
              (weights.array() > 0.).all(),
          "invalid allocation vectors or bounds");
  require(std::isfinite(regularization) && regularization > 0. &&
              std::isfinite(tolerance_setting) && tolerance_setting > 0. &&
              max_iterations > 0,
          "invalid allocation solver settings");
  const auto clamp = [&](const Eigen::VectorXd& value) {
    return value.cwiseMax(lower).cwiseMin(upper).eval();
  };
  // Box-constrained weighted least squares, solved by a primal active-set QP.
  // Minimize ||W(Bu-requested)||^2 + lambda*||u||^2, including rank-deficient
  // B.
  const Eigen::MatrixXd a = weights.asDiagonal() * matrix;
  const Eigen::VectorXd b = weights.cwiseProduct(requested);
  Eigen::MatrixXd hessian = a.transpose() * a;
  hessian.diagonal().array() += regularization;
  const Eigen::VectorXd linear = a.transpose() * b;
  Eigen::VectorXd u = clamp(warm_start);
  require(hessian.allFinite() && linear.allFinite(),
          "allocation scaling overflow");
  std::vector<int> active(static_cast<std::size_t>(n), 0);
  for (Eigen::Index j = 0; j < n; ++j) {
    if (upper[j] == lower[j])
      active[j] = 2;
    else if (u[j] <= lower[j])
      active[j] = -1;
    else if (u[j] >= upper[j])
      active[j] = 1;
  }
  iterations = 0;
  converged = false;
  const double tolerance =
      tolerance_setting * (1. + linear.cwiseAbs().maxCoeff());
  for (int k = 0; k < max_iterations; ++k) {
    iterations = k + 1;
    std::vector<Eigen::Index> free;
    for (Eigen::Index j = 0; j < n; ++j)
      if (active[j] == 0) free.push_back(j);
    Eigen::VectorXd candidate = u;
    const Eigen::Index nf = static_cast<Eigen::Index>(free.size());
    if (nf > 0) {
      Eigen::MatrixXd hff(nf, nf);
      Eigen::VectorXd rhs(nf);
      for (Eigen::Index row = 0; row < nf; ++row) {
        const Eigen::Index i = free[row];
        rhs[row] = linear[i];
        for (Eigen::Index j = 0; j < n; ++j)
          if (active[j] != 0) rhs[row] -= hessian(i, j) * u[j];
        for (Eigen::Index col = 0; col < nf; ++col)
          hff(row, col) = hessian(i, free[col]);
      }
      const Eigen::VectorXd solution = hff.ldlt().solve(rhs);
      if (!solution.allFinite())
        throw std::runtime_error("Dynamics: allocation solve failed");
      for (Eigen::Index j = 0; j < nf; ++j) candidate[free[j]] = solution[j];
    }
    const Eigen::VectorXd direction = candidate - u;
    double fraction = 1.;
    Eigen::Index blocker = -1;
    int blocker_side = 0;
    for (Eigen::Index j = 0; j < n; ++j) {
      if (active[j] != 0) continue;
      double allowed = 1.;
      int side = 0;
      if (candidate[j] < lower[j] && direction[j] < 0.) {
        allowed = (lower[j] - u[j]) / direction[j];
        side = -1;
      } else if (candidate[j] > upper[j] && direction[j] > 0.) {
        allowed = (upper[j] - u[j]) / direction[j];
        side = 1;
      }
      if (side != 0 && allowed < fraction) {
        fraction = std::max(0., allowed);
        blocker = j;
        blocker_side = side;
      }
    }
    u = clamp(u + fraction * direction);
    if (blocker >= 0) {
      active[blocker] = blocker_side;
      u[blocker] = blocker_side < 0 ? lower[blocker] : upper[blocker];
      continue;
    }
    const Eigen::VectorXd gradient = hessian * u - linear;
    Eigen::Index release = -1;
    double violation_max = tolerance;
    for (Eigen::Index j = 0; j < n; ++j) {
      const double violation = active[j] == -1  ? -gradient[j]
                               : active[j] == 1 ? gradient[j]
                                                : 0.;
      if (violation > violation_max) {
        violation_max = violation;
        release = j;
      }
    }
    if (release < 0) {
      converged = true;
      return u;
    }
    active[release] = 0;
  }
  return u;  // feasible iterate; caller can inspect allocation_converged
}

}  // namespace detail
}  // namespace dynamics
