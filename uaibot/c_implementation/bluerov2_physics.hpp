#pragma once
// BlueROV2 Heavy: Fossen dynamics, INERTIAL twist interface, homogeneous pose.
// C++17, Eigen >= 3.3. No ROS or simulator dependency. SI units throughout.
// Equations/nominal coefficients: see README.md and REFERENCES.md.

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace bluerov2 {

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec8 = Eigen::Matrix<double, 8, 1>;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using AllocationMatrix = Eigen::Matrix<double, 6, 8>;

inline Vec6 six(double a, double b, double c, double d, double e, double f) {
  Vec6 x;
  x << a, b, c, d, e, f;
  return x;
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
  // calibrated BlueROV model. See MEASUREMENT_NOISE.md.
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

struct Parameters {
  // Integration and velocity controller. Gains have units 1/s:
  // requested INERTIAL acceleration = gains * inertial twist error.
  double dt = 0.01;
  double max_internal_dt = 0.002;
  Eigen::VectorXd velocity_gains = six(2., 2., 2., 4., 4., 4.);
  bool model_compensation = true;
  MeasurementNoiseParameters measurement_noise;

  // Published nominal Heavy coefficients, von Benzon et al. (2022), Table A1.
  // Fields containing _body are time-invariant vehicle properties, rather
  // than state/command quantities. The simulator rotates them internally.
  double mass = 13.5;                // kg
  double gravity = 9.81;             // m/s^2; paper used 9.82
  double water_density = 1000.;      // kg/m^3 (fresh water)
  double displaced_volume = 0.0134;  // m^3, fully submerged
  Eigen::Matrix3d inertia_at_cg_body =
      Eigen::Vector3d(0.26, 0.23, 0.37).asDiagonal();  // kg m^2
  Eigen::Vector3d center_of_gravity_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d center_of_buoyancy_body = Eigen::Vector3d(0., 0., 0.01);
  Mat6 added_mass_body_at_reference_density =
      six(6.36, 7.12, 18.68, 0.189, 0.135, 0.222).asDiagonal();
  Vec6 linear_damping_body_at_reference_density =
      six(13.7, 0., 33., 0., 0.8, 0.);
  Vec6 quadratic_damping_body_at_reference_density =
      six(141., 217., 190., 1.19, 0.47, 1.5);
  double hydrodynamic_reference_density = 1000.;

  // Spatially uniform, constant, irrotational current in WORLD coordinates.
  Eigen::Vector3d current_world = Eigen::Vector3d::Zero();
  // Additional wrench ABOUT THE VEHICLE ORIGIN, expressed in INERTIAL axes.
  // It enters the plant but is not cancelled by the velocity controller.
  Vec6 external_wrench_world = Vec6::Zero();

  // Geometry matches the previously prepared mesh, x forward/y left/z up.
  // Positions are relative to the chosen body origin (nominal CG at origin).
  std::array<Eigen::Vector3d, 8> thruster_positions_body = {
      {{0.1355, -0.100, -0.0725},
       {0.1355, 0.100, -0.0725},
       {-0.1475, -0.100, -0.0725},
       {-0.1475, 0.100, -0.0725},
       {0.120, -0.218, 0.},
       {0.120, 0.218, 0.},
       {-0.120, -0.218, 0.},
       {-0.120, 0.218, 0.}}};
  std::array<Eigen::Vector3d, 8> positive_thrust_axes_body = {
      {{0.7071067811865475, 0.7071067811865475, 0.},
       {0.7071067811865475, -0.7071067811865475, 0.},
       {-0.7071067811865475, 0.7071067811865475, 0.},
       {-0.7071067811865475, -0.7071067811865475, 0.},
       {0., 0., -1.},
       {0., 0., -1.},
       {0., 0., -1.},
       {0., 0., -1.}}};
  // T200 nominal 16-V static limits: 5.25 / 4.10 kgf, converted using
  // 1 kgf = 9.80665 N. Positive means along positive_thrust_axes_body[i].
  // This sign convention is NOT an ESC/PWM wiring or propeller-spin mapping.
  Vec8 thrust_max = Vec8::Constant(5.25 * 9.80665);
  Vec8 thrust_min = Vec8::Constant(-4.10 * 9.80665);
  // Optional propeller reaction moment/thrust ratios (metres, signed).
  // Zero by default: no unsupported estimate of a T200 reaction coefficient.
  Vec8 reaction_moment_per_newton = Vec8::Zero();

  // Weighted bounded least squares, not pseudoinverse followed by clipping.
  // Weights act on INERTIAL wrench components. Torque residuals are converted
  // to comparable force scales by 1/0.25 m.
  Vec6 allocation_weights = six(1., 1., 1., 4., 4., 4.);
  double allocation_regularization = 1.e-6;
  int allocation_max_iterations = 200;
  double allocation_tolerance = 1.e-9;

  // Zero = ideal force actuator with hard saturation (memoryless interface).
  // Positive = first-order FORCE lag; use the State overload to retain forces.
  // This time constant is an optional modelling choice, not an identified
  // value.
  double thruster_time_constant = 0.;
};

struct Diagnostics {
  // All six-component wrenches use INERTIAL axes, about the vehicle origin.
  // Commands are from the last substep, expressed using the RETURNED pose.
  Vec6 requested_wrench = Vec6::Zero();
  Vec6 allocated_wrench = Vec6::Zero();
  Vec6 applied_wrench = Vec6::Zero();
  Vec6 allocation_residual = Vec6::Zero();
  Vec8 thrust_setpoints = Vec8::Zero();
  int saturated_thrusters = 0;
  int allocation_iterations = 0;
  bool allocation_converged = true;
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

struct State {
  // H = [R p; 0 0 0 1], mapping vehicle/body coordinates into INERTIAL axes.
  // Inertial z points up. Position/orientation have a single source of truth.
  Eigen::Matrix4d pose = [] {
    Eigen::Matrix4d h = Eigen::Matrix4d::Identity();
    h(2, 3) = -2.;
    return h;
  }();
  // Actual INERTIAL geometric twist at the vehicle origin, m/s and rad/s:
  // [p_dot; omega_world], with R_dot = skew(omega_world) * R.
  // This is NOT the translational component of the spatial Lie-algebra twist.
  Eigen::VectorXd twist = Eigen::VectorXd::Zero(6);
  // Needed only when thruster_time_constant > 0; keep this between calls.
  Eigen::VectorXd thruster_forces = Eigen::VectorXd::Zero(8);
  Diagnostics diagnostics;
  double time = 0.;         // true simulation time, seconds
  Measurement measurement;  // most recent feedback sample; never replaces truth
  // Retain the complete State between calls. Copying State copies the random
  // stream by value, so independent rollouts and reproducible clones are safe.
  std::optional<detail::MeasurementNoiseMemory> measurement_noise_state;
};

namespace detail {

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("BlueROV2: " + message);
}

inline void validate_pose(const Eigen::Matrix4d& pose) {
  require(pose.allFinite(), "nonfinite pose");
  require((pose.row(3) - Eigen::RowVector4d(0., 0., 0., 1.)).norm() < 1.e-12,
          "pose must have homogeneous bottom row [0 0 0 1]");
  const Eigen::Matrix3d rotation = pose.topLeftCorner<3, 3>();
  require(
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <
              1.e-6 &&
          std::abs(rotation.determinant() - 1.) < 1.e-6,
      "rotation must be a proper 3x3 rotation matrix");
}

inline void validate_state(const State& s) {
  validate_pose(s.pose);
  require(std::isfinite(s.time) && s.time >= 0.,
          "state time must be finite and nonnegative");
  require(s.twist.size() == 6 && s.twist.allFinite(),
          "actual twist must have six finite entries");
  require(s.thruster_forces.size() == 8 && s.thruster_forces.allFinite(),
          "thruster state must have eight finite entries");
}

// Rotate two geometric 3-vectors at the SAME physical point. No translation
// block: this is not the full SE(3) Adjoint/spatial-origin change of reference.
inline Mat6 orientation_map(const Eigen::Matrix3d& rotation) {
  Mat6 t = Mat6::Zero();
  t.topLeftCorner<3, 3>() = rotation;
  t.bottomRightCorner<3, 3>() = rotation;
  return t;
}

inline Vec6 transport(const Vec6& nu_body) {
  Vec6 c = Vec6::Zero();
  c.head<3>() = nu_body.tail<3>().cross(nu_body.head<3>());
  return c;  // omega x omega is identically zero.
}

struct Model {
  Mat6 rigid_mass = Mat6::Zero();
  Mat6 added_mass = Mat6::Zero();
  Mat6 mass = Mat6::Zero();
  Mat6 inverse_mass = Mat6::Zero();
  AllocationMatrix allocation = AllocationMatrix::Zero();
  Vec6 linear_damping, quadratic_damping;
};

inline Model prepare(const Parameters& p) {
  require(std::isfinite(p.dt) && p.dt > 0., "dt must be positive and finite");
  require(std::isfinite(p.max_internal_dt) && p.max_internal_dt > 0.,
          "max_internal_dt must be positive and finite");
  require(std::ceil(p.dt / p.max_internal_dt) <= 1.e6,
          "too many internal integration steps");
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
          "gains must have six nonnegative finite entries");
  require(std::isfinite(p.mass) && p.mass > 0., "mass must be positive");
  require(std::isfinite(p.gravity) && p.gravity >= 0.,
          "gravity must be nonnegative");
  require(std::isfinite(p.water_density) && p.water_density >= 0.,
          "density must be nonnegative");
  require(std::isfinite(p.displaced_volume) && p.displaced_volume >= 0.,
          "volume must be nonnegative");
  require(std::isfinite(p.hydrodynamic_reference_density) &&
              p.hydrodynamic_reference_density > 0.,
          "reference density must be positive");
  require(p.center_of_gravity_body.allFinite() &&
              p.center_of_buoyancy_body.allFinite() &&
              p.current_world.allFinite() &&
              p.external_wrench_world.allFinite(),
          "nonfinite physical vector");
  require(p.inertia_at_cg_body.allFinite() &&
              p.inertia_at_cg_body.isApprox(p.inertia_at_cg_body.transpose(),
                                            1.e-12),
          "inertia must be symmetric");
  Eigen::LLT<Eigen::Matrix3d> inertia_llt(p.inertia_at_cg_body);
  require(inertia_llt.info() == Eigen::Success,
          "CG inertia must be positive definite");
  require(p.added_mass_body_at_reference_density.allFinite() &&
              p.added_mass_body_at_reference_density.isApprox(
                  p.added_mass_body_at_reference_density.transpose(), 1.e-12),
          "added mass must be symmetric");
  Eigen::SelfAdjointEigenSolver<Mat6> added_eig(
      p.added_mass_body_at_reference_density);
  require(added_eig.info() == Eigen::Success &&
              added_eig.eigenvalues().minCoeff() >= -1.e-10,
          "added mass must be positive semidefinite; use positive inertia "
          "coefficients");
  require(
      p.linear_damping_body_at_reference_density.allFinite() &&
          p.quadratic_damping_body_at_reference_density.allFinite() &&
          (p.linear_damping_body_at_reference_density.array() >= 0.).all() &&
          (p.quadratic_damping_body_at_reference_density.array() >= 0.).all(),
      "damping must be nonnegative");
  require(p.thrust_min.allFinite() && p.thrust_max.allFinite() &&
              (p.thrust_min.array() <= 0.).all() &&
              (p.thrust_max.array() >= 0.).all(),
          "thrust bounds must contain zero");
  require(p.reaction_moment_per_newton.allFinite(),
          "nonfinite reaction-moment coefficient");
  require(p.allocation_weights.allFinite() &&
              (p.allocation_weights.array() > 0.).all(),
          "allocation weights must be positive");
  require(std::isfinite(p.allocation_regularization) &&
              p.allocation_regularization > 0. &&
              std::isfinite(p.allocation_tolerance) &&
              p.allocation_tolerance > 0. && p.allocation_max_iterations > 0,
          "invalid allocation settings");
  require(
      std::isfinite(p.thruster_time_constant) && p.thruster_time_constant >= 0.,
      "thruster time constant must be nonnegative");

  Model m;
  const Eigen::Matrix3d sg = skew(p.center_of_gravity_body);
  m.rigid_mass.topLeftCorner<3, 3>() = p.mass * Eigen::Matrix3d::Identity();
  m.rigid_mass.topRightCorner<3, 3>() = -p.mass * sg;
  m.rigid_mass.bottomLeftCorner<3, 3>() = p.mass * sg;
  m.rigid_mass.bottomRightCorner<3, 3>() =
      p.inertia_at_cg_body - p.mass * sg * sg;
  const double density_scale =
      p.water_density / p.hydrodynamic_reference_density;
  m.added_mass = density_scale * p.added_mass_body_at_reference_density;
  m.linear_damping = density_scale * p.linear_damping_body_at_reference_density;
  m.quadratic_damping =
      density_scale * p.quadratic_damping_body_at_reference_density;
  m.mass = m.rigid_mass + m.added_mass;
  Eigen::LLT<Mat6> mass_llt(m.mass);
  require(mass_llt.info() == Eigen::Success,
          "total inertia must be positive definite");
  m.inverse_mass = mass_llt.solve(Mat6::Identity());
  for (int i = 0; i < 8; ++i) {
    const auto& d = p.positive_thrust_axes_body[i];
    const auto& r = p.thruster_positions_body[i];
    require(d.allFinite() && r.allFinite() && std::abs(d.norm() - 1.) < 1.e-8,
            "thruster axes must be unit vectors and positions finite");
    m.allocation.col(i).head<3>() = d;
    m.allocation.col(i).tail<3>() =
        r.cross(d) + p.reaction_moment_per_newton[i] * d;
  }
  return m;
}

// Kirchhoff/Fossen product C_M(nu)*nu, valid for any constant symmetric M.
// Its power is identically zero: nu.dot(C_M(nu)*nu) == 0.
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
  Vec6 damping = Vec6::Zero();
  Vec6 hydrostatic =
      Vec6::Zero();  // APPLIED wrench = -g(eta) in Fossen's convention
  Vec6 added_current = Vec6::Zero();
};

inline Terms terms(const Eigen::Matrix3d& rotation, const Vec6& nu,
                   const Parameters& p, const Model& m) {
  const Eigen::Vector3d current_body = rotation.transpose() * p.current_world;
  Vec6 relative = nu;
  relative.head<3>() -= current_body;
  Terms t;
  t.coriolis = coriolis_product(m.rigid_mass, nu) +
               coriolis_product(m.added_mass, relative);
  t.damping = ((m.linear_damping.array() +
                m.quadratic_damping.array() * relative.array().abs()) *
               relative.array())
                  .matrix();
  const Eigen::Vector3d weight_body =
      rotation.transpose() * Eigen::Vector3d(0., 0., -p.mass * p.gravity);
  const Eigen::Vector3d buoyancy_body =
      rotation.transpose() *
      Eigen::Vector3d(0., 0., p.water_density * p.gravity * p.displaced_volume);
  t.hydrostatic.head<3>() = weight_body + buoyancy_body;
  t.hydrostatic.tail<3>() = p.center_of_gravity_body.cross(weight_body) +
                            p.center_of_buoyancy_body.cross(buoyancy_body);
  // A current constant in world coordinates rotates in body coordinates.
  // nu_c_dot = [-omega x v_c_body; 0], so M_A * nu_c_dot belongs on the RHS.
  Vec6 current_derivative = Vec6::Zero();
  current_derivative.head<3>() = -nu.tail<3>().cross(current_body);
  t.added_current = m.added_mass * current_derivative;
  return t;
}

inline Vec6 passive_wrench(const Terms& t) {
  return t.hydrostatic - t.coriolis - t.damping + t.added_current;
}

inline Vec8 clamp_forces(const Vec8& f, const Parameters& p) {
  return f.cwiseMax(p.thrust_min).cwiseMin(p.thrust_max);
}

inline Vec8 allocate(const Vec6& requested, const Vec8& warm_start,
                     const Parameters& p,
                     const AllocationMatrix& allocation_world, int& iterations,
                     bool* converged = nullptr) {
  // Minimize ||W(Bf-requested)||^2 + lambda*||f||^2, box-constrained.
  // Feasible primal active-set QP: solve free variables exactly, move to the
  // first blocking bound, and release bounds violating the KKT conditions.
  // This also removes warm-start nullspace thrust after a saturated command.
  const AllocationMatrix a =
      p.allocation_weights.asDiagonal() * allocation_world;
  const Vec6 b = p.allocation_weights.cwiseProduct(requested);
  Eigen::Matrix<double, 8, 8> hessian = a.transpose() * a;
  hessian.diagonal().array() += p.allocation_regularization;
  const Vec8 linear = a.transpose() * b;
  Vec8 f = clamp_forces(warm_start, p);
  // 0: free; -1: lower bound; +1: upper bound; 2: disabled/fixed actuator.
  std::array<int, 8> active{};
  for (int j = 0; j < 8; ++j) {
    if (p.thrust_max[j] == p.thrust_min[j])
      active[j] = 2;
    else if (f[j] <= p.thrust_min[j])
      active[j] = -1;
    else if (f[j] >= p.thrust_max[j])
      active[j] = 1;
  }
  iterations = 0;
  if (converged) *converged = false;
  const double kkt_tolerance =
      p.allocation_tolerance * (1. + linear.cwiseAbs().maxCoeff());
  for (int k = 0; k < p.allocation_max_iterations; ++k) {
    iterations = k + 1;
    std::array<int, 8> free{};
    int nfree = 0;
    for (int j = 0; j < 8; ++j)
      if (active[j] == 0) free[nfree++] = j;
    Vec8 candidate = f;
    if (nfree > 0) {
      Eigen::MatrixXd hff(nfree, nfree);
      Eigen::VectorXd rhs(nfree);
      for (int row = 0; row < nfree; ++row) {
        const int i = free[row];
        rhs[row] = linear[i];
        for (int j = 0; j < 8; ++j)
          if (active[j] != 0) rhs[row] -= hessian(i, j) * f[j];
        for (int column = 0; column < nfree; ++column)
          hff(row, column) = hessian(i, free[column]);
      }
      const Eigen::VectorXd solution = hff.ldlt().solve(rhs);
      if (!solution.allFinite())
        throw std::runtime_error("BlueROV2: allocator linear solve failed");
      for (int j = 0; j < nfree; ++j) candidate[free[j]] = solution[j];
    }
    const Vec8 direction = candidate - f;
    double fraction = 1.;
    int blocker = -1;
    int blocker_side = 0;
    for (int j = 0; j < 8; ++j) {
      if (active[j] != 0) continue;
      double allowed = 1.;
      int side = 0;
      if (candidate[j] < p.thrust_min[j] && direction[j] < 0.) {
        allowed = (p.thrust_min[j] - f[j]) / direction[j];
        side = -1;
      } else if (candidate[j] > p.thrust_max[j] && direction[j] > 0.) {
        allowed = (p.thrust_max[j] - f[j]) / direction[j];
        side = 1;
      }
      if (side != 0 && allowed < fraction) {
        fraction = std::max(0., allowed);
        blocker = j;
        blocker_side = side;
      }
    }
    f = clamp_forces(f + fraction * direction, p);
    if (blocker >= 0) {
      active[blocker] = blocker_side;
      f[blocker] =
          blocker_side < 0 ? p.thrust_min[blocker] : p.thrust_max[blocker];
      continue;
    }
    const Vec8 gradient = hessian * f - linear;
    int release = -1;
    double largest_violation = kkt_tolerance;
    for (int j = 0; j < 8; ++j) {
      const double violation = active[j] == -1  ? -gradient[j]
                               : active[j] == 1 ? gradient[j]
                                                : 0.;
      if (violation > largest_violation) {
        largest_violation = violation;
        release = j;
      }
    }
    if (release < 0) {
      if (converged) *converged = true;
      return f;
    }
    active[release] = 0;
  }
  return f;
}

struct InternalState {
  Eigen::Vector3d position;
  Eigen::Quaterniond q;
  Vec6 nu;
};

inline InternalState to_internal(const State& state) {
  const Eigen::Matrix3d rotation = state.pose.topLeftCorner<3, 3>();
  return {state.pose.topRightCorner<3, 1>(),
          Eigen::Quaterniond(rotation).normalized(),
          orientation_map(rotation).transpose() * state.twist};
}
struct Derivative {
  Eigen::Vector3d position;
  Eigen::Vector4d quaternion;  // Eigen coefficient order: x,y,z,w
  Vec6 nu;
};

inline Vec8 actuator_force(const Vec8& initial, const Vec8& setpoint,
                           double elapsed, double time_constant) {
  if (time_constant == 0.) return setpoint;
  return setpoint + std::exp(-elapsed / time_constant) * (initial - setpoint);
}

inline Derivative derivative(const InternalState& x, const Vec8& forces,
                             const Parameters& p, const Model& m) {
  const Eigen::Matrix3d rotation = x.q.normalized().toRotationMatrix();
  const Terms t = terms(rotation, x.nu, p, m);
  Derivative d;
  d.position = rotation * x.nu.head<3>();
  const Eigen::Quaterniond omega(0., x.nu[3], x.nu[4], x.nu[5]);
  d.quaternion = 0.5 * (x.q * omega).coeffs();
  const Vec6 external_body =
      orientation_map(rotation).transpose() * p.external_wrench_world;
  d.nu = m.inverse_mass *
         (m.allocation * forces + passive_wrench(t) + external_body);
  return d;
}

inline InternalState shifted(const InternalState& x, const Derivative& d,
                             double h) {
  InternalState y = x;
  y.position += h * d.position;
  y.q.coeffs() += h * d.quaternion;
  y.nu += h * d.nu;
  return y;
}

inline void rk4(InternalState& x, Vec8& forces, const Vec8& setpoint, double h,
                const Parameters& p, const Model& m) {
  const Vec8 f0 =
      actuator_force(forces, setpoint, 0., p.thruster_time_constant);
  const Vec8 fm =
      actuator_force(forces, setpoint, 0.5 * h, p.thruster_time_constant);
  const Vec8 f1 = actuator_force(forces, setpoint, h, p.thruster_time_constant);
  const auto k1 = derivative(x, f0, p, m);
  const auto k2 = derivative(shifted(x, k1, h * 0.5), fm, p, m);
  const auto k3 = derivative(shifted(x, k2, h * 0.5), fm, p, m);
  const auto k4 = derivative(shifted(x, k3, h), f1, p, m);
  x.position += (h / 6.) * (k1.position + 2. * k2.position + 2. * k3.position +
                            k4.position);
  x.q.coeffs() += (h / 6.) * (k1.quaternion + 2. * k2.quaternion +
                              2. * k3.quaternion + k4.quaternion);
  x.q.normalize();
  x.nu += (h / 6.) * (k1.nu + 2. * k2.nu + 2. * k3.nu + k4.nu);
  forces = clamp_forces(f1, p);
  if (!x.position.allFinite() || !x.nu.allFinite() || !x.q.coeffs().allFinite())
    throw std::runtime_error(
        "BlueROV2: integration became nonfinite; check parameters and reduce "
        "step size");
}

inline void validate_force_state(const State& state, const Parameters& p) {
  if (p.thruster_time_constant > 0.) {
    require((state.thruster_forces.array() >= p.thrust_min.array() - 1.e-10)
                    .all() &&
                (state.thruster_forces.array() <= p.thrust_max.array() + 1.e-10)
                    .all(),
            "initial actuator forces are outside the configured limits");
  }
}

inline Diagnostics diagnose(const InternalState& x, const Vec6& requested_body,
                            const Vec8& setpoint, const Vec8& forces,
                            const Parameters& p, const Model& m,
                            int iterations = 0, bool converged = true) {
  const Mat6 t = orientation_map(x.q.toRotationMatrix());
  Diagnostics d;
  d.requested_wrench = t * requested_body;
  d.allocated_wrench = t * m.allocation * setpoint;
  d.applied_wrench = t * m.allocation * forces;
  d.allocation_residual = d.requested_wrench - d.allocated_wrench;
  d.thrust_setpoints = setpoint;
  d.allocation_iterations = iterations;
  d.allocation_converged = converged;
  for (int i = 0; i < 8; ++i)
    if (setpoint[i] <= p.thrust_min[i] + 1.e-7 ||
        setpoint[i] >= p.thrust_max[i] - 1.e-7)
      ++d.saturated_thrusters;
  return d;
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

inline State finish(const InternalState& x, const Vec8& forces,
                    const Diagnostics& diagnostic, double time = 0.) {
  require(std::isfinite(time), "simulation clock overflow");
  State out;
  out.pose = Eigen::Matrix4d::Identity();
  out.pose.topRightCorner<3, 1>() = x.position;
  out.pose.topLeftCorner<3, 3>() = x.q.toRotationMatrix();
  out.twist = orientation_map(x.q.toRotationMatrix()) * x.nu;
  out.thruster_forces = forces;
  out.diagnostics = diagnostic;
  out.time = time;
  out.measurement = exact_measurement(x, time);
  return out;
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
    for (int j = 0; j < 12; ++j) {
      const double r = elapsed / n.correlation_times[j];
      const double a = std::exp(-r);
      const double innovation_std =
          std::sqrt(n.colored_variances[j]) * std::sqrt(-std::expm1(-2. * r));
      memory.colored_error[j] =
          a * memory.colored_error[j] +
          innovation_std * memory.normal(memory.generator);
    }
  } else {
    memory.epoch = time;
    for (int j = 0; j < 12; ++j) {
      memory.colored_error[j] = n.stationary_initialization
                                    ? std::sqrt(n.colored_variances[j]) *
                                          memory.normal(memory.generator)
                                    : 0.;
    }
    memory.initialized = true;
  }
  Vec12 e = memory.colored_error;
  for (int j = 0; j < 12; ++j)
    e[j] += std::sqrt(n.white_variances[j]) * memory.normal(memory.generator);
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

inline State step_with_measurements(const Vec6& target,
                                    const Vec8& open_loop_setpoint,
                                    bool closed_loop, const State& state,
                                    const Parameters& p, const Model& m) {
  State sensor = state;
  if (!sensor.measurement_noise_state ||
      !same_noise_settings(sensor.measurement_noise_state->configuration,
                           p.measurement_noise))
    sensor.measurement_noise_state.emplace(p.measurement_noise);
  auto& memory = *sensor.measurement_noise_state;
  const double end = state.time + p.dt;
  require(std::isfinite(end) && end > state.time,
          "dt is outside simulation clock resolution");
  const double tol = clock_tolerance(state.time);
  if (memory.initialized) {
    require(memory.last_sample_time <= state.time + tol &&
                (p.measurement_noise.sample_period == 0. ||
                 memory.next_sample_time >= state.time - tol),
            "inconsistent measurement clock; reset noise state after changing "
            "simulation time");
    require(sensor.measurement.valid,
            "missing cached measurement; reset noise state");
    validate_pose(sensor.measurement.pose);
    require(sensor.measurement.twist.size() == 6 &&
                sensor.measurement.twist.allFinite() &&
                std::isfinite(sensor.measurement.sample_time) &&
                std::abs(sensor.measurement.sample_time -
                         memory.last_sample_time) <= tol,
            "invalid cached measurement; reset noise state");
  }
  auto x = to_internal(state);  // TRUE state, never replaced by a measurement
  Vec8 forces = clamp_forces(state.thruster_forces, p);
  Vec8 setpoint = clamp_forces(open_loop_setpoint, p);
  Vec6 requested_body = m.allocation * open_loop_setpoint;
  int iterations = 0;
  bool converged = true;
  double time = state.time;
  int count = 0;
  while (time < end) {
    if (++count > 2000002)
      throw std::runtime_error(
          "BlueROV2: too many sensor/integration substeps");
    sample_if_due(x, time, memory, sensor.measurement);
    double h = std::min(p.max_internal_dt, end - time);
    if (p.measurement_noise.sample_period > 0.) {
      require(memory.next_sample_time > time,
              "measurement period is below clock resolution");
      h = std::min(h, memory.next_sample_time - time);
    }
    if (closed_loop) {
      // All feedback, including nominal model compensation if enabled,
      // uses the measured pose/twist. The dynamics below use TRUE x.
      const Eigen::Matrix3d measured_rotation =
          sensor.measurement.pose.topLeftCorner<3, 3>();
      const Mat6 t = orientation_map(measured_rotation);
      const Vec6 measured_body = t.transpose() * sensor.measurement.twist;
      const Vec6 acceleration_world =
          p.velocity_gains.cwiseProduct(target - sensor.measurement.twist);
      const Vec6 acceleration_body =
          t.transpose() * acceleration_world - transport(measured_body);
      requested_body = m.mass * acceleration_body;
      if (p.model_compensation)
        requested_body -=
            passive_wrench(terms(measured_rotation, measured_body, p, m));
      setpoint = allocate(t * requested_body, forces, p, t * m.allocation,
                          iterations, &converged);
    }
    rk4(x, forces, setpoint, h, p, m);
    const double next = time + h;
    require(next > time, "integration step is below clock resolution");
    time = next;
    if (end - time <= clock_tolerance(end)) time = end;
  }
  sample_if_due(x, end, memory, sensor.measurement);
  State out = finish(x, forces,
                     diagnose(x, requested_body, setpoint, forces, p, m,
                              iterations, converged),
                     end);
  out.measurement = std::move(sensor.measurement);
  out.measurement_noise_state = std::move(sensor.measurement_noise_state);
  return out;
}

}  // namespace detail

// Both matrices below act on INERTIAL components and depend on the pose.
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

// Time derivative of the INERTIAL geometric twist, including rotating-frame
// transport. Useful for custom controllers and physical checks.
inline Vec6 inertial_acceleration(const State& state,
                                  const Eigen::VectorXd& thrust_forces,
                                  const Parameters& p = Parameters()) {
  detail::validate_state(state);
  detail::require(thrust_forces.size() == 8 && thrust_forces.allFinite(),
                  "forces must have eight finite entries");
  const auto m = detail::prepare(p);
  const Vec8 forces = detail::clamp_forces(thrust_forces, p);
  const auto x = detail::to_internal(state);
  const auto d = detail::derivative(x, forces, p, m);
  return detail::orientation_map(x.q.toRotationMatrix()) *
         (d.nu + detail::transport(x.nu));
}

// Integrate eight force SETPOINTS, with hard limits and optional actuator lag.
inline State step_open_loop(const Eigen::VectorXd& thrust_setpoints,
                            const State& state,
                            const Parameters& p = Parameters()) {
  detail::validate_state(state);
  detail::require(thrust_setpoints.size() == 8 && thrust_setpoints.allFinite(),
                  "setpoints must have eight finite entries");
  const auto m = detail::prepare(p);
  detail::validate_force_state(state, p);
  if (p.measurement_noise.enabled)
    return detail::step_with_measurements(Vec6::Zero(), thrust_setpoints, false,
                                          state, p, m);
  auto x = detail::to_internal(state);
  Vec8 forces = detail::clamp_forces(state.thruster_forces, p);
  const Vec8 setpoint = detail::clamp_forces(thrust_setpoints, p);
  const int count = static_cast<int>(std::ceil(p.dt / p.max_internal_dt));
  const double h = p.dt / count;
  for (int i = 0; i < count; ++i) detail::rk4(x, forces, setpoint, h, p, m);
  return detail::finish(x, forces,
                        detail::diagnose(x, m.allocation * thrust_setpoints,
                                         setpoint, forces, p, m),
                        state.time + p.dt);
}

// Main API. Return the entire State to the next call when motor lag is enabled.
// Target and actual twist are ALWAYS INERTIAL geometric velocities at the
// vehicle origin. Target is held constant in inertial axes during this call.
inline State step(const Eigen::VectorXd& target_twist, const State& state,
                  const Parameters& p = Parameters()) {
  detail::validate_state(state);
  detail::require(target_twist.size() == 6 && target_twist.allFinite(),
                  "target twist must have six finite entries");
  const auto m = detail::prepare(p);
  detail::validate_force_state(state, p);
  if (p.measurement_noise.enabled)
    return detail::step_with_measurements(target_twist, Vec8::Zero(), true,
                                          state, p, m);
  auto x = detail::to_internal(state);
  Vec8 forces = detail::clamp_forces(state.thruster_forces, p);
  Vec6 requested_body = Vec6::Zero();
  Vec8 setpoint = Vec8::Zero();
  int iterations = 0;
  bool converged = false;
  const int count = static_cast<int>(std::ceil(p.dt / p.max_internal_dt));
  const double h = p.dt / count;
  for (int i = 0; i < count; ++i) {
    const Eigen::Matrix3d rotation = x.q.toRotationMatrix();
    const Mat6 t = detail::orientation_map(rotation);
    const Vec6 actual_world = t * x.nu;
    const Vec6 acceleration_world =
        p.velocity_gains.cwiseProduct(target_twist - actual_world);
    // nu_world_dot = T * (nu_body_dot + [omega_b x v_b; 0]).
    // Apply gains in INERTIAL axes before transforming acceleration.
    const Vec6 acceleration_body =
        t.transpose() * acceleration_world - detail::transport(x.nu);
    requested_body = m.mass * acceleration_body;
    if (p.model_compensation)
      requested_body -=
          detail::passive_wrench(detail::terms(rotation, x.nu, p, m));
    setpoint = detail::allocate(t * requested_body, forces, p, t * m.allocation,
                                iterations, &converged);
    detail::rk4(x, forces, setpoint, h, p, m);
  }
  return detail::finish(x, forces,
                        detail::diagnose(x, requested_body, setpoint, forces, p,
                                         m, iterations, converged),
                        state.time + p.dt);
}

// Requested stateless interface. No actuator memory is hidden in the
// parameters.
inline State step(const Eigen::VectorXd& target_twist,
                  const Eigen::Matrix4d& pose,
                  const Eigen::VectorXd& current_twist,
                  const Parameters& p = Parameters()) {
  detail::require(p.thruster_time_constant == 0.,
                  "motor lag needs previous thruster forces: use step(target, "
                  "State, parameters)");
  detail::require(!p.measurement_noise.enabled,
                  "measurement noise needs persistent random/sensor state: use "
                  "step(target, State, parameters)");
  State state;
  state.pose = pose;
  state.twist = current_twist;
  return step(target_twist, state, p);
}

// Convenience overload accepting precisely gains and dt as separate arguments.
inline State step(const Eigen::VectorXd& target_twist,
                  const Eigen::Matrix4d& pose,
                  const Eigen::VectorXd& current_twist,
                  const Eigen::VectorXd& velocity_gains, double dt) {
  Parameters p;
  p.velocity_gains = velocity_gains;
  p.dt = dt;
  return step(target_twist, pose, current_twist, p);
}

}  // namespace bluerov2
