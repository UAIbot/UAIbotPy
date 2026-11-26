#pragma once
// Standard-DH bridge. The dynamics link frame is BEFORE the fixed DH tail.
#include "manipulator_physics.hpp"

namespace manipulator {
enum class InertiaReference { CenterOfMass, DHOrigin };

// Rows [theta, d, alpha, a, joint_type], where 0=revolute, 1=prismatic.
// As in Uaibot's CPP_Manipulator::fk, q REPLACES theta (revolute) or d
// (prismatic); the corresponding DH-table entry is not an additive offset.
// COM and inertia inputs use each AFTER-transform DH frame. Gravity uses the
// fixed robot base, and base_to_dh0 is the transform from DH0 to that base.
inline Parameters parameters_from_dh(
    const Eigen::MatrixXd& table, const Eigen::VectorXd& masses,
    const Eigen::MatrixXd& com_dh,
    const std::vector<Eigen::Matrix3d>& inertias_dh,
    const Eigen::Matrix4d& base_to_dh0 = Eigen::Matrix4d::Identity(),
    const Eigen::Vector3d& gravity_base = Eigen::Vector3d(0., 0., -9.81),
    InertiaReference reference = InertiaReference::CenterOfMass) {
  const Eigen::Index n = table.rows();
  detail::require(n > 0 && n <= 128 && table.cols() == 5 && table.allFinite(),
                  "Manipulator: DH table must be finite n-by-5, 1 <= n <= 128");
  detail::require(masses.size() == n && com_dh.rows() == n &&
                      com_dh.cols() == 3 && masses.allFinite() &&
                      com_dh.allFinite() &&
                      inertias_dh.size() == std::size_t(n),
                  "Manipulator: inconsistent DH inertial data");
  detail::require(reference == InertiaReference::CenterOfMass ||
                      reference == InertiaReference::DHOrigin,
                  "Manipulator: invalid inertia reference");
  Parameters p;
  p.gravity_base = gravity_base;
  p.links.reserve(static_cast<std::size_t>(n));
  Eigen::Matrix4d previous_tail = base_to_dh0;
  for (Eigen::Index i = 0; i < n; ++i) {
    const double kind = table(i, 4);
    detail::require(kind == 0. || kind == 1.,
                    "Manipulator: DH joint_type must be 0 or 1");
    Link link;
    link.joint_type = kind == 0. ? JointType::Revolute : JointType::Prismatic;
    link.parent_to_joint = previous_tail;
    Eigen::Matrix4d tail = Eigen::Matrix4d::Identity();
    const double theta = kind == 0. ? 0. : table(i, 0);
    const double d = kind == 1. ? 0. : table(i, 1);
    tail.topLeftCorner<3, 3>() =
        Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        Eigen::AngleAxisd(table(i, 2), Eigen::Vector3d::UnitX())
            .toRotationMatrix();
    tail.topRightCorner<3, 1>() = Eigen::Vector3d(
        table(i, 3) * std::cos(theta), table(i, 3) * std::sin(theta), d);
    const Eigen::Matrix3d rotation = tail.topLeftCorner<3, 3>();
    const Eigen::Vector3d com = com_dh.row(i).transpose();
    link.mass = masses[i];
    link.center_of_mass_body = rotation * com + tail.topRightCorner<3, 1>();
    Eigen::Matrix3d inertia = inertias_dh[static_cast<std::size_t>(i)];
    if (reference == InertiaReference::DHOrigin)
      inertia += link.mass * dynamics::skew(com) * dynamics::skew(com);
    link.inertia_at_com_body = rotation * inertia * rotation.transpose();
    p.links.push_back(link);
    previous_tail = tail;
  }
  detail::prepare(p);  // Validate physical inertias and converted geometry.
  return p;
}
}  // namespace manipulator
