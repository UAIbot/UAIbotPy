// ne_newton_euler.cpp
// #include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "declarations.h"

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::MatrixXf;
using Eigen::Vector3f;
using Eigen::VectorXf;

Matrix3f create_inertia(double Ixx, double Ixy, double Ixz, double Iyy,
                        double Iyz, double Izz) {
  Matrix3f I;
  I << Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz;
  return I;
}

// --- Newton-Euler implementation (translated) ---
VectorXf recursiveNewtonEuler(const VectorXf &q, const VectorXf &qdot,
                              const VectorXf &qddot, const Vector3f &g0,
                              const Manipulator &robot) {
  const size_t n = q.size();
  // Ensure robot.links has at least n entries
  if (robot.no_links != n) {
    throw std::invalid_argument(
        "Robot number of links does not match size of q."
        " robot.no_links: " +
        std::to_string(robot.no_links) +
        ", q.size(): " + std::to_string(q.size()));
  }

  // masses and inertias — prefer reading from robot, fall back to Get... for
  // demo
  std::vector<float> m = robot.masses;
  // Inertia tensors w.r.t. center of mass
  std::vector<Matrix3f> I = robot.inertia_tensors;
  // Center of mass positions expressed as displacements w.r.t. DH frame
  std::vector<Vector3f> r_cms = robot.com_positions;
  // allocate
  MatrixXf w_i = MatrixXf::Zero(3, n);
  MatrixXf wp_i = MatrixXf::Zero(3, n);
  MatrixXf a_i = MatrixXf::Zero(3, n);
  MatrixXf r_i_im1_i = MatrixXf::Zero(3, n);
  MatrixXf r_i_i_cmi = MatrixXf::Zero(3, n);
  MatrixXf a_ci = MatrixXf::Zero(3, n);

  Vector3f w_i0 = Vector3f::Zero();
  Vector3f wp_i0 = Vector3f::Zero();
  Vector3f a_i0 = Vector3f::Zero();

  // FK transforms
  std::vector<Matrix4f> T = robot.fk(q, robot.htm_world_to_dh0, false).htm_dh;

  for (size_t i = 0; i < n; i++) {
    Matrix3f R_im1_0;
    Vector3f r_0_im1;
    Vector3f z_im1_im1;
    if (i == 0) {
      R_im1_0 = Matrix3f::Identity();
      r_0_im1 = Vector3f::Zero();
      z_im1_im1 = Vector3f(0, 0, 1);
    } else {
      R_im1_0 = T[i - 1].block<3, 3>(0, 0);
      r_0_im1 = T[i - 1].block<3, 1>(0, 3);
      z_im1_im1 = R_im1_0.transpose() * T[i - 1].block<3, 1>(0, 2);
    }

    Matrix3f R_i_im1 = T[i].block<3, 3>(0, 0).transpose() * R_im1_0;

    // w_i[:, i] = R_i_im1 * (w_i0 + qp[i] * z_im1_im1)
    Vector3f tmp_w = w_i0 + qdot[i] * z_im1_im1;
    w_i.col(i) = R_i_im1 * tmp_w;

    // wp_i[:, i] = R_i_im1 * (wp_i0 + qpp[i] * z + cross(qp[i] * w_i0, z) )
    Vector3f cross_term = (qdot[i] * w_i0).cross(z_im1_im1);
    Vector3f tmp_wp = wp_i0 + qddot[i] * z_im1_im1 + cross_term;
    wp_i.col(i) = R_i_im1 * tmp_wp;

    // r_i_im1_i[:, i] = T[i][:3,:3].T * (T[i][:3,3] - r_0_im1)
    Vector3f t_col = T[i].block<3, 1>(0, 3);
    r_i_im1_i.col(i) = T[i].block<3, 3>(0, 0).transpose() * (t_col - r_0_im1);

    // r_i_i_cmi: use robot.links[i].r_cm if set, otherwise fallback
    r_i_i_cmi.col(i) = r_cms[i]; // or GetCmPos() if not set

    // a_i[:, i] = R_i_im1 * a_i0 + cross(wp_i[:,i], r_i_im1_i[:,i]) +
    // cross(w_i[:,i], cross(w_i[:,i], r_i_im1_i[:,i]))
    Vector3f w_i_col_i = w_i.col(i);
    Vector3f r_i_im1_i_col_i = r_i_im1_i.col(i);
    Vector3f wp_i_col_i = wp_i.col(i);
    Vector3f ai = R_i_im1 * a_i0 + wp_i_col_i.cross(r_i_im1_i_col_i) +
                  w_i_col_i.cross(w_i_col_i.cross(r_i_im1_i_col_i));
    a_i.col(i) = ai;

    // a_ci
    Vector3f r_i_i_cmi_col_i = r_i_i_cmi.col(i);
    Vector3f aci = a_i.col(i) + wp_i_col_i.cross(r_i_i_cmi_col_i) +
                   w_i_col_i.cross(w_i_col_i.cross(r_i_i_cmi_col_i));
    a_ci.col(i) = aci;

    // update for next iteration
    w_i0 = w_i.col(i);
    wp_i0 = wp_i.col(i);
    a_i0 = a_i.col(i);
  }

  // Backward recursion
  MatrixXf f_i = MatrixXf::Zero(3, n + 1);
  MatrixXf tau_i = MatrixXf::Zero(3, n + 1);
  VectorXf _u = VectorXf::Zero(n);

  for (int idx = (int)n - 1; idx >= 0; --idx) {
    // gi = T[i][:3,:3].T @ g0
    Vector3f gi = T[idx].block<3, 3>(0, 0).transpose() * g0;
    Vector3f aci_i = a_ci.col(idx);
    Vector3f r_i_i_cmi_col_i = r_i_i_cmi.col(idx);
    Vector3f r_i_im1_i_col_i = r_i_im1_i.col(idx);
    Vector3f w_i_col_i = w_i.col(idx);
    if (idx == (int)n - 1) {
      f_i.col(idx) = m[idx] * (aci_i - gi);
      Vector3f wp_i_col_i = wp_i.col(idx);
      Vector3f f_i_col_i = f_i.col(idx);
      tau_i.col(idx) =
          (-f_i_col_i).cross(r_i_im1_i_col_i + r_i_i_cmi_col_i) +
          I[idx] * wp_i.col(idx) + w_i_col_i.cross(I[idx] * w_i_col_i);
    } else {
      Matrix3f R_i_ip1 =
          T[idx].block<3, 3>(0, 0).transpose() * T[idx + 1].block<3, 3>(0, 0);
      f_i.col(idx) = R_i_ip1 * f_i.col(idx + 1) + m[idx] * (aci_i - gi);

      Vector3f f_i_col_i = f_i.col(idx);
      Vector3f tau_init =
          (-f_i_col_i).cross(r_i_im1_i_col_i + r_i_i_cmi_col_i) +
          I[idx] * wp_i.col(idx) + w_i_col_i.cross(I[idx] * w_i_col_i);

      tau_i.col(idx) = R_i_ip1 * tau_i.col(idx + 1) +
                       (R_i_ip1 * f_i.col(idx + 1)).cross(r_i_i_cmi_col_i) +
                       tau_init;
    }

    if (idx == 0) {
      Vector3f z_im1_im1(0, 0, 1);
      _u[idx] = (tau_i.col(idx).transpose() *
                 T[idx].block<3, 3>(0, 0).transpose() * z_im1_im1)(0, 0);
    } else {
      Matrix3f R_im1_0 = T[idx - 1].block<3, 3>(0, 0);
      Vector3f z_im1_im1 = T[idx - 1].block<3, 3>(0, 0).transpose() *
                           T[idx - 1].block<3, 1>(0, 2);
      _u[idx] =
          (tau_i.col(idx).transpose() * T[idx].block<3, 3>(0, 0).transpose() *
           R_im1_0 * z_im1_im1)(0, 0);
    }
  }

  return _u;
}

// --- Build Euler-Lagrange matrices (M, C*qp, G) ---
std::tuple<MatrixXf, VectorXf, VectorXf>
getEulerLagrangeMatrices(const VectorXf &_q, const VectorXf &_qp,
                         const Vector3f &_g0, const Manipulator &robot) {
  size_t n = _q.size();
  // Gravity vector
  VectorXf G_out = recursiveNewtonEuler(_q, VectorXf::Zero(n),
                                        VectorXf::Zero(n), _g0, robot);
  // Coriolis
  VectorXf Cqp_out =
      recursiveNewtonEuler(_q, _qp, VectorXf::Zero(n), Vector3f::Zero(), robot);

  // Inertia matrix M: columns computed by Newton_Euler with qpp = unit basis
  MatrixXf M_out = MatrixXf::Zero(n, n);
  for (size_t i = 0; i < n; i++) {
    VectorXf e = VectorXf::Zero(n);
    e[i] = 1.0;
    M_out.col(i) =
        recursiveNewtonEuler(_q, VectorXf::Zero(n), e, Vector3f::Zero(), robot);
  }
  return std::make_tuple(M_out, Cqp_out, G_out);
}

// --- Example: build a robot and run ---
// int main() {
//     // Build robot with 6 DOF example (adjust as needed)
//     Robot robot;
//     const size_t n = 6;
//     robot.links.resize(n);
//     // assign masses/inertia from default arrays for testing:
//     auto masses = GetLinkMasses();
//     auto inertias = GetInertia();
//     for (size_t i=0;i<n;i++){
//         if (i < masses.size()) robot.links[i].m = masses[i];
//         else robot.links[i].m = 0.1;
//         if (i < inertias.size()) robot.links[i].I = inertias[i];
//         else robot.links[i].I = Matrix3f::Identity() * 1e-3;
//         robot.links[i].r_cm = GetCmPos(); // same for all in your example
//     }
//
//     VectorXf q(n), qp(n);
//     q << 0.1,0.1,0.1,0.1,0.1,0.1;
//     qp = q; // example
//     Vector3f g0(0.0, 0.0, -9.78);
//
//     MatrixXf M;
//     VectorXf Cqp, G;
//     GetEulerLagrangeMatrices(q, qp, g0, robot, M, Cqp, G);
//
//     std::cout << "Inertia Matrix M:\n" << M << "\n\n";
//     std::cout << "Skew test M - M^T (should be near zero):\n" << M -
//     M.transpose() << "\n\n"; std::cout << "C*qp vector:\n" << Cqp.transpose()
//     << "\n\n"; std::cout << "G vector:\n" << G.transpose() << "\n\n";
//
//     return 0;
// }
