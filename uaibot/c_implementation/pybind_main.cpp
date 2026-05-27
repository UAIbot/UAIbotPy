// Include pybind11 FIRST to avoid conflicts
#include <Python.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Then include other headers
#include <math.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <sstream>
#include <typeinfo>
#include <vector>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/Dense>

#include "declarations.h"
#include "smooth_functions.hpp"

using namespace std;
using namespace std::chrono;
using namespace Eigen;
namespace py = pybind11;

PYBIND11_MODULE(uaibot_cpp_bind, m) {
  m.doc() = "UAIBot C++ interface";

  py::class_<FKResult>(m, "CPP_FKResult")
      .def(py::init<int>())
      .def(py::init<>())
      .def_readonly("htm_dh", &FKResult::htm_dh)
      .def_readonly("jac_v_dh", &FKResult::jac_v_dh)
      .def_readonly("jac_w_dh", &FKResult::jac_w_dh)
      .def_readonly("htm_ee", &FKResult::htm_ee)
      .def_readonly("jac_v_ee", &FKResult::jac_v_ee)
      .def_readonly("jac_w_ee", &FKResult::jac_w_ee)
      .def("get_x_dh", &FKResult::get_x_dh)
      .def("get_y_dh", &FKResult::get_y_dh)
      .def("get_z_dh", &FKResult::get_z_dh)
      .def("get_p_dh", &FKResult::get_p_dh)
      .def("get_Q_dh", &FKResult::get_Q_dh)
      .def("get_x_ee", &FKResult::get_x_ee)
      .def("get_y_ee", &FKResult::get_y_ee)
      .def("get_z_ee", &FKResult::get_z_ee)
      .def("get_p_ee", &FKResult::get_p_ee)
      .def("get_Q_ee", &FKResult::get_Q_ee)
      .def("__str__", &FKResult::toString)
      .def("__repr__", &FKResult::toString);

  py::class_<FKPrimResult>(m, "CPP_FKPrimResult")
      .def(py::init<>())
      .def_readonly("htm_prim", &FKPrimResult::htm_prim)
      .def_readonly("jac_v_prim", &FKPrimResult::jac_v_prim)
      .def_readonly("jac_w_prim", &FKPrimResult::jac_w_prim)
      .def("__str__", &FKPrimResult::toString)
      .def("__repr__", &FKPrimResult::toString);

  py::class_<TaskResult>(m, "CPP_TaskResult")
      .def(py::init<>())
      .def_readonly("task", &TaskResult::task)
      .def_readonly("jac_task", &TaskResult::jac_task)
      .def_readonly("max_error_pos", &TaskResult::max_error_pos)
      .def_readonly("max_error_ori", &TaskResult::max_error_ori)
      .def("__str__", &TaskResult::toString)
      .def("__repr__", &TaskResult::toString);

  py::class_<IKResult>(m, "CPP_IKResult")
      .def(py::init<>())
      .def(py::init<>())
      .def_readonly("qf", &IKResult::qf)
      .def_readonly("error_pos", &IKResult::error_pos)
      .def_readonly("error_ori", &IKResult::error_ori)
      .def_readonly("success", &IKResult::success)
      .def("__str__", &IKResult::toString)
      .def("__repr__", &IKResult::toString);

  py::class_<PrimDistResult>(m, "CPP_PrimDistResult")
      .def_readonly("dist", &PrimDistResult::dist)
      .def_readonly("proj_A", &PrimDistResult::proj_A)
      .def_readonly("proj_B", &PrimDistResult::proj_B)
      .def_readonly("hist_error", &PrimDistResult::hist_error)
      .def("__str__", &PrimDistResult::toString)
      .def("__repr__", &PrimDistResult::toString);

  py::class_<CheckFreeConfigResult>(m, "CPP_CheckFreeConfigResult")
      .def_readonly("isfree", &CheckFreeConfigResult::isfree)
      .def_readonly("message", &CheckFreeConfigResult::message)
      .def_readonly("info", &CheckFreeConfigResult::info);

  py::class_<VectorFieldResult>(m, "CPP_VectorFieldResult")
      .def_readonly("dist", &VectorFieldResult::dist)
      .def_readonly("twist", &VectorFieldResult::twist)
      .def_readonly("index", &VectorFieldResult::index);

  py::class_<DistStructLinkObj>(m, "CPP_DistStructLinkObj")
      .def(py::init<>())
      .def_readwrite("is_null", &DistStructLinkObj::is_null)
      .def_readwrite("link_number", &DistStructLinkObj::link_number)
      .def_readwrite("link_col_obj_number",
                     &DistStructLinkObj::link_col_obj_number)
      .def_readwrite("distance", &DistStructLinkObj::distance)
      .def_readwrite("point_link", &DistStructLinkObj::point_link)
      .def_readwrite("point_object", &DistStructLinkObj::point_object)
      .def_readwrite("jac_distance", &DistStructLinkObj::jac_distance);

  py::class_<DistStructRobotObj>(m, "CPP_DistStructRobotObj")
      .def(py::init<>())
      .def_readwrite("is_null", &DistStructRobotObj::is_null)
      .def_readwrite("jac_dist_mat", &DistStructRobotObj::jac_dist_mat)
      .def_readwrite("dist_vect", &DistStructRobotObj::dist_vect)
      .def_readwrite("list_info", &DistStructRobotObj::list_info);

  py::class_<DistStructLinkLink>(m, "CPP_DistStructLinkLink")
      .def(py::init<>())
      .def_readwrite("is_null", &DistStructLinkLink::is_null)
      .def_readwrite("link_number_1", &DistStructLinkLink::link_number_1)
      .def_readwrite("link_number_2", &DistStructLinkLink::link_number_2)
      .def_readwrite("link_col_obj_number_1",
                     &DistStructLinkLink::link_col_obj_number_1)
      .def_readwrite("link_col_obj_number_2",
                     &DistStructLinkLink::link_col_obj_number_2)
      .def_readwrite("distance", &DistStructLinkLink::distance)
      .def_readwrite("point_link_1", &DistStructLinkLink::point_link_1)
      .def_readwrite("point_link_2", &DistStructLinkLink::point_link_2)
      .def_readwrite("jac_distance", &DistStructLinkLink::jac_distance);

  py::class_<DistStructRobotAuto>(m, "CPP_DistStructRobotAuto")
      .def(py::init<>())
      .def_readwrite("is_null", &DistStructRobotAuto::is_null)
      .def_readwrite("jac_dist_mat", &DistStructRobotAuto::jac_dist_mat)
      .def_readwrite("dist_vect", &DistStructRobotAuto::dist_vect)
      .def_readwrite("list_info", &DistStructRobotAuto::list_info);

  py::class_<ProjResult>(m, "CPP_ProjResult")
      .def(py::init<>())
      .def_readwrite("dist", &ProjResult::dist)
      .def_readwrite("proj", &ProjResult::proj);

  py::class_<AABB>(m, "CPP_AABB")
      .def(py::init<>())
      .def_readwrite("lx", &AABB::lx)
      .def_readwrite("ly", &AABB::ly)
      .def_readwrite("lz", &AABB::lz)
      .def_readwrite("p", &AABB::p);

  py::class_<GeometricPrimitives>(m, "CPP_GeometricPrimitives")
      .def(py::init<>())
      .def_readwrite("htm", &GeometricPrimitives::htm)
      .def_readwrite("points_gp", &GeometricPrimitives::points_gp)
      .def_static(
          "create_box",
          static_cast<GeometricPrimitives (*)(Matrix4f, float, float, float)>(
              &GeometricPrimitives::create_box),
          py::arg("htm"), py::arg("width"), py::arg("depth"), py::arg("height"))
      .def_static("create_cylinder",
                  static_cast<GeometricPrimitives (*)(Matrix4f, float, float)>(
                      &GeometricPrimitives::create_cylinder),
                  py::arg("htm"), py::arg("radius"), py::arg("height"))
      .def_static("create_sphere",
                  static_cast<GeometricPrimitives (*)(Matrix4f, float)>(
                      &GeometricPrimitives::create_sphere),
                  py::arg("htm"), py::arg("radius"))
      .def_static("create_pointcloud",
                  static_cast<GeometricPrimitives (*)(vector<Vector3f> &)>(
                      &GeometricPrimitives::create_pointcloud),
                  py::arg("points"))
      .def_static(
          "create_convexpolytope",
          static_cast<GeometricPrimitives (*)(Matrix4f, MatrixXf, VectorXf)>(
              &GeometricPrimitives::create_convexpolytope),
          py::arg("htm"), py::arg("A"), py::arg("b"))
      .def("to_pointcloud",
           static_cast<GeometricPrimitives (GeometricPrimitives::*)(float)
                           const>(&GeometricPrimitives::to_pointcloud),
           py::arg("disc"))
      .def("dist_to",
           static_cast<PrimDistResult (GeometricPrimitives::*)(
               GeometricPrimitives, float, float, float, int) const>(
               &GeometricPrimitives::dist_to),
           py::arg("prim"), py::arg("h"), py::arg("eps"),
           py::arg("no_iter_max"), py::arg("tol"))
      .def("dist_to",
           static_cast<PrimDistResult (GeometricPrimitives::*)(
               GeometricPrimitives, float, float, float, int, Vector3f) const>(
               &GeometricPrimitives::dist_to),
           py::arg("prim"), py::arg("h"), py::arg("eps"), py::arg("tol"),
           py::arg("no_iter_max"), py::arg("p_A0"))
      .def("projection",
           static_cast<ProjResult (GeometricPrimitives::*)(
               Vector3f, float, float) const>(&GeometricPrimitives::projection),
           py::arg("point"), py::arg("h"), py::arg("eps"))
      .def("get_aabb", static_cast<AABB (GeometricPrimitives::*)() const>(
                           &GeometricPrimitives::get_aabb))
      .def("__str__", &GeometricPrimitives::toString)
      .def("__repr__", &GeometricPrimitives::toString);

  py::class_<Manipulator>(m, "CPP_Manipulator")
      .def(py::init<int>())
      .def(py::init<>())
      .def("fk",
           static_cast<vector<FKResult> (Manipulator::*)(
               const vector<VectorXf> &, const vector<Matrix4f> &, bool) const>(
               &Manipulator::fk),
           py::arg("q"), py::arg("htm_world_base"), py::arg("compute_jac"))
      .def("fk",
           static_cast<FKResult (Manipulator::*)(VectorXf, Matrix4f, bool)
                           const>(&Manipulator::fk),
           py::arg("q"), py::arg("htm_world_base"), py::arg("compute_jac"))
      .def("fk_prim",
           static_cast<FKPrimResult (Manipulator::*)(VectorXf, Matrix4f) const>(
               &Manipulator::fk_prim),
           py::arg("q"), py::arg("htm_world_base"))
      .def("fk_task", &Manipulator::fk_task, py::arg("q"),
           py::arg("htm_world_base"), py::arg("tg_htm"))
      .def("ik",
           static_cast<IKResult (Manipulator::*)(
               Matrix4f, Matrix4f, VectorXf, float, float, int, bool) const>(
               &Manipulator::ik),
           py::arg("tg_htm"), py::arg("htm"), py::arg("q0"), py::arg("p_tol"),
           py::arg("a_tol"), py::arg("no_iter_max"),
           py::arg("ignore_orientation"))
      .def("set_joint_param",
           (void (Manipulator::*)(int, float, float, float, float, int, float,
                                  float))&Manipulator::set_joint_param,
           py::arg("ind_link"), py::arg("theta"), py::arg("d"),
           py::arg("alpha"), py::arg("a"), py::arg("joint_type"),
           py::arg("q_min"), py::arg("q_max"))
      .def("add_geo_prim",
           (void (Manipulator::*)(
               int, GeometricPrimitives))&Manipulator::add_geo_prim,
           py::arg("ind_link"), py::arg("prim"))
      .def("set_htm_extra",
           (void (Manipulator::*)(Matrix4f,
                                  Matrix4f))&Manipulator::set_htm_extra,
           py::arg("htm_world_to_dh0"), py::arg("htm_dhn_to_ee"))
      .def(
          "check_free_configuration",
          static_cast<CheckFreeConfigResult (Manipulator::*)(
              VectorXf, Matrix4f, vector<GeometricPrimitives>, bool, bool,
              float, float, int) const>(&Manipulator::check_free_configuration),
          py::arg("q"), py::arg("htm"), py::arg("obstacles"),
          py::arg("check_joint"), py::arg("check_auto"), py::arg("tol"),
          py::arg("dist_tol"), py::arg("no_iter_max"))
      .def("compute_dist",
           static_cast<DistStructRobotObj (Manipulator::*)(
               GeometricPrimitives, VectorXf, Matrix4f, DistStructRobotObj,
               float, int, float, float, float) const>(
               &Manipulator::compute_dist),
           py::arg("obj"), py::arg("q"), py::arg("htm"),
           py::arg("old_dist_struct"), py::arg("tol"), py::arg("no_iter_max"),
           py::arg("max_dist"), py::arg("h"), py::arg("eps"))
      .def("compute_dist_auto",
           static_cast<DistStructRobotAuto (Manipulator::*)(
               VectorXf, DistStructRobotAuto, float, int, float, float, float)
                           const>(&Manipulator::compute_dist_auto),
           py::arg("q"), py::arg("old_dist_struct"), py::arg("tol"),
           py::arg("no_iter_max"), py::arg("max_dist"), py::arg("h"),
           py::arg("eps"))

      .def("__str__", &Manipulator::toString)
      .def("__repr__", &Manipulator::toString)
      .def("signed_distance", &Manipulator::signedDistance, py::arg("obj"),
           py::arg("q"), py::arg("htm"), py::arg("max_dist"), py::arg("gamma"),
           py::arg("is_conservative") = true, py::arg("epsilon") = 1e-6f)
      .def("signed_distance_auto", &Manipulator::signedDistanceAuto,
           py::arg("q"), py::arg("max_dist"), py::arg("gamma"),
           py::arg("is_conservative") = true);

  m.def("vectorfield_rn", &vectorfield_rn, py::arg("q"), py::arg("q_path"),
        py::arg("alpha"), py::arg("const_velocity"), py::arg("is_closed"),
        py::arg("gamma"));
  m.def("dp_inv_solve", &dp_inv_solve, py::arg("A"), py::arg("b"),
        py::arg("eps"));
  m.def("vectorfield_SE3", &vectorfield_SE3, py::arg("state"), py::arg("curve"),
        py::arg("kt1"), py::arg("kt2"), py::arg("kt3"), py::arg("kn1"),
        py::arg("kn2"),
        py::arg("curve_derivative") = std::vector<Eigen::MatrixXd>(),
        py::arg("delta") = c_delta, py::arg("ds") = c_ds);
  m.def("distance_box2box", &distBox2Box, py::arg("box1"), py::arg("box2"),
        py::arg("gamma"), py::arg("is_conservative") = true,
        py::arg("skip_gradient") = false, py::arg("epsilon") = 1e-6f,
        py::arg("eps_edge") = 1e-6f);
  // m.def("distance_set2set", &distSet2Set, py::arg("vertices_A"),
  // py::arg("vertices_B"),
  //       py::arg("normals_A"), py::arg("normals_B"), py::arg("edge_normals"),
  //       py::arg("gamma"), py::arg("skip_gradient") = false);
  // Overloaded functions:
  m.def(
      "get_candidate_normals",
      [](const GeometricPrimitives &polyhedron1,
         const GeometricPrimitives &polyhedron2, bool is_conservative,
         float epsEdge) {
        return getCandidateNormals(polyhedron1, polyhedron2, is_conservative,
                                   epsEdge);
      },
      py::arg("polyhedron1"), py::arg("polyhedron2"),
      py::arg("is_conservative") = false, py::arg("eps_edge") = 1e-6f);
  m.def(
      "smooth_min",
      [](const std::vector<float> &values, float r) {
        return smoothMinListWithGradient(values, r);
      },
      py::arg("values"), py::arg("r"));
  m.def(
      "smooth_max",
      [](const std::vector<float> &values, float r) {
        return smoothMaxListWithGradient(values, r);
      },
      py::arg("values"), py::arg("r"));
  m.def(
      "distance_set2set",
      [](const GeometricPrimitives &polyhedron1,
         const GeometricPrimitives &polyhedron2, float gamma,
         bool is_conservative, bool skip_gradient, float epsilon) {
        return distSet2Set(polyhedron1, polyhedron2, gamma, is_conservative,
                           skip_gradient, epsilon);
      },
      py::arg("polyhedron1"), py::arg("polyhedron2"), py::arg("gamma"),
      py::arg("is_conservative") = true, py::arg("skip_gradient") = false,
      py::arg("epsilon") = 1e-6f);
  m.def(
      "holder_distance",
      [](std::vector<Eigen::Vector3f> verticesA,
         std::vector<Eigen::Vector3f> verticesB,
         std::vector<Eigen::Vector3f> normalsA,
         std::vector<Eigen::Vector3f> normalsB,
         std::vector<Eigen::Vector3f> edgeNormals, float gamma,
         bool skipGradient, float epsilon) {
        return distSet2Set(verticesA, verticesB, normalsA, normalsB,
                           edgeNormals, gamma, skipGradient, epsilon);
      },
      py::arg("vertices1"), py::arg("vertices2"), py::arg("normals1"),
      py::arg("normals2"), py::arg("edge_normals"), py::arg("gamma"),
      py::arg("skip_gradient") = false, py::arg("epsilon") = 1e-6f);
}
