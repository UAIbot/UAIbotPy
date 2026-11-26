#pragma once
#include <pybind11/pybind11.h>

void bind_dynamics(pybind11::module_& m);
void bind_auv_dynamics(pybind11::module_& m);
void bind_uav_dynamics(pybind11::module_& m);
void bind_manipulator_dynamics(pybind11::module_& m);
void bind_dynamics_robots(pybind11::module_& m);
