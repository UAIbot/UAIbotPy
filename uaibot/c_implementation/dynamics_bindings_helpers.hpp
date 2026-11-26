#pragma once
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "dynamics_bindings.hpp"

namespace dynamics_bindings {
namespace py = pybind11;
// A state copy includes the private generator, cached distribution and clock.
template <class T, class... Options>
void copyable(py::class_<T, Options...>& cls) {
  cls.def("__copy__", [](const T& self) { return T(self); })
      .def(
          "__deepcopy__", [](const T& self, py::dict) { return T(self); },
          py::arg("memo"));
}
}  // namespace dynamics_bindings
