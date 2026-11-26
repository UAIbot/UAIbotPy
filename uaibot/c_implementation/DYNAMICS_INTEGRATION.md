# Uaibot C++ dynamics integration

This integrates the supplied physics headers into the independent
`uaibot_cpp_bind` extension. The Python simulator classes and rendering code are
unchanged. The supplied files are the authority for the model equations;
`feat/torch` supplies the existing geometry/build dependencies. Only
`feat/torch` (`c398924411cebd901484c62ae56a1108c84fa897`) and `feat/fossen`
(`0a527627f733620d201b25a6411bcb4227c6ca6b`) were consulted.

## Layout

| File | Responsibility |
| --- | --- |
| `dynamics.hpp` | Shared rigid-body state, sensor noise, integration and allocation |
| `auv_physics.hpp`, `bluerov2_physics.hpp` | Supplied eight-thruster AUV model; `auv` aliases `bluerov2` |
| `uav_physics.hpp` | UAV model, rotor geometry and controllers |
| `manipulator_physics.hpp` | Generic serial-chain dynamics, inverse/forward dynamics and joint sensors |
| `manipulator_dh.hpp` | Standard-DH geometry and inertial-data conversion |
| `robot.hpp` | Optional C++ owning facades |
| `dynamics_bindings.cpp` | Registers common state and measurement types once |
| `auv_bindings.cpp`, `uav_bindings.cpp`, `manipulator_bindings.cpp` | Model-specific pybind registrations |
| `robot_bindings.cpp` | Bindings for the owning facades |
| `dynamics_bindings.hpp`, `dynamics_bindings_helpers.hpp` | Registration declarations and state-copy support |
| `pybind_main.cpp` | Existing extension API and ordered registration calls |

Geometry, collision detection and vector fields stay in the existing source
files. No unrelated implementation was moved. The empty `CPP_Manipulator`
constructor now initializes its fields through the zero-link constructor,
avoiding undefined reads when conversion is requested before configuration.

## Build

From the repository root, using the same interpreter that will import the module:

```sh
python -m pip install cmake pybind11 numpy
cmake -S uaibot/c_implementation -B build/dynamics \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DUAIBOT_BUILD_DYNAMICS_TESTS=ON
cmake --build build/dynamics --parallel 2
ctest --test-dir build/dynamics --output-on-failure
```

The existing `setup.py` can still build the extension: its module name and CMake
output-directory settings are preserved. No Python package refactor is required.
CMake now requires 3.18, selects C++17, uses target-local settings, and discovers
pybind11 through the selected Python when needed. It prefers an installed Eigen
3.3+ package and otherwise uses `third_party/Eigen`. An explicit fallback path
can be selected with `-DEIGEN3_INCLUDE_DIR=/path/containing/Eigen`.
Automatic pybind LTO is disabled to limit build memory. Platform RPATH settings
use `$ORIGIN` on Linux and `@loader_path` on macOS. Windows remains configured,
but this change was executed and tested on Linux only.

## Extension interface

All names below live directly in `uaibot_cpp_bind`. Stateful functions return a
new state; pass that complete result into the next call. Values are float64.

| Model | Types | Main functions |
| --- | --- | --- |
| Common | `RigidBodyState`, `RigidBodyMeasurement`, `MeasurementNoiseParameters` | `state.reset_measurement_noise()` |
| AUV | `AUVParameters`, `AUVState`, `AUVDiagnostics` | `auv_step`, `auv_step_open_loop`, `auv_step_stateless`, `auv_step_direct`, `auv_allocation_matrix`, `auv_total_mass_matrix`, `auv_inertial_acceleration` |
| UAV | `UAVParameters`, `UAVState`, `UAVDiagnostics`, `UAVRotor`, `UAVVelocityControl` | `uav_step`, `uav_step_open_loop`, `uav_step_wrench`, `uav_step_stateless`, `uav_step_direct`, `uav_configure_actuators`, `uav_rotor_allocation_matrix`, `uav_allocation_matrix`, `uav_total_mass_matrix`, `uav_inertial_acceleration` |
| Manipulator | `ManipulatorParameters`, `ManipulatorState`, `ManipulatorLink`, `ManipulatorJointType`, `ManipulatorDiagnostics`, `JointMeasurement`, `JointMeasurementNoiseParameters`, `ManipulatorInertiaReference` | `manipulator_make_state`, `manipulator_step`, `manipulator_step_open_loop`, `manipulator_step_effort`, `manipulator_mass_matrix`, `manipulator_bias_forces`, `manipulator_inverse_dynamics`, `manipulator_forward_dynamics`, `manipulator_forward_kinematics`, `manipulator_configure_actuators`, `manipulator_parameters_from_dh` |
| Owning facades | `AUVDynamics`, `UAVDynamics`, `ManipulatorDynamics` | `.step(target_velocity)`, `.step_open_loop(actuator_setpoints)`, `.parameters`, `.state` |

The existing `FossenState`, `FossenParameters`, `FossenDiagnostics`,
`FossenMeasurement` and `FossenMeasurementNoiseParameters` names remain aliases.
`fossen_step`, `fossen_step_stateless` and `fossen_step_direct` retain their
arguments, defaults and behavior. Shared measurement types are registered only
once, preventing pybind duplicate-type registration errors.

```python
import numpy as np
import uaibot_cpp_bind as cpp

p = cpp.UAVParameters()
s = cpp.UAVState()
s = cpp.uav_step(np.zeros(6), s, p)  # compensated velocity servo

# Direct physical actuator-output commands bypass the built-in controller.
outputs = np.array([0., 0., p.mass * p.gravity, 0., 0., 0.])
s = cpp.uav_step_open_loop(outputs, s, p)

vehicle = cpp.UAVDynamics(p)
snapshot = vehicle.step(np.zeros(6))
```

The owning facades return snapshots, so a previous result does not change after
later calls. `copy.copy` and `copy.deepcopy` on states or facades include actuator
memory, the random generator, cached normal distribution and sensor clock.
Noise memory is deliberately not writable from Python; use
`reset_measurement_noise()` after manually resetting simulation time. This
clears the cached measurement and restarts sampling at the next call.

STL containers use standard pybind value conversion. For example, assign
`p.links = [link0, link1]`; mutating the temporary list returned by `p.links`
does not update the C++ vector. To edit an element, fetch the list, change it,
then assign the list back. This also applies to external-wrench lists and AUV
thruster-geometry arrays. Nested bound class fields such as
`p.measurement_noise.enabled` can be edited directly.

Compute functions release the GIL. Independent robots can run concurrently;
do not mutate the same parameters/state/facade from another thread during a
call. Model validation raises `ValueError`; Python argument conversion may raise
`TypeError`. Numerical integration failures raise `RuntimeError`.

## DH and inertia conversion

The supplied generic dynamics model uses frames at joint origins, before the
fixed link transform. Uaibot uses standard-DH frames after that transform.
`manipulator_dh.hpp` converts between them without changing the physics model.

`manipulator_parameters_from_dh` accepts an `n x 5` table with columns
`[theta, d, alpha, a, joint_type]`, masses `(n,)`, COM coordinates `(n,3)` and a
list of `n` full `3 x 3` inertia matrices in DH axes. Joint type is 0 for revolute
and 1 for prismatic. As in the existing C++ FK, `q[i]` replaces the revolute
`theta` or prismatic `d`; it is not added to the table value.

The free converter defaults to COM-referenced inertia. Set
`inertia_reference=ManipulatorInertiaReference.DHOrigin` when the supplied
matrix is about the DH-frame origin. It converts
`I_COM = I_DH_origin + mass * skew(com_DH) * skew(com_DH)` before rotating
into joint-frame axes. Matrices must describe physically valid COM inertias.

The existing `CPP_Manipulator` additionally exposes:

```python
p = cpp_robot.dynamics_parameters(
    masses, com_dh, inertias_dh,
    gravity_base=np.array([0., 0., -9.81]),
    inertia_reference=cpp.ManipulatorInertiaReference.DHOrigin,
)
s = cpp.manipulator_make_state(p)
s.q = q
required_effort = cpp.manipulator_inverse_dynamics(q, qd, qdd, p)
s = cpp.manipulator_step_effort(required_effort, s, p)
```

The method reads geometry from that kinematic object and defaults to origin
inertia, matching the conversion present in the old Python `_dyn_model`.
Mass, COM and inertia data remain explicit inputs because the existing C++
kinematic object does not store them. Confirm the convention of each supplied
robot's actual inertial data before choosing the reference.

For fixed tail `F_i`, standard DH gives `T_i(q_i) = motion_z(q_i) * F_i`.
The converter sets the first parent transform to `base_to_dh0`, later parent
transforms to the preceding `F`, and transforms each link's COM/inertia by its
own `F_i`. The last DH/tool frame is not the last dynamics joint frame.
Continue using the existing FK for rendering and tool Jacobians. External
wrenches in the new model use its joint/link frame, not the old DH frame.

`gravity_base` is expressed in the fixed robot-base frame. For a world mounting
rotation `R_world_base`, pass `R_world_base.T @ gravity_world`. The converter
includes `htm_world_to_dh0`; the external robot-world transform and
`htm_dhn_to_ee` stay in the rendering/kinematics layer. Payload inertia must be
included explicitly in the appropriate physical link.

## Scope and behavior

The supplied allocation, actuator dynamics, noise processes and controllers
are retained. The AUV still has eight configurable thrusters and BlueROV2
Heavy defaults; `auv_physics.hpp` is an alias, not a new arbitrary-thruster
Fossen model. UAVs support arbitrary constant 6-by-N actuation. Manipulators
remain fixed-base serial chains.

The AUV noise-free step now rejects `dt` when adding it would overflow or fail
to advance simulation time, consistent with its noisy path and the other
models. Normal simulations are unaffected.

The Python simulator wrappers, automatic animation/state synchronization,
Python robot mass-data collection, contacts and joint-stop handling are outside
this change. No physical experiments or numerical-tuning claims are implied.

## Validation

The extension regression suite uses only stdlib `unittest` and NumPy. It checks
legacy names/calls, shared-type registration, owning snapshots, copied sensor
streams, measurement clocks, invalid settings, UAV freefall and inertial force
mapping, analytic actuator lag, underactuation, quadrotor hover, AUV hydrostatics,
analytic 2R/1P manipulator dynamics, external-wrench virtual work, and the DH
bridge against independently differentiated COM/orientation Jacobians.

An optional independent C++ trajectory comparison builds the same driver in
two separate executables, one per header tree. To enable it, supply a directory
containing the original `dynamics.hpp`, `bluerov2_physics.hpp`,
`uav_physics.hpp` and `manipulator_physics.hpp`:

```sh
cmake -S uaibot/c_implementation -B build/dynamics \
  -DUAIBOT_BUILD_DYNAMICS_TESTS=ON \
  -DUAIBOT_LEGACY_HEADERS_DIR=/absolute/path/to/original/headers
cmake --build build/dynamics --parallel 2
ctest --test-dir build/dynamics --output-on-failure
```

The original snapshots are not required by the normal extension test suite.
Build output still reports pre-existing missing-return warnings in some
geometry helpers in `manipulator.cpp`; those unrelated paths were not changed.

Executed validation: Linux x86_64, GCC 14.2, Python 3.12.14, pybind11 3.1.0,
CMake 4.4.5, and bundled Eigen. The Release extension built successfully;
all 19 extension tests passed. The independent trajectory comparison matched
360 state transitions across all three models, with sensor noise both enabled
and disabled, at `rtol=atol=1e-13`. Both CTest entries passed.
