Modular robot dynamics: AUV, UAV and serial manipulator
A C++17/Eigen library of rigid-body dynamics models, with a shared numerical
and measurement layer. All code is header-only. No ROS or simulator dependency.
Header	Responsibility
dynamics.hpp	Common vehicle pose/twist state, measurement types, frame math, quaternion/Euclidean RK4, white and colored noise, bounded actuator allocation
auv_physics.hpp / bluerov2_physics.hpp	Configurable eight-thruster AUV with the supplied BlueROV2 Heavy defaults; rigid and added inertia, Coriolis, buoyancy, damping, current and thruster lag
uav_physics.hpp	Configurable multirotor/VTOL rigid body, wind/drag, arbitrary constant body actuation matrix, limits/lag and thrust-vector control
manipulator_physics.hpp	Fixed-base open-chain revolute/prismatic robot, recursive Newton–Euler inverse dynamics and full joint-space forward dynamics
robot.hpp	Optional owning Robot<Model> facade, with dynamics::AUV, dynamics::UAV and dynamics::Manipulator aliases


The AUV/UAV state and measurement definitions are now shared. A joint vector
is a different configuration space from a vehicle pose, so the manipulator
has a joint-state/encoder-measurement type. It uses the same RK4, noise-channel
updates and constrained allocator. This keeps coordinate conventions explicit
without converting joint angles into an artificial vehicle pose.
Quick start
Keep the headers together, add this directory and Eigen to your include path,
then use the pure model functions or the optional owning classes:
#include "robot.hpp"
#include "examples/models.hpp"
#include "examples/manipulator_models.hpp"

dynamics::AUV underwater{bluerov2::Parameters{}};
dynamics::UAV aircraft{example_quadrotor()};
dynamics::Manipulator arm{example_planar_arm()};

underwater.step(dynamics::six(.2, 0., 0., 0., 0., 0.));
aircraft.step(dynamics::six(1., 0., 0., 0., 0., .1));
arm.step(Eigen::Vector2d(.1, -.05));
// Returned truth and measurements are in each object's .state.
// Parameters are in .parameters. Own one instance per simulated robot.
Each call advances by that model's parameters.dt. Vehicle velocity commands
are inertial geometric twists [position_dot; angular_velocity_world].
Joint velocity commands are in joint order, rad/s for revolute and m/s for
prismatic joints. Quadrotor mode accepts translation and world-z yaw rate;
roll/pitch-rate targets must be zero because those axes control tilt.

step_open_loop(outputs) bypasses the velocity controller and sends physical
actuator output setpoints to the plant. This is the preferred integration
point for a simulator with an existing controller. All models enforce
actuator bounds and preserve configured actuator lag in their state.

Pure functions remain available, for example:
auto p = example_planar_arm();
auto state = manipulator::make_state(p);
state.q << .2, -.3;
Eigen::Vector2d motor_efforts(0., 0.);
state = manipulator::step_open_loop(motor_efforts, state, p);
auto M = manipulator::mass_matrix(state.q, p);
auto h = manipulator::bias_forces(state.q, state.qd, p);
auto required = manipulator::inverse_dynamics(state.q, state.qd,
                                            Eigen::Vector2d::Zero(), p);
Retain the entire returned State between calls. It holds actuator outputs,
simulation time, sensor sample, colored-error state and random stream. Copying
a state copies the random generator by value and gives a reproducible branch.
Render/use truth from pose/twist or q/qd; feed the separate measurement to
your controller. Enabled measurement noise never overwrites truth.
Manipulator model and geometry
One movable 1-DoF joint is associated with each Link, in serial order.
links[0] connects to a fixed base. Every later link connects to the preceding
link. Set these properties for every link:
- joint_type: Revolute or Prismatic.
- parent_to_joint: homogeneous transform mapping the zero-position joint
  frame into its parent link frame. Its translation is expressed in parent axes.
- axis_joint: unit motion axis in the zero-position joint frame.
- mass, center_of_mass_body, inertia_at_com_body: mass in kg, COM in the
  moving link frame, and full symmetric inertia at the COM in link axes.
- Optional viscous_friction, coulomb_friction, friction_speed and
  reflected_motor_inertia.
Using column vectors, kinematics are
T_base_link_i = T_base_parent * parent_to_joint_i * joint_motion_i(q_i)
For a revolute joint, joint_motion rotates about axis_joint; for a
prismatic joint it translates along that axis. There is no hidden DH-table
convention, joint offset or sign conversion. Encode a fixed rotation/offset in
the transform or adjust the reported coordinate explicitly. Fixed rigid
attachments should be merged into the neighboring link's mass, COM and
inertia using the parallel-axis theorem. Do not approximate a fixed joint by
an enormous servo gain.

forward_kinematics(q,p) returns each link-frame origin pose in the base.
The last origin is at the last joint, not automatically at the tool tip. For
a tool transform T_last_tool, use poses.back() * T_last_tool. Include tool
and payload inertia in the final link when they are rigidly attached.

gravity_base is the gravitational acceleration in fixed-base axes. Rotate
world gravity into those axes if the mounting orientation changes between
simulations. A moving/floating base requires additional coupled dynamics;
these are independent robot models rather than a coupled underwater arm.
Manipulator equations and loads
The plant solves
M(q) qdd + c(q,qd) + g(q) + friction(qd) = B u + tau_external
Recursive Newton–Euler evaluates inertial forces at each COM, full rotational
inertia and velocity products, then propagates forces and moments back to the
base. It includes revolute/prismatic coupling, including prismatic Coriolis
terms. mass_matrix builds inertia columns by zero-gravity inverse-dynamics
calls; forward dynamics uses a positive-definite solve, not an explicit inverse.
This implementation builds M in O(n²) and factors it with a dense solver;
it favors an inspectable implementation over an O(n) articulated-body solver
for very long chains. Supported chain size is 1–128 links.

inverse_dynamics(q,qd,qdd,p) returns the required applied joint effort,
including gravity and friction and subtracting supplied external loads.
bias_forces(q,qd,p) returns the same expression with qdd=0. Thus
M*qdd + bias_forces is the required actuator joint effort. Reflected motor
inertia contributes a nonnegative diagonal term to M. It is a joint-side
approximation; coupled rotor/gear inertias are not synthesized.

Optional external_wrenches_body[i] is an applied [force; torque] about
link i's frame origin, expressed in that link's moving body axes. A force
F applied at link-frame point r contributes [F; r cross F]. A wrench fixed
in world axes must be rotated into link axes as the robot moves. These fields
are held constant in body components during one API call. Decrease dt or
extend the load callback path if your contact solver needs finer updates.

Friction is viscous*qd + coulomb*tanh(qd/friction_speed) and is dissipative.
It approximates moving Coulomb friction smoothly; it does not implement
stiction. Revolute efforts are N m and prismatic efforts are N. Inertias are
kg m²; reflected inertia is kg m² for revolute or kg for prismatic joints.
Joint positions are not automatically wrapped or clamped.
Actuation, underactuation and controllers
The manipulator accepts any constant n x m matrix B mapping actuator
outputs to joint effort. Empty B selects identity; empty bounds select
[-100,100] in channel output units, and empty lag selects ideal actuators.
Select explicit realistic limits before using a physical robot model:
Eigen::MatrixXd B(2,1); B << 1., 0.; // only first joint is driven
manipulator::configure_actuators(p, B, Eigen::VectorXd::Constant(1,-5.),
                               Eigen::VectorXd::Constant(1,5.), .02);
step_effort(joint_effort,state,p) runs bounded allocation before integration.
step(target_joint_velocity,state,p) uses measured joint state to compute
inertia-weighted velocity feedback, with optional nominal bias compensation.
Supplied external loads are excluded from the controller's compensation.
Unattainable effort appears in the allocation residual; underactuated joint
motion still follows full coupling. A general underactuated system needs a
task-specific controller to achieve a particular trajectory. The built-in
servo does not guarantee arbitrary trajectory tracking under rank deficiency.

The shared allocator minimizes weighted wrench/effort error plus a small
positive output regularizer, subject to finite channel bounds that contain
zero. [0,0] disables an actuator. It handles redundant/rank-deficient B and
reports convergence, saturation and residuals. A constant linear B is suitable
for fixed transmissions; configuration-dependent tendons or variable gearing
need an extended model. The UAV has a constant 6-by-N body mapping; the AUV
retains the provided configurable eight-thruster geometry.

Actuator lag is first-order in physical output (force/effort), not RPM,
electrical current or PWM. Stage outputs are evaluated analytically. UAV and
manipulator substeps are limited to a quarter of the fastest positive lag time
constant in addition to max_internal_dt. For the preserved AUV integrator,
choose max_internal_dt small relative to thruster lag. Reduce the step until
the quantities of interest converge; RK4 is not an energy-preserving or stiff
contact integrator.
Measurement noise
Vehicle noise retains the original 12-channel white-plus-colored process,
including the measured-attitude/twist correlation. dynamics.hpp contains the
single implementation, and both vehicle namespaces expose compatible aliases.

Manipulator measurement noise uses 2n channels, [q;qd], with the same shared
white and exact first-order Gauss–Markov update primitives. Empty joint-noise
variance vectors resolve to zero and empty correlation times to one second;
specify realistic encoder/velocity-estimator values:
p.measurement_noise.enabled = true;
p.measurement_noise.white_variances = Eigen::VectorXd::Constant(2*p.links.size(),1.e-6);
p.measurement_noise.colored_variances = Eigen::VectorXd::Constant(2*p.links.size(),4.e-6);
p.measurement_noise.correlation_times = Eigen::VectorXd::Constant(2*p.links.size(),2.);
p.measurement_noise.sample_period = .01;
Entries are variances per sample, not standard deviations or spectral
densities. Position units depend on the joint type. The colored variances are
stationary variances and times are in seconds. Sensors have a separate clock,
sample-and-hold output, timestamps, deterministic seed and optional stationary
initialization. Sample period zero means every internal integration step.
Noise affects observations and closed-loop feedback, not open-loop truth.
Changing noise settings restarts the sensor epoch. Disabling noise yields exact
measurements and clears memory. Reset noise memory when manually resetting
simulation time. Standard-library RNG distributions can differ across platforms.
Build and tests
With Eigen installed and CMake:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
Or run the included dependency-free Python build driver with your Eigen path
(Python orchestrates the C++ compiler; it is not a runtime dependency):
python3 build_and_test.py --eigen /usr/include/eigen3
It builds and runs all three test suites and all three examples. Validation
includes analytical AUV hydrostatics and restoring moment, the previous UAV
physics/controller suite, analytic 1R/1P/2R manipulator equations, mixed
revolute/prismatic geometry with rotated inertias, independent Jacobian and
energy identities, external-load virtual work, actuator lag/allocation, and
measurement clocks/reproducibility. See VALIDATION.md for actual results.

The optional tests/legacy_regression.cpp compares moving AUV/UAV trajectories
against the earlier unmodified headers. It is excluded from the default build
because it needs those original files. Compile it with LEGACY_HEADER set to
the corresponding old header's absolute path and TEST_AUV for the AUV case.
For example, from the package root:
g++ -std=c++17 -O2 -I /usr/include/eigen3 -I . \
  -DLEGACY_HEADER='"/absolute/path/to/original/bluerov2_physics.hpp"' \
  -DTEST_AUV tests/legacy_regression.cpp -o legacy_auv_test
./legacy_auv_test
The optional tests/sensor_legacy_parity.cpp uses LEGACY_UAV_HEADER in the
same way to compare the shared sensor against the previous standalone UAV
sensor. These two optional drivers do not require any old files at runtime.
Compatibility and model fidelity
The uav:: and bluerov2:: parameter fields and function APIs remain available.
The alias auv:: also names the latter model. Keep dynamics.hpp beside each
model header; they are no longer individually standalone files. State layout
now includes a shared base, so this is source compatibility for normal field
access, not a promise of binary ABI or aggregate-initialization compatibility.

These models retain complete rigid-body coupling within their stated scope.
Vehicle parameters, drag, actuator response and sensor defaults require
identification for a specific robot. Manipulator parameters must include
payloads and accurate full link inertias. The AUV model uses fully submerged
empirical Fossen coefficients and uniform current; the UAV uses empirical
airframe drag and fixed-axis output actuators. Neither resolves flow fields.
The manipulator assumes rigid links and a fixed base. Contacts, hard joint
stops, self-collision, flexibility, backlash, gear elasticity and parallel-loop
constraints are not simulated here. Keep environment/contact handling in your
simulator or add an appropriate constrained-dynamics solver. See docs/UAV.md
for the UAV-specific modeling details and REFERENCES.md for equation sources.
