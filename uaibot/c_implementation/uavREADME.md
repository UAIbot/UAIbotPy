onfigurable UAV dynamics
uav_physics.hpp with dynamics.hpp provides C++17/Eigen headers for multirotor/VTOL
rigid-body simulation. It follows the supplied BlueROV2 implementation's
homogeneous-pose, inertial-twist and persistent-state approach. It has no ROS
or simulator dependency. The supplied BlueROV2 header is not needed at runtime.

The default parameters describe an illustrative 1.5 kg rigid body with ideal
six-axis actuation. examples/models.hpp selects an illustrative X quadrotor
by supplying rotor geometry, bounds and lag. Neither configuration is an
identified commercial aircraft model.
Integrating with your simulator
#include "uav_physics.hpp"

uav::Parameters p;
p.mass = 2.0;
p.inertia_at_cg_body = Eigen::Vector3d(.04, .05, .07).asDiagonal();
p.measurement_noise.enabled = true;

uav::State state;
state.pose(2, 3) = 2.0;
uav::Vec6 target = uav::six(1., 0., 0., 0., 0., 0.);

// On each simulation tick; retain the ENTIRE returned State.
state = uav::step(target, state, p);
// Render state.pose. Use state.measurement as estimated feedback.
// Truth remains in state.pose and state.twist.
p.dt is the duration of one API call. Call once per simulation tick. Time,
pose, velocity, actuator state, sensor memory and random-generator state are
owned by State. Copying a state makes a reproducible branch of the rollout.
Do not overwrite truth with state.measurement. The built-in velocity
controller already uses measured feedback when noise is enabled.
API	Command	Behavior
step(target_twist, state, p)	Six inertial geometric velocities	Built-in velocity control, bounded allocation, plant integration
step_wrench(wrench_world, state, p)	Six actuator forces/torques in world axes	Bounded allocation and plant integration; no automatic gravity/drag compensation
step_open_loop(outputs, state, p)	N physical actuator output setpoints	Clamp, actuator lag, plant integration; suitable for an existing flight controller
inertial_acceleration(state, outputs, p)	N instantaneous outputs	True inertial acceleration after saturation, without advancing time or lag
allocation_matrix(pose, p)	Pose	World-frame 6-by-N allocation matrix
total_mass_matrix(pose, p)	Pose	World-frame 6-by-6 rigid-body mass matrix at the pose origin


For an existing flight controller, step_open_loop is usually the clearest
plant interface. Feed state.measurement to that controller at its own update
rate. step_wrench rotates commands using the measured attitude when noise is
enabled, as the reference controller does. To model perfect body-fixed motor
commands independently of estimated attitude, use step_open_loop.

The stateless step(target, pose, twist, p) overload is retained for ideal
actuators with noise disabled. It rejects lag/noise because it cannot preserve
their memory. An overload accepting velocity_gains and dt also exists.
Frames, units and the constant actuation matrix
The convention matches the supplied header: a right-handed world with z up;
body x forward, y left, z up. Convert explicitly if your autopilot uses
NED/FRD. All physical values use SI units.
pose = [ R  position_world ]       R maps body coordinates into world axes
       [ 0        1        ]
twist = [ velocity_world_at_origin; angular_velocity_world ]
R_dot = skew(angular_velocity_world) * R
This is a geometric twist at the chosen vehicle origin; its linear component
is the position derivative. Rotating force and torque at that same point uses
T(R) = diag(R,R), without a translation/SE(3)-adjoint block.

Provide a constant body-frame matrix B, with rows ordered
[Fx, Fy, Fz, Tx, Ty, Tz], relating physical actuator outputs to wrench about
the pose origin:
wrench_body = B * outputs
B_world(R) = diag(R,R) * B
This supports full actuation, rank-deficient underactuation, and redundant
actuation with any supported channel count (1–256). A fully actuated model
needs rank six and sufficient actuator authority within its bounds; rank alone
does not remove one-sided thrust limits. No full-rank assumption is imposed
on the plant or allocator.
uav::AllocationMatrix B(6, number_of_actuators);
// Fill B using YOUR vehicle geometry or measured wrench mapping.
uav::configure_actuators(p, B, lower_limits, upper_limits, .03);
// Optional different lag per actuator:
p.actuator_time_constants[0] = .02;
The helper resizes the actuator bounds and lag vector together. You can also
set these parameter fields directly. Bounds are finite and must contain zero;
[0,0] disables a channel. Their units match the corresponding output channel.
For rotor-thrust inputs, outputs and bounds are in N. For B = I6, the first
three channels are forces in N and the last three are torques in N m.

For a fixed-axis rotor at position r, unit thrust axis d and signed
reaction-moment/thrust ratio kappa (metres), its column is
B_i = [ d; r cross d + kappa*d ]
rotor_allocation_matrix(std::vector<uav::Rotor>) constructs these columns.
The sign of kappa is the applied body reaction torque along d; choose it
from your propeller/motor convention. Positions are measured from the pose
origin, even if that origin differs from the CG. A user-supplied matrix can
include fixed tilted rotors or other constant linear actuator mappings.

If your controller outputs rotor speed, convert to thrust before this API,
e.g. f_i = kT_i * omega_i^2 for a one-way rotor. A constant B maps thrusts,
not raw RPM, PWM or normalized throttle. For a fixed-pitch rotor, the steady
reaction ratio is kQ/kT when torque and thrust both scale with speed squared.
Underactuated multirotor velocity control
Generic VelocityControl::WrenchProjection computes the six-axis inertial
velocity correction, compensates the model when enabled, and allocates the
resulting wrench. This is useful for fully actuated vehicles and for studying
unattainable commands with any supplied B. It cannot independently track
velocity commands outside the available wrench directions.

A conventional quadrotor gets horizontal acceleration by tilting. Select
VelocityControl::ThrustVectoring for its built-in cascaded controller:
#include "examples/models.hpp"
auto p = example_quadrotor(.03);  // B is 6-by-4; four thrust inputs in N
p.measurement_noise.enabled = true;

uav::State state;
state.pose(2, 3) = 2.;
// Start already in hover; leaving outputs empty starts motors at zero thrust.
state.actuator_outputs = Eigen::VectorXd::Constant(4, p.mass * p.gravity / 4.);

// Translation and WORLD-z yaw rate. Roll/pitch rates must be zero.
auto target = uav::six(1., .3, 0., 0., 0., .1);
state = uav::step(target, state, p);
This mode turns the desired world force into a tilt-limited attitude,
commands nonnegative thrust along the current body z axis, and tracks the
desired tilt and world-z yaw rate with bounded torques. Tilt consumes roll and
pitch control authority. Nonzero roll/pitch-rate targets are rejected rather
than treated as independently trackable commands. The yaw input is a rate;
there is no absolute heading reference.

The controller requires the pose origin at the CG and all net actuator forces
parallel to positive body z. It permits disabled or rank-deficient actuator
sets; achievable control still depends on the remaining authority. The plant
and WrenchProjection mode support arbitrary constant B and nonzero CG
offsets. Use your own flight controller for other underactuation patterns,
tilted-rotor allocation strategies, takeoff/landing, or aggressive aerobatics.

velocity_gains.head<3>() sets translational acceleration feedback;
velocity_gains.tail<3>() damps angular-rate error. attitude_gains sets
attitude-error acceleration gains. max_tilt_angle limits requested tilt,
not the actual attitude under disturbances. model_compensation cancels
nominal gravity, drag and rigid-body coupling; external disturbances are
excluded. Tune gains against the actual vehicle's thrust, torque and lag.
Physical model
Internally, velocities are expressed in body axes. For origin-to-CG vector
r_g, S = skew(r_g) and CG inertia J, the rigid-body mass matrix is
M = [ m I    -m S       ]
    [ m S     J - m S S ]

M * nu_dot = B * outputs + gravity_wrench + aerodynamic_wrench
             + external_wrench_body - C(M,nu)*nu
With momentum h = M*nu, the coupling product is
[omega cross h_linear; v cross h_linear + omega cross h_angular].
This handles off-diagonal inertia and CG offset while preserving the
rigid-body energy identity. Inertia must be symmetric, positive definite and
obey the principal-moment triangle inequalities.

Gravity acts at the CG. There is no underwater added mass or buoyancy.
Aerodynamic drag is evaluated at aerodynamic_center_body, using
relative_air_velocity = v_body + omega_body cross r_a - R.transpose()*wind_world
drag_force = -(linear + quadratic*abs(relative_air_velocity))*relative_air_velocity
Angular drag uses the same componentwise law with body angular velocity.
The aerodynamic-center force also contributes r_a cross drag_force to the
torque at the pose origin. All coefficients scale by
air_density/aerodynamic_reference_density. Quadratic coefficients already
include the reference 0.5*rho*Cd*A factor; do not multiply it in twice.
The diagonal drag model dissipates energy relative to uniform still/moving
air, including the force's moment arm.

Allocation minimizes
|| W * (B_world * output_setpoint - requested_wrench_world) ||^2
  + allocation_regularization * ||output_setpoint||^2
subject to actuator_min <= output_setpoint <= actuator_max
It uses a bounded active-set solve with regularization, rather than solving
unconstrained outputs and clipping them. allocation_weights act in world
axes, so choose them to represent force/torque priorities. Small nonzero
regularization introduces a small tracking bias. Diagnose infeasibility,
saturation and iteration limits through state.diagnostics.

Each actuator has an optional first-order output/thrust lag:
output_dot = (setpoint-output)/time_constant. This is not a motor electrical
or rotor-speed model. Lag is evaluated analytically at RK4 stage times;
rigid-body state uses quaternion RK4 with normalized output attitude.
Internal steps respect max_internal_dt, sensor update times and one quarter
of the fastest positive actuator time constant. Very small time constants can
require many steps; zero selects an ideal actuator. Reduce the maximum step
until trajectories converge for the speeds, drag and gains you simulate.
Measurement noise retained from the reference
The measurement-noise parameters, default numbers, random generator,
sampling equations and attitude/twist coupling are retained. Only the
vehicle-specific explanatory comment was updated. The 12 channels are
[position(3), rotation_vector_error(3), linear_velocity(3), angular_velocity(3)].
- White entries are variances per sample, not standard deviations or
  continuous-time noise densities.
- Colored entries are stationary variances of independent first-order
  Gauss–Markov errors. For elapsed sample time h, each channel uses
  a = exp(-h/tau) and innovation variance variance*(1-a*a), evaluated with
  expm1 for small time intervals.
- Position/rotation-vector errors use world axes. Attitude error is applied
  by a rotation exponential on SO(3), not by perturbing matrix elements.
- Velocity errors are added in aligned sensor/body axes and rotated into
  world coordinates with the same measured attitude, retaining the
  attitude/velocity cross-correlations.
- Samples are held between updates and carry measurement.sample_time.
  sample_period = 0 samples at every internal step; a positive period gives
  the sensor its own clock. White variance is not rescaled when the period
  changes.
- The truth integrator uses the true state. Enabled noise affects feedback
  control and observations; it is not injected into physical state.
- The seed and complete random/sensor memory live in State. Changing noise
  settings restarts the noise epoch at the next call. Disabling noise returns
  exact measurements and clears its memory. When manually resetting time,
  also call state.measurement_noise_state.reset().
The inherited default numbers are illustrative aided-estimator errors, not
calibrated IMU/GNSS/barometer specifications. This is an estimated-pose/twist
sensor; it does not synthesize raw accelerometer specific force or GPS messages.
Reproducibility is within the same standard-library implementation;
std::normal_distribution need not generate identical sequences across
different C++ libraries.
Build and validation
Requires Eigen 3.3 or later and a C++17 compiler. To compile without CMake,
run from the package root, adjusting the Eigen include path:
g++ -std=c++17 -O2 -I /usr/include/eigen3 -I . examples/uav.cpp -o uav_example
g++ -std=c++17 -O2 -I /usr/include/eigen3 -I . tests/uav_physics_test.cpp -o uav_physics_test
./uav_physics_test
./uav_example > trajectory.csv
With CMake and an installed Eigen package:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
Validation performed with GCC and Eigen 3.4.0: 700 checks passed, covering
analytical freefall/hover, tilted thrust, angular acceleration, CG offset,
energy/momentum conservation, wind/drag, lag transients, actuator saturation,
allocation KKT conditions over random full/rank-deficient matrices, full-axis
and quadrotor velocity tracking, sensor clock/hold, SO(3), repeatability and
invalid inputs. The quadrotor test with a (1, .35, 0) m/s target and .15
rad/s yaw target reached approximately (0.99936, 0.35189, -0.00000047) m/s
and .15010 rad/s world yaw rate after 8 s with motor lag enabled.

The modular refactor also passes moving-trajectory comparisons against the
previous headers and exact shared-sensor comparisons. See the root
VALIDATION.md for results. The optional legacy regression driver requires
the earlier header files and is excluded from the default build.

These checks validate implementation and numerical behavior, not a particular
airframe. For realistic results, identify mass/inertia, rotor mapping and
reaction signs, thrust limits, actuator lag, aerodynamic drag and sensor
statistics. Check that available thrust supports the chosen vehicle's weight.
Fidelity boundaries
This is a rigid-body multirotor/VTOL model with empirical airframe drag and
force actuators. It does not model fixed-wing lift/stall, rotor inflow/flapping,
propeller speed-dependent thrust changes in flight, rotor gyroscopic or
spin-acceleration reaction torques, propwash interaction, ground effect,
battery/ESC voltage behavior, contact/collisions or structural flexibility.
Wind is uniform and constant per API call; your simulator can update wind and
external wrench between calls. Fixed tilted propulsion is supported by B;
changing rotor tilt needs an updated matrix and its own tilt-actuator model.
Keep contact resolution in the simulator that already owns the environment.
