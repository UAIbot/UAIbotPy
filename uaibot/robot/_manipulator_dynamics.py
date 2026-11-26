"""Manipulator dynamics and synchronization with scene state."""

from copy import copy

import numpy as np

from ._dynamics import dynamics_backend, finite_vector, frame_time, pose_array


class _ManipulatorDynamics:
    @property
    def has_dynamics(self):
        return self.cpp_robot is not None and self.cpp_robot.dynamics is not None

    def _plant(self):
        if not self.has_dynamics:
            dynamics_backend()
            raise RuntimeError(
                "Configure link inertial data with configure_dynamics() first."
            )
        return self.cpp_robot.dynamics

    @property
    def dynamics_parameters(self):
        """Live native model parameters. Link geometry is set by the DH adapter."""
        return self._plant().parameters

    @property
    def dynamics_state(self):
        """Snapshot of truth, actuator outputs, measurements and diagnostics."""
        return copy(self._plant().state)

    @property
    def qd(self):
        """Current joint velocities as a column, in rad/s or m/s."""
        return np.matrix(self._plant().state.qd).T

    @qd.setter
    def qd(self, value):
        self.set_state(qd=value)

    def configure_dynamics(
        self,
        masses,
        com_dh,
        inertias_dh,
        *,
        inertia_reference="com",
        gravity_world=(0.0, 0.0, -9.81),
    ):
        """Attach native dynamics from mass, COM and inertia in each DH frame.

        ``inertia_reference`` is ``'com'`` or ``'dh_origin'``. Gravity is given
        in world axes. Joint geometry and the initial state come from the robot.
        """
        return self._configure_dh_dynamics(
            masses,
            com_dh,
            inertias_dh,
            inertia_reference,
            gravity_world,
            dynamics_backend(),
        )

    def _configure_dh_dynamics(
        self, masses, com_dh, inertias_dh, inertia_reference, gravity_world, cpp
    ):
        com = np.array(com_dh, dtype=float, copy=True)
        if cpp is not None:
            references = {
                "com": cpp.ManipulatorInertiaReference.CenterOfMass,
                "dh_origin": cpp.ManipulatorInertiaReference.DHOrigin,
            }
            if inertia_reference not in references:
                raise ValueError("inertia_reference must be 'com' or 'dh_origin'")
            table = np.array(
                [
                    [link.theta, link.d, link.alpha, link.a, link.joint_type]
                    for link in self._links
                ]
            )
            gravity = np.asarray(self.htm)[:3, :3].T @ finite_vector(
                gravity_world,
                3,
                "gravity_world",
            )
            params = cpp.manipulator_parameters_from_dh(
                table,
                finite_vector(masses, len(self._links), "masses"),
                com,
                inertias_dh,
                self.htm_base_0,
                gravity,
                references[inertia_reference],
            )
            state = cpp.manipulator_make_state(params)
            state.q, state.time = np.asarray(self.q).ravel(), self._state_time
            self.cpp_robot.dynamics = cpp.ManipulatorDynamics(params, state)
        self._com_dh = com
        return self.cpp_robot.parameters if cpp is not None else None

    def set_state(self, *, q=None, qd=None, htm=None, time=None):
        """Set the current truth without appending a playback frame.

        Omitted fields retain their value. Editing truth or time starts a new
        measurement epoch while keeping current actuator outputs.
        """
        q, pose = self._resolve_pose(q, htm)
        if time is None:
            time = self.cpp_robot.state.time if self.has_dynamics else self._state_time
        else:
            time = frame_time(time)
        self._set_pose_and_state(q, pose, time, qd=qd)

    def _resolve_pose(self, q, htm, enforce_joint_limits=False):
        if not isinstance(enforce_joint_limits, bool):
            raise TypeError("enforce_joint_limits must be a boolean")
        q = finite_vector(self.q if q is None else q, len(self._links), "q")
        if enforce_joint_limits:
            limits = np.asarray(self._joint_limit)
            q = np.clip(q, limits[:, 0], limits[:, 1])
        pose = np.asarray(self.htm) if htm is None else pose_array(htm)
        return q, pose

    def _set_pose_and_state(self, q, pose, time, *, qd=None, reset=False):
        if self.has_dynamics or qd is not None:
            plant = self._plant()
            current = plant.state
            state = (
                dynamics_backend().manipulator_make_state(plant.parameters)
                if reset
                else copy(current)
            )
            if qd is not None:
                state.qd = finite_vector(qd, len(self._links), "qd")
            old_pose = np.asarray(self._htm)
            if not (
                time == current.time
                and np.array_equal(q, current.q)
                and np.array_equal(state.qd, current.qd)
                and np.array_equal(pose, old_pose)
            ):
                state.reset_measurement_noise()
            state.q, state.time = q, time
            old_rotation, rotation = old_pose[:3, :3], pose[:3, :3]
            if not np.array_equal(rotation, old_rotation):
                plant.parameters.gravity_base = rotation.T @ (
                    old_rotation @ plant.parameters.gravity_base
                )
            plant.state = state
        self._q = np.matrix(q).T
        self._htm = np.matrix(pose)
        self._state_time = time

    def _joint_input(self, value, name):
        return finite_vector(value, len(self._links), name)

    def _step_dynamics(self, function, command, record):
        plant = self._plant()
        state = function(command, plant.state, plant.parameters)
        plant.state = state
        self._q = np.matrix(state.q).T
        self._state_time = state.time
        if record:
            self._record_ani_frame(state.time)
        return state

    def step(self, target_joint_velocity, *, record=True):
        """Advance the native joint velocity servo by dynamics_parameters.dt."""
        return self._step_dynamics(
            dynamics_backend().manipulator_step,
            self._joint_input(target_joint_velocity, "target_joint_velocity"),
            record,
        )

    def step_open_loop(self, actuator_setpoints, *, record=True):
        """Advance with physical actuator-output setpoints, including lag."""
        return self._step_dynamics(
            dynamics_backend().manipulator_step_open_loop,
            finite_vector(actuator_setpoints, None, "actuator_setpoints"),
            record,
        )

    def step_effort(self, joint_effort, *, record=True):
        """Allocate a requested joint effort to the available actuators."""
        return self._step_dynamics(
            dynamics_backend().manipulator_step_effort,
            self._joint_input(joint_effort, "joint_effort"),
            record,
        )

    def mass_matrix(self, q=None):
        return dynamics_backend().manipulator_mass_matrix(
            self._joint_input(self.q if q is None else q, "q"),
            self.dynamics_parameters,
        )

    def bias_forces(self, q=None, qd=None):
        return dynamics_backend().manipulator_bias_forces(
            self._joint_input(self.q if q is None else q, "q"),
            self._joint_input(self.qd if qd is None else qd, "qd"),
            self.dynamics_parameters,
        )

    def inverse_dynamics(self, q, qd, qdd):
        """Compute required joint effort, including specified external loads."""
        return dynamics_backend().manipulator_inverse_dynamics(
            self._joint_input(q, "q"),
            self._joint_input(qd, "qd"),
            self._joint_input(qdd, "qdd"),
            self.dynamics_parameters,
        )

    def forward_dynamics(self, actuator_outputs):
        """Compute instantaneous joint acceleration for physical actuator outputs."""
        plant = self._plant()
        return dynamics_backend().manipulator_forward_dynamics(
            plant.state,
            finite_vector(actuator_outputs, None, "actuator_outputs"),
            plant.parameters,
        )

    def configure_actuators(self, matrix, lower, upper, time_constant=0.0):
        """Configure a constant joint actuation map, bounds and lag together."""
        cpp, plant = dynamics_backend(), self._plant()
        params = copy(plant.parameters)
        cpp.manipulator_configure_actuators(params, matrix, lower, upper, time_constant)
        initial = cpp.manipulator_make_state(params)
        state = copy(plant.state)
        state.actuator_outputs, state.diagnostics = (
            initial.actuator_outputs,
            initial.diagnostics,
        )
        plant.parameters, plant.state = params, state
