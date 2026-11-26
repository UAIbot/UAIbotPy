"""Free-floating robots backed by Uaibot's native rigid-body dynamics."""
from copy import copy

import numpy as np

from uaibot.simobjects.rigidobject import RigidObject
from ._dynamics import dynamics_backend, finite_vector, frame_time, pose_array
from .robot import Robot


class _RigidBodyRobot(Robot, RigidObject):
    def __init__(self, name='', list_model_3d=None, htm=None, *, twist=None, parameters=None):
        cpp = dynamics_backend()
        state = getattr(cpp, self._kind + 'State')()
        self._cpp_robot = None
        super().__init__(
            [] if list_model_3d is None else list_model_3d, name,
            state.pose if htm is None else pose_array(htm),
        )
        state.pose = np.asarray(self.htm)
        if twist is not None:
            state.twist = finite_vector(twist, 6, 'twist')
        if parameters is None:
            parameters = getattr(cpp, self._kind + 'Parameters')()
        self._cpp_robot = getattr(cpp, self._kind + 'Dynamics')(parameters, state)

    def _physics(self, function):
        return getattr(dynamics_backend(), f'{self._kind.lower()}_{function}')

    @Robot.htm.getter
    def htm(self):
        if self.cpp_robot is not None:
            return np.matrix(self.cpp_robot.state.pose)
        return np.matrix(self._htm)

    @property
    def dynamics_parameters(self):
        """Live native model parameters, including dt and sensor settings."""
        return self.cpp_robot.parameters

    @property
    def dynamics_state(self):
        """Snapshot of truth, actuator outputs, measurements and diagnostics."""
        return copy(self.cpp_robot.state)

    @property
    def twist(self):
        """Inertial geometric velocity [linear; angular] at the robot origin."""
        return np.array(self.cpp_robot.state.twist, copy=True)

    @twist.setter
    def twist(self, value):
        self.set_state(twist=value)

    def _changed_state(self, htm, twist, time):
        current = self.cpp_robot.state
        state = copy(current)
        velocity = current.twist if twist is None else finite_vector(twist, 6, 'twist')
        clock = current.time if time is None else frame_time(time)
        if (clock != current.time or not np.array_equal(htm, current.pose)
                or not np.array_equal(velocity, current.twist)):
            state.reset_measurement_noise()
        state.pose, state.twist, state.time = htm, velocity, clock
        return state

    def set_state(self, *, htm=None, twist=None, time=None):
        """Set truth without writing a playback frame, retaining actuator outputs."""
        pose = self.cpp_robot.state.pose if htm is None else pose_array(htm)
        self.cpp_robot.state = self._changed_state(pose, twist, time)
        self._htm = np.matrix(pose)

    def _record_pose(self, time, pose):
        self._frames.append([time, *np.asarray(pose).ravel().tolist()])
        self._max_time = max(self._max_time, time)

    def add_ani_frame(self, time, htm=None):
        """Set the native pose and clock and append the same pose for playback."""
        time = frame_time(time)
        pose = np.asarray(self.htm) if htm is None else pose_array(htm)
        self.set_state(htm=pose, time=time)
        self._record_pose(time, pose)

    def set_ani_frame(self, htm=None):
        """Restart playback and reset the native clock, twist and actuator state."""
        pose = np.asarray(self.htm) if htm is None else pose_array(htm)
        if self.cpp_robot is not None:
            state = type(self.cpp_robot.state)()
            state.pose = pose
            self.cpp_robot.state = state
        self._htm = np.matrix(pose)
        self._frames = []
        self._max_time = 0.
        self._record_pose(0., pose)

    def _step(self, mode, command, size, record):
        command = finite_vector(command, size, 'command')
        native = self.cpp_robot
        state = self._physics(mode)(command, native.state, native.parameters)
        native.state = state
        self._htm = np.matrix(state.pose)
        if record:
            self._record_pose(state.time, state.pose)
        return state

    def step(self, target_twist, *, record=True):
        """Advance the native velocity servo for one interval in inertial axes."""
        return self._step('step', target_twist, 6, record)

    def step_open_loop(self, actuator_setpoints, *, record=True):
        """Advance with physical actuator-output setpoints, including lag."""
        return self._step('step_open_loop', actuator_setpoints, None, record)

    def allocation_matrix(self):
        """The actuator-to-wrench map, expressed in world axes at the origin."""
        return self._physics('allocation_matrix')(self.cpp_robot.state.pose, self.dynamics_parameters)

    def total_mass_matrix(self):
        """The native mass matrix, expressed in world axes at the origin."""
        return self._physics('total_mass_matrix')(self.cpp_robot.state.pose, self.dynamics_parameters)

    def inertial_acceleration(self, actuator_outputs):
        """Instantaneous geometric acceleration in world axes."""
        return self._physics('inertial_acceleration')(
            self.cpp_robot.state, finite_vector(actuator_outputs, None, 'actuator_outputs'),
            self.dynamics_parameters,
        )


class UAV(_RigidBodyRobot):
    """Configurable aerial robot. Pass model meshes and optional UAVParameters."""

    _kind = 'UAV'

    def step_wrench(self, wrench_world, *, record=True):
        """Allocate an inertial wrench at the vehicle origin and advance."""
        return self._step('step_wrench', wrench_world, 6, record)

    def configure_actuators(self, matrix_body, lower, upper, time_constant=0.):
        """Set the body-frame actuation map and reset the actuator outputs."""
        params = copy(self.dynamics_parameters)
        self._physics('configure_actuators')(params, matrix_body, lower, upper, time_constant)
        self._physics('allocation_matrix')(self.cpp_robot.state.pose, params)
        state = self.dynamics_state
        state.actuator_outputs = np.zeros(params.actuation_matrix_body.shape[1])
        state.diagnostics = dynamics_backend().UAVDiagnostics()
        self.cpp_robot.parameters, self.cpp_robot.state = params, state

    @staticmethod
    def rotor_allocation_matrix(rotors):
        """Construct a body-frame allocation matrix from native UAVRotor objects."""
        return dynamics_backend().uav_rotor_allocation_matrix(rotors)


class AUV(_RigidBodyRobot):
    """Eight-thruster Fossen model with the native BlueROV2 Heavy defaults."""

    _kind = 'AUV'
