import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
from utils import *
from graphics.meshmaterial import *
from simobjects.group import *

from ._set_ani_frame import _set_ani_frame
from ._add_ani_frame import _add_ani_frame

from ._ikm import _ikm
from ._fkm import _fkm
from ._jac_geo import _jac_geo
from ._jac_jac_geo import _jac_jac_geo
from ._jac_ana import _jac_ana

from ._dyn_model import _dyn_model

from ._vector_field import _vector_field
from ._task_function import _task_function
from ._coop_task_function import _coop_task_function
from ._const_control import _const_control
from ._const_control_mod import _const_control_mod

from ._gen_code import _gen_code
from ._update_col_object import _update_col_object
from ._add_col_object import _add_col_object
from ._attach_object import _attach_object
from ._detach_object import _detach_object

from ._compute_dist import _compute_dist
from ._compute_dist_auto import _compute_dist_auto
from ._check_free_configuration import _check_free_configuration

from ._create_kuka_kr5 import _create_kuka_kr5
from ._create_epson_t6 import _create_epson_t6
from ._create_staubli_tx60 import _create_staubli_tx60
from ._create_kuka_lbr_iiwa import _create_kuka_lbr_iiwa
from ._create_abb_crb import _create_abb_crb
from ._create_darwin_mini import _create_darwin_mini
from ._create_kuka_kr5_per import _create_kuka_kr5_per
from ._create_franka_ermika_3 import _create_franka_ermika_3
from robot._create_davinci import _create_davinci
from robot._create_jaco import _create_jaco

#teste
#from robot._create_davinci import _create_davinci

class Robot:
    """
  A class that contains a robot object in uaibot.

  Parameters
  ----------

  name : string
      The robot name.
      (default: 'genRobot').

  htm_base_0 : 4x4 numpy array or 4x4 nested list
      The robot base's configuration.
      (default: 4x4 identity matrix).

  list_base_3d_obj : list of 'uaibot.Model3D' objects
      The list of 3d models of the base of the robot.
      If set to None, there is no base 3d object.
      (default: None).

  links : A list of 'uaibot.Link' objects
      The list of link objects.

  q0 : nd numpy vector or array
      The robot initial configuration.
      (default: zero vector).

  eef_frame_visible : boolean
      Set if the end-effector frame is visible.
      (default: True).

  joint_limit : n x 2 numpy array or None
      A n x 2 numpy array containing the joint limits, either in rad or meters.
      If set to 'None', use very large joint limits.
      (default: None).
  """

    #######################################
    # Attributes
    #######################################

    @property
    def q(self):
        """The current joint configuration."""
        return np.matrix(self._q)

    @property
    def q0(self):
        """The default joint configuration."""
        return np.matrix(self._q0)

    @property
    def htm(self):
        """
        The current base configuration in scenario coordinates.
        A 4x4 homogeneous matrix written is scenario coordinates.
        """
        return np.matrix(self._htm)

    @property
    def htm_base_0(self):
        """
        The constant homogeneous transformation between the HTM of the base and
        the HTM of the first Denavit-Hartenberg frame.
        """
        return np.matrix(self._htm_base_0)

    @property
    def htm_n_eef(self):
        """
        The constant homogeneous transformation between the HTM of the last
        Denavit-Hartenberg frame and the end-effector
        """
        return np.matrix(self._htm_n_eef)

    @property
    def links(self):
        """Data structures containing the links of the robot."""
        return self._links

    @property
    def attached_objects(self):
        """Data structures containing the objects attached into the robot."""
        return self._attached_objects

    @property
    def name(self):
        """Name of the object."""
        return self._name

    @property
    def list_object_3d_base(self):
        """The list of 3d objects that form the base."""
        return self._list_object_3d_base

    @property
    def eef_frame_visible(self):
        """If the frame attached to the end effector is visible"""
        return self._eef_frame_visible

    @property
    def joint_limit(self):
        """A n x 2 numpy array containing the joint limits, either in rad or meters"""
        return self._joint_limit


    #######################################
    # Constructor
    #######################################

    def __init__(self, name, links, list_base_3d_obj=None, htm=np.identity(4), htm_base_0=np.identity(4),
                 htm_n_eef = np.identity(4), q0=None, eef_frame_visible=True, joint_limits = None):
        # Error handling

        if not (Utils.is_a_name(name)):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not Utils.is_a_matrix(htm_base_0, 4, 4):
            raise Exception("The parameter 'htm_base_0' should be a 4x4 homogeneous transformation matrix.")

        if not Utils.is_a_matrix(htm_n_eef, 4, 4):
            raise Exception("The parameter 'htm_n_eef' should be a 4x4 homogeneous transformation matrix.")

        if not (str(type(links)) == "<class 'list'>"):
            raise Exception("The parameter 'links' should be a list of 'uaibot.Link' objects.")
        else:
            for link in links:
                if not Utils.get_uaibot_type(link) == "uaibot.Link":
                    raise Exception("The parameter 'links' should be a list of 'uaibot.Link' objects.")

        n = len(links)

        if not (q0 is None):
            self._q0 = np.matrix(q0).reshape((n,1))
        else:
            self._q0 = np.matrix(np.zeros((n,1)))

        if not (str(type(list_base_3d_obj)) == "<class 'list'>" or (list_base_3d_obj is None)):
            raise Exception("The parameter 'list_base_3d_obj' should be a list of 'uaibot.Model3D' objects.")
        else:
            for i in range(len(list_base_3d_obj)):
                if not (Utils.get_uaibot_type(list_base_3d_obj[i]) == "uaibot.Model3D"):
                    raise Exception(
                        "The parameter 'list_base_3d_obj' should be a list of 'uaibot.Model3D' objects.")

        if not Utils.is_a_vector(self._q0, n):
            raise Exception("The parameter 'q0' should be a " + str(n) + " dimensional vector.")

        if not str(type(eef_frame_visible)) == "<class 'bool'>":
            raise Exception("The parameter 'eef_frame_visible' must be a boolean.")

        if not joint_limits is None:
            self._joint_limit = joint_limits
        else:
            self._joint_limit = np.block([-10 * np.ones((n, 1)) , 10 * np.ones((n, 1))])

        if not Utils.is_a_matrix(self._joint_limit, n, 2):
            raise Exception("The parameter 'joint_limits' should be a " + str(n) + " x 2 numpy array.")

        for i in range(n):
            if self._joint_limit[i, 0]>self._joint_limit[i, 0]:
                raise Exception(
                    "In the parameter 'joint_limits', the minimum value (first column) must be smaller or equal than the maximum value (second column)")

        # end error handling

        self._frames = []
        self._list_object_3d_base = list_base_3d_obj
        self._htm = np.matrix(htm)
        self._name = name
        self._attached_objects = []
        self._links = links
        self._htm_base_0 = htm_base_0
        self._htm_n_eef = htm_n_eef
        self._eef_frame_visible = eef_frame_visible
        self._max_time = 0

        # Set initial total configuration
        self.set_ani_frame(self._q0, self._htm)

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):
        n = len(self.links)

        string = "Robot with name '" + self.name + "': \n\n"
        string += " Number of joints: " + str(n) + "\n"
        string += " Joint types: "

        for i in range(n):
            string += "R" if self._links[i].joint_type == 0 else "P"

        string += "\n"
        string += " Current configuration: " + str([round(num, 3) for num in np.ndarray.tolist(self.q)]) + "\n"
        string += " Current base HTM: \n" + str(self.htm) + "\n"
        string += " Current end-effector HTM: \n" + str(self.fkm())
        return string

    #######################################
    # Methods for configuration changing
    #######################################

    def add_ani_frame(self, time, q=None, htm=None, enforce_joint_limits = False):
        """
    Add a single configuration to the object's animation queue.

    Parameters
    ----------
    time: positive float
        The timestamp of the animation frame, in seconds.
    q : nd numpy vector or array
        The manipulator's joint configuration.
    htm : 4x4 numpy array or 4x4 nested list
        The robot base's configuration.
        (default: the same as the current HTM).
    enforce_joint_limits: boolean
        If True and some q violates the joint limits (seen in the attribute 'joint_limit'), the
        respective q is clamped. Note that it DOES NOT issue any warning for this. So, be aware.
        (default: False).

    Returns
    -------
    None
    """
        return _add_ani_frame(self, time, q, htm, enforce_joint_limits)

    def set_ani_frame(self, q=None, htm=None, enforce_joint_limits = False):
        """
    Reset object's animation queue and add a single configuration to the 
    object's animation queue.

    Parameters
    ----------
    q : nd numpy vector or array
        The manipulator's joint configuration .
        (default: the current  joint configuration (robot.q) for the manipulator, q0).
    htm : 4x4 numpy array or 4x4 nested list
        The robot base's configuration.
        (default: the same as the current HTM).
    enforce_joint_limits: boolean
        If True and some q violates the joint limits (seen in the attribute 'joint_limit'), the
        respective q is clamped. Note that it DOES NOT issue any warning for this. So, be aware.
        (default: False).
    Returns
    -------
    None
    """
        return _set_ani_frame(self, q, htm, enforce_joint_limits)

    #######################################
    # Methods for kinematics model
    #######################################

    def fkm(self, q=None, axis='eef', htm=None):
        """
    Compute the forward kinematics for an axis at a given joint and base
    configuration. Everything is written in the scenario coordinates.

    Parameters
    ----------
    q : nd numpy vector or array
        The manipulator's joint configuration.
        (default: the current  joint configuration (robot.q) for the manipulator).
    axis : string
        For which axis you want to compute the FK:
        'eef': for the end-effector;
        'dh': for all Denavit-Hartenberg axis;
        'com': for all center-of-mass axis.
        (default: 'eef').    
    htm : 4x4 numpy array or 4x4 nested list
        The robot base's configuration.
        (default: the same as the current HTM).

    Returns
    -------
    htm_fk : 4x4 or nx4x4 numpy matrix
        For axis='eef', returns a single htm. For the other cases, return
        n htms as a nx4x4 numpy matrix.
    """
        return _fkm(self, q, axis, htm)

    def ikm(self, htm_target, q0=None, p_tol=0.005, a_tol=5, no_iter_max=2000, ignore_orientation=False):
        """
    Try to solve the inverse kinematic problem for the end-effector, given a
    desired homogeneous transformation matrix. It returns the manipulator
    configuration.

    Important: it disregards the current htm of the base of the robot. That is,
    it assumes that robot.htm = np.identity(4). You can easily consider other
    cases by transforming htm_target as Utils.inv_htm(robot.htm) * htm_target.

    Uses an iterative algorithm.

    The algorithm can fail, throwing an Exception when it happens.

    Parameters
    ----------
    htm_target : 4x4 numpy array or 4x4 nested list
        The target end-effector HTM, written in scenario coordinates.
    q0 : n-dimensional numpy vector or array
        Initial guess for the algorithm for the joint configuration.
        (default: a random joint configuration).
    p_tol : positive float
        The accepted error for the end-effector position, in meters.
        (default: 0.005 m).    
    a_tol : positive float
        The accepted error for the end-effector orientation, in degrees.
        (default: 5 degrees). 
    no_iter_max : positive int
        The maximum number of iterations for the algorithm.
        (default: 2000 iterations).
    ignore_orientation: boolean
        If True, the orientation part of the HTM is ignored. Task is position-only.
        (default: False)

    Returns
    -------
    q : n-dimensional numpy column vector
        The configuration that solves the IK problem.
    """
        return _ikm(self, htm_target, q0, p_tol, a_tol, no_iter_max, ignore_orientation)

    def jac_geo(self, q=None, axis='eef', htm=None):
        """
    Compute the geometric Jacobian for an axis at a given joint and base
    configuration. Also returns the forward kinematics as a by-product.
    Everything is written in the scenario coordinates.

    Parameters
    ----------
    q : nd numpy vector or array
        The manipulator's joint configuration 
        (default: the current  joint configuration (robot.q) for the manipulator).
    axis : string
        For which axis you want to compute the FK:
        'eef': for the end-effector;
        'dh': for all Denavit-Hartenberg axis;
        'com': for all center-of-mass axis.
        (default: 'eef').    
    htm : 4x4 numpy array or 4x4 nested list
        The robot base's configuration.
        (default: the same as the current htm).

    Returns
    -------
    jac_geo : 6 x n or n x 6 x n numpy matrix
        For axis='eef', returns a single 6xn Jacobian. For the other cases, 
        return n Jacobians as a nx6xn numpy matrix.

    htm_out : 4 x 4 or n x 4 x 4 numpy matrix
        For axis='eef', returns a single htm. For the other cases, return
        n htms as a n x 4 x 4 numpy matrix.
    """
        return _jac_geo(self, q, axis, htm)

    def jac_ana(self, q=None, htm=None):
        """
    Compute the analytic Jacobian for the end-effector. The Euler angle
    convention is zyx. Also returns the end-effector htm and Euler angles
    as a by-product.

    Parameters
    ----------
    q : nd numpy vector or array
        The manipulator's joint configuration
        (default: the current  joint configuration (robot.q) for the manipulator).
    htm : 4x4 numpy array or 4x4 nested list
        The robot base's configuration.
        (default: the same as the current htm).

    Returns
    -------
    jac_ana : 6 x n numpy matrix
        The analytic Jacobian.

    htm_eef : 4 x 4 numpy matrix
        The end-effector htm.

    phi : 3x1 numpy matrix
        The euler angles in the z (alpha), y (beta) and x (gamma) convention,
        as a column vector.
    """
        return _jac_ana(self, q, htm)

    def jac_jac_geo(self, q=None, axis='eef', htm=None):
        """
    Compute the Jacobians of the columns of the geometric Jacobian in the joint variable 'q'.
    This can be either to the end-effector frames (axis='eef'), to the Denavit-Hartenberg (DH) frames
    (axis='dh') or to the center-of-mass (COM) frames (axis='com').

    If axis ='dh' or 'com':

    If the robot has n links, jj_geo = robot.jac_jac_geo(q, htm) is a list of lists, in which
    jj_geo[i][j] is a 6 x n matrix that represents the Jacobian of the j-th column of the geometric
    Jacobian matrix of the i-th DH or COM frame.

    jj_geo[i][j] for j>i is not computed, because the j-th column of the geometric Jacobian matrix of the
    i-th DH or COM frame is always the 6 x n zero matrix, regardless of the 'q' and 'htm' chosen.

    jj_geo[i][j] could be alternatively computed numerically. For example, for axis='dh', by defining the function of q
    f = lambda q_var: np.matrix(robot.jac_geo(q = q_var, htm = htm, axis = 'dh')[0][i][j])
    and then computing the numerical Jacobian as Utils.jac(f,q).
    However, this function  is faster and has greater numerical accuracy, since it is computed analytically
    instead of numerically.

    If axis='eef', this is the same as computing axis='dh' but throwing away all but the last list away.


    Parameters
    ----------
    q : n-dimensional numpy vector or array
        The manipulator's joint configuration
        (default: the current  joint configuration (robot.q) for the manipulator).

    axis : string
        For which axis you want to compute the FK:
        'eef': for the end-effector;
        'dh': for all Denavit-Hartenberg axis;
        'com': for all center-of-mass axis.
        (default: 'eef').

    htm : 4x4 numpy array or 4x4 nested list
        The robot base's configuration.
        (default: the same as the current htm).

    Returns
    -------
    jj_geo : list of lists of 6 x n numpy arrays (if axis='dh' or 'com') or a list of 6 x n numpy arrays (if axis=='eef')
        The Jacobian of the j-th column of the geometric Jacobian matrix of the i-th Denavit-Hartenberg frame.

    """
        return _jac_jac_geo(self, q, axis, htm)

    #######################################
    # Methods for dynamics model
    #######################################

    def dyn_model(self, q, qdot):
        """
    Compute the three robot's dynamic model terms, in a given joint configuration 'q'
    and joint speed 'qdot'.

    Parameters
    ----------
    q : n-dimensional numpy vector or array
        The manipulator's joint configuration.

    qdot : n-dimensional numpy vector or array
        The manipulator's joint configuration speed.

    Returns
    -------
    dyn_m : n x n numpy array
        The generalized inertia matrix at the joint configuration q.

    dyn_c : n-dimensional numpy column vector
        The generalized Coriolis-Centrifugal torques at the joint
        configuration q and joint configuration speed qdot.

    dyn_g : n-dimensional numpy column vector
        The generalized gravity torques at the joint configuration q.

    """
        return _dyn_model(self, q, qdot)

    #######################################
    # Methods for control
    #######################################
    @staticmethod
    def vector_field(curve, alpha=1, const_vel=1):
        """
    Computes a handle to a vector field function fun(p). Uses the vector field
    presented in 
    
    "Adriano M. C. Rezende; Vinicius M. Goncalves; Luciano C. A. Pimenta: 
    Constructive Time-Varying Vector Fields for Robot Navigation 
    IEEE Transactions on Robotics (2021)". 
    
    The vector field has constant velocity and use the function 
    G(u) = (2/pi)*atan(alpha*u).


    Parameters
    ----------
    curve : nxm numpy array or nxm nested list
        Curve, described as sampled points. Each one of the n rows should 
        contain a m-dimensional float vector that is the n-th m-dimensional
        sampled point of the curve. 
 
    alpha : positive float
        Controls the vector field behaviour. Greater alpha's imply more 
        robustness to the vector field, but increased velocity and acceleration
        behaviours. Used in G(u) = (2/pi)*atan(alpha*u)
        (default: 1).

    const_vel : positive float
        The constant velocity of the vector field. The signal of this number 
        controls the direction of rotation 
        (default: 1).

    Returns
    -------
    fun: a function handle that you can call as f(p), returning a numpy column vector. 'p' should be a
         m-dimensional vector.
    """

        return _vector_field(curve, alpha, const_vel)

    def task_function(self, htm_des, q=None, htm=None):
        """
    Computes the 6-dimensional task function for end-effector pose control,  
    given a joint configuration, a base configuration and the desired pose 
    'htm_des'.

    The first three entries are position error, and the three last entries are
    orientation error.

    Everything is written in scenario coordinates. 

    Also returns the Jacobian of this function.

    Parameters
    ----------
    htm_des : 4x4 numpy array or 4x4 nested list
        The desired end-effector pose. 
 
    q : nd numpy vector or array
        The manipulator's joint configuration.
        (default: the current  joint configuration (robot.q) for the manipulator).

    htm : 4x4 numpy array or 4x4 nested list
        The robot base's configuration.
        (default: the same as the current htm).

    Returns
    -------
    r : 6-dimensional numpy column vector
        The task function.

    jac_r : 6 x n numpy matrix
        The respective task Jacobian.
    """
        return _task_function(self, htm_des, q, htm)

    @staticmethod
    def coop_task_function(robot_a, robot_b, htm_a_des, htm_a_b_des, q_a=None, q_b=None):
        """
    Computes the 12-dimensional task function for end-effector pose control
    of two robots 'robot_a', 'robot_b' given their respective configurations
    q_a and q_b.

    
    The first three components are relative position error.  

    The second three components are relative orientation error.  

    The third three components are position error for 'robot_a'.

    The third three components are orientation error for 'robot_a'.
    
    Everything is written in scenario coordinates. 
    
    Also returns the Jacobian of this function.

    Parameters
    ----------
    robot_a :robot object
        The first robot.

    robot_b :robot object
        The second robot.

    htm_a_des :4x4 numpy array or 4x4 nested list
        The desired pose for the end-effector of 'robot_a'.

    htm_a_b_des :4x4 numpy array or 4x4 nested list
        The desired relative pose between the end-effector of 'robot_a' and
        'robot_b'. That is, inv(htmA) * htmB.

    q_a : nd numpy vector or array
        'robot_a'' joint configuration
        (default: the current  joint configuration (robot.q) for 'robot_a').

    q_b : md numpy vector or array
        'robot_b'' joint configuration
        (default: the current  joint configuration (robot.q) for 'robot_b').

    Returns
    -------
    r : 12-dimensional numpy column vector
        The task vector.

    jac_r : 12 x (n+m) numpy matrix
        The respective task Jacobian.
    """
        return _coop_task_function(robot_a, robot_b, htm_a_des, htm_a_b_des, q_a, q_b)

    def const_control(self, htm_des, q=None, htm=None, obstacles = [], dict_old_dist_struct=None,
                   eta_obs=0.5, eta_joint=0.5, eta_auto=0.5,
                   dist_safe_obs=0.01, dist_safe_auto=0.01,
                   max_dist_obs=np.inf, max_dist_auto = np.inf, task_rate_fun = 0.5, eps=0.001):
        """
    Computes the constrained joint velocity aiming a target pose (htm_des).
    Consider three kind of constraints: collision with obstacles, joint limits and
    auto collision.

    The obstacles should be simple objects (see Utils.IS_SIMPLE for the list).

    Parameters
    ----------

    htm_des :4x4 numpy array or 4x4 nested list
        The desired pose for the end-effector of the robot.

    q : nd numpy vector or array
        The joint configuration in which we want to compute the control action.
        (default: the current joint configuration for the robot).

    htm : 4x4 numpy array or 4x4 nested list
        The robot base's configuration.
        (default: the same as the current HTM).

    obstacles : list of simple objects
        A list of obstacles as simple objects (see Utils.IS_SIMPLE)
        (default: empty list).

    dict_old_dist_struct: a dictionary that maps each obstacle to a 'DistStructRobotObj'
        Used to speed up computation using data from a previous run. The output
        'dict_dist_struct' of this function can be fed as this parameter in a next
        call. If None is provided, this speed up is ignored.
        (default: None)

    eta_obs : positive float
        Parameter that controls obstacle avoidance, measured in s^(-1)
        (inverse seconds). The rule is that the rate of approximation
        to each obstacle, in meters per seconds, should be at most:

        eta_obs * (distance-safe_distance).

        So, the higher is this value, the less the robot
        is "afraid" of obstacles. Not that if this value is too high
        there can be collisions. If this value is np.inf, obstacle avoidance
        is skipped.
        (default: 0.5 s^(-1)).

    eta_joint : positive float
        Parameter that controls joint limits avoidance, measured in s^(-1)
        (inverse seconds). The rule is that the rate of approximation
        to each joint limit, in meters per seconds, should be at most:

        eta_joint * (distance to joint limit).

        So, the higher is this value, the less the robot
        is "afraid" of joint limits. Not that if this value is too high
        there can be collisions. If this value is np.inf, joint limits
        are skipped.
        (default: 0.5 s^(-1)).

    eta_joint : positive float
        Parameter that controls joint limits avoidance, measured in s^(-1)
        (inverse seconds). The rule is that the rate of approximation
        to each joint limit, in meters per seconds, should be at most:

        eta_joint * (distance to joint limit -safe_distance).

        So, the higher is this value, the less the robot
        is "afraid" of joint limits. Not that if this value is too high
        there can be collisions. If this value is np.inf, joint limits
        are skipped.
        (default: 0.5 s^(-1)).

    eta_auto : positive float
        Parameter that controls auto collision avoidance, measured in s^(-1)
        (inverse seconds). The rule is that the rate of approximation
        between each non-sequential link, in meters per seconds, should be at most:

        eta_auto * (distance -safe_distance).

        So, the higher is this value, the less the robot
        is "afraid" of auto collision. Not that if this value is too high
        there can be collisions. If this value is np.inf, auto collision is
        skipped.
        (default: 0.5 s^(-1)).

    dist_safe_obs : positive float
        Parameter that controls a safety margin to collision to obstacles,
        in meters.
        (default: 0.01 meter).

    dist_safe_auto : positive float
        Parameter that controls a safety margin to auto collision,
        in meters.
        (default: 0.01 meter).

    max_dist_obs : positive float
        Controls the maximum distance considered to skip collision between the object
        and obstacles. See the parameter 'max_dist' in the function 'compute_dist' of
        the robot.
        (default: np.inf).

    max_dist_auto : positive float
        Controls the maximum distance considered when skipping collision between the object
        and itself (auto collision). See the parameter 'max_dist' in the function 'compute_dist' of
        the robot.
        (default: np.inf).

    task_rate_fun : positive float or a function handle
        const_fun should be either a function handle that receives a 6x1 numpy column vector and
        outputs a 6x1 numpy column vector or a positive float.

        If task_rate_fun is a function f(r), it controls the target task function decrease. So we
        aim to have the task function decrease dr/dt as -f(r). In order to be suitable function, there
        should exist a positive definite function V(r) such that grad -V(r).T * f(r) <=0. This is
        not checked and is a responability of the user.

        If task_rate_fun is a positive float, is the same result as using the function handle
        lambda r : task_rate_fun*r. In this case V(r) = ||r||² is suitable.
        (default: 0.5).

    eps : positive float
        Damping factor. Control inputs generated by this procedure can be quite noisy/discontinuous. The
        higher is this value, the less is this behaviour. However, this can generate steady state errors
        in the controller.
        (default: 0.001).

    Returns
    -------

    qdot, failure, r, dict_dist_struct

    qdot : n-dimensional numpy column vector
        The joint velocity that can be used to control.

    failure: boolean
        The controller is computed using a quadratic program. If the problem is unfeasible,
        then this variable returns 'false. This typically happens when the configuration
        'q' violates one of the constraints (collision to obstacle, joint limits or
        auto collision).

    r : 6-dimensional numpy column vector
        The task vector.

    dict_dist_struct : a dictionary that maps each obstacle to a 'DistStructRobotObj'
        Used to speed up computation using data from a previous run. This output can be
        fed as the parameter 'dict_old_dist_struct' in a next call.
    """

        return _const_control(self, htm_des, q, htm, obstacles, dict_old_dist_struct,
                              eta_obs, eta_joint, eta_auto, dist_safe_obs, dist_safe_auto,
                              max_dist_obs, max_dist_auto, task_rate_fun, eps)

    def const_control_mod(self, htm_des, q=None, htm=None, obstacles = [], dict_old_dist_struct=None,
                   eta_obs=0.5, eta_joint=0.5, eta_auto=0.5,
                   dist_safe_obs=0.01, dist_safe_auto=0.01,
                   max_dist_obs=np.inf, max_dist_auto = np.inf, task_rate_fun = 0.5, eps=0.001):


        return _const_control_mod(self, htm_des, q, htm, obstacles, dict_old_dist_struct,
                              eta_obs, eta_joint, eta_auto, dist_safe_obs, dist_safe_auto,
                              max_dist_obs, max_dist_auto, task_rate_fun, eps)
    #######################################
    # Methods for simulation
    #######################################

    def gen_code(self):
        """Generate code for injection."""
        return _gen_code(self)

    def update_col_object(self, time):
        """
        Update internally the objects that compose the collision model to the
        current configuration of the robot.
        """
        _update_col_object(self, time)

    def add_col_object(self, sim):
        """
        Add the objects that compose the collision model to a simulation.

        Parameters
        ----------
        sim : 'Simulation' object
            'Simulation' object.
    """
        _add_col_object(self, sim)

    def attach_object(self, obj):
        """
        Attach an object to the end-effector.
        The list of the type of objects that can be grouped can be seen in 'Utils.IS_GROUPABLE'.

        Parameters
        ----------
        obj : Object that is one of the types in 'Utils.IS_GROUPABLE'
            Object to be attached.
    """
        _attach_object(self, obj)

    def detach_object(self, obj):
        """
        Detach an object (ball, box or cylinder) to the end-effector.

        Parameters
        ----------
        obj : object
            Object to be detached.
    """
        _detach_object(self, obj)

    def set_htm_to_eef(self, htm):
        self._htm_n_eef = htm

    #######################################
    # Robot constructors
    #######################################

    @staticmethod
    def create_kuka_kr5(htm=np.identity(4), name='kukakr5', color="#df6c25", opacity=1, eef_frame_visible=True):
        """
    Create a Kuka KR5 R850, a six-degree of freedom manipulator.
    Thanks Sugi-Tjiu for the 3d model (see https://grabcad.com/library/kuka-kr-5-r850).

    Parameters
    ----------
    htm : 4x4 numpy array or 4x4 nested list
        The initial base configuration for the robot.
        (default: np.identity(4))
 
    name : string
        The robot name.
        (default: 'kukakr5').

    htm : color
        A HTML-compatible string representing the object color.
        (default: '#df6c25')'.

    opacity : positive float between 0 and 1
        The opacity of the robot. 1 = fully opaque and 0 = transparent.
        (default: 1)

    Returns
    -------
    robot : Robot object
        The robot.

    """
        base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits = _create_kuka_kr5(htm, name, color, opacity)
        return Robot(name, links, base_3d_obj, htm, htm_base_0, htm_n_eef, q0, eef_frame_visible, joint_limits)

    @staticmethod
    def create_kuka_kr5_per(htm=np.identity(4), name='kukakr5', color="#df6c25", opacity=1, eef_frame_visible=True, per=0.05):
        """
    Create a Kuka KR5 R850, a six-degree of freedom manipulator.
    Thanks Sugi-Tjiu for the 3d model (see https://grabcad.com/library/kuka-kr-5-r850).

    Parameters
    ----------
    htm : 4x4 numpy array or 4x4 nested list
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: 'kukakr5').

    htm : color
        A HTML-compatible string representing the object color.
        (default: '#df6c25')'.

    opacity : positive float between 0 and 1
        The opacity of the robot. 1 = fully opaque and 0 = transparent.
        (default: 1)

    Returns
    -------
    robot : Robot object
        The robot.

    """
        base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits = _create_kuka_kr5_per(htm, name, color, opacity, per)
        return Robot(name, links, base_3d_obj, htm, htm_base_0, htm_n_eef, q0, eef_frame_visible, joint_limits)

    @staticmethod
    def create_epson_t6(htm=np.identity(4), name='epsont6', color="white", opacity=1, eef_frame_visible=True):
        """
    Create an Epson T6, a SCARA manipulator.
    Thanks KUA for the 3d model (see https://grabcad.com/library/epson-t6-scara-robot-1).

    Parameters
    ----------
    htm : 4x4 numpy array or 4x4 nested list
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: 'epsont6').

    htm : color
        A HTML-compatible string representing the object color.
        (default: 'white')'.

    opacity : positive float between 0 and 1
        The opacity of the robot. 1 = fully opaque and 0 = transparent.
        (default: 1)

    Returns
    -------
    robot : Robot object
        The robot.

    """
        base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits = _create_epson_t6(htm, name, color, opacity)
        return Robot(name, links, base_3d_obj, htm, htm_base_0, htm_n_eef, q0, eef_frame_visible, joint_limits)

    @staticmethod
    def create_staubli_tx60(htm=np.identity(4), name='staublitx60', color="#ff9b00", opacity=1, eef_frame_visible=True):
        """
    Create a Staubli TX60, a six degree of freedom manipulator.
    Model taken from the ROS github repository (https://github.com/ros-industrial/staubli).

    Parameters
    ----------
    htm : 4x4 numpy array or 4x4 nested list
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: 'staublitx60').

    htm : color
        A HTML-compatible string representing the object color.
        (default: '#ff9b00')'.

    opacity : positive float between 0 and 1
        The opacity of the robot. 1 = fully opaque and 0 = transparent.
        (default: 1)

    Returns
    -------
    robot : Robot object
        The robot.

    """
        base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits = _create_staubli_tx60(htm, name, color, opacity)
        return Robot(name, links, base_3d_obj, htm, htm_base_0, htm_n_eef, q0, eef_frame_visible, joint_limits)

    @staticmethod
    def create_kuka_lbr_iiwa(htm=np.identity(4), name='kukalbriiwa', color="silver", opacity=1, eef_frame_visible=True):
        """
    Create a Kuka LBR IIWA 14 R820, a seven degree of freedom manipulator.
    Model taken from the ROS github repository (https://github.com/ros-industrial/kuka_experimental).

    Parameters
    ----------
    htm : 4x4 numpy array or 4x4 nested list
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: 'kukalbriiwa').

    htm : color
        A HTML-compatible string representing the object color.
        (default: 'silver')'.

    opacity : positive float between 0 and 1
        The opacity of the robot. 1 = fully opaque and 0 = transparent.
        (default: 1)

    Returns
    -------
    robot : Robot object
        The robot.

    """
        base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits = _create_kuka_lbr_iiwa(htm, name, color, opacity)
        return Robot(name, links, base_3d_obj, htm, htm_base_0, htm_n_eef, q0, eef_frame_visible, joint_limits)

    @staticmethod
    def create_franka_ermika_3(htm=np.identity(4), name='frankaermika', color="silver", opacity=1, eef_frame_visible=True):
        """
    Create a Franka Ermika 3, a seven degree of freedom manipulator.
    Model taken from the ROS github repository (https://github.com/BolunDai0216/FR3Env/tree/d5218531471cadafd395428f8c2033f6feeb3555/FR3Env/robots/meshes/visual).

    Parameters
    ----------
    htm : 4x4 numpy array or 4x4 nested list
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: 'frankaermika').

    htm : color
        A HTML-compatible string representing the object color.
        (default: 'silver')'.

    opacity : positive float between 0 and 1
        The opacity of the robot. 1 = fully opaque and 0 = transparent.
        (default: 1)

    Returns
    -------
    robot : Robot object
        The robot.

    """
        base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits = _create_franka_ermika_3(htm, name, color, opacity)
        return Robot(name, links, base_3d_obj, htm, htm_base_0, htm_n_eef, q0, eef_frame_visible, joint_limits)

    @staticmethod
    def create_abb_crb(htm=np.identity(4), name='abbcrb', color="white", opacity=1, eef_frame_visible=True):
        """
    Create a ABB CRB 15000, a six degree of freedom manipulator.
    Model taken from the ROS github repository (https://github.com/ros-industrial/abb_experimental).

    Parameters
    ----------
    htm : 4x4 numpy array or 4x4 nested list
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: 'abbcrb').

    htm : color
        A HTML-compatible string representing the object color.
        (default: 'white')'.

    opacity : positive float between 0 and 1
        The opacity of the robot. 1 = fully opaque and 0 = transparent.
        (default: 1)

    Returns
    -------
    robot : Robot object
        The robot.

    """
        base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits = _create_abb_crb(htm, name, color, opacity)
        return Robot(name, links, base_3d_obj, htm, htm_base_0, htm_n_eef, q0, eef_frame_visible, joint_limits)

    @staticmethod
    def create_darwin_mini(htm=np.identity(4), name="darwin_mini", color="#3e3f42", opacity=1, eef_frame_visible=True):
        """
    Create an (oversized) Darwin Mini, a humanoid robot.
    Thanks to Alexandre Le Falher for the 3D model (https://grabcad.com/library/darwin-mini-1).

    Parameters
    ----------
    htm : 4x4 numpy array or 4x4 nested list
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: 'darwin_mini').

    htm : color
        A HTML-compatible string representing the object color.
        (default: '#3e3f42').

    opacity : positive float between 0 and 1
        The opacity of the robot. 1 = fully opaque and 0 = transparent.
        (default: 1)

    Returns
    -------
    robot : Group object
        The robot. It is composed of a group of six objects: the two arms and legs (members of 'Robot' class)
        and the chest and head (both 'RigidObject' class)


    """

        param_arm_left, param_arm_right, param_leg_left, param_leg_right, head, chest = _create_darwin_mini(htm, name,
                                                                                                            color,
                                                                                                            opacity)
        desl_z = htm * Utils.trn([0, 0, -0.18])
        robot_arm_left = Robot(name + "__arm_left", param_arm_left[1], param_arm_left[0],
                               desl_z * Utils.trn([0, 0.14, 1]) * Utils.rotx(-3.14 / 2), param_arm_left[2],
                               param_arm_left[3], [np.pi/2,0.3,0], eef_frame_visible,
                               param_arm_left[5])

        robot_arm_right = Robot(name + "__arm_right", param_arm_right[1], param_arm_right[0],
                                desl_z * Utils.rotz(3.14) * Utils.trn([0, 0.14, 1]) * Utils.rotx(-3.14 / 2),
                                param_arm_right[2],
                                param_arm_right[3], [np.pi/2,0.3,0], eef_frame_visible, param_arm_right[5])

        robot_leg_left = Robot(name + "__leg_left", param_leg_left[1], param_leg_left[0],
                               desl_z * Utils.trn([0, -0.1, 0.7]) * Utils.roty(np.pi / 2) * Utils.rotz(-np.pi / 2),
                               param_leg_left[2], param_leg_left[3], [np.pi/2, 0, 0, np.pi/2], eef_frame_visible, param_leg_left[5])

        robot_leg_right = Robot(name + "__leg_right", param_leg_right[1], param_leg_right[0],
                                desl_z * Utils.trn([0, 0.1, 0.7]) * Utils.roty(np.pi / 2) * Utils.rotz(-np.pi / 2),
                                param_leg_right[2], param_leg_right[3], [np.pi/2, 0, 0, np.pi/2], eef_frame_visible, param_leg_right[5])

        return Group([robot_arm_left, robot_arm_right, robot_leg_left, robot_leg_right, head, chest])
    
    @staticmethod
    def create_davinci(htm=np.identity(4), name="davinci", color=None, opacity=1, eef_frame_visible=True):
        """Create a da Vinci Si, a surgical robot.
        Thanks to Koray Okan for the 3D model (https://grabcad.com/library/da-vinci-surgical-robot-1/details).
        Parameters
        ----------
        htm : 4x4 numpy array or 4x4 nested list
            The initial base configuration for the robot.
            (default: np.identity(4))
        name : string
            The robot name.
            (default: 'davinci').
        color  : string or list of string or None
            A HTML-compatible string representing the object color.
            (default: ["#919090", "#61b3c7", "#a6a6a6", "#707070", "#545353"]).
        opacity : positive float between 0 and 1
            The opacity of the robot. 1 = fully opaque and 0 = transparent.
            (default: 1)
        Returns
        -------
        robot : Group object
            The robot. It is composed of a group of six objects: the four arms (members of 'Robot' class) and the chest 
            ('RigidObject' class)
        """

        return _create_davinci(htm, name, color, opacity, eef_frame_visible)
    
    @staticmethod
    def create_jaco(htm=np.identity(4), name='jaco', color=None, opacity=1, eef_frame_visible=True):
        """Create a Kinova Jaco 2.
        Model taken from Kinova's resources (https://www.kinovarobotics.com/resources?r=79302&s).
        Robot documentation available at https://www.kinovarobotics.com/resources?r=339.
        Parameters
        ----------
        htm : 4x4 numpy array or 4x4 nested list
            The initial base configuration for the robot.
            (default: np.identity(4))
        name : string
            The robot name.
            (default: 'jaco').
        color  : string or list of string or None
            A HTML-compatible string representing the object color.
            (default: ["#3e3f42", "#919090", "#1d1d1f"]).
        opacity : positive float between 0 and 1
            The opacity of the robot. 1 = fully opaque and 0 = transparent.
            (default: 1)
        Returns
        -------
        robot : Group object
            The robot.
        """
        links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits = _create_jaco(htm=htm, name=name, color=color, 
                                                                                   opacity=opacity)
        jaco = Robot(name=name, links=links, list_base_3d_obj=base_3d_obj, htm=np.identity(4), htm_base_0=htm_base_0, 
                     htm_n_eef=htm_n_eef, q0=q0, eef_frame_visible=eef_frame_visible, joint_limits=joint_limits)
        return jaco

    #######################################
    # Distance computation and collision
    #######################################

    def compute_dist(self, obj, q=None, htm=None, old_dist_struct=None, tol=0.0005,
                     no_iter_max=20, max_dist = np.inf):
        """
    Compute the  distance structure from each one of the robot's link to a
    'simple' external object (see Utils.IS_SIMPLE), given a joint and base
    configuration.

    This function can be faster if some distance computations are avoided.
    See the description of the parameter 'max_dist'.

    Use an iterative algorithm, based on projections
    (Von Neumann's cyclic projection algorithm).

    Parameters
    ----------
    obj : a simple object (see Utils.IS_SIMPLE)
        The external object for which the distance structure is going to be 
        computed, for each robot link.

    q : nd numpy vector or array
        The manipulator's joint configuration.
        (default: the current  joint configuration (robot.q) for the manipulator).

    htm : 4x4 numpy array or 4x4 nested list
        The robot base's configuration.
        (default: the same as the current htm).

    old_dist_struct : 'DistStructRobotObj' object
        'DistStructRobotObj' obtained previously for the same robot and external object.
        Can be used to enhance the algorithm speed using the previous closest
        point as an initial guess.
        (default: None).

    tol : positive float
        Tolerance for convergence in the iterative algorithm, in meters.
        (default: 0.0005 m).

    no_iter_max : positive int
        The maximum number of iterations for the algorithm.
        (default: 20 iterations).

    max_dist: positive float
        The algorithm uses an axis aligned bounding box (aabb) to avoid some distance computations.
        The algorithm skips a more precise distance computation if the distance between the aabb
        of the primitive objects composing the link and 'obj' is less than 'max_dist' (in meters).
        (default: infinite).

    Returns
    -------
    dist_struct : 'DistStructRobotObj' object
        Distance struct for each one of the m objects that compose the robot's
        collision model. Contains a list of m 'DistStructLinkObj' objects.
    """

        return _compute_dist(self, obj, q, htm, old_dist_struct, tol, no_iter_max, max_dist)

    def compute_dist_auto(self, q=None, old_dist_struct=None, tol=0.0005,
                     no_iter_max=20, max_dist = np.inf):
        """
    Compute the  distance structure from each one of the robot's links to itself
    (auto collision), given a joint and base configuration.

    This function consider only non-sequential links, since the collision between
    sequential links in the kinematic chain can be verified by checking joint limits.
    This saves times. This verification should be done elsewhere (by checking if the
    configuration is inside the joint limits).

    This function can be faster if some distance computations are avoided.
    See the description of the parameter 'max_dist'.

    Use an iterative algorithm, based on projections
    (Von Neumann's cyclic projection algorithm).

    Parameters
    ----------
    q : nd numpy vector or array
        The manipulator's joint configuration.
        (default: the current  joint configuration (robot.q) for the manipulator).

    old_dist_struct : 'DistStructRobotAuto' object
        'DistStructRobotAuto' obtained previously for the same robot.
        Can be used to enhance the algorithm speed using the previous closest
        point as an initial guess.
        (default: None).

    tol : positive float
        Tolerance for convergence in the iterative algorithm, in meters.
        (default: 0.0005 m).

    no_iter_max : positive int
        The maximum number of iterations for the algorithm.
        (default: 20 iterations).

    max_dist: positive float
        The algorithm uses an axis aligned bounding box (aabb) to avoid some distance computations.
        The algorithm skips a more precise distance computation if the distance between the aabb
        of the primitive objects composing the link and 'obj' is less than 'max_dist' (in meters).
        (default: infinite).

    Returns
    -------
    dist_struct : 'DistStructRobotAuto' object
        Distance struct for each one of the m objects that compose the robot's
        collision model. Contains a list of m 'DistStructLinkLink' objects.
    """

        return _compute_dist_auto(self, q, old_dist_struct, tol, no_iter_max, max_dist)

    def check_free_configuration(self, q=None, htm=None, obstacles=[],
                              check_joint=True, check_auto=True,
                              tol=0.0005, dist_tol=0.005, no_iter_max=20):
        """
    Check if the joint configuration q is in the free configuration space, considering
    joint limits, collision with obstacles and auto collision. It also outputs a message about a
    possible violation.

    For efficiency purposes, the program halts in the first violation it finds (if there is any).
    So, the message, if any, is only about one of the possible violations. There can be more.

    Parameters
    ----------
    q : nd numpy vector or array
        The manipulator's joint configuration.
        (default: the current joint configuration (robot.q) for the manipulator).

    obstacles : list of simple objects
        A list of obstacles as simple objects (see Utils.IS_SIMPLE)
        (default: empty list).

    htm : 4x4 numpy array or 4x4 nested list
        The robot base's configuration.
        (default: the same as the current htm).


    check_joint: boolean
        If joint limits should be considered or not.
        (default: True).

    check_auto: boolean
        If auto collision should be considered or not.
        (default: True).

    tol : positive float
        Tolerance for convergence in the iterative algorithm, in meters.
        (default: 0.0005 m).

    dist_tol : positive float
        The tolerance to consider that two links are colliding.
        (default: 0.005 m).

    no_iter_max : positive int
        The maximum number of iterations for the distance computing algorithm.
        (default: 20 iterations).

    Returns
    -------
    is_free : boolean
        If the configuration is free or not.

    message: string
        A message about what is colliding (is otherwise just 'Ok!').

    info: list
        Contains a list with information of the violation of free space (if there is any, otherwise
        it is empty). The first element is the violation type: either 0 (lower joint limit violated), 1 (upper joint
        limit violated) 2 (collision with obstacles) and 3 (collision between links). The last elements depends on
        the violation type.

        If lower joint limit or upper joint limit, contains which joint was violated.

        If collision with obstacles, contains the list [i,isub,j], containing the index of the link i
        i, the index of the collision object in the link isub and the obstacle index in the list, j.

        If collision, contains the list [i,isub,j,jsub], containing the index of the first link i,
        the index of the collision object in the first link isub, the index of the second link j and
        the index of the collision object in the second link jsub.


    """

        return _check_free_configuration(self, q, htm, obstacles, check_joint, check_auto, tol, dist_tol, no_iter_max)