import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
from utils import *
from graphics.meshmaterial import *
from simobjects.group import *
from simobjects.frame import *

from ._set_ani_frame import _set_ani_frame
from ._add_ani_frame import _add_ani_frame



from ._ikm import _ikm
from ._fkm import _fkm
from ._jac_geo import _jac_geo
from ._jac_jac_geo import _jac_jac_geo
from ._jac_ana import _jac_ana

from ._dyn_model import _dyn_model

from ._vector_field_rn import _vector_field_rn
from ._task_function import _task_function
from ._vector_field_SE3 import _vector_field_SE3, _DELTA as _DELTA_SE3, _DS as _DS_SE3


from ._gen_code import _gen_code
from ._update_col_object import _update_col_object
from ._add_col_object import _add_col_object
from ._attach_object import _attach_object
from ._detach_object import _detach_object

from ._compute_dist import _compute_dist
from ._compute_dist_auto import _compute_dist_auto
from ._check_free_config import _check_free_config

from ._create_kuka_kr5 import _create_kuka_kr5
from ._create_epson_t6 import _create_epson_t6
from ._create_staubli_tx60 import _create_staubli_tx60
from ._create_kuka_lbr_iiwa import _create_kuka_lbr_iiwa
from ._create_abb_crb import _create_abb_crb
from ._create_darwin_mini import _create_darwin_mini
from ._create_franka_emika_3 import _create_franka_emika_3
from ._create_davinci import _create_davinci
from ._create_magician_e6 import _create_magician_e6
from ._create_kinova_gen3 import _create_kinova_gen3
from ._create_jaco import _create_jaco
from ._create_ur_ur3e import _create_ur_ur3e

from .links import *

from ._constrained_control import _constrained_control
from uaibot.simulation.simulation import *

from ._dist_struct_robot_obj import *
from ._dist_struct_robot_auto import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from uaibot.robot import DistStructRobotObj
    from uaibot.robot import DistStructRobotAuto
    from uaibot.robot import Link
    from uaibot.simulation import Simulation

from numpy.typing import ArrayLike, NDArray

import os
if os.environ['CPP_SO_FOUND']=="1":
    import uaibot_cpp_bind as ub_cpp
    from ._to_cpp import _to_cpp

from uaibot.utils.types import HTMatrix, Matrix, Vector, MetricObject, GroupableObject
from typing import Optional, Tuple, List


class Robot:
    """
  A class that contains a robot object in UAIBot.

  Parameters
  ----------

  name : string
      The robot name.
      (default: 'genRobot').

  htm_base_0 : 4x4 numpy matrix
      The transformation between the robot's base and the first DH frame.
      (default: 4x4 identity matrix).

  htm_n_eef : 4x4 numpy matrix
      The transformation between the last DH and the end-effector.
      (default: 4x4 identity matrix).

  list_object_3d_base : list of 'uaibot.Model3D' objects
      The list of 3d models of the base of the robot.
      If set to None, there is no base 3d object.
      (default: None).

  links : A list of 'uaibot.Link' objects
      The list of link objects.

  q0 : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
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
    def q(self) -> Vector:
        """The current joint configuration."""
        return np.matrix(self._q)

    @property
    def q0(self) -> Vector:
        """The default joint configuration."""
        return np.matrix(self._q0)

    @property
    def htm(self) -> HTMatrix:
        """
        The current base configuration in scenario coordinates.
        A 4x4 homogeneous matrix written is scenario coordinates.
        """
        return np.matrix(self._htm)

    @property
    def htm_base_0(self) -> HTMatrix:
        """
        The constant homogeneous transformation between the HTM of the base and
        the HTM of the first Denavit-Hartenberg frame.
        """
        return np.matrix(self._htm_base_0)

    @property
    def htm_n_eef(self) -> HTMatrix:
        """
        The constant homogeneous transformation between the HTM of the last
        Denavit-Hartenberg frame and the end-effector
        """
        return np.matrix(self._htm_n_eef)

    @property
    def links(self) -> List["Link"]:
        """Data structures containing the links of the robot."""
        return self._links

    @property
    def attached_objects(self) -> List[GroupableObject]:
        """Data structures containing the objects attached into the robot."""
        return self._attached_objects

    @property
    def name(self) -> str:
        """Name of the object."""
        return self._name

    @property
    def list_object_3d_base(self) -> List[Model3D]:
        """The list of 3d objects that form the base."""
        return self._list_object_3d_base

    @property
    def eef_frame_visible(self) -> bool:
        """If the frame attached to the end effector is visible"""
        return self._eef_frame_visible

    @property
    def joint_limit(self) -> np.matrix:
        """A n x 2 numpy array containing the joint limits, either in rad or meters"""
        return self._joint_limit

    @property
    def cpp_robot(self):
        """Used in the c++ interface"""
        return self._cpp_robot
    
    @property
    def gravity_vector(self):
        """The gravity vector used in the dynamic model of the robot"""
        return self._gravity_vector

    @gravity_vector.setter
    def gravity_vector(self, vector):
        if not Utils.is_a_vector(vector, 3):
            raise Exception("The parameter 'vector' should be a 3D vector.")
        self._gravity_vector = np.array(vector).reshape((3,1))
        if os.environ['CPP_SO_FOUND']=="1":
            self._cpp_robot.gravity_vector = self._gravity_vector
    #######################################
    # Constructor
    #######################################

    def __init__(self, name: str, links: List["Link"], list_base_3d_obj : Optional[HTMatrix] = None, 
                 htm: HTMatrix =np.identity(4), htm_base_0: HTMatrix = np.identity(4),
                 htm_n_eef: HTMatrix = np.identity(4), q0 : Optional[Vector] = None, 
                 eef_frame_visible: bool =True, joint_limits: Optional[np.matrix] = None) -> "Robot":
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
        
        if name=="":
            name="var_robot_id_"+str(id(self))
            
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
        self._gravity_vector = np.array([0,0,-9.81]).reshape((3,1))

        
        if eef_frame_visible:
            self._eef_frame = Frame(size=0.1)
            
        self._max_time = 0

        #Create the cpp robot
        if os.environ['CPP_SO_FOUND']=="1":
            self._cpp_robot = _to_cpp(self)
            
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
        string += " Current configuration: " + str([round(num[0,0], 3) for num in self.q]) + "\n"
        string += " Current base HTM: \n" + str(self.htm) + "\n"
        string += " Current end-effector HTM: \n" + str(self.fkm())
        return string

    #######################################
    # Methods for configuration changing
    #######################################

    def add_ani_frame(self, time: float, q: Optional[Vector]=None, htm: Optional[HTMatrix]=None, 
                      enforce_joint_limits: bool = False) -> None:
        """
    Add a single configuration to the object's animation queue.

    Parameters
    ----------
    time: positive float
        The timestamp of the animation frame, in seconds.
    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The manipulator's joint configuration.
    htm : 4x4 numpy matrix
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

    def set_ani_frame(self, q: Optional[Vector]=None, htm: Optional[HTMatrix]=None, 
                      enforce_joint_limits: bool = False) -> None:
        """
    Reset object's animation queue and add a single configuration to the 
    object's animation queue.

    Parameters
    ----------
    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The manipulator's joint configuration.
        (default: the current joint configuration (robot.q) for the manipulator, q0).
    htm : 4x4 numpy matrix
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

    def fkm(self, q: Optional[Vector] = None, axis: str ='eef', htm: Optional[HTMatrix]=None, 
            mode: str ='auto') -> List[HTMatrix]:
        """
    Compute the forward kinematics for an axis at a given joint and base
    configuration. Everything is written in the scenario coordinates.

    Parameters
    ----------
    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The manipulator's joint configuration.
        (default: the current  joint configuration (robot.q) for the manipulator).
    axis : string
        For which axis you want to compute the FK:
        'eef': for the end-effector;
        'dh': for all Denavit-Hartenberg axis;
        'com': for all center-of-mass axis.
        (default: 'eef').    
    htm : 4x4 numpy matrix
        The robot base's configuration.
        (default: the same as the current HTM).
    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto').

    Returns
    -------
    htm_fk : list of 4x4 numpy matrices
        For axis='eef', returns a single htm. For the other cases, return
        n htms as a list of 4x4 numpy matrices.
    """
        return _fkm(self, q, axis, htm, mode)

    def ikm(self, htm_tg: HTMatrix, htm: Optional[HTMatrix]=None, q0: Optional[Vector]=None, p_tol: float=0.001, 
            a_tol: float =5, no_iter_max: int =200, ignore_orientation: bool = False, 
            no_tries = 40, check_joint: bool = True, check_auto: bool = True, 
            obstacles: List[MetricObject]=[], mode: str ='auto') -> Vector:
        """
    Try to solve the inverse kinematic problem for the end-effector, given a
    desired homogeneous transformation matrix. It returns the manipulator
    configuration.

    Uses an iterative algorithm.

    The algorithm can fail, throwing an Exception when it happens.

    Parameters
    ----------
    htm_tg : 4x4 numpy matrix
        The target end-effector HTM, written in scenario coordinates.       
    htm : 4x4 numpy matrix
        The pose of the basis of the manipulator.
        (default: 'None' (the current base htm))
    q0 : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        Initial guess for the algorithm for the joint configuration.
        (default: a random joint configuration).
    p_tol : positive float
        The accepted error for the end-effector position, in meters.
        (default: 0.001 m).    
    a_tol : positive float
        The accepted error for the end-effector orientation, in degrees.
        (default: 5 degrees). 
    no_iter_max : positive int
        The maximum number of iterations for the algorithm.
        (default: 200 iterations).
    ignore_orientation: boolean
        If True, the orientation part of the HTM is ignored. Task is position-only.
        (default: False)
    no_tries: positive int.
        How many times the algorithm tries to find a solution.
        (default: 40)
    check_joint: boolen
        If True, consider the joint limits as well.
        (default: True)
    check_auto: boolen
        If True, consider the auto collision of the robot.
        (default: True)
    obstacles: list of 'MetricObject' objects
        List of objects as obstacles to consider.
        (default: Empty list)
    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto').

    Returns
    -------
    q : n x 1 numpy matrix
        The configuration that solves the IK problem.
    """
    
        return _ikm(self, htm_tg, htm, q0, p_tol, a_tol, no_iter_max, ignore_orientation, 
                    no_tries, check_joint, check_auto, obstacles, mode)

    def jac_geo(self, q: Optional[Vector] = None, axis: str ='eef', htm : Optional[HTMatrix]=None, 
                mode: str ='auto') -> Tuple[np.matrix,HTMatrix]:
        """
    Compute the geometric Jacobian for an axis at a given joint and base
    configuration. Also returns the forward kinematics as a by-product.
    Everything is written in the scenario coordinates.

    Parameters
    ----------
    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The manipulator's joint configuration 
        (default: the current  joint configuration (robot.q) for the manipulator).
    axis : string
        For which axis you want to compute the FK:
        'eef': for the end-effector;
        'dh': for all Denavit-Hartenberg axis;
        'com': for all center-of-mass axis.
        (default: 'eef').    
    htm : 4x4 numpy matrix
        The robot base's configuration.
        (default: the same as the current htm).
    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto').

    Returns
    -------
    jac_geo : 6 x n or list of 6 x n numpy matrices
        For axis='eef', returns a single 6xn Jacobian. For the other cases, 
        return n Jacobians as a list of n 6xn numpy matrices.

    htm_out : 4 x 4 or list of 4 x 4 numpy matrices
        For axis='eef', returns a single htm as a 4x4 matrix. For the other cases, return
        n htms as  a list of n 4x4 numpy matrices.
    """
        return _jac_geo(self, q, axis, htm, mode)

    def jac_ana(self, q: Optional[Vector]=None, htm: Optional[HTMatrix]=None) -> Tuple[Matrix,HTMatrix,Vector]:
        """
    Compute the analytic Jacobian for the end-effector. The Euler angle
    convention is zyx. Also returns the end-effector htm and Euler angles
    as a by-product.

    Parameters
    ----------
    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The manipulator's joint configuration
        (default: the current  joint configuration (robot.q) for the manipulator).
    htm : 4x4 numpy matrix
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

    def jac_jac_geo(self, q: Optional[Vector] =None, axis: str ='eef', 
                    htm: Optional[HTMatrix] = None)-> List[np.matrix]:
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
    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The manipulator's joint configuration
        (default: the current  joint configuration (robot.q) for the manipulator).

    axis : string
        For which axis you want to compute the FK:
        'eef': for the end-effector;
        'dh': for all Denavit-Hartenberg axis;
        'com': for all center-of-mass axis.
        (default: 'eef').

    htm : 4x4 numpy matrix
        The robot base's configuration.
        (default: the same as the current htm).

    Returns
    -------
    jj_geo : list of lists of 6 x n numpy arrays (if axis='dh' or 'com') or a list of 6 x n numpy arrays (if axis=='eef')
        The Jacobian of the j-th column of the geometric Jacobian matrix of the i-th Denavit-Hartenberg frame.

    """
        return _jac_jac_geo(self, q, axis, htm)

    def newton_euler(self, q: Optional[Vector] = None, qdot: Optional[Vector] = None, qddot: Optional[Vector] = None) -> Vector:
        """Returns the wrench vector (forces and torques) at the
        effector. Computed using the Newton-Euler algorithm.
        Parameters
        ----------
        q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped
            numpy matrix/numpy array). The manipulator's joint configuration
            (default: the current  joint configuration (robot.q) for the manipulator).
        qdot : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
            The manipulator's joint velocities
            (default: zero vector).
        qddot : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
            The manipulator's joint accelerations
            (default: zero vector).

        Returns
        -------
        wrench : 6 x 1 numpy array
            The wrench at the end-effector, as a 6 x 1 numpy array.
            The first three entries are forces, and the last three
            entries are torques.
        """
        return self.cpp_robot.newtonEuler(q, qdot, qddot)

    def get_euler_lagrange_matrices(self, q: Optional[Vector] = None, qdot: Optional[Vector] = None) -> Tuple[np.array, np.array, np.array]:
        """Returns the matrices of the dynamic model of the robot, in the
        Euler-Lagrange formulation:

        M(q)qddot + C(q,qdot)qdot + g(q) = w
                   |==C_(q,qdot)=|

        Parameters
        ----------
        q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped
            numpy matrix/numpy array). The manipulator's joint configuration
            (default: the current  joint configuration (robot.q) for the manipulator).
        qdot : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
            The manipulator's joint velocities
            (default: zero vector).

        Returns
        -------
        M : n x n numpy array
            The inertia matrix.
        C_ : n x n numpy array
            The Coriolis and centripetal matrix multiplied by qdot.
        g : n x 1 numpy array
            The gravity vector.
        """
        return self.cpp_robot.getEulerLagrangeMatrices(q, qdot)
    #######################################
    # Methods for control
    #######################################
    
    @staticmethod
    def vector_field(q, curve: List[Vector], alpha: float =1, 
                     const_vel: float =1, is_closed: bool = True, gamma: float=10, mode: str ='auto') -> Tuple[np.matrix,float,int]:
        """
    Computes the vector field presented in 
    
    "Adriano M. C. Rezende; Vinicius M. Goncalves; Luciano C. A. Pimenta: 
    Constructive Time-Varying Vector Fields for Robot Navigation 
    IEEE Transactions on Robotics (2021)". 
    
    The vector field has constant velocity and use the function 
    G(p) = (2/pi)*atan(alpha*sqrt(Dist(p))).
    
    in which "Dist(p)" is the Euclidean distance to the curve and alpha>0 a parameter.
    
    If the curve is not closed, the circulation component is modulated by:
    
    R(p) = min(gamma*(1.0 - s(p)),1.0)
    
    in which s(p) in [0,1] is the fraction of the curve travelled at the point p*(p) 
    (i.e., the closest point to the curve to point p) and gamma>0.
    

    Parameters
    ----------
    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The configuration in which the vector field should be computed.

    curve : a list of nD vectors (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        Curve, described as sampled points. 
 
    alpha : positive float
        Controls the vector field behaviour. Greater alpha's imply more 
        robustness to the vector field, but increased velocity and acceleration
        behaviours. Used in G(u) = (2/pi)*atan(alpha*u)
        (default: 1).

    const_vel : positive float
        The constant velocity of the vector field. The signal of this number 
        controls the direction of rotation 
        (default: 1).
        
    is_closed: bool
        If the curve is closed or not.
        (default: True)
        
    gamma: positive float
        The parameter of the function that sends the circulation 
        component to zero when close to the end of the curve. 
        Only active when it 'is_closed' is True.
        (default: 0.5)

    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto').

    Returns
    -------
    qdot : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The velocity generated by the vector field.

    dist : float
        Distance to the curve.

    index : integer
        The index of the closest point to the curve.
    """

        return _vector_field_rn(q, curve, alpha, const_vel, is_closed, gamma, mode)
 
    def task_function(self, htm_tg: HTMatrix, q: Optional[None]=None, 
                      htm: Optional[Vector] = None, mode: str = 'auto') -> Tuple[np.matrix,np.matrix]:
        """
    Computes the 6-dimensional task function for end-effector pose control,  
    given a joint configuration, a base configuration and the desired pose 
    'htm_tg'.

    The first three entries are position error, and the three last entries are
    orientation error.

    Everything is written in scenario coordinates. 

    Also returns the Jacobian of this function.

    Parameters
    ----------
    htm_tg : 4x4 numpy array 
        The target end-effector pose. 
     
    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The manipulator's joint configuration.
        (default: the current  joint configuration (robot.q) for the manipulator).

    htm : 4x4 numpy array 
        The robot base's configuration.
        (default: the same as the current htm).

    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto').

    Returns
    -------
    r : 6 x 1 numpy matrix
        The task function.

    jac_r : 6 x n numpy matrix
        The respective task Jacobian.
    """
    
        return _task_function(self, htm_tg, q, htm, mode)

    def constrained_control(self, htm_tg: HTMatrix, q: Optional[Vector] = None, obstacles: List[MetricObject]=[], 
                            htm: Optional[HTMatrix] = None, Kp: float =  2.0, eta_obs: float = 0.3, eta_auto: float = 0.3, eta_joint: float = 0.3, 
                              eps_to_obs: float = 0.003, h_to_obs: float = 0.003, eps_auto: float = 0.02,  h_auto: float = 0.05, 
                              d_safe_obs: float = 0.02, d_safe_auto: float = 0.002, d_safe_jl: float = (np.pi/180)*5,
                              eps_reg: float = 0.01) -> Tuple[np.matrix, float, float, float, float]:
        

        
        return _constrained_control(self, htm_tg, q, obstacles, htm, Kp, eta_obs, eta_auto, eta_joint, 
                              eps_to_obs, h_to_obs, eps_auto,  h_auto, d_safe_obs, d_safe_auto, d_safe_jl,eps_reg)
    
    @staticmethod
    def vector_field_SE3(
        state: HTMatrix,
        curve: ArrayLike | List[HTMatrix],
        kt1: float = 1.0,
        kt2: float = 1.0,
        kt3: float = 1.0,
        kn1: float = 1.0,
        kn2: float = 1.0,
        curve_derivative: ArrayLike | List[HTMatrix] = [],
        delta: float = _DELTA_SE3,
        ds: float = _DS_SE3,
        mode: str = 'auto'
        ) -> Tuple[NDArray, float, int]:
        r"""Computes the vector field in SE(3) presented in the thesis
        'Felipe B. A. Pessoa, Luciano C. A. Pimenta, Vinicius M. Goncalves: 
        Constructive Vector Fields for Path Following in Matrix Lie Groups,
        Master's Thesis, Universidade Federal de Minas Gerais, 2025'. 
    
        Parameters
        ----------
        state: 4x4 numpy array
            The current state of the robot, as a homogeneous transformation
            matrix.
        curve : nx4x4 numpy array or nx4x4 nested list
            Curve, described as sampled points. Each one of the n rows should 
            contain a 4x4 homogeneous transformation matrix, written in world
            coordinates.
        kt1 : float
            A gain parameter for the tangent component of the vector field.
            (default: 1.0).
        kt2 : float
            A gain parameter for the normal component of the vector field.
            (default: 1.0).
        kt3 : float
            A gain parameter for the binormal component of the vector field.
            (default: 1.0).
        kn1 : float
            A gain parameter for the normal component of the vector field.
            (default: 1.0).
        kn2 : float
            A gain parameter for the binormal component of the vector field.
            (default: 1.0).
        curve_derivative : nx4x4 numpy array or nx4x4 nested list
            The derivative of the curve, as a list of 4x4 homogeneous
            transformation matrices, written in world coordinates.
            If not provided, it is computed numerically.
            (default: empty list).
        delta : float
            Parameter that controls the precision of the normal compoennt
            approximation.
            (default: 1e-3).
        ds : float
            Parameter that controls the precision of the tangent component
            approximation. Ignored if curve_derivative is provided.
            (default: 1e-3).
        mode : string
            'c++' for the c++ implementation, 'python' for the python 
            implementation and 'auto' for automatic ('c++' is available, else 
            'python').
            (default: 'auto').

        Returns
        -------
        qdot : n x 1 numpy array
            The twist generated by the vector field.
        dist : float
            Distance to the curve.
        index : integer
            The index of the closest point to the curve.

        Notes
        -----
        The vector field (:math:`\Psi`) is defined as a combination of tangent
        (:math:`\xi_T`) and normal (:math:`\xi_N`) components [1]_:
        
        .. math:: \Psi = k_T \xi_T + k_N \xi_N,

        where :math:`k_T` and :math:`k_N` are gain parameters for the tangent
        and normal components, respectively. The normal component is computed
        to drive the following distance to zero

        .. math:: D(H) = \min_s \|\log(H^{-1} H_d(s))\|_F,

        where :math:`H` is the current state, :math:`H_d(s)` is a point on the
        curve `curve`.

        The gains are then defined as:

        .. math:: 
            k_N = k_{n1} \tanh(k_{n2} \sqrt{D(H)})\\
            k_T = k_{t1} (1 - k_{t2} \tanh(k_{t3} D(H))),

        where :math:`k_{n1}`, :math:`k_{n2}`, :math:`k_{t1}`, :math:`k_{t2}`,
        and :math:`k_{t3}` are the function parameters `kt1`, `kt2`, `kt3`,
        `kn1`, and `kn2`, respectively.

        References
        ----------
        .. [1] Felipe B. A. Pessoa, Luciano C. A. Pimenta, Vinicius M. Goncalves,
           "Constructive Vector Fields for Path Following in Matrix Lie Groups",
           Master's Thesis, Universidade Federal de Minas Gerais, 2025.
        """
        return _vector_field_SE3(state, curve, kt1, kt2, kt3, kn1, kn2, curve_derivative, delta, ds, mode)
        
        
    #######################################
    # Methods for simulation
    #######################################

    def gen_code(self, port):
        """Generate code for injection."""
        return _gen_code(self, port)

    def update_col_object(self, time: float, mode: str ='auto') -> None:
        """
        Update internally the objects that compose the collision model to the
        current configuration of the robot.
        """
        _update_col_object(self, time, mode)

    def add_col_object(self, sim: "Simulation") -> None:
        """
        Add the objects that compose the collision model to a simulation.

        Parameters
        ----------
        sim : 'Simulation' object
            'Simulation' object.
    """
        _add_col_object(self, sim)

    def attach_object(self, obj: GroupableObject) -> None:
        """
        Attach an object to the end-effector.
        The list of the type of objects that can be grouped can be seen in 'Utils.IS_GROUPABLE'.

        Parameters
        ----------
        obj : Object that is one of the types in 'Utils.IS_GROUPABLE'
            Object to be attached.
    """
        _attach_object(self, obj)

    def detach_object(self, obj: GroupableObject) -> None:
        """
        Detach an object (ball, box or cylinder) to the end-effector.

        Parameters
        ----------
        obj : object
            Object to be detached.
    """
        _detach_object(self, obj)

    def set_htm_to_eef(self, htm: HTMatrix) -> None:
        self._htm_n_eef = htm

    #######################################
    # Robot constructors
    #######################################

    @staticmethod
    def create_kuka_kr5(htm: HTMatrix = np.identity(4), name: str ='', color: str ="#df6c25", 
                        opacity: float =1, eef_frame_visible: bool=True) -> "Robot":
        """
    Create a Kuka KR5 R850, a six-degree of freedom manipulator.
    Thanks Sugi-Tjiu for the 3d model (see https://grabcad.com/library/kuka-kr-5-r850).

    Parameters
    ----------
    htm : 4x4 numpy matrix
        The initial base configuration for the robot.
        (default: np.identity(4))
 
    name : string
        The robot name.
        (default: empty (automatic)).

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
    def create_epson_t6(htm: HTMatrix = np.identity(4), name: str ='', color: str ="white", 
                        opacity: float =1, eef_frame_visible: bool = True)-> "Robot":
        """
    Create an Epson T6, a SCARA manipulator.
    Thanks KUA for the 3d model (see https://grabcad.com/library/epson-t6-scara-robot-1).

    Parameters
    ----------
    htm : 4x4 numpy matrix
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: empty (automatic)).

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
    def create_staubli_tx60(htm: HTMatrix = np.identity(4), name: str = '', color: str ="#ff9b00", 
                            opacity: float =1, eef_frame_visible: bool = True) -> "Robot":
        """
    Create a Staubli TX60, a six degree of freedom manipulator.
    Model taken from the ROS github repository (https://github.com/ros-industrial/staubli).

    Parameters
    ----------
    htm : 4x4 numpy matrix
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: empty (automatic)).

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
    def create_kuka_lbr_iiwa(htm: HTMatrix = np.identity(4), name: str = '', color: str = "", 
                             opacity: float =1, eef_frame_visible: bool = True) -> "Robot":
        """
    Create a Kuka LBR IIWA 14 R820, a seven degree of freedom manipulator.
    Model taken from the ROS github repository (https://github.com/ros-industrial/kuka_experimental).

    Parameters
    ----------
    htm : 4x4 numpy matrix
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: empty (automatic)).

    htm : color
        A HTML-compatible string representing the object color. 
        If set to '' (empty string), set colors from textures.
        (default: '')'.

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
    def create_franka_emika_3(htm: HTMatrix = np.identity(4), name: str = '', color : str = "", 
                              opacity: float = 1, eef_frame_visible: bool = True) -> "Robot":
        """
    Create a Franka Emika 3, a seven degree of freedom manipulator.
    Model taken from the ROS github repository (https://github.com/BolunDai0216/FR3Env/tree/d5218531471cadafd395428f8c2033f6feeb3555/FR3Env/robots/meshes/visual).

    Parameters
    ----------
    htm : 4x4 numpy matrix
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: empty (automatic)).

    htm : color
        A HTML-compatible string representing the object color. 
        If set to '' (empty string), set colors from textures.
        (default: '')'.

    opacity : positive float between 0 and 1
        The opacity of the robot. 1 = fully opaque and 0 = transparent.
        (default: 1)

    Returns
    -------
    robot : Robot object
        The robot.

    """
        base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits = _create_franka_emika_3(htm, name, color, opacity)
        return Robot(name, links, base_3d_obj, htm, htm_base_0, htm_n_eef, q0, eef_frame_visible, joint_limits)

    @staticmethod
    def create_abb_crb(htm : HTMatrix = np.identity(4), name: str = '', color: str = "white", 
                       opacity: float = 1, eef_frame_visible: bool = True) -> "Robot":
        """
    Create a ABB CRB 15000, a six degree of freedom manipulator.
    Model taken from the ROS github repository (https://github.com/ros-industrial/abb_experimental).

    Parameters
    ----------
    htm : 4x4 numpy matrix
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: empty (automatic)).

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
    def create_magician_e6(htm: HTMatrix = np.identity(4), name: str = "", color: str = "", 
                           opacity: float = 1, eef_frame_visible: bool = True) -> "Robot":
        """
    Create a DOBOT Magician E6, a six degree of freedom manipulator.
    Model taken from the ROS github repository (https://github.com/Dobot-Arm/TCP-IP-ROS-6AXis).

    Parameters
    ----------
    htm : 4x4 numpy matrix
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: empty (automatic)).

    htm : color
        A HTML-compatible string representing the object color.
        (default: '')'.

    opacity : positive float between 0 and 1
        The opacity of the robot. 1 = fully opaque and 0 = transparent.
        (default: 1)

    Returns
    -------
    robot : Robot object
        The robot.

    """
        base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits = _create_magician_e6(htm, name, color, opacity)
        return Robot(name, links, base_3d_obj, htm, htm_base_0, htm_n_eef, q0, eef_frame_visible, joint_limits)
    
    @staticmethod
    def create_darwin_mini(htm: HTMatrix = np.identity(4), name: str = "", color: str = "#3e3f42", 
                           opacity: float = 1, eef_frame_visible: bool = True) -> GroupableObject:
        """
    Create an (oversized) Darwin Mini, a humanoid robot.
    Thanks to Alexandre Le Falher for the 3D model (https://grabcad.com/library/darwin-mini-1).

    Parameters
    ----------
    htm : 4x4 numpy matrix
        The initial base configuration for the robot.
        (default: np.identity(4))

    name : string
        The robot name.
        (default: empty (automatic)).

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
    def create_davinci(htm: HTMatrix = np.identity(4), name: str = "", color: str = "#3e3f42", 
                       opacity: float = 1, eef_frame_visible: bool = True) -> GroupableObject:
        """
        Create a da Vinci Si, a surgical robot.
        Thanks to Koray Okan for the 3D model (https://grabcad.com/library/da-vinci-surgical-robot-1/details).
        Created by Felipe Bartelt.
        Parameters
        ----------
        htm : 4x4 numpy matrix
            The initial base configuration for the robot.
            (default: np.identity(4))
        name : string
            The robot name.
            (default: empty (automatic)).
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

        return _create_davinci(htm, name, color, opacity, eef_frame_visible)
    
    @staticmethod
    def create_kinova_gen3(
        htm: HTMatrix = np.eye(4),
        name: str = "",
        color: str = "#e6e1e1",
        opacity: float = 1.0,
        eef_frame_visible: bool = True
        ) -> "Robot":
        """Create a Kinova Gen3-7DoF, a seven degree of freedom 
        manipulator.

        Model from Kinova Resources (https://www.kinovarobotics.com/resources
        Gen3 CAD model (7DoF)). Manipulator parameters taken from the official 
        Kinova Gen3 documentation (GEN3 User Guide), available at
        https://www.kinovarobotics.com/resources.
        There is no gripper in this model, because it is not available.

        Parameters
        ----------
        htm : 4x4 numpy array or 4x4 nested list
            The initial base configuration for the robot.
            (default: np.identity(4))
        name : string
            The robot name.
            (default: empty (automatic)).
        color: string
            A HTML-compatible string representing the object color.
            (default: '#e6e1e1').
        opacity : positive float between 0 and 1
            The opacity of the robot. 1 = fully opaque and 0 = transparent.
            (default: 1)

        Returns
        -------
        robot : Robot object
            The robot.
        """

        return _create_kinova_gen3(htm, name, color, opacity, eef_frame_visible)
    
    @staticmethod
    def create_jaco(
        htm: HTMatrix = np.eye(4),
        name: str = "",
        color: str | list | None = None,
        opacity: float = 1.0,
        eef_frame_visible: bool = True
        ) -> "Robot":
        """Create a Kinova Jaco 2, a six degree of freedom manipulator.
        Model taken from Kinova resources 
        (https://www.kinovarobotics.com/resources?r=79302&s).
        Robot documentation available at 
        https://www.kinovarobotics.com/resources?r=339.
        The gripper is not yet controllable.

        Parameters
        ----------
        htm : 4x4 numpy array or 4x4 nested list
            The initial base configuration for the robot.
            (default: np.identity(4))
        name : string
            The robot name.
            (default: empty (automatic)).
        color : string or list or None
            A HTML-compatible string representing the object color or a list of
            HTML-compatible strings representing the object color. The list 
            should have three elements, representing the color of the links, 
            plastic rings, and the nails, respectively. If less than three 
            elements are provided, the remaining elements will be set to the
            last element of the list. If a single string is provided, it will
            be used for all three elements.
            If set to None, the default color scheme is used, which is a list
            of three strings: ["#3e3f42", "#919090", "#1d1d1f"].
        opacity : positive float between 0 and 1
            The opacity of the robot. 1 = fully opaque and 0 = transparent.
            (default: 1)

        Returns
        -------
        robot : Robot object
            The robot.
        """

        return _create_jaco(htm, name, color, opacity, eef_frame_visible)
    
    @staticmethod
    def create_ur_ur3e(
        htm: HTMatrix = np.eye(4),
        name: str = "",
        color: str | list | None = 'white',
        opacity: float = 1.0,
        eef_frame_visible: bool = True
        ) -> "Robot":

        base_3d_obj, links, htm_base_0, htm_n_eef, q0, joint_limits = _create_ur_ur3e(htm, name, color, opacity)
        return Robot(name, links, base_3d_obj, htm, htm_base_0, htm_n_eef, q0, eef_frame_visible, joint_limits)

    
    
    
    #######################################
    # Distance computation and collision
    #######################################

    def compute_dist(self, obj: MetricObject, q: Optional[Vector] = None, htm: Optional[HTMatrix]=None, 
                     old_dist_struct : Optional["DistStructRobotObj"] = None, tol: float = 0.0005, 
                     no_iter_max: int = 20, max_dist: float = np.inf, h: float = 0, eps: float = 0, 
                     mode: str = 'auto') -> "DistStructRobotObj":
        """
    Compute the  distance structure from each one of the robot's link to a
    'simple' external object (see Utils.IS_SIMPLE), given a joint and base
    configuration.

    This function can be faster if some distance computations are avoided.
    See the description of the parameter 'max_dist'.

    The distance is either Euclidean distance or differentiable.
    
    If h>0 or eps > 0, it computes the Euclidean distance and it uses GJK's algorithm.
    
    Else, it computes the differentiable distance through Generalized Alternating Projection (GAP).
    See the paper 'A Differentiable Distance Metric for Robotics Through Generalized Alternating Projection'.
    This only works in c++ mode, though.

    Parameters
    ----------
    obj : a simple object (see Utils.IS_SIMPLE)
        The external object for which the distance structure is going to be 
        computed, for each robot link.

    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The manipulator's joint configuration.
        (default: the current  joint configuration (robot.q) for the manipulator).

    htm : 4x4 numpy matrix
        The robot base's configuration.
        (default: the same as the current htm).

    old_dist_struct : 'DistStructRobotObj' object
        'DistStructRobotObj' obtained previously for the same robot and external object.
        Can be used to enhance the algorithm speed using the previous closest
        point as an initial guess.
        (default: None).

    tol : positive float
        Convergence criterion of GAP: it stops when ||a[k+1]-a[k]|| < tol.
        Only valid when h > 0 or eps > 0
        (default: 0.0005 m)      

    no_iter_max : positive int 
        Maximum number of iterations of GAP.
        Only valid when h > 0 or eps > 0
        (default: 20 iterations) 

    max_dist: positive float
        The algorithm uses an axis aligned bounding box (aabb) to avoid some distance computations.
        The algorithm skips a more precise distance computation if the distance between the aabb
        of the primitive objects composing the link and 'obj' is less than 'max_dist' (in meters).
        (default: infinite).

    h : nonnegative float
        h parameter in the generalized distance function.
        If h=0 and eps=0, it is simply the Euclidean distance.
        (default: 0) 

    eps : nonnegative float
        h parameter in the generalized distance function.
        If h=0 and eps=0, it is simply the Euclidean distance.
        (default: 0) 
                
    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto').

    Returns
    -------
    dist_struct : 'DistStructRobotObj' object
        Distance struct for each one of the m objects that compose the robot's
        collision model. Contains a list of m 'DistStructLinkObj' objects.
    """

        return _compute_dist(self, obj, q, htm, old_dist_struct, tol, no_iter_max, max_dist, h, eps, mode)

    def compute_dist_auto(self, q: Optional[Vector] = None, old_dist_struct: Optional["DistStructRobotAuto"]=None, 
                          tol: float =0.0005, no_iter_max: int = 20, max_dist: float = np.inf, 
                          h: float = 0, eps: float = 0, mode: str = 'auto') -> "DistStructRobotAuto":
        """
    Compute the  distance structure from each one of the robot's links to itself
    (auto collision), given a joint and base configuration.

    This function consider only non-sequential links, since the collision between
    sequential links in the kinematic chain can be verified by checking joint limits.
    This saves times. This verification should be done elsewhere (by checking if the
    configuration is inside the joint limits).

    This function can be faster if some distance computations are avoided.
    See the description of the parameter 'max_dist'.

    The distance is either Euclidean distance or differentiable.
    
    If h>0 or eps > 0, it computes the Euclidean distance and it uses GJK's algorithm.
    
    Else, it computes the differentiable distance through Generalized Alternating Projection (GAP).
    See the paper 'A Differentiable Distance Metric for Robotics Through Generalized Alternating Projection'.
    This only works in c++ mode, though.

    Parameters
    ----------
    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The manipulator's joint configuration.
        (default: the current  joint configuration (robot.q) for the manipulator).

    old_dist_struct : 'DistStructRobotAuto' object
        'DistStructRobotAuto' obtained previously for the same robot.
        Can be used to enhance the algorithm speed using the previous closest
        point as an initial guess.
        (default: None).

    tol : positive float
        Convergence criterion of GAP: it stops when ||a[k+1]-a[k]|| < tol.
        Only valid when h > 0 or eps > 0.
        (default: 0.0005 m)      

    no_iter_max : positive int 
        Maximum number of iterations of GAP.
        Only valid when h > 0 or eps > 0
        (default: 20 iterations) 

    max_dist: positive float
        The algorithm uses an axis aligned bounding box (aabb) to avoid some distance computations.
        The algorithm skips a more precise distance computation if the distance between the aabb
        of the primitive objects composing the link and 'obj' is less than 'max_dist' (in meters).
        (default: infinite).

    h : nonnegative float
        h parameter in the generalized distance function.
        If h=0 and eps=0, it is simply the Euclidean distance.
        (default: 0) 

    eps : nonnegative float
        h parameter in the generalized distance function.
        If h=0 and eps=0, it is simply the Euclidean distance.
        (default: 0) 
        
    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto').
        
    Returns
    -------
    dist_struct : 'DistStructRobotAuto' object
        Distance struct for each one of the m objects that compose the robot's
        collision model. Contains a list of m 'DistStructLinkLink' objects.
    """

        return _compute_dist_auto(self, q, old_dist_struct, tol, no_iter_max, max_dist, h, eps, mode)

    def check_free_config(self, q: Optional[Vector]=None, htm: Optional[HTMatrix]=None, 
                          obstacles: List[MetricObject]=[], check_joint: bool = True, check_auto: bool = True,
                          tol: float = 0.0005, dist_tol: float = 0.005, no_iter_max: int = 20, 
                          mode: str = 'auto') -> Tuple[bool,str,List]:
        """
    Check if the joint configuration q is in the free configuration space, considering
    joint limits, collision with obstacles and auto collision. It also outputs a message about a
    possible violation.

    For efficiency purposes, the program halts in the first violation it finds (if there is any).
    So, the message, if any, is only about one of the possible violations. There can be more.

    Parameters
    ----------
    q : a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        The manipulator's joint configuration.
        (default: the current joint configuration (robot.q) for the manipulator).

    obstacles : list of simple objects (see Utils.IS_SIMPLE)
        A list of obstacles as simple objects 
        (default: empty list).

    htm : 4x4 numpy matrix
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

    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto').

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

        return _check_free_config(self, q, htm, obstacles, check_joint, check_auto, tol, dist_tol, no_iter_max, mode)

    #######################################################################################################   
    #LEGACY
    def check_free_configuration(self, q: Optional[Vector]=None, htm: Optional[HTMatrix]=None, 
                          obstacles: List[MetricObject]=[], check_joint: bool = True, check_auto: bool = True,
                          tol: float = 0.0005, dist_tol: float = 0.005, no_iter_max: int = 20, 
                          mode: str = 'auto') -> Tuple[bool,str,List]:
      
      
        # Backward compatibility shim
        import warnings
        warnings.warn(
            "'check_free_configuration' is deprecated and will be removed soon, use 'check_free_config' instead.",
            DeprecationWarning
        )
              
        return _check_free_config(self, q, htm, obstacles,
                              check_joint, check_auto,
                              tol, dist_tol, no_iter_max, mode)
     
    #######################################################################################################
