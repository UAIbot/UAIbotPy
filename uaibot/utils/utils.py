from math import *
import numpy as np
from colour import Color
import httplib2
import string as st
from scipy.linalg import null_space
from httplib2 import *
import sys
import quadprog
from .types import *
from typing import Callable
import urllib.request
import io
import hashlib
import os
if os.environ['CPP_SO_FOUND']=="1":
    import uaibot_cpp_bind as ub_cpp
    

class Utils:
    """A library that contains some utilities for UAIbot. All of the functions are static."""

    #######################################
    # Constants
    #######################################

    UAIBOT_NAME_TYPES = ['uaibot.', 'cylinder.', 'box.', 'ball.', 'convexpolytope.', 'robot.', 'simulation.', 'meshmaterial.', 'mtlmeshmaterial.',
                             'glbmeshmaterial.', 'texture.', 'pointlight.', 'frame.', 'model3d.', 'links.', 'pointcloud.', 'arrow.', 'rigidobject.',
                             '.group', '.htmldiv', 'CPP_GeometricPrimitives', 'CPP_DistStructRobotObj','CPP_AABB', '.Pedestrian', '.ObstacleColumn', '.ObstacleThinWall']

    IS_SIMPLE = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.ConvexPolytope']
    
    IS_METRIC = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.PointCloud', 'uaibot.ConvexPolytope']

    IS_GROUPABLE = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.ConvexPolytope', 'uaibot.Frame',
                    'uaibot.RigidObject', 'uaibot.Group', 'uaibot.Robot', 'uaibot.PointLight']

    IS_OBJ_SIM = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.ConvexPolytope', 'uaibot.Robot',
                  'uaibot.PointLight', 'uaibot.Frame', 'uaibot.PointCloud', 'uaibot.Arrow',
                  'uaibot.RigidObject', 'uaibot.Group', 'uaibot.HTMLDiv', 'uaibot.Pedestrian', 'uaibot.ObstacleColumn', 'uaibot.ObstacleThinWall']

    #######################################
    # Basic functions
    #######################################

    @staticmethod
    def cvt(input: Matrix) -> np.matrix:
        if isinstance(input, list) or isinstance(input, tuple):
            if all(isinstance(x, (int, float)) for x in input):
                return np.matrix(input, dtype=np.float64).reshape(-1, 1)
            elif all(isinstance(x, list) for x in input):
                return np.matrix(input, dtype=np.float64)
            else:
                raise ValueError("Entry is not a valid matrix.")
        
        elif isinstance(input, np.ndarray):
            if input.ndim == 1:
                return np.matrix(input.reshape(-1, 1), dtype=np.float64)
            else:
                if input.shape[1] > 1:
                    if input.shape[0]==1:
                        return np.matrix(input, dtype=np.float64).reshape(-1, 1)
                    else:
                        return np.matrix(input, dtype=np.float64)
                else:
                    return np.matrix(input, dtype=np.float64)                    
        else:
            raise TypeError("Entry is not a valid matrix.")                  

        

    @staticmethod
    def S(v: Vector) -> np.matrix:
        """
      Returns a 3x3 matrix that implements the cross product for a 3D vector  
      as a matricial product, that is, a matrix S(v) such that for any other 
      3D column  vector w, S(v)w = cross(v,w).
      
      Parameters
      ----------
      v : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array)
          The vector for which the S matrix will be created.

      Returns
      -------
      S : 3x3 numpy matrix
          A matrix that implements the cross product with v.
      """
        vv = Utils.cvt(v)
        return np.matrix([[0, -vv[2,0], vv[1,0]],
                         [vv[2,0], 0, -vv[0,0]],
                         [-vv[1,0], vv[0,0], 0]])

    @staticmethod
    def rot(axis: Vector, angle: float) -> HTMatrix:
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in an axis.
      
      Parameters
      ----------
      axis : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array)
          The axis of rotation.
      
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 numpy matrix
          The homogeneous transformation matrix.
      """
        a = Utils.cvt(axis)
        a = a / np.linalg.norm(a)
        K = Utils.S(a)
        Q = np.identity(3) + sin(angle) * K + (1 - cos(angle)) * (K * K)
        return np.hstack([np.vstack([Q, np.matrix([0, 0, 0])]), np.matrix([[0], [0], [0], [1]])])

    @staticmethod
    def trn(vector: Vector) -> HTMatrix:
        """
      Homogeneous transformation matrix that represents the displacement
      of a vector
      
      Parameters
      ----------
      vector : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array)
          The displacement vector.
      
      Returns
      -------
      htm : 4x4 numpy matrix
          The homogeneous transformation matrix.
      """
        v = Utils.cvt(vector)
        return np.matrix([[1, 0, 0, v[0,0]],
                         [0, 1, 0, v[1,0]],
                         [0, 0, 1, v[2,0]],
                         [0, 0, 0, 1]])

    @staticmethod
    def rotx(angle: float) -> HTMatrix:
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in the 'x' axis.
      
      Parameters
      ----------
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 numpy matrix
          The homogeneous transformation matrix.
      """
        return np.matrix([[1, 0, 0, 0],
                         [0, cos(angle), -sin(angle), 0],
                         [0, sin(angle), cos(angle), 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def roty(angle: float) -> HTMatrix:
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in the 'y' axis.
      
      Parameters
      ----------
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 numpy matrix
          The homogeneous transformation matrix.
      """
        return np.matrix([[cos(angle), 0, sin(angle), 0],
                         [0, 1, 0, 0],
                         [-sin(angle), 0, cos(angle), 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def rotz(angle: float) -> HTMatrix:
        """
      Homogeneous transformation matrix that represents the rotation of an
      angle in the 'z' axis.
      
      Parameters
      ----------
      angle: float
          The angle of rotation, in radians.

      Returns
      -------
      htm : 4x4 numpy matrix
          The homogeneous transformation matrix.
      """
        return np.matrix([[cos(angle), -sin(angle), 0, 0],
                         [sin(angle), cos(angle), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def htm_rand(trn_min: float =[0.,0.,0.], trn_max: float = [1.,1.,1.], rot: float =np.pi/2) -> HTMatrix:
        """
      Returns a random homogeneous transformation matrix.

      Parameters
      ----------
      trn: float
          Maximum parameter for random translation in x, y and z.
          (default: 1)

      rot: float
          Maximum parameter for random rotation in x, y and z.
          (default: 1)

      Returns
      -------
      htm : 4x4 numpy matrix
          A homogeneous transformation matrix.
      """

        x = np.random.uniform(trn_min[0],trn_max[0])
        y = np.random.uniform(trn_min[1],trn_max[1])
        z = np.random.uniform(trn_min[2],trn_max[2])
        ax = np.random.uniform(-rot / 2, rot / 2)
        ay = np.random.uniform(-rot / 2, rot / 2)
        az = np.random.uniform(-rot / 2, rot / 2)

        return Utils.trn([x,y,z]) * Utils.rotx(ax) * Utils.roty(ay) * Utils.rotz(az)

    @staticmethod
    def inv_htm(htm: HTMatrix) -> HTMatrix:
        """
      Given a homogeneous transformation matrix, compute its inverse.
      It is faster than using numpy.linalg.inv in the case of HTMs.
      
      Parameters
      ----------
      htm: 4X4 numpy matrix
          Homogeneous transformation matrix of the rotation.

      Returns
      -------
      inv_htm: 4X4 numpy array
          The inverse of the transformation matrix.       
      """
        Q = htm[0:3, 0:3]
        p = htm[0:3, 3]

        inv_htm = np.matrix(np.zeros((4, 4)))
        inv_htm[0:3, 0:3] = Q.T
        inv_htm[0:3, 3] = -Q.T * p
        inv_htm[3, 3] = 1

        return inv_htm

    @staticmethod
    def axis_angle(htm: Matrix) -> Tuple[np.matrix, float]:
        """
      Given an homogeneous transformation matrix representing a rotation, 
      return the rotation axis angle.
      
      Parameters
      ----------
      htm: 4X4 numpy matrix
          Homogeneous transformation matrix of the rotation.

      Returns
      -------
      axis : 3D numpy vector
          The rotation axis.

      angle : float
          The rotation angle, in radians.        
      """
      
        Q = htm[0:3, 0:3]
        trace = Q[0, 0] + Q[1, 1] + Q[2, 2]
        angle = acos((trace - 1) / 2)
        G = Q * Q - 2 * cos(angle) * Q + np.identity(3)
        ok = False
        while not ok:
            v = np.matrix(np.random.uniform(-100, 100, size=(3,1)))
            w = np.matrix(np.random.uniform(-100, 100, size=(3,1)))
            r = G * v
            nr = np.linalg.norm(r)
            prod = w.T * r
            if nr > 0.01:
                ortr = w -  (r * prod) / (nr * nr)
                axis = Utils.S(ortr) * (Q * ortr)
                naxis = np.linalg.norm(axis)
                ok = naxis > 0.01

        axis = axis / naxis
        return axis, angle

    @staticmethod
    def euler_angles(htm: Matrix) -> Tuple[float, float, float]:
        """
        Computer the Euler angles of a rotation matrix.
        Find alpha, beta and gamma such that.

        htm = Utils.rotz(alpha) * Utils.roty(beta) * Utils.rotx(gamma).

        Parameters
        ----------
        htm: 4X4 numpy array or nested list
            Homogeneous transformation matrix of the rotation.

        Returns
        -------
        alpha : float
            Rotation in z, in radians.
        beta : float
            Rotation in y, in radians.
        gamma : float
            Rotation in x, in radians.
        """

        Q = np.matrix(htm[0:3, 0:3])
        sy = sqrt(Q[0, 0] ** 2 + Q[1, 0] ** 2)

        if abs(sy) > 0.001:
            gamma = np.arctan2(Q[2, 1], Q[2, 2])
            beta = np.arctan2(-Q[2, 0], sy)
            alpha = np.arctan2(Q[1, 0], Q[0, 0])
        else:
            gamma = np.arctan2(-Q[1, 2], Q[1, 1])
            beta = np.arctan2(-Q[2, 0], sy)
            alpha = 0

        return alpha, beta, gamma


    @staticmethod
    def dp_inv(A: Matrix, eps: float = 0.001) -> np.matrix:
        """
      Compute the damped pseudoinverse of the matrix 'mat'.
      
      Parameters
      ----------
      A: a matrix ((n,m)-element list/tuple, (n,m)-shaped numpy matrix/numpy array)
          The matrix to compute the damped pseudoinverse.
      
      eps: positive float
          The damping factor.
          (default: 0.001).

      Returns
      -------
      pinvA: mxn numpy array
          The damped pseudoinverse of 'mat'.
      """
        A_int =  Utils.cvt(A)
        n = np.shape(A_int)[1]
        return np.linalg.inv(A_int.T * A_int + eps * np.identity(n)) * A_int.T

    @staticmethod
    def dp_inv_solve(A: Matrix, b: Vector, eps: float = 0.001, mode: str ='auto') -> np.matrix:
        """
      Solve the problem of minimizing ||A*x-b||^2 + eps*||x||^2
      It is the same as dp_inv(A,eps)*b, but faster
      
      Parameters
      ----------
      A: a matrix ((n,m)-element list/tuple, (n,m)-shaped numpy matrix/numpy array)
          A Matrix.
          
      b: a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
          b Vector.
                
      eps: positive float
          The damping factor.
          (default: 0.001).

      Returns
      -------
      x: mx1 numpy array
          The solution to the optimization problem.
      """
      
        if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")

        A_cvt = Utils.cvt(A)
        b_cvt = Utils.cvt(b)
        n, m = A_cvt.shape
        
        if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            
            M = np.block([
                [eps * np.eye(m), -A_cvt.T],
                [A_cvt, np.eye(n)]
            ])
            
            rhs = np.concatenate((np.zeros((m,1)), b_cvt))
            solution = np.linalg.solve(M, rhs)
            
            return Utils.cvt(solution[:m])
        else:
            return Utils.cvt(ub_cpp.dp_inv_solve(A_cvt,b_cvt,eps))
                 

    @staticmethod
    def interpolate(points: List[Vector], is_closed : bool =False) -> Callable[[List[float]], List[np.matrix]]:
        """
      Create a function handle that generates an one-time differentiable interpolated data from 'points'.

      The simplest case in when 'points' is a list with m elements. In this case, it will output a function f.
      When this function is evaluated at a scalar t, it will coincide with points[i] when t = i/m, that is,
      f(i/m) = points[i]. This function is once differentiable and periodic with period 1, so f(t+k)=f(t) for
      an integer k.

      The function can also use a n x m numpy array or lists as 'points'. In this case, f(t) is a n x 1 
      numpy vector.

      Finally, t can be a list of k elements instead of just a scalar. In this case, f(t) is a list of n x 1 numpy 
      matrices, in which the k-th element is the same as computing f(t[k]).


      Parameters
      ----------
      points: a list of nD vectors (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
          Points to be interpolated.
          
      is_closed: bool
          If the curve is closed or not (that is, f(0)=f(1)).
          (default: False)

      Returns
      -------
      f: function handle
          The function handle that implements the interpolation. 
      """

        if not Utils.is_a_list_vector(points):
            raise Exception("The parameter 'points' should be a list of vectors with the same dimension.")


        def aux_interpolate_single(arg_points):
            def int_aux_simp(arg_t, arg_n, arg_c):
                tn = arg_n * (arg_t % 1 if is_closed else min(max(arg_t, 0), 1))
                ti = min(floor(tn), arg_n - 1)
                tloc = tn - ti
                coef = arg_c[4 * ti:4 * ti + 4]
                return coef[0] + coef[1] * tloc + coef[2] * tloc**2 + coef[3] * tloc**3

            def int_aux(arg_t, arg_n, arg_c):
                return [int_aux_simp(tt, arg_n, arg_c) for tt in arg_t] if isinstance(arg_t, list) else int_aux_simp(arg_t, arg_n, arg_c)

            n = len(arg_points)
            num_segments = n if is_closed else n - 1
            A_rows = []
            b_vals = []

            # Interpolation constraints (value at start and end of each segment)
            for i in range(num_segments):
                row_start = np.zeros(4 * num_segments)
                row_end = np.zeros(4 * num_segments)
                row_start[4 * i:4 * i + 4] = [1, 0, 0, 0]
                row_end[4 * i:4 * i + 4] = [1, 1, 1, 1]
                A_rows.append(row_start)
                b_vals.append(arg_points[i])
                A_rows.append(row_end)
                b_vals.append(arg_points[(i + 1) % n if is_closed else i + 1])

            # Derivative continuity at junctions (including wraparound if closed)
            for i in range(num_segments):
                i_next = (i + 1) % num_segments
                if not is_closed and i == num_segments - 1:
                    break  # skip final derivative constraint for open curve
                row = np.zeros(4 * num_segments)
                row[4 * i + 1:4 * i + 4] = [1, 2, 3]
                row[4 * i_next + 1:4 * i_next + 4] = [-1, 0, 0]
                A_rows.append(row)
                b_vals.append(0)

            # Assemble matrices
            A = np.vstack(A_rows)
            b = np.array(b_vals).reshape(-1, 1)

            # Regularizer: minimize integral of second derivative squared
            H = np.zeros((4 * num_segments, 4 * num_segments))
            for i in range(num_segments):
                H_i = np.array([
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 4, 6],
                    [0, 0, 6, 12]
                ])
                H[4 * i:4 * i + 4, 4 * i:4 * i + 4] = H_i

            # Solve constrained minimization via KKT system
            m = A.shape[0]
            KKT = np.block([
                [H, A.T],
                [A, np.zeros((m, m))]
            ])
            rhs = np.vstack([np.zeros((4 * num_segments, 1)), b])
            sol = np.linalg.solve(KKT, rhs)

            c = sol[:4 * num_segments].flatten().tolist()
            return lambda s: int_aux(s, num_segments, c)


        def aux_interpolate_multiple(arg_points, t):

            if not (Utils.is_a_vector(t) or Utils.is_a_number(t)):
                raise Exception(
                    "The parameter of the interpolation function must be either a number or a list of numbers.")

            y = np.zeros((0, len(t) if Utils.is_a_vector(t) else 1))
            for i in range(np.shape(Utils.cvt(arg_points[0]))[0]):
                fun = aux_interpolate_single([Utils.cvt(p)[i,0] for p in arg_points])
                fun_out = fun(t)
                fun_out = np.array(fun_out).reshape((1, len(fun_out) if Utils.is_a_vector(fun_out) else 1))
                y = np.block([[y], [fun_out]])

            list_of_points = [np.matrix(y[:, i].reshape(-1, 1)) for i in range(y.shape[1])]
            if isinstance(t, list):
                return list_of_points
            else:
                return list_of_points[0]

        return lambda t: aux_interpolate_multiple(points, t)

    @staticmethod
    def solve_qp(H: Matrix, f: Vector, A: Optional[Matrix] = None, 
                 b: Optional[Vector] = None, A_eq: Optional[Matrix] = None, 
                 b_eq: Optional[Vector] = None) -> np.matrix:
        """
        Solve the convex quadratic optimization problem:
        min_u 0.5 * u'Hu + f'u such that A_eq * u = b_eq and A * u >= b.
        
        Parameters
        ----------
        H : (n,n)-shaped matrix (list/tuple/np.matrix/np.ndarray)
            The H matrix. Must be symmetric positive definite.

        f : (n,)-shaped vector (list/tuple/np.matrix/np.ndarray)
            The f vector.

        A : (m,n)-shaped matrix or 'None', optional
            Inequality constraint matrix (Au >= b). Pass 'None' to omit.

        b : (m,)-shaped vector or 'None', optional
            RHS of inequality constraints. Pass 'None' to omit.

        A_eq : (p,n)-shaped matrix or 'None', optional
            Equality constraint matrix (A_eq * u = b_eq). Pass 'None' to omit.

        b_eq : (p,)-shaped vector or 'None', optional
            RHS of equality constraints. Pass 'None' to omit.
        
        Returns
        -------
        u : (n,1) numpy matrix
            The solution.
        """
        
        if (A is None) and (A_eq is None):
            return -np.linalg.inv(Utils.cvt(H))*(Utils.cvt(f))
        
        H_cvt = np.matrix(H, dtype=np.float64)
        H_cvt = 0.5 * (H_cvt + H_cvt.T)  # Ensure symmetry
        f_cvt = np.array(np.asarray(f).reshape((-1,)), dtype=np.float64)

        # Handle empty A and b
        if (A is None) or (b is None):
            A_ineq = np.empty((0, H_cvt.shape[0]))
            b_ineq = np.empty((0,))
        else:
            A_ineq = np.matrix(A, dtype=np.float64)
            b_ineq = np.array(np.asarray(b).reshape((-1,)), dtype=np.float64)

        # Handle empty A_eq and b_eq
        if (A_eq is None) or (b_eq is None):
            A_eq_cvt = np.empty((0, H_cvt.shape[0]))
            b_eq_cvt = np.empty((0,))
        else:
            A_eq_cvt = np.matrix(A_eq, dtype=np.float64)
            b_eq_cvt = np.array(np.asarray(b_eq).reshape((-1,)), dtype=np.float64)

        # Stack constraints and compute meq
        G = np.vstack([A_eq_cvt, A_ineq])  
        h = np.hstack([b_eq_cvt, b_ineq])
        meq = A_eq_cvt.shape[0]

        result = quadprog.solve_qp(H_cvt, -f_cvt, G.T, h, meq)
        return Utils.cvt(result[0])

    @staticmethod
    def null_space(A: Matrix) -> np.matrix:
        """
        Compute the null space of the matrix A.

        Parameters:
        ----------
        A : a matrix ((n,m)-element list/tuple, (n,m)-shaped numpy matrix/numpy array)
            The matrix to compute the null space.

        Returns
        -------
        A_null : m x r numpy matrix
            The matrix such that its column spam the null space.
        """
        return np.matrix(null_space(Utils.cvt(A)))
    
    @staticmethod
    def get_fishbotics_mp_problems() -> dict:
        """
        Gets a set of many motion planning problems from the 'FishBotics' dataset.
        They were designed for the Franka Emika Panda robot.
        See: https://github.com/fishbotics/robometrics
        
        
        It is a dictionary in which each key is the name of the problem.
        If 'all_problems' is the output of this function, use
         list(all_problems.keys()) to get the keys (the problem names).
        
        Each problem is again a dictionary, which contains the following keys:
        
        'all_obs': a list of UAIBot objects representing the obstacles.
        'q0': the initial configuration.
        'htm_base': the initial pose for the base.
        'htm_tg': the target HTML for the end-effector.
        

        Parameters:
        ----------
        None.

        Returns
        -------
        all_problems : dictionary
            The dictionary with all the problems.
        """          
        
        
        def load_npz_from_url_cached(url, cache_dir=".cache"):
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create a unique filename based on the URL hash
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            local_path = os.path.join(cache_dir, f"{url_hash}.npz")

            # If cached, load from disk
            if os.path.exists(local_path):
                return np.load(local_path, allow_pickle=True)

            # Else, download and save
            with urllib.request.urlopen(url) as response:
                data = response.read()
            with open(local_path, 'wb') as f:
                f.write(data)
            
            return np.load(local_path, allow_pickle=True)


        allproblems_1 = load_npz_from_url_cached("https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/MotionPlanningProblems/fishbotics_mp_problems_part_1.npz")
        allproblems_2 = load_npz_from_url_cached("https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/MotionPlanningProblems/fishbotics_mp_problems_part_2.npz")
        allproblems_3 = load_npz_from_url_cached("https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/MotionPlanningProblems/fishbotics_mp_problems_part_3.npz")
        allproblems_4 = load_npz_from_url_cached("https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/MotionPlanningProblems/fishbotics_mp_problems_part_4.npz")
          
                        
        allproblems_1 = allproblems_1['arr_0'].item()
        allproblems_2 = allproblems_2['arr_0'].item()
        allproblems_3 = allproblems_3['arr_0'].item()
        allproblems_4 = allproblems_4['arr_0'].item()
        
        return {**allproblems_1, **allproblems_2, **allproblems_3, **allproblems_4}

    #######################################
    # Type check and conversion functions
    #######################################

    @staticmethod
    def is_a_number(obj):
        """
      Check if the argument is a float or int number
      
      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """

        return str(type(obj)) in ["<class 'int'>", "<class 'float'>", "<class 'numpy.float64'>"]

    @staticmethod
    def is_a_natural_number(obj):
        """
      Check if the argument is a natural number (integer and >=0)
      
      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """

        return str(type(obj)) == "<class 'int'>" and obj >= 0


        
    @staticmethod
    def is_a_matrix(obj, n=None, m=None):
        """
        Check if the argument is an n x m matrix of integers or floats.

        Parameters
        ----------
        obj: object
            Object to be verified.

        n: positive int
            Number of rows
            (default: it does not matter).

        m: positive int
            Number of columns
            (default: it does not matter).

        Returns
        -------
        is_type: boolean
            If the object is of the type.
        """

        # Check if obj is a numpy array or matrix
        if isinstance(obj, (np.ndarray, np.matrix)):
            # Check shape if dimensions are provided

            shp = np.shape(obj)
            rows = shp[0]
            cols = shp[1] if len(shp)>1 else  1

            if ((n is not None) and (rows != n)) or ((m is not None) and (cols !=m)):
                return False

            # Ensure all elements are integers or floats
            return np.issubdtype(obj.dtype, np.floating) or np.issubdtype(obj.dtype, np.integer)

        # Check if obj is a list (and therefore could represent a matrix)
        if isinstance(obj, list):
            try:
                return Utils.is_a_matrix(np.matrix(obj), n, m)
            except:
                return False
    
    @staticmethod
    def is_a_vector(obj, n=None):
        """
      Check if the argument is a vector of floats.
      A vector is very flexible. For example, a 3D vector (is_a_vector(x,3)) can be:
      x1 = [0,1,2]
      x2 = [[0],[1],[2]]
      x3 = (0,1,2)
      x4 = np.matrix([0,1,2])
      x5 = np.matrix([[0],[1],[2]])
      x6 = np.array([0,1,2])
      x7 = np.array([[0],[1],[2]])
            
      Parameters
      ----------
      obj: object
          Object to be verified.

      n: positive int
          Check if the vector is n-dimensional. If 'None', it does not matter.
          (default: 'None').

      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """
        try:
            p = Utils.cvt(obj)
            if n is None:
                return np.shape(p)[1] == 1
            else:
                return (np.shape(p)[0] == n) and (np.shape(p)[1] == 1)
        except:
            return False
        # return Utils.is_a_matrix(obj, n, 1) or Utils.is_a_matrix(obj, 1, n)

    @staticmethod
    def is_a_list_vector(obj, n=None):
        """
      Check if the argument is a list of vector of floats with the same dimension.

      Parameters
      ----------
      obj: object
          Object to be verified.

      n: positive int
          Check if all the vectors are n-dimensional. If 'None', it does not matter, 
          as long as they are equal.
          (default: 'None').

      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """
              
        if n is None:
            try:
                n_common = np.shape(Utils.cvt(obj[0]))[0]
            except:
                return False
        else:
            n_common = n
        
        for p in obj:
            if not Utils.is_a_vector(p, n_common):
                return False
            
        return True
    
    @staticmethod
    def is_a_pd_matrix(obj, n=None):
        """
      Check if the argument is a symmetric nxn positive (semi)-definite matrix.
      
      Parameters
      ----------
      obj: object
          Object to be verified.

      n: positive int
          Dimension of the square matrix
          (default: it does not matter).
    
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """
        value = Utils.is_a_matrix(obj, n, n)

        if value:
            value = np.allclose(obj, np.transpose(obj), rtol=1e-05, atol=1e-08)

        if value:
            try:
                np.linalg.cholesky(obj)
            except:
                value = False

        return value

    @staticmethod
    def is_a_color(obj):
        """
      Check if the argument is a HTML-compatible string that represents a color.
      
      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """

        try:
            obj = obj.replace(" ", "")
            Color(obj)
            return True
        except:
            return False

    @staticmethod
    def is_a_name(string):
        """
      Check if the argument is a valid name for uaibot objects.
      Only characters [a-z], [A-z], [0-9] and '_' are allowed.
      However, variables should not begin with numbers.

      Parameters
      ----------
      string: string
          Name to be verified.

      Returns
      -------
      is_name: boolean
          If the name is a valid name.
      """

        if str(type(string)) == "<class 'str'>":
            if string=='':
                return True
            else:
                allowed1 = set(st.ascii_lowercase + st.ascii_uppercase + st.digits + "_")
                allowed2 = set(st.ascii_lowercase + st.ascii_uppercase + "_")

                return (set(string) <= allowed1) and (set(string[0]) <= allowed2)
        else:
            return False

    @staticmethod
    def get_uaibot_type(obj):
        """
      Return the UAIBot type of the object.
      Return the empty string if it is not an UAIBot object.
      
      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      obj_type: string
          UAIBot type.
      """
        type_str = str(type(obj))

        ind = -1
        k = 0
        while ind == -1 and k < len(Utils.UAIBOT_NAME_TYPES):
            ind = type_str.find(Utils.UAIBOT_NAME_TYPES[k])
            k += 1

        if ind == -1:
            return ""
        else:
            ind1 = type_str.rfind('.')
            ind2 = type_str.rfind('>')
            return "uaibot." + type_str[ind1 + 1:ind2 - 1]

    @staticmethod
    def is_a_simple_object(obj):
        """
      Check if the argument is a simple object.
      Check the constant 'Utils.IS_SIMPLE' for a list of simple objects.

      Parameters
      ----------
      obj: object
          Object to be verified.

      Returns
      -------
      is_type: boolean
          If the object is of the type.
      """
        return Utils.get_uaibot_type(obj) in Utils.IS_SIMPLE

    @staticmethod
    def is_a_metric_object(obj):
        """
      Check if the argument is a metric object (if distance can be computed).
      Check the constant 'Utils.IS_METRIC' for a list of metric objects.

      Parameters
      ----------
      obj: object
          Object to be verified.

      Returns
      -------
      is_type: boolean
          If the object is of the type.
      """
        return Utils.get_uaibot_type(obj) in Utils.IS_METRIC
    
    @staticmethod
    def is_a_groupable_object(obj):
        """
      Check if the argument is a groupable object.
      Check the constant 'Utils.IS_GROUPABLE' for a list of groupable objects.

      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """
        return Utils.get_uaibot_type(obj) in Utils.IS_GROUPABLE

    @staticmethod
    def is_a_obj_sim(obj):
        """
      Check if the argument is an object that can be put into the simulator.
      Check the constant 'Utils.IS_OBJ_SIM' for a list of objects that can be put in the simulator.
      
      Parameters
      ----------
      obj: object
          Object to be verified.
      
      Returns
      -------
      is_type: boolean
          If the object is of the type.   
      """
        return Utils.get_uaibot_type(obj) in Utils.IS_OBJ_SIM

    @staticmethod
    def is_url_available(url, types):
        """
      Try to access the content of the url 'url'. Also verifies if the content is one of the extensions contained in
      'types' (e.g., types = ['png', 'bmp', 'jpg', 'jpeg'] for images).

      Never throws an Exception, always returning a string with a message. Returns 'ok!' if and only if the
      url was succesfully acessed and has the correct file type.

      Parameters
      ----------
      url: string
          The url string.

      types: list of string
          The desired content extensions.

      Returns
      -------
      message: string
          Message.
      """

        if not (str(type(url)) == "<class 'str'>"):
            return " is not a valid url"

        ind = url.rfind('.')
        filetype = url[ind + 1:]

        if not (filetype in types):
            return " must be an file of the types: '" + str(types) + "'"

        try:
            h = httplib2.Http(disable_ssl_certificate_validation=True) #Change this later
            resp = h.request(url, 'HEAD')
            if int(resp[0]['status']) < 400:
                return "ok!"
            else:
                return " : not able to access '" + url + "'."
        except:
            return " : not able to access '" + url + "'."

    @staticmethod
    def get_environment():
        try:
            import google.colab
            return "Colab"
        except ImportError:
            try:
                from IPython import get_ipython
                if "google.colab" in sys.modules:
                    return "Colab"
                elif get_ipython() is not None:
                    return "Jupyter"
            except ImportError:
                return "Not in Jupyter/Colab"
        return "Local"

    @staticmethod
    def obj_to_cpp(obj, htm=None):
            
            if Utils.get_uaibot_type(obj) != 'uaibot.PointCloud' and htm is None:
                htm = obj.htm
                
                
            if Utils.get_uaibot_type(obj) == 'uaibot.Ball':
                return ub_cpp.CPP_GeometricPrimitives.create_sphere(htm, obj.radius)
            if Utils.get_uaibot_type(obj) == 'uaibot.Box':
                return ub_cpp.CPP_GeometricPrimitives.create_box(htm, obj.width, obj.depth, obj.height)
            if Utils.get_uaibot_type(obj) == 'uaibot.Cylinder':
                return ub_cpp.CPP_GeometricPrimitives.create_cylinder(htm, obj.radius, obj.height)
            if Utils.get_uaibot_type(obj) == 'uaibot.PointCloud':
                return obj.cpp_pointcloud 
            if Utils.get_uaibot_type(obj) == 'uaibot.ConvexPolytope':
                Q = htm[0:3,0:3]
                p = htm[0:3,-1]
                A = obj.A * Q.transpose()
                b = obj.b + obj.A * Q.transpose()*p
                return ub_cpp.CPP_GeometricPrimitives.create_convexpolytope(htm, A, b)
                
    #######################################
    # Distance computation functions
    #######################################

    @staticmethod
    def softmin(x: List[float], h: float) -> float:
        minval = np.min(x)
        s=0

        for val in x:
            s+= exp(-h*(val-minval))

        return minval -(1/h)*np.log(s)

    @staticmethod
    def softselectmin(x: List[Vector], y: List[float], h: float) -> np.matrix:
        minval = np.min(x)
        s = 0

        coef = []
        for val in x:
            coef.append(exp(-(val - minval)/h))
            s += coef[-1]

        coef = [c/s for c in coef]

        sselect = 0 * y[0]
        for i in range(len(coef)):
            sselect += coef[i] * y[i]

        return sselect, minval - h * np.log(s/len(coef))


    @staticmethod
    def softmax(x: List[float], h: float) -> float:
        return Utils.softmin(x,-h)

    @staticmethod
    def softselectmax(x: List[Vector], y: List[float], h: float) -> np.matrix:
        return Utils.softselectmin(x,y,-h)

    @staticmethod
    def compute_aabbdist(obj1: MetricObject , obj2: MetricObject) -> float:

        box1 = obj1.aabb()
        box2 = obj2.aabb()
        delta = box1.htm[0:3, -1] - box2.htm[0:3, -1]

        dx = max(abs(delta[0, 0]) - (box1.width + box2.width) / 2, 0)
        dy = max(abs(delta[1, 0]) - (box1.depth + box2.depth) / 2, 0)
        dz = max(abs(delta[2, 0]) - (box1.height + box2.height) / 2, 0)

        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


    @staticmethod
    def _compute_dist_python(obj_a: MetricObject, obj_b: MetricObject, p_a: Vector, 
                            tol: float =0.001, no_iter_max: int =20) -> Tuple[np.matrix, np.matrix, float, List]:

        converged = False
        i = 0
        
        hist_error = []

        while (not converged) and i < no_iter_max:
            p_a_ant = p_a
            p_b, _ = obj_b.projection(p_a)
            p_a, _ = obj_a.projection(p_b)
            hist_error.append(np.linalg.norm(p_a - p_a_ant))
            converged = hist_error[-1] < tol
            i += 1

        dist = np.linalg.norm(p_a - p_b)

        return p_a, p_b, dist, []
                
    @staticmethod
    def compute_dist(obj_a: MetricObject, obj_b: MetricObject, p_a_init: Optional[Vector] = None, 
                     tol: float =0.001, no_iter_max: int=20, h: float=0, 
                     eps: float = 0, mode: str ='auto') -> Tuple[np.matrix, np.matrix, float, List]:
        """
    Compute Euclidean distance or differentiable distance between two objects.
    
    If h>0 or eps > 0, it computes the Euclidean distance and it uses GJK's algorithm.
    
    Else, it computes the differentiable distance through Generalized Alternating Projection (GAP).
    See the paper 'A Differentiable Distance Metric for Robotics Through Generalized Alternating Projection'.
    This only works in c++ mode, though.
    
    
    Parameters
    ----------
    obj_a : an object of type 'MetricObject' (see Utils.IS_METRIC)
        The first object.
        
    obj_b : an object of type 'MetricObject' (see Utils.IS_METRIC)
        The second object.
        
    p_a_init : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array) or None
        Initial point for closest point in 'obj_a'. If 'None', is set to random.
        (default: None).
    
    tol : positive float
        Convergence criterion of GAP: it stops when ||a[k+1]-a[k]|| < tol.
        Only valid when h > 0 or eps > 0.
        (default: 0.001m).      

    no_iter_max : positive int 
        Maximum number of iterations of GAP.
        Only valid when h > 0 or eps > 0.
        (default: 20 iterations). 

    h : nonnegative float
        h parameter in the generalized distance function.
        If h=0 and eps=0, it is simply the Euclidean distance.
        (default: 0). 

    eps : nonnegative float
        h parameter in the generalized distance function.
        If h=0 and eps=0, it is simply the Euclidean distance.
        (default: 0). 

    mode : string
    'c++' for the c++ implementation, 'python' for the python implementation
    and 'auto' for automatic ('c++' is available, else 'python').
    (default: 'auto').
                                                    
    Returns
    -------
    point_a : 3 x 1 numpy matrix
        Closest point (Euclidean or differentiable) in obj_a.

    point_b : 3 x 1 numpy matrix
        Closest point (Euclidean or differentiable) in obj_b.

    distance : float
        Euclidean or differentiable distance.
        
    hist_error: list of floats
        History of convergence error.    
                
    """

        if mode=='python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            if Utils.get_uaibot_type(obj_a) == 'uaibot.PointCloud' or Utils.get_uaibot_type(obj_b) == 'uaibot.PointCloud':
                raise Exception("Point cloud distance is not supported in Python mode!")
            
            if not Utils.is_a_metric_object(obj_a):
                raise Exception("In python mode, the parameter 'obj_a' must be one of the following: " + str(Utils.IS_METRIC) + ".")

            if not Utils.is_a_metric_object(obj_b):
                raise Exception("In python mode, the parameter 'obj_b' must be one of the following: " + str(Utils.IS_METRIC) + ".")
            
        if mode=='c++':
            if not (Utils.get_uaibot_type(obj_a) == 'uaibot.CPP_GeometricPrimitives' or Utils.is_a_metric_object(obj_a)):
                raise Exception("In c++ mode, the parameter 'obj_a' must be 'uaibot.CPP_GeometricPrimitives' or "+ str(Utils.IS_METRIC) + ".")
            if not (Utils.get_uaibot_type(obj_b) == 'uaibot.CPP_GeometricPrimitives' or Utils.is_a_metric_object(obj_b)):
                raise Exception("In c++ mode, the parameter 'obj_b' must be 'uaibot.CPP_GeometricPrimitives' or "+ str(Utils.IS_METRIC) + ".")            


        if (mode == 'c++') or (mode=='auto' and os.environ['CPP_SO_FOUND']=='1'):
            if Utils.is_a_metric_object(obj_a):
                obj_a_cpp = Utils.obj_to_cpp(obj_a) 
            else:
                obj_a_cpp = obj_a

            if Utils.is_a_metric_object(obj_b):
                obj_b_cpp = Utils.obj_to_cpp(obj_b) 
            else:
                obj_b_cpp = obj_b
           
            
        if p_a_init is None:
            if ((mode == 'c++') or (mode=='auto' and os.environ['CPP_SO_FOUND']=='1')) and h >= 1e-5:
                dist_res = obj_a_cpp.dist_to(obj_b_cpp, 0.0, 0.0, tol, no_iter_max, np.matrix([0,0,0]).reshape((3,1)))
                p_a = Utils.cvt(dist_res.proj_A)
            else:
                p_a = Utils.cvt(np.random.uniform(-3, 3, size=(3,)))
        else:
            p_a = Utils.cvt(p_a_init)


        if not (p_a_init is None or Utils.is_a_vector(p_a_init, 3)):
            raise Exception("The optional parameter 'p_a_init' must be a 3D vector or 'None'.")

        if not Utils.is_a_number(tol) or tol <= 0:
            raise Exception("The optional parameter 'tol' must be a nonnegative number.")

        if not Utils.is_a_natural_number(no_iter_max):
            raise Exception("The optional parameter 'no_iter_max' must be a nonnegative integer.")
        
        if not Utils.is_a_number(h) or h < 0:
            raise Exception("The optional parameter 'h' must be a nonnegative number.")

        if not Utils.is_a_number(eps) or eps < 0:
            raise Exception("The optional parameter 'eps' must be a nonnegative number.")
                
        if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")
        # end error handling

        if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            if h >0 or eps > 0:
                raise Exception("In Python mode, smoothing parameters 'h' and 'eps' must be set to 0!")
            
            return Utils._compute_dist_python(obj_a, obj_b, p_a, tol, no_iter_max)
        else:
            dist_res = obj_a_cpp.dist_to(obj_b_cpp, h, eps, tol, no_iter_max, p_a)
            return Utils.cvt(dist_res.proj_A), Utils.cvt(dist_res.proj_B), dist_res.dist, dist_res.hist_error








