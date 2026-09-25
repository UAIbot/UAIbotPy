import os
import numpy as np
from typing import Optional, List, Tuple
from math import exp
from .basic import cvt
from .types import Vector, MetricObject


def softmin(x: List[float], h: float) -> float:
    minval = np.min(x)
    s = 0

    for val in x:
        s += exp(-h * (val - minval))

    return minval - (1 / h) * np.log(s)


def softselectmin(x: List[Vector], y: List[float], h: float) -> np.matrix:
    minval = np.min(x)
    s = 0

    coef = []
    for val in x:
        coef.append(exp(-(val - minval) / h))
        s += coef[-1]

    coef = [c / s for c in coef]

    sselect = 0 * y[0]
    for i in range(len(coef)):
        sselect += coef[i] * y[i]

    return sselect, minval - h * np.log(s / len(coef))


def softmax(x: List[float], h: float) -> float:
    return softmin(x, -h)


def softselectmax(x: List[Vector], y: List[float], h: float) -> np.matrix:
    return softselectmin(x, y, -h)


def compute_aabbdist(obj1: MetricObject, obj2: MetricObject) -> float:

    box1 = obj1.aabb()
    box2 = obj2.aabb()
    delta = box1.htm[0:3, -1] - box2.htm[0:3, -1]

    dx = max(abs(delta[0, 0]) - (box1.width + box2.width) / 2, 0)
    dy = max(abs(delta[1, 0]) - (box1.depth + box2.depth) / 2, 0)
    dz = max(abs(delta[2, 0]) - (box1.height + box2.height) / 2, 0)

    return np.sqrt(dx**2 + dy**2 + dz**2)


def _compute_dist_python(
    obj_a: MetricObject,
    obj_b: MetricObject,
    p_a: Vector,
    tol: float = 0.001,
    no_iter_max: int = 20,
) -> Tuple[np.matrix, np.matrix, float, List]:

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


def compute_dist(
    obj_a: MetricObject,
    obj_b: MetricObject,
    p_a_init: Optional[Vector] = None,
    tol: float = 0.001,
    no_iter_max: int = 20,
    h: float = 0,
    eps: float = 0,
    mode: str = "auto",
) -> Tuple[np.matrix, np.matrix, float, List]:
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

    if mode == "python" or (mode == "auto" and os.environ["CPP_SO_FOUND"] == "0"):
        if (
            Utils.get_uaibot_type(obj_a) == "uaibot.PointCloud"
            or Utils.get_uaibot_type(obj_b) == "uaibot.PointCloud"
        ):
            raise Exception("Point cloud distance is not supported in Python mode!")

        if not Utils.is_a_metric_object(obj_a):
            raise Exception(
                "In python mode, the parameter 'obj_a' must be one of the following: "
                + str(Utils.IS_METRIC)
                + "."
            )

        if not Utils.is_a_metric_object(obj_b):
            raise Exception(
                "In python mode, the parameter 'obj_b' must be one of the following: "
                + str(Utils.IS_METRIC)
                + "."
            )

    if mode == "c++":
        if not (
            Utils.get_uaibot_type(obj_a) == "uaibot.CPP_GeometricPrimitives"
            or Utils.is_a_metric_object(obj_a)
        ):
            raise Exception(
                "In c++ mode, the parameter 'obj_a' must be 'uaibot.CPP_GeometricPrimitives' or "
                + str(Utils.IS_METRIC)
                + "."
            )
        if not (
            Utils.get_uaibot_type(obj_b) == "uaibot.CPP_GeometricPrimitives"
            or Utils.is_a_metric_object(obj_b)
        ):
            raise Exception(
                "In c++ mode, the parameter 'obj_b' must be 'uaibot.CPP_GeometricPrimitives' or "
                + str(Utils.IS_METRIC)
                + "."
            )

    if (mode == "c++") or (mode == "auto" and os.environ["CPP_SO_FOUND"] == "1"):
        if Utils.is_a_metric_object(obj_a):
            obj_a_cpp = obj_a.cpp_obj
        else:
            obj_a_cpp = obj_a

        if Utils.is_a_metric_object(obj_b):
            obj_b_cpp = obj_b.cpp_obj
        else:
            obj_b_cpp = obj_b

    if p_a_init is None:
        if (
            (mode == "c++") or (mode == "auto" and os.environ["CPP_SO_FOUND"] == "1")
        ) and h >= 1e-5:
            dist_res = obj_a_cpp.dist_to(
                obj_b_cpp,
                0.0,
                0.0,
                tol,
                no_iter_max,
                np.matrix([0, 0, 0]).reshape((3, 1)),
            )
            p_a = Utils.cvt(dist_res.proj_A)
        else:
            p_a = Utils.cvt(np.random.uniform(-3, 3, size=(3,)))
    else:
        p_a = Utils.cvt(p_a_init)

    if not (p_a_init is None or Utils.is_a_vector(p_a_init, 3)):
        raise Exception(
            "The optional parameter 'p_a_init' must be a 3D vector or 'None'."
        )

    if not Utils.is_a_number(tol) or tol <= 0:
        raise Exception("The optional parameter 'tol' must be a nonnegative number.")

    if not Utils.is_a_natural_number(no_iter_max):
        raise Exception(
            "The optional parameter 'no_iter_max' must be a nonnegative integer."
        )

    if not Utils.is_a_number(h) or h < 0:
        raise Exception("The optional parameter 'h' must be a nonnegative number.")

    if not Utils.is_a_number(eps) or eps < 0:
        raise Exception("The optional parameter 'eps' must be a nonnegative number.")

    if mode == "c++" and os.environ["CPP_SO_FOUND"] == "0":
        raise Exception("c++ mode is set, but .so file was not loaded!")
    # end error handling

    if mode == "python" or (mode == "auto" and os.environ["CPP_SO_FOUND"] == "0"):
        if h > 0 or eps > 0:
            raise Exception(
                "In Python mode, smoothing parameters 'h' and 'eps' must be set to 0!"
            )

        return Utils._compute_dist_python(obj_a, obj_b, p_a, tol, no_iter_max)
    else:
        dist_res = obj_a_cpp.dist_to(obj_b_cpp, h, eps, tol, no_iter_max, p_a)
        return (
            Utils.cvt(dist_res.proj_A),
            Utils.cvt(dist_res.proj_B),
            dist_res.dist,
            dist_res.hist_error,
        )
