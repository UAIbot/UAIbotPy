import numpy as np
from typing import Optional, List, Tuple
from math import exp
from .basic import cvt
from .types import Vector, MetricObject
from .typecheck import is_a_number, is_a_natural_number, is_a_vector, is_a_metric_object
from ._backends import get_backend
from ._cpp import GeometryKind, get_cpp_kind, is_a_cpp_object, obj_to_cpp


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
        p_b = cvt(p_b)
        p_a, _ = obj_a.projection(p_b)
        p_a = cvt(p_a)
        hist_error.append(np.linalg.norm(p_a - p_a_ant))
        converged = hist_error[-1] < tol
        i += 1

    dist = np.linalg.norm(p_a - p_b)

    return p_a, p_b, dist, hist_error


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
    """Compute closest points and distance using the user-selected backend.

    mode='python' uses alternating projections and NEVER loads C++. mode='cpp'
    (or the legacy 'c++') requires the extension. 'auto' prefers an importable
    extension and otherwise uses Python. No CPP_SO_FOUND variable is needed.

    Python requires Metric objects supporting projection, excludes registered
    point clouds, and requires h=eps=0. C++ accepts Metric objects with registered
    converters or native CPP_GeometricPrimitives. Nonzero h/eps request the
    backend's differentiable distance. Returns (point_a, point_b, distance,
    convergence_history), with points as 3-by-1 legacy NumPy matrices.
    """

    def scalar(value, label, positive):
        message = label + (
            " must be positive and finite"
            if positive
            else " must be nonnegative and finite"
        )
        if not is_a_number(value):
            raise ValueError(message)
        try:
            value = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(message)
        return value

    tol, h, eps = (
        scalar(tol, "tol", True),
        scalar(h, "h", False),
        scalar(eps, "eps", False),
    )
    if not is_a_natural_number(no_iter_max) or no_iter_max == 0:
        raise ValueError("no_iter_max must be a positive integer")
    no_iter_max = int(no_iter_max)
    if p_a_init is not None:
        if not is_a_vector(p_a_init, 3) or not np.isfinite(np.asarray(p_a_init)).all():
            raise ValueError("p_a_init must be a finite 3-D vector or None")
        p_a = cvt(p_a_init)
    else:
        p_a = None

    backend = get_backend(mode)
    if backend is None:
        if not (is_a_metric_object(obj_a) and is_a_metric_object(obj_b)):
            raise TypeError("Python distance requires two Metric objects")
        if any(get_cpp_kind(obj) is GeometryKind.POINT_CLOUD for obj in (obj_a, obj_b)):
            raise NotImplementedError(
                "Point-cloud distance is not supported by the Python implementation"
            )
        if h != 0 or eps != 0:
            raise ValueError(
                "Python distance requires h=eps=0; select mode='cpp' for smoothing"
            )
        if p_a is None:
            p_a = cvt(np.random.uniform(-3, 3, size=3))
        return _compute_dist_python(obj_a, obj_b, p_a, tol, no_iter_max)

    def native(obj):
        if is_a_cpp_object(obj, backend=backend):
            return obj
        if not is_a_metric_object(obj):
            raise TypeError("C++ distance requires Metric or native primitive objects")
        return obj_to_cpp(obj)

    obj_a_cpp, obj_b_cpp = native(obj_a), native(obj_b)
    if p_a is None:
        if h >= 1.0e-5:
            seed = obj_a_cpp.dist_to(
                obj_b_cpp, 0.0, 0.0, tol, no_iter_max, cvt([0.0, 0.0, 0.0])
            )
            p_a = cvt(seed.proj_A)
        else:
            p_a = cvt(np.random.uniform(-3, 3, size=3))
    result = obj_a_cpp.dist_to(obj_b_cpp, h, eps, tol, no_iter_max, p_a)
    return cvt(result.proj_A), cvt(result.proj_B), result.dist, result.hist_error
