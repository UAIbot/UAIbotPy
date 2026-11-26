"""Class-identity based C++ adapters, without simobject imports.

After defining Ball in ball.py: register_cpp_type(Ball, GeometryKind.SPHERE).
Registration does not import the extension. obj_to_cpp is explicit conversion;
backend-selectable operations bypass it completely in mode='python'.
"""

from enum import Enum, auto
from functools import singledispatch
import numpy as np
from ._arrays import real_array
from ._backends import get_cpp_backend
from .typecheck import is_a_matrix, is_a_number, is_a_vector
from .types import register_uaibot_type

__all__ = [
    "GeometryKind",
    "register_cpp_type",
    "register_cpp_converter",
    "get_cpp_kind",
    "is_a_cpp_object",
    "obj_to_cpp",
]


class GeometryKind(Enum):
    SPHERE = auto()
    BOX = auto()
    CYLINDER = auto()
    POINT_CLOUD = auto()
    CONVEX_POLYTOPE = auto()


def _pose(obj, htm):
    value = obj.htm if htm is None else htm
    if not is_a_matrix(value, 4, 4):
        raise ValueError("htm must be a real 4-by-4 homogeneous pose")
    with np.errstate(over="ignore", invalid="ignore"):
        pose = np.asarray(value, dtype=np.float64)
    rotation = pose[:3, :3]
    if not (
        np.isfinite(pose).all()
        and np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1.0e-8)
        and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-6)
        and np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError("htm must be a finite, proper rigid-body pose")
    return np.matrix(pose, copy=True)


def _positive(value, label):
    if not is_a_number(value):
        raise ValueError(label + " must be a positive finite real scalar")
    try:
        scalar = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(label + " is not representable as a C++ scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(label + " must be a positive finite real scalar")
    return scalar


def _sphere(obj, htm=None):
    pose = _pose(obj, htm)
    radius = _positive(obj.radius, "radius")
    return get_cpp_backend().CPP_GeometricPrimitives.create_sphere(pose, radius)


def _box(obj, htm=None):
    pose = _pose(obj, htm)
    width, depth, height = (
        _positive(obj.width, "width"),
        _positive(obj.depth, "depth"),
        _positive(obj.height, "height"),
    )
    return get_cpp_backend().CPP_GeometricPrimitives.create_box(
        pose, width, depth, height
    )


def _cylinder(obj, htm=None):
    pose = _pose(obj, htm)
    radius, height = _positive(obj.radius, "radius"), _positive(obj.height, "height")
    return get_cpp_backend().CPP_GeometricPrimitives.create_cylinder(
        pose, radius, height
    )


def _pointcloud(obj, htm=None):
    # Preserve the original convention: points are world coordinates; no .htm
    # is needed and the optional htm argument is not applied to point clouds.
    if not is_a_matrix(obj._points, 3, None):
        raise ValueError("point-cloud _points must be a nonempty real 3-by-N matrix")
    with np.errstate(over="ignore", invalid="ignore"):
        points = np.asarray(obj._points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("point-cloud coordinates must be finite")
    columns = [
        np.matrix(points[:, i : i + 1], copy=True) for i in range(points.shape[1])
    ]
    return get_cpp_backend().CPP_GeometricPrimitives.create_pointcloud(columns)


def _polytope(obj, htm=None):
    pose = _pose(obj, htm)
    if not is_a_matrix(obj.A, None, 3):
        raise ValueError("polytope A must be a nonempty real M-by-3 matrix")
    if not is_a_vector(obj.b, real_array(obj.A).shape[0]):
        raise ValueError("polytope b must have one entry per row of A")
    with np.errstate(over="ignore", invalid="ignore"):
        local_a = np.asarray(obj.A, dtype=np.float64)
        local_b = np.asarray(obj.b, dtype=np.float64).reshape(-1, 1)
    # x_world=Q*x_local+p -> A_world=A_local@Q.T; b_world=b_local+A_world@p.
    # * would be elementwise multiplication for ndarray inputs.
    world_a = local_a @ np.asarray(pose[:3, :3]).T
    world_b = local_b + world_a @ np.asarray(pose[:3, 3])
    if not (np.isfinite(world_a).all() and np.isfinite(world_b).all()):
        raise ValueError("polytope inequalities must be finite")
    return get_cpp_backend().CPP_GeometricPrimitives.create_convexpolytope(
        pose, np.matrix(world_a), np.matrix(world_b)
    )


@singledispatch
def _convert(obj, htm=None):
    raise TypeError(
        "No C++ converter registered for "
        + type(obj).__qualname__
        + "; call register_cpp_type or register_cpp_converter after defining the class"
    )


_HANDLERS = {
    GeometryKind.SPHERE: _sphere,
    GeometryKind.BOX: _box,
    GeometryKind.CYLINDER: _cylinder,
    GeometryKind.POINT_CLOUD: _pointcloud,
    GeometryKind.CONVEX_POLYTOPE: _polytope,
}
_KINDS = {handler: kind for kind, handler in _HANDLERS.items()}


def register_cpp_type(cls, kind, *, name=None):
    """Register a class with a built-in adapter; subclasses inherit dispatch.

    Call after defining the class. No extension import or native construction
    occurs. name is an optional legacy display label, not a dispatch key.
    """
    if not isinstance(cls, type):
        raise TypeError("cls must be a class")
    if not isinstance(kind, GeometryKind):
        raise TypeError("kind must be a GeometryKind member")
    register_uaibot_type(cls, name=name)
    _convert.register(cls, _HANDLERS[kind])
    return cls


def register_cpp_converter(cls, converter):
    """Register callable(obj, htm=None) -> native object; return the callable.

    Custom adapters should load their backend lazily. Registration does not
    classify a type as Simple/Metric; its existing ABC inheritance does that.
    """
    if not isinstance(cls, type) or not callable(converter):
        raise TypeError("expected a class and a callable converter")
    register_uaibot_type(cls)
    _convert.register(cls, converter)
    return converter


def get_cpp_kind(obj):
    """Registered GeometryKind or None; never loads the backend."""
    return _KINDS.get(_convert.dispatch(type(obj)))


def is_a_cpp_object(obj, *, backend=None):
    if backend is None:
        try:
            backend = get_cpp_backend()
        except ImportError:
            return False
    native_type = getattr(backend, "CPP_GeometricPrimitives", None)
    return isinstance(native_type, type) and isinstance(obj, native_type)


def obj_to_cpp(obj, htm=None):
    """Explicit C++ conversion; unsupported classes raise TypeError.

    Backend-selectable operations skip this in mode='python'. No Ball/Robot/
    simobjects imports occur anywhere in the dispatch path.
    """
    return _convert(obj, htm)
