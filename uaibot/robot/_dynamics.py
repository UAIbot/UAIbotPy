"""Validation shared by the Python robot interfaces."""

from importlib import import_module

import numpy as np

from uaibot.utils import is_a_matrix, is_a_number, is_a_vector


def dynamics_backend(required=True):
    try:
        cpp = import_module("uaibot_cpp_bind")
    except ImportError:
        cpp = None
    if cpp is not None and hasattr(cpp, "ManipulatorDynamics"):
        return cpp
    if required:
        raise ImportError(
            "Build Uaibot with the C++ dynamics bindings to use robot dynamics."
        )
    return None


def pose_array(value):
    if not is_a_matrix(value, 4, 4):
        raise ValueError("htm must be a 4 x 4 homogeneous transform")
    pose = np.array(value, dtype=float, copy=True)
    r = pose[:3, :3]
    if (
        not np.isfinite(pose).all()
        or np.linalg.norm(pose[3] - [0, 0, 0, 1]) >= 1e-12
        or np.linalg.norm(r.T @ r - np.eye(3)) >= 1e-6
        or abs(np.linalg.det(r) - 1) >= 1e-6
    ):
        raise ValueError("htm must be a finite SE(3) pose with a proper rotation")
    return pose


def finite_vector(value, size, name):
    if not is_a_vector(value, size):
        dimension = "" if size is None else f" of length {size}"
        raise ValueError(f"{name} must be a real vector{dimension}")
    value = np.asarray(value, dtype=float).reshape(-1)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must have finite entries")
    return value


def frame_time(value):
    if not is_a_number(value) or not np.isfinite(value) or value < 0:
        raise ValueError("time must be finite and nonnegative")
    return float(value)
