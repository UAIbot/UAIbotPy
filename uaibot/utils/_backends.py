"""Lazy optional backend loading; no concrete Python robot-class imports."""

from importlib import import_module
from .types import register_uaibot_type


def get_cpp_backend():
    try:
        backend = import_module("uaibot_cpp_bind")
    except ImportError as exc:
        raise ImportError(
            "The uaibot C++ backend could not be loaded. Install/build "
            "uaibot_cpp_bind, or select mode='python' where supported."
        ) from exc
    primitive = getattr(backend, "CPP_GeometricPrimitives", None)
    if isinstance(primitive, type):
        register_uaibot_type(primitive, name="uaibot.CPP_GeometricPrimitives")
    return backend


def get_backend(mode):
    """Explicit python never imports cpp; explicit cpp never silently falls back.

    Auto tries the extension and falls back on import failure. No environment
    variable is required; the actual extension import determines availability.
    """
    if not isinstance(mode, str) or mode not in ("python", "cpp", "c++", "auto"):
        raise ValueError("mode must be 'python', 'cpp', 'c++', or 'auto'")
    if mode == "python":
        return None
    if mode != "auto":
        return get_cpp_backend()
    try:
        return get_cpp_backend()
    except ImportError:
        return None
