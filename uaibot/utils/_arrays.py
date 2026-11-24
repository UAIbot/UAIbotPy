"""Numeric helpers; no concrete classes, facade imports or optional backends."""

from numbers import Integral, Real
import numpy as np


def is_real_number(value):
    return isinstance(value, Real) and not isinstance(
        value, (bool, np.bool_, np.timedelta64)
    )


def is_integral(value):
    return isinstance(value, Integral) and not isinstance(
        value, (bool, np.bool_, np.timedelta64)
    )


def _real_dtype(dtype):
    # NumPy considers timedelta64 an integer, although it is not a numeric length.
    return (
        np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)
    ) and not np.issubdtype(dtype, np.timedelta64)


def _real_contents(value):
    if isinstance(value, np.ma.MaskedArray):
        return False
    if isinstance(value, np.ndarray):
        return _real_dtype(value.dtype)
    if isinstance(value, (list, tuple)):
        return all(_real_contents(item) for item in value)
    return is_real_number(value)


def real_array(value):
    """Return a numeric array or None, without coercing strings/bools.

    Object arrays, ragged sequences and masked data are rejected. Finiteness
    and dimensions are caller-specific requirements, not type properties.
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    try:
        if not _real_contents(value):
            return None
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None
    return array if _real_dtype(array.dtype) else None
