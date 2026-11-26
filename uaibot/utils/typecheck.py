"""Runtime predicates without concrete uaibot-class or backend imports.

Numeric predicates check real numeric types; NaN/infinity remain floats.
Vectors are nonempty (n,), (n,1) or (1,n); matrices are exactly 2-D.
"""

import string as _string
import numpy as np
from ._arrays import is_integral, is_real_number, real_array
from .types import (
    Groupable,
    Metric,
    ObjSim,
    Simple,
    UaibotObject,
    _registered_uaibot_name,
)

__all__ = [
    "is_a_number",
    "is_a_natural_number",
    "is_a_matrix",
    "is_a_vector",
    "is_a_list_vector",
    "is_a_pd_matrix",
    "is_a_color",
    "is_a_name",
    "get_uaibot_type",
    "is_a_simple_object",
    "is_a_metric_object",
    "is_a_groupable_object",
    "is_a_obj_sim",
    "IS_SIMPLE",
    "IS_METRIC",
    "IS_GROUPABLE",
    "IS_OBJ_SIM",
    "UAIBOT_NAME_TYPES",
]

# Compatibility category names now hold classes, not class-name strings.
IS_SIMPLE = (Simple,)
IS_METRIC = (Metric,)
IS_GROUPABLE = (Groupable,)
IS_OBJ_SIM = (ObjSim,)
UAIBOT_NAME_TYPES = (UaibotObject,)


def is_a_number(obj):
    """A Python/NumPy real scalar, excluding bool and complex."""
    return is_real_number(obj)


def is_a_natural_number(obj):
    """A Python/NumPy integer >= 0; booleans are not integers here."""
    return bool(is_integral(obj) and obj >= 0)


def _valid_dimension(n):
    return n is None or (is_integral(n) and n > 0)


def is_a_matrix(obj, n=None, m=None):
    """Check nonempty real 2-D data, optionally of shape (n,m).

    Accept lists, tuples, ndarray and np.matrix. Reject bool/string/complex/
    object data, ragged sequences and nonpositive/nonintegral dimensions.
    """
    if not (_valid_dimension(n) and _valid_dimension(m)):
        return False
    array = real_array(obj)
    return bool(
        array is not None
        and array.ndim == 2
        and array.size > 0
        and (n is None or array.shape[0] == n)
        and (m is None or array.shape[1] == m)
    )


def is_a_vector(obj, n=None):
    """Check a nonempty real flat, row or column vector, optionally length n."""
    if not _valid_dimension(n):
        return False
    array = real_array(obj)
    if array is None or array.size == 0:
        return False
    shape_ok = array.ndim == 1 or (array.ndim == 2 and 1 in array.shape)
    return bool(shape_ok and (n is None or array.size == n))


def is_a_list_vector(obj, n=None):
    """A nonempty list/tuple/array of vectors with equal dimension.

    For a 2-D array, entries are rows. Iterators are not consumed.
    """
    if not _valid_dimension(n) or not isinstance(obj, (list, tuple, np.ndarray)):
        return False
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return False
    if len(obj) == 0 or not is_a_vector(obj[0], n):
        return False
    dimension = real_array(obj[0]).size if n is None else n
    return all(is_a_vector(item, dimension) for item in obj)


def is_a_pd_matrix(obj, n=None):
    """A finite symmetric strictly positive-definite square matrix.

    Singular PSD matrices fail, as in the original Cholesky implementation.
    Symmetry uses the existing rtol=1e-5, atol=1e-8 tolerances.
    """
    if not is_a_matrix(obj, n, n):
        return False
    array = real_array(obj)
    if array.shape[0] != array.shape[1] or not np.isfinite(array).all():
        return False
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(array, dtype=np.float64)
        if not np.isfinite(array).all() or not np.allclose(
            array, array.T, rtol=1.0e-5, atol=1.0e-8
        ):
            return False
        np.linalg.cholesky(0.5 * array + 0.5 * array.T)
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return False
    return True


def is_a_color(obj):
    """A colour.Color-compatible string; needs colour only for string inputs."""
    if not isinstance(obj, str):
        return False
    from colour import Color

    try:
        Color(obj.replace(" ", ""))
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def is_a_name(string):
    """ASCII letters/digits/underscore, no leading digit; empty is valid."""
    if not isinstance(string, str):
        return False
    if string == "":
        return True
    first = _string.ascii_letters + "_"
    return string[0] in first and all(c in first + _string.digits for c in string)


def get_uaibot_type(obj):
    """Legacy display label or ''. Membership uses ABC/class identity.

    Registered subclasses inherit their nearest registered ancestor's label.
    Other UaibotObject descendants use their own class name for display only.
    Algorithms must use ABCs/dispatch rather than comparing these labels.
    """
    name = _registered_uaibot_name(type(obj))
    if name is not None:
        return name
    if isinstance(obj, UaibotObject):
        return "uaibot." + type(obj).__name__
    return ""


def is_a_simple_object(obj):
    return isinstance(obj, Simple)


def is_a_metric_object(obj):
    return isinstance(obj, Metric)


def is_a_groupable_object(obj):
    return isinstance(obj, Groupable)


def is_a_obj_sim(obj):
    return isinstance(obj, ObjSim)
