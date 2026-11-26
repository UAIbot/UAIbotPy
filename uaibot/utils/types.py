from abc import ABC
from typing import Union, List, Tuple, TypeAlias, TYPE_CHECKING, Optional
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Runtime ABCs. Nothing here imports from the rest of uaibot, so this module
# is a safe leaf that both `utils` and every `simobjects` file can import.
# ---------------------------------------------------------------------------


class UaibotObject(ABC):
    """Runtime identity marker without concrete uaibot imports."""


class ObjSim(UaibotObject):
    """Anything that can be placed into a Simulation (has gen_code, name, ...)."""


class Groupable(ObjSim):
    """Anything that can be attached to a Robot or added to a Group."""


class Metric(ObjSim):
    """Anything for which a distance / projection can be computed."""


class Simple(Metric, Groupable):
    """Ball, Box, Cylinder, ConvexPolytope."""


class HtmlDiv(ObjSim):
    """HTMLDiv, kept separate because it's not a physical object."""


_UAIBOT_NAMES = {}


def register_uaibot_type(cls=None, *, name=None):
    """Register class identity and optional display label; usable as decorator.

    Materials/links/Simulation may register without becoming ObjSim/Metric.
    The label is output metadata, never a dispatch/type-checking key.
    """
    if name is not None and (not isinstance(name, str) or not name):
        raise TypeError("name must be a nonempty display string or None")

    def register(concrete):
        if not isinstance(concrete, type):
            raise TypeError("register_uaibot_type expects a class")
        _UAIBOT_NAMES[concrete] = (
            name if name is not None else "uaibot." + concrete.__name__
        )
        UaibotObject.register(concrete)
        return concrete

    return register if cls is None else register(cls)


def _registered_uaibot_name(cls):
    for base in cls.__mro__:
        if base in _UAIBOT_NAMES:
            return _UAIBOT_NAMES[base]
    return None


# ---------------------------------------------------------------------------
# The old Union aliases are still useful for static type checkers (they get
# much better narrowing than ABCs).  At runtime we swap them for the ABCs so
# that isinstance(obj, MetricObject) works.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from uaibot.simobjects import (
        Ball,
        Box,
        ConvexPolytope,
        Cylinder,
        PointCloud,
        Frame,
        RigidObject,
        Group,
        PointLight,
        Arrow,
        HTMLDiv,
    )
    from uaibot.robot import Robot

    MetricObject: TypeAlias = Union[Ball, Box, ConvexPolytope, Cylinder, PointCloud]
    GroupableObject: TypeAlias = Union[
        Ball,
        Box,
        Cylinder,
        ConvexPolytope,
        Frame,
        RigidObject,
        Group,
        Robot,
        PointLight,
    ]
    SimObject: TypeAlias = Union[
        Ball,
        Box,
        Cylinder,
        ConvexPolytope,
        Frame,
        RigidObject,
        Group,
        Robot,
        PointLight,
        Arrow,
        HTMLDiv,
    ]
else:
    MetricObject = Metric
    GroupableObject = Groupable
    SimObject = ObjSim

# Keep the type aliases you already have (Vector, Matrix, HTMatrix)
Vector: TypeAlias = Union[
    List[float], Tuple[float, ...], npt.NDArray[np.float64], np.matrix
]
Matrix: TypeAlias = Union[npt.NDArray[np.float64], np.matrix]
HTMatrix: TypeAlias = np.matrix

__all__ = [
    "UaibotObject",
    "ObjSim",
    "Groupable",
    "Metric",
    "Simple",
    "HtmlDiv",
    "MetricObject",
    "GroupableObject",
    "SimObject",
    "Vector",
    "Matrix",
    "HTMatrix",
    "register_uaibot_type",
]
