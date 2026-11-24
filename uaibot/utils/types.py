from abc import ABC
from typing import Union, List, Tuple, TypeAlias, TYPE_CHECKING, Optional
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Runtime ABCs. Nothing here imports from the rest of uaibot, so this module
# is a safe leaf that both `utils` and every `simobjects` file can import.
# ---------------------------------------------------------------------------


class ObjSim(ABC):
    """Anything that can be placed into a Simulation (has gen_code, name, ...)."""


class Groupable(ObjSim):
    """Anything that can be attached to a Robot or added to a Group."""


class Metric(ObjSim):
    """Anything for which a distance / projection can be computed."""


class Simple(Metric, Groupable):
    """Ball, Box, Cylinder, ConvexPolytope."""


class HtmlDiv(ObjSim):
    """HTMLDiv, kept separate because it's not a physical object."""


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
Vector: TypeAlias = Union[List[float], Tuple[float], npt.NDArray[np.float64], np.matrix]
Matrix: TypeAlias = Union[npt.NDArray[np.float64], np.matrix]
HTMatrix: TypeAlias = np.matrix
