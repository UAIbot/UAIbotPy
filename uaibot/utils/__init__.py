"""
uaibot.utils public surface.

Prefer `from uaibot import utils` and use `utils.foo(...)` directly.
The `Utils` class below is a deprecated shim kept for backwards compatibility.
"""

import sys as _sys

_self = _sys.modules[__name__]

# Import everything that was previously inside the Utils class body.
from .types import *  # noqa: F401,F403
from .basic import *  # noqa: F401,F403
from .linalg import *  # noqa: F401,F403
from .interpolation import *  # noqa: F401,F403
from .typecheck import *  # noqa: F401,F403
from ._cpp import *  # noqa: F401,F403
from .distances import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .data import *  # noqa: F401,F403


class _UtilsMeta(type):
    """
    Makes `Utils.foo` resolve to `uaibot.utils.foo`.

    Any access to an attribute not defined directly on the class body falls
    through to the module. This keeps `Utils.is_a_matrix(...)`,
    `Utils.IS_SIMPLE`, `Utils.rot(...)`, etc. working for existing user code
    without having to re-declare every function as a staticmethod.
    """

    def __getattr__(cls, name):
        # Never intercept dunder lookups — those have special semantics.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            return getattr(_self, name)
        except AttributeError as exc:
            raise AttributeError(f"Utils has no attribute {name!r}") from exc

    def __setattr__(cls, name, value):
        # Setting Utils.X writes through to the module, so
        # `Utils.IS_SIMPLE = ...` at runtime still works as expected.
        setattr(_self, name, value)

    def __dir__(cls):
        # Make tab-completion and dir() show everything on the module.
        return sorted(set(dir(type)) | set(dir(_self)))


class Utils(metaclass=_UtilsMeta):
    """Deprecated façade — use `uaibot.utils` directly.

    .. deprecated::
        Static methods have moved to module-level functions in
        `uaibot.utils.*`. This class will be removed in a future release.
    """

    # Intentionally empty. The metaclass delegates attribute lookup to the
    # `uaibot.utils` module namespace.
    pass


if __name__ != "uaibot.utils":
    raise ImportError(
        f"uaibot.utils was imported as {__name__!r}. "
        f"Fix the offending bare `import utils` to `from uaibot.utils import ...`."
    )

