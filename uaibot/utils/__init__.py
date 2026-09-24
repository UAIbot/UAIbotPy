from .utils import *

if __name__ != "uaibot.utils":
    raise ImportError(
        f"uaibot.utils was imported as {__name__!r}. "
        f"Fix the offending bare `import utils` to `from uaibot.utils import ...`."
    )
