try:
    import torch
except ImportError as exc:
    raise ImportError(
        "uaibot.gpu requires PyTorch. Install it with `pip install torch`."
    ) from exc
