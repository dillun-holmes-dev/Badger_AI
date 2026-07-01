"""VisionHub training, inference, and deployment package."""

try:
    from .core import LazyConfig, LazyCall, instantiate
except ImportError:
    # Keep lightweight metadata imports working when optional ML deps are absent.
    LazyConfig = None
    LazyCall = None
    instantiate = None

__all__ = ["LazyConfig", "LazyCall", "instantiate"]
__version__ = "0.1.0"
