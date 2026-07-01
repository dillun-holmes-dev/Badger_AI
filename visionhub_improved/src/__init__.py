"""Backward-compatible alias for checkpoints saved before the package rename.

Older checkpoints may contain pickled references to modules under ``src``.
The runtime package now lives under ``visionhub``, so this shim lets those
checkpoints unpickle without requiring the old source tree layout.
"""

from importlib import import_module
import sys

_visionhub = import_module("visionhub")
sys.modules[__name__] = _visionhub
