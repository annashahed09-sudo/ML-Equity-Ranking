"""
Signal processing and combination engine.

Handles:
- Signal normalization and calibration
- Signal combination (weighted, rank-based, ML-based)
- Signal orthogonalization (Gram-Schmidt, PCA-based)
- Cross-sectional signal processing
- Time-series signal smoothing
"""

from .combination import SignalCombiner
from .normalization import SignalNormalizer
from .orthogonalization import SignalOrthogonalizer

__all__ = [
    "SignalCombiner",
    "SignalNormalizer",
    "SignalOrthogonalizer",
]
