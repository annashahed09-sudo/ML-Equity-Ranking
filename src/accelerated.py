"""
Utilities for accelerated numeric computing.

This module provides a lightweight abstraction that can run on NumPy by default,
and optionally switches to CuPy if a CUDA-capable environment is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class ArrayBackend:
    """Simple wrapper over NumPy/CuPy style APIs."""

    name: str
    xp: Any

    def asnumpy(self, arr: Any) -> np.ndarray:
        """Convert backend array to NumPy array."""
        if self.name == "cupy":
            return self.xp.asnumpy(arr)
        return np.asarray(arr)


def get_array_backend(prefer_gpu: bool = True) -> ArrayBackend:
    """
    Resolve an array backend.

    Parameters
    ----------
    prefer_gpu : bool
        If True, try to import/use CuPy first. Falls back to NumPy.

    Returns
    -------
    ArrayBackend
        Backend metadata and array module.
    """
    if prefer_gpu:
        try:
            import cupy as cp  # type: ignore

            _ = cp.zeros(1)
            return ArrayBackend(name="cupy", xp=cp)
        except Exception:
            pass

    return ArrayBackend(name="numpy", xp=np)


def normalize_scores(scores: Any, prefer_gpu: bool = True) -> np.ndarray:
    """Normalize prediction scores with optional accelerated arrays."""
    backend = get_array_backend(prefer_gpu=prefer_gpu)
    xp = backend.xp

    arr = xp.asarray(scores)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)

    return backend.asnumpy(arr)
