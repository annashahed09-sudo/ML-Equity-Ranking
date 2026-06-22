"""
Advanced accelerated numeric utilities.

This module offers:
- Backend selection (NumPy/CuPy)
- Optional Numba JIT acceleration for CPU-heavy routines
- Optional CUDA kernels (when numba.cuda is available)
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, util
from typing import Any, Tuple

import numpy as np


def _identity_njit(*args, **kwargs):
    """Fallback decorator with the same call pattern as numba.njit."""
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def wrapper(func):
        return func

    return wrapper


_numba_module = import_module("numba") if util.find_spec("numba") else None
njit = getattr(_numba_module, "njit", _identity_njit)
cuda = getattr(_numba_module, "cuda", None)
HAS_NUMBA_CUDA = cuda is not None


@dataclass
class ArrayBackend:
    """Wrapper around NumPy/CuPy style APIs."""

    name: str
    xp: Any

    def asnumpy(self, arr: Any) -> np.ndarray:
        if self.name == "cupy":
            return self.xp.asnumpy(arr)
        return np.asarray(arr)


def get_array_backend(prefer_gpu: bool = True) -> ArrayBackend:
    """Resolve NumPy or CuPy backend."""
    if prefer_gpu and util.find_spec("cupy"):
        cupy_module = import_module("cupy")
        return ArrayBackend(name="cupy", xp=cupy_module)
    return ArrayBackend(name="numpy", xp=np)


@njit(cache=True)
def _normalize_numba(arr: np.ndarray) -> np.ndarray:
    mean = 0.0
    n = arr.shape[0]
    for i in range(n):
        mean += arr[i]
    mean /= max(n, 1)

    var = 0.0
    for i in range(n):
        d = arr[i] - mean
        var += d * d
    std = np.sqrt(var / max(n, 1))

    out = np.empty_like(arr)
    denom = std + 1e-8
    for i in range(n):
        out[i] = (arr[i] - mean) / denom
    return out


if HAS_NUMBA_CUDA:
    @cuda.jit
    def _normalize_cuda_kernel(inp, mean, std, out):
        i = cuda.grid(1)
        if i < inp.size:
            out[i] = (inp[i] - mean) / (std + 1e-8)


def cuda_available() -> bool:
    """Check whether Numba CUDA is available at runtime."""
    return bool(HAS_NUMBA_CUDA and cuda.is_available())


def normalize_scores(scores: Any, prefer_gpu: bool = True, prefer_numba: bool = True) -> np.ndarray:
    """
    Normalize scores with optional CuPy and Numba acceleration.

    Order of preference:
    1) CuPy (if prefer_gpu)
    2) Numba JIT CPU path (if prefer_numba)
    3) NumPy fallback
    """
    arr = np.asarray(scores, dtype=np.float64)

    backend = get_array_backend(prefer_gpu=prefer_gpu)
    if backend.name == "cupy":
        xp = backend.xp
        gpu_arr = xp.asarray(arr)
        gpu_out = (gpu_arr - gpu_arr.mean()) / (gpu_arr.std() + 1e-8)
        return backend.asnumpy(gpu_out)

    if prefer_numba and _numba_module is not None and arr.ndim == 1 and arr.size > 0:
        return _normalize_numba(arr)

    return (arr - arr.mean()) / (arr.std() + 1e-8)


def normalize_scores_cuda(scores: Any) -> Tuple[np.ndarray, bool]:
    """
    Attempt explicit CUDA-kernel score normalization.

    Returns
    -------
    Tuple[np.ndarray, bool]
        (normalized_array, used_cuda)
    """
    arr = np.asarray(scores, dtype=np.float32)
    if not cuda_available() or arr.ndim != 1:
        out = (arr - arr.mean()) / (arr.std() + 1e-8)
        return out, False

    d_in = cuda.to_device(arr)
    d_out = cuda.device_array_like(d_in)
    mean = arr.mean()
    std = arr.std()

    threads_per_block = 128
    blocks = (arr.size + threads_per_block - 1) // threads_per_block
    _normalize_cuda_kernel[blocks, threads_per_block](d_in, mean, std, d_out)

    return d_out.copy_to_host(), True


@njit(cache=True)
def log_returns_numba(prices: np.ndarray) -> np.ndarray:
    """Compute 1-step log returns with Numba acceleration."""
    n = prices.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = np.nan
    for i in range(1, n):
        out[i] = np.log(prices[i] / prices[i - 1])
    return out
