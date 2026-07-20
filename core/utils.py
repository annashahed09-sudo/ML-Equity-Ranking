"""
Mathematical and software engineering utilities for quantitative research.

Provides statistically sound helper functions for array operations,
validation, parallel processing, and numerical stability.
"""

from __future__ import annotations

import functools
import logging
import multiprocessing
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterator, List, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


# ── Array & DataFrame Utilities ───────────────────────────────────────────

def ensure_array(x: Any, dtype: type = np.float64) -> np.ndarray:
    """Safely convert to numpy array with specified dtype."""
    arr = np.asarray(x, dtype=dtype)
    if np.any(np.isinf(arr)):
        logger.warning(f"Infinity values detected in array of shape {arr.shape}")
        arr = np.where(np.isinf(arr), np.nan, arr)
    return arr


def ensure_dataframe(x: Any, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Safely convert to DataFrame with optional column validation."""
    if isinstance(x, pd.DataFrame):
        df = x.copy()
    elif isinstance(x, pd.Series):
        df = x.to_frame()
    elif isinstance(x, np.ndarray):
        if columns is None and x.ndim == 2:
            columns = [f"col_{i}" for i in range(x.shape[1])]
        df = pd.DataFrame(x, columns=columns)
    else:
        df = pd.DataFrame(x)
    return df


def as_numpy(x: Any) -> np.ndarray:
    """Convert to numpy array, handling pandas objects gracefully."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    return np.asarray(x)


# ── Validation Utilities ──────────────────────────────────────────────────

def validate_probability(p: float, name: str = "probability") -> float:
    """Validate that a value is in [0, 1]."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")
    return p


def validate_positive(value: float, name: str = "value") -> float:
    """Validate that a value is strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_nonnegative(value: float, name: str = "value") -> float:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_window(window: int, name: str = "window") -> int:
    """Validate that a window parameter is a positive integer."""
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"{name} must be a positive integer, got {window}")
    return window


def validate_weights(weights: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    """Validate that weights sum to approximately 1.0."""
    w = np.asarray(weights, dtype=np.float64)
    total = np.sum(w)
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Weights must sum to 1.0, sum={total:.6f}")
    return w


# ── Statistical Utilities ─────────────────────────────────────────────────

def winsorize_series(
    series: pd.Series,
    limits: tuple[float, float] = (0.01, 0.99)
) -> pd.Series:
    """
    Winsorize a series by clipping extreme values at specified quantiles.
    
    Uses a robust quantile estimator to handle fat-tailed distributions
    common in financial data.
    """
    lower = series.quantile(limits[0], interpolation="higher")
    upper = series.quantile(limits[1], interpolation="lower")
    return series.clip(lower=lower, upper=upper)


def zscore_normalize(
    series: pd.Series,
    method: str = "standard",
    ddof: int = 1
) -> pd.Series:
    """
    Z-score normalize a series with numerical stability.
    
    Parameters
    ----------
    series : pd.Series
        Input series to normalize
    method : str
        'standard': (x - mean) / std
        'robust': (x - median) / mad
        'gaussian_rank': Gaussian rank transform
    ddof : int
        Delta degrees of freedom for std computation
    """
    valid = series.dropna()
    if len(valid) < 2:
        return series * np.nan

    if method == "standard":
        mean = valid.mean()
        std = valid.std(ddof=ddof)
        if std < 1e-12:
            return series * 0.0
        normalized = (series - mean) / std
        
    elif method == "robust":
        median = valid.median()
        mad = np.median(np.abs(valid - median))
        if mad < 1e-12:
            return series * 0.0
        normalized = (series - median) / (mad * 1.4826)  # Consistent estimator
        
    elif method == "gaussian_rank":
        # Van der Waerden rank transform
        ranks = valid.rank(method="average")
        normalized_values = scipy_stats.norm.ppf(
            (ranks - 0.5) / len(ranks)
        )
        normalized = series.copy()
        normalized[valid.index] = normalized_values
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.replace([np.inf, -np.inf], np.nan)


def cross_sectional_zscore(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "date",
    method: str = "standard",
    min_obs: int = 5
) -> pd.Series:
    """
    Compute cross-sectional z-scores per time step.
    
    For each group (typically date), computes z-scores across all assets,
    ensuring the cross-sectional ranking structure is preserved.
    """
    if df[value_col].isnull().all():
        return pd.Series(np.nan, index=df.index)

    def _zscore_group(group: pd.DataFrame) -> pd.Series:
        if len(group) < min_obs:
            return pd.Series(np.nan, index=group.index)
        return zscore_normalize(group[value_col], method=method)

    return df.groupby(group_col, group_keys=False).apply(_zscore_group)


def rank_transform(series: pd.Series, pct: bool = True) -> pd.Series:
    """Transform values to ranks or percentile ranks."""
    ranks = series.rank(method="average", na_option="keep")
    if pct:
        return ranks / ranks.max()
    return ranks


def gaussian_rank_transform(series: pd.Series) -> pd.Series:
    """
    Gaussian (normal) rank transform for cross-sectional normalization.
    
    Maps ranks to quantiles of the standard normal distribution,
    producing approximately normally distributed features.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return series * np.nan
    
    ranks = valid.rank(method="average")
    quantiles = (ranks - 0.5) / len(ranks)
    # Clip to avoid infinite z-scores at extremes
    quantiles = np.clip(quantiles, 1e-15, 1 - 1e-15)
    transformed = scipy_stats.norm.ppf(quantiles)
    
    result = series.copy()
    result[valid.index] = transformed
    return result


# ── Timing & Performance ──────────────────────────────────────────────────

@contextmanager
def timing(name: str = "block") -> Generator[None, None, None]:
    """Context manager for timing code execution."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{name} completed in {elapsed:.3f}s")


def timeit(func: Callable) -> Callable:
    """Decorator for timing function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


# ── Parallel Processing ───────────────────────────────────────────────────

def parallel_map(
    func: Callable[[T], U],
    items: Sequence[T],
    n_jobs: int = -1,
    chunk_size: int = 1,
    desc: str = "Processing"
) -> List[U]:
    """
    Parallel map with progress tracking.
    
    Parameters
    ----------
    func : Callable
        Function to apply to each item
    items : Sequence
        Items to process
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    chunk_size : int
        Chunk size for load balancing
    desc : str
        Description for logging
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    if n_jobs == 1 or len(items) <= 1:
        return [func(item) for item in items]

    results: List[U] = []
    with multiprocessing.Pool(processes=n_jobs) as pool:
        async_results = []
        for item in items:
            async_results.append(pool.apply_async(func, (item,)))
        for i, ar in enumerate(async_results):
            try:
                results.append(ar.get(timeout=3600))
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                results.append(None)  # type: ignore
    return results


def batch_iterator(
    items: Sequence[T],
    batch_size: int = 100
) -> Iterator[List[T]]:
    """Yield successive batches from a sequence."""
    for i in range(0, len(items), batch_size):
        yield list(items[i:i + batch_size])


# ── Numerical Utilities ───────────────────────────────────────────────────

def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax computation."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def entropy(probabilities: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0]  # Remove zero probabilities
    return -np.sum(p * np.log(p))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence between two distributions."""
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance (symmetric, bounded [0, 1])."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return np.sqrt(0.5 * (kl_divergence(p, m) + kl_divergence(q, m)))
