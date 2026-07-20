"""
Signal normalization utilities.

Standardizes raw signals into comparable units:
- Z-score normalization (cross-sectional)
- Rank transformation
- Quantile mapping
- Sigmoid scaling
- Robust scaling (median/MAD)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.utils import zscore_normalize, gaussian_rank_transform


class SignalNormalizer:
    """Collection of signal normalization methods."""
    
    @staticmethod
    def zscore(
        series: pd.Series,
        cross_sectional: bool = True,
        group_col: Optional[str] = None,
    ) -> pd.Series:
        """
        Z-score normalize a signal.
        
        If cross_sectional=True, normalizes per time step (grouped by date).
        """
        if cross_sectional and group_col is not None:
            from core.utils import cross_sectional_zscore
            return cross_sectional_zscore(
                series.to_frame("value").reset_index(),
                "value", group_col
            )
        return zscore_normalize(series)
    
    @staticmethod
    def rank(series: pd.Series, pct: bool = True) -> pd.Series:
        """Convert signal to (percentile) ranks."""
        ranks = series.rank(method="average", na_option="keep")
        if pct:
            return ranks / ranks.max()
        return ranks
    
    @staticmethod
    def gaussian_rank(series: pd.Series) -> pd.Series:
        """Gaussian rank transform (Van der Waerden)."""
        return gaussian_rank_transform(series)
    
    @staticmethod
    def quantile(series: pd.Series, n_quantiles: int = 5) -> pd.Series:
        """Map signal to quantile buckets."""
        return pd.qcut(series.rank(method="first"), n_quantiles, labels=False) + 1
    
    @staticmethod
    def sigmoid(series: pd.Series, scale: float = 1.0) -> pd.Series:
        """Map signal to (0, 1) using sigmoid function."""
        return 1.0 / (1.0 + np.exp(-series / scale))
    
    @staticmethod
    def robust(series: pd.Series) -> pd.Series:
        """Robust scaling using median and MAD."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad < 1e-12:
            return series * 0.0
        return (series - median) / (mad * 1.4826)
    
    @staticmethod
    def clip(series: pd.Series, limits: tuple[float, float] = (0.01, 0.99)) -> pd.Series:
        """Winsorize signal at specified quantiles."""
        lower = series.quantile(limits[0])
        upper = series.quantile(limits[1])
        return series.clip(lower, upper)
