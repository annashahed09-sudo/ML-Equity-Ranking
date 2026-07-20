"""
Feature engineering utilities for the research pipeline.

Provides forward return computation and feature selection.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def compute_forward_returns(
    df: pd.DataFrame,
    forward_periods: int = 1,
    price_col: str = "close",
    method: str = "log",
) -> pd.DataFrame:
    """
    Compute forward (next-period) returns for ranking targets.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with date, ticker columns
    forward_periods : int
        Number of periods ahead to compute returns
    price_col : str
        Price column to use
    method : str
        'log' for log returns, 'simple' for simple returns
    
    Returns
    -------
    pd.DataFrame
        Original data with 'forward_return' column added
    """
    result = df.copy()
    
    def _forward_return(group: pd.DataFrame) -> pd.Series:
        prices = group[price_col].values
        shifted = np.roll(prices, -forward_periods)
        shifted[-forward_periods:] = np.nan
        
        if method == "log":
            fwd_returns = np.log(shifted / prices)
        else:
            fwd_returns = shifted / prices - 1
        
        return pd.Series(fwd_returns, index=group.index)
    
    result["forward_return"] = df.groupby("ticker", group_keys=False).apply(
        _forward_return
    )
    
    return result


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic feature set for research pipeline.
    
    This is a thin wrapper that calls the factor factory.
    For full factor computation, use factors.factory.compute_all_factors.
    """
    from factors.factory import compute_all_factors
    
    factor_result = compute_all_factors(df)
    
    result = df.copy()
    for col in factor_result.factor_values.columns:
        result[col] = factor_result.factor_values[col]
    
    return result
