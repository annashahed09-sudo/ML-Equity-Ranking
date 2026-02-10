"""
Feature engineering for cross-sectional equity prediction.

Key principle: All features are standardized cross-sectionally (across the asset universe
at each time step), NOT temporally. This preserves the cross-sectional ranking structure
while preventing lookahead bias.

Feature naming convention:
  - _mom: momentum (price-based)
  - _vol: volatility
  - _trend: distance from trend
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler


def compute_rolling_momentum(
    df: pd.DataFrame,
    window: int = 20,
    column: str = 'close'
) -> pd.Series:
    """
    Compute rolling momentum (return) for each asset.

    Momentum[t] = (close[t] - close[t-window]) / close[t-window]

    Parameters
    ----------
    df : pd.DataFrame
        Data for a single asset (sorted by date), must have 'close' column
    window : int, default 20
        Lookback period in days
    column : str, default 'close'
        Column to compute momentum on

    Returns
    -------
    pd.Series
        Rolling momentum values
    """
    return df[column].pct_change(window).bfill()


def compute_rolling_volatility(
    df: pd.DataFrame,
    window: int = 20,
    column: str = 'close'
) -> pd.Series:
    """
    Compute rolling volatility (annualized standard deviation of returns).

    Parameters
    ----------
    df : pd.DataFrame
        Data for a single asset (sorted by date)
    window : int, default 20
        Lookback period in days
    column : str, default 'close'
        Column to compute returns on

    Returns
    -------
    pd.Series
        Rolling volatility (annualized)
    """
    returns = df[column].pct_change()
    volatility = returns.rolling(window).std() * np.sqrt(252)  # annualize
    return volatility


def compute_trend_distance(
    df: pd.DataFrame,
    window: int = 60,
    column: str = 'close'
) -> pd.Series:
    """
    Compute distance from trend (simple moving average).

    Distance[t] = (close[t] - SMA[t]) / SMA[t]

    Parameters
    ----------
    df : pd.DataFrame
        Data for a single asset (sorted by date)
    window : int, default 60
        SMA window
    column : str, default 'close'
        Column to compute trend distance on

    Returns
    -------
    pd.Series
        Normalized distance from trend
    """
    sma = df[column].rolling(window).mean()
    return (df[column] - sma) / sma


def cross_sectional_standardize(
    series: pd.Series,
    group_key: str = None,
    date_key: str = None
) -> pd.Series:
    """
    Standardize a feature cross-sectionally (per time step).

    Use this to standardize features within each date group, preserving
    cross-sectional relationships.

    Parameters
    ----------
    series : pd.Series
        Feature values (expected to have multi-index or groupby info)
    group_key : str, optional
        Key for grouping (e.g., 'date' if series is indexed by date/ticker)
    date_key : str, optional
        Alternative: date column name if series is part of a DataFrame

    Returns
    -------
    pd.Series
        Standardized feature values
    """
    if group_key:
        return series.groupby(level=group_key, group_keys=False).apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    else:
        return (series - series.mean()) / (series.std() + 1e-8)


def compute_features(
    df: pd.DataFrame,
    momentum_windows: List[int] = None,
    volatility_window: int = 20,
    trend_window: int = 60,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Compute all feature engineering for a full OHLCV dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: date, ticker, open, high, low, close, volume
        Must be sorted by (date, ticker)
    momentum_windows : List[int], optional
        List of momentum lookback windows. Default: [5, 20]
    volatility_window : int, default 20
        Volatility rolling window
    trend_window : int, default 60
        Trend (SMA) window
    normalize : bool, default True
        If True, apply cross-sectional standardization to each feature

    Returns
    -------
    pd.DataFrame
        Original data + feature columns (ending in _mom, _vol, _trend, etc.)
        Rows with NaN features are retained (for visibility; filter as needed)
    """
    if momentum_windows is None:
        momentum_windows = [5, 20]
    
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    result = df.copy()
    
    # Compute features per asset
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        asset_data = df[mask].copy()
        
        # Momentum features
        for window in momentum_windows:
            col_name = f'momentum_{window}_mom'
            result.loc[mask, col_name] = compute_rolling_momentum(asset_data, window=window).values
        
        # Volatility feature
        result.loc[mask, 'volatility_vol'] = compute_rolling_volatility(
            asset_data, window=volatility_window
        ).values
        
        # Trend distance feature
        result.loc[mask, 'trend_distance_trend'] = compute_trend_distance(
            asset_data, window=trend_window
        ).values
    
    # Cross-sectional standardization
    if normalize:
        feature_cols = [col for col in result.columns if any(
            col.endswith(suffix) for suffix in ['_mom', '_vol', '_trend']
        )]
        
        for col in feature_cols:
            result[col] = result.groupby('date')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
    
    return result


def compute_forward_returns(
    df: pd.DataFrame,
    forward_periods: int = 1,
    column: str = 'close'
) -> pd.DataFrame:
    """
    Compute next-period log returns for ranking (target variable).

    forward_return[t] = log(close[t+forward_periods] / close[t])

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame
    forward_periods : int, default 1
        Number of periods ahead to compute returns
    column : str, default 'close'
        Price column

    Returns
    -------
    pd.DataFrame
        Original data with 'forward_return' column added
    """
    result = df.copy()
    result['forward_return'] = result.groupby('ticker')[column].apply(
        lambda x: np.log(x.shift(-forward_periods) / x)
    ).values
    
    return result


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of all feature columns (end with _mom, _vol, _trend)."""
    return [col for col in df.columns if any(
        col.endswith(suffix) for suffix in ['_mom', '_vol', '_trend']
    )]
