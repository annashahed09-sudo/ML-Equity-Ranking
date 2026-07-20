"""
Liquidity factor implementations.

Implements:
- Average Daily Volume (ADV)
- Turnover ratio
- Amihud Illiquidity (price impact per dollar volume)
- Dollar volume
- Bid-Ask spread proxy
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import FactorType
from core.utils import validate_window
from .base import FactorComputer, FactorResult


class AverageDailyVolume(FactorComputer):
    """Average Daily Volume: rolling mean of daily volume."""
    
    def __init__(self, window: int = 60, **kwargs):
        super().__init__(name=f"adv_{window}d", factor_type=FactorType.LIQUIDITY, **kwargs)
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "volume"])
        
        def _compute(group: pd.DataFrame) -> pd.Series:
            return group["volume"].rolling(self.window).mean()
        
        values = df.groupby("ticker", group_keys=False).apply(_compute)
        # Log transform for normality
        values = np.log1p(values)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"Average Daily Volume: {self.window}-day rolling mean (log-scaled)")


class TurnoverRatio(FactorComputer):
    """Turnover ratio: volume relative to shares outstanding."""
    
    def __init__(self, window: int = 60, **kwargs):
        super().__init__(name=f"turnover_{window}d", factor_type=FactorType.LIQUIDITY, **kwargs)
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "volume"])
        
        if "shares_outstanding" in df.columns:
            def _compute(group: pd.DataFrame) -> pd.Series:
                turnover = group["volume"] / group["shares_outstanding"].clip(lower=1)
                return turnover.rolling(self.window).mean()
            values = df.groupby("ticker", group_keys=False).apply(_compute)
        else:
            # Proxy: volume / volume mean
            def _compute(group: pd.DataFrame) -> pd.Series:
                vol_mean = group["volume"].rolling(self.window).mean()
                return group["volume"] / (vol_mean + 1e-6)
            values = df.groupby("ticker", group_keys=False).apply(_compute)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"Turnover ratio: {self.window}-day rolling mean turnover")


class AmihudIlliquidity(FactorComputer):
    """
    Amihud Illiquidity Ratio (Amihud, 2002).
    
    Measures price impact per dollar of volume. Higher values = more illiquid.
    Computed as: |return| / dollar_volume
    
    One of the most widely used microstructure-based illiquidity measures.
    """
    
    def __init__(self, window: int = 60, **kwargs):
        super().__init__(name=f"amihud_{window}d", factor_type=FactorType.LIQUIDITY, **kwargs)
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close", "volume"])
        
        def _compute(group: pd.DataFrame) -> pd.Series:
            returns = group["close"].pct_change().abs()
            dollar_vol = group["close"] * group["volume"]
            illiquidity = returns / (dollar_vol + 1e-12)
            # Winsorize and smooth
            illiquidity = illiquidity.clip(upper=illiquidity.quantile(0.99))
            return illiquidity.rolling(self.window).mean()
        
        values = df.groupby("ticker", group_keys=False).apply(_compute)
        # Higher illiquidity is "worse" so we negate for consistency
        values = -np.log1p(values)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"Amihud Illiquidity: {self.window}-day rolling price impact (negated for liquidity)")


class DollarVolume(FactorComputer):
    """Dollar volume: price × volume trend."""
    
    def __init__(self, window: int = 60, **kwargs):
        super().__init__(name=f"dollar_vol_{window}d", factor_type=FactorType.LIQUIDITY, **kwargs)
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close", "volume"])
        
        def _compute(group: pd.DataFrame) -> pd.Series:
            dollar_vol = group["close"] * group["volume"]
            return np.log(dollar_vol.rolling(self.window).mean())
        
        values = df.groupby("ticker", group_keys=False).apply(_compute)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"Dollar volume: {self.window}-day rolling mean (log-scaled)")
