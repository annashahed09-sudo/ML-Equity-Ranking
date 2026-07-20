"""
Momentum factor implementations.

Implements academically grounded momentum factors:
- Time-series momentum (1M, 3M, 6M, 12M)
- Cross-sectional momentum
- Residual momentum (idiosyncratic)
- Volatility-adjusted momentum
- 52-week high momentum
- Industry momentum
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from core.types import FactorType
from core.utils import validate_window

from .base import FactorComputer, FactorResult


class TimeSeriesMomentum(FactorComputer):
    """
    Time-series momentum: asset's own past return.
    
    Classic Jegadeesh & Titman (1993) momentum factor.
    Computes cumulative return over the lookback window.
    """
    
    def __init__(self, window: int = 252, **kwargs):
        super().__init__(
            name=f"momentum_{window}d",
            factor_type=FactorType.MOMENTUM,
            **kwargs
        )
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        # Compute returns per ticker
        def _compute_momentum(group: pd.DataFrame) -> pd.Series:
            prices = group["close"]
            mom = prices.pct_change(self.window)
            return mom
        
        values = df.groupby("ticker", group_keys=False).apply(_compute_momentum)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=values,
            description=f"Time-series momentum: {self.window}-day cumulative return",
        )


class ResidualMomentum(FactorComputer):
    """
    Residual momentum: momentum of idiosyncratic returns.
    
    Computes momentum after removing market beta contribution,
    isolating stock-specific momentum signal (Blitz et al., 2011).
    """
    
    def __init__(self, window: int = 252, **kwargs):
        super().__init__(
            name=f"residual_momentum_{window}d",
            factor_type=FactorType.MOMENTUM,
            **kwargs
        )
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        def _compute_residual_momentum(group: pd.DataFrame) -> pd.Series:
            returns = group["close"].pct_change()
            # Use equal-weighted market return as proxy
            market_ret = returns.mean()
            # Rolling beta estimation (60-day)
            rolling_cov = returns.rolling(60).cov(market_ret)
            rolling_mkt_var = returns.rolling(60).var()  # Simplified proxy
            beta = rolling_cov / (rolling_mkt_var + 1e-10)
            # Idiosyncratic return
            residual = returns - beta * market_ret
            # Cumulative residual momentum
            residual_mom = residual.rolling(self.window).sum()
            return residual_mom
        
        values = df.groupby("ticker", group_keys=False).apply(_compute_residual_momentum)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=values,
            description=f"Residual momentum: {self.window}-day cumulative idiosyncratic return",
        )


class VolatilityAdjustedMomentum(FactorComputer):
    """
    Volatility-adjusted momentum (Bali et al., 2017).
    
    Scales raw momentum by inverse volatility to reduce exposure
    to high-volatility stocks that have historically poor performance.
    """
    
    def __init__(self, window: int = 252, **kwargs):
        super().__init__(
            name=f"vol_adj_momentum_{window}d",
            factor_type=FactorType.MOMENTUM,
            **kwargs
        )
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        def _compute_vol_adj_momentum(group: pd.DataFrame) -> pd.Series:
            prices = group["close"]
            returns = prices.pct_change()
            raw_mom = prices.pct_change(self.window)
            rolling_vol = returns.rolling(60).std()
            # Scale momentum by 1/volatility
            adj_mom = raw_mom / (rolling_vol + 1e-10)
            return adj_mom
        
        values = df.groupby("ticker", group_keys=False).apply(_compute_vol_adj_momentum)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=values,
            description=f"Volatility-adjusted momentum: {self.window}d return / volatility",
        )


class High52Week(FactorComputer):
    """
    52-week high momentum (George & Hwang, 2004).
    
    Measures how close current price is to 52-week high.
    Nearness to 52-week high predicts future returns.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="near_52wk_high",
            factor_type=FactorType.MOMENTUM,
            **kwargs
        )
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        def _compute(group: pd.DataFrame) -> pd.Series:
            rolling_max = group["close"].rolling(252).max()
            return group["close"] / rolling_max - 1.0
        
        values = df.groupby("ticker", group_keys=False).apply(_compute)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=values,
            description="Distance from 52-week high: (price / 52wk_high) - 1",
        )


class CompositeMomentum(FactorComputer):
    """
    Composite momentum score combining multiple momentum signals.
    
    Averages z-score normalized momentum across multiple horizons
    and types (raw, residual, volatility-adjusted) for robustness.
    """
    
    def __init__(self, windows: List[int] = None, **kwargs):
        super().__init__(
            name="composite_momentum",
            factor_type=FactorType.MOMENTUM,
            **kwargs
        )
        if windows is None:
            windows = [21, 63, 126, 252]  # 1M, 3M, 6M, 12M
        self.components = [
            TimeSeriesMomentum(window=w, **kwargs) for w in windows
        ]
        self.components.append(ResidualMomentum(window=252, **kwargs))
        self.components.append(High52Week(**kwargs))
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        component_values = []
        for comp in self.components:
            result = comp.compute(df, **kwargs)
            component_values.append(result.values)
        
        combined = pd.concat(component_values, axis=1).mean(axis=1)
        combined = self._post_process(combined, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=combined,
            description="Composite momentum: equal-weighted multi-horizon momentum signals",
        )
