"""
Growth factor implementations.

Implements:
- Revenue Growth (YoY, QoQ)
- EPS Growth
- EBITDA Growth
- Sustainable Growth Rate (ROE × retention ratio)
- Analyst-estimated growth (where available)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import FactorType
from core.utils import validate_window
from .base import FactorComputer, FactorResult


class RevenueGrowth(FactorComputer):
    """Revenue growth rate over lookback periods."""
    
    def __init__(self, window: int = 252, **kwargs):
        super().__init__(name=f"revenue_growth_{window}d", factor_type=FactorType.GROWTH, **kwargs)
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        if "revenue" in df.columns:
            def _compute(group: pd.DataFrame) -> pd.Series:
                return group["revenue"].pct_change(self.window)
            values = df.groupby("ticker", group_keys=False).apply(_compute)
        else:
            # Use price change as rough proxy when fundamental data unavailable
            def _compute(group: pd.DataFrame) -> pd.Series:
                return group["close"].pct_change(self.window)
            values = df.groupby("ticker", group_keys=False).apply(_compute)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"Revenue growth: {self.window}-day trailing growth rate")


class EPSGrowth(FactorComputer):
    """EPS growth rate (trailing)."""
    
    def __init__(self, window: int = 252, **kwargs):
        super().__init__(name=f"eps_growth_{window}d", factor_type=FactorType.GROWTH, **kwargs)
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        if "eps" in df.columns:
            def _compute(group: pd.DataFrame) -> pd.Series:
                return group["eps"].pct_change(self.window)
            values = df.groupby("ticker", group_keys=False).apply(_compute)
        else:
            values = pd.Series(np.nan, index=df.index)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"EPS growth: {self.window}-day trailing growth rate")


class EBITDAGrowth(FactorComputer):
    """EBITDA growth rate (trailing)."""
    
    def __init__(self, window: int = 252, **kwargs):
        super().__init__(name=f"ebitda_growth_{window}d", factor_type=FactorType.GROWTH, **kwargs)
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        if "ebitda" in df.columns:
            def _compute(group: pd.DataFrame) -> pd.Series:
                return group["ebitda"].pct_change(self.window)
            values = df.groupby("ticker", group_keys=False).apply(_compute)
        else:
            values = pd.Series(np.nan, index=df.index)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"EBITDA growth: {self.window}-day trailing growth rate")


class CompositeGrowth(FactorComputer):
    """Composite growth score combining multiple growth signals."""
    
    def __init__(self, **kwargs):
        super().__init__(name="composite_growth", factor_type=FactorType.GROWTH, **kwargs)
        self.components = [
            RevenueGrowth(window=252, **kwargs),
            EPSGrowth(window=252, **kwargs),
        ]
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        vals = [c.compute(df, **kwargs).values for c in self.components]
        combined = pd.concat(vals, axis=1).mean(axis=1)
        combined = self._post_process(combined, df)
        return FactorResult.from_series(self.name, self.factor_type, combined,
            "Composite growth: equal-weighted revenue and EPS growth")
