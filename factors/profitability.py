"""
Profitability factor implementations.

Implements:
- Gross Margin
- Operating Margin
- Net Margin
- Return on Assets (ROA)
- Asset Turnover
- Capital Intensity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import FactorType
from .base import FactorComputer, FactorResult


class GrossMargin(FactorComputer):
    """Gross Margin: (revenue - COGS) / revenue."""
    
    def __init__(self, **kwargs):
        super().__init__(name="gross_margin", factor_type=FactorType.PROFITABILITY, **kwargs)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        if "gross_profit" in df.columns and "revenue" in df.columns:
            values = df["gross_profit"] / df["revenue"].clip(lower=1e-6)
        else:
            values = pd.Series(np.nan, index=df.index)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            "Gross Margin: gross profit divided by revenue")


class OperatingMargin(FactorComputer):
    """Operating Margin: operating income / revenue."""
    
    def __init__(self, **kwargs):
        super().__init__(name="operating_margin", factor_type=FactorType.PROFITABILITY, **kwargs)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        if "operating_income" in df.columns and "revenue" in df.columns:
            values = df["operating_income"] / df["revenue"].clip(lower=1e-6)
        else:
            values = pd.Series(np.nan, index=df.index)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            "Operating Margin: operating income divided by revenue")


class NetMargin(FactorComputer):
    """Net Margin: net income / revenue."""
    
    def __init__(self, **kwargs):
        super().__init__(name="net_margin", factor_type=FactorType.PROFITABILITY, **kwargs)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        if "net_income" in df.columns and "revenue" in df.columns:
            values = df["net_income"] / df["revenue"].clip(lower=1e-6)
        else:
            values = pd.Series(np.nan, index=df.index)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            "Net Margin: net income divided by revenue")


class ReturnOnAssets(FactorComputer):
    """Return on Assets: net income / total assets."""
    
    def __init__(self, **kwargs):
        super().__init__(name="roa", factor_type=FactorType.PROFITABILITY, **kwargs)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        if "net_income" in df.columns and "total_assets" in df.columns:
            values = df["net_income"] / df["total_assets"].clip(lower=1e-6)
        else:
            values = pd.Series(np.nan, index=df.index)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            "Return on Assets: net income divided by total assets")


class CompositeProfitability(FactorComputer):
    """Composite profitability score."""
    
    def __init__(self, **kwargs):
        super().__init__(name="composite_profitability", factor_type=FactorType.PROFITABILITY, **kwargs)
        self.components = [
            GrossMargin(**kwargs),
            OperatingMargin(**kwargs),
            ReturnOnAssets(**kwargs),
        ]
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        vals = [c.compute(df, **kwargs).values for c in self.components]
        combined = pd.concat(vals, axis=1).mean(axis=1)
        combined = self._post_process(combined, df)
        return FactorResult.from_series(self.name, self.factor_type, combined,
            "Composite profitability: equal-weighted gross/operating margins and ROA")
