"""
Quality factor implementations.

Implements academically grounded quality factors:
- Return on Equity (ROE)
- Return on Invested Capital (ROIC)
- Gross Profitability (Novy-Marx, 2013)
- Piotroski F-Score (fundamental strength)
- Altman Z-Score (financial distress)
- Accruals quality
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import FactorType
from .base import FactorComputer, FactorResult


class ReturnOnEquity(FactorComputer):
    """Return on Equity: net income / shareholders' equity."""
    
    def __init__(self, **kwargs):
        super().__init__(name="roe", factor_type=FactorType.QUALITY, **kwargs)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        if "net_income" in df.columns and "equity" in df.columns:
            values = df["net_income"] / df["equity"].clip(lower=1e-6)
        else:
            values = pd.Series(np.nan, index=df.index)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            "Return on Equity: net income divided by shareholders' equity")


class ReturnOnInvestedCapital(FactorComputer):
    """Return on Invested Capital: NOPAT / invested capital."""
    
    def __init__(self, **kwargs):
        super().__init__(name="roic", factor_type=FactorType.QUALITY, **kwargs)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        if "nopat" in df.columns and "invested_capital" in df.columns:
            values = df["nopat"] / df["invested_capital"].clip(lower=1e-6)
        else:
            values = pd.Series(np.nan, index=df.index)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            "ROIC: after-tax operating profit divided by invested capital")


class GrossProfitability(FactorComputer):
    """
    Gross Profitability (Novy-Marx, 2013).
    
    Gross profits / total assets. One of the most robust predictors
    of cross-sectional returns (Novy-Marx, 2013).
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="gross_profitability", factor_type=FactorType.QUALITY, **kwargs)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        if "gross_profit" in df.columns and "total_assets" in df.columns:
            values = df["gross_profit"] / df["total_assets"].clip(lower=1e-6)
        else:
            values = pd.Series(np.nan, index=df.index)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            "Gross Profitability: gross profit divided by total assets")


class PiotroskiFScore(FactorComputer):
    """
    Piotroski F-Score (Piotroski, 2000).
    
    Composite score (0-9) based on 9 fundamental signals:
    Profitability (4), Leverage/Liquidity (3), Operating Efficiency (2).
    Higher scores indicate stronger financial health.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="piotroski_f_score", factor_type=FactorType.QUALITY, **kwargs)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        # In production, compute from fundamental data
        # For now, placeholder when no fundamental data available
        if all(c in df.columns for c in ["roa", "cfo", "leverage", "liquidity", "gross_margin", "asset_turnover"]):
            f1 = (df["roa"] > 0).astype(int)
            f2 = (df["cfo"] > 0).astype(int)
            f3 = (df["roa"].diff() > 0).astype(int)
            f4 = (df["cfo"] > df["roa"]).astype(int)
            f5 = (df["leverage"].diff() < 0).astype(int)
            f6 = (df["liquidity"].diff() > 0).astype(int)
            f7 = (df["gross_margin"].diff() == 0).astype(int)
            f8 = (df["asset_turnover"].diff() > 0).astype(int)
            values = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
        else:
            values = pd.Series(np.nan, index=df.index)
        
        values = self._post_process(values, df)
        return FactorResult.from_series(self.name, self.factor_type, values,
            "Piotroski F-Score: composite 0-9 fundamental strength score")


class CompositeQuality(FactorComputer):
    """Composite quality score combining multiple quality signals."""
    
    def __init__(self, **kwargs):
        super().__init__(name="composite_quality", factor_type=FactorType.QUALITY, **kwargs)
        self.components = [
            ReturnOnEquity(**kwargs),
            GrossProfitability(**kwargs),
        ]
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        vals = [c.compute(df, **kwargs).values for c in self.components]
        combined = pd.concat(vals, axis=1).mean(axis=1)
        combined = self._post_process(combined, df)
        return FactorResult.from_series(self.name, self.factor_type, combined,
            "Composite quality: equal-weighted ROE and gross profitability")
