"""
Volatility and risk factor implementations.

Implements:
- Realized volatility (various windows)
- Downside deviation
- Beta (CAPM)
- Idiosyncratic volatility
- Maximum drawdown
- Value at Risk
- Skewness and kurtosis
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import FactorType
from core.utils import validate_window
from .base import FactorComputer, FactorResult


class RealizedVolatility(FactorComputer):
    """Realized volatility: rolling standard deviation of returns."""
    
    def __init__(self, window: int = 60, annualize: bool = True, **kwargs):
        super().__init__(name=f"realized_vol_{window}d", factor_type=FactorType.VOLATILITY, **kwargs)
        self.window = validate_window(window)
        self.annualize = annualize
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        def _compute(group: pd.DataFrame) -> pd.Series:
            returns = group["close"].pct_change()
            vol = returns.rolling(self.window).std()
            if self.annualize:
                vol = vol * np.sqrt(252)
            return vol
        
        values = df.groupby("ticker", group_keys=False).apply(_compute)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"Realized volatility: {self.window}-day rolling std * sqrt(252)")


class DownsideDeviation(FactorComputer):
    """Downside deviation: volatility of negative returns only."""
    
    def __init__(self, window: int = 60, **kwargs):
        super().__init__(name=f"downside_dev_{window}d", factor_type=FactorType.VOLATILITY, **kwargs)
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        def _compute(group: pd.DataFrame) -> pd.Series:
            returns = group["close"].pct_change()
            downside = returns.where(returns < 0, 0)
            dev = downside.rolling(self.window).std() * np.sqrt(252)
            return dev
        
        values = df.groupby("ticker", group_keys=False).apply(_compute)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"Downside deviation: {self.window}-day rolling std of negative returns")


class MarketBeta(FactorComputer):
    """Market beta: rolling CAPM beta relative to equal-weighted market."""
    
    def __init__(self, window: int = 252, **kwargs):
        super().__init__(name=f"beta_{window}d", factor_type=FactorType.VOLATILITY, **kwargs)
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        # Compute market returns (equal-weighted)
        returns = df.groupby("ticker")["close"].transform(lambda x: x.pct_change())
        market_ret = returns.groupby(df["date"]).transform("mean")
        
        def _compute_beta(group: pd.DataFrame) -> pd.Series:
            g_ret = group["close"].pct_change()
            m_ret = market_ret.loc[group.index]
            cov = g_ret.rolling(self.window).cov(m_ret)
            mkt_var = m_ret.rolling(self.window).var()
            beta = cov / (mkt_var + 1e-10)
            return beta
        
        values = df.groupby("ticker", group_keys=False).apply(_compute_beta)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"Market beta: {self.window}-day rolling CAPM beta")


class IdiosyncraticVolatility(FactorComputer):
    """Idiosyncratic volatility: volatility of CAPM residual returns."""
    
    def __init__(self, window: int = 60, **kwargs):
        super().__init__(name=f"idio_vol_{window}d", factor_type=FactorType.VOLATILITY, **kwargs)
        self.window = validate_window(window)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        returns = df.groupby("ticker")["close"].transform(lambda x: x.pct_change())
        market_ret = returns.groupby(df["date"]).transform("mean")
        
        def _compute(group: pd.DataFrame) -> pd.Series:
            g_ret = group["close"].pct_change()
            m_ret = market_ret.loc[group.index]
            beta = g_ret.rolling(60).cov(m_ret) / (m_ret.rolling(60).var() + 1e-10)
            residual = g_ret - beta * m_ret
            idio_vol = residual.rolling(self.window).std() * np.sqrt(252)
            return idio_vol
        
        values = df.groupby("ticker", group_keys=False).apply(_compute)
        values = self._post_process(values, df)
        
        return FactorResult.from_series(self.name, self.factor_type, values,
            f"Idiosyncratic volatility: {self.window}-day residual vol")
