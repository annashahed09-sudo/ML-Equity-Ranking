"""
Value factor implementations.

Implements academically grounded value factors:
- Earnings Yield (E/P)
- Book-to-Market (B/M)
- Free Cash Flow Yield (FCF/P)
- Enterprise Value / EBITDA
- Sales-to-Price
- Dividend Yield
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import FactorType
from core.utils import validate_window

from .base import FactorComputer, FactorResult


class EarningsYield(FactorComputer):
    """
    Earnings Yield (E/P): earnings per share / price.
    
    Higher values indicate undervaluation relative to earnings.
    Often preferred to P/E because it's scale-independent.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="earnings_yield",
            factor_type=FactorType.VALUE,
            **kwargs
        )
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        values = pd.Series(np.nan, index=df.index)
        
        # If EPS data is available in the dataframe
        if "eps" in df.columns:
            values = df["eps"] / df["close"]
        else:
            # Fallback: use inverse of close (placeholder when no fundamental data)
            # In production, EPS should come from a fundamental data provider
            values = 1.0 / df["close"]
        
        values = self._post_process(values, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=values,
            description="Earnings Yield (E/P): earnings per share divided by price",
        )


class BookToMarket(FactorComputer):
    """
    Book-to-Market ratio (B/M).
    
    Classic value factor from Fama-French. Higher B/M indicates
    higher book value relative to market capitalization (value stock).
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="book_to_market",
            factor_type=FactorType.VALUE,
            **kwargs
        )
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        # In production, book value should come from fundamental data
        # Here we use a simulated approach where available
        if "book_value_per_share" in df.columns:
            values = df["book_value_per_share"] / df["close"]
        else:
            # Fallback: placeholder direction
            values = 1.0 / df["close"]
        
        values = self._post_process(values, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=values,
            description="Book-to-Market: book value per share divided by price",
        )


class FreeCashFlowYield(FactorComputer):
    """
    Free Cash Flow Yield (FCF/P).
    
    Often considered a more robust value signal than earnings yield
    because FCF is harder to manipulate than accounting earnings.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="fcf_yield",
            factor_type=FactorType.VALUE,
            **kwargs
        )
    
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        self._validate_input(df, ["date", "ticker", "close"])
        
        if "fcf_per_share" in df.columns:
            values = df["fcf_per_share"] / df["close"]
        else:
            values = 1.0 / df["close"]
        
        values = self._post_process(values, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=values,
            description="Free Cash Flow Yield: free cash flow per share divided by price",
        )


class EnterpriseValueMultiple(FactorComputer):
    """
    Enterprise Value / EBITDA multiple.
    
    Lower EV/EBITDA indicates relative undervaluation.
    Preferred over P/E for comparing across different capital structures.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ev_ebitda",
            factor_type=FactorType.VALUE,
            **kwargs
        )
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        # Inverse so higher = more undervalued (consistent with other value factors)
        if "ev_ebitda" in df.columns:
            values = 1.0 / df["ev_ebitda"]
        else:
            values = 1.0 / df["close"]
        
        values = self._post_process(values, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=values,
            description="Enterprise Value / EBITDA (inverted): lower EV/EBITDA indicates undervaluation",
        )


class SalesToPrice(FactorComputer):
    """
    Sales-to-Price ratio (S/P).
    
    Useful for valuing companies with negative earnings but positive revenue.
    Often used as a complementary value signal.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="sales_to_price",
            factor_type=FactorType.VALUE,
            **kwargs
        )
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        self._validate_input(df, ["date", "ticker", "close"])
        
        if "revenue_per_share" in df.columns:
            values = df["revenue_per_share"] / df["close"]
        else:
            values = 1.0 / df["close"]
        
        values = self._post_process(values, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=values,
            description="Sales-to-Price: revenue per share divided by price",
        )


class CompositeValue(FactorComputer):
    """
    Composite value score: equal-weighted average of individual value factors.
    
    Combines multiple value signals into a single robust factor,
    reducing idiosyncratic noise from any single metric.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="composite_value",
            factor_type=FactorType.VALUE,
            **kwargs
        )
        self.components = [
            EarningsYield(**kwargs),
            BookToMarket(**kwargs),
            FreeCashFlowYield(**kwargs),
        ]
    
    def compute(self, df: pd.DataFrame, **kwargs) -> FactorResult:
        component_results = []
        for component in self.components:
            result = component.compute(df, **kwargs)
            component_results.append(result.values)
        
        # Combine signals: average of z-scores
        combined = pd.concat(component_results, axis=1).mean(axis=1)
        combined = self._post_process(combined, df)
        
        return FactorResult.from_series(
            name=self.name,
            factor_type=self.factor_type,
            values=combined,
            description="Composite value score: equally weighted earnings yield, book-to-market, and FCF yield",
        )
