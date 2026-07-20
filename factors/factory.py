"""
Factor factory: unified factor computation pipeline.

Computes all factor families and returns a consolidated DataFrame
with factor values, diagnostics, and quality checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from core.types import FactorName, FactorType
from factors.value import (
    EarningsYield,
    BookToMarket,
    FreeCashFlowYield,
    CompositeValue,
)
from factors.momentum import (
    TimeSeriesMomentum,
    ResidualMomentum,
    VolatilityAdjustedMomentum,
    High52Week,
    CompositeMomentum,
)
from factors.quality import (
    ReturnOnEquity,
    GrossProfitability,
    CompositeQuality,
)
from factors.volatility import (
    RealizedVolatility,
    DownsideDeviation,
    MarketBeta,
    IdiosyncraticVolatility,
)
from factors.liquidity import (
    AverageDailyVolume,
    AmihudIlliquidity,
)
from factors.growth import (
    CompositeGrowth,
)
from factors.profitability import (
    CompositeProfitability,
)
from .base import FactorComputer, FactorResult

logger = logging.getLogger(__name__)


@dataclass
class FactorCatalog:
    """Catalog of all available factor computers with metadata."""
    
    factors: Dict[str, FactorComputer] = field(default_factory=dict)
    
    @classmethod
    def create_default(cls) -> "FactorCatalog":
        """Create catalog with all default factor definitions."""
        catalog = cls()
        
        # Value factors
        catalog.register(CompositeValue())
        catalog.register(EarningsYield())
        catalog.register(BookToMarket())
        
        # Momentum factors
        for w in [21, 63, 126, 252]:
            catalog.register(TimeSeriesMomentum(window=w))
        catalog.register(CompositeMomentum())
        catalog.register(ResidualMomentum())
        catalog.register(High52Week())
        
        # Quality factors
        catalog.register(CompositeQuality())
        catalog.register(ReturnOnEquity())
        catalog.register(GrossProfitability())
        
        # Volatility factors
        for w in [21, 60, 252]:
            catalog.register(RealizedVolatility(window=w))
        catalog.register(DownsideDeviation())
        catalog.register(MarketBeta())
        catalog.register(IdiosyncraticVolatility())
        
        # Liquidity factors
        catalog.register(AverageDailyVolume(window=60))
        catalog.register(AmihudIlliquidity(window=60))
        
        # Growth factors
        catalog.register(CompositeGrowth())
        
        # Profitability factors
        catalog.register(CompositeProfitability())
        
        return catalog
    
    def register(self, factor: FactorComputer) -> None:
        """Register a factor computer."""
        self.factors[factor.name] = factor
    
    def get(self, name: str) -> FactorComputer:
        """Get factor by name."""
        if name not in self.factors:
            raise KeyError(f"Factor '{name}' not found in catalog")
        return self.factors[name]
    
    def names(self) -> List[str]:
        """Get list of all registered factor names."""
        return sorted(self.factors.keys())


@dataclass
class FactorComputationResult:
    """Result of computing all factors for a dataset."""
    
    factor_values: pd.DataFrame  # Wide format: columns = factors
    factor_results: Dict[FactorName, FactorResult]
    quality_warnings: Dict[FactorName, List[str]]
    computation_time: float = 0.0
    n_factors_computed: int = 0
    n_factors_failed: int = 0


def compute_all_factors(
    df: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    neutralize: str = "none",
) -> FactorComputationResult:
    """
    Compute all specified factors for the input dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with date, ticker, close columns
    factor_names : List[str], optional
        Specific factors to compute. If None, computes all.
    neutralize : str
        Neutralization: 'none', 'sector', 'market'
    
    Returns
    -------
    FactorComputationResult
        All factor values and diagnostic information
    """
    import time
    start = time.perf_counter()
    
    catalog = FactorCatalog.create_default()
    
    if factor_names is not None:
        computers = [catalog.get(name) for name in factor_names]
    else:
        computers = list(catalog.factors.values())
    
    factor_values = pd.DataFrame(index=df.index)
    factor_results: Dict[str, FactorResult] = {}
    quality_warnings: Dict[str, List[str]] = {}
    n_failed = 0
    
    for computer in computers:
        try:
            result = computer.compute(df)
            factor_results[computer.name] = result
            factor_values[computer.name] = result.values
            warnings = computer._check_quality(result)
            if warnings:
                quality_warnings[computer.name] = warnings
                for w in warnings:
                    logger.warning(f"Factor '{computer.name}': {w}")
        except Exception as e:
            logger.error(f"Factor '{computer.name}' failed: {e}")
            n_failed += 1
            factor_values[computer.name] = float("nan")
    
    elapsed = time.perf_counter() - start
    
    return FactorComputationResult(
        factor_values=factor_values,
        factor_results=factor_results,
        quality_warnings=quality_warnings,
        computation_time=elapsed,
        n_factors_computed=len(factor_results),
        n_factors_failed=n_failed,
    )
