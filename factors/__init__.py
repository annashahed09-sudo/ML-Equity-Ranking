"""
Academically grounded factor computation engine.

Provides implementations of value, momentum, quality, volatility, liquidity,
growth, and profitability factors used in institutional quantitative research.
"""

from .base import FactorComputer, FactorResult
from .value import (
    EarningsYield,
    BookToMarket,
    FreeCashFlowYield,
    CompositeValue,
)
from .momentum import (
    TimeSeriesMomentum,
    ResidualMomentum,
    VolatilityAdjustedMomentum,
    High52Week,
    CompositeMomentum,
)
from .quality import (
    ReturnOnEquity,
    GrossProfitability,
    PiotroskiFScore,
    CompositeQuality,
)
from .volatility import (
    RealizedVolatility,
    DownsideDeviation,
    MarketBeta,
    IdiosyncraticVolatility,
)
from .liquidity import (
    AverageDailyVolume,
    AmihudIlliquidity,
)
from .growth import (
    RevenueGrowth,
    EPSGrowth,
    CompositeGrowth,
)
from .profitability import (
    GrossMargin,
    OperatingMargin,
    NetMargin,
    CompositeProfitability,
)
from .factory import FactorCatalog, compute_all_factors

__all__ = [
    "FactorComputer",
    "FactorResult",
    "FactorCatalog",
    "compute_all_factors",
    # Value
    "EarningsYield",
    "BookToMarket",
    "FreeCashFlowYield",
    "CompositeValue",
    # Momentum
    "TimeSeriesMomentum",
    "ResidualMomentum",
    "VolatilityAdjustedMomentum",
    "High52Week",
    "CompositeMomentum",
    # Quality
    "ReturnOnEquity",
    "GrossProfitability",
    "PiotroskiFScore",
    "CompositeQuality",
    # Volatility
    "RealizedVolatility",
    "DownsideDeviation",
    "MarketBeta",
    "IdiosyncraticVolatility",
    # Liquidity
    "AverageDailyVolume",
    "AmihudIlliquidity",
    # Growth
    "RevenueGrowth",
    "EPSGrowth",
    "CompositeGrowth",
    # Profitability
    "GrossMargin",
    "OperatingMargin",
    "NetMargin",
    "CompositeProfitability",
]
