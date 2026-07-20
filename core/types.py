"""
Domain types and data structures for the quantitative equity research platform.

All core types used across the platform are defined here to avoid circular
imports and ensure type consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Type Aliases ──────────────────────────────────────────────────────────

Ticker = str
Date = str  # ISO format: "YYYY-MM-DD"
FactorName = str
SignalName = str


# ── Enums ─────────────────────────────────────────────────────────────────

class FactorType(str, Enum):
    """Academically grounded factor categories."""
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    GROWTH = "growth"
    PROFITABILITY = "profitability"
    SIZE = "size"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    TECHNICAL = "technical"
    COMPOSITE = "composite"


class SignalType(str, Enum):
    """Types of signals produced by the platform."""
    RANKING = "ranking"
    ALPHA = "alpha"
    SENTIMENT = "sentiment"
    FACTOR = "factor"
    ENSEMBLE = "ensemble"
    RISK = "risk"


class MarketRegime(str, Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    RECOVERY = "recovery"
    CRISIS = "crisis"


class RebalanceFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


# ── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class AssetReturn:
    """Return data for a single asset at a point in time."""
    ticker: Ticker
    date: Date
    total_return: float          # Total return (incl. dividends)
    log_return: float            # Log return
    excess_return: float         # Return - risk-free rate
    abnormal_return: Optional[float] = None  # Alpha (CAPM-adjusted)


@dataclass
class FactorExposure:
    """A single factor exposure for a ticker on a date."""
    ticker: Ticker
    date: Date
    factor_name: FactorName
    factor_type: FactorType
    raw_value: float
    zscore: float = 0.0          # Cross-sectional z-score
    percentile: float = 0.5      # Cross-sectional percentile (0-1)
    sector_neutral: float = 0.0  # Sector-neutral value
    market_neutral: float = 0.0  # Market-neutral value
    winsorized: float = 0.0      # Winsorized value
    is_extreme: bool = False     # Flag for extreme values


@dataclass
class PortfolioWeight:
    """Portfolio weight for a single asset."""
    ticker: Ticker
    weight: float                # Portfolio weight (positive = long, negative = short)
    date: Date
    signal_strength: float = 0.0  # Raw signal strength
    sector: str = ""
    market_cap: float = 0.0
    constrained: bool = False     # Whether weight was modified by constraints


@dataclass
class RankedAsset:
    """A ranked asset with its score and metadata."""
    ticker: Ticker
    rank: int
    score: float                 # Model score
    predicted_return: float = 0.0
    expected_direction: str = "neutral"
    sector: str = ""
    factors: Dict[str, float] = field(default_factory=dict)
    signal_components: Dict[str, float] = field(default_factory=dict)
    prediction_interval: Tuple[float, float] = (0.0, 0.0)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    strategy_name: str
    start_date: Date
    end_date: Date
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_turnover: float
    avg_holding_days: float
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    drawdown_series: pd.Series
    monthly_returns: Optional[pd.DataFrame] = None
    sector_exposures: Optional[pd.DataFrame] = None
    factor_exposures: Optional[pd.DataFrame] = None
    turnover_series: Optional[pd.Series] = None


@dataclass
class FactorReturn:
    """Time series of returns for a single factor."""
    factor_name: FactorName
    factor_type: FactorType
    returns: pd.Series
    cumulative_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    t_stat: float
    p_value: float


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_id: str
    model_name: str
    factor_set: List[str]
    validation_strategy: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None
    predictions: Optional[pd.DataFrame] = None
    backtest: Optional[BacktestResult] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    git_commit: str = ""
    notes: str = ""
