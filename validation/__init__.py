"""
Validation framework for time-series aware model evaluation.

Provides walk-forward validation, purged cross-validation,
backtesting engine, and comprehensive performance metrics.
"""

from .walk_forward import (
    WalkForwardSplitter,
    ExpandingWindowSplitter,
)
from .purged_cv import PurgedCrossValidation
from .backtesting import (
    BacktestEngine,
    BacktestResult,
)
from .metrics import (
    compute_information_coefficient,
    compute_rank_ic,
    compute_quantile_returns,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_win_rate,
    compute_turnover,
    compute_long_short_returns,
)

__all__ = [
    "WalkForwardSplitter",
    "ExpandingWindowSplitter",
    "PurgedCrossValidation",
    "BacktestEngine",
    "BacktestResult",
    "compute_information_coefficient",
    "compute_rank_ic",
    "compute_quantile_returns",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compute_max_drawdown",
    "compute_win_rate",
    "compute_turnover",
    "compute_long_short_returns",
]
