"""
Tests for the validation framework — walk-forward, purged CV, backtesting, metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.walk_forward import WalkForwardSplitter, ExpandingWindowSplitter
from validation.purged_cv import PurgedCrossValidation
from validation.metrics import (
    compute_information_coefficient, compute_rank_ic,
    compute_quantile_returns, compute_long_short_returns,
    compute_sharpe_ratio, compute_sortino_ratio, compute_calmar_ratio,
    compute_max_drawdown, compute_drawdown_duration,
    compute_win_rate, compute_profit_factor, compute_turnover,
    compute_information_ratio, compute_ic_summary,
)


@pytest.fixture
def sample_predictions() -> pd.DataFrame:
    """Generate sample predictions data."""
    rng = np.random.default_rng(42)
    n_days = 100
    n_tickers = 20
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    data = []
    for d in dates:
        for t in range(n_tickers):
            score = rng.normal(0, 1)
            forward_ret = 0.5 * score + rng.normal(0, 1)
            data.append({
                "date": d,
                "ticker": f"T{t:02d}",
                "model_score": score,
                "forward_return": forward_ret,
            })
    return pd.DataFrame(data)


class TestWalkForwardSplitter:
    """Walk-forward validation tests."""

    def test_split_counts(self):
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=1000, freq="B")})
        splitter = WalkForwardSplitter(n_splits=5, test_size=100, min_train_size=400)
        folds = splitter.split(df)
        assert len(folds) == 5

    def test_fold_dates(self):
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=1000, freq="B")})
        splitter = WalkForwardSplitter(n_splits=3, test_size=200, min_train_size=400)
        folds = splitter.split(df)

        for i, fold in enumerate(folds):
            train_start, train_end = fold.train_dates
            test_start, test_end = fold.test_dates
            assert train_start < train_end
            assert test_start < test_end
            assert train_end <= test_start  # No overlap


class TestMetrics:
    """Performance metric computation tests."""

    def test_information_coefficient(self):
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        returns = pd.Series([2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0])
        ic = compute_information_coefficient(scores, returns, method="spearman")
        assert abs(ic) <= 1.0

    def test_rank_ic(self, sample_predictions):
        ic_series = compute_rank_ic(sample_predictions)
        assert len(ic_series) > 0

    def test_quantile_returns(self, sample_predictions):
        quantile_rets = compute_quantile_returns(sample_predictions, n_quantiles=5)
        assert len(quantile_rets) > 0
        if len(quantile_rets) > 0:
            assert quantile_rets["quantile"].nunique() == 5

    def test_long_short_returns(self, sample_predictions):
        ls = compute_long_short_returns(sample_predictions, long_pct=0.2, short_pct=0.2)
        assert len(ls) > 0
        assert "long_short_return" in ls.columns

    def test_sharpe_ratio(self):
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        sharpe = compute_sharpe_ratio(returns)
        assert not np.isnan(sharpe)

    def test_sortino_ratio(self):
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        sortino = compute_sortino_ratio(returns)
        assert not np.isnan(sortino)

    def test_calmar_ratio(self):
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        calmar = compute_calmar_ratio(returns)
        assert not np.isnan(calmar)

    def test_max_drawdown(self):
        returns = pd.Series(np.array([0.01, 0.02, 0.03, -0.05, -0.03, 0.01, 0.02]))
        mdd = compute_max_drawdown(returns)
        assert mdd < 0

    def test_win_rate(self):
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03])
        wr = compute_win_rate(returns)
        assert wr == 0.6

    def test_profit_factor(self):
        returns = pd.Series([0.01, -0.005, 0.02, -0.005, 0.03])
        pf = compute_profit_factor(returns)
        assert pf > 1.0

    def test_turnover(self):
        current = pd.Series({"A": 0.3, "B": 0.7})
        previous = pd.Series({"A": 0.5, "B": 0.5})
        turnover = compute_turnover(current, previous)
        assert abs(turnover - 0.4) < 1e-6

    def test_ic_summary(self):
        ic_series = pd.Series(np.random.default_rng(42).normal(0.05, 0.03, 100))
        summary = compute_ic_summary(ic_series)
        assert "mean_ic" in summary
        assert "ic_sharpe" in summary
        assert "pct_positive" in summary
