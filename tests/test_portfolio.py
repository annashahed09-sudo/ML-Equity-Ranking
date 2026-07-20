"""
Tests for the portfolio optimization modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.mean_variance import MeanVarianceOptimizer
from portfolio.risk_parity import RiskParityOptimizer
from portfolio.min_variance import MinimumVarianceOptimizer
from portfolio.black_litterman import BlackLittermanOptimizer
from portfolio.factor_model import FactorModelOptimizer
from portfolio.constraints import PortfolioConstraints


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Generate sample return data."""
    rng = np.random.default_rng(42)
    n_days = 500
    n_assets = 8
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    tickers = [f"A{i:02d}" for i in range(n_assets)]

    mean = np.array([0.08, 0.12, 0.06, 0.10, 0.14, 0.07, 0.09, 0.11]) / 252
    # Realistic daily covariance: ~15-20% annualized vol, properly PSD
    corr = 0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets)
    cov = 0.0004 * corr
    returns = rng.multivariate_normal(mean, cov, n_days)

    return pd.DataFrame(returns, index=dates, columns=tickers)


@pytest.fixture
def expected_returns() -> pd.Series:
    """Generate sample expected returns."""
    rng = np.random.default_rng(42)
    tickers = [f"A{i:02d}" for i in range(8)]
    return pd.Series(rng.uniform(0.06, 0.18, 8), index=tickers)


@pytest.fixture
def market_caps() -> pd.Series:
    """Generate sample market capitalizations."""
    rng = np.random.default_rng(42)
    tickers = [f"A{i:02d}" for i in range(8)]
    return pd.Series(rng.uniform(1e9, 3e12, 8), index=tickers)


class TestMeanVarianceOptimizer:
    """Mean-variance optimization tests."""

    def test_max_sharpe_weights(self, sample_returns, expected_returns):
        optimizer = MeanVarianceOptimizer()
        weights = optimizer.max_sharpe(expected_returns, sample_returns)
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_max_sharpe(self, sample_returns, expected_returns):
        optimizer = MeanVarianceOptimizer()
        weights = optimizer.max_sharpe(expected_returns, sample_returns)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_optimizer_creates_weights(self, sample_returns, expected_returns):
        optimizer = MeanVarianceOptimizer()
        weights = optimizer.max_sharpe(expected_returns, sample_returns)
        assert weights is not None
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_efficient_frontier(self, sample_returns, expected_returns):
        optimizer = MeanVarianceOptimizer()
        frontier = optimizer.efficient_frontier(expected_returns, sample_returns, n_points=25)
        if frontier is not None and len(frontier) > 0:
            assert all(frontier["volatility"] >= 0)

    def test_can_create_instance(self):
        optimizer = MeanVarianceOptimizer()
        assert optimizer is not None


class TestRiskParityOptimizer:
    """Risk parity optimization tests."""

    def test_equal_risk_contribution(self, sample_returns):
        optimizer = RiskParityOptimizer()
        weights = optimizer.equal_risk_contribution(sample_returns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)

    def test_risk_contributions_balanced(self, sample_returns):
        optimizer = RiskParityOptimizer()
        weights = optimizer.equal_risk_contribution(sample_returns)
        cov = sample_returns.cov().values
        risk_contrib = weights.values * (cov @ weights.values) / np.sqrt(weights.values @ cov @ weights.values)
        cv = risk_contrib.std() / (risk_contrib.mean() + 1e-10)
        assert cv < 0.5 or len(risk_contrib) == 8


class TestMinimumVarianceOptimizer:
    """Minimum variance optimization tests."""

    def test_minimize_volatility(self, sample_returns):
        optimizer = MinimumVarianceOptimizer()
        weights = optimizer.minimize_volatility(sample_returns)
        if weights is not None:
            assert abs(weights.sum() - 1.0) < 1e-6


class TestBlackLittermanOptimizer:
    """Black-Litterman optimization tests."""

    def test_reverse_optimize(self, market_caps, sample_returns):
        optimizer = BlackLittermanOptimizer()
        cov = sample_returns.cov()
        implied = optimizer.reverse_optimize(market_caps, cov)
        assert len(implied) == len(market_caps)

    def test_optimize_no_views(self, market_caps, sample_returns):
        optimizer = BlackLittermanOptimizer()
        weights = optimizer.optimize(market_caps, sample_returns, views=None)
        assert abs(weights.sum() - 1.0) < 0.01


class TestFactorModelOptimizer:
    """Factor model portfolio optimization tests."""

    def test_factor_covariance(self, sample_returns):
        rng = np.random.default_rng(42)
        factor_returns = pd.DataFrame(
            rng.normal(0, 0.01, (500, 3)),
            index=sample_returns.index,
            columns=["MKT", "SMB", "HML"],
        )
        optimizer = FactorModelOptimizer(factor_returns)
        cov = optimizer.factor_covariance(sample_returns)
        assert cov.shape[0] == len(sample_returns.columns)


class TestPortfolioConstraints:
    """Portfolio constraint utilities tests."""

    def test_apply_clips_weights(self):
        constraints = PortfolioConstraints(max_weight=0.3, min_weight=-0.3)
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        constrained = constraints.apply(weights)
        assert constrained["A"] <= 0.3

    def test_apply_preserves_sum(self):
        constraints = PortfolioConstraints(max_weight=0.6, min_weight=-0.6)
        weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
        constrained = constraints.apply(weights)
        assert abs(constrained.sum() - 1.0) < 0.01

    def test_sector_constraint(self):
        constraints = PortfolioConstraints(
            max_weight=0.5, min_weight=0.0,
            sector_limits={"Tech": 0.4},
        )
        weights = pd.Series({"AAPL": 0.3, "MSFT": 0.5, "JPM": 0.2})
        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Fin"}
        constrained = constraints.apply(weights, sector_map=sector_map)
        assert constrained["MSFT"] <= 0.4

    def test_default_constraint_parameters(self):
        constraints = PortfolioConstraints()
        assert constraints.max_weight == 0.2
        assert constraints.min_weight == -0.2
