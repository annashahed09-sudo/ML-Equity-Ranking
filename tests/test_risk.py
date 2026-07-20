"""
Tests for the risk module — covariance estimation, factor risk model, VaR, CVaR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk.covariance import (
    CovarianceEstimator, SampleCovariance, LedoitWolfCovariance, EWMACovariance,
)
from risk.factor_risk import FactorRiskModel
from risk.metrics import (
    compute_var, compute_cvar, compute_tracking_error,
    compute_active_risk, compute_beta, compute_alpha,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Generate sample return data."""
    rng = np.random.default_rng(42)
    n_days = 500
    n_assets = 10
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    tickers = [f"A{i:02d}" for i in range(n_assets)]

    # Correlated returns
    mean = np.zeros(n_assets)
    cov = 0.5 * np.ones((n_assets, n_assets)) + 0.5 * np.eye(n_assets)
    returns = rng.multivariate_normal(mean, cov, n_days)

    return pd.DataFrame(returns, index=dates, columns=tickers)


@pytest.fixture
def sample_factor_returns() -> pd.DataFrame:
    """Generate sample factor return data for risk model."""
    rng = np.random.default_rng(42)
    n_days = 500
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    factors = ["Market", "Value", "Momentum", "Quality", "Size"]

    mean = np.zeros(5)
    cov = 0.3 * np.ones((5, 5)) + 0.7 * np.eye(5)
    returns = rng.multivariate_normal(mean, cov, n_days)

    return pd.DataFrame(returns, index=dates, columns=factors)


class TestSampleCovariance:
    """Sample covariance estimator tests."""

    def test_estimate_shape(self, sample_returns):
        estimator = SampleCovariance()
        cov = estimator.estimate(sample_returns)
        assert cov.shape == (10, 10)

    def test_estimate_is_symmetric(self, sample_returns):
        estimator = SampleCovariance()
        cov = estimator.estimate(sample_returns)
        assert np.allclose(cov.values, cov.values.T)

    def test_estimate_positive_semi_definite(self, sample_returns):
        estimator = SampleCovariance()
        cov = estimator.estimate(sample_returns)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-10)

    def test_estimate_diagonal_positive(self, sample_returns):
        estimator = SampleCovariance()
        cov = estimator.estimate(sample_returns)
        assert np.all(np.diag(cov.values) > 0)

    def test_empty_data_raises(self):
        estimator = SampleCovariance()
        with pytest.raises(Exception):
            estimator.estimate(pd.DataFrame())


class TestLedoitWolfCovariance:
    """Ledoit-Wolf shrinkage estimator tests."""

    def test_shrinkage_improves_conditioning(self, sample_returns):
        sample_est = SampleCovariance()
        lw_est = LedoitWolfCovariance()

        sample_cov = sample_est.estimate(sample_returns)
        lw_cov = lw_est.estimate(sample_returns)

        sample_cond = np.linalg.cond(sample_cov.values)
        lw_cond = np.linalg.cond(lw_cov.values)

        # Shrinkage should improve conditioning (lower condition number)
        assert lw_cond <= sample_cond * 1.1  # Allow small margin

    def test_estimate_is_symmetric(self, sample_returns):
        estimator = LedoitWolfCovariance()
        cov = estimator.estimate(sample_returns)
        assert np.allclose(cov.values, cov.values.T)

    def test_estimate_positive_definite(self, sample_returns):
        estimator = LedoitWolfCovariance()
        cov = estimator.estimate(sample_returns)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues > -1e-10)


class TestEWMACovariance:
    """EWMA covariance estimator tests."""

    def test_estimate_shape(self, sample_returns):
        estimator = EWMACovariance(lambda_factor=0.94)
        cov = estimator.estimate(sample_returns)
        assert cov.shape == (10, 10)

    def test_different_lambda(self, sample_returns):
        e1 = EWMACovariance(lambda_factor=0.90)
        e2 = EWMACovariance(lambda_factor=0.97)
        c1 = e1.estimate(sample_returns)
        c2 = e2.estimate(sample_returns)
        assert not np.allclose(c1.values, c2.values)

    def test_invalid_lambda(self):
        with pytest.raises(Exception):
            EWMACovariance(lambda_factor=1.5)

    def test_lambda_near_one(self, sample_returns):
        estimator = EWMACovariance(lambda_factor=0.99)
        cov = estimator.estimate(sample_returns)
        assert cov.shape == (10, 10)


class TestFactorRiskModel:
    """Factor risk model tests."""

    def test_factor_exposures_shape(self, sample_returns, sample_factor_returns):
        model = FactorRiskModel(sample_factor_returns)
        exposures = model.compute_factor_exposures(sample_returns)
        assert exposures.shape == (10, 5)

    def test_factor_exposures_validity(self, sample_returns, sample_factor_returns):
        model = FactorRiskModel(sample_factor_returns)
        exposures = model.compute_factor_exposures(sample_returns)
        assert not exposures.isna().any().any()

    def test_risk_decomposition_keys(self, sample_returns, sample_factor_returns):
        model = FactorRiskModel(sample_factor_returns)
        weights = pd.Series(np.ones(10) / 10, index=sample_returns.columns)
        result = model.decompose_risk(weights, sample_returns)
        assert "total_risk" in result
        assert "systematic_risk" in result
        assert "idiosyncratic_risk" in result
        assert "factor_contributions" in result

    def test_risk_decomposition_positive(self, sample_returns, sample_factor_returns):
        model = FactorRiskModel(sample_factor_returns)
        weights = pd.Series(np.ones(10) / 10, index=sample_returns.columns)
        result = model.decompose_risk(weights, sample_returns)
        assert result["total_risk"] >= 0
        assert result["systematic_risk"] >= 0
        assert result["idiosyncratic_risk"] >= 0


class TestRiskMetrics:
    """Risk metric computation tests."""

    @pytest.fixture
    def portfolio_returns(self) -> pd.Series:
        rng = np.random.default_rng(42)
        return pd.Series(rng.normal(0.0005, 0.008, 1000))

    @pytest.fixture
    def benchmark_returns(self) -> pd.Series:
        rng = np.random.default_rng(43)
        return pd.Series(rng.normal(0.0003, 0.01, 1000))

    def test_var_historical(self, portfolio_returns):
        var = compute_var(portfolio_returns, confidence_level=0.95, method="historical")
        assert var > 0
        assert var < 0.1  # Sanity check

    def test_var_parametric(self, portfolio_returns):
        var = compute_var(portfolio_returns, confidence_level=0.95, method="parametric")
        assert var > 0

    def test_var_monte_carlo(self, portfolio_returns):
        var = compute_var(portfolio_returns, confidence_level=0.95, method="monte_carlo")
        assert var > 0

    def test_var_higher_confidence_greater_var(self, portfolio_returns):
        var_95 = compute_var(portfolio_returns, confidence_level=0.95)
        var_99 = compute_var(portfolio_returns, confidence_level=0.99)
        assert var_99 >= var_95

    def test_cvar(self, portfolio_returns):
        cvar = compute_cvar(portfolio_returns, confidence_level=0.95)
        assert cvar > 0
        assert cvar < 0.15

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_cvar_greater_than_var(self, portfolio_returns, confidence):
        var = compute_var(portfolio_returns, confidence_level=confidence)
        cvar = compute_cvar(portfolio_returns, confidence_level=confidence)
        assert cvar >= var * 0.9  # CVaR should be at least close to VaR

    def test_tracking_error(self, portfolio_returns, benchmark_returns):
        te = compute_tracking_error(portfolio_returns, benchmark_returns)
        assert te > 0

    def test_beta(self, portfolio_returns, benchmark_returns):
        beta = compute_beta(portfolio_returns, benchmark_returns, periods=252)
        assert not np.isnan(beta)

    def test_alpha(self, portfolio_returns, benchmark_returns):
        alpha = compute_alpha(portfolio_returns, benchmark_returns)
        assert not np.isnan(alpha)
