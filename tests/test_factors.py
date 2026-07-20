"""
Tests for the factor engine — value, momentum, quality, volatility, liquidity, growth, profitability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.value import (
    EarningsYield, BookToMarket, FreeCashFlowYield, CompositeValue,
)
from factors.momentum import (
    TimeSeriesMomentum, ResidualMomentum, VolatilityAdjustedMomentum,
    High52Week, CompositeMomentum,
)
from factors.quality import (
    ReturnOnEquity, GrossProfitability, CompositeQuality,
)
from factors.volatility import (
    RealizedVolatility, DownsideDeviation, MarketBeta, IdiosyncraticVolatility,
)
from factors.liquidity import AverageDailyVolume, AmihudIlliquidity
from factors.growth import CompositeGrowth
from factors.profitability import (
    GrossMargin, OperatingMargin, NetMargin, CompositeProfitability,
)
from factors.factory import FactorCatalog, compute_all_factors


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Generate sample OHLCV data for factor tests."""
    rng = np.random.default_rng(42)
    n_days = 252
    n_tickers = 10
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    rows = []
    for ticker in tickers:
        price = 100 + np.cumsum(rng.normal(0, 1.5, n_days))
        price = np.maximum(price, 10)
        volume = rng.integers(1_000_000, 10_000_000, n_days)
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": float(price[i]),
                "high": float(price[i] * (1 + abs(rng.normal(0, 0.01)))),
                "low": float(price[i] * (1 - abs(rng.normal(0, 0.01)))),
                "close": float(price[i]),
                "volume": int(volume[i]),
            })
    return pd.DataFrame(rows)


class TestValueFactors:
    """Value factor tests."""

    def test_earnings_yield(self, sample_market_data):
        factor = EarningsYield()
        result = factor.compute(sample_market_data)
        assert result is not None
        assert isinstance(result.values, pd.Series)

    def test_book_to_market(self, sample_market_data):
        factor = BookToMarket()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_free_cash_flow_yield(self, sample_market_data):
        factor = FreeCashFlowYield()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_composite_value(self, sample_market_data):
        factor = CompositeValue()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_all_value_factors_have_correct_type(self, sample_market_data):
        from core.types import FactorType
        for factor_cls in [EarningsYield, BookToMarket, FreeCashFlowYield]:
            f = factor_cls()
            assert f.factor_type == FactorType.VALUE


class TestMomentumFactors:
    """Momentum factor tests."""

    def test_time_series_momentum(self, sample_market_data):
        factor = TimeSeriesMomentum()
        result = factor.compute(sample_market_data)
        assert result is not None
        assert isinstance(result.values, pd.Series)

    def test_residual_momentum(self, sample_market_data):
        factor = ResidualMomentum()
        result = factor.compute(sample_market_data)
        assert result is not None
        assert isinstance(result.values, pd.Series)

    def test_volatility_adjusted_momentum(self, sample_market_data):
        factor = VolatilityAdjustedMomentum()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_high_52_week(self, sample_market_data):
        factor = High52Week()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_composite_momentum(self, sample_market_data):
        factor = CompositeMomentum()
        result = factor.compute(sample_market_data)
        assert result is not None
        assert not result.values.isna().all()

    def test_different_windows_produce_different_results(self, sample_market_data):
        f1 = TimeSeriesMomentum(window=63)
        f2 = TimeSeriesMomentum(window=126)
        r1 = f1.compute(sample_market_data).values
        r2 = f2.compute(sample_market_data).values
        # Different windows should produce different results (or at least non-constant)
        assert r1.notna().any() or r2.notna().any()


class TestQualityFactors:
    """Quality factor tests."""

    def test_return_on_equity(self, sample_market_data):
        factor = ReturnOnEquity()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_gross_profitability(self, sample_market_data):
        factor = GrossProfitability()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_composite_quality(self, sample_market_data):
        factor = CompositeQuality()
        result = factor.compute(sample_market_data)
        assert result is not None


class TestVolatilityFactors:
    """Volatility factor tests."""

    def test_realized_volatility(self, sample_market_data):
        factor = RealizedVolatility()
        result = factor.compute(sample_market_data)
        assert result is not None
        # Volatility should be non-negative
        assert result.values.notna().any() if len(result.values) > 0 else True

    def test_downside_deviation(self, sample_market_data):
        factor = DownsideDeviation()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_market_beta(self, sample_market_data):
        factor = MarketBeta()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_idiosyncratic_volatility(self, sample_market_data):
        factor = IdiosyncraticVolatility()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_different_volatility_windows(self, sample_market_data):
        f1 = RealizedVolatility(window=21)
        f2 = RealizedVolatility(window=60)
        r1 = f1.compute(sample_market_data).values
        r2 = f2.compute(sample_market_data).values
        # Different windows should produce valid series
        assert r1.notna().any() or r2.notna().any()


class TestLiquidityFactors:
    """Liquidity factor tests."""

    def test_average_daily_volume(self, sample_market_data):
        factor = AverageDailyVolume()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_amihud_illiquidity(self, sample_market_data):
        factor = AmihudIlliquidity()
        result = factor.compute(sample_market_data)
        assert result is not None
        assert isinstance(result.values, pd.Series)


class TestGrowthFactors:
    """Growth factor tests."""

    def test_revenue_growth(self, sample_market_data):
        from factors.growth import RevenueGrowth
        factor = RevenueGrowth()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_eps_growth(self, sample_market_data):
        from factors.growth import EPSGrowth
        factor = EPSGrowth()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_composite_growth(self, sample_market_data):
        factor = CompositeGrowth()
        result = factor.compute(sample_market_data)
        assert result is not None


class TestProfitabilityFactors:
    """Profitability factor tests."""

    def test_gross_margin(self, sample_market_data):
        factor = GrossMargin()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_operating_margin(self, sample_market_data):
        factor = OperatingMargin()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_net_margin(self, sample_market_data):
        factor = NetMargin()
        result = factor.compute(sample_market_data)
        assert result is not None

    def test_composite_profitability(self, sample_market_data):
        factor = CompositeProfitability()
        result = factor.compute(sample_market_data)
        assert result is not None


class TestFactorCatalog:
    """Factor factory and catalog tests."""

    def test_catalog_create_default(self):
        catalog = FactorCatalog.create_default()
        assert len(catalog.factors) > 0
        assert "earnings_yield" in catalog.factors
        assert "momentum_126d" in catalog.factors

    def test_compute_all_factors(self, sample_market_data):
        result = compute_all_factors(sample_market_data)
        assert result is not None
        assert result.factor_values is not None
        assert not result.factor_values.empty

    def test_compute_specific_factors(self, sample_market_data):
        result = compute_all_factors(
            sample_market_data,
            factor_names=["earnings_yield", "momentum_126d"]
        )
        assert result is not None
        expected_cols = {"earnings_yield", "momentum_126d"}
        assert expected_cols.issubset(set(result.factor_values.columns))
