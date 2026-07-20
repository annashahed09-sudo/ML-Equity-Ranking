"""
Tests for the core module — types, exceptions, and mathematical utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.types import (
    AssetReturn, FactorExposure, PortfolioWeight, RankedAsset,
    BacktestResult, ExperimentResult, FactorType, SignalType,
    MarketRegime, RebalanceFrequency,
)
from core.exceptions import (
    QuantsError, DataError, ModelError, ValidationError,
    PortfolioError, RiskError, ConfigError, FactorError,
    SignalError, BacktestError, ConvergenceError, NumericError,
    DataQualityError, LookAheadBiasError, LeakageError,
)
from core.utils import (
    ensure_array, ensure_dataframe, validate_probability,
    validate_positive, validate_window, validate_weights,
    winsorize_series, zscore_normalize, cross_sectional_zscore,
    rank_transform, gaussian_rank_transform, stable_softmax,
    entropy, kl_divergence, jensen_shannon_distance,
)


class TestTypes:
    """Domain types initialization and validation."""

    def test_asset_return_defaults(self):
        ar = AssetReturn(ticker="AAPL", date="2024-01-01",
                         total_return=0.05, log_return=0.0488,
                         excess_return=0.04)
        assert ar.ticker == "AAPL"
        assert ar.abnormal_return is None

    def test_factor_exposure_defaults(self):
        fe = FactorExposure(ticker="AAPL", date="2024-01-01",
                            factor_name="Momentum", factor_type=FactorType.MOMENTUM,
                            raw_value=1.5)
        assert fe.zscore == 0.0
        assert fe.is_extreme is False

    def test_portfolio_weight_defaults(self):
        pw = PortfolioWeight(ticker="AAPL", weight=0.1, date="2024-01-01")
        assert pw.constrained is False
        assert pw.sector == ""

    def test_ranked_asset_defaults(self):
        ra = RankedAsset(ticker="AAPL", rank=1, score=0.85)
        assert ra.expected_direction == "neutral"
        assert ra.confidence == 0.0

    def test_factor_type_values(self):
        assert FactorType.VALUE.value == "value"
        assert FactorType.MOMENTUM.value == "momentum"

    def test_market_regime_values(self):
        assert MarketRegime.BEAR.value == "bear"
        assert MarketRegime.CRISIS.value == "crisis"


class TestExceptions:
    """Exception hierarchy and instantiation."""

    def test_base_exception(self):
        e = QuantsError("test error")
        assert str(e) == "test error"
        assert e.detail is None

    def test_base_exception_with_detail(self):
        e = QuantsError("test error", detail="additional info")
        assert e.detail == "additional info"

    def test_exception_hierarchy(self):
        assert issubclass(DataError, QuantsError)
        assert issubclass(ModelError, QuantsError)
        assert issubclass(ValidationError, QuantsError)
        assert issubclass(PortfolioError, QuantsError)
        assert issubclass(RiskError, QuantsError)
        assert issubclass(ConfigError, QuantsError)
        assert issubclass(FactorError, QuantsError)
        assert issubclass(SignalError, QuantsError)
        assert issubclass(BacktestError, QuantsError)
        assert issubclass(ConvergenceError, QuantsError)
        assert issubclass(NumericError, QuantsError)
        assert issubclass(DataQualityError, DataError)
        assert issubclass(LookAheadBiasError, ValidationError)
        assert issubclass(LeakageError, ValidationError)

    def test_all_exceptions_instantiate(self):
        for exc in [DataError, ModelError, ValidationError, PortfolioError,
                    RiskError, ConfigError, FactorError, SignalError,
                    BacktestError, ConvergenceError, NumericError]:
            e = exc("test")
            assert str(e) == "test"


class TestUtils:
    """Mathematical and data utilities."""

    def test_ensure_array_from_list(self):
        result = ensure_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_ensure_array_inf_to_nan(self):
        result = ensure_array([1.0, np.inf, 3.0])
        assert np.isnan(result[1])

    def test_ensure_dataframe_from_series(self):
        s = pd.Series([1, 2, 3], name="test")
        df = ensure_dataframe(s)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 1)

    def test_validate_probability_valid(self):
        assert validate_probability(0.5) == 0.5
        assert validate_probability(0.0) == 0.0
        assert validate_probability(1.0) == 1.0

    def test_validate_probability_invalid(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_probability(-0.1)
        with pytest.raises(ValueError, match="must be in"):
            validate_probability(1.5)

    def test_validate_positive(self):
        assert validate_positive(5.0) == 5.0
        with pytest.raises(ValueError):
            validate_positive(0.0)
        with pytest.raises(ValueError):
            validate_positive(-1.0)

    def test_validate_window(self):
        assert validate_window(10) == 10
        with pytest.raises(ValueError):
            validate_window(0)
        with pytest.raises(ValueError):
            validate_window(-5)
        with pytest.raises(ValueError):
            validate_window(3.5)

    def test_validate_weights(self):
        weights = np.array([0.5, 0.3, 0.2])
        result = validate_weights(weights)
        assert np.allclose(result, weights)

        with pytest.raises(ValueError):
            validate_weights(np.array([0.5, 0.5, 0.5]))

    def test_winsorize_series(self):
        s = pd.Series([1, 2, 3, 100, 200, 4, 5, 6, 7, 8, 9, 10])
        result = winsorize_series(s, limits=(0.1, 0.9))
        assert result.max() < 200
        assert result.min() >= 1

    def test_zscore_normalize_standard(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = zscore_normalize(s, method="standard")
        # Result mean should be approximately 0
        assert abs(result.mean()) < 1.0

    def test_zscore_normalize_single_value(self):
        s = pd.Series([5.0, 5.0, 5.0])
        result = zscore_normalize(s)
        assert all(r == 0.0 or np.isnan(r) for r in result)

    def test_rank_transform(self):
        s = pd.Series([0.1, 0.5, 0.9, 0.3, 0.7])
        result = rank_transform(s, pct=True)
        assert result.max() == 1.0
        assert result.min() > 0

    def test_gaussian_rank_transform(self):
        s = pd.Series([0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.0])
        result = gaussian_rank_transform(s)
        assert not result.isna().all()

    def test_stable_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        result = stable_softmax(x)
        assert abs(result.sum() - 1.0) < 1e-10
        assert result[2] > result[1] > result[0]

    def test_entropy(self):
        p = np.array([0.5, 0.5])
        assert abs(entropy(p) - 0.693) < 0.01

    def test_kl_divergence(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        assert kl_divergence(p, q) < 1e-10

    def test_jensen_shannon_distance(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        # JSD between [1,0] and [0,1] should be max possible: sqrt(ln(2)) ≈ 0.832
        jsd = jensen_shannon_distance(p, q)
        assert 0.8 < jsd < 0.85

    def test_cross_sectional_zscore(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"] * 5 + ["2024-01-02"] * 5,
            "ticker": [f"T{i}" for i in range(5)] * 2,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
        })
        result = cross_sectional_zscore(df, "value", group_col="date")
        assert len(result) == 10
        assert not result.isna().any()
