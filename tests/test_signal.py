"""
Tests for signal processing — combination, normalization, orthogonalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_processing.combination import SignalCombiner
from signal_processing.normalization import SignalNormalizer
from signal_processing.orthogonalization import SignalOrthogonalizer


@pytest.fixture
def sample_signals() -> pd.DataFrame:
    """Generate correlated sample signals."""
    rng = np.random.default_rng(42)
    n = 500
    base = rng.normal(0, 1, n)
    return pd.DataFrame({
        "momentum": base + 0.3 * rng.normal(0, 1, n),
        "value": 0.7 * base + 0.5 * rng.normal(0, 1, n),
        "quality": 0.5 * base + 0.7 * rng.normal(0, 1, n),
        "growth": -0.3 * base + 0.9 * rng.normal(0, 1, n),
    })


@pytest.fixture
def forward_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 1, 500))


class TestSignalCombination:
    """Signal combination tests using SignalCombiner class."""

    def test_equal_weight(self, sample_signals):
        combined = SignalCombiner.equal_weight(sample_signals)
        assert len(combined) == len(sample_signals)
        assert not combined.isna().any()

    def test_rank_average(self, sample_signals):
        combined = SignalCombiner.rank_average(sample_signals)
        assert len(combined) == len(sample_signals)

    def test_weighted(self, sample_signals):
        weights = {"momentum": 0.4, "value": 0.3, "quality": 0.2, "growth": 0.1}
        combined = SignalCombiner.weighted(sample_signals, weights)
        assert len(combined) == len(sample_signals)

    def test_ic_weighted(self, sample_signals, forward_returns):
        combined = SignalCombiner.ic_weighted(sample_signals, forward_returns)
        assert len(combined) == len(sample_signals)

    def test_optimal(self, sample_signals, forward_returns):
        combined = SignalCombiner.optimal(sample_signals, forward_returns)
        assert len(combined) == len(sample_signals)

    def test_different_methods_produce_different_results(self, sample_signals):
        w1 = SignalCombiner.equal_weight(sample_signals)
        w2 = SignalCombiner.rank_average(sample_signals)
        assert not w1.equals(w2)


class TestSignalNormalization:
    """Signal normalization tests using SignalNormalizer class."""

    def test_zscore(self, sample_signals):
        for col in sample_signals.columns:
            normalized = SignalNormalizer.zscore(sample_signals[col])
            if normalized.notna().sum() > 0:
                assert abs(normalized.mean()) < 0.5 or np.isnan(normalized.mean())

    def test_rank(self, sample_signals):
        for col in sample_signals.columns:
            normalized = SignalNormalizer.rank(sample_signals[col], pct=True)
            assert normalized.max() <= 1.0
            assert normalized.min() >= 0

    def test_gaussian_rank(self, sample_signals):
        for col in sample_signals.columns:
            normalized = SignalNormalizer.gaussian_rank(sample_signals[col])
            assert not normalized.isna().all()

    def test_sigmoid(self, sample_signals):
        for col in sample_signals.columns:
            normalized = SignalNormalizer.sigmoid(sample_signals[col])
            assert normalized.min() >= 0
            assert normalized.max() <= 1

    def test_robust(self, sample_signals):
        for col in sample_signals.columns:
            normalized = SignalNormalizer.robust(sample_signals[col])
            assert not normalized.isna().all()

    def test_clip(self, sample_signals):
        signals = sample_signals.copy()
        signals.iloc[0, 0] = 100
        for col in signals.columns:
            clipped = SignalNormalizer.clip(signals[col], limits=(0.01, 0.99))
            assert clipped.max() <= signals[col].max()


class TestSignalOrthogonalization:
    """Signal orthogonalization tests using SignalOrthogonalizer class."""

    def test_gram_schmidt(self, sample_signals):
        order = list(sample_signals.columns)
        orthogonal = SignalOrthogonalizer.gram_schmidt(sample_signals, order)
        assert orthogonal.shape == sample_signals.shape

    def test_pca(self, sample_signals):
        orthogonal = SignalOrthogonalizer.pca(sample_signals, n_components=4)
        assert orthogonal.shape == sample_signals.shape

    def test_residualize(self, sample_signals):
        target = sample_signals["momentum"]
        conditioning = sample_signals[["value", "quality"]]
        residual = SignalOrthogonalizer.residualize(target, conditioning)
        assert len(residual) == len(sample_signals)

    def test_orthogonal_signals_uncorrelated(self, sample_signals):
        order = list(sample_signals.columns)
        orthogonal = SignalOrthogonalizer.gram_schmidt(sample_signals, order)
        corr = orthogonal.corr().fillna(0)
        off_diag = corr.values[np.triu_indices_from(corr.values, k=1)]
        assert all(abs(c) < 0.5 for c in off_diag)
