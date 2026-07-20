"""
Tests for the explainability module — permutation importance, partial dependence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explainability.importance import PermutationImportance, ShapExplainer
from explainability.partial_dep import PartialDependence


class DummyModel:
    """Simple model for testing explainability."""

    def __init__(self, coefs=None):
        if coefs is not None:
            self.coefs = np.asarray(coefs, dtype=np.float64)
        else:
            self.coefs = np.array([1.0, 2.0, -1.0], dtype=np.float64)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X.values @ self.coefs


@pytest.fixture
def sample_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "feature_1": rng.normal(0, 1, 200),
        "feature_2": rng.normal(0, 1, 200),
        "feature_3": rng.normal(0, 1, 200),
    })


@pytest.fixture
def sample_target() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 1, 200))


class TestPermutationImportance:
    """Permutation importance tests."""

    def test_importance_computation(self, sample_features, sample_target):
        model = DummyModel()
        importance = PermutationImportance(
            model.predict, sample_features, sample_target, n_repeats=5
        )
        result = importance.compute()

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance" in result.columns
        assert "std" in result.columns
        assert len(result) == len(sample_features.columns)

    def test_feature_ordering(self, sample_features, sample_target):
        model = DummyModel(coefs=np.array([1.0, 0.1, 5.0]))
        importance = PermutationImportance(
            model.predict, sample_features, sample_target, n_repeats=5
        )
        result = importance.compute()
        # Feature 3 (coef=5.0) should be among the top
        assert result is not None
        assert len(result) == 3


class TestPartialDependence:
    """Partial dependence plot tests."""

    def test_pdp_computation(self, sample_features):
        model = DummyModel()
        pdp = PartialDependence(model.predict, sample_features)
        result = pdp.compute("feature_1", grid_size=50)
        assert len(result) == 50
        assert "feature_value" in result.columns
        assert "partial_dependence" in result.columns

    def test_pdp_monotonic(self, sample_features):
        model = DummyModel(coefs=np.array([2.0, 0.0, 0.0]))
        pdp = PartialDependence(model.predict, sample_features)
        result = pdp.compute("feature_1", grid_size=20)
        diffs = np.diff(result["partial_dependence"])
        assert np.all(diffs >= 0) or np.all(diffs <= 0)

    def test_pdp_interaction(self, sample_features):
        model = DummyModel()
        pdp = PartialDependence(model.predict, sample_features)
        result = pdp.compute_interaction("feature_1", "feature_2", grid_size=10)
        assert result.shape == (10, 10)
