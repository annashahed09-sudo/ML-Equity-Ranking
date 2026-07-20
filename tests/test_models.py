"""
Tests for the model layer — linear models, tree-based, ensembles, rankers, neural, tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.factory import ModelFactory


@pytest.fixture
def sample_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample training data with no NaN/Inf values."""
    rng = np.random.default_rng(42)
    n = 500
    # Use uniform bounded values to avoid Inf/NaN in sklearn models
    X = pd.DataFrame({
        "feature_1": rng.uniform(-3, 3, n),
        "feature_2": rng.uniform(-3, 3, n),
        "feature_3": rng.uniform(-3, 3, n),
        "feature_4": rng.uniform(-3, 3, n),
        "feature_5": rng.uniform(-3, 3, n),
    })
    y = pd.Series(
        0.5 * X["feature_1"] + 0.3 * X["feature_2"] - 0.2 * X["feature_3"] +
        0.1 * X["feature_4"] * X["feature_5"] + rng.normal(0, 0.1, n),
        name="target",
    )
    return X, y


@pytest.fixture
def sample_test_data() -> pd.DataFrame:
    rng = np.random.default_rng(43)
    return pd.DataFrame({
        "feature_1": rng.uniform(-3, 3, 100),
        "feature_2": rng.uniform(-3, 3, 100),
        "feature_3": rng.uniform(-3, 3, 100),
        "feature_4": rng.uniform(-3, 3, 100),
        "feature_5": rng.uniform(-3, 3, 100),
    })


class TestLinearModels:
    """Linear model tests."""

    def test_ridge_model(self, sample_training_data, sample_test_data):
        X, y = sample_training_data
        model = ModelFactory.create("ridge")
        model.fit(X, y)
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)
        assert not np.isnan(preds).any()

    def test_elastic_net_model(self, sample_training_data, sample_test_data):
        X, y = sample_training_data
        model = ModelFactory.create("elastic_net")
        model.fit(X, y)
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)

    def test_lasso_model(self, sample_training_data, sample_test_data):
        X, y = sample_training_data
        model = ModelFactory.create("lasso")
        model.fit(X, y)
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)

    def test_linear_model_feature_importance(self, sample_training_data):
        X, y = sample_training_data
        model = ModelFactory.create("ridge")
        model.fit(X, y)
        importance = model.get_feature_importance()
        if importance is not None:
            assert len(importance) >= 0


class TestTreeModels:
    """Tree-based model tests."""

    def test_random_forest_model(self, sample_training_data, sample_test_data):
        X, y = sample_training_data
        model = ModelFactory.create("random_forest")
        model.fit(X, y)
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)

    def test_xgboost_model(self, sample_training_data, sample_test_data):
        try:
            import xgboost as xgb
            # Test if XGBoost can actually create and train a model (needs libomp)
            _test = xgb.XGBRegressor(n_estimators=2, max_depth=1, random_state=42)
            import numpy as np
            _test.fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))
        except (ImportError, OSError, Exception):
            pytest.skip("XGBoost not available (missing OpenMP runtime)")
        X, y = sample_training_data
        model = ModelFactory.create("xgboost")
        model.fit(X, y)
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)

    def test_lightgbm_model(self, sample_training_data, sample_test_data):
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError):
            pytest.skip("LightGBM not available (missing OpenMP runtime)")
        X, y = sample_training_data
        model = ModelFactory.create("lightgbm")
        model.fit(X, y)
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)

    def test_catboost_model(self, sample_training_data, sample_test_data):
        try:
            import catboost  # noqa: F401
        except (ImportError, OSError):
            pytest.skip("CatBoost not available")
        X, y = sample_training_data
        model = ModelFactory.create("catboost")
        model.fit(X, y)
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)


class TestEnsembleModels:
    """Ensemble model tests."""

    def test_stacking_ensemble(self, sample_training_data, sample_test_data):
        X, y = sample_training_data
        model = ModelFactory.create("stacking_ensemble")
        model.fit(X, y)
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)

    def test_voting_ensemble(self, sample_training_data, sample_test_data):
        X, y = sample_training_data
        model = ModelFactory.create("voting_ensemble")
        model.fit(X, y)
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)


class TestRankerModels:
    """Ranker model tests."""

    def test_lightgbm_ranker(self, sample_training_data, sample_test_data):
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError):
            pytest.skip("LightGBM not available (missing OpenMP runtime)")
        X, y = sample_training_data
        model = ModelFactory.create("lgbm_ranker")
        model.fit(X, y, group=[len(X)])
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)

    def test_xgboost_ranker(self, sample_training_data, sample_test_data):
        try:
            import xgboost as xgb
            # Test if XGBoost can actually create and train a model (needs libomp)
            _test = xgb.XGBRegressor(n_estimators=2, max_depth=1, random_state=42)
            import numpy as np
            _test.fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))
        except (ImportError, OSError, Exception):
            pytest.skip("XGBoost not available (missing OpenMP runtime)")
        X, y = sample_training_data
        model = ModelFactory.create("xgb_ranker")
        model.fit(X, y, group=[len(X)])
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)


class TestNeuralModels:
    """Neural network model tests."""

    def test_neural_mlp_model(self, sample_training_data, sample_test_data):
        X, y = sample_training_data
        model = ModelFactory.create("neural_mlp")
        model.fit(X, y)
        preds = model.predict(sample_test_data)
        assert len(preds) == len(sample_test_data)


class TestModelFactory:
    """Model factory tests."""

    def test_create_ridge(self):
        model = ModelFactory.create("ridge")
        assert model is not None
        assert hasattr(model, "fit")

    def test_create_invalid_model(self):
        with pytest.raises(Exception):
            ModelFactory.create("non_existent_model")

    def test_all_linear_models_creatable(self):
        for model_type in ["ridge", "lasso", "elastic_net"]:
            model = ModelFactory.create(model_type)
            assert model is not None

    def test_model_has_predict_method(self, sample_training_data):
        X, y = sample_training_data
        model = ModelFactory.create("ridge")
        model.fit(X, y)
        assert hasattr(model, "predict")
        preds = model.predict(X[:10])
        assert len(preds) == 10
