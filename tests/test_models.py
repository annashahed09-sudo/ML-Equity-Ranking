import pandas as pd
import numpy as np
from src.models import (
    RidgeModel,
    GradientBoostingModel,
    QuantumInspiredModel,
    RandomForestModel,
    HistGBModel,
    NeuralMLPModel,
    AdvancedEnsembleModel,
    create_model,
)


def _sample_data(n=120, p=5):
    X = pd.DataFrame(np.random.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = np.random.randn(n)
    return X, y


def test_core_models_predict_shape():
    X, y = _sample_data(120, 4)
    models = [
        RidgeModel(prefer_gpu=False, prefer_numba=False),
        GradientBoostingModel(n_estimators=20, prefer_gpu=False, prefer_numba=False),
        QuantumInspiredModel(n_components=32, prefer_gpu=False, prefer_numba=False),
    ]
    for model in models:
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (120,)


def test_advanced_models_predict_shape():
    X, y = _sample_data(90, 4)
    models = [
        RandomForestModel(n_estimators=20, prefer_gpu=False, prefer_numba=False),
        HistGBModel(max_iter=50, prefer_gpu=False, prefer_numba=False),
        NeuralMLPModel(hidden_layer_sizes=(8,), max_iter=1000, prefer_gpu=False, prefer_numba=False),
        AdvancedEnsembleModel(prefer_gpu=False, prefer_numba=False),
    ]
    for model in models:
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (90,)


def test_create_model_factory_aliases():
    aliases = [
        ('qi', QuantumInspiredModel),
        ('rf', RandomForestModel),
        ('hgb', HistGBModel),
        ('mlp', NeuralMLPModel),
        ('ensemble', AdvancedEnsembleModel),
    ]
    for alias, klass in aliases:
        model = create_model(alias, prefer_gpu=False, prefer_numba=False)
        assert isinstance(model, klass)
