import pandas as pd
import numpy as np
from src.models import RidgeModel, GradientBoostingModel, QuantumInspiredModel, create_model


def test_ridge_model():
    X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
    y = np.random.randn(100)
    model = RidgeModel(prefer_gpu=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)


def test_gb_model():
    X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
    y = np.random.randn(100)
    model = GradientBoostingModel(n_estimators=10, prefer_gpu=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)


def test_quantum_inspired_model():
    X = pd.DataFrame(np.random.randn(80, 4), columns=['a', 'b', 'c', 'd'])
    y = np.random.randn(80)
    model = QuantumInspiredModel(n_components=32, prefer_gpu=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (80,)


def test_create_model_factory_quantum_alias():
    model = create_model('qi', n_components=16, prefer_gpu=False)
    assert isinstance(model, QuantumInspiredModel)
