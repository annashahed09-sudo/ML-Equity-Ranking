import pandas as pd
import numpy as np
from src.models import RidgeModel, GradientBoostingModel

def test_ridge_model():
    X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
    y = np.random.randn(100)
    model = RidgeModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)

def test_gb_model():
    X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
    y = np.random.randn(100)
    model = GradientBoostingModel(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)
