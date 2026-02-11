"""Model wrappers for cross-sectional equity prediction."""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline

from .accelerated import normalize_scores


class CrossSectionalModel:
    def __init__(self, name: str, prefer_gpu: bool = True, prefer_numba: bool = True):
        self.name = name
        self.model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        self.prefer_gpu = prefer_gpu
        self.prefer_numba = prefer_numba

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_feature_importance(self) -> Optional[pd.Series]:
        return None


class RidgeModel(CrossSectionalModel):
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(name="Ridge", **kwargs)
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_scaled = self.feature_scaler.transform(X)
        scores = self.model.predict(X_scaled)
        return normalize_scores(scores, self.prefer_gpu, self.prefer_numba)


class GradientBoostingModel(CrossSectionalModel):
    def __init__(self, n_estimators: int = 150, max_depth: int = 3, learning_rate: float = 0.05, **kwargs):
        super().__init__(name="GradientBoosting", **kwargs)
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
        )
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = X.columns.tolist()
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_scaled = self.feature_scaler.transform(X)
        scores = self.model.predict(X_scaled)
        return normalize_scores(scores, self.prefer_gpu, self.prefer_numba)

    def get_feature_importance(self) -> Optional[pd.Series]:
        if not self.is_fitted or self.feature_names is None:
            return None
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)


class QuantumInspiredModel(CrossSectionalModel):
    def __init__(self, n_components: int = 512, gamma: float = 1.0, alpha: float = 0.5, random_state: int = 42, **kwargs):
        super().__init__(name="QuantumInspired", **kwargs)
        self.model = Pipeline([
            ("rff", RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_scaled = self.feature_scaler.transform(X)
        scores = self.model.predict(X_scaled)
        return normalize_scores(scores, self.prefer_gpu, self.prefer_numba)


class RandomForestModel(CrossSectionalModel):
    def __init__(self, n_estimators: int = 300, max_depth: Optional[int] = 8, **kwargs):
        super().__init__(name="RandomForest", **kwargs)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = X.columns.tolist()
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_scaled = self.feature_scaler.transform(X)
        scores = self.model.predict(X_scaled)
        return normalize_scores(scores, self.prefer_gpu, self.prefer_numba)


class HistGBModel(CrossSectionalModel):
    def __init__(self, max_depth: int = 6, learning_rate: float = 0.04, max_iter: int = 250, **kwargs):
        super().__init__(name="HistGB", **kwargs)
        self.model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_scaled = self.feature_scaler.transform(X)
        scores = self.model.predict(X_scaled)
        return normalize_scores(scores, self.prefer_gpu, self.prefer_numba)


class NeuralMLPModel(CrossSectionalModel):
    def __init__(self, hidden_layer_sizes=(64, 32), alpha: float = 1e-3, max_iter: int = 500, **kwargs):
        super().__init__(name="NeuralMLP", **kwargs)
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_scaled = self.feature_scaler.transform(X)
        scores = self.model.predict(X_scaled)
        return normalize_scores(scores, self.prefer_gpu, self.prefer_numba)


class AdvancedEnsembleModel(CrossSectionalModel):
    """Stacked ensemble over linear, tree, and kernel-style learners."""

    def __init__(self, **kwargs):
        super().__init__(name="AdvancedEnsemble", **kwargs)
        self.model = StackingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=0.5)),
                ("rf", RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42, n_jobs=-1)),
                ("hgb", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=42)),
            ],
            final_estimator=Ridge(alpha=0.5),
            passthrough=True,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_scaled = self.feature_scaler.transform(X)
        scores = self.model.predict(X_scaled)
        return normalize_scores(scores, self.prefer_gpu, self.prefer_numba)


def create_model(model_type: str, **kwargs) -> CrossSectionalModel:
    key = model_type.lower()
    if key == "ridge":
        return RidgeModel(**kwargs)
    if key in ["gradient_boosting", "gb"]:
        return GradientBoostingModel(**kwargs)
    if key in ["quantum", "quantum_inspired", "qi"]:
        return QuantumInspiredModel(**kwargs)
    if key in ["random_forest", "rf"]:
        return RandomForestModel(**kwargs)
    if key in ["hist_gradient_boosting", "histgb", "hgb"]:
        return HistGBModel(**kwargs)
    if key in ["neural_mlp", "mlp", "nn"]:
        return NeuralMLPModel(**kwargs)
    if key in ["advanced_ensemble", "ensemble", "stacking"]:
        return AdvancedEnsembleModel(**kwargs)
    raise ValueError(f"Unknown model type: {model_type}")
