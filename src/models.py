"""
Model wrappers for cross-sectional equity prediction.

Models output normalized scores (roughly standard normal distribution) that represent
the model's ranking of each asset at each time step. These scores are used directly
for portfolio construction.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline

from .accelerated import normalize_scores


class CrossSectionalModel:
    """
    Base class for cross-sectional prediction models.

    Models are trained on standardized features and output normalized scores.
    """

    def __init__(self, name: str, prefer_gpu: bool = True):
        self.name = name
        self.model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        self.prefer_gpu = prefer_gpu

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Return feature importances if available."""
        return None


class RidgeModel(CrossSectionalModel):
    """Ridge regression baseline for linear cross-sectional signals."""

    def __init__(self, alpha: float = 1.0, prefer_gpu: bool = True):
        super().__init__(name="Ridge", prefer_gpu=prefer_gpu)
        self.alpha = alpha
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
        return normalize_scores(scores, prefer_gpu=self.prefer_gpu)


class GradientBoostingModel(CrossSectionalModel):
    """Gradient Boosting for capturing nonlinear cross-sectional patterns."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        random_state: int = 42,
        prefer_gpu: bool = True,
    ):
        super().__init__(name="GradientBoosting", prefer_gpu=prefer_gpu)
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
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
        return normalize_scores(scores, prefer_gpu=self.prefer_gpu)

    def get_feature_importance(self) -> Optional[pd.Series]:
        if not self.is_fitted or self.feature_names is None:
            return None

        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)


class QuantumInspiredModel(CrossSectionalModel):
    """
    Quantum-inspired regression model.

    Uses random Fourier feature expansion (RBFSampler) as a practical approximation
    to high-dimensional kernel mappings that are commonly used in quantum-inspired ML.
    """

    def __init__(
        self,
        n_components: int = 256,
        gamma: float = 1.0,
        alpha: float = 1.0,
        random_state: int = 42,
        prefer_gpu: bool = True,
    ):
        super().__init__(name="QuantumInspired", prefer_gpu=prefer_gpu)
        self.model = Pipeline(
            [
                ("rff", RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)),
                ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
            ]
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
        return normalize_scores(scores, prefer_gpu=self.prefer_gpu)


def create_model(model_type: str, **kwargs) -> CrossSectionalModel:
    """Factory function to create model instances."""
    key = model_type.lower()
    if key == "ridge":
        return RidgeModel(**kwargs)
    if key in ["gradient_boosting", "gb"]:
        return GradientBoostingModel(**kwargs)
    if key in ["quantum", "quantum_inspired", "qi"]:
        return QuantumInspiredModel(**kwargs)
    raise ValueError(f"Unknown model type: {model_type}")
