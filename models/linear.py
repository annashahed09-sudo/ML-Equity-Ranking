"""
Linear model implementations for cross-sectional equity prediction.

Implements:
- Ridge regression (L2 regularization)
- ElasticNet (L1 + L2)
- Lasso regression (L1 regularization)

All models include automatic feature scaling and
cross-sectional score normalization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler

from core.utils import zscore_normalize
from .base import BaseModel


class RidgeModel(BaseModel):
    """
    Ridge regression with L2 regularization.
    
    Implements Tikhonov regularization for shrinkage of coefficients.
    Well-suited as a baseline for cross-sectional factor models.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        **kwargs
    ):
        super().__init__(name="Ridge", **kwargs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._scaler = StandardScaler()
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> Ridge:
        X_scaled = self._scaler.fit_transform(X)
        model = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
        )
        model.fit(X_scaled, y)
        return model
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        scores = self._model.predict(X_scaled)
        return zscore_normalize(pd.Series(scores)).values
    
    def _get_params(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "fit_intercept": self.fit_intercept}
    
    def _compute_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None or not self._feature_names:
            return None
        return pd.Series(
            np.abs(self._model.coef_),
            index=self._feature_names,
        ).sort_values(ascending=False)


class ElasticNetModel(BaseModel):
    """
    Elastic Net regression (L1 + L2 regularization).
    
    Combines L1 (Lasso) and L2 (Ridge) penalties. Useful when
    feature selection is desired but Lasso alone is too aggressive.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        **kwargs
    ):
        super().__init__(name="ElasticNet", **kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self._scaler = StandardScaler()
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> ElasticNet:
        X_scaled = self._scaler.fit_transform(X)
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(X_scaled, y)
        return model
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)
    
    def _get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
        }
    
    def _compute_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None or not self._feature_names:
            return None
        return pd.Series(
            np.abs(self._model.coef_),
            index=self._feature_names,
        ).sort_values(ascending=False)


class LassoModel(BaseModel):
    """
    Lasso regression (L1 regularization).
    
    Performs feature selection by shrinking coefficients to zero.
    Useful for sparse factor models with many irrelevant features.
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        **kwargs
    ):
        super().__init__(name="Lasso", **kwargs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self._scaler = StandardScaler()
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> Lasso:
        X_scaled = self._scaler.fit_transform(X)
        model = Lasso(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(X_scaled, y)
        return model
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)
    
    def _get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
        }
    
    def _compute_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None or not self._feature_names:
            return None
        return pd.Series(
            np.abs(self._model.coef_),
            index=self._feature_names,
        ).sort_values(ascending=False)
