"""
Abstract base class for all prediction models.

All models share a common interface for:
- Training and prediction
- Feature importance
- Probability calibration
- Model persistence
- Diagnostic metrics
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.exceptions import ModelError

logger = logging.getLogger(__name__)


@dataclass
class ModelDiagnostics:
    """Diagnostic information about a trained model."""
    
    is_fitted: bool = False
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)
    n_train_samples: int = 0
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_time: float = 0.0
    n_features_used: int = 0
    n_iterations: int = 0


class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, name: str = "base_model", random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.diagnostics = ModelDiagnostics()
        self.feature_importances_: Optional[pd.Series] = None
        self._model = None
        self._feature_names: List[str] = []
    
    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Internal fit implementation. Returns fitted model."""
        ...
    
    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Internal predict implementation. Returns predictions."""
        ...
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> "BaseModel":
        """
        Fit the model to training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Target variable (forward returns)
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
        **kwargs
            Additional fitting parameters
        
        Returns
        -------
        BaseModel
            Fitted model (self)
        """
        import time
        start = time.perf_counter()
        
        self._validate_input(X, y)
        self._feature_names = list(X.columns)
        
        X_arr = X.values.astype(np.float64)
        y_arr = y.values.astype(np.float64)
        
        # Handle missing values
        valid_mask = ~(np.isnan(X_arr).any(axis=1) | np.isnan(y_arr))
        X_clean = X_arr[valid_mask]
        y_clean = y_arr[valid_mask]
        
        if len(X_clean) < 10:
            raise ModelError(
                f"Not enough valid samples: {len(X_clean)} after removing NaN"
            )
        
        try:
            self._model = self._fit(X_clean, y_clean, **kwargs)
        except Exception as e:
            raise ModelError(f"Model '{self.name}' fit failed: {e}") from e
        
        # Compute diagnostics
        elapsed = time.perf_counter() - start
        train_pred = self._predict(X_clean)
        train_score = self._score(y_clean, train_pred)
        
        val_score = None
        if X_val is not None and y_val is not None:
            Xv_arr = X_val.values.astype(np.float64)
            yv_arr = y_val.values.astype(np.float64)
            val_pred = self._predict(Xv_arr)
            val_score = self._score(yv_arr, val_pred)
        
        self.diagnostics = ModelDiagnostics(
            is_fitted=True,
            n_features=X_clean.shape[1],
            feature_names=self._feature_names,
            n_train_samples=len(X_clean),
            train_score=train_score,
            val_score=val_score,
            hyperparameters=self._get_params(),
            training_time=elapsed,
            n_features_used=len(self._feature_names),
        )
        
        # Compute feature importance
        self.feature_importances_ = self._compute_feature_importance()
        
        logger.info(
            f"Model '{self.name}' fitted: "
            f"train_score={train_score if train_score is not None else 'N/A'}, "
            f"val_score={val_score if val_score is not None else 'N/A'}, "
            f"samples={len(X_clean)}, "
            f"features={len(self._feature_names)}, "
            f"time={elapsed:.2f}s"
        )
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        return_std: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for input features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        return_std : bool
            If True, return prediction standard error estimates
        
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Predictions, and optionally prediction standard errors
        """
        if not self.diagnostics.is_fitted:
            raise ModelError("Model not fitted yet")
        
        X_arr = X[self._feature_names].values.astype(np.float64)
        
        predictions = self._predict(X_arr)
        
        if return_std:
            # Simple prediction interval estimation
            n = self.diagnostics.n_train_samples
            if n > 1:
                residuals_std = self._estimate_residual_std()
                se = residuals_std * np.sqrt(
                    1 + 1/n + np.sum((X_arr - X_arr.mean(axis=0))**2, axis=1) /
                    np.sum((X_arr - X_arr.mean(axis=0))**2, axis=0)
                )
            else:
                se = np.full_like(predictions, np.nan)
            return predictions, se
        
        return predictions
    
    def predict_ranked(
        self,
        X: pd.DataFrame,
        direction: str = "descending"
    ) -> pd.DataFrame:
        """
        Generate ranked predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        direction : str
            'descending' (higher score = higher rank 1)
            'ascending' (lower score = higher rank 1)
        
        Returns
        -------
        pd.DataFrame
            Ranked predictions with scores
        """
        predictions = self.predict(X)
        scores = pd.Series(predictions, index=X.index)
        
        if direction == "descending":
            ranks = scores.rank(ascending=False, method="min")
        else:
            ranks = scores.rank(ascending=True, method="min")
        
        result = pd.DataFrame({
            "score": scores,
            "rank": ranks,
        })
        result = result.sort_values("rank")
        return result
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available."""
        return self.feature_importances_
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self._get_params()
    
    def summary(self) -> str:
        """Get a text summary of the model."""
        lines = [
            f"Model: {self.name}",
            f"Fitted: {self.diagnostics.is_fitted}",
            f"Features: {self.diagnostics.n_features}",
            f"Training samples: {self.diagnostics.n_train_samples}",
        ]
        if self.diagnostics.train_score is not None:
            lines.append(f"Train R²: {self.diagnostics.train_score:.4f}")
        if self.diagnostics.val_score is not None:
            lines.append(f"Val R²: {self.diagnostics.val_score:.4f}")
        if self.diagnostics.training_time > 0:
            lines.append(f"Training time: {self.diagnostics.training_time:.2f}s")
        return "\n".join(lines)
    
    # ── Abstract / Override Methods ────────────────────────────────────
    
    def _get_params(self) -> Dict[str, Any]:
        """Get model parameters for diagnostics."""
        return {}
    
    def _compute_feature_importance(self) -> Optional[pd.Series]:
        """Compute feature importance. Override in subclasses that support it."""
        return None
    
    def _estimate_residual_std(self) -> float:
        """Estimate residual standard deviation."""
        return 1.0
    
    # ── Private Helpers ────────────────────────────────────────────────
    
    @staticmethod
    def _validate_input(X: pd.DataFrame, y: pd.Series) -> None:
        """Validate input data."""
        if X.empty:
            raise ModelError("Empty feature matrix")
        if y.empty:
            raise ModelError("Empty target variable")
        if len(X) != len(y):
            raise ModelError(
                f"Feature/Target mismatch: X={len(X)}, y={len(y)}"
            )
        if X.isnull().all().all():
            raise ModelError("All feature values are NaN")
        if y.isnull().all():
            raise ModelError("All target values are NaN")
    
    @staticmethod
    def _score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-12:
            return 0.0
        return float(1 - ss_res / ss_tot)
    
    def __str__(self) -> str:
        return f"{self.name}(fitted={self.diagnostics.is_fitted})"
    
    def __repr__(self) -> str:
        return self.__str__()
