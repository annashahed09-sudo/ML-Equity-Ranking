"""
Ensemble model implementations for robust cross-sectional prediction.

Implements:
- Stacking ensemble with meta-learner
- Voting ensemble (average of model predictions)
- Weighted ensemble with learned weights
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from core.exceptions import ModelError
from .base import BaseModel, ModelDiagnostics
from .linear import RidgeModel


class StackingEnsemble(BaseModel):
    """
    Stacking ensemble with base models and meta-learner.
    
    Combines predictions from heterogeneous base models using a
    meta-learner (default: Ridge regression) trained on out-of-fold
    predictions to prevent overfitting.
    """
    
    def __init__(
        self,
        base_models: Optional[List[BaseModel]] = None,
        meta_learner: Optional[BaseModel] = None,
        use_cv_predictions: bool = True,
        n_folds: int = 5,
        **kwargs
    ):
        # Remove 'name' from kwargs if present to avoid duplicate arg error
        kwargs.pop("name", None)
        super().__init__(name="StackingEnsemble", **kwargs)
        self.base_models = base_models or self._default_base_models()
        self.meta_learner = meta_learner or RidgeModel(alpha=0.5)
        self.use_cv_predictions = use_cv_predictions
        self.n_folds = n_folds
        self._scaler = StandardScaler()
    
    def _default_base_models(self) -> List[BaseModel]:
        """Create default diverse base model set."""
        return [
            RidgeModel(alpha=1.0, name="Ridge_1"),
            RidgeModel(alpha=0.1, name="Ridge_01"),
        ]
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        from sklearn.model_selection import KFold
        
        # Train base models and store fitted models
        for model in self.base_models:
            model._model = model._fit(X, y)
        
        # Generate meta-features
        if self.use_cv_predictions:
            kf = KFold(n_splits=self.n_folds, shuffle=False)
            meta_features = np.zeros((len(X), len(self.base_models)))
            for i, model in enumerate(self.base_models):
                for train_idx, val_idx in kf.split(X):
                    model_copy = model.__class__(**model._get_params())
                    model_copy._model = model_copy._fit(X[train_idx], y[train_idx])
                    meta_features[val_idx, i] = model_copy._predict(X[val_idx])
        else:
            meta_features = np.column_stack([
                model._predict(X) for model in self.base_models
            ])
        
        # Train meta-learner
        meta_scaled = self._scaler.fit_transform(meta_features)
        self.meta_learner._model = self.meta_learner._fit(meta_scaled, y)
        
        return self.meta_learner._model
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        # Generate base predictions
        base_preds = np.column_stack([
            model._predict(X) for model in self.base_models
        ])
        # Meta prediction
        meta_scaled = self._scaler.transform(base_preds)
        return self.meta_learner._predict(meta_scaled)


class VotingEnsemble(BaseModel):
    """
    Voting ensemble: simple average of diverse model predictions.
    
    Equal-weight averaging of base model predictions. The simplicity
    of equal weighting often outperforms learned weighting schemes
    in out-of-sample settings.
    """
    
    def __init__(
        self,
        base_models: Optional[List[BaseModel]] = None,
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        # Remove 'name' from kwargs if present to avoid duplicate arg error
        kwargs.pop("name", None)
        super().__init__(name="VotingEnsemble", **kwargs)
        self.base_models = base_models or self._default_base_models()
        self._weights = weights
    
    def _default_base_models(self) -> List[BaseModel]:
        return [
            RidgeModel(alpha=1.0, name="Ridge"),
            RidgeModel(alpha=0.5, name="Ridge_05"),
        ]
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        for model in self.base_models:
            model._model = model._fit(X, y)
        return None
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.column_stack([m._predict(X) for m in self.base_models])
        if self._weights:
            weights = np.array(self._weights) / sum(self._weights)
            return preds @ weights
        return preds.mean(axis=1)
