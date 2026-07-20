"""
Tree-based model implementations.

Implements:
- Random Forest (baseline ensemble)
- XGBoost (gradient boosting with regularization)
- LightGBM (efficient gradient boosting)
- CatBoost (ordered boosting with categorical support)

All models support feature importance and probability calibration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 300, max_depth: Optional[int] = 8,
                 min_samples_leaf: int = 5, n_jobs: int = -1, **kwargs):
        super().__init__(name="RandomForest", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs

    def _fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        model.fit(X, y)
        return model

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def _compute_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None or not self._feature_names:
            return None
        return pd.Series(self._model.feature_importances_, index=self._feature_names).sort_values(ascending=False)

    def _get_params(self) -> Dict[str, Any]:
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf}


class XGBoostModel(BaseModel):
    def __init__(self, n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, min_child_weight: int = 1,
                 **kwargs):
        super().__init__(name="XGBoost", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight

    def _fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("Install xgboost: pip install xgboost")
        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            verbosity=0,
        )
        model.fit(X, y)
        return model

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def _compute_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None or not self._feature_names:
            return None
        importance = self._model.feature_importances_
        return pd.Series(importance, index=self._feature_names).sort_values(ascending=False)


class LightGBMModel(BaseModel):
    def __init__(self, num_leaves: int = 31, learning_rate: float = 0.05,
                 n_estimators: int = 500, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_alpha: float = 0.0,
                 reg_lambda: float = 0.0, num_threads: int = -1, **kwargs):
        super().__init__(name="LightGBM", **kwargs)
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.num_threads = num_threads

    def _fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("Install lightgbm: pip install lightgbm")
        model = lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            num_threads=self.num_threads,
            random_state=self.random_state,
            verbose=-1,
        )
        model.fit(X, y)
        return model

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def _compute_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None or not self._feature_names:
            return None
        importance = self._model.feature_importances_
        return pd.Series(importance, index=self._feature_names).sort_values(ascending=False)


class CatBoostModel(BaseModel):
    def __init__(self, iterations: int = 500, depth: int = 6,
                 learning_rate: float = 0.05, l2_leaf_reg: float = 3.0,
                 random_seed: int = 42, **kwargs):
        super().__init__(name="CatBoost", **kwargs)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

    def _fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("Install catboost: pip install catboost")
        model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=False,
        )
        model.fit(X, y)
        return model

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def _compute_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None or not self._feature_names:
            return None
        importance = self._model.get_feature_importance()
        return pd.Series(importance, index=self._feature_names).sort_values(ascending=False)
