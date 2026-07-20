"""
Learning-to-rank model implementations.

Implements:
- LightGBM Ranker (LambdaRank)
- XGBoost Ranker (pairwise ranking)

Ranking models are preferred over regression for equity ranking tasks
since we care about relative ordering, not absolute return prediction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseModel


class LightGBMRanker(BaseModel):
    """
    LightGBM LambdaRank implementation for cross-sectional ranking.
    
    Learning-to-rank models optimize directly for ranking metrics
    (NDCG, MAP) rather than regression metrics. This is preferred
    for equity ranking where relative ordering matters.
    """
    
    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        num_threads: int = -1,
        **kwargs
    ):
        super().__init__(name="LightGBMRanker", **kwargs)
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.num_threads = num_threads
        self._feature_names_ = []
    
    def _fit(self, X: np.ndarray, y: np.ndarray, group: Optional[np.ndarray] = None) -> Any:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required for LightGBMRanker. Install: pip install lightgbm")
        
        # Create query groups (required for ranking)
        if group is None:
            # Default: all data is one group
            group = np.array([len(X)])
        
        train_data = lgb.Dataset(
            X, label=y, group=group,
            feature_name=self._feature_names if self._feature_names else None,
        )
        
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5, 10],
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "num_threads": self.num_threads,
            "verbosity": -1,
            "seed": self.random_state,
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data],
        )
        
        self._feature_names_ = model.feature_name()
        return model
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
    
    def _compute_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None:
            return None
        import lightgbm as lgb
        importance = self._model.feature_importance(importance_type="gain")
        return pd.Series(importance, index=self._feature_names_).sort_values(ascending=False)


class XGBoostRanker(BaseModel):
    """
    XGBoost ranking implementation.
    
    Uses pairwise ranking objective optimized for NDCG.
    Well-suited for cross-sectional ranking of equities.
    """
    
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        **kwargs
    ):
        super().__init__(name="XGBoostRanker", **kwargs)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
    
    def _fit(self, X: np.ndarray, y: np.ndarray, group: Optional[np.ndarray] = None) -> Any:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is required for XGBoostRanker. Install: pip install xgboost")
        
        params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@5",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "seed": self.random_state,
            "verbosity": 0,
        }
        
        if group is None:
            group = np.array([len(X)])
        
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(group)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False,
        )
        
        return model
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is required")
        dmatrix = xgb.DMatrix(X)
        return self._model.predict(dmatrix)
    
    def _compute_feature_importance(self) -> Optional[pd.Series]:
        if self._model is None or not self._feature_names:
            return None
        importance = self._model.get_score(importance_type="gain")
        return pd.Series(importance).reindex(self._feature_names, fill_value=0).sort_values(ascending=False)
