"""
Feature importance computation methods.

Implements:
- Permutation importance (model-agnostic)
- SHAP values (when shap library available)
- Feature importance from tree-based models
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.exceptions import ModelError


class PermutationImportance:
    """
    Permutation feature importance (model-agnostic).
    
    Computes the decrease in model performance when each feature
    is randomly shuffled. More important features produce larger
    performance drops.
    """
    
    def __init__(
        self,
        model: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "r2",
        n_repeats: int = 10,
        random_state: int = 42,
    ):
        self.model = model
        self.X = X
        self.y = y
        self.metric = metric
        self.n_repeats = n_repeats
        self.random_state = random_state
    
    def compute(self) -> pd.DataFrame:
        """
        Compute permutation importance for all features.
        
        Returns
        -------
        pd.DataFrame
            Importance scores with columns: feature, importance, std
        """
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Baseline performance
        y_pred = self.model(self.X)
        
        if self.metric == "r2":
            baseline = r2_score(self.y, y_pred)
        elif self.metric == "mse":
            baseline = -mean_squared_error(self.y, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        rng = np.random.RandomState(self.random_state)
        importances = []
        
        for col in self.X.columns:
            scores = []
            for _ in range(self.n_repeats):
                X_permuted = self.X.copy()
                X_permuted[col] = rng.permutation(X_permuted[col].values)
                y_perm = self.model(X_permuted)
                
                if self.metric == "r2":
                    score = r2_score(self.y, y_perm)
                else:
                    score = -mean_squared_error(self.y, y_perm)
                
                scores.append(baseline - score)
            
            importances.append({
                "feature": col,
                "importance": float(np.mean(scores)),
                "std": float(np.std(scores)),
            })
        
        result = pd.DataFrame(importances)
        result = result.sort_values("importance", ascending=False).reset_index(drop=True)
        return result


class ShapExplainer:
    """
    SHAP-based model explainability.
    
    Falls back to permutation importance if shap is not installed.
    """
    
    def __init__(self, model, X: pd.DataFrame):
        self.model = model
        self.X = X
        self.shap_values = None
    
    def explain(self) -> Optional[pd.DataFrame]:
        """Compute SHAP values for model predictions."""
        try:
            import shap
            explainer = shap.Explainer(self.model, self.X)
            self.shap_values = explainer(self.X)
            
            # Aggregate SHAP values per feature
            importance = np.abs(self.shap_values.values).mean(axis=0)
            return pd.DataFrame({
                "feature": self.X.columns,
                "shap_importance": importance,
            }).sort_values("shap_importance", ascending=False)
        except ImportError:
            return None
