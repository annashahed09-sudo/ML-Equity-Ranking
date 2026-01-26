"""
Model wrappers for cross-sectional equity prediction.

Models output normalized scores (roughly standard normal distribution) that represent
the model's ranking of each asset at each time step. These scores are used directly
for portfolio construction.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class CrossSectionalModel:
    """
    Base class for cross-sectional prediction models.
    
    Models are trained on standardized features and output normalized scores.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit model on training features and target.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (features as columns)
        y : pd.Series
            Target (next-period cross-sectional rank or returns)
        """
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict normalized scores.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        np.ndarray
            Normalized prediction scores (approximately zero-mean, unit variance)
        """
        raise NotImplementedError
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Return feature importances if available."""
        return None


class RidgeModel(CrossSectionalModel):
    """
    Ridge regression baseline for linear cross-sectional signals.
    
    Simple, interpretable, and good for understanding feature relationships.
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__(name="Ridge")
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Ridge regression on (standardized) features."""
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and normalize scores."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        X_scaled = self.feature_scaler.transform(X)
        scores = self.model.predict(X_scaled)
        
        # Normalize to approximately standard normal
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        
        return scores


class GradientBoostingModel(CrossSectionalModel):
    """
    Gradient Boosting for capturing nonlinear cross-sectional patterns.
    
    Requires careful tuning to avoid overfitting in weak signal regime.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        random_state: int = 42
    ):
        super().__init__(name="GradientBoosting")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state
        )
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Gradient Boosting on features."""
        self.feature_names = X.columns.tolist()
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and normalize scores."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        X_scaled = self.feature_scaler.transform(X)
        scores = self.model.predict(X_scaled)
        
        # Normalize to approximately standard normal
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        
        return scores
    
    def get_feature_importance(self) -> pd.Series:
        """Return feature importances from Gradient Boosting."""
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)


def create_model(model_type: str, **kwargs) -> CrossSectionalModel:
    """
    Factory function to create model instances.

    Parameters
    ----------
    model_type : str
        'ridge' or 'gradient_boosting'
    **kwargs
        Model-specific hyperparameters

    Returns
    -------
    CrossSectionalModel
        Fitted or unfitted model instance
    """
    if model_type.lower() == 'ridge':
        return RidgeModel(**kwargs)
    elif model_type.lower() in ['gradient_boosting', 'gb']:
        return GradientBoostingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
