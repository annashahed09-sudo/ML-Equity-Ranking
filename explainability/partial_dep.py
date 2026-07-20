"""
Partial dependence and Individual Conditional Expectation (ICE) plots.

Implements:
- Partial dependence (marginal effect of a feature)
- ICE (per-instance feature effect)
- Feature interaction analysis
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd


class PartialDependence:
    """
    Partial dependence computation.
    
    Shows the marginal effect of a feature on the model's predictions
    after averaging out the effects of all other features.
    """
    
    def __init__(self, model: Callable, X: pd.DataFrame):
        self.model = model
        self.X = X
    
    def compute(
        self,
        feature: str,
        grid_size: int = 50,
        ice: bool = False,
    ) -> pd.DataFrame:
        """
        Compute partial dependence for a single feature.
        
        Parameters
        ----------
        feature : str
            Feature name
        grid_size : int
            Number of grid points
        ice : bool
            If True, also return ICE curves
        
        Returns
        -------
        pd.DataFrame
            Partial dependence values
        """
        if feature not in self.X.columns:
            raise ValueError(f"Feature '{feature}' not found")
        
        # Create grid
        values = self.X[feature]
        grid = np.linspace(values.min(), values.max(), grid_size)
        
        partial = []
        ice_curves = []
        
        for val in grid:
            X_modified = self.X.copy()
            X_modified[feature] = val
            preds = self.model(X_modified)
            partial.append(preds.mean())
            
            if ice:
                ice_curves.append(preds)
        
        result = pd.DataFrame({
            "feature_value": grid,
            "partial_dependence": partial,
        })
        
        if ice and ice_curves:
            ice_df = pd.DataFrame(
                np.column_stack(ice_curves),
                columns=[f"ice_{i}" for i in range(len(grid))],
            )
            result = pd.concat([result, ice_df], axis=1)
        
        return result
    
    def compute_interaction(
        self,
        feature1: str,
        feature2: str,
        grid_size: int = 20,
    ) -> pd.DataFrame:
        """
        Compute 2-way partial dependence (feature interaction).
        
        Returns a DataFrame with the 2D grid of predictions.
        """
        grid1 = np.linspace(self.X[feature1].min(), self.X[feature1].max(), grid_size)
        grid2 = np.linspace(self.X[feature2].min(), self.X[feature2].max(), grid_size)
        
        results = []
        for v1 in grid1:
            row = []
            for v2 in grid2:
                X_modified = self.X.copy()
                X_modified[feature1] = v1
                X_modified[feature2] = v2
                row.append(self.model(X_modified).mean())
            results.append(row)
        
        return pd.DataFrame(
            results,
            index=[f"{feature1}={v:.2f}" for v in grid1],
            columns=[f"{feature2}={v:.2f}" for v in grid2],
        )
