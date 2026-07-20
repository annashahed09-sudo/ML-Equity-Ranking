"""
Signal orthogonalization to remove redundancy between correlated signals.

Implements:
- Gram-Schmidt orthogonalization (sequential)
- PCA-based orthogonalization
- Residualization (regression-based)
- Symmetric orthogonalization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from core.exceptions import SignalError


class SignalOrthogonalizer:
    """
    Orthogonalization methods for correlated signals.
    
    Removing common variance between signals improves diversification
    and reduces multicollinearity in downstream models.
    """
    
    @staticmethod
    def gram_schmidt(signals: pd.DataFrame, order: list[str]) -> pd.DataFrame:
        """
        Gram-Schmidt orthogonalization.
        
        Sequentially orthogonalizes signals in the given order.
        Earlier signals are preserved, later signals are adjusted
        to remove correlation with earlier ones.
        """
        orthogonalized = pd.DataFrame(index=signals.index)
        seen = []
        
        for name in order:
            if name not in signals.columns:
                continue
            
            signal = signals[name].values.copy()
            
            # Project out previously orthogonalized signals
            for prev_name in seen:
                prev = orthogonalized[prev_name].values
                if np.std(prev) > 1e-12:
                    projection = (np.dot(signal, prev) / np.dot(prev, prev)) * prev
                    signal = signal - projection
            
            orthogonalized[name] = signal
            seen.append(name)
        
        return orthogonalized
    
    @staticmethod
    def pca(signals: pd.DataFrame, n_components: int | None = None) -> pd.DataFrame:
        """
        PCA-based orthogonalization.
        
        Transforms signals to principal components, removing
        all linear correlations. If n_components < n_signals,
        this also reduces dimensionality.
        """
        if signals.empty:
            raise SignalError("Empty signals DataFrame")
        
        n = signals.shape[1]
        if n_components is None:
            n_components = n
        
        # Handle missing values
        clean = signals.fillna(signals.mean())
        
        pca = PCA(n_components=min(n_components, n))
        components = pca.fit_transform(clean)
        
        result = pd.DataFrame(
            components,
            index=signals.index,
            columns=[f"PC{i+1}" for i in range(components.shape[1])],
        )
        
        return result
    
    @staticmethod
    def residualize(
        target_signal: pd.Series,
        conditioning_signals: pd.DataFrame,
    ) -> pd.Series:
        """
        Residualize target signal with respect to conditioning signals.
        
        Returns the part of the target signal that is orthogonal to
        (uncorrelated with) the conditioning signals.
        """
        from sklearn.linear_model import LinearRegression
        
        clean_target = target_signal.fillna(target_signal.mean())
        clean_conditioning = conditioning_signals.fillna(conditioning_signals.mean())
        
        model = LinearRegression()
        model.fit(clean_conditioning, clean_target)
        predicted = model.predict(clean_conditioning)
        
        residual = clean_target - predicted
        return pd.Series(residual, index=target_signal.index, name=target_signal.name)
