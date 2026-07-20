"""
Neural network model implementations.

Implements:
- Multi-layer Perceptron with skip connections
- Feature normalization for neural network training
- Early stopping and learning rate scheduling
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class NeuralMLPModel(BaseModel):
    """
    Multi-layer Perceptron regressor for equity ranking.
    
    Uses ReLU activations, Adam optimizer, and early stopping.
    Feature scaling is applied internally for neural network stability.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        **kwargs
    ):
        super().__init__(name="NeuralMLP", **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self._scaler = StandardScaler()
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> MLPRegressor:
        X_scaled = self._scaler.fit_transform(X)
        model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
            verbose=False,
        )
        model.fit(X_scaled, y)
        return model
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)
    
    def _get_params(self) -> Dict[str, Any]:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
        }
