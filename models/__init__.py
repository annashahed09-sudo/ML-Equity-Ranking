"""
Machine learning model implementations for cross-sectional equity prediction.

Provides a comprehensive suite of models including:
- Linear models (Ridge, ElasticNet, Lasso)
- Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost)
- Neural networks (MLP, simple transformers)
- Learning-to-rank models (LightGBM Ranker, XGBoost Ranker)
- Ensemble methods (stacking, voting)
- Hyperparameter optimization (Bayesian, grid)
"""

from .base import BaseModel
from .linear import RidgeModel, ElasticNetModel, LassoModel
from .tree import RandomForestModel, XGBoostModel, LightGBMModel, CatBoostModel
from .neural import NeuralMLPModel
from .ensemble import StackingEnsemble, VotingEnsemble
from .ranker import LightGBMRanker, XGBoostRanker
from .factory import ModelFactory
from .tuning import BayesianOptimizer, GridOptimizer

__all__ = [
    "BaseModel",
    "RidgeModel",
    "ElasticNetModel",
    "LassoModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "NeuralMLPModel",
    "StackingEnsemble",
    "VotingEnsemble",
    "LightGBMRanker",
    "XGBoostRanker",
    "ModelFactory",
    "BayesianOptimizer",
    "GridOptimizer",
]
