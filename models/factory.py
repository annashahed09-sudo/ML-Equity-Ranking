"""
Model factory: creates models by name with parameter injection.

Supports dynamic model creation from configuration, enabling
experiment-driven model selection without code changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import settings
from core.exceptions import ModelError

from .base import BaseModel
from .linear import RidgeModel, ElasticNetModel, LassoModel
from .ensemble import StackingEnsemble, VotingEnsemble
from .ranker import LightGBMRanker, XGBoostRanker

# Optional tree models (require extra dependencies)
try:
    from .tree import RandomForestModel, XGBoostModel, LightGBMModel, CatBoostModel
    HAS_XGB = True
    HAS_LGBM = True
    HAS_CATBOOST = True
except ImportError:
    HAS_XGB = HAS_LGBM = HAS_CATBOOST = False

try:
    from .neural import NeuralMLPModel
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False


_MODEL_REGISTRY: Dict[str, type] = {
    "ridge": RidgeModel,
    "elastic_net": ElasticNetModel,
    "lasso": LassoModel,
    "stacking_ensemble": StackingEnsemble,
    "voting_ensemble": VotingEnsemble,
    "lgbm_ranker": LightGBMRanker,
    "xgb_ranker": XGBoostRanker,
}

_MODEL_REGISTRY["random_forest"] = RandomForestModel
_MODEL_REGISTRY["rf"] = RandomForestModel

if HAS_XGB:
    _MODEL_REGISTRY["xgboost"] = XGBoostModel
    _MODEL_REGISTRY["xgb_ranker"] = XGBoostRanker

if HAS_LGBM:
    _MODEL_REGISTRY["lightgbm"] = LightGBMModel
    _MODEL_REGISTRY["lgbm_ranker"] = LightGBMRanker

if HAS_CATBOOST:
    _MODEL_REGISTRY["catboost"] = CatBoostModel

if HAS_NEURAL:
    _MODEL_REGISTRY["neural_mlp"] = NeuralMLPModel
    _MODEL_REGISTRY["mlp"] = NeuralMLPModel


class ModelFactory:
    """Creates and configures model instances by name."""
    
    @classmethod
    def create(
        cls,
        model_type: str,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance by type name.
        
        Parameters
        ----------
        model_type : str
            Model type key (e.g., 'ridge', 'lightgbm', 'xgboost')
        **kwargs
            Model-specific parameters (overrides defaults)
        
        Returns
        -------
        BaseModel
            Configured model instance
        
        Raises
        ------
        ModelError
            If model type is unknown or dependencies are missing
        """
        key = model_type.lower().replace("-", "_")
        
        if key not in _MODEL_REGISTRY:
            available = list(_MODEL_REGISTRY.keys())
            raise ModelError(
                f"Unknown model type: '{model_type}'. "
                f"Available: {available}"
            )
        
        model_class = _MODEL_REGISTRY[key]
        
        # Merge default params from settings with provided kwargs
        params = cls._default_params(key)
        params.update(kwargs)
        
        return model_class(**params)
    
    @classmethod
    def available_models(cls) -> List[str]:
        """List available model types."""
        return list(_MODEL_REGISTRY.keys())
    
    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """Register a custom model class."""
        _MODEL_REGISTRY[name.lower()] = model_class
    
    @staticmethod
    def _default_params(model_type: str) -> Dict[str, Any]:
        """Get default parameters for a model type."""
        defaults = {
            "ridge": {"alpha": 1.0},
            "elastic_net": {
                "alpha": settings.ELASTIC_NET_ALPHA,
                "l1_ratio": settings.ELASTIC_NET_L1_RATIO,
            },
            "lasso": {"alpha": 0.01},
            "lightgbm": {
                "num_leaves": settings.LIGHTGBM_NUM_LEAVES,
                "learning_rate": settings.LIGHTGBM_LEARNING_RATE,
                "n_estimators": settings.LIGHTGBM_N_ESTIMATORS,
            },
            "xgboost": {
                "max_depth": settings.XGBOOST_MAX_DEPTH,
                "learning_rate": settings.XGBOOST_LEARNING_RATE,
                "n_estimators": settings.XGBOOST_N_ESTIMATORS,
            },
            "catboost": {
                "depth": settings.CATBOOST_DEPTH,
                "learning_rate": settings.CATBOOST_LEARNING_RATE,
                "iterations": settings.CATBOOST_ITERATIONS,
            },
        }
        return defaults.get(model_type, {})
