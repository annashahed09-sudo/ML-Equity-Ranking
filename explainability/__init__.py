"""
Model explainability and interpretability module.

Implements:
- SHAP (SHapley Additive exPlanations)
- Permutation feature importance
- Partial dependence plots
- Individual conditional expectation (ICE)
- Feature interaction analysis
- Model confidence estimation
"""

from .importance import PermutationImportance
from .partial_dep import PartialDependence

__all__ = [
    "PermutationImportance",
    "PartialDependence",
]
