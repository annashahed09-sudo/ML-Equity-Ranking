"""
Research pipeline for systematic quantitative experimentation.

Orchestrates:
- Factor computation
- Model training and evaluation
- Walk-forward validation
- Portfolio backtesting
- Performance reporting
- Experiment tracking
"""

from .pipeline import ResearchPipeline, PipelineResult
from .experiments import ExperimentTracker

__all__ = [
    "ResearchPipeline",
    "PipelineResult",
    "ExperimentTracker",
]
