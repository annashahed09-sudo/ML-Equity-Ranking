"""
Experiment tracking for systematic research.

Records:
- Model configurations
- Hyperparameters
- Performance metrics
- Feature sets used
- Git commit hash
- Execution time
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import settings
from core.types import ExperimentResult

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Lightweight experiment tracker.
    
    Records experiment configurations and results to disk for
    later analysis and comparison.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.EXPERIMENT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_experiment: Optional[ExperimentResult] = None
    
    def start_experiment(
        self,
        name: str,
        model_name: str,
        factor_set: List[str],
        validation_strategy: str,
        hyperparameters: Dict[str, Any],
    ) -> str:
        """Start a new experiment and return its ID."""
        experiment_id = f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
        
        self._current_experiment = ExperimentResult(
            experiment_id=experiment_id,
            model_name=model_name,
            factor_set=factor_set,
            validation_strategy=validation_strategy,
            hyperparameters=hyperparameters,
            metrics={},
            timestamp=datetime.now(),
        )
        
        logger.info(f"Starting experiment: {experiment_id}")
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Record metrics for the current experiment."""
        if self._current_experiment is not None:
            self._current_experiment.metrics.update(metrics)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Record additional parameters."""
        if self._current_experiment is not None:
            self._current_experiment.hyperparameters.update(params)
    
    def finish_experiment(self, duration: Optional[float] = None) -> ExperimentResult:
        """Finalize the current experiment and save to disk."""
        if self._current_experiment is None:
            raise RuntimeError("No active experiment")
        
        if duration is not None:
            self._current_experiment.duration_seconds = duration
        
        # Get git commit
        try:
            self._current_experiment.git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            self._current_experiment.git_commit = "unknown"
        
        # Save to disk
        self._save_experiment(self._current_experiment)
        
        result = self._current_experiment
        self._current_experiment = None
        
        logger.info(
            f"Experiment {result.experiment_id} completed: "
            f"{result.metrics}"
        )
        
        return result
    
    def _save_experiment(self, experiment: ExperimentResult) -> None:
        """Save experiment to JSON file."""
        filepath = self.output_dir / f"{experiment.experiment_id}.json"
        
        data = {
            "experiment_id": experiment.experiment_id,
            "model_name": experiment.model_name,
            "factor_set": experiment.factor_set,
            "validation_strategy": experiment.validation_strategy,
            "hyperparameters": experiment.hyperparameters,
            "metrics": experiment.metrics,
            "timestamp": experiment.timestamp.isoformat(),
            "duration_seconds": experiment.duration_seconds,
            "git_commit": experiment.git_commit,
            "notes": experiment.notes,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load a saved experiment."""
        filepath = self.output_dir / f"{experiment_id}.json"
        if not filepath.exists():
            return None
        
        with open(filepath) as f:
            data = json.load(f)
        
        return ExperimentResult(
            experiment_id=data["experiment_id"],
            model_name=data["model_name"],
            factor_set=data["factor_set"],
            validation_strategy=data["validation_strategy"],
            hyperparameters=data["hyperparameters"],
            metrics=data["metrics"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration_seconds=data["duration_seconds"],
            git_commit=data["git_commit"],
            notes=data.get("notes", ""),
        )
    
    def list_experiments(self) -> pd.DataFrame:
        """List all saved experiments."""
        records = []
        for filepath in sorted(self.output_dir.glob("*.json")):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                records.append({
                    "experiment_id": data["experiment_id"],
                    "model": data["model_name"],
                    "metrics": str(data.get("metrics", {})),
                    "duration": data.get("duration_seconds", 0),
                    "timestamp": data.get("timestamp", ""),
                })
            except Exception:
                continue
        
        return pd.DataFrame(records) if records else pd.DataFrame()
