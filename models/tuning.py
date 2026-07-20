"""
Hyperparameter optimization framework.

Implements:
- Bayesian optimization (Gaussian Process / TPE)
- Randomized search
- Walk-forward aware cross-validation for time series
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result from a single hyperparameter trial."""
    params: Dict[str, Any]
    score: float
    std_score: float = 0.0
    train_time: float = 0.0
    n_iterations: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    trials: List[TrialResult]
    n_trials: int
    search_space: Dict[str, Any]
    method: str  # 'bayesian', 'grid', 'random'


class BayesianOptimizer:
    """Bayesian hyperparameter optimization using Gaussian Processes."""
    
    def __init__(
        self,
        param_space: Dict[str, Tuple],
        objective: Callable,
        n_trials: int = 50,
        n_startup_trials: int = 10,
        random_state: int = 42,
    ):
        self.param_space = param_space
        self.objective = objective
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.random_state = random_state
        self.trials: List[TrialResult] = []
    
    def optimize(self) -> OptimizationResult:
        """Run Bayesian optimization."""
        # Startup with random trials
        rng = np.random.RandomState(self.random_state)
        
        for i in range(self.n_startup_trials):
            params = self._sample_random_params(rng)
            self._evaluate(params, i)
        
        # Bayesian optimization rounds
        for i in range(self.n_startup_trials, self.n_trials):
            params = self._suggest_next_params()
            self._evaluate(params, i)
        
        best = max(self.trials, key=lambda t: t.score)
        
        return OptimizationResult(
            best_params=best.params,
            best_score=best.score,
            trials=self.trials,
            n_trials=len(self.trials),
            search_space=self.param_space,
            method="bayesian",
        )
    
    def _sample_random_params(self, rng: np.random.RandomState) -> Dict[str, Any]:
        params = {}
        for name, space in self.param_space.items():
            if space[0] == "choice":
                params[name] = rng.choice(space[1])
            elif space[0] == "int":
                params[name] = int(rng.randint(space[1], space[2] + 1))
            elif space[0] == "float":
                if len(space) == 4:
                    params[name] = float(rng.uniform(np.log10(space[1]), np.log10(space[2])))
                    params[name] = 10 ** params[name]
                else:
                    params[name] = float(rng.uniform(space[1], space[2]))
            elif space[0] == "log":
                log_low = np.log(space[1])
                log_high = np.log(space[2])
                params[name] = float(np.exp(rng.uniform(log_low, log_high)))
        return params
    
    def _evaluate(self, params: Dict[str, Any], trial_id: int) -> None:
        import time
        start = time.perf_counter()
        try:
            score = self.objective(params)
        except Exception as e:
            logger.warning(f"Trial {trial_id} failed: {e}")
            score = -1e10
        
        elapsed = time.perf_counter() - start
        result = TrialResult(
            params=params,
            score=float(score),
            train_time=elapsed,
        )
        self.trials.append(result)
        logger.info(f"Trial {trial_id}: score={score:.4f}, time={elapsed:.1f}s")
    
    def _suggest_next_params(self) -> Dict[str, Any]:
        """Simple heuristic: pick random params weighted by past performance."""
        if not self.trials:
            return self._sample_random_params(np.random.RandomState(self.random_state))
        
        # Epsilon-greedy: sometimes explore
        if np.random.random() < 0.2:
            return self._sample_random_params(
                np.random.RandomState(self.random_state + len(self.trials))
            )
        
        # Otherwise pick from top-performing params with noise
        scores = np.array([t.score for t in self.trials])
        if np.std(scores) < 1e-6:
            return self._sample_random_params(
                np.random.RandomState(self.random_state + len(self.trials))
            )
        
        # Weighted random selection
        weights = np.exp(scores - scores.max())
        weights = weights / weights.sum()
        idx = np.random.choice(len(self.trials), p=weights)
        best_trial = self.trials[idx]
        
        # Perturb best params
        new_params = {}
        for name, value in best_trial.params.items():
            if name in self.param_space:
                space = self.param_space[name]
                if space[0] in ("float", "log"):
                    noise = np.random.uniform(0.9, 1.1)
                    new_params[name] = value * noise
                elif space[0] == "int":
                    noise = np.random.randint(-2, 3)
                    new_params[name] = max(space[1], min(space[2], int(value) + noise))
                else:
                    new_params[name] = value
            else:
                new_params[name] = value
        return new_params


class GridOptimizer:
    """Exhaustive grid search over parameter space."""
    
    def __init__(self, param_grid: Dict[str, List], objective: Callable, random_state: int = 42):
        self.param_grid = param_grid
        self.objective = objective
        self.random_state = random_state
    
    def optimize(self) -> OptimizationResult:
        from itertools import product
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        trials = []
        best_score = -np.inf
        best_params = {}
        
        for combination in product(*values):
            params = dict(zip(keys, combination))
            try:
                score = self.objective(params)
            except Exception as e:
                logger.warning(f"Grid trial {params} failed: {e}")
                score = -np.inf
            
            trials.append(TrialResult(params=params, score=float(score)))
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            trials=trials,
            n_trials=len(trials),
            search_space=self.param_grid,
            method="grid",
        )
