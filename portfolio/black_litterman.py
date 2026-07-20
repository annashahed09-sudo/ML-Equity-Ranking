"""
Black-Litterman portfolio optimization model.

The Black-Litterman model combines prior market equilibrium returns with
investor views to produce a posterior expected return estimate, then optimizes
using mean-variance.

Key reference:
    Black, F. & Litterman, R. (1992). "Global Portfolio Optimization."
    Financial Analysts Journal, 48(5), 28-43.

    Idzorek, T. (2004). "A Step-by-Step Guide to the Black-Litterman Model."
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint

from core.exceptions import PortfolioError, ConvergenceError
from risk.covariance import CovarianceEstimator, SampleCovariance


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization model.

    Steps:
    1. Compute prior (market equilibrium) expected returns via reverse optimization
    2. Incorporate investor views with confidence levels
    3. Compute posterior expected returns and covariance
    4. Optimize using mean-variance on posterior estimates
    """

    def __init__(
        self,
        covariance_estimator: Optional[CovarianceEstimator] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.05,
        max_weight: float = 0.2,
        min_weight: float = -0.2,
    ):
        """
        Parameters
        ----------
        covariance_estimator : CovarianceEstimator
            Method for estimating the covariance matrix
        risk_aversion : float
            Market risk aversion coefficient (lambda)
            Typical range: 1.0 - 5.0. Higher = more risk-averse.
        tau : float
            Uncertainty scaling factor for prior covariance
            Typical range: 0.01 - 0.10. Lower = more confidence in prior.
        risk_free_rate : float
            Annualized risk-free rate
        max_weight : float
            Maximum position weight
        min_weight : float
            Minimum position weight (negative for shorting)
        """
        self.covariance_estimator = covariance_estimator or SampleCovariance()
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight

    def reverse_optimize(
        self,
        market_caps: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute implied equilibrium returns via reverse optimization.

        From CAPM, the market portfolio is optimal given equilibrium returns.
        We reverse-engineer these returns from market weights and covariance.

        Π = λ * Σ * w_market

        Parameters
        ----------
        market_caps : pd.Series
            Market capitalizations for each asset
        covariance : pd.DataFrame
            N x N covariance matrix

        Returns
        -------
        pd.Series
            Implied equilibrium excess returns
        """
        common = market_caps.index.intersection(covariance.index)
        if len(common) < 2:
            raise PortfolioError("Need at least 2 assets for reverse optimization")

        w_market = market_caps[common] / market_caps[common].sum()
        sigma = covariance.loc[common, common].values
        implied_returns = self.risk_aversion * sigma @ w_market.values

        return pd.Series(implied_returns, index=common, name="implied_return")

    def compute_posterior(
        self,
        prior_returns: pd.Series,
        covariance: pd.DataFrame,
        views: List[Dict],
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute Black-Litterman posterior expected returns and covariance.

        Parameters
        ----------
        prior_returns : pd.Series
            Prior expected returns (from reverse optimization or other)
        covariance : pd.DataFrame
            Prior covariance matrix
        views : List[Dict]
            Investor views, each with:
                - 'type': 'absolute' or 'relative'
                - 'assets': list of asset names
                - 'return': expected return (for absolute) or
                            return differential (for relative)
                - 'confidence': confidence level in [0, 1]
                - 'weights' (optional): for relative views, the
                  weights on each asset (must sum to 0)

        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            (posterior_returns, posterior_covariance)
        """
        common = prior_returns.index.intersection(covariance.index)
        n = len(common)
        prior = prior_returns[common].values
        sigma = covariance.loc[common, common].values

        # Build view matrix P and view vector Q
        m = len(views)
        P = np.zeros((m, n))
        Q = np.zeros(m)
        omega = np.zeros((m, m))  # View uncertainty matrix

        for i, view in enumerate(views):
            assets = view.get("assets", [])
            idx = [common.get_loc(a) for a in assets if a in common]

            if view["type"] == "absolute":
                # Single asset absolute view
                if len(idx) == 1:
                    P[i, idx[0]] = 1.0
                    Q[i] = view["return"]
                    uncertainty = sigma[idx[0], idx[0]] * (1 - view.get("confidence", 0.5))
                    omega[i, i] = uncertainty

            elif view["type"] == "relative":
                # Relative view between assets
                vw = view.get("weights", None)
                if vw is None and len(idx) >= 2:
                    # Equal weight long/short
                    n_assets = len(idx)
                    for j, jdx in enumerate(idx):
                        P[i, jdx] = 1.0 / n_assets if j < n_assets // 2 else -1.0 / n_assets
                    Q[i] = view["return"]
                elif vw is not None:
                    for asset, w in vw.items():
                        if asset in common:
                            P[i, common.get_loc(asset)] = w
                    Q[i] = view["return"]

                # View uncertainty: scaled by diagonal of P*Sigma*P'
                view_var = P[i, :] @ sigma @ P[i, :]
                uncertainty = view_var * (1 - view.get("confidence", 0.5))
                omega[i, i] = max(uncertainty, 1e-10)

        # Black-Litterman posterior mean
        # μ_post = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} * [(τΣ)^{-1}Π + P'Ω^{-1}Q]
        tau_sigma_inv = np.linalg.inv(self.tau * sigma + 1e-12 * np.eye(n))
        omega_inv = np.linalg.inv(omega + 1e-12 * np.eye(m))

        A = tau_sigma_inv + P.T @ omega_inv @ P
        b = tau_sigma_inv @ prior + P.T @ omega_inv @ Q

        try:
            posterior_mean = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            posterior_mean = np.linalg.lstsq(A, b, rcond=None)[0]

        # Posterior covariance
        # Σ_post = Σ + [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}
        posterior_cov = sigma + np.linalg.inv(A)

        return (
            pd.Series(posterior_mean, index=common, name="posterior_return"),
            pd.DataFrame(posterior_cov, index=common, columns=common),
        )

    def optimize(
        self,
        market_caps: pd.Series,
        returns: pd.DataFrame,
        views: Optional[List[Dict]] = None,
    ) -> pd.Series:
        """
        Full Black-Litterman optimization workflow.

        1. Estimate covariance matrix from historical returns
        2. Compute market-cap weighted equilibrium returns (reverse optimization)
        3. Incorporate views to get posterior estimates
        4. Mean-variance optimize on posterior estimates

        Parameters
        ----------
        market_caps : pd.Series
            Market capitalizations for each asset
        returns : pd.DataFrame
            T x N historical returns matrix
        views : List[Dict], optional
            Investor views (see compute_posterior for format)

        Returns
        -------
        pd.Series
            Optimal portfolio weights
        """
        cov = self.covariance_estimator.estimate(returns)

        common = market_caps.index.intersection(cov.index)
        if len(common) < 2:
            raise PortfolioError("Need at least 2 assets after intersection")

        market_caps = market_caps[common]
        cov = cov.loc[common, common]

        # Step 1: Prior - implied equilibrium returns
        prior = self.reverse_optimize(market_caps, cov)

        # Step 2: Posterior - incorporate views
        if views and len(views) > 0:
            posterior_mean, posterior_cov = self.compute_posterior(
                prior, cov, views
            )
        else:
            posterior_mean = prior
            posterior_cov = cov

        # Step 3: Mean-variance optimization on posterior
        # Max: μ'w - (λ/2) * w'Σw
        n = len(posterior_mean)

        def objective(weights: np.ndarray) -> float:
            ret = weights @ posterior_mean.values
            risk = weights @ posterior_cov.values @ weights
            return -(ret - 0.5 * self.risk_aversion * risk)

        constraints = LinearConstraint(np.ones(n), 1.0, 1.0)
        bounds = Bounds([self.min_weight] * n, [self.max_weight] * n)

        # Initial guess: equal weight
        x0 = np.array([1.0 / n] * n)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            raise ConvergenceError(
                f"Black-Litterman optimization failed: {result.message}"
            )

        return pd.Series(result.x, index=posterior_mean.index, name="bl_weights")
