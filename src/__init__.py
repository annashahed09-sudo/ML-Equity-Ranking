"""
DEPRECATED — Legacy module from the original ML-Equity-Ranking v1.x.

This module is preserved for backward compatibility only.
All functionality has been migrated to the new modular platform.

Please use the new modules directly:
    config/         → Configuration management
    core/           → Domain types, exceptions, utilities
    data/           → Data loading and quality
    factors/        → Factor engine (7 families)
    models/         → Model layer (12 model types)
    validation/     → Walk-forward, purged CV, metrics
    risk/           → Covariance estimation, VaR, factor risk
    portfolio/      → MVO, risk parity, Black-Litterman
    signal/         → Combination, normalization, orthogonalization
    nlp/            → Financial sentiment analysis
    news/           → News ingestion and event detection
    explainability/ → SHAP, permutation importance, PDP
    research/       → Orchestrated pipeline
    api/            → FastAPI server
    dashboard/      → Streamlit dashboard

This legacy module will be removed in a future release.
"""

import warnings

warnings.warn(
    "The 'src' module is deprecated. Use the new modular platform imports instead: "
    "config, core, factors, models, validation, risk, portfolio, signal, "
    "nlp, news, explainability, research, api, dashboard.",
    DeprecationWarning,
    stacklevel=2,
)
