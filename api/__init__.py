"""
FastAPI-based serving layer for the quantitative research platform.

Provides:
- Prediction endpoints (single ticker, batch, S&P 500 simulation)
- Factor analysis endpoints
- Portfolio optimization endpoints
- Risk analysis endpoints
- Health and status endpoints
"""

from .app import create_app

__all__ = ["create_app"]
