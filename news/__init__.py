"""
News intelligence module for market-aware analysis.

Provides:
- RSS feed ingestion from financial news sources
- Entity extraction (ticker, company, person)
- Event classification (earnings, M&A, regulatory, macro)
- Market-moving event detection
- News-to-factor integration
"""

from .ingestion import NewsIngestion
from .events import EventDetector

__all__ = [
    "NewsIngestion",
    "EventDetector",
]
