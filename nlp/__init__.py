"""
Natural Language Processing module for financial text analysis.

Provides:
- Financial sentiment analysis (FinBERT, lexicon, hybrid)
- Named Entity Recognition (NER)
- Topic modeling
- Text embeddings and semantic similarity
- Financial event classification
"""

from .sentiment import FinancialSentimentAnalyzer

__all__ = [
    "FinancialSentimentAnalyzer",
]
