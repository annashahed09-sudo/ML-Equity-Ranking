"""
Financial sentiment analysis with multiple backends.

Supports:
- FinBERT (when transformers library available)
- Lexicon-based (financial word lists)
- Hybrid approach (combining both)
- Confidence scoring and calibration
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


FINANCIAL_POSITIVE_WORDS: set = {
    "beat", "growth", "strong", "bullish", "upside", "improve", "record",
    "profit", "outperform", "positive", "expansion", "surge", "rally",
    "upgrade", "exceed", "momentum", "robust", "accelerate", "optimistic",
    "dividend", "buyback", "innovation", "efficiency", "synergy",
}

FINANCIAL_NEGATIVE_WORDS: set = {
    "miss", "weak", "bearish", "downgrade", "risk", "loss", "decline",
    "lawsuit", "underperform", "negative", "contraction", "plunge", "crash",
    "volatile", "uncertainty", "provision", "write-down", "default",
    "bankruptcy", "restructuring", "layoff", "investigation", "penalty",
}


class LexiconSentiment:
    """Simple lexicon-based financial sentiment analyzer."""
    
    def score(self, text: str) -> float:
        """Score text in [-1, 1] using financial word lists."""
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        if not tokens:
            return 0.0
        pos = sum(1 for t in tokens if t in FINANCIAL_POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in FINANCIAL_NEGATIVE_WORDS)
        return float((pos - neg) / len(tokens))


class FinBERTSentiment:
    """FinBERT sentiment analysis (when transformers is available)."""
    
    def __init__(self):
        self._pipeline = None
    
    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                max_length=512,
                truncation=True,
            )
            logger.info("Loaded FinBERT model")
        except ImportError:
            logger.warning("FinBERT not available (install transformers)")
            self._pipeline = None
    
    def score(self, text: str) -> tuple[float, float]:
        """Score text using FinBERT. Returns (score, confidence)."""
        self._load()
        if self._pipeline is None:
            return 0.0, 0.0
        
        try:
            result = self._pipeline(text[:512])[0]
            label = result["label"]
            confidence = result["score"]
            
            if label == "positive":
                return confidence, confidence
            elif label == "negative":
                return -confidence, confidence
            else:
                return 0.0, confidence
        except Exception as e:
            logger.warning(f"FinBERT inference failed: {e}")
            return 0.0, 0.0


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    score: float  # [-1, 1]
    label: str  # positive, negative, neutral
    confidence: float  # [0, 1]
    method: str  # 'finbert', 'lexicon', 'hybrid'
    summary: str = ""


class FinancialSentimentAnalyzer:
    """
    Hybrid financial sentiment analyzer.
    
    Uses FinBERT when available, falls back to lexicon.
    Aggregates sentiment over multiple texts for robustness.
    """
    
    def __init__(self, method: str = "hybrid"):
        self.method = method
        self.finbert = FinBERTSentiment()
        self.lexicon = LexiconSentiment()
    
    def analyze(self, texts: List[str]) -> SentimentResult:
        """
        Analyze sentiment of a collection of texts.
        
        Parameters
        ----------
        texts : List[str]
            Collection of text snippets (headlines, reviews, etc.)
        
        Returns
        -------
        SentimentResult
            Aggregated sentiment with confidence
        """
        if not texts:
            return SentimentResult(0.0, "neutral", 0.0, self.method)
        
        scores = []
        confidences = []
        
        for text in texts:
            if not text.strip():
                continue
            
            if self.method in ("finbert", "hybrid"):
                fs, fc = self.finbert.score(text)
            else:
                fs, fc = 0.0, 0.0
            
            if self.method == "lexicon" or (self.method == "hybrid" and fc < 0.5):
                ls = self.lexicon.score(text)
                if self.method == "hybrid":
                    fs = 0.3 * fs + 0.7 * ls
                    fc = max(fc, 0.3)
                else:
                    fs = ls
                    fc = 0.3
            
            scores.append(fs)
            confidences.append(fc)
        
        if not scores:
            return SentimentResult(0.0, "neutral", 0.0, self.method)
        
        # Aggregate: weighted average by confidence
        weights = np.array(confidences)
        if weights.sum() < 1e-6:
            weights = np.ones_like(weights)
        
        avg_score = float(np.average(scores, weights=weights))
        avg_confidence = float(np.average(confidences, weights=weights))
        
        # Classify
        if avg_score > 0.1:
            label = "positive"
        elif avg_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return SentimentResult(
            score=avg_score,
            label=label,
            confidence=avg_confidence,
            method=self.method,
            summary=self._summarize(texts),
        )
    
    @staticmethod
    def _summarize(texts: List[str]) -> str:
        """Generate a concise summary of the texts."""
        if len(texts) == 1:
            return texts[0][:200]
        
        # Extract key terms
        words = []
        for t in texts:
            words.extend(re.findall(r"[A-Z][a-z]+", t))
        
        from collections import Counter
        top = Counter(words).most_common(5)
        return f"Key terms: {', '.join(w for w, c in top)}"
