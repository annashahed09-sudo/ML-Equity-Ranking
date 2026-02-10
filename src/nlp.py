"""NLP utilities for market review analysis and sentiment scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


POSITIVE_WORDS = {
    "growth", "beat", "strong", "bullish", "upside", "improve", "record", "profit", "outperform"
}
NEGATIVE_WORDS = {
    "miss", "weak", "bearish", "downgrade", "risk", "loss", "decline", "lawsuit", "underperform"
}


@dataclass
class ReviewInsight:
    sentiment_score: float
    label: str
    confidence: float
    summary: str


class FinancialSentimentAnalyzer:
    """Hybrid NLP sentiment analyzer with trainable classifier + lexicon fallback."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> None:
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.fitted = True

    def _lexicon_score(self, text: str) -> float:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        if not tokens:
            return 0.0
        pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        return float((pos - neg) / max(len(tokens), 1))

    def predict_scores(self, texts: List[str]) -> np.ndarray:
        if self.fitted:
            X = self.vectorizer.transform(texts)
            probs = self.model.predict_proba(X)[:, 1]
            return probs * 2 - 1  # map to [-1, 1]
        return np.array([self._lexicon_score(t) for t in texts], dtype=float)

    def summarize_reviews(self, texts: List[str], top_k: int = 3) -> str:
        all_tokens: List[str] = []
        for t in texts:
            all_tokens.extend(re.findall(r"[a-zA-Z]{4,}", t.lower()))
        if not all_tokens:
            return "No review content available."

        freq: Dict[str, int] = {}
        for tok in all_tokens:
            if tok in {"this", "that", "with", "from", "have", "were", "will", "into"}:
                continue
            freq[tok] = freq.get(tok, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return "Top discussion themes: " + ", ".join([f"{w} ({c})" for w, c in top])

    def build_insight(self, texts: List[str]) -> ReviewInsight:
        scores = self.predict_scores(texts)
        score = float(np.mean(scores)) if len(scores) else 0.0
        confidence = float(np.clip(np.std(scores) * -1 + 1, 0.0, 1.0)) if len(scores) else 0.0

        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return ReviewInsight(
            sentiment_score=score,
            label=label,
            confidence=confidence,
            summary=self.summarize_reviews(texts),
        )
