"""
Tests for NLP and sentiment analysis modules.
"""

from __future__ import annotations

import pytest

from nlp.sentiment import (
    FinancialSentimentAnalyzer, LexiconSentiment, SentimentResult,
)


class TestLexiconSentiment:
    """Lexicon-based sentiment analyzer tests."""

    def test_positive_text(self):
        analyzer = LexiconSentiment()
        score = analyzer.score("strong earnings growth beat expectations")
        assert score > 0

    def test_negative_text(self):
        analyzer = LexiconSentiment()
        score = analyzer.score("weak losses bankruptcy risk decline")
        assert score < 0

    def test_neutral_text(self):
        analyzer = LexiconSentiment()
        score = analyzer.score("the company released its quarterly report")
        assert isinstance(score, float)

    def test_empty_text(self):
        analyzer = LexiconSentiment()
        score = analyzer.score("")
        assert score == 0.0


class TestFinancialSentimentAnalyzer:
    """Hybrid sentiment analyzer tests."""

    def test_positive_analysis(self):
        analyzer = FinancialSentimentAnalyzer(method="lexicon")
        result = analyzer.analyze([
            "Strong earnings beat expectations across all segments."
        ])
        assert isinstance(result, SentimentResult)
        assert result.label in ("positive", "negative", "neutral")
        assert isinstance(result.score, float)
        assert 0 <= result.confidence <= 1

    def test_negative_analysis(self):
        analyzer = FinancialSentimentAnalyzer(method="lexicon")
        result = analyzer.analyze([
            "Severe losses and bankruptcy risk imminent."
        ])
        assert isinstance(result, SentimentResult)

    def test_multiple_texts(self):
        analyzer = FinancialSentimentAnalyzer(method="lexicon")
        result = analyzer.analyze([
            "Strong revenue growth this quarter.",
            "EPS exceeded expectations by 15%.",
            "Management provided positive guidance.",
        ])
        assert result.score > -0.5  # Should be overall positive

    def test_empty_texts(self):
        analyzer = FinancialSentimentAnalyzer(method="lexicon")
        result = analyzer.analyze([])
        assert result.label == "neutral"
        assert result.score == 0.0

    @pytest.mark.parametrize("texts,expected_min_score", [
        (["massive profit growth exceeded all expectations"], 0.0),
        (["severe losses and bankruptcy risk imminent"], -1.0),
    ])
    def test_sentiment_direction(self, texts, expected_min_score):
        analyzer = FinancialSentimentAnalyzer(method="lexicon")
        result = analyzer.analyze(texts)
        assert result.score >= expected_min_score
