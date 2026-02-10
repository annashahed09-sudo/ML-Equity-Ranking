from src.nlp import FinancialSentimentAnalyzer


def test_financial_sentiment_lexicon_and_summary():
    analyzer = FinancialSentimentAnalyzer()
    texts = [
        "Strong growth and record profit with bullish guidance",
        "Weak quarter and downgrade risk with decline",
    ]
    scores = analyzer.predict_scores(texts)
    assert len(scores) == 2

    insight = analyzer.build_insight(texts)
    assert insight.label in {"positive", "neutral", "negative"}
    assert isinstance(insight.summary, str)
    assert len(insight.summary) > 0


def test_financial_sentiment_trainable_classifier():
    analyzer = FinancialSentimentAnalyzer()
    texts = ["strong growth and beat", "weak miss and decline", "bullish upside", "lawsuit risk and loss"]
    labels = [1, 0, 1, 0]
    analyzer.fit(texts, labels)
    scores = analyzer.predict_scores(["strong upside", "weak decline"])
    assert len(scores) == 2
