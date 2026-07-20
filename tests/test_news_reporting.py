from pathlib import Path

import pytest

from src.news import build_evidence_narrative, filter_geopolitical_evidence, NewsEvidence


def test_news_evidence_filter_and_narrative():
    evidence = [
        NewsEvidence(source="New York Times Business", title="Central bank rate decision rattles markets", link="https://example.com/1", published="today"),
        NewsEvidence(source="The Economist Latest", title="Culture notes", link="https://example.com/2", published="today"),
    ]
    filtered = filter_geopolitical_evidence(evidence)
    assert len(filtered) == 1
    narrative = build_evidence_narrative(filtered)
    assert "New York Times Business" in narrative
    assert "Central bank" in narrative


def test_generate_pdf_report(tmp_path: Path):
    pytest.importorskip("matplotlib", reason="matplotlib not installed")
    from src.reporting import generate_pdf_report
    from src.sp500 import run_sp500_simulation
    from tests.test_sp500 import _prepared_data
    
    result = run_sp500_simulation(
        start_date="2022-01-01",
        end_date="2022-06-01",
        model_type="ridge",
        limit=6,
        n_splits=2,
        test_size=120,
        min_train_size=240,
        prepared_data=_prepared_data(),
    )
    evidence = [NewsEvidence(source="The Economist Latest", title="Oil and trade risks rise", link="https://example.com/e", published="today")]
    output = generate_pdf_report(result, tmp_path / "report.pdf", evidence)
    assert output.exists()
    assert output.stat().st_size > 0
