from src.cli import _parse_tickers


def test_parse_tickers():
    out = _parse_tickers('aapl, msft , nvda')
    assert out == ['AAPL', 'MSFT', 'NVDA']
