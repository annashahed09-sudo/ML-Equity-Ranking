"""Privacy and regulatory guardrails for generated market reports."""

from __future__ import annotations

PRIVACY_NOTICE = """
Privacy notice: this tool processes ticker symbols, OHLCV market data, optional review text,
and source-attributed public news links only for the current analysis session. It does not need
personal financial account data, social security numbers, credentials, or non-public personal
information. Do not upload sensitive personal information.
""".strip()

REGULATORY_NOTICE = """
Regulatory notice: outputs are research signals and scenario-analysis artifacts, not financial,
investment, legal, tax, or accounting advice. Predictions are uncertain and may be wrong. Users
must perform independent due diligence, risk review, and compliance checks before making any
investment decision.
""".strip()


def combined_disclaimer() -> str:
    """Return privacy and regulatory disclaimers for UI/API/PDF outputs."""
    return f"{PRIVACY_NOTICE}\n\n{REGULATORY_NOTICE}"
