"""
Financial event detection from news and text data.

Detects:
- Earnings announcements and surprises
- Guidance changes
- M&A activity
- Executive changes
- Regulatory actions
- Macroeconomic events
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FinancialEvent:
    """A detected financial event."""
    
    type: str  # earnings, guidance, ma, executive, regulatory, macro
    ticker: str
    date: str
    title: str
    description: str
    confidence: float
    sentiment: float = 0.0
    source: str = ""
    url: str = ""


# Event detection patterns
EARNINGS_PATTERNS = re.compile(
    r"(?:earnings|quarterly results|Q\d|fiscal|revenue|EPS)", re.I
)
GUIDANCE_PATTERNS = re.compile(
    r"(?:guidance|outlook|forecast|projection|expect)", re.I
)
MA_PATTERNS = re.compile(
    r"(?:acqui|merger|takeover|buyout|acquisition|merg)", re.I
)
EXECUTIVE_PATTERNS = re.compile(
    r"(?:CEO|CFO|appoint|resign|hire|fired|executive|board)", re.I
)
REGULATORY_PATTERNS = re.compile(
    r"(?:SEC|regulat|fine|settle|investigation|compliance)", re.I
)
MACRO_PATTERNS = re.compile(
    r"(?:GDP|inflation|interest rate|fed|unemployment|PMI)", re.I
)


class EventDetector:
    """
    Detects financial events from news text.
    
    Uses pattern matching and keyword detection to identify
    market-relevant events.
    """
    
    def __init__(self):
        self.ticker_pattern = re.compile(r"\b[A-Z]{1,5}\b")
    
    def detect(self, title: str, summary: str = "", ticker_hint: Optional[str] = None) -> List[FinancialEvent]:
        """
        Detect financial events in a news headline/article.
        
        Parameters
        ----------
        title : str
            News headline
        summary : str
            Article summary/body
        ticker_hint : str, optional
            Known ticker if available
        
        Returns
        -------
        List[FinancialEvent]
            Detected events
        """
        text = f"{title} {summary}"
        events = []
        
        # Try to extract ticker
        if ticker_hint:
            ticker = ticker_hint
        else:
            tickers = self.ticker_pattern.findall(text)
            ticker = tickers[0] if tickers else "UNKNOWN"
        
        # Detect event types
        if EARNINGS_PATTERNS.search(text):
            events.append(FinancialEvent(
                type="earnings",
                ticker=ticker,
                date="",
                title=title,
                description=summary[:300],
                confidence=0.7,
            ))
        
        if GUIDANCE_PATTERNS.search(text):
            events.append(FinancialEvent(
                type="guidance",
                ticker=ticker,
                date="",
                title=title,
                description=summary[:300] if summary else "",
                confidence=0.6,
            ))
        
        if MA_PATTERNS.search(text):
            events.append(FinancialEvent(
                type="ma",
                ticker=ticker,
                date="",
                title=title,
                description=summary[:300] if summary else "",
                confidence=0.8,
            ))
        
        if EXECUTIVE_PATTERNS.search(text):
            events.append(FinancialEvent(
                type="executive",
                ticker=ticker,
                date="",
                title=title,
                description=summary[:300] if summary else "",
                confidence=0.5,
            ))
        
        if REGULATORY_PATTERNS.search(text):
            events.append(FinancialEvent(
                type="regulatory",
                ticker=ticker,
                date="",
                title=title,
                description=summary[:300] if summary else "",
                confidence=0.6,
            ))
        
        return events
    
    def detect_batch(
        self,
        titles: List[str],
        summaries: Optional[List[str]] = None,
    ) -> List[List[FinancialEvent]]:
        """Detect events for multiple headlines."""
        if summaries is None:
            summaries = [""] * len(titles)
        
        return [
            self.detect(t, s)
            for t, s in zip(titles, summaries)
        ]
