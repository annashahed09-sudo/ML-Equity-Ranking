"""News and geopolitics evidence collection for market reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class NewsEvidence:
    """A compact, source-attributed news item used as model context evidence."""

    source: str
    title: str
    link: str
    published: str
    summary: str = ""


DEFAULT_NEWS_FEEDS = {
    "New York Times Business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "New York Times World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "The Economist Latest": "https://www.economist.com/latest/rss.xml",
}


GEOPOLITICAL_KEYWORDS = {
    "war", "sanction", "election", "inflation", "rate", "central bank", "oil", "trade",
    "tariff", "china", "russia", "middle east", "supply chain", "recession", "growth",
}


def _text_or_empty(element: Optional[ET.Element]) -> str:
    return "" if element is None or element.text is None else element.text.strip()


def fetch_rss_feed(source: str, url: str, limit: int = 10, timeout: int = 10) -> List[NewsEvidence]:
    """Fetch a single RSS feed and return compact evidence items."""
    request = Request(url, headers={"User-Agent": "ML-Equity-Ranking/1.0 research tool"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (TimeoutError, URLError, OSError):
        return []

    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []

    items: List[NewsEvidence] = []
    for item in root.findall(".//item")[:limit]:
        title = _text_or_empty(item.find("title"))
        link = _text_or_empty(item.find("link"))
        published = _text_or_empty(item.find("pubDate"))
        summary = _text_or_empty(item.find("description"))
        if title and link:
            items.append(
                NewsEvidence(
                    source=source,
                    title=title,
                    link=link,
                    published=published,
                    summary=summary,
                )
            )
    return items


def collect_market_news(
    feeds: Optional[dict[str, str]] = None,
    limit_per_source: int = 5,
) -> List[NewsEvidence]:
    """Collect source-attributed evidence from NYT and The Economist RSS feeds."""
    selected_feeds = feeds or DEFAULT_NEWS_FEEDS
    evidence: List[NewsEvidence] = []
    for source, url in selected_feeds.items():
        evidence.extend(fetch_rss_feed(source, url, limit=limit_per_source))
    return evidence


def filter_geopolitical_evidence(evidence: Iterable[NewsEvidence], limit: int = 10) -> List[NewsEvidence]:
    """Keep items whose title/summary indicate macro or geopolitical relevance."""
    selected: List[NewsEvidence] = []
    for item in evidence:
        haystack = f"{item.title} {item.summary}".lower()
        if any(keyword in haystack for keyword in GEOPOLITICAL_KEYWORDS):
            selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def build_evidence_narrative(evidence: Iterable[NewsEvidence]) -> str:
    """Create a concise, source-attributed narrative for reports."""
    items = list(evidence)
    if not items:
        return "No external NYT/Economist evidence was available at report generation time."

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bullets = [f"Evidence collected at {generated}:"]
    for item in items:
        bullets.append(f"- {item.source}: {item.title} ({item.link})")
    return "\n".join(bullets)
