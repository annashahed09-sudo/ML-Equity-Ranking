"""
News ingestion from RSS feeds with caching and deduplication.

Supports multiple financial news sources.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsItem:
    """A single news article with metadata."""
    source: str
    title: str
    link: str
    published: str
    summary: str = ""
    tickers: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


DEFAULT_FEEDS: Dict[str, str] = {
    "NYT Business": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "NYT World": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "FT World": "https://www.ft.com/rss/world",
    "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
}

GEOPOLITICAL_KEYWORDS: set = {
    "war", "sanction", "election", "inflation", "rate", "central bank",
    "oil", "trade", "tariff", "china", "russia", "middle east",
    "supply chain", "recession", "growth", "gdp", "interest rate",
    "federal reserve", "fed", "stimulus", "infrastructure",
}


class NewsIngestion:
    """News ingestion from RSS feeds."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or settings.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch(
        self,
        feeds: Optional[Dict[str, str]] = None,
        limit_per_source: int = 5,
        timeout: int = 10,
    ) -> List[NewsItem]:
        """Fetch news from RSS feeds."""
        selected = feeds or DEFAULT_FEEDS
        all_items: List[NewsItem] = []
        
        for source, url in selected.items():
            items = self._fetch_feed(source, url, limit_per_source, timeout)
            all_items.extend(items)
            logger.debug(f"Fetched {len(items)} items from {source}")
        
        # Deduplicate by title
        seen_titles: set = set()
        unique_items: List[NewsItem] = []
        for item in all_items:
            if item.title.lower() not in seen_titles:
                seen_titles.add(item.title.lower())
                unique_items.append(item)
        
        return unique_items
    
    def _fetch_feed(
        self,
        source: str,
        url: str,
        limit: int,
        timeout: int,
    ) -> List[NewsItem]:
        """Fetch a single RSS feed."""
        request = Request(url, headers={
            "User-Agent": "ML-Equity-Ranking/2.0 research tool (academic)"
        })
        
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except (TimeoutError, URLError, OSError) as e:
            logger.warning(f"Failed to fetch {source}: {e}")
            return []
        
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as e:
            logger.warning(f"Failed to parse {source}: {e}")
            return []
        
        items: List[NewsItem] = []
        for item in root.findall(".//item")[:limit]:
            title = self._get_text(item, "title")
            link = self._get_text(item, "link")
            published = self._get_text(item, "pubDate")
            summary = self._get_text(item, "description")
            
            if title and link:
                items.append(NewsItem(
                    source=source,
                    title=title,
                    link=link,
                    published=published,
                    summary=summary,
                ))
        
        return items
    
    @staticmethod
    def _get_text(element: ET.Element, tag: str) -> str:
        el = element.find(tag)
        if el is None or el.text is None:
            return ""
        return el.text.strip()
    
    def filter_geopolitical(
        self,
        items: Iterable[NewsItem],
        limit: int = 10,
    ) -> List[NewsItem]:
        """Keep items with geopolitical/macro relevance."""
        selected: List[NewsItem] = []
        for item in items:
            haystack = f"{item.title} {item.summary}".lower()
            if any(kw in haystack for kw in GEOPOLITICAL_KEYWORDS):
                selected.append(item)
            if len(selected) >= limit:
                break
        return selected
    
    @staticmethod
    def to_dataframe(items: List[NewsItem]) -> pd.DataFrame:
        """Convert news items to DataFrame for analysis."""
        return pd.DataFrame([
            {
                "source": item.source,
                "title": item.title,
                "link": item.link,
                "published": item.published,
                "summary": item.summary[:200] if item.summary else "",
            }
            for item in items
        ])
