"""
knowledge/gfg_scraper.py
─────────────────────────
Fetches GeeksforGeeks ACD articles and returns clean plain-text.
Uses requests + BeautifulSoup.  Respects robots.txt spirit by caching.

Upgrade:
  • Robust user-agent headers
  • DuckDuckGo web search fallback if scraping fails
  • Retry logic with exponential backoff
"""

from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from knowledge.curated_topics import GFG_ACD_URLS

# Local cache directory (avoids hitting GFG on every startup)
_CACHE_DIR = Path("./data/gfg_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


def _cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"{h}.txt"


def _web_search_fallback(topic: str) -> Optional[str]:
    """
    Fallback: search the web for the topic using DuckDuckGo Search.
    Returns the best result's text content, or None.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"{topic} automata compiler design site:geeksforgeeks.org",
                max_results=3,
            ))
        if results:
            # Try to fetch the first result URL
            for r in results:
                url = r.get("href", r.get("link", ""))
                if url:
                    text = fetch_article(url, use_cache=True, skip_fallback=True)
                    if text:
                        return text
            # If no URL worked, return snippet text
            snippets = [r.get("body", "") for r in results if r.get("body")]
            return "\n\n".join(snippets) if snippets else None
    except ImportError:
        pass
    except Exception as exc:
        print(f"[GFG Scraper] DuckDuckGo fallback failed: {exc}")

    return None


def fetch_article(
    url: str,
    use_cache: bool = True,
    skip_fallback: bool = False,
    max_retries: int = 2,
) -> Optional[str]:
    """
    Download a GFG article and return its main text content.
    Returns None on failure.
    """
    cpath = _cache_path(url)
    if use_cache and cpath.exists():
        return cpath.read_text(encoding="utf-8")

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            break
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            print(f"[GFG Scraper] Failed to fetch {url} after {max_retries + 1} attempts: {exc}")
            return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # GFG main article content lives in .article-body or .entry-content
    content_div = (
        soup.find("div", class_="article-body")
        or soup.find("div", class_="entry-content")
        or soup.find("article")
    )
    if not content_div:
        return None

    # Remove code blocks' noise — keep them but strip ads/nav
    for tag in content_div.find_all(["script", "style", "nav", "aside", "footer"]):
        tag.decompose()

    text = content_div.get_text(separator="\n", strip=True)
    if not text or len(text) < 100:
        return None

    cpath.write_text(text, encoding="utf-8")
    return text


def load_all_gfg_articles(
    progress_callback=None, delay: float = 0.5
) -> list[dict]:
    """
    Fetch all curated ACD articles from GFG.
    Falls back to web search if direct scraping fails.

    Returns
    -------
    list of {"topic": str, "url": str, "text": str}
    """
    docs = []
    for idx, (topic, url) in enumerate(GFG_ACD_URLS):
        if progress_callback:
            progress_callback(idx, len(GFG_ACD_URLS), topic)

        text = fetch_article(url)

        # Fallback: web search
        if not text:
            text = _web_search_fallback(topic)

        if text:
            docs.append({"topic": topic, "url": url, "text": text})
        time.sleep(delay)  # polite delay
    return docs
