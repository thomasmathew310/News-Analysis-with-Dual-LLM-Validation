"""
news_fetcher.py — Fetch recent news articles from NewsAPI.

Primary function:
    fetch_news(api_key, query, max_results, days) -> list[dict]

Design goals:
- Clean interface for main.py to call
- Robust error handling (timeouts, non-200 responses, rate limits)
- Normalized article schema (so LLM prompts are consistent)
- De-duplication (NewsAPI can return duplicates across pages)
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
DEFAULT_TIMEOUT_SEC = 15
MAX_PAGE_SIZE = 100  # NewsAPI pageSize limit


class NewsAPIError(RuntimeError):
    """Raised when NewsAPI returns an error response or unexpected payload."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso_z(dt: datetime) -> str:
    # NewsAPI accepts ISO-8601 timestamps; Z is a common UTC marker.
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_article_id(article: Dict[str, Any]) -> str:
    """
    Create a stable identifier for an article using URL (best) or title+publishedAt.
    This helps join raw_articles -> analysis -> validated results reliably.
    """
    url = (article.get("url") or "").strip()
    if url:
        base = url
    else:
        base = f"{(article.get('title') or '').strip()}|{(article.get('publishedAt') or '').strip()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def _normalize_article(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw NewsAPI article into a consistent schema for downstream LLM analysis.
    """
    source_obj = raw.get("source") or {}
    normalized = {
        "id": _stable_article_id(raw),
        "title": (raw.get("title") or "").strip(),
        "description": (raw.get("description") or "").strip(),
        "content": (raw.get("content") or "").strip(),
        "url": (raw.get("url") or "").strip(),
        "source": (source_obj.get("name") or "").strip(),
        "author": (raw.get("author") or "").strip(),
        "publishedAt": (raw.get("publishedAt") or "").strip(),
    }

    # Helpful derived field for prompting: a single “best effort” text blob.
    # Prefer content; fallback to description/title.
    blob_parts = [normalized["content"], normalized["description"], normalized["title"]]
    normalized["text"] = "\n\n".join([p for p in blob_parts if p])

    return normalized


def fetch_news(
    api_key: str,
    query: str,
    max_results: int = 15,
    days: int = 7,
    *,
    language: str = "en",
    sort_by: str = "publishedAt",
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch recent news articles from NewsAPI.

    Args:
        api_key: NewsAPI key (pass from env in main.py)
        query: Search query, e.g. "India politics" or "India government"
        max_results: Target number of articles to return (10–15 recommended)
        days: Lookback window in days (e.g., 7)
        language: NewsAPI language filter (default "en")
        sort_by: One of NewsAPI's sort modes (default "publishedAt")
        timeout_sec: HTTP timeout
        session: Optional requests.Session for testability/reuse

    Returns:
        List of normalized article dicts.

    Raises:
        ValueError: for invalid inputs
        NewsAPIError: for API errors/unexpected payloads
    """
    if not api_key or not api_key.strip():
        raise ValueError("NewsAPI api_key is missing/empty.")
    if not query or not query.strip():
        raise ValueError("Query is missing/empty.")
    if max_results <= 0:
        return []
    if days <= 0:
        raise ValueError("days must be >= 1.")

    s = session or requests.Session()

    to_dt = _utc_now()
    from_dt = to_dt - timedelta(days=days)

    # NewsAPI pagination
    page = 1
    page_size = min(MAX_PAGE_SIZE, max_results)

    headers = {"X-Api-Key": api_key.strip()}

    results: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_ids: set[str] = set()

    # Basic rate-limit/backoff handling: retry on 429 a few times.
    max_retries = 3
    backoff_sec = 2.0

    while len(results) < max_results:
        params = {
            "q": query,
            "from": _to_iso_z(from_dt),
            "to": _to_iso_z(to_dt),
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size,
            "page": page,
        }

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = s.get(NEWSAPI_BASE_URL, params=params, headers=headers, timeout=timeout_sec)
            except requests.Timeout as e:
                raise NewsAPIError(f"NewsAPI request timed out after {timeout_sec}s.") from e
            except requests.RequestException as e:
                raise NewsAPIError("NewsAPI request failed due to a network/client error.") from e

            # Handle rate limits politely
            if resp.status_code == 429 and attempt <= max_retries:
                logging.warning("NewsAPI rate limited (429). Backing off for %.1fs...", backoff_sec)
                time.sleep(backoff_sec)
                backoff_sec *= 2
                continue

            if resp.status_code != 200:
                # NewsAPI usually returns JSON with {status, code, message}
                try:
                    payload = resp.json()
                except Exception:
                    payload = {"message": resp.text[:2000]}
                code = payload.get("code") if isinstance(payload, dict) else None
                msg = payload.get("message") if isinstance(payload, dict) else str(payload)
                raise NewsAPIError(f"NewsAPI error HTTP {resp.status_code} (code={code}): {msg}")

            try:
                data = resp.json()
            except Exception as e:
                raise NewsAPIError("NewsAPI returned non-JSON response.") from e

            if not isinstance(data, dict) or data.get("status") != "ok":
                raise NewsAPIError(f"Unexpected NewsAPI payload: {data!r}")

            raw_articles = data.get("articles") or []
            if not raw_articles:
                # No more results
                return results

            # Normalize + filter + dedupe
            for raw in raw_articles:
                if not isinstance(raw, dict):
                    continue
                norm = _normalize_article(raw)

                # Filter out obviously unusable items
                if not norm["title"] and not norm["description"] and not norm["content"]:
                    continue
                if not norm["url"]:
                    # URL is important for reporting and de-dupe; skip if missing
                    continue

                if norm["url"] in seen_urls or norm["id"] in seen_ids:
                    continue

                seen_urls.add(norm["url"])
                seen_ids.add(norm["id"])
                results.append(norm)

                if len(results) >= max_results:
                    break

            # If we didn't get enough articles, go next page.
            page += 1
            break

        # Safety stop: NewsAPI has practical limits (and infinite loops are bad)
        if page > 10:  # 10 pages * 100 = 1000 potential scan, more than enough for this assignment
            logging.warning("Pagination exceeded 10 pages; stopping early with %d articles.", len(results))
            break

    return results
