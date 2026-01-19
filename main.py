"""
main.py — Entry point for the News Analyzer pipeline.

Pipeline:
1) Fetch recent news from NewsAPI (news_fetcher.py)
2) Analyze each article using LLM#1 (llm_analyzer.py)
   - Primary: Gemini (uses GEMINI_API_KEY + GEMINI_MODEL from .env)
   - Fallback: Groq (uses GROQ_API_KEY + GROQ_MODEL from .env) if Gemini is rate-limited / unavailable
3) Validate/correct each analysis using LLM#2 (llm_validator.py)
   - Primary: OpenRouter (uses OPENROUTER_API_KEY + OPENROUTER_MODEL from .env)
   - Fallback: Groq (uses GROQ_API_KEY + GROQ_MODEL from .env) if OpenRouter model is unavailable / service errors
4) Save:
   - output/raw_articles.json
   - output/analysis_results.json
   - output/final_report.md

Expected function interfaces:
- news_fetcher.fetch_news(api_key: str, query: str, max_results: int, days: int) -> list[dict]
- llm_analyzer.analyze_article(article: dict, api_key: str, model: str | None = None) -> dict
- llm_validator.validate_analysis(article: dict, analysis: dict, api_key: str, model: str | None = None) -> dict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from news_fetcher import fetch_news
from llm_analyzer import analyze_article
from llm_validator import validate_analysis


DEFAULT_OUTPUT_DIR = Path("output")
RAW_ARTICLES_FILENAME = "raw_articles.json"
ANALYSIS_RESULTS_FILENAME = "analysis_results.json"
FINAL_REPORT_FILENAME = "final_report.md"

logger = logging.getLogger("news-analyzer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="News Analyzer (fetch -> analyze -> validate -> save).")
    parser.add_argument("--query", type=str, default="India politics", help="NewsAPI query string.")
    parser.add_argument("--n", type=int, default=15, help="Number of articles to fetch (10–15 recommended).")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days for recent articles.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory path.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stable_article_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _source_name(article: Dict[str, Any]) -> Optional[str]:
    src = article.get("source")
    if isinstance(src, dict):
        return src.get("name")
    if isinstance(src, str):
        return src
    return None


def _published_at(article: Dict[str, Any]) -> Optional[str]:
    return article.get("publishedAt") or article.get("published_at")


def filter_and_dedupe_articles(articles: List[Dict[str, Any]], max_n: int) -> List[Dict[str, Any]]:
    """Filter invalid items and dedupe by URL for cleaner runs."""
    cleaned: List[Dict[str, Any]] = []
    seen_urls = set()

    for a in articles or []:
        if not isinstance(a, dict):
            continue

        url = (a.get("url") or "").strip()
        title = (a.get("title") or "").strip()

        if not url or not title:
            continue
        if url in seen_urls:
            continue

        seen_urls.add(url)
        cleaned.append(a)

        if len(cleaned) >= max_n:
            break

    return cleaned


def adjudicate_final(llm1: Dict[str, Any], llm2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide what to publish as final fields.

    Current recommended validator shape (from llm_validator.py):
      {
        "gist": "...",
        "sentiment": "positive|negative|neutral",
        "tone": "...",
        "verdict": "accepted|corrected",
        "notes": "...",
        "_meta": {"provider": "...", "model": "..."}
      }

    Backward-compatible with older shapes that used pass/fail + corrected_fields.
    """
    llm1 = llm1 or {}
    llm2 = llm2 or {}

    # Default: trust LLM#1 fields
    final = {
        "gist": llm1.get("gist"),
        "sentiment": llm1.get("sentiment"),
        "tone": llm1.get("tone"),
        "status": "validator_unknown",
    }

    verdict_raw = llm2.get("verdict") or llm2.get("result") or llm2.get("status")
    verdict = str(verdict_raw).lower() if verdict_raw is not None else ""

    # New schema: accepted/corrected
    if verdict in {"accepted"}:
        final["status"] = "validated_pass"
        return final

    if verdict in {"corrected"}:
        final["status"] = "validated_with_corrections"
        for k in ("gist", "sentiment", "tone"):
            if llm2.get(k):
                final[k] = llm2.get(k)
        return final

    # Older schema: pass/fail + corrected_fields
    corrected = llm2.get("corrected_fields") or llm2.get("corrections") or {}
    if verdict in {"pass", "ok", "true"}:
        final["status"] = "validated_pass"
        return final

    if verdict in {"fail", "false", "error"}:
        final["status"] = "disputed"
        if isinstance(corrected, dict) and corrected:
            for k in ("gist", "sentiment", "tone"):
                if corrected.get(k):
                    final[k] = corrected[k]
            final["status"] = "validated_with_corrections"
        return final

    return final


def _format_validator_line(llm2_block: Dict[str, Any]) -> str:
    """Produce a single-line validation statement for the report."""
    if (llm2_block or {}).get("status") != "ok":
        err = (llm2_block or {}).get("error") or "validator unavailable"
        return f"Validator unavailable: {err}"

    v = (llm2_block or {}).get("result") or {}
    verdict = str(v.get("verdict") or "").lower()
    notes = v.get("notes") or v.get("issues") or v.get("validation")

    if verdict == "accepted":
        return "✓ Accepted." if not notes else f"✓ Accepted. Notes: {notes}"

    if verdict == "corrected":
        return "↺ Corrected." if not notes else f"↺ Corrected. Notes: {notes}"

    # Backward compatibility
    if verdict == "pass":
        return "✓ Correct." if not notes else f"✓ Correct. Notes: {notes}"
    if verdict == "fail":
        return "✗ Issues found." if not notes else f"✗ Issues: {notes}"

    return f"Validator responded (verdict={v.get('verdict')!r})."


def build_report_md(results: List[Dict[str, Any]], run_meta: Dict[str, Any]) -> str:
    final_sentiments = []
    for r in results:
        s = (r.get("final") or {}).get("sentiment")
        if s:
            final_sentiments.append(str(s).strip().capitalize())

    sentiment_counts = Counter(final_sentiments)

    analyzed_n = len(results)
    date_utc = run_meta.get("run_time_utc") or utc_now_iso()

    lines: List[str] = []
    lines.append("# News Analysis Report")
    lines.append("")
    lines.append(f"**Date:** {date_utc} (UTC)")
    lines.append(f"**Articles Analyzed:** {analyzed_n}")
    lines.append("**Source:** NewsAPI")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Positive: {sentiment_counts.get('Positive', 0)} articles")
    lines.append(f"- Negative: {sentiment_counts.get('Negative', 0)} articles")
    lines.append(f"- Neutral: {sentiment_counts.get('Neutral', 0)} articles")
    lines.append("")
    lines.append("## Detailed Analysis")
    lines.append("")

    for i, r in enumerate(results, start=1):
        a = r.get("article") or {}
        llm1 = (r.get("llm1") or {}).get("result") or {}
        final = r.get("final") or {}

        title = a.get("title") or "(no title)"
        url = a.get("url") or "(no url)"
        publisher = a.get("source") or "(unknown source)"
        published = a.get("published_at") or "(unknown time)"

        lines.append(f"### Article {i}: {title}")
        lines.append(f"- **Source:** {url}")
        lines.append(f"- **Publisher:** {publisher}")
        lines.append(f"- **Published:** {published}")
        lines.append(f"- **Gist:** {final.get('gist') or llm1.get('gist') or '(missing)'}")
        lines.append(f"- **LLM#1 Sentiment:** {llm1.get('sentiment') or '(missing)'}")
        lines.append(f"- **LLM#2 Validation:** {_format_validator_line(r.get('llm2') or {})}")
        lines.append(f"- **Tone:** {final.get('tone') or llm1.get('tone') or '(missing)'}")

        if final.get("status") == "validated_with_corrections":
            lines.append(f"- **Final Sentiment (corrected):** {final.get('sentiment')}")
            lines.append(f"- **Final Tone (corrected):** {final.get('tone')}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    setup_logging(args.debug)
    load_dotenv()

    # Required for running the pipeline
    newsapi_key = require_env("NEWSAPI_KEY")
    gemini_key = require_env("GEMINI_API_KEY")
    openrouter_key = require_env("OPENROUTER_API_KEY")

    # Optional but recommended (enables fallback in llm_analyzer/llm_validator)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = os.getenv("GROQ_MODEL", "").strip()
    if not groq_key or not groq_model:
        logger.warning(
            "Groq fallback is not fully configured (GROQ_API_KEY/GROQ_MODEL missing). "
            "If Gemini/OpenRouter fail, the run may have more LLM errors."
        )

    out_dir = Path(args.output_dir)
    ensure_output_dir(out_dir)

    raw_path = out_dir / RAW_ARTICLES_FILENAME
    results_path = out_dir / ANALYSIS_RESULTS_FILENAME
    report_path = out_dir / FINAL_REPORT_FILENAME

    logger.info("Starting pipeline | query=%r n=%d days=%d", args.query, args.n, args.days)

    # 1) Fetch
    try:
        fetched = fetch_news(api_key=newsapi_key, query=args.query, max_results=args.n, days=args.days)
    except Exception as e:
        logger.exception("Fetch failed: %s", e)
        return 1

    if not isinstance(fetched, list) or not fetched:
        logger.error("No articles returned from fetch step.")
        save_json(raw_path, fetched or [])
        return 1

    # Hygiene: filter + dedupe
    articles = filter_and_dedupe_articles(fetched, max_n=args.n)
    save_json(raw_path, articles)
    logger.info("Fetched=%d | After filter/dedupe=%d", len(fetched), len(articles))

    if not articles:
        logger.error("All fetched articles were filtered out (missing url/title).")
        return 1

    # 2) Analyze + 3) Validate
    enriched_results: List[Dict[str, Any]] = []
    for idx, article in enumerate(articles, start=1):
        url = (article.get("url") or "").strip()
        article_id = stable_article_id(url) if url else f"no_url_{idx}"

        meta = {
            "article_id": article_id,
            "title": article.get("title"),
            "source": _source_name(article),
            "published_at": _published_at(article),
            "url": url,
        }

        logger.info("Processing %d/%d | %s", idx, len(articles), article_id)

        llm1_block: Dict[str, Any] = {"status": "error", "error": None, "result": None}
        llm2_block: Dict[str, Any] = {"status": "skipped", "error": None, "result": None}
        final_block: Dict[str, Any] = {"status": "skipped"}

        # LLM#1 (Primary: Gemini; fallback handled inside llm_analyzer.py)
        try:
            analysis = analyze_article(article=article, api_key=gemini_key, model=None)
            llm1_block = {"status": "ok", "error": None, "result": analysis}
        except Exception as e:
            logger.exception("LLM#1 analysis failed for %s: %s", article_id, e)
            llm1_block = {"status": "error", "error": str(e), "result": None}

        # LLM#2 (Primary: OpenRouter; fallback handled inside llm_validator.py) (only if LLM#1 succeeded)
        if llm1_block["status"] == "ok" and isinstance(llm1_block["result"], dict):
            try:
                validation = validate_analysis(
                    article=article,
                    analysis=llm1_block["result"],
                    api_key=openrouter_key,
                    model=None,
                )
                llm2_block = {"status": "ok", "error": None, "result": validation}
            except Exception as e:
                logger.exception("LLM#2 validation failed for %s: %s", article_id, e)
                llm2_block = {"status": "error", "error": str(e), "result": None}

            # Final adjudication
            try:
                final_block = adjudicate_final(llm1_block["result"], llm2_block.get("result") or {})
            except Exception as e:
                logger.exception("Adjudication failed for %s: %s", article_id, e)
                final_block = {
                    "gist": llm1_block["result"].get("gist"),
                    "sentiment": llm1_block["result"].get("sentiment"),
                    "tone": llm1_block["result"].get("tone"),
                    "status": "adjudication_error",
                    "error": str(e),
                }
        else:
            final_block = {"status": "analysis_failed"}

        enriched_results.append(
            {
                "article": meta,
                "llm1": llm1_block,
                "llm2": llm2_block,
                "final": final_block,
            }
        )

    run_meta = {
        "run_time_utc": utc_now_iso(),
        "query": args.query,
        "requested_n": args.n,
        "fetched_n": len(articles),
        "days": args.days,
        "llm1_provider": "gemini",
        "llm2_provider": "openrouter",
        "gemini_model": os.getenv("GEMINI_MODEL", ""),
        "openrouter_model": os.getenv("OPENROUTER_MODEL", ""),
        "groq_fallback_enabled": bool(groq_key and groq_model),
        "groq_model": groq_model,
    }

    payload = {"run": run_meta, "results": enriched_results}
    save_json(results_path, payload)
    logger.info("Saved analysis results to %s", results_path.as_posix())

    report_md = build_report_md(enriched_results, run_meta)
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Saved report to %s", report_path.as_posix())

    pass_count = sum(1 for r in enriched_results if (r.get("final") or {}).get("status") == "validated_pass")
    corrected_count = sum(
        1 for r in enriched_results if (r.get("final") or {}).get("status") == "validated_with_corrections"
    )
    disputed_count = sum(1 for r in enriched_results if (r.get("final") or {}).get("status") == "disputed")
    analyzer_errors = sum(1 for r in enriched_results if (r.get("llm1") or {}).get("status") != "ok")
    validator_errors = sum(1 for r in enriched_results if (r.get("llm2") or {}).get("status") == "error")

    logger.info(
        "Done | total=%d pass=%d corrected=%d disputed=%d analyzer_errors=%d validator_errors=%d",
        len(enriched_results),
        pass_count,
        corrected_count,
        disputed_count,
        analyzer_errors,
        validator_errors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
