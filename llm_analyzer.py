"""
llm_analyzer.py — LLM#1 (Primary) Analysis

Behavior:
- Primary provider: Gemini
- Fallback provider: Groq (ONLY if Gemini is rate-limited / quota-exhausted / unavailable)

Important constraint:
- This file must NOT contain real model names.
- Model names must come ONLY from .env.
- The only model string allowed in code is "__MODEL_NAME".
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MODEL_PLACEHOLDER = "__MODEL_NAME"


class LLMAnalysisError(RuntimeError):
    """Raised when analysis fails (Gemini + fallback Groq)."""


@dataclass(frozen=True)
class AnalyzerConfig:
    # Gemini (primary)
    gemini_api_key_env: str = "GEMINI_API_KEY"
    gemini_model_env: str = "GEMINI_MODEL"
    gemini_default_model: str = MODEL_PLACEHOLDER  # placeholder only
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 512
    gemini_retries_before_fallback: int = 1

    # Groq (fallback)
    groq_api_key_env: str = "GROQ_API_KEY"
    groq_model_env: str = "GROQ_MODEL"
    groq_default_model: str = MODEL_PLACEHOLDER  # placeholder only
    groq_temperature: float = 0.2
    groq_max_tokens: int = 512

    # Article handling
    max_article_chars_default: int = 12000


def analyze_article(
    article: Dict[str, Any],
    api_key: str,
    model: Optional[str] = None,
    *,
    request_id: Optional[str] = None,
    max_article_chars: int = 12000,
    cfg: AnalyzerConfig = AnalyzerConfig(),
) -> Dict[str, Any]:
    """
    Analyze a news article using Gemini first, fallback to Groq only if Gemini is rate-limited/unavailable.

    Notes:
    - `api_key` is treated as the Gemini key (kept for backward compatibility with your pipeline).
    - `model` (if passed) overrides Gemini model. Prefer setting GEMINI_MODEL in .env.
    """
    article_text = _build_article_text(article, max_chars=max_article_chars or cfg.max_article_chars_default)
    prompt = _build_prompt(article_text)

    gemini_model = model or os.getenv(cfg.gemini_model_env) or cfg.gemini_default_model
    if gemini_model == MODEL_PLACEHOLDER:
        raise LLMAnalysisError(
            f"Missing Gemini model. Set {cfg.gemini_model_env} in .env (no hardcoded model names allowed)."
        )

    # 1) Try Gemini (small retry budget, then fallback)
    try:
        raw = _generate_with_gemini(
            prompt,
            api_key=api_key,
            model=gemini_model,
            temperature=cfg.gemini_temperature,
            max_output_tokens=cfg.gemini_max_output_tokens,
            retries=cfg.gemini_retries_before_fallback,
            request_id=request_id,
        )
        parsed = _parse_analysis_json(raw)
        return _finalize(parsed, provider="gemini", model=gemini_model, request_id=request_id)

    except Exception as gemini_exc:
        if not _should_fallback_from_gemini(gemini_exc):
            raise LLMAnalysisError(f"Gemini analysis failed (non-fallback error): {gemini_exc}") from gemini_exc

        logger.warning(
            "Gemini failed (fallback to Groq). request_id=%s error=%s",
            request_id,
            repr(gemini_exc),
        )

    # 2) Fallback to Groq
    groq_key = os.getenv(cfg.groq_api_key_env)
    if not groq_key:
        raise LLMAnalysisError(
            f"Gemini failed and {cfg.groq_api_key_env} is not set, so Groq fallback cannot run."
        )

    groq_model = os.getenv(cfg.groq_model_env) or cfg.groq_default_model
    if groq_model == MODEL_PLACEHOLDER:
        raise LLMAnalysisError(
            f"Missing Groq model. Set {cfg.groq_model_env} in .env (no hardcoded model names allowed)."
        )

    try:
        raw = _generate_with_groq(
            prompt,
            api_key=groq_key,
            model=groq_model,
            temperature=cfg.groq_temperature,
            max_tokens=cfg.groq_max_tokens,
        )
        parsed = _parse_analysis_json(raw)
        return _finalize(parsed, provider="groq", model=groq_model, request_id=request_id)
    except Exception as groq_exc:
        raise LLMAnalysisError(f"Groq fallback also failed: {groq_exc}") from groq_exc


# ---------------------------
# Prompting
# ---------------------------

def _build_article_text(article: Dict[str, Any], *, max_chars: int) -> str:
    title = (article.get("title") or "").strip()
    desc = (article.get("description") or "").strip()
    content = (article.get("content") or "").strip()

    merged = "\n\n".join(
        [x for x in [
            f"TITLE:\n{title}" if title else "",
            f"DESCRIPTION:\n{desc}" if desc else "",
            f"CONTENT:\n{content}" if content else "",
        ] if x]
    ).strip()

    if not merged:
        merged = "TITLE:\n(unknown)\n\nCONTENT:\n(empty article text)"

    if max_chars and len(merged) > max_chars:
        merged = merged[:max_chars].rstrip() + "\n\n[TRUNCATED]"
    return merged


def _build_prompt(article_text: str) -> str:
    return f"""
You are a news analyst.

Task:
Given the article below, produce a STRICT JSON object with exactly these keys:
- "gist": 1–2 sentences summarizing the article.
- "sentiment": one of ["positive","negative","neutral"] about the overall story.
- "tone": a short label such as "urgent", "analytical", "balanced", "critical", "investigative", "satirical", "explanatory", or similar.

Rules:
- Output ONLY valid JSON (no markdown, no code fences, no extra text).
- Keep "tone" to 1–3 words.

ARTICLE:
{article_text}
""".strip()


# ---------------------------
# Gemini (Primary)
# ---------------------------

_GEMINI_CLIENT_CACHE: Dict[str, Any] = {}


def _generate_with_gemini(
    prompt: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    request_id: Optional[str],
) -> str:
    last_exc: Optional[BaseException] = None

    for attempt in range(retries + 1):
        try:
            return _gemini_generate_once(
                prompt,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break

            if _looks_like_rate_limit(exc):
                backoff = min(2.0 * (attempt + 1), 4.0)
                logger.info(
                    "Gemini rate-limited; retrying after %.1fs. request_id=%s",
                    backoff,
                    request_id,
                )
                time.sleep(backoff)
                continue

            time.sleep(0.5 * (attempt + 1))

    raise last_exc if last_exc else RuntimeError("Gemini failed with unknown error.")


def _gemini_generate_once(
    prompt: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    """
    Preferred: google-genai (new SDK).
    Fallback: google-generativeai (legacy SDK), if installed.
    """
    try:
        from google import genai
        from google.genai import types

        client = _GEMINI_CLIENT_CACHE.get(api_key)
        if client is None:
            client = genai.Client(api_key=api_key)
            _GEMINI_CLIENT_CACHE[api_key] = client

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        text = getattr(resp, "text", None)
        if not text or not str(text).strip():
            raise LLMAnalysisError("Gemini returned empty text.")
        return str(text).strip()

    except ImportError:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        resp = m.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
        text = getattr(resp, "text", None)
        if not text or not str(text).strip():
            raise LLMAnalysisError("Gemini (legacy SDK) returned empty text.")
        return str(text).strip()


def _should_fallback_from_gemini(exc: BaseException) -> bool:
    if _looks_like_rate_limit(exc):
        return True

    msg = str(exc).lower()
    transient_markers = [
        "timeout",
        "timed out",
        "temporarily unavailable",
        "service unavailable",
        "internal error",
        "connection reset",
        "connection aborted",
        "connection error",
        "failed to establish a new connection",
        "503",
        "500",
        "502",
        "504",
    ]
    return any(m in msg for m in transient_markers)


def _looks_like_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    markers = [
        "429",
        "too many requests",
        "rate limit",
        "quota",
        "resource exhausted",
        "exceeded",
        "tpm",
        "rpm",
        "requests per minute",
        "tokens per minute",
    ]
    return any(m in msg for m in markers)


# ---------------------------
# Groq (Fallback)
# ---------------------------

_GROQ_CLIENT_CACHE: Dict[str, Any] = {}


def _generate_with_groq(
    prompt: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        from groq import Groq
    except ImportError as exc:
        raise LLMAnalysisError("Missing dependency: pip install groq") from exc

    client = _GROQ_CLIENT_CACHE.get(api_key)
    if client is None:
        client = Groq(api_key=api_key)
        _GROQ_CLIENT_CACHE[api_key] = client

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = completion.choices[0].message.content
    if not text or not str(text).strip():
        raise LLMAnalysisError("Groq returned empty text.")
    return str(text).strip()


# ---------------------------
# JSON parsing / output normalization
# ---------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_analysis_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = _JSON_RE.search(raw)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise LLMAnalysisError(f"Model output was not valid JSON: {raw[:400]}")


def _finalize(parsed: Dict[str, Any], *, provider: str, model: str, request_id: Optional[str]) -> Dict[str, Any]:
    gist = str(parsed.get("gist", "")).strip()
    sentiment = str(parsed.get("sentiment", "")).strip().lower()
    tone = str(parsed.get("tone", "")).strip()

    if sentiment not in {"positive", "negative", "neutral"}:
        sentiment = "neutral"

    if not gist:
        gist = "(missing gist)"
    if not tone:
        tone = "balanced"

    return {
        "gist": gist,
        "sentiment": sentiment,
        "tone": tone,
        "_meta": {
            "provider": provider,
            "model": model,  # comes from .env (no hardcoded model names in code)
            **({"request_id": request_id} if request_id else {}),
        },
    }
