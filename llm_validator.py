"""
llm_validator.py — LLM#2 Validation (Primary: OpenRouter, Fallback: Groq)

Requirement:
- No real model names in code. Models must be set in .env only.
- Only allowed model string in code is "__MODEL_NAME".

Interface expected by main.py:
- validate_analysis(article: dict, analysis: dict, api_key: str, model: Optional[str] = None) -> dict
  where api_key is the OpenRouter key (kept for backward compatibility).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

MODEL_PLACEHOLDER = "__MODEL_NAME"


class LLMValidationError(RuntimeError):
    """Raised when validation fails (OpenRouter + fallback Groq)."""


@dataclass(frozen=True)
class ValidatorConfig:
    # OpenRouter (primary)
    openrouter_api_key_env: str = "OPENROUTER_API_KEY"
    openrouter_model_env: str = "OPENROUTER_MODEL"
    openrouter_default_model: str = MODEL_PLACEHOLDER
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_timeout_s: int = 30
    openrouter_temperature: float = 0.0
    openrouter_max_tokens: int = 600
    openrouter_retries_before_fallback: int = 1

    # Groq (fallback)
    groq_api_key_env: str = "GROQ_API_KEY"
    groq_model_env: str = "GROQ_MODEL"
    groq_default_model: str = MODEL_PLACEHOLDER
    groq_temperature: float = 0.0
    groq_max_tokens: int = 600

    # Article handling
    max_article_chars_default: int = 12000


def validate_analysis(
    article: Dict[str, Any],
    analysis: Dict[str, Any],
    api_key: str,
    model: Optional[str] = None,
    *,
    request_id: Optional[str] = None,
    max_article_chars: int = 12000,
    cfg: ValidatorConfig = ValidatorConfig(),
) -> Dict[str, Any]:
    """
    Validate/correct LLM#1 analysis using OpenRouter first, fallback to Groq only if OpenRouter model is unavailable.

    Notes:
    - `api_key` is treated as OpenRouter API key (kept for backward compatibility).
    - `model` (if passed) overrides OpenRouter model. Prefer setting OPENROUTER_MODEL in .env.
    """
    article_text = _build_article_text(article, max_chars=max_article_chars or cfg.max_article_chars_default)
    prompt = _build_prompt(article_text, analysis)

    openrouter_model = model or os.getenv(cfg.openrouter_model_env) or cfg.openrouter_default_model
    if openrouter_model == MODEL_PLACEHOLDER:
        raise LLMValidationError(
            f"Missing OpenRouter model. Set {cfg.openrouter_model_env} in .env (no hardcoded model names allowed)."
        )

    # 1) Try OpenRouter (small retry budget)
    try:
        raw = _generate_with_openrouter(
            prompt,
            api_key=api_key or os.getenv(cfg.openrouter_api_key_env) or "",
            model=openrouter_model,
            temperature=cfg.openrouter_temperature,
            max_tokens=cfg.openrouter_max_tokens,
            timeout_s=cfg.openrouter_timeout_s,
            retries=cfg.openrouter_retries_before_fallback,
            request_id=request_id,
            base_url=cfg.openrouter_base_url,
        )
        parsed = _parse_json(raw)
        return _finalize(parsed, provider="openrouter", model=openrouter_model, request_id=request_id)

    except Exception as or_exc:
        # Fallback ONLY when OpenRouter model is not available / service unavailable
        if not _should_fallback_from_openrouter(or_exc):
            raise LLMValidationError(f"OpenRouter validation failed (non-fallback error): {or_exc}") from or_exc

        logger.warning(
            "OpenRouter failed (fallback to Groq). request_id=%s error=%s",
            request_id,
            repr(or_exc),
        )

    # 2) Fallback to Groq
    groq_key = os.getenv(cfg.groq_api_key_env)
    if not groq_key:
        raise LLMValidationError(
            f"OpenRouter failed and {cfg.groq_api_key_env} is not set, so Groq fallback cannot run."
        )

    groq_model = os.getenv(cfg.groq_model_env) or cfg.groq_default_model
    if groq_model == MODEL_PLACEHOLDER:
        raise LLMValidationError(
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
        parsed = _parse_json(raw)
        return _finalize(parsed, provider="groq", model=groq_model, request_id=request_id)
    except Exception as groq_exc:
        raise LLMValidationError(f"Groq fallback also failed: {groq_exc}") from groq_exc


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


def _build_prompt(article_text: str, analysis: Dict[str, Any]) -> str:
    # We ask the validator to either confirm or correct, but always return gist/sentiment/tone.
    gist = str(analysis.get("gist", "")).strip()
    sentiment = str(analysis.get("sentiment", "")).strip()
    tone = str(analysis.get("tone", "")).strip()

    return f"""
You are validating an LLM-generated news analysis.

Given:
1) The article text
2) The initial analysis (gist/sentiment/tone)

Your job:
- If the initial analysis is good, return the same values.
- If it is wrong or misleading, correct it.

Output STRICT JSON ONLY with exactly these keys:
- "gist": 1–2 sentences
- "sentiment": one of ["positive","negative","neutral"]
- "tone": 1–3 words
- "verdict": one of ["accepted","corrected"]
- "notes": a short reason (1 sentence)

Rules:
- Output ONLY valid JSON (no markdown, no extra text).

INITIAL_ANALYSIS:
{{
  "gist": {json.dumps(gist)},
  "sentiment": {json.dumps(sentiment)},
  "tone": {json.dumps(tone)}
}}

ARTICLE:
{article_text}
""".strip()


# ---------------------------
# OpenRouter (Primary)
# ---------------------------

_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def _generate_with_openrouter(
    prompt: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
    request_id: Optional[str],
    base_url: str,
) -> str:
    if not api_key:
        raise LLMValidationError("OpenRouter API key is missing (OPENROUTER_API_KEY or api_key param).")

    last_exc: Optional[BaseException] = None

    for attempt in range(retries + 1):
        try:
            return _openrouter_once(
                prompt,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                base_url=base_url,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break

            # If OpenRouter is rate-limited or transient, brief backoff
            if _looks_like_rate_limit(exc) or _looks_like_transient(exc):
                backoff = min(1.5 * (attempt + 1), 3.0)
                logger.info("OpenRouter transient/rate-limited; retrying after %.1fs request_id=%s", backoff, request_id)
                time.sleep(backoff)
                continue

            time.sleep(0.4 * (attempt + 1))

    raise last_exc if last_exc else RuntimeError("OpenRouter failed with unknown error.")


def _openrouter_once(
    prompt: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    base_url: str,
) -> str:
    session = _get_session()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but commonly recommended by OpenRouter:
        # "HTTP-Referer": "https://your-app.example",
        # "X-Title": "news-analyzer",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = session.post(base_url, headers=headers, json=payload, timeout=timeout_s)
    if r.status_code >= 400:
        # include response body to help detection of model-not-available
        raise LLMValidationError(f"OpenRouter HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LLMValidationError(f"OpenRouter response parsing error: {data}") from exc

    if not text or not str(text).strip():
        raise LLMValidationError("OpenRouter returned empty text.")
    return str(text).strip()


def _should_fallback_from_openrouter(exc: BaseException) -> bool:
    # User requirement: fallback when OpenRouter model is not available.
    # We detect common “model not available” indicators plus major unavailability.
    msg = str(exc).lower()

    model_unavailable_markers = [
        "model not found",
        "not found",
        "no endpoints",
        "no endpoints found",
        "model is not available",
        "is not available",
        "does not exist",
        "unknown model",
        "invalid model",
        "not supported",
        "provider unavailable",
    ]
    if any(m in msg for m in model_unavailable_markers):
        return True

    # Also treat these as "not available" in practice:
    if _looks_like_rate_limit(exc) or _looks_like_transient(exc):
        return True

    return False


def _looks_like_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ["timeout", "timed out", "503", "502", "504", "service unavailable", "bad gateway"])


def _looks_like_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ["429", "too many requests", "rate limit", "quota", "exceeded"])


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
        raise LLMValidationError("Missing dependency: pip install groq") from exc

    client = _GROQ_CLIENT_CACHE.get(api_key)
    if client is None:
        client = Groq(api_key=api_key)
        _GROQ_CLIENT_CACHE[api_key] = client

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = completion.choices[0].message.content
    if not text or not str(text).strip():
        raise LLMValidationError("Groq returned empty text.")
    return str(text).strip()


# ---------------------------
# JSON parsing / output normalization
# ---------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()

    # strict JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # extract first JSON object if model adds noise
    m = _JSON_RE.search(raw)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise LLMValidationError(f"Model output was not valid JSON: {raw[:400]}")


def _finalize(parsed: Dict[str, Any], *, provider: str, model: str, request_id: Optional[str]) -> Dict[str, Any]:
    gist = str(parsed.get("gist", "")).strip()
    sentiment = str(parsed.get("sentiment", "")).strip().lower()
    tone = str(parsed.get("tone", "")).strip()
    verdict = str(parsed.get("verdict", "")).strip().lower()
    notes = str(parsed.get("notes", "")).strip()

    if sentiment not in {"positive", "negative", "neutral"}:
        sentiment = "neutral"
    if not gist:
        gist = "(missing gist)"
    if not tone:
        tone = "balanced"
    if verdict not in {"accepted", "corrected"}:
        verdict = "accepted" if notes == "" else "corrected"

    return {
        "gist": gist,
        "sentiment": sentiment,
        "tone": tone,
        "verdict": verdict,
        "notes": notes,
        "_meta": {
            "provider": provider,
            "model": model,  # comes from .env (no hardcoded model names in code)
            **({"request_id": request_id} if request_id else {}),
        },
    }
