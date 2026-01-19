"""
tests/test_analyzer.py

Unit tests for:
- llm_analyzer.analyze_article (LLM#1)
- llm_validator.validate_analysis (LLM#2)

These tests:
- Do NOT call real external APIs
- Monkeypatch internal LLM calls to return deterministic outputs
- Validate required fields + basic normalization/robustness

Important:
- This file does NOT rely on Python import paths.
- It loads llm_analyzer.py and llm_validator.py by absolute file path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


# ----------------------------
# Load modules by absolute path (no sys.path / PYTHONPATH dependency)
# ----------------------------

ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "llm_analyzer.py"
VALIDATOR_PATH = ROOT / "llm_validator.py"

if not ANALYZER_PATH.exists():
    raise RuntimeError(f"Cannot find {ANALYZER_PATH}. Check filename/path.")
if not VALIDATOR_PATH.exists():
    raise RuntimeError(f"Cannot find {VALIDATOR_PATH}. Check filename/path.")


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # register so monkeypatch works normally
    spec.loader.exec_module(module)
    return module


llm_analyzer = _load_module("llm_analyzer", ANALYZER_PATH)
llm_validator = _load_module("llm_validator", VALIDATOR_PATH)

LLMAnalysisError = getattr(llm_analyzer, "LLMAnalysisError", Exception)
LLMValidationError = getattr(llm_validator, "LLMValidationError", Exception)


# ----------------------------
# Helpers
# ----------------------------

ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}


def _find_key_recursive(obj: Any, key: str) -> Optional[Any]:
    """Find a key anywhere in a nested dict/list structure (returns first match)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None


@pytest.fixture
def sample_article() -> Dict[str, Any]:
    return {
        "source": {"id": None, "name": "Example News"},
        "author": "Reporter",
        "title": "Government announces new policy initiative",
        "description": "The government announced a new policy aimed at improving infrastructure.",
        "content": "In a press conference today, officials outlined a new policy initiative...",
        "url": "https://example.com/news/policy",
        "publishedAt": "2026-01-18T00:00:00Z",
    }


def _set_required_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Your code enforces that models come from .env.
    For unit tests, set dummy strings (NOT real model names).
    """
    monkeypatch.setenv("GEMINI_MODEL", "test-gemini-model")
    monkeypatch.setenv("OPENROUTER_MODEL", "test-openrouter-model")
    monkeypatch.setenv("GROQ_MODEL", "test-groq-model")

    # Keys are not used (we patch network calls), but keep set for completeness
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def _patch_llm_analyzer_gemini(monkeypatch: pytest.MonkeyPatch, response_payload: Dict[str, Any]) -> None:
    """
    Patch Gemini path so analyze_article() never imports SDKs or calls network.
    Preferred patch point: llm_analyzer._gemini_generate_once
    """
    _set_required_model_env(monkeypatch)
    fake_text = json.dumps(response_payload)

    if hasattr(llm_analyzer, "_gemini_generate_once"):
        monkeypatch.setattr(llm_analyzer, "_gemini_generate_once", lambda *a, **k: fake_text)
        return

    # Backward compatibility if your code uses a different helper
    if hasattr(llm_analyzer, "_generate"):
        monkeypatch.setattr(llm_analyzer, "_generate", lambda *a, **k: fake_text)
        return

    raise RuntimeError("Could not patch llm_analyzer; expected _gemini_generate_once or _generate to exist.")


def _patch_llm_validator_openrouter(monkeypatch: pytest.MonkeyPatch, response_payload: Dict[str, Any]) -> None:
    """
    Patch OpenRouter path so validate_analysis() never calls network.
    Preferred patch point: llm_validator._openrouter_once
    """
    _set_required_model_env(monkeypatch)
    fake_text = json.dumps(response_payload)

    if hasattr(llm_validator, "_openrouter_once"):
        monkeypatch.setattr(llm_validator, "_openrouter_once", lambda *a, **k: fake_text)
        return

    # Backward compatibility for older shapes
    for attr in ("_generate", "_call_openrouter", "_call_llm", "_post_openrouter"):
        if hasattr(llm_validator, attr):
            monkeypatch.setattr(llm_validator, attr, lambda *a, **k: fake_text)
            return

    raise RuntimeError("Could not patch llm_validator; expected _openrouter_once or legacy helper to exist.")


def _patch_llm_validator_openrouter_fail_and_groq_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    groq_response_payload: Dict[str, Any],
) -> None:
    """
    Force OpenRouter to fail with 'model not found' so fallback triggers,
    then patch Groq fallback to succeed.
    """
    _set_required_model_env(monkeypatch)

    if not hasattr(llm_validator, "_openrouter_once"):
        raise RuntimeError("Expected llm_validator._openrouter_once to exist for fallback test.")

    def _fail_openrouter(*a, **k):
        raise LLMValidationError("OpenRouter HTTP 404: model not found")

    monkeypatch.setattr(llm_validator, "_openrouter_once", _fail_openrouter)

    fake_groq_text = json.dumps(groq_response_payload)

    if hasattr(llm_validator, "_generate_with_groq"):
        monkeypatch.setattr(llm_validator, "_generate_with_groq", lambda *a, **k: fake_groq_text)
        return

    raise RuntimeError("Expected llm_validator._generate_with_groq to exist for fallback test.")


# ----------------------------
# Tests
# ----------------------------

def test_analyze_article_returns_required_fields(monkeypatch: pytest.MonkeyPatch, sample_article: Dict[str, Any]) -> None:
    _patch_llm_analyzer_gemini(
        monkeypatch,
        {
            "gist": "The government announced a new policy initiative to improve infrastructure.",
            "sentiment": "Positive",
            "tone": "Analytical",
        },
    )

    result = llm_analyzer.analyze_article(article=sample_article, api_key="fake-api-key", model=None)

    gist = _find_key_recursive(result, "gist")
    sentiment = _find_key_recursive(result, "sentiment")
    tone = _find_key_recursive(result, "tone")

    assert isinstance(gist, str) and gist.strip()
    assert isinstance(tone, str) and tone.strip()
    assert isinstance(sentiment, str) and sentiment.strip().lower() in ALLOWED_SENTIMENTS


def test_analyze_article_handles_empty_article_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_llm_analyzer_gemini(
        monkeypatch,
        {
            "gist": "No meaningful content was provided in the article fields.",
            "sentiment": "Neutral",
            "tone": "Balanced",
        },
    )

    empty_article = {"title": "", "description": "", "content": "", "url": ""}

    result = llm_analyzer.analyze_article(article=empty_article, api_key="fake-api-key", model=None)

    gist = _find_key_recursive(result, "gist")
    sentiment = _find_key_recursive(result, "sentiment")
    tone = _find_key_recursive(result, "tone")

    assert isinstance(gist, str) and gist.strip()
    assert isinstance(tone, str) and tone.strip()
    assert isinstance(sentiment, str) and sentiment.strip().lower() in ALLOWED_SENTIMENTS


def test_validate_analysis_corrects_invalid_sentiment(monkeypatch: pytest.MonkeyPatch, sample_article: Dict[str, Any]) -> None:
    _patch_llm_validator_openrouter(
        monkeypatch,
        {
            "gist": "The government announced a new policy initiative to improve infrastructure.",
            "sentiment": "Neutral",
            "tone": "Analytical",
            "verdict": "corrected",
            "notes": "Corrected sentiment to neutral because the article is informational and non-opinionated.",
        },
    )

    bad_analysis = {"gist": "Gov announced something.", "sentiment": "HAPPY", "tone": "Analytical"}

    result = llm_validator.validate_analysis(
        article=sample_article,
        analysis=bad_analysis,
        api_key="fake-api-key",
        model=None,
    )

    sentiment = _find_key_recursive(result, "sentiment")
    notes = _find_key_recursive(result, "notes") or _find_key_recursive(result, "validation")

    assert isinstance(sentiment, str) and sentiment.strip().lower() in ALLOWED_SENTIMENTS
    assert notes is None or (isinstance(notes, str) and notes.strip())


def test_validator_falls_back_to_groq_when_openrouter_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    sample_article: Dict[str, Any],
) -> None:
    _patch_llm_validator_openrouter_fail_and_groq_success(
        monkeypatch,
        groq_response_payload={
            "gist": "The government announced a new policy initiative to improve infrastructure.",
            "sentiment": "Neutral",
            "tone": "Analytical",
            "verdict": "accepted",
            "notes": "OpenRouter unavailable; validated using fallback model.",
        },
    )

    analysis = {"gist": "Gov announced something.", "sentiment": "NEUTRAL", "tone": "Analytical"}

    result = llm_validator.validate_analysis(
        article=sample_article,
        analysis=analysis,
        api_key="fake-openrouter-key",
        model=None,
    )

    provider = _find_key_recursive(result, "provider")
    sentiment = _find_key_recursive(result, "sentiment")

    assert isinstance(sentiment, str) and sentiment.strip().lower() in ALLOWED_SENTIMENTS
    assert provider == "groq"
