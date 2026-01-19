# DESIGN_DOC.md — News Analysis with Dual LLM Validation (Gemini + OpenRouter with Groq Fallback)

## 1) Overview

This repository implements a small, production-style **news analysis pipeline** for recent Indian politics coverage. The system:
1) fetches recent articles from **NewsAPI**
2) generates **gist + sentiment + tone** using **LLM#1 (Gemini primary)**
3) validates/corrects that analysis using **LLM#2 (OpenRouter primary)**
4) uses **Groq as a fallback provider** when the primary LLM provider fails due to rate limits, quota, timeouts, or transient server errors
5) exports JSON artifacts and a human-readable report

The design focuses on **structured outputs**, **auditability**, and **reliability under API constraints**.

---

## 2) Problem Statement

Given a topic query such as `"India politics"` (or `"India government"`), the pipeline should fetch ~10–15 recent articles and, for each article:
- generate a 1–2 sentence **gist**
- classify **sentiment** as **positive / negative / neutral**
- classify **tone** such as urgent/analytical/satirical/balanced/etc.
- validate the analysis with a second LLM and correct if needed

The pipeline must save outputs to disk and avoid committing API keys.

---

## 3) Goals and Non-Goals

### Goals
- **Correctness of workflow:** fetch → analyze → validate → export results.
- **Structured output:** enforce JSON outputs from LLMs so downstream logic is deterministic.
- **Dual-LLM validation:** reduce single-model mistakes and improve quality consistency.
- **Reliability:** automatic fallback when a provider hits RPM/TPM/quota issues or transient failures.
- **Reproducibility:** save raw inputs and both analysis stages to support debugging and review.

### Non-Goals
- External fact-checking against authoritative databases (the system analyzes only the article text provided by NewsAPI).
- UI/dashboard (the deliverable is CLI + saved artifacts).
- Long-term storage, streaming, or distributed processing (kept simple for take-home scope).

---

## 4) High-Level Architecture

### Pipeline Flow
┌──────────────────────────┐
│ 1) CLI / Config          │
│ - Read args: query,n,days│
│ - Load .env API keys     │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ 2) Fetch News (NewsAPI)  │
│ news_fetcher.fetch_news  │
│ - call NewsAPI           │
│ - normalize fields       │
│ - dedupe articles        │
└─────────────┬────────────┘
              │  articles[]
              v
┌──────────────────────────┐
│ 3) Analyze (LLM #1)      │
│ llm_analyzer.analyze_article
│ Primary: Gemini          │
│ - prompt -> JSON         │
│ - parse + schema-check   │
│ If Gemini fails:         │
│   retry -> fallback Groq │
└─────────────┬────────────┘
              │  analysis[]
              v
┌──────────────────────────┐
│ 4) Validate (LLM #2)     │
│ llm_validator.validate_analysis
│ Primary: OpenRouter      │
│ - compare vs article     │
│ - correct if needed      │
│ - output JSON            │
│ If OpenRouter fails:     │
│   retry -> fallback Groq │
└─────────────┬────────────┘
              │  validation[]
              v
┌──────────────────────────┐
│ 5) Export + Report       │
│ main.py writes:          │
│ - output/raw_articles.json
│ - output/analysis_results.json
│ - output/final_report.md │
└──────────────────────────┘


### Component Responsibilities

- **`main.py`**
  - Orchestrates the end-to-end run
  - Loads environment variables
  - Calls fetch/analyze/validate functions
  - Writes output artifacts + final report

- **`news_fetcher.py`**
  - Fetches recent articles from NewsAPI using query + days window
  - Normalizes article fields
  - Performs lightweight deduplication (typical: title/source/url hash)

- **`llm_analyzer.py` (LLM#1)**
  - Builds analysis prompt from article content/description
  - Enforces JSON-only output format
  - **Primary:** Gemini
  - **Fallback:** Groq (only if Gemini fails)
  - Parses/validates JSON response and adds metadata `_meta`

- **`llm_validator.py` (LLM#2)**
  - Builds validation prompt using article + LLM#1 output
  - Validates/corrects gist/sentiment/tone
  - **Primary:** OpenRouter
  - **Fallback:** Groq (only if OpenRouter fails)
  - Parses/validates JSON response and adds metadata `_meta`

---

## 5) Data Flow (Step-by-step)

1. **Configuration**
   - Load `.env` (keys + optional model overrides)
   - Read CLI args (query, n, days)

2. **Fetch**
   - `news_fetcher.fetch_news(NEWSAPI_KEY, query, max_results=n, days=days)`
   - Output: list of normalized article dicts

3. **Analyze (LLM#1)**
   - For each article:
     - `llm_analyzer.analyze_article(article, GEMINI_API_KEY, ...)`
     - Gemini is attempted first
     - If Gemini fails (rate limit/quota/timeout/5xx/invalid JSON), retry briefly then fallback to Groq
     - Output: JSON analysis (gist/sentiment/tone) + `_meta`

4. **Validate (LLM#2)**
   - For each article + analysis:
     - `llm_validator.validate_analysis(article, analysis, OPENROUTER_API_KEY, ...)`
     - OpenRouter is attempted first
     - If OpenRouter fails, retry briefly then fallback to Groq
     - Output: validated JSON (gist/sentiment/tone/is_valid/notes) + `_meta`

5. **Export**
   - Save raw fetched articles
   - Save combined results (article + analysis + validation)
   - Generate a Markdown report summarizing results

---

## 6) Data Contracts (Schemas)

### 6.1 Normalized Article (from `news_fetcher.py`)
The system expects each article as a dictionary containing (where available):
- `title`: string
- `source`: dict or string (often `{"name": "..."}`)
- `url`: string
- `publishedAt`: string
- `description`: string (optional)
- `content`: string (optional)

For prompting, the system uses:
- `content` if present, else `description`, else empty string  
and truncates text to a safe maximum length to keep prompts bounded.

---

### 6.2 LLM#1 Output (Analyzer)
**Required keys**
- `gist`: string (1–2 sentences)
- `sentiment`: one of `["positive", "negative", "neutral"]`
- `tone`: string (e.g., urgent/analytical/satirical/balanced/other)

**Metadata**
- `_meta.provider_used`: `"gemini"` or `"groq"`
- `_meta.model_used`: string
- `_meta.fallback_reason`: optional string when fallback occurs

**Example**
```json
{
  "gist": "The government introduced a new policy proposal focused on administrative reforms.",
  "sentiment": "neutral",
  "tone": "analytical",
  "_meta": {
    "provider_used": "gemini",
    "model_used": "gemini-1.5-flash"
  }
}
### 6.3 LLM#2 Output (Validator)
**Required keys**
- `gist`: string (may be corrected)
- `sentiment`: one of `["positive", "negative", "neutral"]`
- `tone`: string
- `is_valid`: boolean
- `notes`: short string explaining what was corrected or why it is valid

**Metadata**
- `_meta.provider_used`: `"openrouter"` or `"groq"`
- `_meta.model_used`: string
- `_meta.fallback_reason`: optional string when fallback occurs

**Example**
```json
{
  "gist": "Opposition parties criticized the proposed change, arguing it weakens oversight mechanisms.",
  "sentiment": "negative",
  "tone": "balanced",
  "is_valid": false,
  "notes": "LLM#1 summary missed the opposition criticism; sentiment adjusted to negative.",
  "_meta": {
    "provider_used": "openrouter",
    "model_used": "google/gemini-2.0-flash-exp"
  }
}
## 6.4 Combined Result Record (saved in `output/analysis_results.json`)

For each article, the pipeline stores:

- `article`: normalized article dict  
- `analysis`: LLM#1 output JSON  
- `validation`: LLM#2 output JSON  

This structure preserves full lineage for auditing and debugging.

---

## 7) Prompting Strategy

### JSON-only prompting
Both LLM stages require the model to output **strict JSON only**, because:
- downstream parsing becomes deterministic
- report generation becomes straightforward
- it reduces brittle string parsing logic

### Bounded inputs
NewsAPI text can be long or truncated. To keep latency and token usage stable:
- use `content` when available, else `description`
- truncate to `max_article_chars`

### Guardrails
- `sentiment` is limited to a fixed set: `positive | negative | neutral`
- `tone` is encouraged to be one of a small known set (urgent/analytical/satirical/balanced/other)
- the validator is instructed to correct mismatches, not just “agree”

### Robust JSON parsing
If an LLM wraps JSON with extra text:
- attempt direct `json.loads`
- if that fails, extract the first `{ ... }` block and parse it
- validate required keys and allowed values

---

## 8) Reliability, Error Handling, and Fallbacks

### 8.1 Why fallbacks are necessary
LLM providers may fail due to:
- HTTP 429 (RPM/TPM rate limits)
- quota exhaustion
- timeouts
- transient server errors (502/503/504)
- empty responses
- malformed / non-JSON output

### 8.2 Fallback policy (implemented)
- **Analyzer:** Gemini primary → Groq fallback  
  Fallback occurs only after Gemini fails (after limited retries).
- **Validator:** OpenRouter primary → Groq fallback  
  Fallback occurs only after OpenRouter fails (after limited retries).

### 8.3 Retry strategy (primary only)
- retry a small number of times for transient failures
- use simple backoff between attempts
- then switch to fallback to keep the pipeline moving

### 8.4 Failure visibility (observability)
Each stage includes `_meta` fields capturing:
- provider used (`gemini` / `openrouter` / `groq`)
- model used
- fallback reason (when applicable)

This makes it easy to explain pipeline behavior during evaluation.

---

## 9) Configuration and Secrets

All API keys are loaded from `.env` and must not be committed.

### Required environment variables
- `NEWSAPI_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`

### Optional model overrides
- `GEMINI_MODEL` (default: `gemini-1.5-flash`)
- `OPENROUTER_MODEL` (example: `google/gemini-2.0-flash-exp`)
- `GROQ_MODEL` (default: `llama-3.1-8b-instant`)

---

## 10) Output Artifacts

The pipeline writes these artifacts for auditability and review:

### `output/raw_articles.json`
- normalized and deduped articles fetched from NewsAPI
- helpful to verify what the LLMs actually received

### `output/analysis_results.json`
- combined record per article: `{article, analysis, validation}`
- includes provider metadata to show fallback usage

### `output/final_report.md`
- human-readable summary
- typically includes:
  - per-article final gist/sentiment/tone
  - aggregate counts (e.g., sentiment distribution)
  - notes where validator corrected the analyzer

---

## 11) Testing Strategy

### Unit tests (recommended)
- JSON parsing helper:
  - valid JSON
  - JSON surrounded by extra text
  - invalid JSON
- schema validation:
  - missing keys
  - invalid sentiment label
- deduplication hashing (if implemented in fetcher)

### Integration test
- run with small parameters for quick checks:
  - `--n 3 --days 3`
- confirm:
  - outputs are created
  - JSON is valid
  - fallback works when a primary key is intentionally invalid (optional test)

---

## 12) Limitations and Future Improvements

### Limitations
- NewsAPI content can be truncated; analysis quality depends on available text.
- sentiment/tone is subjective and depends on prompt/model interpretation.
- no external fact-checking beyond the provided article text.

### Future improvements
- caching (avoid re-analyzing the same URL)
- better dedupe (canonical URL + fuzzy title matching)
- structured logging and metrics:
  - provider latency
  - fallback rates
  - JSON parse error rate
- add a confidence score for sentiment/tone
- language detection + multilingual support

---

## 13) How to Run

### Install dependencies
```bash
pip install -r requirements.txt

NEWSAPI_KEY=...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
GROQ_API_KEY=...

# Optional
GEMINI_MODEL=gemini-1.5-flash
OPENROUTER_MODEL=google/gemini-2.0-flash-exp
GROQ_MODEL=llama-3.1-8b-instant

# Run
python main.py --query "India politics" --n 10 --days 7
