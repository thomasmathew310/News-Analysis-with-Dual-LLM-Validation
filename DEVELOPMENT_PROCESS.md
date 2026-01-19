# DEVELOPMENT_PROCESS.md — News Analysis with Dual LLM Validation

## 1) What I built (in one paragraph)
I implemented a small, production-style pipeline that fetches recent articles about Indian politics from NewsAPI, runs an **LLM analysis pass** (gist + sentiment + tone), then runs a **second LLM validation pass** to confirm or correct the first result, and finally saves (a) raw inputs, (b) per-article analyses, and (c) a human-readable summary report.

---

## 2) Requirements I extracted from the prompt
**Functional**
- Fetch **10–15** recent articles for `"India politics"` / `"India government"`.
- For each article, produce:
  - **Gist** (1–2 sentences)
  - **Sentiment**: positive / negative / neutral
  - **Tone**: e.g., urgent / analytical / satirical / balanced (open set, but controlled)
- Validate the analysis with a second LLM (“dual LLM validation”).
- Save outputs as files (JSON + report).

**Non-functional**
- Clean Python, modular structure, readable logs.
- No API keys in code (use `.env`).
- Document thinking + AI prompts (this file).
- Deterministic enough to run repeatedly without chaos (same inputs → similar outputs, within LLM limits).

---

## 3) Architecture (how I decomposed the system)
I split the pipeline into 4 responsibilities to keep each module small and testable:

1. **news_fetcher.py**
   - Talks to NewsAPI.
   - Returns a normalized list of `article` dicts.

2. **llm_analyzer.py (LLM #1)**
   - Takes one article.
   - Produces structured output: gist/sentiment/tone (+ a short evidence snippet).

3. **llm_validator.py (LLM #2)**
   - Takes the original article + LLM#1 analysis output.
   - Returns either:
     - “accepted” (no change), or
     - “corrected” (updated fields + short reason).

4. **main.py**
   - Orchestrates: fetch → analyze → validate → save → summary.
   - Handles CLI args, output directory, logging, and resilience.

This separation keeps “IO code” (API calls, file writes) away from “decision code” (prompting + parsing), which made debugging and iteration much faster.

---

## 4) Data model decisions (what I store and why)
### Article normalization
News articles from APIs are messy (missing `content`, empty `description`, duplicates). I normalize per article:
- `title`, `source`, `publishedAt`, `url`
- best-available text for analysis:
  - prefer `content`
  - else `description`
  - else `title`

### Stable ID per article
I generate a stable `article_id` (hash of URL). Reason:
- avoids duplicates
- makes output easy to diff across runs
- lets me re-run without confusion

### Output artifacts
- `output/raw_articles.json`
  - The exact inputs used (debugging + reproducibility).
- `output/analysis_results.json`
  - For each article: LLM#1 output + LLM#2 validation output + final chosen values.
- `output/final_report.md`
  - Human-readable report: sentiment distribution + per-article summary + validation outcome.

---

## 5) Prompting strategy (how I controlled LLM behavior)
The biggest risk was not “LLM quality”; it was **output stability**. If the LLM returns inconsistent formats, the pipeline breaks.

So I used these rules:
1. **Ask for JSON** with an explicit schema.
2. **Keep fields small and enumerable** (sentiment is a fixed set).
3. **Grounding instruction**: use only the provided article text; do not assume external facts.
4. **Defensive parsing**: tolerate minor formatting issues and fail gracefully when output cannot be recovered.

---

## 6) Prompt log (the exact style of prompts I used)
I kept prompts short, structured, and machine-parseable.

### 6.1 LLM #1 (Analyzer) — gist/sentiment/tone
**Goal:** produce an initial structured analysis.

**Prompt pattern (representative):**
- Role: “You are a news analyst. Use only the given article text.”
- Output: strict JSON.

Inputs provided:
- Title, Source, Published time
- Text (content/description/title fallback)

Required JSON schema:
- `gist` (string, 1–2 sentences)
- `sentiment` (“positive” | “negative” | “neutral”)
- `tone` (one label)
- `evidence` (short phrase from the article supporting the choice)

Why `evidence`?
- It forces the model to “point” at something in the text, which reduces hallucination and makes validation easier.

### 6.2 LLM #2 (Validator) — accept/correct
**Goal:** treat LLM#1 output as a hypothesis and check it against the text.

Prompt behavior:
- Provide article text + LLM#1 JSON.
- Ask validator to output JSON:
  - `verdict`: “accepted” or “corrected”
  - corrected fields (`gist`, `sentiment`, `tone`) when needed
  - `notes`: short reason(s)

Key instruction:
- “If the analysis is acceptable, accept it. Only correct when there’s a clear mismatch with the article text.”

This reduces unnecessary rewrites and makes validator behavior more consistent.

---

## 7) Engineering choices that improved reliability
### 7.1 Deduplication
NewsAPI can return duplicates/syndicated versions. I dedupe primarily by URL-derived ID.

### 7.2 Token control / text truncation
Some `content` fields are long or contain boilerplate. I truncate article text to a safe limit so requests stay stable and within free-tier limits.

### 7.3 Logging
I log:
- number of fetched articles
- which article is being processed
- analyzer/validator success/failure
- file paths written

This makes the run auditable without opening JSON files.

### 7.4 Dependency injection for tests
The LLM modules are structured so tests can monkeypatch internal LLM calls and return deterministic outputs. This ensures:
- tests do not require API keys
- CI/local runs stay deterministic

---

## 8) How I used AI assistance (and how I kept ownership)
I used AI like a pair programmer for drafts, but I kept control by:
- defining module boundaries and function contracts first (inputs/outputs)
- using AI mainly for scaffolding (argument parsing, request boilerplate, initial prompt wording)
- manually revising correctness-critical parts:
  - error handling and retries
  - JSON parsing robustness
  - logging clarity and output schema
  - keeping model names and keys in `.env` (no hardcoding)

Rule of thumb:
- If it affects correctness, schema, or failure modes, I reviewed and rewrote until I could explain it confidently.

---

## 9) Quick validation I performed (manual + programmatic)
### Manual spot checks
For a few articles, I verified:
- gist matches the article (no extra claims)
- sentiment is reasonable given the language
- tone label matches style (analytical vs urgent, etc.)
- validator corrections are justified (not random rewrites)

### Programmatic sanity checks
- gist is non-empty
- sentiment is always in {positive, negative, neutral}
- tone is non-empty
- output files exist and JSON is valid

### Gemini API verification (connectivity smoke test)
Before running the full pipeline, I verified Gemini credentials and model configuration using a minimal one-line test. This confirms:
- `.env` is being loaded correctly
- `GEMINI_API_KEY` is valid
- `GEMINI_MODEL` points to an available model
- the GenAI client can call `generate_content`

Command executed from the project root:

```powershell
python -c "import os; from dotenv import load_dotenv; load_dotenv(); from google import genai; c=genai.Client(api_key=os.environ['GEMINI_API_KEY']); m=os.environ['GEMINI_MODEL']; r=c.models.generate_content(model=m, contents='Say OK'); print(r.text)"
