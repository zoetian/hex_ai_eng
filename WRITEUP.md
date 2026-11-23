# Hecks NL→SQL Agent — Writeup

## TL;DR

| Version | Change | Hard pass rate |
|---|---|---|
| v0 baseline | only `submit_answer` tool, generic prompt | 0/64 (0%) |
| v1 | discovery tools + auto-generated schema/guide index in system prompt + validated submit | 41/64 (64.1%) |
| v2 | agent loop always nudges to call a tool; guide-application rules; `rule_checklist` on submit; schema mentions in guide index | 41/64 (64.1%) — 5 fixed / 5 regressed (noise) |
| v3 | stricter (case-sensitive) guide→schema routing | 46/64 (71.9%) |
| v3-high | v3 + `reasoning.effort=high` | 44/64 (68.8%) — only 5 mismatches, but 12 hit max iterations |
| v4 | soft step-budget warning + transient network retries | 45/64 (70.3%) |
| v4-high | v4 + `reasoning.effort=high` | 43/64 (67.2%) — model ignored the soft warning (17 max-iteration) |
| **v5 (final, recommended)** | **hard budget: last 3 steps expose only `submit_answer`** | **48/64 (75.0%)**, 0 non-submissions, ~12s median, ~4.0M tokens |
| v5-high | v5 + `reasoning.effort=high` | 47/64 (73.4%) — 7 mismatches, 10 max-iteration; ~40s median, ~9.8M tokens |

Model: `openai/gpt-oss-120b:nitro` (repo default) throughout.

## How I worked

1. **Map the levers.** Traced `evaluate.py → create_tools() / Agent(OpenRouterConfig(api_key)) → Agent.run()`.
   Only four things shape behavior: the tool set, the system prompt, the model config, and the
   agent loop. The eval/compare code and the guides are off-limits.
2. **Baseline first.** 0%: the agent could not see the database at all and hallucinated tables.
3. **Error analysis drives every change.** `scripts/analyze_run.py` prints gold vs. submitted
   SQL and which guides were read for each failure; `scripts/diff_runs.py` diffs two runs
   case-by-case (fixed / regressed / still failing), which is essential because run-to-run
   variance on 64 cases is roughly ±3–5 cases.

## What I built

### Tools (all discovered at runtime — no hand-maintained schema map)
- `describe_schema(schema, tables?)` — tables, approx row counts, columns + types from the DuckDB catalog.
- `profile_column(table, column)` — NULL/distinct counts, min/max, top values. Surfaces sentinel
  values, codes and casing (e.g. `charge_code` is `''` for 99.99% of rows, `'RF'` for refunds).
- `run_sql(query)` — read-only exploratory execution with a 60s timeout and a compact preview.
- `search_guides(keywords)` / `read_guide(name)` — grep + full read of the guide directory
  (path overridable via `HECKS_GUIDES_DIR`).
- `submit_answer(rule_checklist, query)` — **executes the query before accepting it**; SQL errors
  are returned to the model to fix, and an empty result needs a confirming resubmission. The
  required `rule_checklist` argument forces the model to reconcile its query with each
  applicable rule before submitting (a structured self-check).

### System prompt (`framework/prompts.py`)
- Auto-generated **schema index** (schema → tables) and **guide index** (guide → title →
  schemas it mentions). This saves several exploratory steps per question.
- A workflow (route → read guide in full → inspect → verify → submit) and general rules for
  applying guidance: scope/clarification paragraphs are authoritative, data-quality exclusions
  apply to all measures, use the guide's category labels, use `FILTER` for per-measure
  conditions, aggregate to the stated grain before thresholds, explicit question > guide.

### Agent loop (`framework/agent.py`) and client (`framework/llm.py`)
- If the model stops with plain text (it often wrote SQL as prose), nudge it to call a tool
  (up to 3×) instead of ending with no submission. Fixed 7/7 no-submission failures.
- Step budget: a warning when 5 iterations remain, and in the last 3 iterations only
  `submit_answer` is exposed. The soft warning alone did not work (v4) — a hard constraint did.
- Retry transient network errors (TLS resets, connection errors, 5xx), not just 429/timeouts.
- `HECKS_REASONING_EFFORT` env var to run reasoning-effort experiments without code changes.

## Key findings

1. **Tools were worth far more than anything else** (0 → 64%). The model can't answer what it can't see.
2. **Retrieval/routing was the next bottleneck, not SQL skill.** Credit-card questions were being
   answered with the banking guide because naive case-insensitive matching tagged
   `financial_operations` with the `Credit` schema ("Credit transactions" is a Czech deposit type).
   Case-sensitive matching + requiring a title/code-literal mention or ≥2 mentions fixed routing (+5).
3. **Higher reasoning effort trades steps for accuracy.** With `effort=high`, mismatches dropped
   from 17 to 5, but 12 runs exhausted the 30-step budget; ~3.5× median latency and ~2× tokens.
4. **Hard constraints beat soft prompts for a weaker model.** The model ignored "only 5 steps
   left"; removing the exploration tools from the request worked for default reasoning.
5. **Noise matters.** v1→v2 had identical totals but 10 cases flipped. Decisions should be made on
   repeated runs and per-case diffs, not single totals.

## Overfitting guardrails
- No schema- or table-specific logic anywhere; indexes are generated from the catalog and the
  guide directory, so a new warehouse or rewritten guides need no code changes.
- Prompt rules were *motivated* by hard-set failures but written as general analytics principles
  (grain, per-measure filters, labels, scope paragraphs), not answers to specific questions.
- Guides untouched; evaluation code untouched except `create_tools()`.

## Recommendation
Ship v5 with default reasoning: best pass rate observed (75%), every question gets an answer,
~3.5× faster and ~2.5× cheaper than high reasoning. High reasoning is the more *accurate* config
when it finishes (7 mismatches vs 16) — it becomes the better choice once the step-budget issue
below is solved, so that is the highest-value next step.

## Known limitations / next steps
- **Make the hard budget airtight for high reasoning**: even when only `submit_answer` is offered,
  the model sometimes still calls hidden tools (copied from history) and the loop executes them.
  Reject non-finishing tools during the forced phase, and/or fall back to auto-submitting the last
  successful `run_sql` query.
- **Preserve reasoning across turns**: the loop never sends `reasoning_details` back to the API, so
  the reasoning model loses its prior chain of thought every turn — a likely cause of it re-reading
  the same guide 4–7 times. Passing reasoning back should cut steps and tokens substantially.
- **Run-to-run variance**: run each config 3× and report mean ± spread; consider self-consistency
  (sample N candidate queries, execute, majority-vote on result dataframes).
- **Rule application is still the main failure mode** with gpt-oss-120b (e.g. classification labels,
  per-measure exclusions). Ideas: a dedicated "extract applicable rules" sub-call that turns the
  guide into an explicit checklist before SQL writing; a verifier sub-agent that critiques the query
  against the guide.
- **Guide routing** is lexical; embeddings or an LLM router would handle guides that never name
  the schema. Some false positives remain (common words that are also schema names).
- **Context hygiene**: the model sometimes re-reads the same guide many times; caching/dedup or
  pinning the selected guide into the prompt would cut tokens.
- **Model choice**: a stronger model is likely the single biggest remaining lever, at a cost/latency tradeoff.

## How to run
```bash
unzip hecks.duckdb.zip && uv sync
echo "OPENROUTER_API_KEY=..." > .env && source .env
uv run evaluate --api-key "$OPENROUTER_API_KEY" --concurrency 12 --split hard
HECKS_REASONING_EFFORT=high uv run evaluate --api-key "$OPENROUTER_API_KEY" --concurrency 12
uv run python scripts/analyze_run.py --show-failures       # latest run
uv run python scripts/diff_runs.py logs/run_A/evals_hard logs/run_B/evals_hard
```
