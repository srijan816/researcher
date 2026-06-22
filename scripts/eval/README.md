# Deep-Research Eval Harness

Benchmark questions, a job runner, and an LLM judge for measuring deep-research
report quality end to end against a running backend.

```text
scripts/eval/
  questions.yaml   18 benchmark questions with judging rubrics
  run_eval.py      submits jobs, polls to completion, saves artifacts
  judge.py         LLM judge -> scorecard.md / scorecard.json (+ --compare)
  runs/            output (gitignored runtime data; one folder per run)
```

## 1. Start the local stack

```bash
./scripts/run_minimax_deep_research.sh
curl -fsS http://127.0.0.1:9000/health
```

## 2. Run a subset of questions

```bash
# 2-question smoke run (medium-depth questions keep it fast)
uv run python scripts/eval/run_eval.py \
  --questions fin-hbm-memory-market,dec-vector-db-selection \
  --run-name smoke

# Full benchmark, two jobs at a time
uv run python scripts/eval/run_eval.py --run-name baseline --concurrency 2

# Override depth for everything (cheaper iteration)
uv run python scripts/eval/run_eval.py --depth medium --run-name baseline-medium
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `AIQ_EVAL_BASE_URL` | `http://127.0.0.1:9000` | backend base URL |
| `AIQ_EVAL_API_KEY` | unset | optional `Authorization: Bearer` token |
| `AIQ_EVAL_JOB_TIMEOUT_SECONDS` | `2400` | per-job polling timeout |

Each question writes `scripts/eval/runs/<run-name>/<qid>/` containing
`report.md`, `job.json`, `state.json`, `metrics.json` (and `error.json` on
failure). Re-running the same `--run-name` skips questions that already have a
`report.md`, so interrupted runs resume where they left off. Job failures are
recorded and the run continues.

## 3. Judge the run

```bash
uv run python scripts/eval/judge.py smoke
```

The judge calls an Anthropic-compatible `/v1/messages` endpoint once per
report (structured JSON output, retried with backoff) and scores five
dimensions 0-10: citation support, factual accuracy, comprehensiveness,
recency (only where the rubric requires it), and report cleanliness. It writes
`scorecard.md` and `scorecard.json` into the run directory and prints the
markdown table.

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `AIQ_EVAL_JUDGE_BASE_URL` | `$ANTHROPIC_BASE_URL` or `https://api.minimax.io/anthropic` | judge endpoint |
| `AIQ_EVAL_JUDGE_MODEL` | `MiniMax-M3` | judge model name |
| `MINIMAX_API_KEY` / `ANTHROPIC_API_KEY` | unset | judge auth (first non-empty wins) |

Judge only some questions with `--only id1,id2`.

## 4. Read the scorecard

`scorecard.md` has one row per question (each dimension, the question overall,
and efficiency columns pulled from `metrics.json` — search/tool calls and wall
duration) plus dimension means and an overall mean. `scorecard.json` is the
machine-readable equivalent used for comparisons.

## 5. Compare two runs

Run the same questions under two run names (e.g. before/after a prompt or
budget change), judge both, then:

```bash
uv run python scripts/eval/judge.py --compare \
  scripts/eval/runs/baseline/scorecard.json \
  scripts/eval/runs/candidate/scorecard.json
```

This prints a per-question overall-score diff and a dimension-mean diff table.

## Tests

```bash
uv run --extra dev pytest tests/eval
uv run ruff check scripts/eval tests/eval
```

The tests validate the questions schema, the judge's pure prompt-building and
parsing functions, and scorecard aggregation. They make no network calls.
