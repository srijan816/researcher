# Deep Research App — Integration Brief

A short guide for tools and agents that consume this research service. It explains
what the app does, how a question becomes a report, what quality to expect, and
how to submit work and retrieve output.

## What this is

A self-hosted deep-research pipeline built on NVIDIA AI-Q / NeMo Agent Toolkit,
powered by MiniMax M3 for all reasoning and synthesis. Given a research question
it plans, searches the live web (plus a local corpus of previously scraped
pages), verifies claims against stored evidence, and writes a long-form cited
report. Everything runs on our own infrastructure — searches go through
self-hosted SearXNG/Websurfx/DDGS with optional Exa, page extraction through
Scrapling, and result reranking through NVIDIA NIM cloud models.

## Content pathway (question → report)

1. **Clarify & approve.** The clarifier checks whether the question is
   answerable as asked. If scope is ambiguous it asks one short clarifying
   question. It then proposes a research plan (title + section TOC) for
   approval. Once approved, the plan becomes the scope contract for the run.
2. **Plan.** A planner agent converts the approved scope into an executable
   plan: 3–5 researcher tasks, each with target claims, search budgets,
   required source classes, and seed queries.
3. **Research (parallel).** Researcher agents execute their tasks
   concurrently. Each search pass: discovery across multiple engines →
   cross-encoder reranking (NIM) → scraping of the top candidates → extraction
   quality scoring → recency/freshness scoring. Every scraped page is persisted
   as an artifact and indexed; repeat lookups hit a TTL'd scrape cache and a
   local embedding corpus (~28k docs) before touching the live web.
4. **Verify.** A deterministic adversarial verifier re-checks high-risk claims
   against the stored evidence (not against model memory). Citations are
   matched to actual scraped URLs with fuzzy canonicalization; unverifiable
   claims are flagged or dropped.
5. **Gap-fill.** If verification exposes coverage gaps, a bounded gap-fill
   loop (budget depends on depth tier) runs targeted follow-up research.
6. **Synthesize.** An orchestrator with extended thinking writes the final
   report following the approved TOC, drawing only on the verified evidence
   packet. Verified facts about recurring entities persist to a cross-run fact
   ledger that seeds future jobs.

## Output contract

- The deliverable is a long-form markdown report (`/report.md`) following the
  approved section TOC, with inline numbered citations resolving to real
  scraped source URLs.
- Reports include a claim-verification summary (verified / partially verified /
  unverified counts). Treat unverified claims as leads, not facts.
- Typical depth ("deeper" tier): 40–77 distinct sources, 5 researcher tasks,
  multi-thousand-word report. Runtime is roughly 20–45 minutes depending on
  depth and topic.

## Depth tiers

| Tier | Use for | Behavior |
|---|---|---|
| `shallow` | Quick lookups, single-fact questions | Few searches, no gap-fill, fast |
| `medium` | Focused briefs | Moderate budget, 1 gap-fill round |
| `deeper` | Standard deep research (default for reports) | Max 5 researcher tasks, 16 searches each, 2 gap-fill rounds |
| `deep` | Exhaustive audits | Largest budgets and reserve, 3 gap-fill rounds |

## How to submit and get output

Base URL: `http://localhost:9000` on the server (proxied publicly through the
app UI). All endpoints require the deployment's auth unless called internally.

**Routed chat (simplest).** `POST /chat` with the user query. Shallow questions
return inline answers; research questions return
`{"status": "deep_research_running", "job_id": "..."}`.

**Direct job submission.**

```
POST /v1/jobs/async/submit
{
  "agent_type": "deep_researcher",
  "input": "<the research question>",
  "research_depth": "deeper",
  "webhook_url": "https://example.com/hook"   // optional, called on terminal status
}
```

**Tracking and retrieval.**

- `GET /v1/jobs/async/job/{job_id}` — status (`submitted|running|success|failure|interrupted`)
- `GET /v1/jobs/async/job/{job_id}/stream` — SSE event stream (progress, tool calls, artifacts)
- `GET /v1/jobs/async/job/{job_id}/report` — final report; `?format=md` for raw markdown, `?format=docx` for Word
- `GET /v1/jobs/async/job/{job_id}/state` — tool calls, outputs, and source list
- `POST /v1/jobs/async/job/{job_id}/cancel` / `.../resume` — lifecycle control
- `POST /v1/jobs/queue` — durable server-side queue for scheduled/batch research

A ready-made CLI wrapper exists at `.agents/skills/aiq-research/scripts/aiq.py`
(`chat`, `submit`, `research_poll`, `report`, `stream`, `cancel`).

Prefer polling status or using a webhook over holding the SSE stream open;
jobs survive client disconnects and backend restarts (checkpoint auto-resume).
Submit one job per question — duplicate submissions of the same question burn
budget without improving the answer.

## Prompting for the best output

The pipeline rewards specificity. The approved plan's TOC literally becomes
the report's section structure, so the more shape the question carries, the
better the plan.

- **State the real question, not a topic.** "What are the best practices for X,
  and what evidence supports each?" beats "Tell me about X".
- **Name the entities and timeframe** you care about (companies, models,
  papers, "since 2024"). Recency words ("latest", "current") automatically
  trigger time-bounded search.
- **Say who it's for and what shape you want** — audience, length, and whether
  you want a comparison, a how-to, or a state-of-the-art survey.
- **Ask for evidence standards when they matter**: "prefer primary sources /
  peer-reviewed papers / official filings" is honored by the planner's
  source-class requirements.
- **Don't pre-chunk the work.** Submit one well-scoped question rather than
  many fragments; the planner decomposes better than upstream callers can.
- Expect the clarifier to ask at most one question or present a plan for
  approval. Automated callers can approve the proposed plan by replying
  `approve` or pre-empt it by making the original prompt unambiguous.
