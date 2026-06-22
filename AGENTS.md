# Deep Research App Operating Guide

This repository is no longer a plain NVIDIA AI-Q demo. It is Srijan's production
deep-research system: a FastAPI + Dask backend, a Next.js UI, AI-Q/NAT agent
registration, DeepAgents/LangGraph research workflows, local search/scrape
sources, and an Oracle app2 deployment.

Use this file as the first context for any coding agent working here, including
Codex, Claude Code, Antigravity, or a direct shell session.

## What This App Does

- Accepts chat and async research jobs through `frontends/aiq_api`.
- Shows the research process in `frontends/ui` through SSE events, job state,
  artifacts, thoughts, tools, files, and final reports.
- Runs AI-Q agents from `src/aiq_agent`, especially:
  - `clarifier` for intent, plan preview, and HITL approval.
  - `shallow_researcher` for smaller research runs.
  - `deep_researcher` for medium/deeper/deep multi-agent research.
  - `claude_research` as an experimental Claude Code-backed research lane.
- Uses virtual DeepAgents files such as `/shared/plan.json`, `/shared/*.md`,
  and `/report.md`. These are LangGraph state files, not normal disk files.
- Persists scraped/source artifacts through backend job/event storage so the UI
  can recover jobs and reports after reconnects.
- Supports source quality classification, search budgets, report verification,
  Claude Code run folders, queue/batch jobs, and terminal webhooks.

## Important Local Paths

- Project root: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research`
- Backend API: `frontends/aiq_api/src/aiq_api`
- Frontend UI: `frontends/ui`
- Core agents: `src/aiq_agent/agents`
- Shared research utilities: `src/aiq_agent/common`
- Search/scrape source package: `sources/searxng_jina_web_search`
- Active configs: `configs/`
- Oracle sync script: `scripts/sync_app2_deep_research.sh`
- Claude Code runner/tooling: `scripts/run_claude_research.sh`,
  `scripts/claude_research_tool.py`
- Claude research guide: `docs/claude-research-operating-guide.md`

## Oracle App2 Deployment

Oracle app2 is the production-like target the user tests most often.

- Oracle repo path: `/opt/stacks/app2/deep-research`
- Backend health URL on Oracle host: `http://127.0.0.1:9000/health`
- Primary app service names include:
  - `app2-aiq-agent`
  - `app2-aiq-blueprint-ui`
  - `app2-aiq-postgres`
  - `app2-aiq-searxng`
  - `app2-aiq-websurfx`
  - `app2-aiq-websurfx-redis`
  - `app2-aiq-kokoro`

After every code or config change that affects the deep-research app, sync to
Oracle app2 and restart the affected service(s).

- Default to restarting the backend on Oracle after backend, prompt, planner,
  research, source, model-routing, webhook, or middleware changes.
- Restart the frontend when UI code changes.
- If the change touches shared runtime behavior and the impact is unclear,
  restart both backend and frontend.
- Do not leave a changed deployment running on old code just because a job is
  active.
- Oracle sync/restart entrypoint:
  `./scripts/sync_app2_deep_research.sh --apply --restart all`
- The sync script can finish with `curl: (56) Recv failure: Connection reset by
  peer` right after containers start. If that happens, run a health poll before
  assuming failure:
  `ssh oracle 'for i in $(seq 1 40); do curl -fsS http://127.0.0.1:9000/health && exit 0; sleep 3; done; docker logs --tail 160 app2-aiq-agent; exit 1'`

For documentation-only files such as this one, direct-copying to the Oracle repo
is acceptable when no running code changes, but keep GitHub updated too.

## Research Depth And Model Routing

Research tiers are defined in `src/aiq_agent/common/research_depth.py`.
Treat those budgets as product behavior, not incidental constants.

- `shallow`: smallest run, fast, limited source target.
- `medium`: latency-first mode with reduced evidence budget.
- `deeper`: main high-value mode for most serious user research.
- `deep`: largest mode for highest-stakes or very broad research.

Model routing and thinking-mode decisions have changed several times. Before
editing them, inspect the current configs and router code instead of assuming
old behavior. General intent:

- Keep cheap/fast steps small-context and deterministic when possible.
- Use larger-context/stronger reasoning only where it actually improves quality:
  orchestration, final synthesis, verification, large-document reasoning, and
  Claude Code delegation.
- Avoid sending huge contexts to every researcher loop; that previously caused
  slow, repetitive, and failure-prone planning/writing.

## Search And Scrape Stack

The app currently uses self-hosted search and scraping tools, not arbitrary
external search products unless configured.

- SearXNG and DDGS-style search live under `sources/searxng_jina_web_search`.
- Websurfx is available on Oracle and can route selected tiers/searches.
- Scrapling/Playwright/Patchright/curl-cffi are used for web content gathering.
- Search-query quality matters: keep generated queries precise, not giant pasted
  prompts. Prefer several focused full queries over one overloaded query.

When changing search behavior, verify:

- Which engine is actually called in job events.
- Whether scrape artifacts are created.
- Whether citations in the final report trace to usable source records.
- Whether source classification leaves too many important sources as `unknown`.

## DeepAgents Filesystem Rules

DeepAgents file tools write to shared LangGraph state.

- Researchers must not race on the same path.
- Use per-task files for parallel researcher output.
- Orchestrator/planner should merge deterministically.
- Do not assume `write_file` content is real until a subsequent step can read
  the same file from graph state.
- If a tool read returns `[omitted]`, do not let the model spiral into repeated
  rewrites. Use structured tools, smaller artifacts, or deterministic backend
  reads where possible.

## Claude Code Lane

Claude Code is experimental but important. It should be used when it adds real
orchestration value, not for basic search.

Good Claude Code tasks:

- Research-direction memos before large plans.
- Gap analysis after first-pass research.
- Contradiction/source-quality review.
- Long-context synthesis over many artifacts.
- Data extraction or chart/table generation that benefits from code execution.
- Final structured artifact generation when the normal planner/writer is brittle.

Claude Code prompts must specify:

- The exact task boundary.
- The run directory or expected artifact paths.
- The exact output file names and formats.
- Whether web research is allowed.
- Which local tool script to call if search/scrape is needed.
- How the caller will read the result.

Never send a vague "help plan this" prompt and expect the backend to infer what
file to wait for.

## Webhooks

Async API submissions may include terminal webhook fields:

- `webhook_url`
- `webhook_headers`
- `webhook_secret`

The backend sends a best-effort `job.terminal` POST when a job succeeds, fails,
or is interrupted. HMAC signing uses:

- Header: `X-AIQ-Webhook-Timestamp`
- Header: `X-AIQ-Webhook-Signature: sha256=<hex>`
- Message: `timestamp + "." + raw_body`

Webhook failures must not fail the research job.

## Testing Expectations

Before reporting completion:

- Run focused Python tests for backend changes with `uv run --extra dev pytest`.
- Run `uv run ruff check` on touched Python files.
- For frontend changes, run the relevant `npm` checks/build inside
  `frontends/ui` when feasible.
- For deployment-affecting changes, sync to Oracle app2 and verify
  `http://127.0.0.1:9000/health`.
- If changing research workflow behavior, inspect at least one actual job trace
  or run a small smoke job. Unit tests alone do not catch planner loops, broken
  file commits, or missing UI events.

## Git And Release Hygiene

- Keep commits focused and descriptive.
- Push the current branch to GitHub after meaningful changes when the user asks
  for Oracle/GitHub parity.
- Do not commit local secrets, `.env` files, generated bulky run artifacts, or
  stale one-off audit documents unless they describe current product behavior.
- Prefer updating current-state docs over adding many speculative docs.

## High-Risk Areas

Be especially careful around:

- Planner commit/write-plan logic.
- Search budget allocation and exhaustion behavior.
- Model routing/thinking-mode changes.
- DeepAgents shared file writes.
- SSE event naming and UI event recovery.
- Citation/reference integrity.
- Source classification and quality gates.
- Oracle Docker build context and service restarts.

The product goal is not "more agent steps." The goal is fast, auditable,
research-backed output that can support real decisions.
