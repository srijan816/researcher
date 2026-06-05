# NVDA/MiniMax Deep Research Context

This repository is a customized NVIDIA AI-Q / NeMo Agent Toolkit deep research system. It has been adapted into a MiniMax M3 powered research application with a Next.js UI, async research jobs, persisted job/event state, self-hosted SearXNG search, optional Exa and NotebookLM source harvesting, Kokoro narration, local debate transcript search, source artifacts, and app2 deployment on the Oracle VPS.

The local project root is:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
```

The Oracle app2 deployment root is:

```bash
/opt/stacks/app2/deep-research
```

The default SSH target for the Oracle VPS is:

```bash
oracle
```

Do not print secrets, API keys, `.env` contents, tokens, or passwords in logs, chat, screenshots, or docs. Local and server environment files are intentionally excluded from sync.

## What This Repo Is

At a high level, this repo is a deep research agent stack.

It contains:

- A FastAPI backend exposed through the AI-Q API plugin.
- A Next.js frontend UI.
- NVIDIA AI-Q / NAT workflow configuration.
- MiniMax M3 LLM configuration.
- Deep research, shallow research, clarifier, and chat researcher agents.
- Self-hosted SearXNG plus Jina/Scrapling style extraction.
- Optional Exa search.
- Optional NotebookLM source harvesting.
- Local debate transcript search over a prepared corpus.
- Persistent async job/event storage.
- Scrape artifact storage.
- Kokoro ONNX text-to-speech narration.
- Docker Compose deployment for local-ish production and Oracle app2.

The current active research config is:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/configs/config_cli_minimax_ddgs.yml
```

Despite the filename containing `ddgs`, the config currently uses a hybrid search setup:

- `notebooklm_source_harvest_tool` for shallow/experimental NotebookLM source harvesting.
- `exa_web_search_tool` when `EXA_API_KEY` is configured.
- `web_search_tool` backed by SearXNG/Jina/Scrapling-style extraction.
- `advanced_web_search_tool` for larger/deeper SearXNG-based research.
- `stock_quote_tool` for Stooq-backed stock quotes.
- `debate_transcript_search_tool` for local debate transcript retrieval.

## Main Local Commands

Start the local MiniMax Deep Research stack:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/run_minimax_deep_research.sh
```

This script does more than simply start the app:

- Stops existing local backend/frontend processes.
- Installs local editable plugins and extraction dependencies.
- Prepares the debate transcript corpus if source folders exist.
- Ensures Kokoro ONNX model files are present.
- Ensures SearXNG is available.
- Starts the backend and frontend through `./scripts/start_deep_research.sh`.

Stop the local stack:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/stop_deep_research.sh
```

The stop script checks for active async jobs first. If jobs are still running, it refuses to stop unless forced:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
FORCE_STOP_DEEP_RESEARCH=1 ./scripts/stop_deep_research.sh
```

Use force only when you are sure it is okay to interrupt running research.

Default local URLs:

```text
Backend:  http://localhost:9000
Frontend: http://localhost:3000
SearXNG:  http://localhost:8080
```

The local start script binds the backend to `0.0.0.0`, so it can also print a LAN URL for phone/tablet testing.

## Important Local Runtime Paths

Local runtime logs and PID files:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/.deep-research-runtime
```

Local job and checkpoint files may exist in the repo root:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/jobs.db
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/checkpoints.db
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/checkpoints.db-wal
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/checkpoints.db-shm
```

These are runtime state, not source code. They can become large and are excluded from the app2 sync script. Do not commit them.

Local scrape artifacts live under:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/data/scrape_artifacts
```

Scrape artifacts are compressed per-job source captures. They are useful for auditing what was fetched and why a report cited a source.

Kokoro model assets live locally under:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/data/kokoro-models
```

## Oracle app2 Deployment

The Oracle VPS app2 deployment uses this compose file:

```bash
/opt/stacks/app2/deep-research/deploy/compose/docker-compose.app2.yaml
```

The local source copy of that file is:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/deploy/compose/docker-compose.app2.yaml
```

App2 containers use app2-specific names and volumes so they do not disturb app1:

```text
app2-aiq-agent
app2-aiq-blueprint-ui
app2-aiq-kokoro
app2-aiq-postgres
app2-aiq-searxng
```

App2 host port bindings:

```text
Backend:  127.0.0.1:9000 -> aiq-agent:8000
Frontend: 127.0.0.1:3110 -> frontend:3000
Kokoro:   127.0.0.1:3111 -> kokoro:8000
Postgres: private Docker network only
SearXNG:  private Docker network only
```

App2 Docker volumes:

```text
app2_deep_research_aiq_data
app2_deep_research_postgres_data
app2_deep_research_searxng_cache
app2_deep_research_kokoro_data
```

These volumes preserve:

- Async jobs.
- Event history.
- Checkpoints.
- User/auth state.
- Scrape artifacts and data written under `/app/data`.
- SearXNG cache.
- Kokoro model files.

Never run `docker compose down -v` on app2 unless you explicitly want to erase persisted state.

## Syncing Local Source to Oracle app2

The sync script is:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/scripts/sync_app2_deep_research.sh
```

Operational rule: after any code/config change that should affect app2, sync to Oracle and restart the affected app2 service immediately. If the change is backend, restart backend; if UI, restart frontend; if the impact spans both, restart `all`. Do not skip the restart just because a job is currently running.

Dry run:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/sync_app2_deep_research.sh
```

Apply source sync without restart:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/sync_app2_deep_research.sh --apply
```

Apply and rebuild/restart only the frontend:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/sync_app2_deep_research.sh --apply --restart frontend
```

Apply and rebuild/restart only the backend:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/sync_app2_deep_research.sh --apply --restart backend
```

Apply and rebuild/restart backend, frontend, Kokoro, and SearXNG:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/sync_app2_deep_research.sh --apply --restart all
```

The sync script defaults to:

```text
APP2_REMOTE_HOST=oracle
APP2_REMOTE_ROOT=/opt/stacks/app2/deep-research
```

You can override those:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
APP2_REMOTE_HOST=oracle APP2_REMOTE_ROOT=/opt/stacks/app2/deep-research ./scripts/sync_app2_deep_research.sh --apply --restart frontend
```

The sync script intentionally excludes:

- `.git/`
- `.venv/`
- `node_modules/`
- `.next/`
- `.deep-research-runtime/`
- `.pytest_cache/`
- `.ruff_cache/`
- `.tmp/`
- `data/`
- `.env`
- `.env.*`
- `deploy/.env`
- `*.db`
- `*.sqlite`
- `*.log`
- `__pycache__/`
- recovered local report files

That means secrets and local runtime state are not copied by normal sync.

## Server-Side Environment Files

Local environment files:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/deploy/.env
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/.env
```

App2 server environment file:

```bash
/opt/stacks/app2/deep-research/deploy/.env
```

Do not print these files. They contain keys, auth settings, tokens, passwords, and production settings.

If app2 is missing a capability, inspect the key names only, not values:

```bash
ssh oracle "cd /opt/stacks/app2/deep-research && grep -E '^[A-Z0-9_]+=' deploy/.env | cut -d= -f1 | sort"
```

## Health Checks

Local backend:

```bash
curl -fsS http://127.0.0.1:9000/health
```

Local frontend:

```bash
curl -fsSI http://127.0.0.1:3000/
```

App2 backend from the VPS:

```bash
ssh oracle "curl -fsS http://127.0.0.1:9000/health"
```

App2 frontend from the VPS:

```bash
ssh oracle "curl -fsSI http://127.0.0.1:3110/ | sed -n '1,8p'"
```

App2 Kokoro:

```bash
ssh oracle "curl -fsS http://127.0.0.1:3111/health"
```

App2 containers:

```bash
ssh oracle "docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' | grep app2-aiq"
```

## Public app2 Access

The app2 public site is normally routed through nginx to the frontend bound at `127.0.0.1:3110`.

The public hostname used in prior work is:

```text
https://app2.sniperip.com
```

The backend remains bound to localhost on the VPS. App1 can call it from the same machine/container network through the configured deep research base URL.

Expected backend base URL for app1 integration depends on app1 Docker networking:

- If app1 can reach the host gateway: `http://host.docker.internal:9000`
- If app1 is on the same Docker network as app2 backend: `http://app2-aiq-agent:8000`
- If app1 calls host localhost from the VPS host process: `http://127.0.0.1:9000`

The target variable in app1 is:

```text
DEEP_RESEARCH_BASE_URL
```

## API Surface

Main async job endpoints:

```text
GET  /health
GET  /v1/data_sources
GET  /v1/jobs/async/agents
GET  /v1/jobs/async/jobs?limit=50
POST /v1/jobs/async/submit
GET  /v1/jobs/async/job/{job_id}
GET  /v1/jobs/async/job/{job_id}/stream
GET  /v1/jobs/async/job/{job_id}/stream/{last_event_id}
POST /v1/jobs/async/job/{job_id}/cancel
POST /v1/jobs/async/job/{job_id}/resume
GET  /v1/jobs/async/job/{job_id}/state
GET  /v1/jobs/async/job/{job_id}/report
GET  /v1/jobs/async/job/{job_id}/report?format=md
```

Submit a job locally:

```bash
curl -X POST http://127.0.0.1:9000/v1/jobs/async/submit \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "deep_researcher",
    "input": "Research the best ways to live well in the age of AI.",
    "research_depth": "deep"
  }'
```

Poll status:

```bash
curl http://127.0.0.1:9000/v1/jobs/async/job/{job_id}
```

Fetch state/artifacts:

```bash
curl http://127.0.0.1:9000/v1/jobs/async/job/{job_id}/state
```

Fetch final report as JSON:

```bash
curl http://127.0.0.1:9000/v1/jobs/async/job/{job_id}/report
```

Fetch final report as markdown:

```bash
curl http://127.0.0.1:9000/v1/jobs/async/job/{job_id}/report?format=md
```

If the report is not ready, the report endpoint returns progress fields such as `has_report`, `report_ready`, `terminal`, `status`, `message`, and URLs for status/state/report/stream.

## Authentication and Users

The backend supports local user authentication when enabled by environment:

```text
REQUIRE_AUTH
AIQ_LOCAL_AUTH_ENABLED
AIQ_AUTH_TOKEN_SECRET
AIQ_AUTH_DB_URL
AIQ_LOCAL_USERS
AIQ_LOCAL_ADMIN_USERS
AIQ_AUTH_TOKEN_TTL_SECONDS
```

The local auth route code lives under:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/aiq_api/src/aiq_api/routes/auth.py
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/aiq_api/src/aiq_api/auth
```

The configured user set from prior app work is:

```text
srijan
mai
jami
naveen
saurav
sreyan
```

`srijan` is the administrator user. Admin access is intended to see all research history; other users should see only their own research. Lesson-plan-triggered research is associated with `srijan`.

The frontend also has local session/history behavior, but the backend now has database-backed conversation/job routes so history can be available across devices rather than only browser localStorage.

## UI Capabilities

The frontend lives at:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui
```

The UI currently supports:

- Chat-style research interaction.
- Clarifier questions.
- Plan approval/rejection/revision.
- Deep research progress streaming.
- Research panel with thinking, tools, tasks, sources, and report tabs.
- Async job restoration from backend state.
- Server-backed job/history access.
- Data source selection, including web search and optional local debate transcripts.
- Kokoro report narration controls.
- Copy affordances for prompts/reports.
- Dark-only theme.
- A `srijan`-only cinematic homepage treatment and active-research waiting animation.

The Next.js frontend proxies backend calls through `/api/*` routes, so browser clients usually talk to the frontend, and the frontend talks to the backend using `BACKEND_URL`.

Key frontend proxy files:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/app/api/jobs/async/[...path]/route.ts
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/app/api/v1/[...path]/route.ts
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/app/api/chat/route.ts
```

## Research Agent Architecture

Core agent folders:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/chat_researcher
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/clarifier
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/shallow_researcher
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher
```

The broad flow is:

1. User enters a research query in the UI or calls the async API.
2. Intent classifier decides whether this is normal chat/meta or research.
3. Clarifier may ask follow-up questions if the scope is too ambiguous.
4. Clarifier generates a user-facing plan preview.
5. User approves, rejects, or revises the plan.
6. Shallow or deep researcher runs, depending on selected/intended depth.
7. Planner creates tasks/queries/sections.
8. Researcher agents call search/extraction/local tools.
9. Source artifacts and tool events are persisted.
10. Orchestrator synthesizes findings into `/report.md`.
11. Quality gates check report existence, citations, source spread, claim table signals, and failure state.
12. UI and API expose status, state, report, sources, and artifacts.
13. Optional Kokoro narration can generate audio from the report.

Deep research uses a multi-role setup:

- `planner_llm`: MiniMax M3 planner profile.
- `researcher_llm`: MiniMax M3 general research profile.
- `orchestrator_llm`: MiniMax M3 synthesis profile with larger output budget.

## Search and Source Capabilities

The source registry in `configs/config_cli_minimax_ddgs.yml` declares two source groups:

```text
web_search
debate_transcripts
```

`web_search` includes:

- NotebookLM source harvest.
- Exa web search.
- SearXNG-backed web search.
- Advanced SearXNG-backed web search.
- Stooq stock quote.

`debate_transcripts` includes:

- Local debate transcript search.

The local debate transcript tool is disabled by default in the data-source registry. Users can enable it from the frontend when they want local transcript retrieval.

The debate transcript corpus is prepared by:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/scripts/prepare_debate_transcript_corpus.py
```

Default source transcript folders:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data/notebooklm_exports/debate_transcripts
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data/youtube_recent_transcripts_2022_2026
```

Default prepared corpus output:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data/prepared_debate_corpus
```

The app2 compose file mounts the repo-local prepared corpus at:

```text
/app/data/prepared_debate_corpus
```

The app2 backend uses:

```text
DEBATE_TRANSCRIPT_CORPUS_JSONL=/app/data/prepared_debate_corpus/chunks/search_chunks.jsonl
```

If new transcript source files are added locally, rerun the prep script or `./scripts/run_minimax_deep_research.sh`, then sync the prepared corpus/source changes to app2.

## NotebookLM Status

NotebookLM source harvesting is wired as an experimental source path, mainly for shallow-research testing. The config currently allows NotebookLM to run in `deep` mode with multiple queries and source harvesting limits.

Important NotebookLM settings in the config:

```text
query_count: 3
timeout_seconds: 3600
max_sources: 36
max_fulltext_sources: 24
include_generated_reports: false
delete_notebook_after: false
```

NotebookLM is intended to contribute sources, not replace the system's own researcher. The preferred direction is parallel contribution: NotebookLM harvests sources while the normal researcher independently searches and reasons, then the orchestrator combines the evidence.

## Kokoro Narration

Kokoro exists in two forms:

1. Local Next.js route fallback under:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/app/api/tts/kokoro/route.ts
```

2. App2 dedicated Docker service:

```text
app2-aiq-kokoro
```

App2 frontend points at Kokoro with:

```text
KOKORO_TTS_URL=http://kokoro:8000
```

App2 host health check:

```bash
ssh oracle "curl -fsS http://127.0.0.1:3111/health"
```

The intended narration behavior is:

- Use Onyx-style/default voice where configured in UI.
- Chunk long reports on sentence boundaries, not mid-word.
- Generate/report audio continuously for long reports.
- Keep playback controls usable on desktop and mobile.

## Quality and Reliability Work Already Present

The repo includes or references several quality improvements:

- Durable scrape artifacts under `data/scrape_artifacts`.
- Source classification attached to events/artifacts.
- Source-quality checks in the async job runner.
- Claim-table summarization hooks.
- Stronger prompts for clarifier, planner, researcher, and orchestrator.
- Report recovery path when artifacts exist but `/report.md` is missing.
- Ghost job reaper for stale async jobs.
- Resume endpoint for failed/interrupted jobs.
- Stop script guard that avoids killing active local jobs accidentally.
- API report polling fields so clients know whether a report is ready, failed, or still running.

Useful audit script:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
python scripts/audit_deep_research_job.py --help
```

## Development Checks

Frontend checks:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui
npm run type-check -- --pretty false
npm run lint -- --quiet
npm run test
```

Backend/Python checks depend on the active environment, but common commands are:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
uv run pytest
uv run ruff check .
```

Because this repo often has active local changes and large runtime files, check status before broad edits:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
git status --short
```

Do not revert unrelated dirty files unless explicitly instructed.

## Storage and Bloat Notes

Local and server runtime state can grow quickly.

Common large local files:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/checkpoints.db
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/checkpoints.db-wal
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/jobs.db
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/data/scrape_artifacts
```

Common app2 storage consumers:

- Docker build cache.
- Old Docker images.
- `app2_deep_research_postgres_data`.
- `app2_deep_research_aiq_data`.
- SearXNG cache.
- Kokoro model volume.
- Source tree copies under `/opt/stacks/app2/deep-research`.

Safe inspection commands:

```bash
ssh oracle "df -h"
ssh oracle "docker system df"
ssh oracle "sudo du -xh /opt/stacks/app2/deep-research 2>/dev/null | sort -h | tail -30"
ssh oracle "docker volume ls | grep app2_deep_research"
```

Do not delete app2 Docker volumes unless the user explicitly approves data loss.

## App1 Relationship

App1 is separate and must not be disturbed when deploying app2.

App1 uses app2's backend as a deep research service through:

```text
DEEP_RESEARCH_BASE_URL
```

Expected endpoints used by app1:

```text
GET  /health
POST /v1/jobs/async/submit
GET  /v1/jobs/async/job/{job_id}
GET  /v1/jobs/async/job/{job_id}/report
GET  /v1/jobs/async/job/{job_id}/state
```

When wiring app1, restart only app1 web if needed. Do not restart app1 database or unrelated services.

## Practical Mental Model

Local machine:

- Best for code edits, prompt edits, UI development, quick testing, transcript corpus preparation, and local debugging.
- Uses local `.env` and local runtime files.
- Can run backend/frontend directly through scripts.

Oracle app2:

- Best for persistent hosted usage.
- Runs Docker Compose services.
- Keeps persistent jobs, checkpoints, auth, scrape artifacts, SearXNG cache, and Kokoro models in Docker volumes.
- Should receive source updates through `scripts/sync_app2_deep_research.sh`.
- Should not receive local `.env`, local DBs, local caches, or local runtime folders through rsync.

App1:

- Separate application.
- Calls app2 deep research API.
- Should not be touched unless specifically wiring or restarting app1 integration.

## Most Important Rules

1. Do not expose secrets.
2. Do not overwrite app1.
3. Do not run `docker compose down -v` on app2 unless data loss is intended.
4. Use `./scripts/sync_app2_deep_research.sh` for source sync to Oracle.
5. Use `--restart frontend`, `--restart backend`, or `--restart all` deliberately.
6. Check active jobs before stopping local services.
7. Treat `data/`, `jobs.db`, `checkpoints.db`, `.deep-research-runtime/`, `.venv/`, `.next/`, and `node_modules/` as runtime/build state, not normal source sync payload.
8. When debugging report quality, inspect `/v1/jobs/async/job/{job_id}/state`, persisted events, scrape artifacts, and source-class/citation metrics before changing prompts.
9. Keep broader research quality fixes architectural where possible: source quality, claim grounding, budget allocation, planning discipline, synthesis verification, and resumability beat narrow one-off prompt patches.
