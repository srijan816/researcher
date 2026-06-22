# Deep Research App Context

This file is the high-level operating context for agents working in this
repository. Read it before editing code, prompts, configs, deployment files, or
research workflow behavior.

The project started from NVIDIA AI-Q / NeMo Agent Toolkit examples, but it has
evolved into Srijan's production deep-research system. It combines:

- A FastAPI backend in `frontends/aiq_api`.
- A Next.js UI in `frontends/ui`.
- NVIDIA AI-Q / NAT workflow registration.
- DeepAgents / LangGraph research workflows.
- MiniMax M3 through the Anthropic-compatible API.
- Self-hosted search through SearXNG, Websurfx, and DDGS fallback paths.
- Scrapling / Playwright / Patchright / curl-cffi page extraction.
- Persistent async jobs, event streams, scrape artifacts, and reports.
- Optional Claude Code-backed research lanes.
- Oracle app2 Docker Compose deployment.

The product goal is not to maximize agent steps. The goal is fast, auditable,
source-grounded research output that can support decisions, content creation,
meeting/interview preparation, debate lesson construction, and enterprise-style
research-backed workflows.

Do not print secrets, API keys, `.env` contents, tokens, cookies, passwords, or
private auth material in chat, docs, logs, screenshots, or commits.

## Canonical Paths

Local project root:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
```

Oracle app2 deployment root:

```bash
/opt/stacks/app2/deep-research
```

Default Oracle SSH target:

```bash
oracle
```

Primary runtime config:

```bash
configs/config_cli_minimax_ddgs.yml
```

Primary local/Oracle sync script:

```bash
scripts/sync_app2_deep_research.sh
```

## Top-Level Code Map

Backend API:

```bash
frontends/aiq_api/src/aiq_api
```

Important backend areas:

- `routes/jobs.py`: async job routes, SSE streaming, job state/report endpoints,
  resume/recovery behavior, queue-visible job metadata.
- `jobs/submit.py`: job submission/resume glue.
- `jobs/runner.py`: Dask job execution, post-run quality checks, report
  recovery, artifact emission, terminal webhook dispatch.
- `jobs/event_store.py`: persisted SSE/job event store. PostgreSQL uses
  LISTEN/NOTIFY, SQLite uses polling.
- `jobs/callbacks.py`: maps LangChain/agent callbacks into UI-visible events.
- `jobs/webhooks.py`: best-effort terminal webhook delivery.
- `auth/` and `routes/auth.py`: local auth/session/user handling.

Frontend UI:

```bash
frontends/ui
```

Important frontend areas:

- `src/features/chat/store.ts`: main persisted chat/research state, message
  handling, batch queue state, restoration behavior.
- `src/features/chat/hooks/use-deep-research.ts`: async research SSE handling,
  report/artifact/thought/tool/citation integration.
- `src/features/chat/hooks/use-batch-research-queue.ts`: client-side queued
  research runner; approved queued items submit one after another.
- `src/features/layout/components/InputArea.tsx`: prompt box, depth selector,
  engine selector, queue button, source controls.
- `src/features/layout/components/ResearchPanel.tsx`: plan, batch, task,
  thinking, citation, file, and report display.
- `src/features/layout/components/SessionsPanel.tsx`: history/session UI.
- `src/app/api/*`: Next.js proxy routes from browser to backend.

Core AI agents:

```bash
src/aiq_agent/agents
```

Important agent folders:

- `clarifier`: intent, clarification, and user-facing plan preview.
- `shallow_researcher`: small bounded research runs.
- `deep_researcher`: medium/deeper/deep multi-agent research workflow.
- `claude_research`: experimental Claude Code-backed lane.
- `chat_researcher`: ordinary chat/research routing support.

Shared agent utilities:

```bash
src/aiq_agent/common
```

Important utilities:

- `research_depth.py`: source targets, max task counts, parallel lanes, and
  search budgets for shallow/medium/deeper/deep.
- `source_classification.py`: URL/source class taxonomy.
- `source_quality_gates.py`: source distribution/diversity quality checks.
- `scrape_artifacts.py`: source capture persistence.
- `citation_verification.py`: citation/source registry helpers.

Search/scrape package:

```bash
sources/searxng_jina_web_search
```

This package provides the main web search function. Despite the historical name,
it now supports SearXNG, DDGS, Websurfx, hybrid discovery, and Scrapling-based
page extraction.

Claude Code tooling:

```bash
scripts/run_claude_research.sh
scripts/claude_research_tool.py
docs/claude-research-operating-guide.md
```

## The Research Flow

The normal AIQ research path works like this:

1. User submits a prompt from the UI or the async API.
2. The frontend sends the request through a Next.js proxy route to the FastAPI
   backend.
3. Backend creates an async job, persists a `job.submitted` event, and dispatches
   work through the configured AI-Q / Dask runner.
4. Clarifier/intent logic decides whether this is chat, shallow research, or a
   deep-research job.
5. For UI-originated research, a user-facing plan may be generated and approved,
   revised, rejected, or auto-approved depending on the path.
6. Deep research builds a plan with task decomposition, source strategy, budget
   profile, and researcher task assignments.
7. Researcher agents run search/scrape/local tools. They should use focused
   queries, not pasted full prompts.
8. Tool results, thoughts, files, citations, scrape artifacts, and source records
   are emitted into the event store and shown in the UI via SSE.
9. Orchestrator/synthesizer writes the final report, usually as `/report.md` in
   DeepAgents virtual state.
10. Backend post-run quality gates check report existence, citations, source
    spread, source quality, and recoverable report artifacts.
11. Final report is exposed through `/report`, persisted in job output, and shown
    in the UI report tab.
12. Optional terminal webhooks notify API clients when the job succeeds, fails, or
    is interrupted.

The UI can also queue multiple research tasks. Queue entries are stored in the
chat store, auto-approved after a grace period, and submitted sequentially. The
queue is a frontend orchestration feature; the backend still sees ordinary async
jobs.

## Research Tiers

Research tiers live in:

```bash
src/aiq_agent/common/research_depth.py
```

Current tiers:

- `shallow`: smallest run, quick answer, limited sources.
- `medium`: latency-first mode with a reduced evidence budget.
- `deeper`: default serious-research mode; higher source target and up to four
  parallel researcher tasks.
- `deep`: largest mode for broad or high-stakes research.

Treat these budgets as product behavior, not incidental constants. Changing
them affects speed, cost, source quality, UI expectations, and report length.

The depth config controls:

- Source target range.
- Maximum researcher tasks.
- Maximum parallel researcher tasks.
- Search calls per researcher task.
- Planner search allowance.
- Advanced/web/stock search limits.
- Nominal versus adaptive reserve budget.
- Planner guidance injected into prompts.

## Model Routing

The active config is MiniMax M3-first:

```bash
configs/config_cli_minimax_ddgs.yml
```

Important intent:

- Planner and researcher loops generally use M3 with thinking disabled for speed
  and tool-call reliability.
- Synthesis can use a larger-context M3 profile. Some synthesis profiles have
  thinking enabled, but this must be used carefully because thinking can increase
  latency and repeated-output failure modes.
- Deeper-specific planner/research profiles exist and are currently M3
  fast/no-thinking profiles unless environment overrides change the model names.
- Avoid sending huge contexts to every researcher loop. Use large context where
  it matters: final synthesis, verification, long-document reasoning, and Claude
  Code delegation.

Before changing model routing, inspect both:

```bash
configs/config_cli_minimax_ddgs.yml
src/aiq_agent/agents/deep_researcher
```

Do not assume older M2.7 behavior is still active. Environment variables may
override model names.

## Search And Scrape Stack

The main configured source group is `web_search`.

Current important tools:

- `exa_web_search_tool`: optional; works only when `EXA_API_KEY` is configured.
- `web_search_tool`: SearXNG/DDGS hybrid discovery with Scrapling extraction.
- `advanced_web_search_tool`: larger search tool; on Oracle app2 it can use
  `websurfx_hybrid` through `AIQ_ADVANCED_DISCOVERY_BACKEND`.
- `stock_quote_tool`: Stooq-backed quotes.
- `debate_transcript_search_tool`: local prepared debate transcript corpus;
  disabled by default in the data-source registry.
- `local_corpus_search_tool`: embedding/BM25 search over previously scraped
  pages (see Local Research Corpus below). Agents should consult it before
  spending web budget on topics likely covered by prior runs.

Search results are reranked before scraping. Ranking layers:

- Heuristic ranking (term match, domain authority, text signals, freshness).
- Optional cross-encoder reranking of the top candidates through NVIDIA NIM
  (`AIQ_RERANK_BACKEND=nvidia` with `NVIDIA_API_KEY`, default model
  `nvidia/llama-nemotron-rerank-1b-v2`) or a local model
  (`AIQ_RERANK_BACKEND=local`). Reranking fails open to heuristic order.

Recency handling: queries with recency intent ("latest", "news", explicit
recent years) automatically set SearXNG `time_range`/DDGS `timelimit`;
best-effort `published_at` extraction feeds a freshness score and is emitted
on documents and scrape artifacts.

Extraction quality: every document carries `<extraction_quality>`
(`full`/`partial`/`snippet_only`/`metadata_only`) plus the explicit fallback
chain, so weak extracts are visible downstream instead of silently identical
to full reads.

Scrape cache: durable scrape artifacts are indexed in
`<artifact_root>/artifacts_index.db` and consulted before fetching
(`AIQ_SCRAPE_CACHE_ENABLED`, TTL `AIQ_SCRAPE_CACHE_TTL_SECONDS`, shorter
`AIQ_SCRAPE_CACHE_RECENT_TTL_SECONDS` for recency queries). Cache hits are
marked `<cached>true</cached>` and still recorded as artifacts for the
current job. Discovery and scraping are rate-limited per backend
(`AIQ_SEARCH_DISCOVERY_CONCURRENCY`, `AIQ_SCRAPE_CONCURRENCY`) to protect the
self-hosted SearXNG/Websurfx containers.

Oracle app2 has:

- SearXNG container: `app2-aiq-searxng`
- Websurfx container: `app2-aiq-websurfx`
- Websurfx Redis: `app2-aiq-websurfx-redis`

App2 compose defaults include:

```text
SEARXNG_URL=http://searxng:8080
WEBSURFX_URL=http://websurfx:8080
AIQ_ADVANCED_DISCOVERY_BACKEND=websurfx_hybrid
WEBSURFX_ENGINES=Searx,Brave,DuckDuckGo,LibreX,Mojeek,Qwant,Startpage,Yahoo,Bing,SepiaSearch
```

The search function can merge results from multiple discovery lanes and then
extract pages with Scrapling. Search failures often come from overloaded query
strings, upstream search blocks, or extraction timeouts. Fix those by improving
query generation, timeout handling, discovery backend routing, and artifact
recovery, not by adding narrow one-off prompt patches.

## DeepAgents Virtual Files

DeepAgents file tools write to LangGraph state, not normal disk.

Common virtual paths:

```text
/shared/plan.json
/shared/*.md
/shared/*.json
/report.md
```

Rules:

- Parallel researchers must not write the same shared file path.
- Use per-task files for parallel researcher outputs.
- Merge deterministically in Python or orchestrator glue.
- Never trust that a write succeeded until a later turn/tool can read the same
  virtual file from graph state.
- If a read returns `[omitted]`, do not let the model repeatedly rewrite the
  same file. Use smaller files, structured tools, deterministic backend reads, or
  file-specific artifacts.

## Event Store And SSE

The UI depends on durable job events. The event store lives in:

```bash
frontends/aiq_api/src/aiq_api/jobs/event_store.py
```

The SSE stream lives in:

```bash
frontends/aiq_api/src/aiq_api/routes/jobs.py
```

PostgreSQL deployments use `LISTEN/NOTIFY`. Channel names must remain under the
PostgreSQL 63-byte identifier limit, so code must use `get_job_events_channel()`
rather than building a channel directly from the full job id.

Important reliability rule:

- Persist events first.
- Commit them.
- Then notify SSE listeners.

This prevents a notification failure from rolling back event history. If event
history disappears, the UI may show a fallback title such as a shortened job id
and fetch the report directly from `job_info.output`, making the run look stale
or detached from the query.

When debugging event/UI issues, check:

```bash
ssh oracle "docker logs --tail 200 app2-aiq-agent"
ssh oracle "docker exec app2-aiq-postgres psql -U aiq -d aiq_jobs -c 'select job_id,status,created_at,updated_at,length(output) from job_info order by updated_at desc limit 10;'"
ssh oracle "docker exec app2-aiq-postgres psql -U aiq -d aiq_jobs -c 'select job_id,count(*) from job_events group by job_id order by max(id) desc limit 10;'"
```

## Persistence Model

Backend state can live in SQLite locally or PostgreSQL on Oracle.

PostgreSQL tables on app2 include:

- `job_info`: job status, output, timestamps.
- `job_events`: persisted SSE/job events.
- `job_access`: ownership and auth visibility.
- `local_auth_users`: local users.
- `api_keys`: API key state.
- `ui_conversations`: server-backed UI conversation metadata.
- `summaries`: generated summaries.

Scrape/source artifacts live under the app data volume and local `data/` tree.
They are useful for auditing citation quality.

Do not delete Docker volumes or local DBs unless data loss is intended.

## Source Quality And Citation Integrity

The system has infrastructure for:

- Source classification.
- Source quality gates.
- Citation/source artifacts.
- Post-run report checks.
- Report recovery from artifacts.

Important files:

```bash
src/aiq_agent/common/source_classification.py
src/aiq_agent/common/source_quality_gates.py
src/aiq_agent/common/citation_verification.py
src/aiq_agent/common/scrape_artifacts.py
frontends/aiq_api/src/aiq_api/jobs/runner.py
```

Quality problems to watch:

- Too many sources classified as `unknown`.
- Citation references duplicated or pointing nowhere.
- Report claims with numbers/dates but weak sourcing.
- Sources cited for claims they do not support.
- Stale funding/product/model data.
- Inverted findings where the report states the opposite of the source.
- Broken `/report.md` or report text that includes "I will now..." thought
  traces.

The long-term fix is structured claim/evidence artifacts, adversarial
verification, clean reference generation, and post-run source checks. Avoid
papering over factual issues with one-off query-specific prompt rules.

Adversarial claim verification now runs as a deterministic backend step after
the research dossier is compiled and before final synthesis
(`src/aiq_agent/agents/deep_researcher/adversarial_verifier.py`). High-risk
claims (numeric/date/superlative/causal) are checked one-by-one against their
stored evidence extracts by the verifier LLM (`verifier_llm` in the config,
profile `minimax_m3_verifier_llm`). Results land in
`/shared/verification_report.json`; contradicted claims are flagged in the
claim table and must be dropped or hedged in synthesis. Controls:
`AIQ_ADVERSARIAL_VERIFIER_ENABLED`, `AIQ_VERIFIER_MAX_CLAIMS`,
`AIQ_VERIFIER_CONCURRENCY`. The UI shows a verification stats strip above the
report when the file is present.

Gap-filling is a bounded loop: the progress snapshot reports rounds used vs
allowed (shallow 0 / medium 1 / deeper 2 / deep 3, override
`AIQ_GAPFILL_MAX_ROUNDS`) and remaining reserve; gap-fill spends only the
adaptive reserve.

Citation matching is fuzzy: trivial URL variants (scheme, www/m/amp,
trailing slash, fragments, tracking params) no longer get valid `[N]`
citations deleted; real ambiguity is still rejected.

A persistent cross-run fact ledger (`src/aiq_agent/common/fact_ledger_store.py`,
SQLite at `AIQ_FACT_LEDGER_DB`, kill switch `AIQ_FACT_LEDGER_ENABLED`) stores
verified entity facts on job completion and primes the planner with
"previously verified facts" matching the query, labeled for re-verification.

## Local Research Corpus

Every scraped page persists as an artifact; the corpus index makes them
searchable across jobs:

```bash
sources/local_corpus_search/          # NAT tool package
scripts/build_corpus_index.py         # incremental ingest CLI (cron-able)
```

Embeddings via NVIDIA NIM (`NVIDIA_API_KEY`,
`AIQ_CORPUS_EMBED_MODEL=nvidia/llama-nemotron-embed-1b-v2`) with a pure-python
BM25 fallback when no key is set. Index lives at `AIQ_CORPUS_DB`
(default `./data/corpus_index.db`). Rebuild with `--rebuild` when switching
modes; use `--backfill-embeddings` to fill in vectors for chunks stored while
the embedding endpoint was unavailable (e.g. after a NIM model EOL). The
`--backfill-embeddings` pass is resilient: a transient embedding failure skips
only that batch and the run continues, so it self-completes across re-runs.
The `local_corpus_search_tool` is registered in the `web_search`
data-source group and returns web-search-style `<document>` blocks with
real source URLs so citations stay valid.

**Vector index (zvec).** SQLite stays the durable content/metadata store; a
derived [zvec](https://github.com/alibaba/zvec) HNSW index
(`./data/corpus_index.zvec/`, alongside the DB) serves dense retrieval ~80x
faster than the previous in-memory full scan and without the per-query
multi-hundred-MB matrix load (measured: ~1000ms → ~12ms/query at 80k chunks,
recall@10 100%). `scripts/build_corpus_index.py` syncs it incrementally after
every ingest/backfill (`--no-vector-index` to skip; `vector_index_state` tracks
synced chunk ids). The index is a rebuildable cache: `core.search` prefers zvec
(dense ANN + scalar filter) and falls back to the SQLite numpy/BM25 path
whenever zvec is unavailable or a query errors, so behaviour degrades gracefully.
`zvec` is declared in the package deps and installed in the Dockerfile; it is
scoped to `local_corpus_search` only.

## Evaluation Harness

```bash
scripts/eval/questions.yaml   # 18 rubric-scored benchmark questions
scripts/eval/run_eval.py      # submit + poll + collect reports via async API
scripts/eval/judge.py         # MiniMax LLM judge -> scorecard.md/json, --compare
```

Run a subset before/after behavior changes and compare scorecards
(citation support, factual accuracy, comprehensiveness, recency,
cleanliness, efficiency). Do not change budgets, prompts, model routing, or
search behavior without an eval comparison when feasible.

## Claude Code Lane

The Claude Code lane is experimental but important. It is intended for tasks
where autonomous file/artifact discipline is better than ordinary agent loops.

Entry points:

```bash
src/aiq_agent/agents/claude_research
scripts/run_claude_research.sh
scripts/claude_research_tool.py
docs/claude-research-operating-guide.md
```

Claude Code can be routed through MiniMax M3 by setting Anthropic-compatible
environment variables. The runner uses `MINIMAX_API_KEY` as the Anthropic auth
token when configured for MiniMax.

Good Claude Code tasks:

- High-quality research-direction memo before a large plan.
- Gap analysis after first-pass research.
- Contradiction/source-quality review.
- Long-context synthesis over many artifacts.
- Structured run-folder generation.
- Data extraction, tables, timelines, chart generation, or code-assisted
  analysis.
- Final artifact generation when ordinary planner/writer loops are brittle.

Bad Claude Code tasks:

- Basic web search.
- One-page summarization.
- Simple formatting.
- Real-time user conversation.

Claude prompts must always specify:

- Exact task boundary.
- Whether research/search is allowed.
- Run directory.
- Required output files and formats.
- Tool script to call if search/scrape is needed.
- Completion signal the caller should wait for.

The run-folder contract should be explicit:

```text
runs/<run-id>/
  plan.md
  queries.json
  sources.json
  notes/
  source_summaries/
  contradictions.md
  gaps.md
  research.md
  final.md
```

If the UI claims Claude Code was used, verify actual artifacts and events, not
just labels.

## Batch Queue

The batch queue is currently frontend-driven.

Important files:

```bash
frontends/ui/src/features/chat/hooks/use-batch-research-queue.ts
frontends/ui/src/features/chat/store.ts
frontends/ui/src/features/layout/components/BatchResearchQueue.tsx
frontends/ui/src/features/layout/components/InputArea.tsx
```

Behavior:

- User queues research tasks from the input box.
- Queue entries persist in the chat store.
- User can approve queued items manually.
- Items auto-approve after a grace period.
- Approved items submit one at a time.
- `researchEngine === "claude_code"` submits `claude_researcher`; otherwise it
  submits `deep_researcher`.

This is not a server-side queue yet. Refresh/persistence depends on frontend
state behavior.

A durable server-side queue also exists (`/v1/jobs/queue`, table
`research_queue`, scheduler in
`frontends/aiq_api/src/aiq_api/jobs/queue_scheduler.py`). It supports
priorities, `scheduled_for` timestamps, and `recurrence_seconds` for
recurring research (e.g. weekly monitoring briefs); recurring items
re-enqueue themselves after each run. Controls: `AIQ_QUEUE_ENABLED`,
`AIQ_QUEUE_POLL_SECONDS`, `AIQ_QUEUE_MAX_CONCURRENT`. The frontend batch
queue has not yet been migrated onto it.

## Webhooks

Async submit payloads may include:

```json
{
  "webhook_url": "https://example.com/research-complete",
  "webhook_headers": {"Authorization": "Bearer ..."},
  "webhook_secret": "optional secret"
}
```

Terminal webhooks are best-effort. They must not fail the research job.

Signing headers:

```text
X-AIQ-Webhook-Timestamp
X-AIQ-Webhook-Signature: sha256=<hex>
```

Signing message:

```text
timestamp + "." + raw_body
```

## Local Development

Start local stack:

```bash
./scripts/run_minimax_deep_research.sh
```

Stop local stack:

```bash
./scripts/stop_deep_research.sh
```

Force stop only if interrupting active local jobs is acceptable:

```bash
FORCE_STOP_DEEP_RESEARCH=1 ./scripts/stop_deep_research.sh
```

Default local URLs:

```text
Backend:  http://localhost:9000
Frontend: http://localhost:3000
SearXNG:  http://localhost:8080
```

Local runtime files:

```bash
.deep-research-runtime/
jobs.db
checkpoints.db
checkpoints.db-wal
checkpoints.db-shm
data/scrape_artifacts/
data/kokoro-models/
```

Treat these as runtime state, not source. Do not commit them.

## Oracle App2 Deployment

Oracle app2 is the production-like environment the user tests most often.

Compose file:

```bash
deploy/compose/docker-compose.app2.yaml
```

Important app2 services:

```text
app2-aiq-agent
app2-aiq-blueprint-ui
app2-aiq-postgres
app2-aiq-searxng
app2-aiq-websurfx
app2-aiq-websurfx-redis
app2-aiq-kokoro
```

Host bindings:

```text
Backend health: http://127.0.0.1:9000/health
Frontend host:  http://127.0.0.1:3110
Kokoro health:  http://127.0.0.1:3111/health
Public site:    https://app2.sniperip.com
```

After every code or config change that affects app2 behavior, sync and restart.
The user prefers Oracle to be restarted after changes rather than left running
old code.

Common command:

```bash
./scripts/sync_app2_deep_research.sh --apply --restart all
```

The sync script may end with:

```text
curl: (56) Recv failure: Connection reset by peer
```

That often happens while the backend is warming up after containers start. Do a
health poll before assuming failure:

```bash
ssh oracle 'for i in $(seq 1 40); do curl -fsS http://127.0.0.1:9000/health && exit 0; sleep 3; done; docker logs --tail 160 app2-aiq-agent; exit 1'
```

Do not run `docker compose down -v` on app2 unless the user explicitly approves
data loss. The volumes contain jobs, auth, checkpoints, artifacts, source cache,
and Kokoro model data.

## Environment Files

Local environment files may exist at:

```bash
.env
deploy/.env
```

Oracle environment file:

```bash
/opt/stacks/app2/deep-research/deploy/.env
```

Never print values from these files. If you need to inspect available variable
names:

```bash
ssh oracle "cd /opt/stacks/app2/deep-research && grep -E '^[A-Z0-9_]+=' deploy/.env | cut -d= -f1 | sort"
```

## API Surface

Important backend endpoints:

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
GET  /v1/jobs/async/job/{job_id}/report?format=docx
POST /v1/jobs/queue
GET  /v1/jobs/queue
POST /v1/jobs/queue/{id}/approve
POST /v1/jobs/queue/{id}/cancel
DELETE /v1/jobs/queue/{id}
```

Jobs emit a persisted `job.metrics` event before the terminal event with
per-tool search counts, document/cache-hit counts, and phase durations; the
eval harness consumes these for efficiency scoring.

Example local submit:

```bash
curl -X POST http://127.0.0.1:9000/v1/jobs/async/submit \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "deep_researcher",
    "input": "Research the best ways to live well in the age of AI.",
    "research_depth": "deeper"
  }'
```

For authenticated Oracle calls, use the UI session/proxy or an API key. Do not
hard-code tokens in docs.

## Authentication

Backend auth supports local users and API keys. Important areas:

```bash
frontends/aiq_api/src/aiq_api/routes/auth.py
frontends/aiq_api/src/aiq_api/auth
frontends/aiq_api/src/aiq_api/jobs/access.py
```

Relevant environment variable names include:

```text
REQUIRE_AUTH
AIQ_LOCAL_AUTH_ENABLED
AIQ_AUTH_TOKEN_SECRET
AIQ_AUTH_DB_URL
AIQ_LOCAL_USERS
AIQ_LOCAL_ADMIN_USERS
AIQ_AUTH_TOKEN_TTL_SECONDS
```

Do not assume history is public. Job visibility should respect auth/access
tables. `srijan` is the admin user in the deployed app.

## Kokoro Narration

Kokoro narration exists as:

- A dedicated app2 service: `app2-aiq-kokoro`.
- A frontend API route fallback:
  `frontends/ui/src/app/api/tts/kokoro/route.ts`.

Frontend app2 uses:

```text
KOKORO_TTS_URL=http://kokoro:8000
```

Narration should chunk long reports cleanly and keep controls usable on desktop
and mobile.

## Testing Expectations

For Python/backend changes:

```bash
uv run --extra dev pytest <focused tests>
uv run ruff check <touched python files>
```

For frontend changes:

```bash
cd frontends/ui
npm run build
npm test -- <focused tests>
```

For search package changes:

```bash
uv run --extra dev pytest sources/searxng_jina_web_search/tests
uv run ruff check sources/searxng_jina_web_search
```

For deployment-affecting changes:

```bash
./scripts/sync_app2_deep_research.sh --apply --restart all
ssh oracle "curl -fsS http://127.0.0.1:9000/health"
```

If changing research behavior, inspect an actual job trace or run a small smoke
job. Unit tests do not catch planner loops, missing file commits, missing SSE
events, broken source traces, or poor final synthesis.

## Debugging Playbooks

If the UI shows a shortened query/job id:

1. Check whether `job.submitted` exists in `job_events`.
2. Check event count for that job.
3. Check `job_info.output`.
4. Check backend logs for event-store failures.
5. Verify `get_job_events_channel()` is used for LISTEN/NOTIFY.

If research is stuck in planning:

1. Inspect thought/tool events in the UI or `job_events`.
2. Check whether `/shared/plan.json` was actually written and readable.
3. Look for JSON validation loops.
4. Check model timeout/repetition guard logs.
5. Confirm planner budget and forced commit behavior.

If search budget exhausts early:

1. Inspect the plan budget profile.
2. Count search calls per researcher task.
3. Check whether one branch consumed most budget.
4. Check search query length and complexity.
5. Preserve reserve for gap-filling and weak-source replacement.

If final report quality is poor:

1. Check citation/source artifacts.
2. Check source class distribution.
3. Inspect scrape artifacts for source content.
4. Look for unsupported numeric/date claims.
5. Check reference list integrity.
6. Prefer structural fixes: claim/evidence tables, source-quality scoring,
   verifier loops, and better plan decomposition.

If Oracle seems bloated:

```bash
ssh oracle "df -h"
ssh oracle "docker system df"
ssh oracle "sudo du -xh /opt/stacks/app2/deep-research 2>/dev/null | sort -h | tail -30"
ssh oracle "docker volume ls | grep app2_deep_research"
```

Do not delete volumes without explicit approval.

## Git And Hygiene

- Check `git status --short` before broad edits.
- Do not revert unrelated dirty files.
- Keep commits focused.
- Do not commit secrets, `.env`, runtime DBs, scrape dumps, `.venv`,
  `node_modules`, `.next`, or bulky generated artifacts.
- Prefer updating current-state docs over adding many speculative docs.
- Push to GitHub when the user asks for GitHub/Oracle parity.

## High-Risk Areas

Be especially careful around:

- Planner commit/write-plan logic.
- Model routing and thinking-mode changes.
- Search budgets and parallelism.
- Search query generation.
- DeepAgents virtual file writes.
- Event store and SSE recovery.
- Source classification and quality gates.
- Citation/reference generation.
- Report recovery and final output cleanup.
- Claude Code prompt contracts.
- Batch queue state.
- Auth/job access.
- Oracle Docker volumes and restart behavior.

## Current Operating Principles

1. Keep research auditable: plans, queries, sources, notes, citations, and final
   reports must be traceable.
2. Keep queries focused: do not search with giant pasted prompts.
3. Spend budget deliberately: allocate across branches and preserve reserve for
   gap-filling.
4. Use source quality early: prefer primary, official, academic, authoritative,
   and recent sources when the claim requires them.
5. Do not overfit prompt patches to one failed query.
6. Use M3 large-context power mainly where it helps: synthesis, verification,
   complex reasoning, and Claude Code delegation.
7. Persist before streaming: UI recovery matters more than real-time elegance.
8. Restart Oracle after behavior changes.
9. Protect secrets and persistent state.
10. The final report should be useful, cited, calibrated, and clean; no thought
    traces, broken references, or unsupported precision should ship.
