# Deep Research Process Audit

This document audits how the customized NVDA/MiniMax Deep Research platform works today, where it is strong, where it is fragile, and what fixes would improve quality without destabilizing the app.

It is based on the current local repository at:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
```

It also accounts for the Oracle app2 deployment at:

```bash
/opt/stacks/app2/deep-research
```

This audit intentionally describes only technologies that are part of the current active stack: NVIDIA AI-Q / NeMo Agent Toolkit, MiniMax M3, FastAPI, Next.js, Dask, DeepAgents/LangGraph-style virtual filesystem behavior, SearXNG/DDGS-style web discovery, Scrapling-style extraction, NotebookLM source harvesting where enabled, Stooq stock quotes, local debate transcript search, Postgres/SQLite job storage, Kokoro ONNX narration, and Docker Compose on app2.

## Executive Summary

The platform is a sophisticated multi-agent research system. A user query flows through intent classification, optional clarification, plan approval, async job submission, planner/researcher/orchestrator execution, web and local-source collection, scrape artifact persistence, citation verification, source-quality gates, report storage, and optional Kokoro audio narration.

The most important thing to understand is that the system is not a single model call. It is a pipeline. Quality depends on how well each stage hands structured evidence to the next one.

The biggest real remaining limitations are:

1. Post-run quality gates can still fail a job after the expensive research has already happened.
2. Search budget exhaustion is handled better than before, but the model can still spend too much of the budget before synthesis or gap-filling.
3. User-facing plan previews can still become generic when the plan-generation repair path falls back.
4. Source quality is measured after the report, but not yet strongly used as a live steering signal during research.
5. Claim-table and fact-ledger concepts exist, but they are not yet fully deterministic enough to make every report claim traceable to evidence.
6. Long synthesis remains the highest-risk phase because MiniMax M3 must compress many intermediate notes, citations, and quality constraints into a final report.
7. Local debate transcript search depends on a prepared static index; new transcript files are not automatically indexed unless the preparation script runs.
8. Local SQLite can still be a concurrency bottleneck under heavy telemetry, though Oracle app2 uses Postgres for the main job/checkpoint path.
9. Scrape artifacts are valuable for auditability, but need retention and size discipline so app2 disk use does not grow forever.

The highest-value fix is to change late fatal quality gates into quality-audited success where possible. A report that is non-empty, on-topic, and usable should be delivered with visible quality warnings instead of being converted into a failure after 10-30 minutes of work. Hard failure should be reserved for empty reports, raw provider/thinking output, severe scope drift, or no usable report artifact.

The second highest-value fix is to make source quality and budget state visible to the model while it is still researching, not only after the report is written.

## Current Active Architecture

The system has these major pieces:

- Next.js UI in `frontends/ui`
- FastAPI / AI-Q API backend in `frontends/aiq_api`
- NAT workflow configuration in `configs/config_cli_minimax_ddgs.yml`
- Deep research agent code in `src/aiq_agent/agents/deep_researcher`
- Shallow research agent code in `src/aiq_agent/agents/shallow_researcher`
- Clarifier and intent agents in `src/aiq_agent/agents/clarifier` and `src/aiq_agent/agents/chat_researcher`
- Common reliability and audit modules in `src/aiq_agent/common`
- Local and app2 scripts in `scripts`
- Oracle app2 Docker Compose in `deploy/compose/docker-compose.app2.yaml`

The active config is:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/configs/config_cli_minimax_ddgs.yml
```

The MiniMax model roles are split:

- `minimax_m3_planner_llm` for intent, clarification, and planning
- `minimax_m3_llm` for most research turns
- `minimax_m3_synthesis_llm` for final synthesis/orchestration

The current active search and retrieval sources are:

- Self-hosted SearXNG-based web discovery
- DDGS-style fallback discovery inside the web search plugin
- Scrapling-style extraction
- NotebookLM source harvesting for the shallow/experimental path when credentials/runtime are available
- Stooq stock quotes
- Local debate transcript search over a prepared JSONL corpus

Kokoro ONNX is used for report narration after a report is available.

## Local vs Oracle app2

### Local

Local project root:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
```

Start:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/run_minimax_deep_research.sh
```

Stop:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/stop_deep_research.sh
```

The start script installs local editable packages, prepares the debate transcript corpus, ensures Kokoro model files, ensures SearXNG, then starts the backend and frontend.

Local URLs are normally:

```text
Backend:  http://localhost:9000
Frontend: http://localhost:3000
SearXNG:  http://localhost:8080
```

Local runtime data may include SQLite files:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/jobs.db
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/checkpoints.db
```

Local scrape artifacts live under:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/data/scrape_artifacts
```

### Oracle app2

Oracle app2 source root:

```bash
/opt/stacks/app2/deep-research
```

App2 compose file:

```bash
/opt/stacks/app2/deep-research/deploy/compose/docker-compose.app2.yaml
```

App2 services:

```text
app2-aiq-agent
app2-aiq-blueprint-ui
app2-aiq-kokoro
app2-aiq-postgres
app2-aiq-searxng
```

App2 host bindings:

```text
Backend:  127.0.0.1:9000
Frontend: 127.0.0.1:3110
Kokoro:   127.0.0.1:3111
Postgres: Docker network only
SearXNG:  Docker network only
```

App2 state is preserved in Docker volumes:

```text
app2_deep_research_aiq_data
app2_deep_research_postgres_data
app2_deep_research_searxng_cache
app2_deep_research_kokoro_data
```

These volumes are important. Do not remove them unless you intend to erase job history, checkpoint data, scrape artifacts, search cache, or Kokoro model data.

Sync local source to app2:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/sync_app2_deep_research.sh --apply
```

Sync and restart backend:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/sync_app2_deep_research.sh --apply --restart backend
```

Sync and restart frontend:

```bash
cd /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
./scripts/sync_app2_deep_research.sh --apply --restart frontend
```

The sync script excludes `.env`, runtime databases, logs, local data, node modules, Python caches, and recovered reports. That is correct. Secrets and local runtime junk should not be copied as source.

## Research Flow From Query to Output

### 1. User starts from UI or API

The user can start research from the UI or by calling the async API.

Main async API endpoints expected by app1 and other clients:

```text
GET  /health
POST /v1/jobs/async/submit
GET  /v1/jobs/async/job/{job_id}
GET  /v1/jobs/async/job/{job_id}/state
GET  /v1/jobs/async/job/{job_id}/report
```

The UI streams events as the job runs. The API stores job state and event history so clients can poll.

### 2. Intent classifier decides whether this is research

The intent classifier uses MiniMax M3 planner model. Its job is to distinguish meta/chat requests from research requests and assign a depth tier when possible.

Depth tiers are:

```text
shallow
deeper
deep
```

The default normalized depth is configured in `src/aiq_agent/common/research_depth.py`.

### 3. Clarifier may ask a question

If the system thinks the query is underspecified, the clarifier can ask a focused question before research starts.

This is useful when the query could mean very different things. It is risky when the clarifier asks generic questions or when the user's answer is later treated as the whole query. That exact failure mode has happened before: the user answered something like "1 and 4," and the plan title became "1 and 4."

The correct behavior should be:

- Preserve the original user query as the root task.
- Treat clarification answers as constraints or preferences.
- Never replace the research topic with the clarification answer.

This is a real area to harden.

### 4. Plan generation and approval

The plan preview is user-facing. It should tell the user what the system intends to research.

Current risk:

- If plan JSON is malformed or weak, fallback sections can leak into the UI.
- Generic sections like "Requirements," "Architecture," "Failure Paths," and "Implementation Plan" are sometimes technically valid but too vague for a serious research plan.

The internal plan may be richer than the preview, but that does not solve the user trust problem. If a user approves a vague plan, the system has effectively received permission to do vague research.

Best fix:

- Keep fallback as a crash guard, but do not silently present it as a high-quality plan.
- Run one focused plan-repair pass.
- If repair fails, ask the user to retry/revise rather than showing a generic fallback plan.
- Preserve both the original query and clarification answers in the plan object.

### 5. Async job submission

Once approved, the backend creates an async job. The job runs in a Dask/NAT runtime and writes event logs to the configured job store.

On app2, job and checkpoint state are backed by Postgres through compose environment variables:

```text
NAT_JOB_STORE_DB_URL
AIQ_CHECKPOINT_DB
AIQ_SUMMARY_DB
```

Locally, SQLite may be used unless those variables point elsewhere.

### 6. Planner runs inside deep research

For deep/deeper jobs, the planner prepares a deeper internal plan. It may produce:

- task analysis
- report sections
- researcher task list
- source targets
- constraints
- entities or claims, depending on prompt path

The planner is allowed some search budget for grounding. Current budgets:

```text
shallow planner: up to 4 search calls
deeper planner: up to 12 search calls
deep planner: up to 24 search calls
```

Planner search can improve downstream quality, but it must not behave like the whole research phase. If planner search consumes attention but does not translate into better researcher tasks, it becomes waste.

Best direction:

- Let the planner ground enough to create a good task decomposition.
- Require it to produce a budget allocation per task.
- Require it to identify source classes needed for the claims it expects the report to make.
- Avoid planner-driven source-count inflation where it tells researchers to collect more than the configured budget can support.

### 7. Orchestrator dispatches researcher tasks

The orchestrator coordinates researcher agents. It reads the plan, sends tasks, watches files in the virtual filesystem, and eventually writes `/report.md`.

Current depth budgets:

```text
shallow:
  target sources: 10-20
  max researcher tasks: 2
  max parallel researcher tasks: 1
  search calls per researcher: 8
  advanced search-family calls: 20

deeper:
  target sources: 32-64
  max researcher tasks: 5
  max parallel researcher tasks: 2
  search calls per researcher: 12
  advanced search-family calls: 64

deep:
  target sources: 90-150+
  max researcher tasks: 10
  max parallel researcher tasks: 2
  search calls per researcher: 14
  advanced search-family calls: 140
```

These are not infinite. If the model acts as if "deep" means "keep searching until everything is perfect," it will hit the budget.

### 8. Search and extraction

Researchers use tools based on the active data source set.

For web research, the main path is self-hosted SearXNG discovery plus extraction. The web search plugin uses a hybrid discovery backend and has request timeouts and retry limits.

For shallow/experimental source harvesting, NotebookLM source harvesting can be available if the deployment has the runtime and credentials set up. In the active deep research agent config, NotebookLM is excluded from deep-research-agent tools, so it is not the main deep synthesis path today.

For stock data, Stooq quote lookup is available.

For debate materials, local transcript search uses:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data/prepared_debate_corpus/chunks/search_chunks.jsonl
```

On app2, the equivalent path is mounted read-only under:

```bash
/app/data/prepared_debate_corpus/chunks/search_chunks.jsonl
```

Transcript search only sees what has been prepared into the JSONL index. Dropping new raw transcripts into a source folder is not enough unless the preparation script runs.

### 9. Source registry and scrape artifacts

As tools return URLs and extracted text, the system registers sources and persists scrape artifacts.

Scrape artifact storage is handled by:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/scrape_artifacts.py
```

Artifacts include:

- job id
- URL
- title
- extraction status
- content hash
- researcher/task identity
- tool name
- source class
- compressed content
- content length
- created timestamp

This is a major strength. It means fetched documents can be audited later instead of disappearing from `/large_tool_results`.

Limitation:

- Storage must be managed.
- Artifacts are compressed, but long-running use can still grow app2 disk usage.
- Retention policy and per-artifact caps should be explicit.

### 10. Virtual filesystem notes

DeepAgents-style researcher notes are written to virtual paths such as:

```text
/shared/...
/report.md
```

These are not ordinary durable OS files during the agent run. They are part of the agent state and event stream.

This matters because parallel researcher tasks can race if they write the same virtual path. The safer pattern is one file per researcher or task, followed by deterministic merge.

### 11. Claim table and fact ledger

The repo now has claim-table infrastructure:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/claim_table.py
```

The purpose is good: convert research into atomic, verifiable claims with status:

```text
verified
partially_verified
unverified
```

This is the right long-term direction. However, the current system still relies too much on prompt compliance rather than deterministic orchestration.

What should eventually happen:

1. Planner creates expected atomic claims.
2. Researchers resolve claims with cited evidence.
3. Python merges claim fragments deterministically.
4. Orchestrator writes from the validated claim table and compact notes.
5. Quality gates audit claim coverage and calibration.

If this is fully enforced, the system becomes much less likely to produce confident unsupported claims.

### 12. Synthesis

The orchestrator reads researcher notes, source registry, claim/fact artifacts when present, and writes `/report.md`.

This is the hardest stage for MiniMax M3 because the model must:

- compress many notes
- preserve the original scope
- avoid drifting into adjacent themes
- use enough citations
- cite only verified sources
- not include internal workflow text
- not output raw thinking blocks
- satisfy quality gates

Long/deep jobs are most likely to fail here.

Best structural fix:

- Move toward section-level synthesis.
- Have each researcher produce compact, source-backed section briefs.
- Merge claim tables deterministically.
- Final writer uses briefs and claim table, not raw full search history.

### 13. Citation verification and report cleanup

After report generation, citation verification checks whether inline references and bibliography entries correspond to sources the system actually saw.

The system already has deterministic URL repair and normalization. It can resolve some malformed citations through exact, normalized, prefix, child-path, and query-subset matching.

Remaining risk:

- If citation repair fails, citations can be stripped.
- Stripping citations reduces cited-source counts.
- Reduced cited-source counts can trigger post-run quality failures.

The pasted analysis is directionally right about this cascade, but the current code is more advanced than a simple "delete every mismatch" approach.

Best fix:

- Keep deterministic citation repair.
- Add a final reconciliation step for near-miss URLs before stripping.
- When stripping still happens, downgrade the report with a warning instead of failing if the report is otherwise usable.

### 14. Post-run quality gates

Post-run quality evaluation currently checks for problems such as:

- missing `/report.md`
- no citation-use events
- failed researcher tasks
- cited sources too sparse compared with collected sources
- source quality gate failures
- missing or invalid fact ledger for entity-heavy plans
- central entities with zero verified ledger facts
- low first-party citation share for entity-heavy reports
- invalid or under-resolved claim table

These checks are useful. The problem is that they currently happen late and can raise a fatal `RuntimeError`.

This is the highest-friction failure mode: the user may wait through a long run, the report may exist, and then the backend marks it failed because an audit rule fired.

Better behavior:

- Hard-fail only empty report, raw provider/thinking output, severe scope drift, or no usable report artifact.
- For source diversity, citation sparsity, weak source mix, missing ledger, or under-resolved claim table, deliver the report with `quality_warnings`.
- Make the UI show a visible quality audit banner.
- Let the user request "refine sources" or "repair citations" instead of losing the report.

### 15. Report storage and retrieval

Final reports are stored as job output and can be retrieved through API endpoints. Job history and events are backend state, not just browser cache.

On app2, Postgres and app2 Docker volumes are the important state stores.

For API clients, the best pattern is:

1. Submit async job.
2. Poll job status.
3. Poll state/events for progress.
4. Fetch report endpoint only when status or state says a report is ready.
5. If report is not ready, handle "not ready" as progress, not failure.
6. If completed with warnings, display report and warnings together.

### 16. Kokoro narration

Kokoro is a separate service on app2:

```text
app2-aiq-kokoro
```

It is bound on the VPS host at:

```text
127.0.0.1:3111
```

The frontend talks to it over the Docker network using:

```text
http://kokoro:8000
```

Kokoro should run after the report is ready. The best narration behavior is:

- split the final report into sentence-safe chunks
- avoid cutting in the middle of words
- generate chunks continuously
- append chunks into a playlist-like player
- let the user seek, skip, pause, and resume

The known limitation is that long report narration needs chunk orchestration; a single generated audio chunk is not enough.

## Variations by Research Type

### Shallow

Shallow research is intended for smaller jobs, faster responses, and lower source targets.

Current budget:

```text
target sources: 10-20
max researcher tasks: 2
parallel researcher tasks: 1
search calls per researcher: 8
advanced search-family budget: 20
```

NotebookLM source harvesting may be part of the shallow path when runtime/auth are available. This makes shallow slower than a normal quick search if NotebookLM harvesting is enabled, but it is useful as a test pathway for external source collection.

Limitations:

- It can still exhaust budget if the planner creates too many objectives.
- It should not pretend to be comprehensive.
- It needs clear UI expectations if NotebookLM harvesting makes it slow.

### Deeper

Deeper is the balanced tier.

Current budget:

```text
target sources: 32-64
max researcher tasks: 5
parallel researcher tasks: 2
search calls per researcher: 12
advanced search-family budget: 64
```

This is usually the right default for serious but not exhaustive research.

Limitations:

- Can still hit budget on broad prompts.
- Needs good planner decomposition.
- Needs reserve preserved for gap-filling and synthesis.

### Deep

Deep is the highest-budget tier.

Current budget:

```text
target sources: 90-150+
max researcher tasks: 10
parallel researcher tasks: 2
search calls per researcher: 14
advanced search-family budget: 140
```

Deep does not mean infinite. It means more source collection, more researcher tasks, and longer synthesis.

Limitations:

- Highest chance of late synthesis pressure.
- Highest chance of over-searching if the planner interprets "deep" as a source-count obligation rather than an evidence-sufficiency target.
- Highest need for budget allocation and live source-quality steering.

### Lesson-plan research

Lesson-based research has special risks because it often has two related but different anchors:

- lesson topic
- final motion / debate motion

Correct behavior:

- The lesson topic is the primary scope.
- The motion is a relevance anchor.
- Research should not drift to one intersection unless the query explicitly asks for it.

Known failure pattern:

- A lesson about a broad topic like "art and activism" can drift into one subcase like feminist art activism if early searches over-index that intersection.

Best fix:

- Give lesson research a budget allocator across lesson topic, motion-specific context, case studies, arguments, counterarguments, and teaching examples.
- Require the planner to preserve broad topic coverage before allowing deep dives into one subtheme.

### Local transcript research

Local transcript research uses a prepared corpus. It is useful for debate-specific argument interaction, framing, clashes, rebuttal patterns, and analogies.

It is not a magic folder search over raw files. The prepared index must exist and be current.

Best fix:

- Add startup freshness checks.
- If raw transcript files are newer than the prepared JSONL, show a warning.
- Optionally rebuild automatically during startup or through a dedicated admin action.

### Stock/finance research

Stock quote lookup uses Stooq. It can provide current quote-style data, but broad valuation research still depends on web sources and should be treated cautiously.

Limitations:

- Quotes are one input, not full valuation.
- Any recommendation-like report needs source calibration and caveats.
- Specific numbers must be cited to source classes appropriate to the claim.

## What the Attached System Analysis Got Right

### Fatal late-stage gates are real

The analysis is correct that quality gates can fire after the report has already been generated. This can turn a useful report into a failed job.

This is the most important practical issue to fix.

### Citation stripping can cascade

The analysis is directionally correct. If citations are stripped, the cited-source count drops, and quality gates can then fail.

However, the current implementation already has deterministic URL repair and normalization. The real problem is not "no healing exists." The real problem is that remaining unresolved citations can still lead to hard failure.

### Thinking-only MiniMax stalls are real

The system has middleware specifically to repair thinking-only behavior. That means the issue is real.

However, it is partially mitigated. The remaining risk is that repeated repair adds latency/context and can still fail if the model never produces usable content.

### Search budget exhaustion is real

The analysis is correct that budget exhaustion can trap the run if the model keeps trying to search.

However, the quoted budget numbers in that analysis are stale. Current budgets are much larger and depth-aware. The system also has search-budget repair middleware that tells the model to stop searching and can strip exhausted search calls.

The remaining issue is planner and prompt behavior: the model can still spend budget inefficiently before the repair middleware has to intervene.

### Transcript index staleness is real

The analysis is correct. The transcript search corpus is static until prepared again.

### SQLite locking is real locally

Local SQLite can lock under heavy event/checkpoint writes. The current code has batching and a 30-second timeout, so it is not unmitigated.

On Oracle app2, this is much less central because Postgres is used for production job/checkpoint state.

### Scrape and telemetry bloat are real but overstated

The system already truncates many tool payloads and compresses scrape artifacts. Still, long-term storage bloat is real, especially on app2.

The better framing is storage lifecycle management, not immediate unavoidable OOM.

## Current Reliability Mechanisms

The repo already includes several important protections:

- Tool budget middleware
- Search-budget exhaustion repair
- Sequential search limiting
- Parallel researcher task limiting
- Tool retry middleware
- Tool-name repair
- Thinking-only repair
- Source registry middleware
- Tool-result pruning
- Citation verification
- Source classification
- Source quality gates
- Scrape artifact persistence
- Post-run quality checks
- Report recovery from event artifacts
- Stop script that avoids interrupting active jobs unless forced

This matters because the system is not naive. The next improvements should not pile on more prompt text. They should make existing protections earlier, softer where appropriate, and more deterministic.

## Core Limitations

### MiniMax M3 is capable but needs structure

MiniMax M3 can do strong research when the task is decomposed well. It is less reliable when asked to:

- obey many buried instructions
- produce perfect JSON every time
- synthesize very large intermediate state
- stop searching when the prompt still implies more evidence is required
- distinguish a user-facing final answer from internal reasoning
- keep citation discipline across a long report

The architecture should assume that the model needs bounded tasks, explicit artifacts, validation, repair, and deterministic merge steps.

### The current pipeline still has too much late judgment

The system often detects quality problems after the fact. Post-run gates are valuable for audit, but quality improves more when the same information steers the next researcher task.

For instance:

- Late: "final report failed because too many citations came from one domain."
- Better: "after researcher batch 1, source diversity is weak; next task must target primary/official/academic sources."

### The plan is not always good enough as a contract

If the user approves a vague plan, downstream research can be vague. If the plan title is truncated or based on a clarification answer, the job can drift from the real query.

The plan preview should be treated as a formal contract. Bad plans should be repaired or rejected before approval.

### Deep research can over-search

The system now has budgets, but a model can still burn through them if the planner creates too many broad tasks or if researcher agents interpret "deep" as "keep searching until source count is huge."

Deep research needs evidence sufficiency, not blind source accumulation.

### Final synthesis is the bottleneck

Research collection can be parallelized. Synthesis is harder. It has to integrate everything coherently and avoid hallucinations.

The more raw material the orchestrator must read, the more likely it is to:

- omit important facts
- cite weakly
- drift
- produce generic prose
- fail output formatting
- hit thinking-only or empty-report problems

### Source quality is not yet a first-class research objective

Source classification exists. Source quality gates exist. But the planner/researcher path is not yet fully driven by source-class targets.

A high-quality research system should know, before searching:

- what claims it expects to make
- what source classes those claims require
- whether each claim is verified, partially verified, or unverified

## Highest-Value Fixes

### 1. Convert non-critical quality failures into warnings

This is the best immediate fix.

Current problem:

- A report can exist and be useful.
- A quality gate fires late.
- The job becomes failed.
- The user sees failure instead of a report.

Recommended behavior:

- Keep hard failure for empty report, raw provider/thinking output, severe drift, or no usable report.
- Treat source diversity, citation sparsity, weak authority mix, missing ledger, or under-resolved claim table as `quality_warnings`.
- Deliver the report.
- Show warnings clearly in the UI and API.

This preserves user value while keeping honesty.

### 2. Add in-flight source quality steering

After each researcher batch, compute:

- distinct domains
- source class distribution
- dominant domain share
- number of primary/official/academic-style sources
- collected vs cited candidate source count

Then inject a compact instruction into the next task:

```text
Current source mix is weak: 2 domains, 70% content-marketing-style sources, 0 primary/official/academic sources. Next searches must target stronger sources and avoid repeating the dominant domain.
```

This turns quality gates from punishment into guidance.

### 3. Add deterministic budget allocation

Planner should output a budget plan:

```json
{
  "research_budget": {
    "total_search_calls": 64,
    "planner_reserved": 8,
    "researcher_allocations": [
      {"task": "case studies", "search_calls": 12},
      {"task": "counterarguments", "search_calls": 10}
    ],
    "gap_fill_reserve": 8,
    "synthesis_reserve": 0
  }
}
```

The actual numbers should come from `research_depth.py`, not model imagination.

For lesson-specific research, allocate across topic, motion, cases, arguments, counterarguments, and teaching examples.

For broader research, allocate more symmetrically unless the planner gives a strong reason.

### 4. Make plan preview repair mandatory

If the plan preview has a title that is just a truncated query or clarification answer, or sections that are generic templates, run repair.

Plan preview quality rules:

- Title should summarize the actual user request.
- Sections should be specific enough to approve.
- Clarification answers should not replace the query.
- No generic fallback should be silently presented as a good plan.

### 5. Make claim-table merge deterministic

The claim table should become a hard artifact, not just a prompt idea.

Recommended flow:

1. Planner creates claim targets.
2. Each researcher writes claim fragments to a per-task file.
3. Python validates and merges fragments.
4. Orchestrator receives a compact claim summary.
5. Final report claims must trace to verified or partially verified claims.

This improves truthfulness more than adding more prompt text.

### 6. Use section-level synthesis for long runs

For deep research:

- Each researcher writes notes.
- A section summarizer converts notes into compact evidence-backed briefs.
- Final writer reads section briefs plus claim table.

This reduces context pressure during final synthesis.

### 7. Add transcript freshness checks

On startup or admin action:

- compare raw transcript file mtimes against prepared corpus mtime
- warn if stale
- optionally rebuild

This prevents local research from silently using old data.

### 8. Add retention and caps for scrape artifacts

Recommended:

- per-artifact max content length
- preserve hash and metadata even when truncating
- retention policy by age or disk threshold
- admin script to summarize and clean old artifacts

This protects app2 disk space without losing auditability.

## Fixes To Avoid

### Do not just raise budgets

Raising budgets can help specific jobs, but it does not solve inefficient planning. It can make synthesis harder by producing more raw material.

Raise budgets only with:

- allocation
- reserve
- in-flight source monitoring
- clear synthesis transition rules

### Do not add many more prompt paragraphs

MiniMax M3 is already handling long prompts. More text can reduce compliance.

Prefer:

- structured artifacts
- deterministic validators
- small repair prompts
- shorter high-priority instructions

### Do not make every warning fatal

Fatal gates are useful for preventing garbage, but many quality issues should degrade the confidence label, not erase the report.

### Do not restart app2 blindly

There may be active jobs. Use scripts and status checks. Restart only the service needed.

## Recommended Implementation Order

### Phase 1: Deliver Reports With Quality Warnings

Modify the backend completion path so non-critical post-run quality problems are saved as warnings rather than raising fatal errors.

Keep hard errors for:

- empty final report
- raw provider/thinking output
- severe scope drift
- no usable report artifact

Expected value:

- fewer "research finished but failed" experiences
- less user frustration
- immediate access to imperfect but useful reports

Risk:

- low to medium
- API/UI must expose warnings clearly

### Phase 2: Plan Preview Repair

Detect weak plans before approval:

- title too short/generic
- title equals clarification answer
- title is truncated weirdly
- sections are generic templates
- original query missing from plan context

Run a repair pass or ask the user to retry.

Expected value:

- less drift
- higher user trust
- better downstream task decomposition

Risk:

- low

### Phase 3: In-Flight Source Quality Steering

Expose source quality summary to orchestrator after each batch.

Expected value:

- better source mix
- fewer post-run source-quality warnings
- less reliance on SEO/listicle sources

Risk:

- medium
- must avoid making researchers over-search

### Phase 4: Budget Allocator

Make the planner allocate search budget by task within configured tier limits.

Expected value:

- less budget exhaustion
- better reserve for gap filling
- clearer transition from search to synthesis

Risk:

- medium
- needs careful tests on shallow/deeper/deep

### Phase 5: Deterministic Claim Table

Move claim-table merge/validation into Python and make it a required artifact for deeper/deep research where feasible.

Expected value:

- much better factuality
- better citation discipline
- clearer uncertainty

Risk:

- medium to high
- changes core research behavior

### Phase 6: Section-Level Synthesis

Add compact section briefs before final report synthesis.

Expected value:

- fewer final synthesis failures
- better long-report coherence
- less context pressure

Risk:

- medium

### Phase 7: Storage and Transcript Maintenance

Add transcript freshness checks and scrape artifact cleanup tools.

Expected value:

- less app2 disk growth
- fewer stale local-search failures

Risk:

- low

## Open Questions

### Should shallow remain a NotebookLM experiment?

Right now shallow can be used as a source-harvesting test path. That is useful, but it changes user expectations because shallow may become slower.

Recommendation:

- Make this explicit in the UI.
- Consider labels like "Shallow Web" and "NotebookLM Assisted" if both are supported.

### Should quality warnings become a formal status?

A new status like `SUCCESS_WITH_WARNINGS` is attractive, but it may break clients that expect existing status values.

Safer first step:

- keep `SUCCESS`
- add `quality_warnings` to output metadata
- emit quality audit events
- update UI to show warnings

### How much should app1 see?

App1 should receive:

- job id
- current status
- progress/state
- final report
- quality warnings
- source quality summary

App1 should not need to understand every internal agent event.

## Conclusion

This platform is already significantly more advanced than a basic web-search wrapper. It has async jobs, plan approval, multiple research depths, source registry, scrape artifacts, source classification, quality gates, recovery paths, local transcript search, and app2 deployment with Postgres-backed persistence.

The main issue is that some of the quality architecture is still positioned too late in the pipeline. The system often discovers that a report has weak sources, sparse citations, or under-resolved claims only after the report has already been generated. That creates the frustrating pattern where research appears to work, then the final job fails.

The best near-term improvement is to deliver usable reports with honest quality warnings instead of failing them. The best medium-term improvement is to make source quality, budget state, and claim resolution steer the research while it is still happening. The best long-term improvement is deterministic claim-table orchestration plus section-level synthesis.

In simple terms: the next stage is not "more search." It is better control over what the search is trying to prove, when enough evidence is enough, and how the final report is allowed to make claims.
