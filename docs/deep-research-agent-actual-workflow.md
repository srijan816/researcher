# Deep Research Agent Actual Workflow

Last audited: 2026-06-01
Local project root: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research`
Oracle app2 deploy root: `/opt/stacks/app2/deep-research`
Live app2 backend: `http://127.0.0.1:9000` on the Oracle host
Public UI: `https://app2.sniperip.com`

This document describes how the current Deep Research system actually works in code, including the real prompt templates and the failure modes observed in the latest running job. It is not a design ideal.

## Quick Map

Main backend entry points:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/aiq_api/src/aiq_api/routes/jobs.py`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/aiq_api/src/aiq_api/jobs/submit.py`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/aiq_api/src/aiq_api/jobs/runner.py`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/agent.py`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/custom_middleware.py`

Main frontend event/client code:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/adapters/api/deep-research-client.ts`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/features/chat/store.ts`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/features/layout/components/ResearchPanel.tsx`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/features/layout/components/ThinkingTab.tsx`

Prompt templates currently loaded by the system:

- Intent routing: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/chat_researcher/prompts/intent_classification.j2`
- Clarification: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/clarifier/prompts/research_clarification.j2`
- User-facing plan preview: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/clarifier/prompts/plan_generation.j2`
- Deep orchestrator: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/orchestrator.j2`
- Deep planner: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/planner.j2`
- Deep researcher: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/researcher.j2`
- Source registry given to final writer: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/source_registry.j2`
- Shallow researcher: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/shallow_researcher/prompts/researcher.j2`

## What Runs on Oracle Right Now

Docker containers on app2:

- `app2-aiq-agent`: backend API, host `127.0.0.1:9000`, container port `8000`
- `app2-aiq-blueprint-ui`: UI, host `127.0.0.1:3110`, container port `3000`
- `app2-aiq-kokoro`: Kokoro TTS, host `127.0.0.1:3111`, container port `8000`
- `app2-aiq-postgres`: async jobs, auth, conversations, checkpoints
- `app2-aiq-searxng`: self-hosted metasearch, internal port `8080`

The backend config is:

- `/opt/stacks/app2/deep-research/configs/config_cli_minimax_ddgs.yml`

The current backend reports these tool facts at startup:

- Shallow available tools: `debate_transcript_search_tool`, `stock_quote_tool`, `web_search_tool`
- Deep available tools: `advanced_web_search_tool`, `debate_transcript_search_tool`, `stock_quote_tool`
- `exa_web_search_tool` is registered but unavailable unless `EXA_API_KEY` exists.

## End-to-End Flow

### 1. User submits a request

The UI sends the user prompt to the backend through the local Next.js API proxy, then the FastAPI backend. For async jobs, the important backend endpoint is:

- `POST /v1/jobs/async/submit`

Registered in:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/aiq_api/src/aiq_api/routes/jobs.py`

The job submission includes:

- `agent_type`
- `input`
- optional `job_id`
- optional `data_sources`
- optional `research_depth`

The backend authorizes the user, stores job access metadata, submits the job through Dask, and records submission events in Postgres.

### 2. History and access control

Jobs are stored in the `aiq_jobs` Postgres database.

Important tables:

- `job_info`
- `job_events`
- `job_access`
- `ui_conversations`
- `local_auth_users`
- `api_keys`

The API requires an authenticated principal for job access. The current app2 backend logs show:

- `AuthMiddleware registered (require_auth=True, validators=['LocalUserTokenValidator', 'DatabaseAPIKeyValidator'])`

The job history endpoint:

- `GET /v1/jobs/async/jobs?limit=50`

returns jobs visible to the logged-in principal. This is why history is supposed to come from the database, not only from browser cache.

### 3. Job events and live UI streaming

Every running job emits events into `job_events`. The UI reads them with SSE:

- `GET /v1/jobs/async/job/{job_id}/stream`
- `GET /v1/jobs/async/job/{job_id}/stream/{last_event_id}`

The event stream starts from historical events and continues with live events.

Important event types include:

- `job.submitted`
- `job.started`
- `job.heartbeat`
- `workflow.start`
- `workflow.end`
- `llm.start`
- `llm.chunk`
- `llm.end`
- `tool.start`
- `tool.end`
- `artifact.update`
- `final_report`
- `job.completed`
- `job.failed`
- `job.interrupted`

Important note about time display:

The backend stores real event timestamps in `job_events.created_at` and often also embeds a nested ISO timestamp inside `event_data`. The frontend has several paths that still default to `new Date()` when reconstructing events or local store objects. The likely source of "everything shows the current time after refresh" is in:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/features/chat/store.ts`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/adapters/api/deep-research-client.ts`

Concrete examples:

- `addDeepResearchLLMStep` creates `timestamp: new Date()`
- `addDeepResearchToolCall` creates `timestamp: new Date()`
- `addDeepResearchFile` creates or updates timestamps with `new Date()`
- event callbacks do receive `timestamp?: string`, but some store actions do not preserve it all the way through.

So the timestamp issue is probably a frontend reconstruction bug, not a backend event-store issue.

## Research Modes

The depth tiers are defined in:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/research_depth.py`

Current values:

| Tier | Target Sources | Max Researcher Tasks | Max Parallel | Search Calls Per Researcher | Planner Search Limit | Advanced Search Limit |
|---|---:|---:|---:|---:|---:|---:|
| `shallow` | 10-20 | 2 | 1 | 8 | 4 | 20 |
| `deeper` | 32-64 | 5 | 3 | 12 | 12 | 64 |
| `deep` | 90-150+ | 10 | 3 | 14 | 24 | 140 |

Important practical detail:

The search budget is enforced by `ToolBudgetMiddleware`. It collapses web search tools into a shared `search` family. Planner search has separate scoped limits, for example `planner:search`, so planner calls do not spend the main researcher search pool.

## Intent and Clarification Path

Before deep research, the chat researcher/clarifier path can classify and ask for clarification.

### Intent classifier

Template:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/chat_researcher/prompts/intent_classification.j2`

Actual behavior:

- Classifies query as `meta` or `research`.
- Extracts named entities.
- Chooses `shallow` or `deep`.
- Uses entity density override: 3+ named entities routes to deep.

Important excerpt:

```text
IF INTENT IS "RESEARCH":
Determine the depth.
- shallow: Single main question, factual lookup, 2-3 tool calls, no complex comparison.
- deep: Multi-faceted, explicit comparisons, trend analysis, strategy/roadmaps, or "comprehensive" requests.
- Entity density override: If the query names 3 or more specific products, models, tools, companies, people, statutes, or organizations, choose "deep" regardless of other signals.
```

### Clarification prompt

Template:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/clarifier/prompts/research_clarification.j2`

Actual behavior:

- It should avoid over-clarifying.
- It can ask a single high-impact question.
- It should not ask generic "business vs personal vs technical" questions unless the query is genuinely broad.

Important excerpt:

```text
TOP PRIORITY: Do not over-clarify. If the request is reasonably researchable, return `needs_clarification: false` and let the planner make assumptions explicit.
```

### User-facing plan preview

Template:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/clarifier/prompts/plan_generation.j2`

Actual behavior:

- Generates only a lightweight UI-facing plan: title plus 4-7 section names.
- This is not the same as `/shared/plan.json`.
- If the user approves it, code preloads a compact `/shared/plan.json` from the approved title/sections.
- If the plan is detected as generic fallback, the deep planner is supposed to ignore it and create a new specific plan.

Important excerpt:

```text
If the user already provided explicit report dimensions, preserve them in the user-facing plan. Do not replace a detailed request with generic buckets.
For ranked-list reports ... the plan MUST include ranking criteria, the ranked list itself, quantified impact/evidence, examples, maturity/technology/adoption analysis, and outlook when requested.
Never return placeholder ... Landscape, Recent Evidence and Signals, Capability Gaps, Adoption Risks and Recommendations, Requirements, Architecture and Interfaces, Failure Paths and Guardrails, Implementation Plan...
```

## Deep Research Agent Construction

Main file:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/agent.py`

The constructor:

1. Receives role-based LLMs from `LLMProvider`.
2. Receives registered tools from the config.
3. Loads prompts from the prompt directory.
4. Creates a `SourceRegistryMiddleware` to capture URLs returned by search/source tools.
5. Defines orchestration-only tools:
   - `think`
   - `defer_task`
   - `get_verified_sources`
   - `get_source_quality_snapshot`
6. Builds middleware for orchestrator, planner, and researcher scopes.

## Role-Specific Middleware

Defined in:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/custom_middleware.py`
- wired in `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/agent.py`

Current middleware stack:

1. `EmptyContentFixMiddleware`
2. `ToolNameSanitizationMiddleware`
3. `PlanFileValidationMiddleware`
4. `TaskBatchLimitMiddleware`
5. `SequentialSearchMiddleware`
6. `SearchBudgetExhaustionRepairMiddleware`
7. `ToolBudgetMiddleware`
8. `ToolRetryMiddleware`
9. `SourceRegistryMiddleware`
10. `ToolResultPruningMiddleware`
11. `ThinkingOnlyRepairMiddleware`
12. `ModelRetryMiddleware`

### PlanFileValidationMiddleware

Added to stop invalid `/shared/plan.json` writes.

It intercepts only:

- `write_file` to `/shared/plan.json`
- `write_file` to `shared/plan.json`
- `write_file` to `/plan.json`
- `write_file` to `plan.json`

It requires:

- `task_analysis`
- `report_title`
- `report_toc`
- `constraints`
- `output_style`
- `queries`

If the JSON is invalid or missing required fields, the write is rejected and the planner receives `PLAN_FILE_VALIDATION_FAILED`.

This protects downstream research from an incomplete plan, but it can also expose a model loop if M3 keeps trying to write oversized invalid JSON.

### ToolResultPruningMiddleware

Current role-specific limits:

| Scope | keep_last_n | max_chars | recent_max_chars | max_tool_call_arg_chars |
|---|---:|---:|---:|---:|
| planner | 4 | 500 | 10000 | 1600 |
| researcher | 4 | 700 | 12000 | 1800 |
| orchestrator | 8 | 1200 | 40000 | 5000 |

This means planner/researcher loops stay compact. M3's large context is mainly reserved for final synthesis through evidence files and the orchestrator context.

Important nuance:

This middleware trims historical tool call arguments before future model calls. It does not truncate the live `write_file.content` before the tool runs. So if a file is invalid or missing `queries`, that usually means the model emitted invalid/incomplete content, not that the runtime cut off the live write.

## Deep Workflow Step by Step

### Step 0. Preload approved plan or structured lesson plan

Before the deep agent runs, `DeepResearcherAgent._inject_approved_plan_if_available()` checks:

- If `/plan.json` or `/shared/plan.json` already exists, keep it.
- If the user query has structured lesson fields like `Exact lesson topic:` and `Final debate motion:`, build a deterministic topic-first plan.
- If the clarifier produced an approved plan, build `/shared/plan.json` from it.
- If the approved plan looks generic, ignore it and let the deep planner run.

Important code:

- `_extract_structured_lesson_scope`
- `_build_structured_lesson_plan_json`
- `_build_plan_json_from_approved_context`
- `_is_generic_approved_plan`

### Step 1. Orchestrator starts

Prompt template:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/orchestrator.j2`

The orchestrator is told:

```text
You are never a direct-answer chatbot in this workflow. Every invocation, even one that looks simple, must run the deep research process:
- Track progress with `write_todos`.
- Use an existing `/shared/plan.json`, an approved plan, or the `planner-agent`.
- Delegate factual gathering to `researcher-agent`.
- Ensure at least one search/source-capturing tool result has been used before final writing.
- Never answer from memory, never claim web research is unnecessary, and never invent source URLs.
```

The orchestrator either reads an existing plan or delegates to planner-agent.

### Step 2. Planner-agent creates `/shared/plan.json`

Prompt template:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/planner.j2`

The planner is supposed to:

- Analyze the user request.
- Anticipate claim types.
- Inventory named entities.
- Create a source strategy.
- Create a budget plan.
- Create a TOC.
- Create constraints.
- Create search queries.
- Write `/shared/plan.json`.

The planner prompt explicitly requires a final JSON structure with:

- `task_analysis`
- `report_title`
- `report_toc`
- `fact_ledger_targets`
- `constraints`
- `queries`

Important excerpt:

```text
Every query must be bound to one or more target claims. The TOC organizes the final report; claim-bound queries drive evidence gathering.
```

Actual weakness observed:

M3 can produce very large planner JSON, hit an 8192-output-token cap, and stop mid-string. The validator then rejects it. If the planner repeats the same oversized JSON, it loops in planner repair rather than moving to research.

### Step 3. Researcher-agent runs plan queries

Prompt template:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/researcher.j2`

The orchestrator reads `/shared/plan.json`, then sends researcher-agent tasks. Each task should include:

- query text
- target claims
- source-class expectations
- relevant constraints

The researcher is supposed to use search/source tools, then write:

- `/shared/[query_topic].txt`
- `/shared/claims/claims_[short_query_topic].json`
- `/shared/section_briefs/[short_query_topic].md`
- `/shared/extracts/[short_query_topic].json`
- optional `/shared/fact_ledger_[topic].json`

Important excerpt:

```text
Your assigned query may include `target_claims` from `/shared/plan.json`. If so, your primary job is to resolve those claims, not merely to write prose.
```

Researcher failures this prompt tries to prevent:

- hallucinated specs
- derivative-blog statistics
- single-source dominance
- motion-specific drift in lesson prompts
- searching after budget exhaustion

### Step 4. Source registry captures URLs

Middleware:

- `SourceRegistryMiddleware`

It captures URLs only from configured source/search tool results. It ignores internal tools such as `think`.

The orchestrator later calls `get_verified_sources`, which renders:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/source_registry.j2`

The source registry includes source classes where available:

- `first_party`
- `primary_issuer`
- `academic`
- `authoritative_third_party`
- `trade_press`
- `vendor_marketing`
- `content_marketing`
- `forum`
- `unknown`

### Step 5. Runtime merges structured artifacts

The agent merges:

- `/shared/claims/claims_*.json` into `/shared/claim_table.json`
- `/shared/fact_ledger_*.json` into `/shared/fact_ledger.json`
- `/shared/extracts/*.json` plus claim table into `/shared/evidence_packet.json`

Relevant code:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/claim_table.py`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/fact_ledger.py`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/evidence_packet.py`
- `_merge_structured_research_artifacts_into_result` in `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/agent.py`

This is the M3 large-context path. The goal is not to make every subagent run with huge context; it is to give final synthesis a structured evidence packet.

### Step 6. Orchestrator synthesizes

The orchestrator must:

1. `ls /shared/`
2. read all files
3. read `/shared/evidence_packet.json` if present
4. read `/shared/claim_table.json`
5. read `/shared/fact_ledger.json` when entity targets exist
6. call `get_verified_sources`
7. optionally write `/shared/consolidated_findings.md`
8. write `/report.md`

The prompt explicitly says:

```text
Write from the claim table first. The notes explain context, but the claim table controls which factual claims can be stated and how strongly.
```

### Step 7. Report validation and recovery

After DeepAgents returns, code checks the result:

- `DeepResearcherAgent._is_report_complete`
- `_extract_report_content_from_result`
- `_compile_report_from_artifacts`
- `verify_citations`
- `sanitize_report`
- `report_matches_request_scope`

A complete report must:

- be longer than `_MIN_REPORT_LENGTH` (currently `1500`)
- have markdown section headers
- have a sources/references section
- not be only provider thinking blocks
- not be a known model-failure report
- match the requested scope
- have sources captured by source registry

If the final output is too short or thinking-only, `_compile_report_from_artifacts` may build a report from saved artifacts. That is a fallback path; it is meant to avoid total loss, but it should not be the main happy path.

### Step 8. API exposes progress and report

Important endpoints:

- `GET /health`
- `POST /v1/jobs/async/submit`
- `GET /v1/jobs/async/job/{job_id}`
- `GET /v1/jobs/async/job/{job_id}/state`
- `GET /v1/jobs/async/job/{job_id}/stream`
- `GET /v1/jobs/async/job/{job_id}/report`
- `GET /v1/jobs/async/job/{job_id}/report?format=markdown`
- `POST /v1/jobs/async/job/{job_id}/cancel`
- `POST /v1/jobs/async/job/{job_id}/resume`

If the report is not ready, the markdown report endpoint returns `202 Accepted` with report-ready fields.

## Latest Observed Failure: Planner JSON Loop

Observed live Oracle job:

- `lesson-education-as-a-tool-to-prepare-f-1p7sotx`
- status at audit time: `running`
- created: `2026-06-01 11:24:20 UTC`
- events at audit time: `242`
- last meaningful behavior: planner-agent repeatedly tries to write `/shared/plan.json`, hits JSON errors, and emits repair thoughts.

Observed database/event facts:

```text
job_id: lesson-education-as-a-tool-to-prepare-f-1p7sotx
status: running
events: 242
event range: 2026-06-01 11:24:20 UTC to 2026-06-01 11:34:51 UTC
```

Observed validator rejection:

```text
Rejected incomplete /shared/plan.json write:
content is not valid JSON:
Unterminated string starting at line 504 column 23
```

Observed planner thought chunks:

```text
I keep getting JSON errors from nested quotes. Let me rewrite the file carefully...
There's another error at line 504...
```

What this means:

1. The planner produced an enormous JSON plan.
2. The output hit the model/tool generation boundary around 8192 output tokens.
3. The JSON ended mid-string.
4. `PlanFileValidationMiddleware` correctly rejected the bad file.
5. The planner responded by trying to repair the same giant JSON, not by simplifying.
6. The job then sat in planner retry/heartbeat mode.

This is a real workflow problem, but it is not the same as "search is slow." It is a planning-output shape problem.

The core fix should be architectural:

- Do not ask M3 planner to hand-write a 500-line JSON file.
- Use a smaller, stricter plan schema.
- Let Python deterministically expand a compact plan into the full internal schema.
- Or provide a dedicated `write_plan` tool that accepts typed fields and serializes JSON safely.

## Current Prompt Sizes

These are line counts as of this audit:

| Prompt | Lines |
|---|---:|
| `orchestrator.j2` | 301 |
| `planner.j2` | 279 |
| `researcher.j2` | 262 |
| `source_registry.j2` | 21 |
| `plan_generation.j2` | 130 |
| `research_clarification.j2` | 171 |
| `intent_classification.j2` | 59 |
| `shallow_researcher/prompts/researcher.j2` | 71 |

This matters because M3 may have a huge context window, but short/medium-context speed and instruction reliability can still degrade when we ask it to write long structured JSON by hand.

## Prompt Template Inventory

This section records the actual prompt templates used. The full source of truth is the file path listed for each prompt. Do not copy prompt text from this document when changing behavior; edit the template file directly.

### Intent Classification Template

Path:

`/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/chat_researcher/prompts/intent_classification.j2`

Key actual instructions:

```text
Classify the query as "meta" or "research".
Extract proper nouns from the query that name specific products, models, tools, companies, people, statutes, organizations, places, or other identifiable entities.
IF INTENT IS "RESEARCH": Determine the depth.
Entity density override: If the query names 3 or more specific products, models, tools, companies, people, statutes, or organizations, choose "deep".
```

Output schema:

```json
{
  "intent": "meta|research",
  "meta_response": "string|null",
  "research_depth": "shallow|deep|null",
  "depth_reasoning": "string|null",
  "named_entities": [
    {"name": "Specific Entity Name", "type": "product|model|tool|company|person|statute|organization|place|other"}
  ]
}
```

### Clarification Template

Path:

`/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/clarifier/prompts/research_clarification.j2`

Key actual instructions:

```text
TOP PRIORITY: Do not over-clarify.
Clarify only when missing context would materially change the research.
Only ask for clarification when the question is genuinely ambiguous, the scope is so broad narrowing would improve research, or critical context is missing.
```

Output schema:

```json
{
  "needs_clarification": true,
  "clarification_question": "..."
}
```

or:

```json
{
  "needs_clarification": false,
  "clarification_question": null
}
```

### User-Facing Plan Preview Template

Path:

`/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/clarifier/prompts/plan_generation.j2`

Purpose:

- Creates the plan the user sees and approves in the UI.
- This is intentionally lightweight.
- It should preserve explicit dimensions from the user query.

Important actual rule:

```text
If the user already provided explicit report dimensions, preserve them in the user-facing plan. Do not replace a detailed request with generic buckets.
```

Output schema:

```json
{
  "title": "Topic-Specific Research Title",
  "sections": [
    "Topic-Specific Section One"
  ]
}
```

### Deep Planner Template

Path:

`/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/planner.j2`

Purpose:

- Creates the executable internal `/shared/plan.json`.
- Much more detailed than the UI plan preview.

Required plan output shape:

```json
{
  "task_analysis": {},
  "report_title": "...",
  "report_toc": [],
  "fact_ledger_targets": {},
  "constraints": [],
  "queries": []
}
```

Important actual rule:

```text
Every query must be bound to one or more target claims.
The TOC organizes the final report; claim-bound queries drive evidence gathering.
```

Current risk:

The template asks the model to produce many nested fields and examples. For lesson prompts, the orchestrator can also ask the planner to produce user-specific fields such as `topic`, `motion`, `toc`, `sub_areas`, and `fact_ledger_targets`. This can create a very large JSON object and trigger mid-string truncation.

### Deep Researcher Template

Path:

`/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/researcher.j2`

Purpose:

- Executes one assigned query or packed query group.
- Uses search tools.
- Writes notes, claim fragments, section briefs, and extracts.

Important actual rule:

```text
Your assigned query may include `target_claims` from `/shared/plan.json`.
If so, your primary job is to resolve those claims, not merely to write prose.
```

Required files:

- `/shared/[query_topic_x].txt`
- `/shared/claims/claims_[short_query_topic].json`
- `/shared/section_briefs/[short_query_topic].md`
- `/shared/extracts/[short_query_topic].json`

### Deep Orchestrator Template

Path:

`/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/orchestrator.j2`

Purpose:

- Coordinates planning, research, synthesis, report writing, final verification.

Important actual rule:

```text
You are never a direct-answer chatbot in this workflow.
Every invocation, even one that looks simple, must run the deep research process.
```

M3-specific instruction:

```text
This workflow is optimized for M3's large context window.
Researchers write notes, claim fragments, and extract fragments; the runtime merges those into `/shared/claim_table.json` and `/shared/evidence_packet.json`.
```

### Source Registry Template

Path:

`/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/source_registry.j2`

Purpose:

- Presents verified tool-returned URLs to the final writer.
- Includes source class annotations.

Important actual rule:

```text
Use ONLY these sources when writing the final report.
Each has been verified as a real tool result and annotated with a source class.
```

### Shallow Researcher Template

Path:

`/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/shallow_researcher/prompts/researcher.j2`

Purpose:

- Rapid citation-backed answer.
- Uses available tools directly.
- Does not run the full planner/researcher/orchestrator deep workflow.

Important actual rule:

```text
Loop Prevention: Max 2 calls per tool. If results are empty, change your search string once or move to synthesis.
```

## Failure Modes to Watch

### 1. Planner JSON explosion

Symptoms:

- Planner emits thousands of tokens.
- UI shows repeated thoughts about JSON validation.
- `/shared/plan.json` write is rejected.
- Job keeps heartbeating with no research starting.

Cause:

- Planner is being asked to hand-write too much nested JSON.
- M3 reaches output token cap or makes an escaping error.

Best fix:

- Replace raw `write_file('/shared/plan.json')` with a typed `write_plan` tool or a compact schema that Python expands.

### 2. UI timestamp reset after refresh

Symptoms:

- Tool calls and thoughts all show current time after refresh.

Likely cause:

- Frontend store actions default to `new Date()` when rehydrating activity, even though SSE events have persisted timestamps.

Best fix:

- Thread backend `timestamp` into every `addDeepResearch*` store action.
- For historical events, use `event_data.timestamp` or `job_events.created_at`, never client receipt time.

### 3. M3 slower than M2.7 for medium tasks

Symptoms:

- Planner/researcher calls are slower.
- Long-ish outputs hit 8192 output token caps.

Likely cause:

- M3 may be better for large synthesis but slower for short/medium planning/tool loops.
- Current workflow still uses the same model family across roles unless config routes differently.

Possible improvement:

- Use M2.7/high-speed or a smaller model for clarifier, UI plan preview, and planner JSON.
- Use M3 for final synthesis over evidence packet.

### 4. Search budget exhaustion loops

Symptoms:

- Model says budget exhausted, then tries another search.

Current guard:

- `SearchBudgetExhaustionRepairMiddleware`
- `ToolBudgetMiddleware`

Known remaining issue:

- If the model loops without issuing search calls, budget middleware cannot stop it; a separate stuck-phase watchdog is needed.

### 5. Thinking-only final output

Symptoms:

- Report view shows provider thinking blocks or empty final report.

Current guards:

- `ThinkingOnlyRepairMiddleware`
- `_compile_report_from_artifacts`
- final report completeness checks

## Recommended Next Fixes

Highest-value next fixes, in order:

1. Add a deterministic `write_plan` tool.
   - It should accept typed fields or a compact plan object.
   - Python serializes valid `/shared/plan.json`.
   - This directly fixes the latest planner JSON loop.

2. Add planner retry compression.
   - If `PLAN_FILE_VALIDATION_FAILED` happens once, the repair prompt should forbid full verbose schema expansion and require a compact plan under a hard field/character budget.

3. Route planner/clarifier to faster short-context model if available.
   - M3 is useful for final synthesis.
   - Medium JSON-writing loops may be better on a faster model or with deterministic tooling.

4. Fix frontend event timestamps.
   - Preserve backend event timestamps on all reconstructed thinking/tool/file/activity objects.

5. Add phase-stuck watchdog.
   - If a job stays in planner-agent for more than a threshold with repeated validation errors, fail fast with actionable error or switch to deterministic plan builder.

## Useful Diagnosis Commands

Check app2 health from Oracle:

```bash
ssh oracle 'curl -fsS http://127.0.0.1:9000/health'
```

List recent jobs:

```bash
ssh oracle "cd /opt/stacks/app2/deep-research && docker compose --env-file deploy/.env -f deploy/compose/docker-compose.app2.yaml exec -T postgres psql -U aiq -d aiq_jobs -P pager=off -c \"select job_id,status,created_at,updated_at,left(coalesce(error,''),180) as error,length(coalesce(output,'')) as output_len from job_info order by created_at desc limit 8;\""
```

Inspect recent events for a job:

```bash
ssh oracle "cd /opt/stacks/app2/deep-research && docker compose --env-file deploy/.env -f deploy/compose/docker-compose.app2.yaml exec -T postgres psql -U aiq -d aiq_jobs -P pager=off -c \"select id,event_type,created_at,left(event_data,320) as event from job_events where job_id='JOB_ID_HERE' order by id desc limit 35;\""
```

Search backend logs for planner validation failures:

```bash
ssh oracle 'docker logs --since=45m app2-aiq-agent 2>&1 | grep -E "PLAN_FILE_VALIDATION_FAILED|Rejected incomplete|JSON validation|Unterminated|string|planner-agent|plan.json" | tail -n 200'
```

Verify live pruning limits inside the backend container:

```bash
ssh oracle 'cd /opt/stacks/app2/deep-research && docker compose --env-file deploy/.env -f deploy/compose/docker-compose.app2.yaml exec -T aiq-agent python - <<PY
from aiq_agent.common.research_depth import RESEARCH_DEPTH_CONFIGS
from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent
print("parallel.deeper", RESEARCH_DEPTH_CONFIGS["deeper"].max_parallel_researcher_tasks)
print("parallel.deep", RESEARCH_DEPTH_CONFIGS["deep"].max_parallel_researcher_tasks)
for scope in ["planner", "researcher", "orchestrator"]:
    m = DeepResearcherAgent._tool_result_pruning_middleware_for_scope(scope)
    print(scope, m.keep_last_n, m.max_chars, m.recent_max_chars, m.max_tool_call_arg_chars)
PY'
```
