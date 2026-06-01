# MiniMax M3 Planner Stability Fix Plan

Last updated: 2026-06-01 19:50 HKT
Project root: `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research`
Primary target: local repo first, then Oracle app2 at `/opt/stacks/app2/deep-research`

This plan merges the three external recommendations with the current repo state. It is written as an implementation plan, not a design wish list.

## Executive Diagnosis

The current deep research failure is primarily a planning serialization failure.

The system upgraded to MiniMax M3, but the planner phase still behaves like a model should hand-author a large, strict `/shared/plan.json` file through the generic DeepAgents `write_file` tool. On complex requests, especially debate/lesson or high-dimensional deep research prompts, the planner tries to produce a 400-700 line JSON artifact containing:

- task analysis
- claim profile
- source strategy
- entity inventory
- TOC
- constraints
- fact ledger targets
- queries with target claims
- output style

That output is currently capped at `8192` tokens in `configs/config_cli_minimax_ddgs.yml`:

```yaml
minimax_m3_planner_llm:
  max_tokens: 8192
```

The latest failing jobs show the planner repeatedly hitting that ceiling, producing invalid JSON such as:

```text
Unterminated string starting at line 504 column 23
```

`PlanFileValidationMiddleware` correctly rejects the bad plan, but the repair instruction asks the same model to rewrite compact JSON through the same fragile path. That creates a loop.

The problem is not that M3 is too weak. It is that the workflow is using M3 in a brittle way:

1. It asks the model to serialize raw JSON by hand.
2. It combines multiple schemas in one generated file.
3. It lets a failed planner retry the same oversized shape.
4. It uses M3's large context in the planner/research loops, where shorter contexts and structured tools are more reliable.
5. It reserves too little output budget for the current planner artifact while simultaneously encouraging verbose planning.

## What Is Already Good

The system already has several strong pieces worth preserving:

- Role-based LLM routing exists through `LLMProvider` and `LLMRole`.
- DeepAgents virtual filesystem already persists `/shared/*` inside graph state.
- The code already preloads deterministic plans for some structured lesson and stock-screen cases.
- `PlanFileValidationMiddleware` is useful as a backstop.
- The M3 evidence-packet path is directionally right: use M3's large context in final synthesis over merged evidence, not in every subagent loop.
- Source classification, source quality gates, claim fragments, fact ledgers, and evidence packets exist or are partially wired.
- Research depth profiles and scoped budgets exist.

So the fix should not rebuild the whole app. It should harden the planning gate and simplify what the planner must emit.

## Guiding Principles

1. The model should not hand-write large JSON.
2. The planner should emit the minimum executable plan, not a giant report blueprint.
3. Python should own serialization, validation, schema expansion, and repair fallback.
4. M3 should be reserved for final synthesis and truly long-context evidence reading.
5. Planner/researcher loops should stay compact.
6. A validation failure should trigger a different path, not the same attempt again.
7. User-facing plan quality and internal execution plan quality should be separated.
8. Existing running jobs should be protected when possible; rebuild only after current jobs finish or are explicitly stopped.

## Phase 0: Immediate Containment

Goal: reduce active failures before the deeper structural fix lands.

### 0.1 Increase Planner Output Budget

File:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/configs/config_cli_minimax_ddgs.yml`

Change:

```yaml
minimax_m3_planner_llm:
  max_tokens: 32768
  request_timeout: 480
```

Do not treat this as the real fix. It only prevents easy truncation. Raw JSON writing can still fail on escaping and schema confusion.

Acceptance:

- Planner no longer regularly shows `8192 out` in traces.
- Any remaining plan failure is no longer mostly truncation.

Risk:

- Slightly slower bad planner attempts if structural fix is not in place.

### 0.2 Add a Better Validation Failure Message

File:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/custom_middleware.py`

Current behavior:

- `PlanFileValidationMiddleware` rejects invalid JSON and tells the model to rewrite compact JSON.

Temporary improvement:

- Update the message to explicitly say the previous attempt was too large or malformed.
- Cap repair shape:
  - maximum 5 top-level sections
  - maximum 8 queries for `deeper`
  - maximum 12 queries for `deep`
  - no prose paragraphs inside JSON values

This is still not enough by itself, but it reduces repeated giant repairs.

Acceptance:

- On validation failure, the next planner attempt is visibly shorter.

## Phase 1: Add a Typed `write_plan` Tool

Goal: eliminate malformed `/shared/plan.json` at the source.

This is the highest-value fix. The planner should call a typed tool whose arguments are structured fields. Python then serializes and writes valid JSON to the DeepAgents virtual filesystem.

### 1.1 Create Plan Schema Models

New file:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/plan_schema.py`

Define Pydantic v2 models:

- `PlanTargetClaim`
- `PlanQuery`
- `PlanTocItem`
- `PlanConstraint`
- `PlanOutputStyle`
- `PlanTaskAnalysis`
- `WritePlanInput`

Keep the tool input compact. Do not require every currently optional field in the model-facing input.

Recommended tool input:

```python
class WritePlanInput(BaseModel):
    report_title: str
    report_toc: list[PlanTocItem]
    queries: list[PlanQuery]
    constraints: list[PlanConstraint | str]
    output_style: PlanOutputStyle | dict[str, Any] | None = None
    task_analysis: PlanTaskAnalysis | dict[str, Any] | None = None
    fact_ledger_targets: dict[str, list[dict[str, Any]]] | None = None
```

Python should fill defaults:

- `task_analysis.source_strategy`
- `task_analysis.claim_profile`
- `budget_profile`
- `output_style.mode`
- `output_style.target`
- `output_style.avoid`

### 1.2 Implement Tool Using DeepAgents StateBackend

New file:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/plan_tools.py`

Implementation approach:

- Use `langchain_core.tools.StructuredTool` or `@tool(args_schema=WritePlanInput)`.
- Accept a `ToolRuntime`.
- Use `deepagents.backends.StateBackend().write("/shared/plan.json", json.dumps(plan, indent=2))`.
- Also write `/plan.json` for compatibility if needed.
- If `/shared/plan.json` already exists, read/edit or return a clear message telling the orchestrator to use the existing plan.

Why this works:

- DeepAgents' own `write_file` uses `StateBackend` internally.
- `StateBackend` queues updates through LangGraph config, so this can safely mutate the same virtual filesystem.

Important:

- The model must never pass a giant raw JSON string to this tool.
- The model passes typed arguments; Python owns JSON serialization.

Acceptance:

- A planner can create `/shared/plan.json` without calling `write_file`.
- Nested quotes in section titles or hook questions cannot break JSON.
- Tests prove `/shared/plan.json` validates after `write_plan`.

### 1.3 Register Tool for Planner and Orchestrator

File:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/agent.py`

Change:

- Add `write_plan` to `self.orchestrator_tools`.
- Ensure `self.all_tools` includes it.
- The planner-agent already receives `self.all_tools`, so it will see the tool.

Careful point:

- `PlanFileValidationMiddleware` only intercepts generic `write_file` writes. It can remain as a backstop.
- The new tool should validate before writing and return a concise error if the typed fields are incomplete.

Acceptance:

- Planner traces show `write_plan`, not `write_file`, for plan creation.
- Existing `read_file("/shared/plan.json")` continues to work for orchestrator.

## Phase 2: Rewrite Planner Prompt Around `write_plan`

Goal: stop asking the planner to output giant JSON.

File:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/planner.j2`

Current problem:

- The prompt gives a huge JSON schema and ends with:

```text
You MUST write the plan using the write_file filesystem tool with file_path="/shared/plan.json"
```

Change:

- Replace raw JSON output instructions with typed tool instructions:

```text
You MUST create the plan by calling write_plan.
Do not call write_file for /shared/plan.json.
Do not hand-write JSON.
```

Keep:

- claim anticipation
- source strategy
- source diversity
- numeric claim verification
- lesson topic vs motion scope rule
- budget guidance

Remove or compress:

- the full JSON schema example
- duplicate claim_profile list variants
- verbose field-by-field schema text
- anything that encourages paragraphs inside JSON values

New planner behavior:

1. optionally use 0-2 planning searches
2. think briefly
3. call `write_plan`
4. return a human-readable summary of plan title, sections, and query count

Acceptance:

- `planner.j2` becomes materially shorter.
- The planner no longer emits a massive code-fenced JSON blob.
- Planner output does not hit output limit in normal cases.

## Phase 3: Collapse Dual Schema and Expand in Python

Goal: make lesson/debate and other structured requests deterministic.

Files:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/agent.py`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/plan_schema.py`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/plan_tools.py`

Current partial support:

- `_extract_structured_lesson_scope`
- `_build_structured_lesson_plan_json`
- `_build_plan_json_from_approved_context`
- `_is_generic_approved_plan`

Problem:

- When the approved plan is weak or generic, the system drops into planner-agent.
- The planner then tries to merge lesson-specific fields with the general plan schema by hand.

Fix:

- Create one canonical compact internal representation.
- For structured lesson prompts, Python should build the skeleton:
  - report title
  - mandatory lesson sections
  - broad topic queries first
  - examples/case-studies second
  - motion bridge last
  - output style with `topic_anchor` and `motion_anchor`
- The model only supplies improved query text or extra constraints through `write_plan`.

Better path:

1. Clarifier creates user-facing plan.
2. If that plan is generic but the original query is rich, do not use generic sections.
3. Python derives a query-specific initial plan from the original prompt.
4. Planner-agent may refine via `write_plan`, but cannot rebuild the whole schema from scratch.

Acceptance:

- A prompt like "Conduct a comprehensive deep research report on top 10 AI use cases in 2026..." creates a plan with sections matching the requested dimensions, not:
  - `Landscape`
  - `Recent Evidence and Signals`
  - `Capability Gaps`
  - `Adoption Risks and Recommendations`
- A lesson prompt keeps the broad topic primary and the debate motion bounded.

## Phase 4: Add Planner Circuit Breaker

Goal: never let planner validation loop forever.

Files:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/custom_middleware.py`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/agent.py`

### 4.1 Track Plan Validation Failures Per Session

Use a context variable similar to the existing tool-budget counters:

- `_session_plan_validation_failures`

Increment when `PlanFileValidationMiddleware` rejects `/shared/plan.json`.

### 4.2 Escalate After Two Failures

After 2 failures:

- Stop saying "try again with compact JSON."
- Return a message instructing the orchestrator to:
  - use deterministic plan builder, or
  - call `write_plan` with a minimal plan, or
  - synthesize a compact skeleton without further planner retries.

### 4.3 Optional Runner-Level Stuck Phase Detection

File:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/aiq_api/src/aiq_api/jobs/runner.py`

Add status detection:

- if same job emits repeated planner JSON validation errors
- or planner phase exceeds a threshold with no files/researcher events
- emit a job event:

```json
{
  "type": "quality.warning",
  "phase": "planning",
  "message": "Planner validation loop detected; switching to deterministic compact plan path."
}
```

Acceptance:

- No job can sit in planner JSON repair loop indefinitely.
- The user gets a visible warning rather than silent heartbeats.

## Phase 5: Model Routing and M3 Usage Policy

Goal: use M3 where it helps and avoid it where it causes slow loops.

File:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/configs/config_cli_minimax_ddgs.yml`

Current:

- intent classifier: `minimax_m3_planner_llm`
- clarifier: `minimax_m3_planner_llm`
- planner: `minimax_m3_planner_llm`
- researcher: `minimax_m3_llm`
- synthesis: `minimax_m3_synthesis_llm`

Recommended:

- Keep M3 synthesis for final report and evidence packet reading.
- Keep M3 researcher if its quality is good, but maintain compact pruning.
- For planner and clarifier:
  - prefer a faster short-context model if available through env, or
  - keep M3 but require typed tools and shorter output.

Practical config:

```yaml
minimax_m3_planner_llm:
  max_tokens: 32768
  request_timeout: 480
```

Optional env-based route:

```yaml
model_name: ${MINIMAX_PLANNER_MODEL:-MiniMax-M3}
```

This already exists, so if MiniMax M2.7 remains available, Oracle/local can set:

```text
MINIMAX_PLANNER_MODEL=<fast planner model>
```

Do not depend on model routing as the primary fix. The typed plan tool is still required.

Acceptance:

- Planner traces are shorter.
- Synthesis still uses M3.
- Model labels in UI match actual configured model names.

## Phase 6: User-Facing Plan Quality

Goal: prevent every clear request from producing a fallback-looking approval plan.

Files:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/clarifier/prompts/plan_generation.j2`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/clarifier/agent.py`

Current issue:

- The user-facing plan can collapse to generic headings, for example:
  - `Landscape`
  - `Recent Evidence and Signals`
  - `Capability Gaps`
  - `Adoption Risks and Recommendations`

Fix:

- Treat generic plan preview as invalid when the original query has explicit structure.
- If the user gives dimensions, lists, rankings, output format, or numbered requirements, the plan preview must preserve those dimensions.
- Add a deterministic plan-preview repair:
  - parse obvious numbered requirements
  - preserve top-level user wording
  - create a title from the core noun phrase, not the first truncated words

For the AI use case prompt, expected preview:

```text
Title: Top 10 Highest-Value AI Use Cases in 2026

Sections:
1. Executive Summary and Ranking Method
2. Ranked Top 10 AI Use Cases
3. Use Case Evidence: ROI, Examples, Maturity, Enablers, Barriers, Beneficiaries
4. Cross-Use-Case Patterns and Adoption Constraints
5. 2027-2028 Outlook for AI Value Creation
```

For the curiosity engine prompt, expected preview:

```text
Title: Building AI Lifelong Curiosity Engines

Sections:
1. Motivation Science and Lifelong Learning Foundations
2. Interest Modeling and User Profiling
3. Long-Tail Discovery and Recommendation Architecture
4. Teaching, Scaffolding, and Adaptive Dialogue
5. Knowledge, Content Verification, and Evaluation Stack
```

Acceptance:

- Generic fallback sections are not shown for rich prompts.
- The plan preview title is never just a truncated first clause.
- Approval plan and internal `/shared/plan.json` are aligned but not identical.

## Phase 7: Search Budget and Synthesis Transition

Goal: when budget is exhausted, move to synthesis instead of searching forever.

Files:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/custom_middleware.py`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/researcher.j2`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/prompts/orchestrator.j2`

Already present:

- `SearchBudgetExhaustionRepairMiddleware`
- `ToolBudgetMiddleware`
- researcher prompt says to write notes when budget is exhausted.

Likely problem:

- the model may receive budget exhaustion, think about more searches, then retry variants.
- orchestrator may launch more researcher tasks because it does not see enough files or the plan implies more coverage.

Fix:

- When any search family is exhausted:
  - mark that scope exhausted in state
  - force next researcher action to `write_file`
  - tell orchestrator that this branch is done
- Add a stronger event:

```text
research.branch_exhausted
```

- Orchestrator must not launch another researcher for the same target claim after branch exhaustion unless user upgrades budget.

Future UI feature:

- show `Budget exhausted: synthesize now` and `Upgrade this job to deep budget`
- if no action after 2 minutes, synthesize

Acceptance:

- Exhaustion leads to notes and synthesis, not repeated search.
- User can see budget state in UI.

## Phase 8: Frontend Timestamp and History Repair

Goal: make the UI history faithful after refresh.

Files:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/features/chat/store.ts`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/ui/src/adapters/api/deep-research-client.ts`
- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/frontends/aiq_api/src/aiq_api/jobs/callbacks.py`

Current symptom:

- after refresh, historical thoughts/tool calls show current time.

Likely cause:

- frontend store actions default to `new Date()` during rehydration.

Fix:

- Thread backend timestamps through parsed SSE events.
- Every `addDeepResearch*` action should preserve `event.timestamp`, `event_data.timestamp`, or `job_events.created_at`.
- Only use `new Date()` for genuinely live client-created events with no server timestamp.

Acceptance:

- Refreshing a running job preserves original event times.
- Tool calls no longer all show the current refresh time.

## Phase 9: Tests

Add tests in this order.

### Unit Tests

New:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/tests/aiq_agent/agents/deep_researcher/test_plan_tools.py`

Cover:

- `write_plan` creates valid `/shared/plan.json`.
- nested quotes in titles/queries are escaped safely.
- missing queries are rejected before write.
- existing plan behavior is deterministic.
- output includes all validator-required top-level keys.

Existing file to extend:

- `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/tests/aiq_agent/agents/deep_researcher/test_custom_middleware.py`

Cover:

- validation failure count
- compact repair message
- escalation after repeated failures

### Prompt Tests

Add tests that render:

- planner prompt includes `write_plan`
- planner prompt does not instruct `write_file` for `/shared/plan.json`
- plan_generation rejects generic fallback sections for rich prompts

### Integration Smoke Tests

Run local or app2 API jobs:

1. Rich AI use-case report:
   - verify user-facing plan is specific
   - verify `/shared/plan.json` has queries
   - verify planner does not loop

2. Debate lesson prompt:
   - verify topic-first plan
   - verify motion bridge is bounded
   - verify no drift into only one subtopic

3. Curiosity engine prompt:
   - verify frontend plan resembles a real research plan
   - verify planning completes

4. Shallow research:
   - verify NotebookLM pathway is removed
   - verify shallow uses normal configured tools

## Phase 10: Deployment Order

Recommended safe order:

1. Commit local plan/tool/schema changes.
2. Run unit tests locally.
3. Run one local backend smoke job.
4. Wait for active Oracle app2 jobs to finish unless user explicitly says stop.
5. Sync code to Oracle with the existing app2 sync script.
6. Rebuild only `aiq-agent` first.
7. Verify:
   - `curl http://127.0.0.1:9000/health` on VPS
   - app2 UI loads
   - a tiny shallow job works
   - one deep planning-only smoke reaches research phase
8. Rebuild frontend only if Phase 6 or Phase 8 UI changes are included.
9. Clean old Oracle build artifacts after the new build is healthy.

## Acceptance Criteria for the Whole Fix

The fix is complete only when all of these are true:

- Planner no longer writes `/shared/plan.json` through raw `write_file`.
- Planner calls `write_plan` or uses a deterministic preloaded plan.
- `/shared/plan.json` always contains:
  - `task_analysis`
  - `report_title`
  - `report_toc`
  - `constraints`
  - `output_style`
  - `queries`
- Complex prompts do not get stuck in planner-agent.
- User-facing plan preview is specific for structured prompts.
- Search budget exhaustion moves branches toward notes/synthesis.
- Timestamps survive refresh.
- M3 remains the final synthesis model.
- Local and Oracle app2 configs agree.

## Highest-Value First Implementation Slice

If we want the smallest set of changes with the biggest benefit, do this:

1. Raise planner `max_tokens` to `32768`.
2. Add `write_plan` typed tool.
3. Rewrite planner prompt to call `write_plan` and remove raw JSON schema dump.
4. Add repeated plan-validation circuit breaker.
5. Fix user-facing plan fallback sections.
6. Deploy backend to app2 and run one deep smoke test.

This slice directly addresses the current catastrophic loop without waiting on broader UI improvements.

## What Not To Do

Do not only raise `max_tokens` and call it fixed. That reduces truncation but keeps raw JSON fragility.

Do not remove `PlanFileValidationMiddleware`. It is correctly catching bad state writes.

Do not use M3's 1M context as permission to pass huge intermediate state to every loop. Keep M3's large context for final evidence synthesis.

Do not let planner failures silently fallback to generic plans. Generic plans are a quality failure when the user prompt is specific.

Do not rebuild Oracle while a valuable long job is running unless the user explicitly stops it or accepts interruption.
