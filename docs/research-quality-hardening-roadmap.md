# Research Quality Hardening Roadmap

This document turns the latest research-quality analysis into a concrete roadmap for this repo.

It complements:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/docs/deep-research-process-audit.md
```

The core conclusion is simple: the next major quality jump will not come from merely increasing search budget. It will come from making unsupported generation structurally difficult.

## Why The Priority Changes

The prior process audit correctly identified late quality gates, search-budget exhaustion, generic plan previews, source-quality weakness, and final synthesis pressure as the main pain points.

The new analysis adds one sharper point: in deep-research systems, the biggest failure is often not that the model misunderstands the task. The bigger failure is that the final report generation stage turns incomplete or weakly integrated evidence into polished, confident prose.

That means our quality roadmap should prioritize generation hardening:

1. Claim-table-driven generation
2. Section-level synthesis
3. Citation reconciliation as a separate pass
4. In-flight source and budget steering
5. Plan preview repair
6. Quality warnings instead of report loss

This does not mean search is unimportant. It means search only becomes valuable when the final writer is forced to use the evidence honestly.

## Current Repo Position

The repo already has pieces of the right architecture:

- Source classification:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/source_classification.py
```

- Source quality gates:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/source_quality_gates.py
```

- Claim table schema:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/claim_table.py
```

- Scrape artifact storage:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/scrape_artifacts.py
```

- Search budget and repair middleware:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/agents/deep_researcher/custom_middleware.py
```

- Depth-specific budget config:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/src/aiq_agent/common/research_depth.py
```

The remaining gap is enforcement. Several quality mechanisms exist as prompts or post-run checks. They need to become deterministic substrate.

## The Real Bottleneck

The most fragile stage is final synthesis.

Research collection can be parallelized and audited. Final synthesis must compress many notes, preserve scope, obey citation rules, avoid generic padding, avoid internal workflow leakage, handle uncertainty, and write a coherent report.

That is exactly where MiniMax M3 is most likely to fail because it is being asked to do too many high-discipline tasks at once.

The correct architectural move is to remove discipline from the prompt and put it into artifacts:

- Researchers resolve claims.
- Python validates and merges claim fragments.
- Section writers summarize bounded slices.
- Final writer writes from claim table plus section briefs.
- Citation pass reconciles claims to URLs.
- Verifier checks final claims against claim table.

## Target Architecture

### Stage 1: Planner Produces A Claim Map

The planner should not only produce a table of contents. It should produce a structured claim map:

```json
{
  "claim_profile": {
    "claim_density": "medium",
    "claims": [
      {
        "claim_id": "C1",
        "claim_text": "The report needs to establish what the main market demand signal is.",
        "claim_type": "trend",
        "preferred_source_classes": ["primary_issuer", "academic", "authoritative_third_party"]
      }
    ]
  }
}
```

The table of contents organizes the final answer. The claim map controls what the final answer is allowed to assert.

### Stage 2: Planner Allocates Budget By Claim Groups

The planner should receive tier limits from `research_depth.py` and allocate them.

The model should never invent a budget. It should fill a shape like this:

```json
{
  "budget_plan": {
    "tier": "deeper",
    "total_search_calls": 64,
    "planner_reserve": 8,
    "researcher_allocations": [
      {
        "task_id": "R1",
        "target_claim_ids": ["C1", "C2", "C3"],
        "search_calls": 12,
        "source_priority": ["primary_issuer", "academic"]
      }
    ],
    "gap_fill_reserve": 8
  }
}
```

This directly addresses the repeated problem where the system keeps searching because the prompt implies source-count completion rather than evidence sufficiency.

### Stage 3: Researchers Resolve Claims

Each researcher receives a bounded contract:

- task objective
- included claim ids
- excluded scope
- search budget
- preferred source classes
- output path
- required claim-fragment schema

Each researcher writes:

```text
/shared/researcher_notes/{task_id}.md
/shared/claims/{task_id}.json
/shared/section_briefs/{task_id}.md
```

The important rule is that researchers never write the shared claim table directly. They write per-task fragments.

### Stage 4: Python Merges Claim Fragments

The merge must be deterministic Python, not an LLM instruction.

Merge rules:

- Validate every fragment with Pydantic.
- Deduplicate by `claim_id`.
- Prefer `verified` over `partially_verified` over `unverified`.
- Preserve multiple evidence URLs when useful.
- Mark unresolved claims explicitly.
- Write the merged table to:

```text
/shared/claim_table.json
```

This makes the claim table the substrate of generation instead of a late audit artifact.

### Stage 5: Section-Level Synthesis

Before final report synthesis, the system should produce section briefs.

A section brief is not a raw note dump. It should include:

- section title
- claims used
- verified evidence bullets
- uncertainty bullets
- citation candidates
- what must not be claimed

The final writer should read section briefs plus the merged claim table, not the entire raw search history.

This is the biggest way to reduce final synthesis pressure.

### Stage 6: Final Writer Uses Only Claim Table + Briefs

The final writer should be constrained:

- Verified claims can be stated normally with citation.
- Partially verified claims require hedging.
- Unverified claims must be omitted or explicitly framed as unknown.
- Claims not present in the claim table cannot be introduced as facts.

This is the main anti-fabrication control.

### Stage 7: Separate Citation Reconciliation

Citation should not rely only on the final writer.

After report draft:

1. Extract report claims.
2. Match report claims to claim ids.
3. Attach or verify citation URLs.
4. Repair near-miss URLs deterministically.
5. Only strip citations after repair fails.
6. If stripping weakens the report, warn instead of hiding the report.

This is similar in spirit to using a separate citation agent, but can start as deterministic code plus a small repair pass.

### Stage 8: Final Verifier

The verifier should score:

- factual claim support rate
- partial-support rate
- unsupported claim count
- citation support rate
- source class match rate
- source concentration
- missing essential verified claims

The result should be stored as a quality artifact, not only used to fail jobs.

## Revised Priority Order

### Priority 1: Warnings Instead Of Report Loss

This remains first because it protects user value immediately.

Change the backend so non-critical quality failures produce:

```json
{
  "status": "success",
  "quality_warnings": [...]
}
```

Do not fail the job for:

- weak source diversity
- high source concentration
- missing claim table
- under-resolved claim table
- sparse citations when a report exists

Still fail the job for:

- empty report
- raw provider/thinking output
- severe scope drift
- no usable report artifact

### Priority 2: Generation Hardening

This should move ahead of broad budget increases.

Implement:

- per-researcher claim fragments
- deterministic claim merge
- section briefs
- final writer constrained by claim table

This directly attacks unsupported polished prose.

### Priority 3: Citation Reconciliation

Implement a separate pass that checks report claims and citation mappings.

Minimum viable version:

- use existing source registry
- run deterministic URL repair
- identify unsupported or weakly supported claim sentences
- emit quality notes

Later version:

- small verifier model pass for claim-to-source alignment

### Priority 4: In-Flight Source Steering

After each researcher batch, compute:

- domain diversity
- source-class mix
- dominant-domain share
- unresolved claim count
- budget remaining

Then steer the next task.

The steering message should be short and operational, not a long prompt addition.

### Priority 5: Budget Allocator

Tie search budgets to claim groups and researcher contracts.

This avoids two bad outcomes:

- spending budget too early
- continuing to search after evidence sufficiency is reached

### Priority 6: Plan Preview Repair

Repair or reject bad user-facing plans before approval.

The plan preview must not be a generic fallback. It is the contract the user approves.

## MiniMax M3-Specific Rules

MiniMax M3 should be treated as capable but structure-sensitive.

Use it for:

- bounded researcher tasks
- extracting source-backed claims
- writing compact section briefs
- synthesizing from already-structured evidence

Avoid asking it to:

- maintain citation discipline across a huge raw context
- infer all source-quality logic from prose
- produce perfect JSON without validation
- decide by itself when enough research is enough
- read long instructions where the critical rule is buried

Practical rules:

1. Put the final task instruction after the source material for long synthesis calls.
2. Use small schemas and validate them.
3. On invalid JSON, repair with the validation error.
4. Give explicit permission to say "not verified."
5. Keep researcher contracts tight.
6. Prefer one file per researcher over shared writes.
7. Give the final writer fewer, better artifacts.

## Test Set Needed

Create a 20-task golden evaluation set before major prompt/model changes.

It should include:

- shallow current-events query
- deeper thematic business query
- deep product/model comparison query
- lesson topic plus debate motion
- local transcript query
- source-heavy historical/policy query
- stock quote / finance-support query
- broad philosophical/advice query
- query with likely weak public evidence
- query requiring first-party documentation

Each run should be scored on:

- factual accuracy
- citation accuracy
- completeness
- source quality
- scope adherence
- tool efficiency
- final report usability

This gives us a way to tell whether changes actually improve output rather than merely sounding better architecturally.

## Metrics To Track

### Fabrication Metrics

- Unsupported factual claims per report
- Unsupported factual claims per 1,000 words
- False citation rate
- Fabricated fact rate

### Evidence Metrics

- verified claims
- partially verified claims
- unverified claims
- claim support rate
- source class match rate
- distinct cited domains
- dominant-domain share

### Efficiency Metrics

- search calls used
- search calls remaining at synthesis
- researcher tasks spawned
- elapsed time by stage
- final synthesis retries
- reports completed with warnings

### User Experience Metrics

- plan approval rate
- plan repair rate
- report failure rate
- report-with-warning rate
- time to first useful artifact
- time to final report
- API report retrieval success rate

## What Not To Do

Do not solve quality by only increasing search budget.

Do not keep adding long prompt patches for every failure mode.

Do not make source-quality warnings fatal unless the report is unusable.

Do not let the final writer invent claims to make the report feel complete.

Do not force deep research to cite every collected source. The goal is enough high-quality cited evidence, not citation stuffing.

Do not route all complexity into the final synthesis call.

## Concrete First Implementation Slice

The best first engineering slice after quality-warning delivery is:

1. Add per-researcher claim-fragment output path:

```text
/shared/claims/{task_id}.json
```

2. Add deterministic merge helper:

```text
merge_claim_fragments(filesystem) -> /shared/claim_table.json
```

3. Add per-researcher section brief path:

```text
/shared/section_briefs/{task_id}.md
```

4. Update orchestrator so final synthesis reads:

```text
/shared/claim_table.json
/shared/section_briefs/*.md
verified source registry
```

5. Add a post-report verifier artifact:

```text
quality_verification.json
```

This is narrow enough to implement incrementally and important enough to change output quality.

## Expected Outcome

If implemented properly, this should reduce:

- confident unsupported claims
- reports that look polished but are weakly evidenced
- source drift
- final synthesis failures
- citation stripping cascades
- repeated "search one more time" loops

It should improve:

- report factuality
- citation faithfulness
- source diversity
- scope adherence
- user trust
- API reliability

The main idea is that MiniMax M3 does not need to be a frontier model in one giant step. It needs a workflow where every step is bounded, auditable, and evidence-constrained.
