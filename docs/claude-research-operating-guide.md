# Claude Research Operating Guide

This guide defines the concurrent Claude Code research engine. It is designed to
run beside the existing AI-Q deep research workflow first, then become a
candidate replacement for the research worker once it proves better on real
jobs.

The core idea is simple: keep the app shell, job store, UI, SearXNG, Websurfx,
scrapers, source memory, and verification layers, but let Claude Code own the
research work inside a strict artifact contract.

## Non-Negotiable Contract

Every Claude research run writes to one run folder:

```text
runs/<job-or-topic>/
  README.md
  plan.md
  queries.json
  sources.json
  notes/
  source_summaries/
  contradictions.md
  gaps.md
  research.md
  final.md
  logs/
```

The app must not trust stdout as the result. Stdout is progress only. The app
reads files from the run folder.

During the run, update `logs/progress.md` after each phase with short,
observable progress notes. Do not put hidden chain-of-thought there. Record what
was done, which modules/sources changed state, what gaps remain, and what will
happen next. The UI streams this file while Claude Code is still running.
After receiving a run prompt, your first file write should be
`logs/progress.md` with a one-sentence note that planning has started.

## Depth Accountability

Depth is a contract, not a label. Use the selected tier to size the plan, source
collection, and verification effort. These are targets and floors for normal
queries; if a query is genuinely narrow or the user explicitly requests a tiny
run, explain the undershoot in `gaps.md`.

| Tier | Modules | Target searches | Candidate sources | Sources read/summarized | Expected rigor |
| --- | ---: | ---: | ---: | ---: | --- |
| shallow | 2-4 | 4-8 | 6-12 | 4-8 | concise answer with explicit gaps |
| medium | 4-6 | 8-14 | 16-28 | 10-16 | balanced research compile with source table and gap check |
| deeper | 5-7 | 18-32 | 35-70 | 20-35 | module evidence, contradictions, gap-fill pass, calibrated final report |
| deep | 7-10 | 35-60 | 70-140 | 40-70 | exhaustive dossier, counterevidence, contradiction table, validation pass |

For all tiers:

- every major module in `queries.json` must be represented in `notes/` or
  `research.md`
- high-risk numbers, rankings, recency claims, and named-entity facts need
  authoritative support
- source quality matters more than raw count; prefer official, academic,
  primary-data, standards, regulator, analyst, and reputable-news sources over
  generic blogs
- use `gaps.md` to justify any missing module, weak source, or intentional
  depth reduction

## What Claude Code Owns

Claude Code owns:

- query understanding
- plan decomposition
- source strategy
- search query generation
- deciding when enough evidence exists
- gap analysis
- contradiction tracking
- synthesis
- final report writing

Claude Code does not own:

- serving the web UI
- user sessions
- async job persistence
- SSE streaming
- app health checks
- source and report verification policy
- deployment lifecycle

## Tools Exposed To Claude Code

Claude Code should use this repo-local CLI instead of improvising web access:

```bash
python scripts/claude_research_tool.py init-run --run-dir runs/example
python scripts/claude_research_tool.py search "query text" --backend websurfx_hybrid --run-dir runs/example
python scripts/claude_research_tool.py scrape "https://example.com/page" --run-dir runs/example
```

The tool uses:

- Websurfx for advanced meta-search when available
- SearXNG for general self-hosted search
- Scrapling for page extraction when available
- HTTP plus BeautifulSoup fallback when Scrapling fails

The default Oracle endpoints are provided through environment variables:

```text
SEARXNG_URL=http://searxng:8080
WEBSURFX_URL=http://websurfx:8080
WEBSURFX_ENGINES=Searx,Brave,DuckDuckGo,LibreX,Mojeek,Qwant,Startpage,Yahoo,Bing,SepiaSearch
```

Local development usually uses:

```text
SEARXNG_URL=http://127.0.0.1:8080
WEBSURFX_URL=http://127.0.0.1:8081
```

## Required Run Phases

### Phase 0: Plan

Before searching, write `plan.md` with:

- main question
- user deliverable
- audience and use case
- assumptions
- subquestions/modules
- likely authoritative source types
- risky claims needing verification
- search strategy
- stop condition

Also write `queries.json`:

```json
{
  "modules": [
    {
      "id": "M1",
      "title": "Module title",
      "goal": "What this module must resolve",
      "budget_percent": 25,
      "queries": [
        {
          "query": "short search-engine friendly query",
          "source_goal": "primary | academic | trade | first_party | counterevidence"
        }
      ]
    }
  ]
}
```

Budget percentages must sum to 100. Keep queries short and searchable.

### Phase 1: Discovery

Use `claude_research_tool.py search` for each module. Prefer Websurfx hybrid for
deeper research:

```bash
python scripts/claude_research_tool.py search "precise query" \
  --backend websurfx_hybrid \
  --limit 8 \
  --run-dir runs/example
```

Search results are automatically appended to `sources.json` when `--run-dir` is
provided.

### Phase 2: Source Reading

Scrape promising URLs:

```bash
python scripts/claude_research_tool.py scrape "https://source.url" \
  --run-dir runs/example \
  --max-chars 12000
```

Each scrape writes a Markdown source summary in `source_summaries/`.

For each source summary, capture:

- source URL
- source title
- source type or class if known
- claims supported
- numbers/dates
- limitations
- bias risk
- whether it supports or contradicts other sources

### Phase 3: Notes And Contradictions

Write module notes to `notes/<module-id>.md`.

Update `contradictions.md` whenever:

- two sources disagree
- a source contradicts the likely answer
- a number differs across sources
- a finding is stale or date-sensitive

Update `gaps.md` whenever:

- a required module lacks evidence
- a hard number lacks primary support
- a source is weak but currently necessary
- the final output would need a hedge

### Phase 4: Research Compile

Before writing the final report, write `research.md` with:

- executive synthesis of findings
- source table
- claim-by-claim evidence
- unresolved gaps
- contradictions and caveats
- recommended conclusion
- final report outline

The final writer must use `research.md`, not raw memory, as the main substrate.

### Phase 5: Final Report

Write `final.md` in the user-requested format. The final report must:

- preserve the user's actual task and audience
- cite sources with URLs or source IDs
- hedge weakly supported claims
- avoid precise numbers without evidence
- include source notes or limitations where relevant

## Stop Conditions

Stop searching when:

- each major module has enough evidence for a defensible answer
- high-risk claims have primary or authoritative support, or are explicitly
  hedged as unverified
- additional search is repeating the same sources
- `gaps.md` contains no blocker-level gaps

Do not exhaust search budget for its own sake.

## Failure Rules

If search fails:

- simplify the query
- switch backend from Websurfx hybrid to SearXNG
- try one broader and one narrower query
- record the failure in `logs/search_failures.md`

If scraping fails:

- try the source homepage, PDF, press release, or cached page
- use another source if the claim is not source-specific
- record the failure in `logs/scrape_failures.md`

If a claim cannot be verified:

- leave it out, or
- include it with an explicit hedge and explain the limitation

## Integration Target

The intended app integration is a new `claude_research` worker:

1. FastAPI creates a job and run folder.
2. FastAPI launches Claude Code with this guide and the user query.
3. Claude Code uses `scripts/claude_research_tool.py`.
4. UI streams logs and watches artifact files.
5. Backend runs existing source quality, citation, and fact audit layers over
   `final.md`.

This lets Claude Code replace the brittle research loop without replacing the
product shell.
