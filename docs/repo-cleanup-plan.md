# Deep Research Repo Cleanup Plan

This repo currently mixes three different kinds of material:

1. Source code and configuration that should be versioned and deployed.
2. Local runtime state that should be preserved but never committed.
3. Generated research artifacts and model/cache files that should live in durable storage or an archive path, not in the code tree.

The cleanup goal is not to delete useful research history. The goal is to make the code path clean enough that a deploy can be trusted, reviewed, and repeated.

## Current Local Bloat Snapshot

Observed on the local workspace before the app2 deploy:

- `.deep-research-runtime/`: about 196 MB, mostly local backend/frontend logs.
- `data/`: about 357 MB.
- `data/kokoro-models/`: about 337 MB, generated/downloaded Kokoro model files.
- `data/scrape_artifacts/`: about 19 MB, fetched research documents.
- Root-level local artifacts include `jobs.db`, `checkpoints.db`, `summaries.db`, `backend.log`, `ui.log`, `latest_checkpoint.blob`, and recovered report markdown files.

## Keep In Repo

These are code or reproducible configuration and should remain in the repo:

- `src/aiq_agent/**`
- `frontends/aiq_api/**`
- `frontends/ui/**`
- `sources/**`
- `configs/**`
- `deploy/**`, except server-local `.env`
- `scripts/**`
- `tests/**`
- `docs/**`
- `pyproject.toml`, `uv.lock`, package manifests, Docker files, compose files

## Keep Out Of Repo

These are runtime/cache/data artifacts and should be ignored or stored elsewhere:

- `.deep-research-runtime/`
- `.tmp/`
- `.venv/`
- `node_modules/`
- `.next/`
- `*.db`, `*.sqlite`, `*.log`
- `data/kokoro-models/`
- `data/scrape_artifacts/`
- `latest_checkpoint.blob`
- `recovered_report_*.md`
- server-local files such as `deploy/.env`

## Clean Pathway

### Phase 1: Guardrails

Keep `.gitignore` aligned with generated artifacts so routine research runs do not pollute `git status`.

Use the existing app2 sync script for deployment:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/scripts/sync_app2_deep_research.sh --apply --restart all
```

The script already excludes secrets, `.git`, virtualenvs, `node_modules`, local databases, logs, `.deep-research-runtime`, `data/`, and recovered reports.

### Phase 2: Separate Runtime State

Local runtime state should be under one path:

```text
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/.deep-research-runtime/
```

Durable server runtime state should stay in Docker volumes on app2:

- Postgres volume for async jobs/checkpoints.
- AIQ data volume for agent runtime state.
- SearXNG cache volume.
- Kokoro model/cache volume.

Do not use `docker compose down -v` unless intentionally wiping state.

### Phase 3: Separate Research Artifacts

Fetched document artifacts should be treated as durable job artifacts, not source files.

Recommended local structure:

```text
data/scrape_artifacts/       # generated, ignored
data/kokoro-models/          # generated, ignored
external-corpora/            # optional local-only corpus checkout, ignored unless intentionally tracked
```

Recommended server structure:

```text
/opt/stacks/app2/deep-research-data/
```

That server path can hold large corpora or generated assets separately from `/opt/stacks/app2/deep-research`, which should remain the deployable source tree.

### Phase 4: Split Feature Work Into Reviewable Groups

The current working tree contains several independent feature groups. Before committing, split them into reviewable chunks:

- Deployment/runtime: compose, Dockerfile, app2 sync/start/stop scripts.
- Authentication/history: local users, API keys, conversation storage.
- UI/UX: mobile layout, history panels, copy buttons, mode selector, report tab, Kokoro controls.
- Research quality: source classification, fact ledger, prompt hardening, post-run gates.
- Local research sources: debate transcript preparation/search and NotebookLM source harvest.

This keeps future debugging much easier: if a deploy breaks, we can identify whether the cause is UI, auth, research pipeline, or infrastructure.

### Phase 5: Optional Archive Command

After verifying no active local job needs these files, archive local generated artifacts outside the repo:

```bash
mkdir -p /Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/_archives/nvda-deep-research-runtime
```

Then move old logs, recovered reports, and obsolete local DBs there. Do this only after confirming the current app and any in-progress research no longer need them.

## Deployment Rule

Deploy source from the repo. Persist state in databases, Docker volumes, or explicit data roots. Do not deploy browser cache, local logs, local SQLite files, recovered reports, temporary files, or model caches as source.
