# Documentation

This folder is intentionally small. It should describe the current app, not
old implementation plans or historical debugging notes.

Current documents:

- `deep-research-agent-actual-workflow.md` - how the deep research agent
  actually runs today, including prompt/template locations and live Oracle
  deployment shape.
- `deep-research-pipeline.md` - end-to-end query-to-output pipeline overview.
- `deep-research-api.md` - backend API/auth/job usage notes.

Remove or archive obsolete roadmap, audit, and one-off fix-plan documents
instead of keeping them beside current-state docs.

## Building the Documentation

## Prerequisites

```bash
# Install doc dependencies from pyproject.toml
uv pip install -e ".[docs]"
```

## Build

```bash
make -C docs html
```

## Preview

```bash
python -m http.server --directory docs/build/html 8080
# Open http://localhost:8080
```

## Link Check

```bash
make -C docs linkcheck
```
