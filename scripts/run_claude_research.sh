#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAUDE_BIN="${AIQ_CLAUDE_CODE_PATH:-claude}"
RUN_ROOT="${AIQ_CLAUDE_RESEARCH_RUN_ROOT:-$ROOT_DIR/runs}"
DEPTH="${AIQ_CLAUDE_RESEARCH_DEPTH:-deeper}"
QUERY=""
RUN_ID=""

for env_file in "$ROOT_DIR/.env" "$ROOT_DIR/deploy/.env"; do
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_claude_research.sh --query "research question" [--run-id id] [--depth medium|deeper|deep]
  scripts/run_claude_research.sh --query-file prompt.txt [--run-id id]

Environment:
  MINIMAX_API_KEY                         Used as ANTHROPIC_AUTH_TOKEN when provider=minimax.
  AIQ_CLAUDE_CODE_PROVIDER=minimax         Default provider adapter.
  AIQ_CLAUDE_CODE_BASE_URL                 Default: https://api.minimax.io/anthropic
  AIQ_CLAUDE_CODE_MODEL                    Default: MiniMax-M3
  AIQ_CLAUDE_CODE_BYPASS_PERMISSIONS=true  Adds --dangerously-skip-permissions.
  SEARXNG_URL                              Search endpoint.
  WEBSURFX_URL                             Websurfx endpoint.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --query)
      QUERY="${2:-}"
      shift 2
      ;;
    --query-file)
      QUERY="$(cat "${2:-}")"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --depth)
      DEPTH="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$QUERY" ]]; then
  echo "Missing --query or --query-file" >&2
  usage >&2
  exit 2
fi

if ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
  echo "Claude Code CLI not found: $CLAUDE_BIN" >&2
  exit 127
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(printf '%s' "$QUERY" | shasum -a 1 | awk '{print substr($1,1,12)}')"
fi

RUN_DIR="$RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

PYTHON_BIN="${AIQ_CLAUDE_RESEARCH_PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/claude_research_tool.py" init-run --run-dir "$RUN_DIR" >/dev/null

export ANTHROPIC_BASE_URL="${AIQ_CLAUDE_CODE_BASE_URL:-https://api.minimax.io/anthropic}"
if [[ "${AIQ_CLAUDE_CODE_PROVIDER:-minimax}" == "minimax" && -n "${MINIMAX_API_KEY:-}" ]]; then
  export ANTHROPIC_AUTH_TOKEN="${AIQ_CLAUDE_CODE_API_KEY:-$MINIMAX_API_KEY}"
  unset ANTHROPIC_API_KEY
  export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-${AIQ_CLAUDE_CODE_MODEL:-MiniMax-M3}}"
  export ANTHROPIC_DEFAULT_SONNET_MODEL="${ANTHROPIC_DEFAULT_SONNET_MODEL:-$ANTHROPIC_MODEL}"
  export ANTHROPIC_DEFAULT_OPUS_MODEL="${ANTHROPIC_DEFAULT_OPUS_MODEL:-$ANTHROPIC_MODEL}"
  export ANTHROPIC_DEFAULT_HAIKU_MODEL="${ANTHROPIC_DEFAULT_HAIKU_MODEL:-$ANTHROPIC_MODEL}"
  export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}"
fi

PERMISSION_ARGS=("--permission-mode" "${AIQ_CLAUDE_CODE_PERMISSION_MODE:-auto}")
if [[ "${AIQ_CLAUDE_CODE_BYPASS_PERMISSIONS:-}" =~ ^(1|true|yes|on)$ ]]; then
  PERMISSION_ARGS=("--permission-mode" "bypassPermissions" "--dangerously-skip-permissions")
fi

PROMPT="$(cat <<PROMPT_EOF
You are Claude Code running the concurrent Claude Research engine for this repository.

Read and obey this operating guide:
$ROOT_DIR/docs/claude-research-operating-guide.md

Run folder:
$RUN_DIR

Research depth:
$DEPTH

Depth accountability:
Follow the tier table in the operating guide for "$DEPTH". Treat it as the
normal floor for modules, searches, candidate sources, source summaries, and
gap closure. If the task is intentionally too narrow for that depth, explain the
exception in $RUN_DIR/gaps.md.

User query:
$QUERY

You MUST use the repo-local tool for search and scraping:
$PYTHON_BIN $ROOT_DIR/scripts/claude_research_tool.py

Required final artifacts:
- $RUN_DIR/plan.md
- $RUN_DIR/queries.json
- $RUN_DIR/sources.json
- $RUN_DIR/logs/progress.md
- $RUN_DIR/contradictions.md
- $RUN_DIR/gaps.md
- $RUN_DIR/research.md
- $RUN_DIR/final.md

While working, update $RUN_DIR/logs/progress.md after each phase with
observable progress notes. Do not put hidden chain-of-thought there.

Do not treat stdout as the deliverable. The app will inspect the files above.
When all required artifacts are complete, print exactly:
CLAUDE_RESEARCH_COMPLETE $RUN_DIR
PROMPT_EOF
)"

printf '%s' "$PROMPT" | "$CLAUDE_BIN" \
  --bare \
  --print \
  --output-format text \
  --input-format text \
  "${PERMISSION_ARGS[@]}" \
  --add-dir "$ROOT_DIR" \
  --add-dir "$RUN_DIR"
