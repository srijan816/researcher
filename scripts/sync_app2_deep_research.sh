#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REMOTE_HOST="${APP2_REMOTE_HOST:-oracle}"
REMOTE_ROOT="${APP2_REMOTE_ROOT:-/opt/stacks/app2/deep-research}"
COMPOSE_FILE="${REMOTE_ROOT}/deploy/compose/docker-compose.app2.yaml"
ENV_FILE="${REMOTE_ROOT}/deploy/.env"

APPLY=0
RESTART="none"

usage() {
  cat <<'EOF'
Sync the local NVDA/MiniMax Deep Research project to app2 on the Oracle VPS.

Default mode is a dry run. It never copies deploy/.env or local runtime data.

Usage:
  ./scripts/sync_app2_deep_research.sh [--apply] [--restart none|backend|frontend|all]

Examples:
  ./scripts/sync_app2_deep_research.sh
  ./scripts/sync_app2_deep_research.sh --apply
  ./scripts/sync_app2_deep_research.sh --apply --restart backend
  ./scripts/sync_app2_deep_research.sh --apply --restart all

Environment overrides:
  APP2_REMOTE_HOST=oracle
  APP2_REMOTE_ROOT=/opt/stacks/app2/deep-research
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --restart)
      RESTART="${2:-}"
      shift 2
      ;;
    --help|-h)
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

case "$RESTART" in
  none|backend|frontend|all) ;;
  *)
    echo "--restart must be one of: none, backend, frontend, all" >&2
    exit 2
    ;;
esac

RSYNC_ARGS=(
  -az
  --itemize-changes
  --omit-dir-times
  --exclude='.git/'
  --exclude='.DS_Store'
  --exclude='.venv/'
  --exclude='node_modules/'
  --exclude='.next/'
  --exclude='.deep-research-runtime/'
  --exclude='.pytest_cache/'
  --exclude='.ruff_cache/'
  --exclude='.tmp/'
  --exclude='/data/'
  --exclude='/tests/knowledge_layer_tests/data/'
  --include='.env.example'
  --exclude='.env'
  --exclude='.env.*'
  --exclude='*.db'
  --exclude='*.db-*'
  --exclude='*.sqlite'
  --exclude='*.sqlite-*'
  --exclude='*.tsbuildinfo'
  --exclude='*.log'
  --exclude='backend.log'
  --exclude='*.pyc'
  --exclude='__pycache__/'
  --exclude='*.egg-info/'
  --exclude='latest_checkpoint.blob'
  --exclude='recovered_report_*.md'
  --exclude='deploy/.env'
)

if [ "$APPLY" -eq 0 ]; then
  RSYNC_ARGS+=(--dry-run)
  echo "Dry run only. Add --apply to push changes."
fi

echo "Syncing ${PROJECT_ROOT}/ -> ${REMOTE_HOST}:${REMOTE_ROOT}/"
rsync "${RSYNC_ARGS[@]}" "${PROJECT_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"

if [ "$APPLY" -eq 0 ] || [ "$RESTART" = "none" ]; then
  exit 0
fi

remote_compose() {
  ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && docker compose --env-file '$ENV_FILE' -f '$COMPOSE_FILE' $*"
}

case "$RESTART" in
  backend)
    remote_compose up -d --build --no-deps aiq-agent
    ;;
  frontend)
    remote_compose up -d --build --no-deps frontend
    ;;
  all)
    remote_compose up -d --build aiq-agent frontend kokoro searxng websurfx websurfx-redis
    ;;
esac

ssh "$REMOTE_HOST" "curl -fsS http://127.0.0.1:9000/health >/dev/null && docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' | grep app2-aiq"
