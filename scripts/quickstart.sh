#!/usr/bin/env bash
# =============================================================================
# Deep Research — one-command quick start
# =============================================================================
# Builds and launches the full stack with Docker Compose. The only thing you
# need is a MiniMax API key in deploy/.env.
#
#   ./scripts/quickstart.sh
#
# Re-run any time to rebuild and restart. Pass --pull to use prebuilt images
# instead of building locally (faster, if images are published).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$ROOT/deploy/.env"
ENV_EXAMPLE="$ROOT/deploy/.env.example"
COMPOSE_FILE="$ROOT/deploy/compose/docker-compose.yaml"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
err()  { printf '\033[31m%s\033[0m\n' "$1" >&2; }

# --- 1. Preconditions ---------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  err "Docker is not installed. Install Docker Desktop or Docker Engine first:"
  err "  https://docs.docker.com/get-docker/"
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  err "The 'docker compose' plugin is not available. Update Docker to a recent version."
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  err "Docker daemon is not running. Start Docker and re-run this script."
  exit 1
fi

# --- 2. Ensure deploy/.env exists --------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
  bold "Creating deploy/.env from template..."
  cp "$ENV_EXAMPLE" "$ENV_FILE"
fi

# --- 3. Require a MiniMax API key --------------------------------------------
MINIMAX_KEY="$(grep -E '^MINIMAX_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
if [ -z "${MINIMAX_KEY// }" ]; then
  err ""
  err "MINIMAX_API_KEY is not set in deploy/.env"
  err "Open deploy/.env, paste your key after MINIMAX_API_KEY=, then re-run:"
  err "  ./scripts/quickstart.sh"
  err ""
  err "Get a key at https://www.minimax.io/ (platform → API keys)."
  exit 1
fi

# --- 4. Launch ---------------------------------------------------------------
FRONTEND_PORT="$(grep -E '^FRONTEND_PORT=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

bold "Starting the Deep Research stack (this builds images on first run — a few minutes)..."
if [ "${1:-}" = "--pull" ]; then
  docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
else
  docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build
fi

# --- 5. Wait for backend health ----------------------------------------------
BACKEND_PORT="$(grep -E '^PORT=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
bold "Waiting for the backend to become healthy..."
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 3
done

bold ""
bold "Deep Research is up."
bold "  UI:       http://localhost:${FRONTEND_PORT}"
bold "  API:      http://localhost:${BACKEND_PORT}/health"
bold ""
echo "Logs:  docker compose -f deploy/compose/docker-compose.yaml logs -f aiq-agent"
echo "Stop:  docker compose -f deploy/compose/docker-compose.yaml down"
