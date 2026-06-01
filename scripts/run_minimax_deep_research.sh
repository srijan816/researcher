#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export AIQ_CONFIG_FILE="${AIQ_CONFIG_FILE:-configs/config_cli_minimax_ddgs.yml}"
export BACKEND_PORT="${BACKEND_PORT:-9000}"
export FRONTEND_PORT="${FRONTEND_PORT:-3000}"
export NEXT_PORT="${NEXT_PORT:-3001}"
export SEARXNG_URL="${SEARXNG_URL:-http://localhost:8080}"
export AIQ_RESEARCH_ROOT="${AIQ_RESEARCH_ROOT:-$PROJECT_ROOT}"
export DEBATE_TRANSCRIPT_NOTEBOOKLM_SOURCE_DIR="${DEBATE_TRANSCRIPT_NOTEBOOKLM_SOURCE_DIR:-/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data/notebooklm_exports/debate_transcripts}"
export DEBATE_TRANSCRIPT_RECENT_SOURCE_DIR="${DEBATE_TRANSCRIPT_RECENT_SOURCE_DIR:-/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data/youtube_recent_transcripts_2022_2026}"
export DEBATE_TRANSCRIPT_OUTPUT_DIR="${DEBATE_TRANSCRIPT_OUTPUT_DIR:-/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data/prepared_debate_corpus}"
export DEBATE_TRANSCRIPT_CORPUS_JSONL="${DEBATE_TRANSCRIPT_CORPUS_JSONL:-$DEBATE_TRANSCRIPT_OUTPUT_DIR/chunks/search_chunks.jsonl}"
export KOKORO_MODEL_DIR="${KOKORO_MODEL_DIR:-$PROJECT_ROOT/data/kokoro-models}"
export KOKORO_PYTHON="${KOKORO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export KOKORO_MODEL_URL="${KOKORO_MODEL_URL:-https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx}"
export KOKORO_VOICES_URL="${KOKORO_VOICES_URL:-https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin}"

kill_pid_tree() {
    local pid="$1"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    local child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        kill_pid_tree "$child"
    done

    kill "$pid" 2>/dev/null || true
}

clear_port() {
    local name="$1"
    local port="$2"
    local pids

    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -z "$pids" ]; then
        return 0
    fi

    printf "Clearing stale %s listener(s) on port %s: %s\n" "$name" "$port" "$pids"
    local pid
    for pid in $pids; do
        kill_pid_tree "$pid"
    done
}

wait_for_searxng() {
    local attempt=1
    local max_attempts=30
    printf "Waiting for SearXNG"
    while [ "$attempt" -le "$max_attempts" ]; do
        if curl -fsS "${SEARXNG_URL%/}/search?q=health&format=json" >/dev/null 2>&1; then
            printf "\nSearXNG is ready: %s\n" "$SEARXNG_URL"
            return 0
        fi
        printf "."
        sleep 1
        attempt=$((attempt + 1))
    done
    printf "\nSearXNG did not become ready at %s\n" "$SEARXNG_URL"
    return 1
}

ensure_searxng() {
    if curl -fsS "${SEARXNG_URL%/}/search?q=health&format=json" >/dev/null 2>&1; then
        printf "SearXNG already responding: %s\n" "$SEARXNG_URL"
        return 0
    fi

    if command -v docker >/dev/null 2>&1; then
        printf "Starting SearXNG with Docker Compose\n"
        docker compose --env-file "$PROJECT_ROOT/deploy/.env" \
            -f "$PROJECT_ROOT/deploy/compose/docker-compose.yaml" \
            up -d searxng
        wait_for_searxng
        return 0
    fi

    printf "SearXNG is not responding at %s and Docker is not available.\n" "$SEARXNG_URL"
    printf "Start SearXNG first, then rerun this script.\n"
    exit 1
}

download_if_missing() {
    local destination="$1"
    local url="$2"
    if [ -s "$destination" ]; then
        return 0
    fi

    mkdir -p "$(dirname "$destination")"
    printf "Downloading %s\n" "$(basename "$destination")"
    local tmp="${destination}.tmp.$$"
    rm -f "$tmp"
    curl -fL --retry 3 --connect-timeout 30 --max-time 600 "$url" -o "$tmp"
    mv "$tmp" "$destination"
}

ensure_kokoro() {
    printf "Preparing Kokoro ONNX narration assets\n"
    download_if_missing "$KOKORO_MODEL_DIR/kokoro-v1.0.onnx" "$KOKORO_MODEL_URL"
    download_if_missing "$KOKORO_MODEL_DIR/voices-v1.0.bin" "$KOKORO_VOICES_URL"

    "$KOKORO_PYTHON" - <<'PY'
from pathlib import Path
from kokoro_onnx import Kokoro
import os

model_dir = Path(os.environ["KOKORO_MODEL_DIR"])
model = model_dir / "kokoro-v1.0.onnx"
voices = model_dir / "voices-v1.0.bin"
if not model.exists() or not voices.exists():
    raise SystemExit(f"Kokoro model files are missing in {model_dir}")

Kokoro(str(model), str(voices))
print(f"Kokoro ONNX ready: {model_dir}")
PY

    if ! command -v "${FFMPEG_BIN:-ffmpeg}" >/dev/null 2>&1; then
        printf "ffmpeg was not found; Kokoro narration will serve WAV audio instead of MP3.\n"
    fi
}

prepare_debate_transcript_corpus() {
    if [ ! -d "$DEBATE_TRANSCRIPT_NOTEBOOKLM_SOURCE_DIR" ] && [ ! -d "$DEBATE_TRANSCRIPT_RECENT_SOURCE_DIR" ]; then
        printf "No local debate transcript source folders found; skipping corpus preparation.\n"
        return 0
    fi

    printf "Preparing local debate transcript corpus\n"
    python3 "$PROJECT_ROOT/scripts/prepare_debate_transcript_corpus.py" \
        --source-dir "$DEBATE_TRANSCRIPT_NOTEBOOKLM_SOURCE_DIR" \
        --recent-source-dir "$DEBATE_TRANSCRIPT_RECENT_SOURCE_DIR" \
        --output-dir "$DEBATE_TRANSCRIPT_OUTPUT_DIR"
}

printf "Stopping existing local Deep Research backend/frontend processes\n"
FORCE_STOP_DEEP_RESEARCH=1 "$PROJECT_ROOT/scripts/stop_deep_research.sh" || true

clear_port backend "$BACKEND_PORT"
clear_port frontend "$FRONTEND_PORT"
clear_port frontend-next "$NEXT_PORT"

printf "Installing local workflow plugins and extraction dependencies\n"
uv pip install \
    -e "$PROJECT_ROOT/sources/debate_transcript_search" \
    -e "$PROJECT_ROOT/sources/searxng_jina_web_search" \
    -e "$PROJECT_ROOT/sources/stooq_stock_quote" \
    -e "$PROJECT_ROOT/frontends/aiq_api" \
    "kokoro-onnx>=0.5.0,<0.6" >/dev/null

prepare_debate_transcript_corpus
ensure_kokoro
ensure_searxng

printf "Starting MiniMax Deep Research workflow\n"
exec "$PROJECT_ROOT/scripts/start_deep_research.sh"
