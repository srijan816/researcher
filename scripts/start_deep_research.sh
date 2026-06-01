#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
UI_DIR="$PROJECT_ROOT/frontends/ui"
RUNTIME_DIR="${AIQ_RUNTIME_DIR:-$PROJECT_ROOT/.deep-research-runtime}"

CONFIG_FILE="${AIQ_CONFIG_FILE:-configs/config_cli_minimax_ddgs.yml}"
BACKEND_PORT="${BACKEND_PORT:-9000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
BACKEND_URL="http://localhost:${BACKEND_PORT}"
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"
BACKEND_SESSION="${BACKEND_SESSION:-deep-research-backend}"
FRONTEND_SESSION="${FRONTEND_SESSION:-deep-research-frontend}"

mkdir -p "$RUNTIME_DIR"

detect_lan_host() {
    if [ -n "${LAN_HOST:-}" ]; then
        printf "%s" "$LAN_HOST"
        return 0
    fi

    local ip
    ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
    if [ -z "$ip" ]; then
        ip="$(ipconfig getifaddr en1 2>/dev/null || true)"
    fi
    if [ -z "$ip" ] && command -v hostname >/dev/null 2>&1; then
        ip="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
    fi
    if [ -n "$ip" ]; then
        printf "%s" "$ip"
    fi
}

is_pid_running() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid
        pid="$(cat "$pid_file")"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

is_backend_ready() {
    curl -s -f "${BACKEND_URL}/health" >/dev/null 2>&1 || curl -s -f "${BACKEND_URL}/docs" >/dev/null 2>&1
}

is_frontend_ready() {
    curl -s -f "$FRONTEND_URL" >/dev/null 2>&1
}

wait_for_backend() {
    local attempt=1
    local max_attempts=90
    printf "Waiting for backend"
    while [ "$attempt" -le "$max_attempts" ]; do
        if is_backend_ready; then
            printf "\nBackend is ready: %s\n" "$BACKEND_URL"
            return 0
        fi
        printf "."
        sleep 1
        attempt=$((attempt + 1))
    done
    printf "\nBackend did not answer within %s seconds. Check %s/backend.log\n" "$max_attempts" "$RUNTIME_DIR"
    return 1
}

start_backend() {
    local pid_file="$RUNTIME_DIR/backend.pid"
    local log_file="$RUNTIME_DIR/backend.log"

    if is_pid_running "$pid_file"; then
        printf "Backend already started by this script: %s (PID %s)\n" "$BACKEND_URL" "$(cat "$pid_file")"
        return 0
    fi

    if is_backend_ready; then
        printf "Backend already responding at %s\n" "$BACKEND_URL"
        return 0
    fi

    if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
        printf "Config file not found: %s\n" "$CONFIG_FILE"
        exit 1
    fi

    printf "Starting backend at %s\n" "$BACKEND_URL"
    if command -v tmux >/dev/null 2>&1; then
        tmux kill-session -t "$BACKEND_SESSION" 2>/dev/null || true
        tmux new-session -d -s "$BACKEND_SESSION" -c "$PROJECT_ROOT" \
            "bash -lc 'if [ -f deploy/.env ]; then set -a; source deploy/.env; set +a; fi; export AIQ_DEV_ENV=e2e BACKEND_URL=\"$BACKEND_URL\" NEXT_PUBLIC_BACKEND_URL=\"$BACKEND_URL\" PYTHONWARNINGS=\"\${PYTHONWARNINGS:-ignore}\"; exec uv run nat serve --config_file \"$CONFIG_FILE\" --host 0.0.0.0 --port \"$BACKEND_PORT\"' >> \"$log_file\" 2>&1"
        tmux display-message -p -t "$BACKEND_SESSION" "#{pane_pid}" >"$pid_file"
        wait_for_backend || true
        return
    fi

    PROJECT_ROOT="$PROJECT_ROOT" BACKEND_URL="$BACKEND_URL" NEXT_PUBLIC_BACKEND_URL="$BACKEND_URL" \
        CONFIG_FILE="$CONFIG_FILE" BACKEND_PORT="$BACKEND_PORT" nohup bash -c '
        set -euo pipefail
        cd "$1"
        if [ -f "$PROJECT_ROOT/deploy/.env" ]; then
            set -a
            source "$PROJECT_ROOT/deploy/.env"
            set +a
        fi
        export AIQ_DEV_ENV=e2e
        export BACKEND_URL="$BACKEND_URL"
        export NEXT_PUBLIC_BACKEND_URL="$BACKEND_URL"
        export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
        exec uv run nat serve --config_file "$CONFIG_FILE" --host 0.0.0.0 --port "$BACKEND_PORT"
    ' bash "$PROJECT_ROOT" >"$log_file" 2>&1 &
    local pid=$!
    printf "%s\n" "$pid" >"$pid_file"
    disown "$pid" 2>/dev/null || true
    wait_for_backend || true
}

start_frontend() {
    local pid_file="$RUNTIME_DIR/frontend.pid"
    local log_file="$RUNTIME_DIR/frontend.log"

    if is_pid_running "$pid_file"; then
        printf "Frontend already started by this script: %s (PID %s)\n" "$FRONTEND_URL" "$(cat "$pid_file")"
        return 0
    fi

    if is_frontend_ready; then
        printf "Frontend already responding at %s\n" "$FRONTEND_URL"
        return 0
    fi

    if [ ! -d "$UI_DIR" ]; then
        printf "UI directory not found: %s\n" "$UI_DIR"
        exit 1
    fi

    if [ ! -d "$UI_DIR/node_modules" ]; then
        printf "Installing frontend dependencies\n"
        cd "$UI_DIR"
        npm ci
    fi

    printf "Starting frontend at %s\n" "$FRONTEND_URL"
    if command -v tmux >/dev/null 2>&1; then
        tmux kill-session -t "$FRONTEND_SESSION" 2>/dev/null || true
        tmux new-session -d -s "$FRONTEND_SESSION" -c "$UI_DIR" \
            "bash -lc 'if [ -f \"$PROJECT_ROOT/deploy/.env\" ]; then set -a; source \"$PROJECT_ROOT/deploy/.env\"; set +a; fi; export AIQ_DEV_ENV=e2e BIND_HOST=\"$BIND_HOST\" AIQ_RESEARCH_ROOT=\"\${AIQ_RESEARCH_ROOT:-$PROJECT_ROOT}\" BACKEND_URL=\"$BACKEND_URL\" NEXT_PUBLIC_BACKEND_URL=\"$BACKEND_URL\"; exec npm run dev' >> \"$log_file\" 2>&1"
        tmux display-message -p -t "$FRONTEND_SESSION" "#{pane_pid}" >"$pid_file"
        return
    fi

    PROJECT_ROOT="$PROJECT_ROOT" BACKEND_URL="$BACKEND_URL" NEXT_PUBLIC_BACKEND_URL="$BACKEND_URL" BIND_HOST="$BIND_HOST" nohup bash -c '
        set -euo pipefail
        cd "$1"
        if [ -f "$PROJECT_ROOT/deploy/.env" ]; then
            set -a
            source "$PROJECT_ROOT/deploy/.env"
            set +a
        fi
        export AIQ_DEV_ENV=e2e
        export BIND_HOST="$BIND_HOST"
        export AIQ_RESEARCH_ROOT="${AIQ_RESEARCH_ROOT:-$PROJECT_ROOT}"
        export BACKEND_URL="$BACKEND_URL"
        export NEXT_PUBLIC_BACKEND_URL="$BACKEND_URL"
        exec npm run dev
    ' bash "$UI_DIR" >"$log_file" 2>&1 &
    local pid=$!
    printf "%s\n" "$pid" >"$pid_file"
    disown "$pid" 2>/dev/null || true
}

start_backend
start_frontend
LAN_HOST_DETECTED="$(detect_lan_host || true)"

printf "\nDeep Research is starting.\n"
printf "Backend:  %s\n" "$BACKEND_URL"
printf "Frontend: %s\n" "$FRONTEND_URL"
if [ -n "$LAN_HOST_DETECTED" ]; then
    printf "LAN:      http://%s:%s\n" "$LAN_HOST_DETECTED" "$FRONTEND_PORT"
fi
printf "Logs:     %s\n" "$RUNTIME_DIR"
printf "Stop:     ./scripts/stop_deep_research.sh\n"
