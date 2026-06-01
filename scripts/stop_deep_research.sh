#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUNTIME_DIR="${AIQ_RUNTIME_DIR:-$PROJECT_ROOT/.deep-research-runtime}"
BACKEND_PORT="${BACKEND_PORT:-9000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
NEXT_PORT="${NEXT_PORT:-3001}"
BACKEND_URL="${BACKEND_URL:-http://localhost:${BACKEND_PORT}}"
BACKEND_SESSION="${BACKEND_SESSION:-deep-research-backend}"
FRONTEND_SESSION="${FRONTEND_SESSION:-deep-research-frontend}"

active_jobs_from_json() {
    if command -v jq >/dev/null 2>&1; then
        jq -r '.jobs[]? | select(.status == "submitted" or .status == "running") | [.job_id, .status, (.title // .input // "")] | @tsv'
        return 0
    fi

    if command -v node >/dev/null 2>&1; then
        node -e '
let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => { raw += chunk; });
process.stdin.on("end", () => {
  let payload;
  try { payload = JSON.parse(raw); } catch { return; }
  for (const job of payload.jobs || []) {
    if (job.status === "submitted" || job.status === "running") {
      const title = job.title || job.input || "";
      console.log([job.job_id, job.status, String(title).replace(/\s+/g, " ").slice(0, 120)].join("\t"));
    }
  }
});
'
        return 0
    fi

    return 1
}

guard_active_jobs() {
    if [ "${FORCE_STOP_DEEP_RESEARCH:-}" = "1" ]; then
        return 0
    fi

    local jobs_json
    jobs_json="$(curl -fsS "${BACKEND_URL}/v1/jobs/async/jobs?limit=200" 2>/dev/null || true)"
    if [ -z "$jobs_json" ]; then
        return 0
    fi

    local active_jobs
    active_jobs="$(printf "%s" "$jobs_json" | active_jobs_from_json || true)"
    if [ -z "$active_jobs" ]; then
        return 0
    fi

    printf "Refusing to stop Deep Research because active jobs are still running or queued:\n"
    printf "%s\n" "$active_jobs" | while IFS="$(printf '\t')" read -r job_id status title; do
        printf "  - %s [%s] %s\n" "$job_id" "$status" "$title"
    done
    printf "\nWait for them to finish, cancel them in the UI, or run:\n"
    printf "  FORCE_STOP_DEEP_RESEARCH=1 ./scripts/stop_deep_research.sh\n"
    exit 1
}

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

stop_service() {
    local name="$1"
    local pid_file="$RUNTIME_DIR/${name}.pid"

    if [ ! -f "$pid_file" ]; then
        printf "%s was not started by ./scripts/start_deep_research.sh\n" "$name"
        return 0
    fi

    local pid
    pid="$(cat "$pid_file")"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        printf "Stopping %s (PID %s)\n" "$name" "$pid"
        kill_pid_tree "$pid"
    else
        printf "%s PID is not running\n" "$name"
    fi

    rm -f "$pid_file"
}

stop_port() {
    local name="$1"
    local port="$2"
    local pids

    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -z "$pids" ]; then
        return 0
    fi

    printf "Stopping %s listener(s) on port %s: %s\n" "$name" "$port" "$pids"
    local pid
    for pid in $pids; do
        kill_pid_tree "$pid"
    done
}

guard_active_jobs

stop_service frontend
stop_service backend
tmux kill-session -t "$FRONTEND_SESSION" 2>/dev/null || true
tmux kill-session -t "$BACKEND_SESSION" 2>/dev/null || true
stop_port frontend "$FRONTEND_PORT"
stop_port frontend-next "$NEXT_PORT"
stop_port backend "$BACKEND_PORT"

printf "Deep Research services stopped.\n"
