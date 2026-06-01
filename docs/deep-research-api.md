# Deep Research API Integration Guide

This guide documents the current MiniMax Deep Research interface in this repository. It is written for developers who want to start the app, authenticate, submit research jobs, stream progress, retrieve reports, and recover failed jobs without using the browser UI.

The recommended service integration path is the async jobs API under `/v1/jobs/async`. It does not require an interactive approval step and is suitable for backend-to-backend callers.

## Base URLs

Repository root:

```bash
/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research
```

Local defaults started by `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/scripts/run_minimax_deep_research.sh`:

```bash
BACKEND_URL=http://localhost:9000
FRONTEND_URL=http://localhost:3000
SEARXNG_URL=http://localhost:8080
```

The backend exposes:

- Health: `GET /health`
- API key management: `/v1/api-keys`
- Data sources: `GET /v1/data_sources`
- Async research jobs: `/v1/jobs/async/...`
- Interactive browser/chat flow: WebSocket `/websocket` and NAT chat routes

## Start And Stop

These commands can be run from any current working directory:

```bash
cp "/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/deploy/.env.example" "/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/deploy/.env"
# Edit "/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/deploy/.env" and set at least MINIMAX_API_KEY.

"/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/scripts/run_minimax_deep_research.sh"
```

The script:

- Stops stale local backend/frontend processes.
- Installs local editable packages for the AI-Q API frontend, SearXNG/Jina search, and stock quotes.
- Starts SearXNG through Docker Compose if it is not already listening.
- Starts the backend on `http://localhost:9000`.
- Starts the frontend on `http://localhost:3000`.

Stop services:

```bash
"/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/scripts/stop_deep_research.sh"
```

Check whether the backend is ready:

```bash
curl -sS http://localhost:9000/health
```

Expected healthy response:

```json
{"status":"healthy"}
```

If startup appears stuck at `Waiting for backend`, inspect:

```bash
tail -n 200 "/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/.deep-research-runtime/backend.log"
tail -n 200 "/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/.deep-research-runtime/frontend.log"
```

Common startup issues:

- `MINIMAX_API_KEY` is missing.
- Port `9000`, `3000`, `3001`, or `8080` is already in use.
- Docker is not running and SearXNG is not already available.
- `uv lock` or dependency installation failed after a merge.

## Required Server Configuration

For `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/configs/config_cli_minimax_ddgs.yml`:

```bash
MINIMAX_API_KEY=...       # Required. MiniMax M3 is used for routing, clarifying, planning, research, and synthesis.
SEARXNG_URL=http://localhost:8080  # Optional locally. The start script defaults this.
EXA_API_KEY=...           # Optional. Enables Exa as a higher-quality primary search provider when configured.
AIQ_API_KEYS=...          # Optional. Comma-separated static bearer keys for external callers.
REQUIRE_AUTH=true         # Optional. Set true for authenticated external API access.
AIQ_EXTERNAL_HOSTNAMES=research.example.com  # Optional. Hosts treated as external by auth/path filtering.
NAT_JOB_STORE_DB_URL=sqlite+aiosqlite:////Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/jobs.db
AIQ_API_KEY_DB_URL=...    # Optional. Defaults to NAT_JOB_STORE_DB_URL for generated aiq_ keys.
AIQ_SCRAPE_ARTIFACT_DIR=/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/data/scrape_artifacts
```

There is no committed universal working API key in this repository. In local development with `REQUIRE_AUTH=false`, no client API key is required. In an authenticated deployment, use either a generated `aiq_...` key or one of the static keys configured in `AIQ_API_KEYS`.

If you override `NAT_JOB_STORE_DB_URL` and use generated `aiq_...` keys, set `AIQ_API_KEY_DB_URL` to the same database unless you intentionally want keys stored separately.

## Authentication

All examples below use:

```bash
export BASE_URL=http://localhost:9000
AUTH_ARGS=()
if [ -n "${AIQ_API_KEY:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer $AIQ_API_KEY")
fi
```

### Local Development, No Auth

By default `REQUIRE_AUTH=false`. Local callers can omit `Authorization`:

```bash
curl -sS "$BASE_URL/v1/jobs/async/agents"
```

### Static Bearer Keys

For deployments that want a fixed service key, configure:

```bash
export REQUIRE_AUTH=true
export AIQ_API_KEYS='replace-with-a-long-random-secret' # pragma: allowlist secret
```

Callers use:

```bash
export AIQ_API_KEY='replace-with-a-long-random-secret' # pragma: allowlist secret
curl -sS "$BASE_URL/v1/jobs/async/agents" \
  -H "Authorization: Bearer $AIQ_API_KEY"
```

Use a high-entropy value. Do not commit this value to the repo.

### Generated `aiq_...` API Keys

Generated keys are stored hashed in the job database. The plaintext key is returned once.

With auth disabled locally, you can generate a development key:

```bash
curl -sS -X POST "$BASE_URL/v1/api-keys" \
  -H "Content-Type: application/json" \
  -d '{"name":"local-dev"}'
```

Example shape:

```json
{
  "id": "4d2e...",
  "name": "local-dev",
  "prefix": "aiq_abcd1234",
  "created_at": "2026-05-21T15:00:00+00:00",
  "last_used_at": null,
  "key": "aiq_full_plaintext_key_returned_once"
}
```

Use the returned `key`:

```bash
export AIQ_API_KEY='aiq_full_plaintext_key_returned_once' # pragma: allowlist secret
curl -sS "$BASE_URL/v1/jobs/async/agents" \
  -H "Authorization: Bearer $AIQ_API_KEY"
```

In production with `REQUIRE_AUTH=true`, create generated keys using an already verified JWT or existing admin auth path:

```bash
curl -sS -X POST "$BASE_URL/v1/api-keys" \
  -H "Authorization: Bearer $ID_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"partner-service"}'
```

List keys for the current principal:

```bash
curl -sS "$BASE_URL/v1/api-keys" \
  -H "Authorization: Bearer $ID_TOKEN"
```

Revoke a key:

```bash
curl -sS -X DELETE "$BASE_URL/v1/api-keys/{key_id}" \
  -H "Authorization: Bearer $ID_TOKEN"
```

### JWT Auth

The middleware also supports JWT validators. Configure issuer/audience and install/register the validator package used by your deployment:

```bash
REQUIRE_AUTH=true
AIQ_JWT_ISSUER=https://accounts.example.com
AIQ_JWT_AUDIENCE=your-audience
```

Callers then send:

```bash
Authorization: Bearer <jwt>
```

## Direct Async Research API

This is the path external developers should normally use.

### List Agents

```bash
curl -sS "$BASE_URL/v1/jobs/async/agents" \
  "${AUTH_ARGS[@]}"
```

Expected agent:

```json
{
  "agents": [
    {
      "agent_type": "deep_researcher",
      "description": "..."
    }
  ]
}
```

### List Data Sources

```bash
curl -sS "$BASE_URL/v1/data_sources" \
  "${AUTH_ARGS[@]}"
```

With the MiniMax/SearXNG config, the primary source id is usually `web_search`. The configured tool chain includes optional Exa, SearXNG/Jina web search, DDGS fallback, and stock quote lookup.

### Submit A Job

```bash
JOB_ID=$(
  curl -sS -X POST "$BASE_URL/v1/jobs/async/submit" \
    "${AUTH_ARGS[@]}" \
    -H "Content-Type: application/json" \
    -d '{
      "agent_type": "deep_researcher",
      "input": "Research the latest AI technologies and identify what capabilities are still missing. Include current evidence, trade-offs, risks, and practical recommendations.",
      "research_depth": "deeper",
      "data_sources": ["web_search"],
      "expiry_seconds": 604800
    }' | jq -r .job_id
)
echo "$JOB_ID"
```

Submit body:

```json
{
  "agent_type": "deep_researcher",
  "input": "Required research prompt",
  "data_sources": ["web_search"],
  "research_depth": "shallow | deeper | deep",
  "job_id": "optional-custom-id",
  "expiry_seconds": 600
}
```

Fields:

- `agent_type`: Use `deep_researcher` for direct report generation.
- `input`: Natural-language research task. Include scope, output needs, and constraints.
- `data_sources`: Optional data source ids. Omit or use `["web_search"]` for the default configured search source.
- `research_depth`: `shallow`, `deeper`, or `deep`.
- `job_id`: Optional custom id, max 64 chars, alphanumeric plus `_` and `-`.
- `expiry_seconds`: Optional retention window, min `600`, max `604800` seconds. The MiniMax config defaults to `604800` seconds.

Depth behavior:

| Tier | Target sources | Researcher tasks | Parallel researcher batch | Typical use |
| --- | ---: | ---: | ---: | --- |
| `shallow` | 5-10 | up to 2 | 1 | Fast answers, narrow questions |
| `deeper` | 20-40 | up to 5 | 2 | Default balanced reports |
| `deep` | 60-100+ | up to 10 | 2 | Exhaustive scans and market landscapes |

Each researcher may keep up to 2 independent search tool calls in the same model response. The runner still enforces global tool budgets, retries failed tools once, and trims larger search bursts.

### Submit With An Already Approved Plan

The direct async API does not run a separate approve/reject prompt. If you already have a plan, embed it in the input:

```bash
curl -sS -X POST "$BASE_URL/v1/jobs/async/submit" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "deep_researcher",
    "research_depth": "deeper",
    "input": "Research the latest AI technologies and identify what capabilities are still missing.\n\n## Clarification Context\n\n**Approved Research Plan**\n\nTitle: Latest AI Technologies and Missing Capabilities\n\nSections:\n- AI Technology Landscape\n- Recent Evidence and Signals\n- Capability Gaps\n- Adoption Risks and Recommendations"
  }'
```

The deep researcher recognizes this clarification context and treats it as a scope constraint.

The submit response already includes the URLs a caller should use next:

```json
{
  "job_id": "abc123",
  "status": "submitted",
  "agent_type": "deep_researcher",
  "has_report": false,
  "report_ready": false,
  "terminal": false,
  "poll_after_seconds": 10,
  "message": "Research is still running. Poll the status_url or report_url again.",
  "status_url": "/v1/jobs/async/job/abc123",
  "report_url": "/v1/jobs/async/job/abc123/report",
  "state_url": "/v1/jobs/async/job/abc123/state",
  "stream_url": "/v1/jobs/async/job/abc123/stream"
}
```

### Get Job Status

```bash
curl -sS "$BASE_URL/v1/jobs/async/job/$JOB_ID" \
  "${AUTH_ARGS[@]}"
```

Successful responses include direct relative links that another program can
follow without hardcoding endpoint shapes:

```json
{
  "job_id": "abc123",
  "status": "success",
  "agent_type": "deep_researcher",
  "error": null,
  "created_at": "2026-05-21T16:45:19.601033",
  "updated_at": "2026-05-21T16:55:46.206146",
  "has_report": true,
  "report_ready": true,
  "terminal": true,
  "poll_after_seconds": null,
  "message": "Final report is ready.",
  "status_url": "/v1/jobs/async/job/abc123",
  "report_url": "/v1/jobs/async/job/abc123/report",
  "state_url": "/v1/jobs/async/job/abc123/state",
  "stream_url": "/v1/jobs/async/job/abc123/stream"
}
```

For active jobs, `status` is usually `submitted` or `running`,
`report_ready` is `false`, `terminal` is `false`, and `poll_after_seconds`
tells a simple polling client how long to wait before checking again.

For failed/interrupted jobs, `terminal` is `true`; inspect `error` and
`state_url`, and call `resume` only if you want to retry/recover.

Statuses:

- `submitted`
- `running`
- `success`
- `failure`
- `interrupted`
- `not_found`

### Stream Progress With SSE

```bash
curl -N "$BASE_URL/v1/jobs/async/job/$JOB_ID/stream" \
  "${AUTH_ARGS[@]}"
```

SSE frame format:

```text
id: 123
event: artifact.update
data: {"type":"artifact.update","data":{...}}
```

Important event types:

- `stream.mode`: replay/live mode. Historical events are replayed before live events.
- `job.status`: terminal or current job status.
- `job.heartbeat`: long-running job is still alive.
- `workflow.start`, `workflow.end`: agent/subagent phase boundaries.
- `llm.start`, `llm.chunk`, `llm.end`: model activity.
- `tool.start`, `tool.end`: tool calls and results.
- `artifact.update`: todos, source discoveries, cited source uses, output snapshots, files, scrape artifacts.
- `job.recovered`: a usable report was recovered from persisted artifacts after a failure.
- `job.error`: failure details.

Reconnect from the last seen event id:

```bash
curl -N "$BASE_URL/v1/jobs/async/job/$JOB_ID/stream/$LAST_EVENT_ID" \
  "${AUTH_ARGS[@]}"
```

### Retrieve Final Report

```bash
curl -sS "$BASE_URL/v1/jobs/async/job/$JOB_ID/report" \
  "${AUTH_ARGS[@]}" | jq -r .report_markdown
```

Response shape:

```json
{
  "job_id": "abc123",
  "has_report": true,
  "report_ready": true,
  "terminal": true,
  "poll_after_seconds": null,
  "message": "Final report is ready.",
  "status": "success",
  "content_type": "text/markdown",
  "report": "# Research Report\n...",
  "report_markdown": "# Research Report\n...",
  "status_url": "/v1/jobs/async/job/abc123",
  "report_url": "/v1/jobs/async/job/abc123/report",
  "state_url": "/v1/jobs/async/job/abc123/state",
  "stream_url": "/v1/jobs/async/job/abc123/stream",
  "sources_found": 34,
  "sources_cited": 12,
  "found_urls": ["https://example.com/source"],
  "cited_urls": ["https://example.com/source"]
}
```

`has_report` is `false` until the job has produced a final report.
Polling `/report` before the report is ready is not an error; the JSON response
will include `status`, `report_ready: false`, `terminal`, `poll_after_seconds`,
and the same URLs.
If the workflow only recovered an intermediate artifact that drifted away from
the submitted request scope, the API will not treat it as a final report and the
job will stay report-unready until a scope-matching synthesis exists.

If your caller wants the report as a raw Markdown payload instead of JSON:

```bash
curl -sS "$BASE_URL/v1/jobs/async/job/$JOB_ID/report?format=markdown" \
  "${AUTH_ARGS[@]}" \
  -o /tmp/deep-research-report.md
```

If raw Markdown is requested before the final report exists, the endpoint returns
HTTP `202` with a JSON status payload instead of a misleading 404. Retry after
`poll_after_seconds`, or stream `stream_url` for real-time events.

For the currently observed lesson package job, the exact call is:

```bash
curl -sS "$BASE_URL/v1/jobs/async/job/lesson-1779381918993-university-education/report?format=markdown" \
  "${AUTH_ARGS[@]}" \
  -o /tmp/university-education-report.md
```

### Retrieve Artifacts And Source Counts

```bash
curl -sS "$BASE_URL/v1/jobs/async/job/$JOB_ID/state" \
  "${AUTH_ARGS[@]}" | jq .
```

Response includes:

- `artifacts.tools`: tool calls and outputs.
- `artifacts.outputs`: search snapshots, report/file artifacts, durable scrape artifact records.
- `artifacts.sources.found`: count of collected source URLs.
- `artifacts.sources.cited`: count of URLs actually cited in the final report.
- `artifacts.sources.found_urls`
- `artifacts.sources.cited_urls`

Durable scrape artifacts are persisted as compressed JSON under:

```text
${AIQ_SCRAPE_ARTIFACT_DIR:-/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/data/scrape_artifacts}/{job_id}/...
```

Each stored scrape JSON contains URL, title, extraction status, content hash, researcher, tool name, content length, created timestamp, and content.

### Cancel A Running Job

```bash
curl -sS -X POST "$BASE_URL/v1/jobs/async/job/$JOB_ID/cancel" \
  "${AUTH_ARGS[@]}"
```

Cancel sets status to `interrupted` and attempts to cancel the Dask task.

### Resume Or Recover A Failed Job

```bash
curl -sS -X POST "$BASE_URL/v1/jobs/async/job/$JOB_ID/resume" \
  "${AUTH_ARGS[@]}"
```

Resume behavior:

- If the job is still `submitted` or `running`, it returns the current status.
- If the job is already `success`, it returns success.
- If a usable report can be recovered from persisted events, it marks the job `success` without another model call.
- Otherwise, it requeues the same job id with recovered notes, source lists, and output artifacts as virtual files.

## Quality Gates

At completion the runner checks:

- `/report.md` exists as a persisted report artifact.
- Researcher tasks did not end with failure-looking output.
- Cited source count is not far below collected relevant sources.
- The final report emitted `citation_use` events.

If gates fail but a usable report artifact exists, the job may be marked success with recovery metadata and a `job.recovered` event. If no usable report exists, the job is marked `failure`.

## Interactive Plan Approval Flow

There are two different modes:

1. Direct async jobs: `POST /v1/jobs/async/submit`. No separate approval endpoint. The job starts immediately.
2. Interactive chat/UI flow: WebSocket `/websocket` through the browser client. This can ask clarifying questions and show a plan approval prompt before submitting the async job.

The interactive clarifier is configured in `/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/nvda-deep-research/configs/config_cli_minimax_ddgs.yml`:

```yaml
clarifier_agent:
  max_turns: 3
  enable_plan_approval: true
```

Approval responses accepted by the clarifier:

```text
approve
approved
yes
ok
proceed
continue
go ahead
looks good
y
accept
skip
```

Reject responses:

```text
reject
rejected
no
cancel
stop
abort
n
```

Any other response is treated as feedback and the plan is regenerated. Examples:

```text
Focus only on section 3.
Add competitors and risks.
Remove generic background and focus on deployment gaps.
```

Interactive timing:

- HITL prompt response timeout is `300` seconds.
- If a prompt times out, the backend sends `skip`.
- For plan approval, `skip` is treated as approval.
- Plan feedback is capped by `max_plan_iterations`, currently default `10`; after that the clarifier auto-approves the latest plan.
- Clarification questions are capped by `max_turns`, currently `3`.

For service callers that use the chat/WebSocket route and cannot answer HITL prompts, send:

```text
X-AIQ-Mode: headless
```

Headless mode skips the clarifier so the workflow does not wait for user interaction. For most service integrations, prefer the direct async jobs API instead.

## Follow-Ups

The async job API does not mutate a completed job with follow-up questions. Use one of these patterns:

- Submit a new async job whose `input` includes the previous report or the specific follow-up scope.
- Use the interactive chat/WebSocket flow with a stable conversation id if you need conversational follow-up behavior.
- Use `POST /v1/jobs/async/job/{job_id}/resume` only for failed or interrupted jobs, not for asking a new question.

## Job Retention And Expiry

`expiry_seconds` controls how long completed jobs remain available in the job store. Valid request range:

```text
600 seconds to 604800 seconds
```

The MiniMax config defaults to `604800` seconds, which is seven days. Periodic cleanup removes expired job metadata, events, and access rows. Durable scrape artifact files are stored on disk and are not currently deleted by the async job cleanup loop.

## Python Client Example

```python
import json
import os
import sys
import time
from urllib.request import Request, urlopen

BASE_URL = os.getenv("BASE_URL", "http://localhost:9000")
API_KEY = os.getenv("AIQ_API_KEY")


def headers(content_type=False):
    h = {}
    if content_type:
        h["Content-Type"] = "application/json"
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def request_json(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(f"{BASE_URL}{path}", data=data, headers=headers(body is not None), method=method)
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


submit = request_json(
    "POST",
    "/v1/jobs/async/submit",
    {
        "agent_type": "deep_researcher",
        "input": "Research the latest AI technologies and what capabilities are still missing.",
        "research_depth": "deeper",
        "data_sources": ["web_search"],
        "expiry_seconds": 604800,
    },
)
job_id = submit["job_id"]
print("job_id:", job_id)

while True:
    status = request_json("GET", submit["status_url"])
    print(status["message"])
    if status["report_ready"] or status["terminal"]:
        break
    time.sleep(status.get("poll_after_seconds") or 10)

if status["report_ready"]:
    report = request_json("GET", status["report_url"])
    print(report.get("report_markdown") or "")
else:
    print(status, file=sys.stderr)
```

## Node.js Client Example

```js
const BASE_URL = process.env.BASE_URL || "http://localhost:9000";
const API_KEY = process.env.AIQ_API_KEY;

const headers = (json = false) => ({
  ...(json ? { "Content-Type": "application/json" } : {}),
  ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
});

async function api(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { ...headers(Boolean(options.body)), ...(options.headers || {}) },
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

const submit = await api("/v1/jobs/async/submit", {
  method: "POST",
  body: JSON.stringify({
    agent_type: "deep_researcher",
    input: "Research the latest AI technologies and what capabilities are still missing.",
    research_depth: "deeper",
    data_sources: ["web_search"],
    expiry_seconds: 604800,
  }),
});

const jobId = submit.job_id;
console.log("job_id", jobId);

for (;;) {
  const status = await api(submit.status_url);
  console.log(status.message);
  if (status.report_ready || status.terminal) break;
  await new Promise((resolve) => setTimeout(resolve, (status.poll_after_seconds || 10) * 1000));
}

const status = await api(submit.status_url);
if (status.report_ready) {
  const report = await api(status.report_url);
  console.log(report.report_markdown);
} else {
  console.error(status);
}
```

## Operational Checklist For External Developers

1. Confirm backend health with `GET /health`.
2. Decide auth mode:
   - Local dev: no key needed if `REQUIRE_AUTH=false`.
   - Production: use `Authorization: Bearer <aiq_or_static_or_jwt_token>`.
3. Confirm available agents with `GET /v1/jobs/async/agents`.
4. Confirm data sources with `GET /v1/data_sources`.
5. Submit `agent_type=deep_researcher`.
6. Stream `stream_url` or poll `status_url`.
7. Fetch `report_url` when `report_ready=true`.
8. Fetch `/state` for source counts, tool activity, and artifact metadata.
9. Use `/resume` for failed/interrupted jobs.
10. Use `/cancel` for jobs that should stop.
