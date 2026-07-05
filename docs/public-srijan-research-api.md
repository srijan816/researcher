# GenAlphAI Research — Public API

Agentic deep-research engine. Submit a question; it plans the work, runs multi-source web research, adversarially verifies claims against sources, and returns a cited markdown report — optionally illustrated with generated photorealistic images.

```text
Base URL: https://app2.sniperip.com
```

## Authentication

Every `/v1` request requires a bearer token issued by the operator (tokens start with `aiq_` and do not expire until revoked):

```bash
-H 'Authorization: Bearer aiq_YOUR_TOKEN'
```

`GET /about` and `GET /health` are unauthenticated.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/about` | Service info and endpoint map (no auth) |
| GET | `/health` | Liveness check (no auth) |
| POST | `/v1/jobs/async/submit` | Start a research job |
| GET | `/v1/jobs/async/job/{job_id}` | Job status |
| GET | `/v1/jobs/async/job/{job_id}/report` | Final report (JSON, `?format=md`, `?format=docx`) |
| GET | `/v1/jobs/async/job/{job_id}/stream` | SSE progress stream (`/stream/{last_event_id}` to resume) |
| GET | `/v1/jobs/async/job/{job_id}/images/{filename}` | Fetch a generated report image |
| POST | `/v1/jobs/async/job/{job_id}/images/generate` | Backfill images into a finished report |

## Submit a research job

```bash
curl -X POST 'https://app2.sniperip.com/v1/jobs/async/submit' \
  -H 'Authorization: Bearer aiq_YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_type": "deep_researcher",
    "input": "Current best open evidence on compact nuclear batteries",
    "research_depth": "deeper",
    "include_images": true,
    "image_count": 3
  }'
```

Response:

```json
{
  "job_id": "abc123",
  "status": "submitted",
  "agent_type": "deep_researcher",
  "poll_after_seconds": 10,
  "status_url": "/v1/jobs/async/job/abc123",
  "report_url": "/v1/jobs/async/job/abc123/report",
  "state_url": "/v1/jobs/async/job/abc123/state",
  "stream_url": "/v1/jobs/async/job/abc123/stream"
}
```

### Request fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `agent_type` | string | required | Use `"deep_researcher"` |
| `input` | string | required | The research question |
| `research_depth` | string | `"deeper"` | One of `shallow`, `medium`, `deeper`, `deep` (larger tiers use more sources and verification budget) |
| `include_images` | bool | `false` | Opt in to generated photorealistic report images |
| `image_count` | int \| null | `null` | Preferred number of images, **1–4**. Only meaningful with `include_images=true`; when omitted the server uses **3**. Values outside 1–4 return HTTP 422 |
| `data_sources` | string[] \| null | server default | e.g. `["web_search"]` |
| `job_id` | string \| null | auto | Custom ID, `[a-zA-Z0-9_-]`, max 64 chars |
| `expiry_seconds` | int \| null | 86400 | 600–604800 |
| `webhook_url` / `webhook_headers` / `webhook_secret` | — | none | Optional terminal-status webhook (HMAC-signed when a secret is set) |

## Poll status

```bash
curl -H 'Authorization: Bearer aiq_YOUR_TOKEN' \
  'https://app2.sniperip.com/v1/jobs/async/job/abc123'
```

Polling guidance: wait `poll_after_seconds` (typically 10 s) between polls until `terminal: true` or `report_ready: true`. Deep jobs commonly take several minutes; `message` always contains a human-readable next action.

## Get the report

```bash
# JSON envelope (report + source stats)
curl -H 'Authorization: Bearer aiq_YOUR_TOKEN' \
  'https://app2.sniperip.com/v1/jobs/async/job/abc123/report'

# Raw markdown
curl -H 'Authorization: Bearer aiq_YOUR_TOKEN' \
  'https://app2.sniperip.com/v1/jobs/async/job/abc123/report?format=md'

# Word document
curl -H 'Authorization: Bearer aiq_YOUR_TOKEN' -o report.docx \
  'https://app2.sniperip.com/v1/jobs/async/job/abc123/report?format=docx'
```

`?format=md` returns HTTP 202 with a JSON status body while the report is not ready yet.

## Stream progress (SSE)

```bash
curl -N -H 'Authorization: Bearer aiq_YOUR_TOKEN' \
  'https://app2.sniperip.com/v1/jobs/async/job/abc123/stream'
```

Events include tool calls, intermediate artifacts, and the final report. To resume after a disconnect, use `/stream/{last_event_id}` with the last event ID you received.

## Generated images

When `include_images=true`, after the report is finalized the server plans 1–4 (default 3) illustration opportunities and generates **hyper-realistic photographic** images (cinematic photography style; never charts, diagrams, sketches, or text-bearing graphics). Image generation is fail-open: if it fails, the report is delivered unchanged.

Images are embedded in the report markdown as:

```markdown
![Caption](/api/jobs/async/job/abc123/images/image-1.jpg)
```

**Important for API consumers:** embedded image URLs use the `/api/` prefix (an internal UI proxy path). To fetch image bytes over the public API, rewrite `/api/` → `/v1/`:

```text
/api/jobs/async/job/abc123/images/image-1.jpg
  → https://app2.sniperip.com/v1/jobs/async/job/abc123/images/image-1.jpg
```

```bash
curl -H 'Authorization: Bearer aiq_YOUR_TOKEN' -o image-1.jpg \
  'https://app2.sniperip.com/v1/jobs/async/job/abc123/images/image-1.jpg'
```

Image requests carry the same auth/ownership checks as job status. Responses are `image/jpeg` or `image/png`.

## Backfill images into an existing report

Add images to a finished job that was submitted without `include_images` (idempotent — reports that already contain generated images are left untouched):

```bash
curl -X POST -H 'Authorization: Bearer aiq_YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"image_count": 2}' \
  'https://app2.sniperip.com/v1/jobs/async/job/abc123/images/generate'
```

The JSON body is optional; `image_count` accepts 1–4 and defaults to 3 when omitted. Response:

```json
{"ok": true, "images_added": 2, "already_had_images": false}
```

Errors: `409` if the job is not finished or has no usable report text; `500` if planning/generation fails or exceeds its time budget.

## Service info

```bash
curl 'https://app2.sniperip.com/about'   # endpoint map, auth notes (no auth required)
curl 'https://app2.sniperip.com/health'  # {"status": "healthy"}
```
