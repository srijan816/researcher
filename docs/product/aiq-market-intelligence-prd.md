# Product Requirements Document: AI-Q Market Intelligence

Status: Draft v1 for founder review  
Date: 2026-06-22  
Product owner: Founder/operator until a dedicated PM is assigned  
Primary system: `/opt/stacks/app2/deep-research`  
Working product name: AI-Q Market Intelligence

## 1. Executive Decision

Build a SaaS product that turns app2's deep-research engine into an evidence-grade market intelligence workspace for AI builders, founders, investors, analysts, and operator-led content teams.

Do not launch a generic "deep research assistant." OpenAI, Google, Perplexity, and Anthropic already ship broad research/search capabilities. The winning wedge is a repeatable workflow: users create companies, models, markets, vendors, or technologies to monitor; the system produces cited dossiers, change briefs, source ledgers, and scheduled intelligence updates.

The product should sell outcomes:

- "Know what changed in an AI market, company, product category, or competitor set."
- "Get a cited analyst-grade dossier with every claim traceable."
- "Maintain a living market brief that updates on schedule."
- "Export research into memos, newsletters, investor notes, client reports, or internal strategy docs."

## 2. Why This Product, Not Generic Deep Research

### Current frontier-platform overlap

The market has moved quickly. Generic deep research is now a feature, not a standalone moat.

- OpenAI exposes deep research through the Responses API with deep-research models, web search, MCP, file search, and `max_tool_calls` cost controls.
  Source: https://developers.openai.com/api/docs/guides/deep-research
- ChatGPT paid plans include deep research as a native feature.
  Source: https://openai.com/chatgpt/pricing/
- Google Gemini API has a Deep Research Agent that plans, searches, synthesizes, cites, supports MCP, file inputs, and visualizations.
  Source: https://ai.google.dev/gemini-api/docs/interactions/deep-research
- Perplexity sells Sonar Deep Research through API pricing with token, citation, search, and reasoning charges.
  Source: https://docs.perplexity.ai/docs/getting-started/pricing
- Perplexity consumer and enterprise products already package deep research, multiple models, internal files, audit logs, and compliance controls.
  Source: https://www.perplexity.ai/enterprise/pricing
- Anthropic provides Claude web search with citations and charges per search plus token costs.
  Source: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
- Claude paid plans include Research and web search.
  Source: https://www.anthropic.com/pricing

### What has succeeded recently

The strongest AI SaaS outcomes have not come from undifferentiated model wrappers. They have come from vertical or workflow-specific systems with trust, workflow depth, and enterprise usefulness.

- Harvey raised at an $11B valuation by focusing on legal work, custom agents, and embedded legal engineering rather than generic chat.
  Source: https://www.harvey.ai/blog/harvey-raises-growth-round-at-dollar11-billion-valuation-co-led-by-gic-and-sequoia
- Hebbia raised $130M Series B by positioning as an AI product layer for financial institutions and law firms doing high-stakes analysis.
  Source: https://www.hebbia.com/blog/hebbia-raises-usd130m-series-b
- Glean raised $150M Series F at a $7.2B valuation around enterprise knowledge and workplace AI.
  Source: https://www.glean.com/blog/glean-series-f-announcement

The lesson: sell a workflow with domain-specific trust and integration, not "we also summarize webpages."

## 3. App2 Baseline: What Already Exists

App2 is already a strong technical foundation.

Verified locally on 2026-06-22:

- Live backend health is passing on `127.0.0.1:9000/health`.
- Live frontend is running on `127.0.0.1:3110`.
- Docker stack includes backend, frontend, Postgres, SearXNG, Websurfx, Redis, and Kokoro.
- Postgres has real usage: 422 total jobs, 354 success, 59 interrupted, 9 failure.
- Recent jobs show long-running research workflows completing successfully with citation verification and reference rebuilding.
- Logs also show real reliability issues: permission denied mirroring `.deep-research-runtime`, source-quality warnings, model retry failures, worker heartbeat loss, and Next.js stale server-action errors after deployments.

Existing product capabilities from app2:

- Next.js research UI with auth-required mode, research-depth selector, engine selector, data-source controls, progress tabs, citations, and report rendering.
- FastAPI async job API.
- Research depths: shallow, medium, deeper, deep.
- Search through SearXNG/Websurfx/DDGS and optional Exa/local corpus tools.
- Durable scrape artifacts and job event persistence.
- Evidence packet, fact ledger, source quality checks, citation verification, and reference rebuilding.
- API keys, local auth, job ownership, conversation persistence, queue, scheduled/recurring jobs, webhooks.
- Markdown report artifacts and PDF/export-related UI code.

Important local references:

- Core README: `/opt/stacks/app2/deep-research/README.md`
- API README: `/opt/stacks/app2/deep-research/frontends/aiq_api/README.md`
- Job API route: `/opt/stacks/app2/deep-research/frontends/aiq_api/src/aiq_api/routes/jobs.py`
- Depth budgets: `/opt/stacks/app2/deep-research/src/aiq_agent/common/research_depth.py`
- Live compose file: `/opt/stacks/app2/deep-research/deploy/compose/docker-compose.app2.yaml`
- Research workflow doc: `/opt/stacks/app2/deep-research/docs/deep-research-pipeline.md`
- API doc: `/opt/stacks/app2/deep-research/docs/deep-research-api.md`

## 4. Product Options Evaluated

### Option A: Evidence-grade AI market intelligence workspace

Build a workspace for users to research and monitor AI companies, models, APIs, categories, regulations, pricing, benchmarks, funding, launches, and competitive movement.

Strengths:

- Strong fit with app2's cited reports, source ledger, scheduled jobs, and depth tiers.
- Differentiated from generic chat because the product owns a repeatable workflow.
- Attractive to AI founders, analysts, investors, content operators, agencies, and product teams.
- Can start self-serve and grow into team/API/enterprise.
- Can use public web sources first, then add private docs later.

Risks:

- Must prove report quality and freshness.
- Needs strong source controls to avoid stale or weak claims.
- Pricing must be credit-based because deep jobs have variable cost.

Decision: Choose this.

### Option B: Deep research API for developers

Expose app2 as a hosted async research API with webhooks, report artifacts, source ledgers, and depth controls.

Strengths:

- App2 already has much of the API surface.
- Easier to meter than a broad UI product.
- Developer buyers understand async jobs and usage-based billing.

Risks:

- Directly competes with OpenAI, Google, Perplexity, Anthropic, Exa, and emerging retrieval APIs.
- Requires strong latency, uptime, SDKs, docs, rate limits, and support.
- Weak moat unless app2 offers unique source policies, artifact transparency, or deployability.

Decision: Add as a secondary channel after the workspace is useful.

### Option C: Enterprise private diligence workspace

Build for legal, finance, procurement, compliance, or investment diligence with private document ingestion plus web research.

Strengths:

- Highest willingness to pay.
- Strong precedent from Harvey, Hebbia, and Glean.
- Private data plus cited external research can be valuable.

Risks:

- Requires SSO, SCIM, RBAC, audit logs, private connectors, retention policies, security reviews, possibly SOC 2, and support.
- App2 is not yet hardened enough.
- Sales cycle is longer.

Decision: Do not start here. Build toward this after app2 has reliability, tenancy, and source governance.

## 5. Product Thesis

AI professionals do not need another chat box. They need a dependable research operating system that watches markets, gathers primary sources, flags what changed, produces cited memos, and preserves the evidence trail.

AI-Q Market Intelligence will be the "living analyst desk" for AI markets.

The user experience should be:

1. Create a watchlist item: company, model, market, policy topic, API, product category, or competitor set.
2. Choose a research objective: brief, dossier, comparison, pricing monitor, launch analysis, benchmark monitor, or risk scan.
3. Approve or edit the research plan.
4. Receive progress, sources, gaps, and report artifacts while the job runs.
5. Review a final cited report with evidence score, source quality, unresolved gaps, and changed-since-last-run summary.
6. Export or schedule recurring updates.

## 6. Target Customers

### Primary ICP: AI builders and operators

Examples:

- AI startup founders tracking competitors, pricing, models, funding, and customer positioning.
- Product managers evaluating APIs, model providers, and feature gaps.
- Developer-relations and content teams producing credible AI market reports.
- Agencies and consultants preparing client strategy briefs.
- Independent analysts and newsletter operators.

Why they buy:

- They already spend time manually reading release notes, docs, pricing pages, benchmarks, blogs, GitHub, and news.
- Existing AI tools answer questions but do not maintain structured watchlists and evidence ledgers.
- They need citations and freshness because AI markets change weekly.

### Secondary ICP: investment and strategy teams

Examples:

- VC scouts and associates.
- Corporate strategy teams.
- M&A/product diligence teams.

Why they buy:

- Repeatable company/category dossiers save analyst hours.
- Source ledgers and changed-since-last-run summaries make updates faster.

### Later ICP: enterprise knowledge teams

Examples:

- Legal, procurement, compliance, and internal research desks.

Why later:

- They need private connectors, SSO, data controls, audit logs, and vendor review.

## 7. Jobs To Be Done

1. When I am evaluating a market, I want a structured cited dossier so I can make a decision without manually collecting 50 sources.
2. When a competitor changes pricing, releases a model, changes terms, or launches a feature, I want to know quickly and see the source.
3. When I publish research, I want citations and source quality warnings so I do not embarrass myself with stale or unsupported claims.
4. When my team researches the same category repeatedly, I want a saved topic with past runs, deltas, and reusable artifacts.
5. When I need to brief a client or investor, I want export-ready output in a consistent format.
6. When a deep run fails, I want a partial artifact, clear reason, and retry/resume path rather than a silent loss.

## 8. Core Product

### 8.1 Research Workspace

The main workspace should be a dense, operational interface, not a landing page. It should include:

- Saved watchlists.
- Recent research runs.
- Draft and completed dossiers.
- Scheduled monitors.
- Credit balance and current queue.
- Failed or warning-state jobs requiring review.

### 8.2 Dossier Builder

Users create research from templates:

- Company dossier.
- Competitor comparison.
- Market landscape.
- Pricing intelligence.
- API/provider evaluation.
- Model benchmark and release monitor.
- Funding and traction scan.
- Regulation/policy tracker.
- Weekly change brief.
- Source-first fact check.

Each template defines:

- Required inputs.
- Preferred source classes.
- Output schema.
- Minimum source requirements.
- Required freshness window.
- Depth recommendation.
- Evidence scoring rubric.

### 8.3 Watchlists

Watchlists are persistent monitored entities.

Watchlist types:

- Company.
- Product.
- Model.
- API provider.
- Market/category.
- Person/team.
- Regulation/policy topic.
- GitHub repository.
- Pricing page.
- Custom URL set.

Watchlist fields:

- Name.
- Description.
- Entity type.
- Canonical URLs.
- Competitors.
- Required sources.
- Blocked sources.
- Update cadence.
- Default template.
- Owner/team.
- Notification destinations.

### 8.4 Scheduled Intelligence

Users schedule recurring jobs:

- Daily change scan.
- Weekly brief.
- Monthly landscape refresh.
- Triggered URL/pricing monitor.
- Manual deep-dive.

Scheduled outputs:

- Changed facts.
- New sources.
- Removed or stale claims.
- Pricing changes.
- Launches.
- Funding/news.
- Benchmark changes.
- Suggested follow-up questions.

### 8.5 Evidence Ledger

Every report must expose an evidence layer:

- Sources found.
- Sources cited.
- Primary vs secondary vs user-provided source classification.
- Citation verification status.
- Unsupported claims.
- Numeric claims needing stronger sources.
- Stale source warnings.
- Source freshness.
- Removed/invalid citations.
- Gaps and unresolved questions.

This should be treated as a first-class product surface, not hidden logs.

### 8.6 Report Outputs

MVP report types:

- Executive brief.
- Full dossier.
- Comparison table.
- Change brief.
- Source ledger.

Export formats:

- Markdown.
- PDF.
- Copy-to-clipboard.
- Public/private share link.
- Webhook JSON.

Post-MVP exports:

- DOCX.
- Google Docs.
- Notion.
- Airtable.
- CSV tables.
- API artifact bundle.

### 8.7 API

The API should be a paid feature, not the initial positioning.

MVP API endpoints:

- Submit research job.
- Get job status.
- Stream job events.
- Fetch report.
- Fetch evidence ledger.
- Fetch source list.
- Submit watchlist item.
- Configure webhook.

App2 already has much of this through the async job API.

## 9. MVP Scope

### MVP must include

- Authenticated single-user and team-ready accounts.
- Stripe billing.
- Credit balance and credit debits by research depth.
- Research workspace.
- Dossier templates.
- Saved reports and job history.
- Watchlists.
- Scheduled weekly monitor.
- Source/evidence panel.
- Report quality warnings visible in UI.
- Markdown/PDF export.
- Webhook callback for completed jobs.
- Admin panel for job inspection and user support.
- Usage and cost tracking per job.
- Basic email notifications.
- Terms, privacy policy, refund/failed-job policy.

### MVP should not include

- Full enterprise SSO.
- SOC 2 claims.
- Private Slack/GDrive/Notion connectors.
- Browser extension.
- Mobile app.
- Multi-model marketplace.
- Generic chat-first experience.
- Unsupported claims like "guaranteed accurate."

## 10. Functional Requirements

### Account and auth

REQ-AUTH-001: Users can sign up, log in, reset password, and manage profile.  
REQ-AUTH-002: Every job belongs to exactly one user and optionally one organization.  
REQ-AUTH-003: Users cannot access another user's jobs, reports, artifacts, or webhooks.  
REQ-AUTH-004: API keys are scoped to user or organization and can be revoked.  
REQ-AUTH-005: Admin users can inspect jobs for support with audit logging.

### Organizations

REQ-ORG-001: Users can create an organization.  
REQ-ORG-002: Organization owners can invite members.  
REQ-ORG-003: Roles: owner, admin, member, viewer.  
REQ-ORG-004: Billing, credits, watchlists, and reports can belong to an organization.  
REQ-ORG-005: Team report sharing is controlled by org membership.

### Research job submission

REQ-JOB-001: User can submit a research prompt from a template or freeform input.  
REQ-JOB-002: User can choose depth: quick, standard, dossier, exhaustive. Internally these map to app2 shallow, medium, deeper, deep.  
REQ-JOB-003: User sees estimated credits and estimated completion time before starting.  
REQ-JOB-004: User can approve or edit the generated plan before expensive runs.  
REQ-JOB-005: System creates idempotent job IDs to avoid double-billing on retry.  
REQ-JOB-006: Jobs can be cancelled. Cancelled jobs charge only for consumed resources or a fixed partial credit according to policy.

### Job execution and recovery

REQ-RUN-001: Job state transitions must be explicit: queued, planning, awaiting_approval, researching, synthesizing, verifying, completed, completed_with_warnings, failed, cancelled, interrupted, recovered.  
REQ-RUN-002: Interrupted jobs must expose resume if artifacts are sufficient.  
REQ-RUN-003: If final synthesis fails but artifacts exist, system attempts recovery into a partial report.  
REQ-RUN-004: Users must see failure reason in plain language.  
REQ-RUN-005: Failed jobs should not consume full credits unless a usable report is delivered.  
REQ-RUN-006: Backend restarts must not orphan running jobs without marking them recoverable or failed.

### Evidence and quality

REQ-EVID-001: Every final report has a source ledger.  
REQ-EVID-002: Every final report has cited source count and found source count.  
REQ-EVID-003: Every final report has citation verification status.  
REQ-EVID-004: Every final report shows source-quality warnings.  
REQ-EVID-005: Numeric claims must be flagged when not backed by primary or high-quality sources.  
REQ-EVID-006: Reports with serious quality warnings must be marked completed_with_warnings, not clean success.  
REQ-EVID-007: User can inspect removed citations and reasons.  
REQ-EVID-008: User can rerun only the gap-fill stage where possible.

### Watchlists and scheduled jobs

REQ-WATCH-001: User can create a watchlist item.  
REQ-WATCH-002: User can attach canonical URLs and preferred sources.  
REQ-WATCH-003: User can schedule recurring runs.  
REQ-WATCH-004: System shows changes since the previous successful run.  
REQ-WATCH-005: System sends email/webhook when scheduled research completes.  
REQ-WATCH-006: Users can pause, resume, or delete schedules.

### Billing and credits

REQ-BILL-001: User can subscribe through Stripe.  
REQ-BILL-002: User sees current credit balance.  
REQ-BILL-003: Each job reserves credits before execution.  
REQ-BILL-004: Each job finalizes actual credit charge after terminal state.  
REQ-BILL-005: Admin can grant, refund, or adjust credits.  
REQ-BILL-006: Credit ledger is immutable except through explicit adjustment entries.  
REQ-BILL-007: API usage and UI usage share the same credit wallet.

### Admin and support

REQ-ADMIN-001: Admin can search users, orgs, jobs, subscriptions, and failed runs.  
REQ-ADMIN-002: Admin can replay job event timeline.  
REQ-ADMIN-003: Admin can see model/search costs, source counts, duration, warnings, and failure reason.  
REQ-ADMIN-004: Admin can trigger safe retry or refund.  
REQ-ADMIN-005: Admin actions are audit logged.

## 11. Nonfunctional Requirements

### Reliability

Target beta SLOs:

- API health uptime: 99.0%.
- Job acceptance success: 99.0%.
- Completed-or-recovered rate for paid jobs: 95%.
- Clean completion rate for paid jobs: 90%.
- No silent job loss.
- No full charge for failed jobs without usable output.

Target v1 paid SLOs:

- API health uptime: 99.5%.
- Job acceptance success: 99.5%.
- Completed-or-recovered rate: 97%.
- Clean completion rate: 93%.
- P95 queue start time under 2 minutes for standard jobs when capacity is available.

### Performance

Initial completion targets:

- Quick scan: 1-3 minutes.
- Standard brief: 4-10 minutes.
- Dossier: 10-30 minutes.
- Exhaustive landscape: 30-75 minutes.

These are product targets, not guarantees. Deep research is variable because web search, extraction, and model latency vary.

### Security

- TLS everywhere for public endpoints.
- Secure cookies.
- CSRF protection where applicable.
- API key hashing at rest.
- Secrets only through environment/secret manager, never committed.
- Per-user and per-org row-level access checks in backend.
- Rate limits by user/org/API key/IP.
- Prompt-injection mitigations for untrusted web content.
- Artifact retention policy.
- Admin access audit logs.

### Privacy and compliance

MVP positioning must avoid enterprise compliance claims until verified.

Required policies:

- Terms of service.
- Privacy policy.
- Data retention policy.
- Acceptable use policy.
- Copyright/source-use policy.
- Refund and failed-job credit policy.

## 12. Pricing and Packaging

Pricing must respect consumer anchors. ChatGPT, Claude, and Perplexity have strong $20/month anchors. A generic research wrapper cannot start at $499/month.

Use credits because costs vary by depth.

Recommended credit mapping:

- Quick scan: 1 credit.
- Standard brief: 3 credits.
- Evidence dossier: 8 credits.
- Exhaustive landscape: 20 credits.
- Scheduled change scan: 1-2 credits depending on scope.

Recommended launch pricing:

| Plan | Price | Included usage | Target user |
| --- | ---: | --- | --- |
| Free trial | $0 | 3 credits, no API | Evaluation |
| Solo | $39/mo | 20 credits/mo | Independent analyst/founder |
| Pro | $99/mo | 70 credits/mo | Heavy individual/operator |
| Team | $299/mo | 250 credits/mo, 3 seats | Small team/agency |
| Growth/API | $799/mo | 800 credits/mo, API/webhooks, priority queue | Teams building repeatable workflows |
| Enterprise | Custom | SSO, retention, private sources, support | Later only |

Overage recommendation:

- $2 per additional credit for Solo/Pro.
- $1.50 per additional credit for Team/Growth.
- Hard monthly spend cap by default.

Pricing rationale:

- Low enough to compete with AI subscriptions.
- High enough to cover variable search/model costs.
- Credits make deep runs economically bounded.
- Team/Growth tiers monetize scheduled monitoring and API usage.

Before public launch, instrument actual COGS per job:

- Model input/output tokens.
- Search calls.
- Extraction calls.
- Job duration.
- Retry count.
- Sources fetched.
- Sources cited.
- Failed/recovered ratio.
- Human support minutes.

Do not finalize margins until 200+ representative paid-intent jobs are measured.

## 13. Data Model

New or formalized SaaS tables:

- `users`
- `organizations`
- `organization_members`
- `subscriptions`
- `credit_wallets`
- `credit_ledger`
- `plans`
- `plan_limits`
- `watchlists`
- `watchlist_sources`
- `scheduled_research`
- `research_templates`
- `research_runs`
- `research_run_costs`
- `research_run_quality`
- `research_reports`
- `source_ledgers`
- `report_exports`
- `webhook_endpoints`
- `webhook_deliveries`
- `admin_audit_log`

Existing app2 tables to preserve or extend:

- `job_info`
- `job_events`
- `job_access`
- `api_keys`
- `research_queue`
- `ui_conversations`
- `local_auth_users`

Critical rule: billing state must not be inferred only from job status. Billing needs its own immutable credit ledger.

## 14. Architecture

Initial architecture:

1. Next.js UI.
2. FastAPI app2 backend.
3. Postgres for users, jobs, events, billing state, schedules.
4. Redis or Postgres queue for scheduled jobs and retries.
5. SearXNG/Websurfx/DDGS/optional Exa for retrieval.
6. Model provider layer, initially MiniMax M3 as configured, with provider abstraction for future OpenAI/Gemini/Anthropic fallback.
7. Artifact store for scrape artifacts, evidence packets, reports, and exports.
8. Stripe for subscriptions and payments.
9. Email provider for notifications.
10. Observability stack for logs, metrics, traces, and alerts.

Required architecture changes:

- Move artifacts to a writable durable path or object storage.
- Make job runner restart-safe.
- Add cost-metering middleware around model/search calls.
- Add credit reservation/finalization around job lifecycle.
- Add source-quality status to final job status and UI, not only logs.
- Add deployment version handling so stale Next.js server-action requests force reload cleanly.
- Add backups and restore drills for Postgres and artifacts.

## 15. Reliability Fixes Required Before Charging

These are launch blockers:

1. Fix `.deep-research-runtime` permission error.
2. Fix reproducible backend test environment; current `.venv` is broken.
3. Fix reproducible frontend test command; current UI test command cannot find `vitest` from host context.
4. Ensure release containers or CI can run smoke tests.
5. Add job cost tracking.
6. Add credit reservation and refund behavior.
7. Mark jobs with source-quality warnings as warning-state in UI.
8. Add retry/resume policy for heartbeat loss and backend restarts.
9. Add queue concurrency limits by plan.
10. Add DB/artifact backups.
11. Add alerts for failed jobs, interrupted jobs, queue stalls, high model errors, artifact write errors, and webhook failures.
12. Add versioned frontend deployment reload behavior.

## 16. User Experience Requirements

### First-run onboarding

User should not start with a blank chat box.

First screen after signup:

- Choose goal: track competitors, research a company, compare providers, monitor pricing, build a market map, create weekly AI brief.
- Add 1-5 entities or URLs.
- Pick report cadence.
- Generate first quick scan.

### Research creation

The create flow should have:

- Template selector.
- Inputs.
- Source preferences.
- Depth selector with credit estimate.
- Output format.
- Schedule toggle.
- Plan preview.
- Start button.

### Report review

The report page should show:

- Executive summary.
- Key changes.
- Evidence score.
- Warnings.
- Report body.
- Sources.
- Removed/invalid citations.
- Gaps.
- Export/share actions.
- Rerun/gap-fill actions.

### Workspace navigation

Navigation:

- Dashboard.
- Research.
- Watchlists.
- Reports.
- Sources.
- Schedules.
- API.
- Billing.
- Settings.

## 17. Launch Metrics

Activation:

- Signup to first completed report.
- Percentage of users creating a watchlist.
- Percentage approving a plan.

Engagement:

- Reports completed per active user per week.
- Scheduled monitors created.
- Watchlist entities per account.
- Export/share actions per report.

Quality:

- Clean completion rate.
- Completed-with-warnings rate.
- Failure/interruption rate.
- Average sources found/cited by template.
- User rating per report.
- Rerun/gap-fill rate.

Revenue:

- Free-to-paid conversion.
- Credit burn per paid account.
- Gross margin by depth.
- Expansion from Solo to Pro/Team.
- Overage revenue.

Retention:

- Week 4 retained users.
- Scheduled monitor retention.
- Reports viewed after email/webhook notification.

## 18. Go-To-Market

Assumption to validate: the existing website/blog audience is AI builders and AI-curious operators. Validate this with analytics before committing budget.

Initial offer:

"AI market intelligence reports with verifiable sources, delivered on demand or on schedule."

Launch assets:

- Public example dossier: "2026 AI Coding Agent Landscape."
- Public example comparison: "OpenAI vs Anthropic vs Google vs Perplexity Deep Research APIs."
- Public example pricing monitor: "AI search and deep research API pricing tracker."
- Waitlist landing page.
- Founder-led blog posts showing source-led research workflows.
- Weekly free AI market brief generated by the product, with citations.

Conversion path:

1. Reader lands on public report.
2. Report shows source ledger and "create your own monitored dossier."
3. User signs up for free credits.
4. User creates first watchlist.
5. User receives first scheduled change brief.
6. Upgrade prompt appears when credits or schedules run out.

Sales motion:

- Self-serve for Solo/Pro.
- Light sales for Team/Growth.
- No enterprise sales until product has reliability and security posture.

## 19. Build Plan

### Phase 0: Stabilize app2 for paid beta, 1-2 weeks

- Fix artifact permission issue.
- Fix local/CI test environment.
- Add backend and frontend smoke tests.
- Add job status quality propagation.
- Add backup/restore process.
- Add logs/metrics/alerts for job failures.
- Add deployment version reload handling.

Exit criteria:

- A paid-intent test user can run 20 jobs without silent loss.
- Failed jobs are visible and explainable.
- Admin can inspect job timeline and artifacts.
- Smoke tests run from CI or a documented command.

### Phase 1: SaaS shell and billing, 2-4 weeks

- Accounts/orgs.
- Stripe subscriptions.
- Credit wallet and ledger.
- Plan limits.
- Billing page.
- Credit reservation/finalization.
- Admin support panel.
- Basic email notifications.

Exit criteria:

- User can sign up, pay, receive credits, run jobs, and see debits.
- Admin can refund credits.
- Job double-submit does not double-charge.

### Phase 2: Market intelligence workflow, 3-5 weeks

- Watchlists.
- Research templates.
- Scheduled weekly monitors.
- Changed-since-last-run summaries.
- Evidence panel.
- Report library.
- Markdown/PDF exports.

Exit criteria:

- User can create a company/category watchlist and receive a scheduled cited brief.
- Report page exposes quality warnings and source ledger.
- Product no longer feels like generic chat.

### Phase 3: API and growth features, 3-6 weeks

- API docs and keys.
- Webhook management.
- SDK examples.
- Team collaboration.
- Shared report links.
- Public example reports.
- Usage analytics.

Exit criteria:

- Growth plan customer can automate recurring research through API/webhook.
- Public examples drive self-serve signup.

### Phase 4: Enterprise path, later

- SSO/SAML.
- SCIM.
- Audit logs.
- Configurable retention.
- Private connectors.
- DPA/security package.
- Source allow/block policies per org.
- Dedicated deployment option.

## 20. Acceptance Criteria For Public Paid Launch

The product can launch paid only when:

- Billing and credits are live.
- Job cost tracking is live.
- Failed paid jobs do not silently consume full credits.
- App2 test/smoke path is reproducible.
- Artifacts write reliably.
- Reports expose quality warnings.
- Admin can inspect and refund.
- Backups are configured and tested.
- Terms/privacy/refund/source policy are published.
- At least 25 beta users have completed 100+ real research runs.
- Clean/recovered completion rate is at least 95% in beta.
- At least 20% of active beta users create a watchlist or schedule.
- At least 5 users say they would be disappointed if access were removed.

## 21. Major Risks

### Risk: Frontier platforms commoditize the core

Mitigation:

- Focus on watchlists, templates, changed-since-last-run, source policy, evidence ledger, exports, and repeatability.
- Offer provider abstraction later so the product can use frontier deep research models where useful.

### Risk: Costs exceed subscription revenue

Mitigation:

- Credit system.
- Depth limits.
- Queue controls.
- Spend caps.
- Model/search cost tracking.
- Use cheaper modes for monitoring and expensive modes only for full dossiers.

### Risk: Research quality is inconsistent

Mitigation:

- Template-specific source requirements.
- Primary-source preference.
- Evidence scores.
- Warning states.
- Gap-fill reruns.
- Report ratings.
- Regression eval set.

### Risk: Web scraping/source access becomes unreliable

Mitigation:

- Multiple search providers.
- Prefer official APIs and public primary pages.
- Cache source artifacts with retention policy.
- Let users provide canonical URLs.
- Add paid source integrations only when justified.

### Risk: Enterprise buyers ask for security before product is ready

Mitigation:

- Do not sell enterprise prematurely.
- Be honest about beta status.
- Publish clear data practices.
- Build audit/retention/SSO after core retention exists.

## 22. Open Questions

1. What is the actual current website/blog traffic, audience mix, and conversion intent?
2. What is the domain/brand for this product?
3. Should MiniMax remain default, or should app2 add OpenAI/Gemini/Anthropic fallback for reliability?
4. What are current model/search COGS per shallow/medium/deeper/deep job?
5. Which initial templates should be first: company dossier, pricing monitor, competitor comparison, or weekly market brief?
6. Will the product use current local auth or migrate to Clerk/Auth0/Supabase Auth?
7. What retention period is acceptable for artifacts and customer reports?
8. What support promise can be made during beta?

## 23. Immediate Next Actions

1. Instrument COGS per job.
2. Fix artifact permission issue.
3. Fix test environment.
4. Add Stripe credit ledger design.
5. Build watchlist schema.
6. Build first three templates:
   - AI company dossier.
   - Competitor comparison.
   - Pricing/API monitor.
7. Produce one public demo report from app2.
8. Recruit 10 beta users from the existing audience.

## 24. Founder-Level Positioning

Short version:

AI-Q Market Intelligence is a cited research workspace for tracking fast-moving AI markets. It turns messy web research into living dossiers, change briefs, and evidence ledgers that teams can trust, schedule, export, and automate.

Long version:

AI teams waste hours tracking model launches, pricing changes, funding rounds, benchmark claims, docs updates, API shifts, and competitor positioning. Generic AI assistants can answer one-off questions, but they do not preserve a durable evidence trail or maintain living market intelligence. AI-Q Market Intelligence uses app2's deep-research engine to plan, search, collect evidence, verify citations, flag weak claims, and deliver repeatable market dossiers on demand or on schedule.

The moat is not the model. The moat is the workflow, source discipline, watchlist memory, scheduled updates, evidence ledger, and accumulated customer research graph.
