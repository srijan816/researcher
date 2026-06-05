# Oracle App2 Operating Rule

- After every code or config change that affects the deep-research app, sync to Oracle app2 and restart the affected service(s).
- Default to restarting the backend on Oracle after backend, prompt, planner, research, or middleware changes.
- Restart the frontend when UI code changes.
- If the change touches shared runtime behavior and the impact is unclear, restart both backend and frontend.
- Do not leave a changed deployment running on old code just because a job is active.
- Oracle sync/restart entrypoint: `./scripts/sync_app2_deep_research.sh --apply --restart all`
