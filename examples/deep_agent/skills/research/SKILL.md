---
name: research
description: >
  Topic-agnostic deep research — search, fetch, cite, verify, analyze images,
  delegate in parallel, and deliver a Markdown brief under reports/. Use for
  any subject (science, policy, product, incident, market, …).
---

# Research skill (any topic)

You are doing **deep research on whatever topic the user asked**. The domain
changes; the loop does not.

Prefer the packaged skill via ``create_deep_agent(..., profile="research")``
or ``skills=["research"]`` — this example copy mirrors it for demos.

## Loop

1. **Plan** — `write_todos` with concrete steps; keep one item `in_progress`.
2. **Search** — `web_search` until you have 3–5 solid primary candidates.
3. **Fetch** — `extract_text` / `fetch_url`. Large results → `.offload/` slices.
4. **Cite** — `register_source` + `verify_source`; `register_claim` for key findings.
5. **Images** — `discover_images` → `fetch_image` → `analyze_image` when useful.
6. **Delegate** — `task` / `task_batch` for isolated sub-angles.
7. **Compact** — `compact_conversation` when chat is heavy.
8. **Deliver** — `reports/<slug>.md` + bibliography.
9. **Close** — one todo update, then STOP.

## Quality bar

- Prefer primary sources. Never invent URLs.
- Report under `reports/` is required.
