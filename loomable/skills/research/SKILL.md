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

## Loop

1. **Plan** — `write_todos` with concrete steps; keep one item `in_progress`.
2. **Search** — `web_search` until you have 3–5 solid primary candidates
   (or the budget is tight). Prefer primary docs over SEO blogs.
3. **Fetch** — `extract_text` / `fetch_url` on the best URLs. Large results may
   be offloaded to `.offload/` — use `read_file(path, offset, limit)` / `grep`.
4. **Cite** — `register_source(url, title, summary, quote?)` for every source
   you rely on. Call `verify_source` on key URLs. Link findings with
   `register_claim(claim, source_id, quote)`.
5. **Images** (when useful) — `discover_images` → `fetch_image` → `analyze_image`;
   keep notes under `images/`.
6. **Delegate** — `task` / `task_batch` for isolated sub-angles; specialists
   share this workspace. Use `subagent_type` when named specialists exist.
7. **Compact** — if chat is heavy, `compact_conversation` with a short checkpoint.
8. **Deliver** — write `reports/<slug>.md` with findings + `format_bibliography`.
9. **Close** — one `update_todo` to mark done, then **STOP** with a final answer.
   Do not keep updating todos after the report exists.

## Report template

```markdown
# <Title>

## Summary
...

## Findings
...

## Visual evidence
...

## Sources
(from format_bibliography)
```

## Quality bar

- Prefer primary sources over SEO blogs.
- Never invent URLs — only cite `register_source` entries.
- If search fails, say so and use what you have.
- The Markdown report under `reports/` is required — bibliography alone is not enough.
