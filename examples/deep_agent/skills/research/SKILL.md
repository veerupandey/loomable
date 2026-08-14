---
name: research
description: Deep multimodal research workflow — search, fetch, cite, analyze images, write briefs.
---

# Research deep agent skill

Use this workflow for long-horizon research tasks.

## Loop

1. **Plan** — `write_todos` with concrete steps; keep one item `in_progress`.
2. **Search** — `web_search` for 3–8 candidate sources.
3. **Fetch** — `extract_text` / `fetch_url` on the best URLs. Large results may be
   offloaded to `.offload/` — use `read_file` / `grep` on those paths.
4. **Cite** — `register_source(url, title, summary, quote?)` for every source you rely on.
5. **Images** — `discover_images` on a page, then `fetch_image` + `analyze_image`;
   keep notes under `images/`.
6. **Delegate** — use `task` for isolated sub-research; specialists share this workspace.
7. **Deliver** — write `reports/<slug>.md` with findings + paste `format_bibliography`.
8. **Close** — mark todos completed; store durable user facts via `memory` when available.

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
