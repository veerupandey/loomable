# Research quality checklist

Use this as a final pass before calling the research deliverable done. Pull it
in with `list_skill_resources(name="research")` /
`read_skill_resource(name="research", path="references/checklist.md")` when
you want the detail without loading it into every turn.

## Sources

- [ ] At least 3–5 primary sources for a substantive brief (fewer only when
      the budget is explicitly tight — say so if you fall short).
- [ ] Every claim in the report traces back to a `register_source` entry via
      `register_claim`; no invented URLs.
- [ ] Key sources were checked with `verify_source` (not just fetched once).
- [ ] Prefer primary documents (papers, official docs, filings) over SEO
      blogs or aggregator summaries.

## Findings

- [ ] Findings are concrete (numbers, dates, named entities) — not vague
      generalities.
- [ ] Conflicting sources are noted, not silently dropped.
- [ ] Visual evidence (if used) is captured via `discover_images` /
      `fetch_image` / `analyze_image` and referenced in the report.

## Deliverable

- [ ] A Markdown report exists under `reports/` (bibliography alone is not
      enough).
- [ ] The report follows the template: Summary → Findings → Visual evidence
      (optional) → Sources.
- [ ] `format_bibliography` output is included or pasted into the report.
- [ ] Todos are marked completed in at most one `update_todo` call, then the
      run stops with a final answer — no further tool calls after the report
      exists.

## Process hygiene

- [ ] Large intermediate research dumps went to workspace files, not chat.
- [ ] `compact_conversation` was used if the conversation got long.
- [ ] Specialists (`task` / `task_batch`) were used for isolated sub-angles
      when that kept the parent context smaller.
