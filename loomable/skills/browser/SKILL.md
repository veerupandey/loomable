---
name: browser
description: >
  Drive a headless browser through MCP (Lightpanda recommended). Navigate,
  extract markdown/semantic trees, click/fill forms, and store notes in the
  workspace. Use when the user needs live page interaction beyond fetch_url.
---

# Browser skill (Lightpanda MCP + CDP)

You control the browser through **MCP tools**, not a built-in CDP client.
Preferred server: **Lightpanda** native MCP (`lightpanda mcp`).

## Setup (already configured by the host)

Typical loomable mount::

    mcp_servers=[{
      "id": "lightpanda",
      "description": "Lightpanda headless browser",
      "command": "lightpanda",
      "args": ["mcp"],
    }]

If discovery is on (default for deep agents):

1. `search_mcp(query="lightpanda")` or `search_mcp(query="browser")`
2. `activate_mcp_server(server_id="lightpanda")` when deferred
3. `search_tools` / `activate_tool` for `goto`, `markdown`, `links`, `click`,
   `fill`, `evaluate`, `semantic_tree` (exact names depend on the server)

## Loop

1. **Activate** the browser MCP server if tools are not yet callable.
2. **Navigate** with `goto(url=...)`. Treat "Navigated successfully" as soft —
   verify with a content tool next.
3. **Extract** with `markdown` or `semantic_tree` — never dump full HTML into chat.
4. **Interact** with `click` / `fill` / `evaluate` only when needed.
5. **Persist** useful extracts under workspace `notes/browser/` via `write_file`.
6. **Cite** important URLs with `register_source` when doing research.

## Rules

- Prefer Lightpanda MCP over spinning Playwright yourself.
- CDP (`connect_over_cdp`) is an advanced host integration — only use if the
  host exposed custom CDP tools; do not assume they exist.
- Obey robots / site ToS when the server is started with `--obey_robots`.
- If Lightpanda cannot render a complex SPA, say so and fall back to
  `fetch_url` / `extract_text` or ask the host for a Chrome CDP backend.

## Done

Stop when the user’s browser question is answered and any required notes/report
paths are written. Do not keep clicking after the deliverable exists.
