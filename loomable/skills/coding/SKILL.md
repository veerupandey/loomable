---
name: coding
description: >
  Explore and change a codebase: repo_map → code_search / find_symbol →
  read slices → plan with todos → edit workspace files → run tests in the
  sandbox. Use for deep code tasks on an indexed repository.
---

# Coding skill (deep code)

You are working on a **real repository** indexed for navigation. Prefer
structured tools over dumping whole files into chat.

## Loop

1. **Orient** — call `repo_map` once to see the layout and key symbols.
2. **Locate** — `code_search(query=...)` for behavior; `find_symbol(name=...)`
   for exact defs. Then `read_file` / `grep` for the slices you need.
3. **Plan** — `write_todos` with concrete edit/test steps; one item in_progress.
4. **Edit** — `write_file` / `edit_file` under the workspace (or repo root the
   host configured). Keep diffs small and purposeful.
5. **Verify** — `run_python` / `run_shell` inside the sandbox (tests, linters).
   Do not use shell for destructive host commands.
6. **Deliver** — summarize what changed, which tests ran, and residual risks.
   Mark todos completed and STOP.

## Rules

- Never paste huge files into chat — use offset/limit reads and search hits.
- Prefer `code_search` before inventing new modules that may already exist.
- If the index looks stale, say so; the host can rebuild `CodeIndex`.
- Sandbox tools may require confirmation — wait for approval when prompted.
- Browser/MCP is out of scope unless the user asked for docs in a browser.

## Done

Stop when the user’s code goal is met (or blocked with a clear reason) and any
requested report/notes paths are written.
