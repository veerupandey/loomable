# AGENTS.md

## Cursor Cloud specific instructions

`loomable` is a single Python library (package `loomable/`, requires Python >= 3.11)
for building AI agents (Agent / Loop / Flow), plus optional serving adapters. It is
a pip/uv-installable SDK, not a standing service. Package manager is **uv**
(`pyproject.toml` + `uv.lock`).

### Environment
- Dependencies are refreshed automatically by the startup update script
  (`uv sync --extra toolkits`, which also installs the `dev` dependency group).
  No manual install is needed at session start.
- `uv` is installed at `~/.local/bin` and is on `PATH` via `~/.bashrc`. Prefer
  `uv run <cmd>` so commands use the project virtualenv (`.venv`).

### Test / run commands (nothing here needs API keys — providers are stubbed)
- Tests: `uv run pytest tests/` (README documents this). Individual dirs:
  `tests/unit`, `tests/integration`, `tests/properties`, `tests/toolkits`,
  `tests/benchmarks`.
- No linter/formatter is configured (no ruff/flake8/black config). "Lint" is not
  a defined step; use `uv run python -m compileall loomable` for a static sanity
  check if needed.
- The runnable **application** is the FastAPI serving adapter
  (`loomable/serve/fastapi_adapter.py`, `FastAPIAdapter(built_agent).app()`),
  run in dev mode with `uv run uvicorn <module>:app --reload`. It exposes
  `GET /health`, `POST /run`, `POST /run/stream`. There is also an MCP stdio
  adapter (`loomable/serve/mcp_adapter.py`).

### Non-obvious caveats
- To exercise an agent end-to-end without credentials, inject a scripted provider
  via `ModelSpec(provider="scripted", provider_impl=<obj with async complete()>)`
  and pass `capabilities=ModelCapabilities()` — this is the pattern the test suite
  uses. Real provider classes (`OpenAIProvider`, `AzureOpenAIProvider`, etc.) read
  credentials from env vars; the example scripts in `examples/` need a live
  provider key (e.g. `OPENAI_API_KEY`, or the `AZURE_OPENAI_*` trio).
- `result.tool_activity` holds `ToolOutcome` objects (`call_id`, `result`,
  `error`) — there is no `tool_name` on them; read `outcome.result.content`.
- Two tests fail on the current dependency versions and are unrelated to
  application code (do not "fix" them by hand as part of unrelated work):
  - `tests/integration/test_transport_parity.py::test_transport_parity_equivalent_output`
    — the installed `mcp` package exposes `CallToolResult.isError`, but the test
    reads `.is_error` (library API drift).
  - `tests/properties/test_decorated_function_tool.py::...test_tool_description_defaults_to_docstring`
    — `inspect.getdoc` expands tabs, so a docstring like `"0\t0"` no longer equals
    `docstring.strip()`.
- Some example/docs helpers import `loomable.display` (e.g.
  `examples/subagents/01_simple_delegation.py`, `pp()`, `delegation_outputs()`),
  but `loomable/display.py` does not exist in the repo; those specific examples
  will fail on that import.
