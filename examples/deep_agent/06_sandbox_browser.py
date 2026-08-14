"""Deep agent with sandbox (Python/shell) + optional Lightpanda browser MCP.

Illustrates the layering:

- **Tools**: ``code_exec`` / ``shell`` → soft sandbox under ``workspace/.sandbox``
- **Skill**: bundled ``browser`` playbook (how to use MCP browser tools)
- **MCP**: Lightpanda (or any browser MCP) supplies the actual browser tools

Run (no network required for the scripted smoke path)::

    python examples/deep_agent/06_sandbox_browser.py

Live Lightpanda (optional)::

    # install/start Lightpanda per https://lightpanda.io/docs/usage/mcp
    DEEP_AGENT_BROWSER=1 python examples/deep_agent/06_sandbox_browser.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from loomable.agent import ModelSpec, create_deep_agent
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.skills import list_bundled_skills

ROOT = Path(__file__).resolve().parent / ".workspace_sandbox_browser"
ROOT.mkdir(parents=True, exist_ok=True)


class _Noop:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        tools = [t.get("function", {}).get("name") for t in (request.tools or [])]
        return ModelResponse(
            content=(
                "sandbox demo ready; "
                f"advertised={sorted(n for n in tools if n)[:12]}"
            )
        )


async def main() -> None:
    assert "browser" in list_bundled_skills()

    mcp_servers = []
    if os.environ.get("DEEP_AGENT_BROWSER") == "1":
        mcp_servers.append(
            {
                "id": "lightpanda",
                "description": "Lightpanda headless browser MCP",
                "command": os.environ.get("LIGHTPANDA_BIN", "lightpanda"),
                "args": ["mcp"],
            }
        )

    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Noop()),
        workspace=ROOT,
        profile="general",
        skills=["browser"],
        web_search=False,
        url_fetch=False,
        citations=False,
        images=False,
        code_exec=True,
        shell=True,
        mcp_servers=mcp_servers or None,
        enable_task_tool=False,
        modalities="text",
        use_llm_summarizer=False,
    )
    built = agent.build()
    catalog = {t.name for t in built.discovery.catalog.tools}
    print("bundled_skills", list_bundled_skills())
    print("has_run_python", "run_python" in catalog or "run_python" in built.tool_runtime._tools)
    print("has_run_shell", "run_shell" in catalog or "run_shell" in built.tool_runtime._tools)
    print("mcp_servers", [s.get("id") for s in mcp_servers] or "(none — set DEEP_AGENT_BROWSER=1)")
    result = await agent.arun("Confirm sandbox tools are available.")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
