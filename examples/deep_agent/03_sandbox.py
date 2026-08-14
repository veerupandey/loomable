"""Sandbox — run Python (and optional shell) against workspace files.

USE WHEN: The agent should compute on local files instead of guessing.
``code_exec=True`` / ``shell=True`` attach ``run_python`` / ``run_shell`` on a
soft subprocess sandbox under ``workspace/.sandbox``.

Optional browser: set ``DEEP_AGENT_BROWSER=1`` to attach a Lightpanda MCP
server plus the bundled ``browser`` skill
(https://lightpanda.io/docs/usage/mcp).

``arun()`` builds the agent. You do not call ``agent.build()``.

Requires a live LLM key — see ``.env.example``.

Run::

    python examples/deep_agent/03_sandbox.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import create_deep_agent

ROOT = Path(__file__).resolve().parent / ".workspace_sandbox"


def _seed_orders() -> Path:
    data = ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "orders.csv").write_text(
        "order_id,sku,qty,unit_cents\n"
        "1,WIDGET,3,900\n"
        "2,SPROCKET,1,1200\n"
        "3,WIDGET,10,900\n"
        "4,BOLT,40,25\n"
        "5,SPROCKET,2,1200\n",
        encoding="utf-8",
    )
    return data / "orders.csv"


def _tool_names(result) -> list[str]:
    names: list[str] = []
    for outcome in result.tool_activity or []:
        meta = (outcome.result.metadata or {}) if outcome.result else {}
        name = meta.get("tool_name")
        if name:
            names.append(str(name))
    return names


async def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = _seed_orders()

    mcp_servers = None
    skills = None
    if os.environ.get("DEEP_AGENT_BROWSER") == "1":
        skills = ["browser"]
        mcp_servers = [
            {
                "id": "lightpanda",
                "description": "Lightpanda headless browser",
                "command": os.environ.get("LIGHTPANDA_BIN", "lightpanda"),
                "args": ["mcp"],
            }
        ]

    agent = create_deep_agent(
        require_provider(),
        profile="general",
        workspace=ROOT,
        code_exec=True,
        shell=True,
        discovery_core="code",
        require_tools=["run_python", "write_file:reports/"],
        web_search=False,
        url_fetch=False,
        citations=False,
        skills=skills,
        mcp_servers=mcp_servers,
    )
    result = await agent.arun(
        f"Read {csv_path.name} under data/. Call run_python to compute total "
        "revenue in cents and the top SKU by revenue from the CSV. Write "
        "reports/summary.md with those two numbers. Do not guess the totals."
    )
    print(result.output.text() or "(no final text)")
    print("tools:", ", ".join(_tool_names(result)) or "(none)")
    summary = ROOT / "reports" / "summary.md"
    if summary.is_file():
        print("\n--- reports/summary.md ---")
        print(summary.read_text(encoding="utf-8")[:4000])
    else:
        print(f"No report at {summary}")


if __name__ == "__main__":
    asyncio.run(main())
