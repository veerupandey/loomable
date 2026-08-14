"""Deep research — planning, web search, citations, workspace files.

USE WHEN: The task is long-horizon: search sources, take notes, and deliver a
cited brief. ``profile="research"`` loads the bundled research skill, todo
list, workspace FS, web search, URL fetch, and a deliverable gate
(``reports/`` + ``register_source``).

``arun()`` builds the agent. You do not call ``agent.build()``.

Requires a live LLM key — see ``.env.example``.

Run::

    python examples/deep_agent/01_research.py
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

ROOT = Path(__file__).resolve().parent / ".workspace_research"
TOPIC = os.environ.get(
    "DEEP_RESEARCH_TOPIC",
    "How do long-horizon agents use planning, a filesystem, and subagents, "
    "and how is that different from a plain ReAct tool loop?",
)


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
    agent = create_deep_agent(
        require_provider(),
        profile="research",
        workspace=ROOT,
    )
    result = await agent.arun(
        f"{TOPIC}\n\n"
        "Search the web, register the sources you use, and write the brief "
        "to reports/brief.md with a bibliography."
    )
    print(result.output.text() or "(no final text)")
    print("tools:", ", ".join(_tool_names(result)) or "(none)")
    brief = ROOT / "reports" / "brief.md"
    if brief.is_file():
        print("\n--- reports/brief.md ---")
        print(brief.read_text(encoding="utf-8")[:4000])
    else:
        reports = ROOT / "reports"
        found = sorted(p for p in reports.rglob("*") if p.is_file()) if reports.is_dir() else []
        if found:
            print(f"\n--- {found[0].relative_to(ROOT)} ---")
            print(found[0].read_text(encoding="utf-8")[:4000])
        else:
            print(f"No report under {ROOT / 'reports'}")


if __name__ == "__main__":
    asyncio.run(main())
