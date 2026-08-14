"""Team inherits ``knowledge_base=`` onto members (high-level API).

``Team``, ``Case``, ``Workflow``, and ``Flow`` take the same ``knowledge_base=``
kwarg as ``Agent``. Members without their own KB get the shared search tools.

Run::

    python examples/agents/08_team_knowledge_base.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from loomable import Agent, Team
from loomable.retrieval import KnowledgeBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _offline import scripted_model  # noqa: E402

ROOT = Path(__file__).resolve().parent / ".team_kb_demo"
ROOT.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    docs = ROOT / "policy.md"
    docs.write_text("# Ship policy\n\nNever ship without review.\n", encoding="utf-8")

    member = Agent(
        scripted_model(
            [
                {"tool": "search_knowledge", "args": {"query": "ship review", "k": 2}},
                "Review required before ship (policy).",
            ]
        ),
        role="Reviewer",
        use_llm_summarizer=False,
        max_tool_iterations=4,
    )
    team = Team(
        [member],
        model=scripted_model(["unused — broadcast runs members directly"]),
        mode="broadcast",
        knowledge_base=KnowledgeBase(sources=[docs], name="knowledge"),
    )
    assert "search_knowledge" in member.build().tool_runtime._tools

    result = await team.arun("Can we ship tonight?")
    print((result.output.text() or "").strip())


if __name__ == "__main__":
    asyncio.run(main())
