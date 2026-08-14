"""Team inherits ``knowledge_base=`` — live sequential Agents.

Members without their own KB get the shared search tools. Sequential mode
passes each Agent the previous Agent's output — no parse glue.

Requires a real LLM key — see ``.env.example``.

Run::

    python examples/agents/08_team_knowledge_base.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, Team
from loomable.retrieval import KnowledgeBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

ROOT = Path(__file__).resolve().parent / ".team_kb_demo"
ROOT.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    model = require_provider()
    docs = ROOT / "policy.md"
    docs.write_text("# Ship policy\n\nNever ship without review.\n", encoding="utf-8")

    researcher = Agent(
        model,
        role="Researcher",
        goal="Look up ship policy in the knowledge base",
        instructions="Use search_knowledge. Quote the policy in one sentence.",
        max_tool_iterations=4,
    )
    advisor = Agent(
        model,
        role="Advisor",
        goal="Decide whether we can ship tonight",
        instructions=(
            "You receive the researcher's finding as your input. "
            "Give a clear yes/no decision and one-line reason."
        ),
    )

    team = Team(
        [researcher, advisor],
        model=model,
        mode="sequential",
        knowledge_base=KnowledgeBase(sources=[docs], name="knowledge"),
    )
    assert "search_knowledge" in researcher.build().tool_runtime._tools

    result = await team.arun("Can we ship tonight without a review?")
    print((result.output.text() or "").strip())


if __name__ == "__main__":
    asyncio.run(main())
