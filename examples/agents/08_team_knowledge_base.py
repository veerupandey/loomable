"""Team inherits ``knowledge_base=`` — members are Agents, not parse steps.

``Team`` / ``Workflow`` / ``Case`` take the same ``knowledge_base=`` as ``Agent``.
Members without their own KB get the shared search tools.

For pipelines, prefer ``Team(mode="sequential")`` or ``Workflow.step(...)`` with
Agents — each Agent already reads the prior Agent's output. No glue parsers.

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

    researcher = Agent(
        scripted_model(
            [
                {"tool": "search_knowledge", "args": {"query": "ship review", "k": 2}},
                "Policy says: never ship without review.",
            ]
        ),
        role="Researcher",
        goal="Look up ship policy in the knowledge base",
        use_llm_summarizer=False,
        max_tool_iterations=4,
    )
    # Sequential Team: this Agent receives the researcher's output as input.
    advisor = Agent(
        scripted_model([{"echo": "Decision based on prior finding → {input}"}]),
        role="Advisor",
        goal="Decide whether we can ship tonight",
        use_llm_summarizer=False,
    )

    team = Team(
        [researcher, advisor],
        model=scripted_model(["coordinator unused in hard sequential mode"]),
        mode="sequential",
        knowledge_base=KnowledgeBase(sources=[docs], name="knowledge"),
    )
    assert "search_knowledge" in researcher.build().tool_runtime._tools

    result = await team.arun("Can we ship tonight?")
    print((result.output.text() or "").strip())


if __name__ == "__main__":
    asyncio.run(main())
