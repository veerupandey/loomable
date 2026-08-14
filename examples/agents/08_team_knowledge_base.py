"""Team inherits ``knowledge_base=`` onto members that do not already have one.

``Team``, ``Case``, ``Workflow``, and ``Flow`` take the same ``knowledge_base=``
kwarg as ``Agent``. Members without their own KB get the shared search tools.

Run::

    python examples/agents/08_team_knowledge_base.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loomable.agent import Agent, ModelSpec, Team
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.retrieval import KnowledgeBase

ROOT = Path(__file__).resolve().parent / ".team_kb_demo"
ROOT.mkdir(parents=True, exist_ok=True)


def _seed() -> Path:
    docs = ROOT / "policy.md"
    docs.write_text(
        "# Ship policy\n\nNever ship without review.\n",
        encoding="utf-8",
    )
    return docs


def _tool_names(request: ModelRequest) -> set[str]:
    names: set[str] = set()
    for t in request.tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = (fn or {}).get("name") or t.get("name")
        if name:
            names.add(str(name))
    return names


class _Member:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        names = _tool_names(request)
        if "search_knowledge" not in names:
            return ModelResponse(content=f"FAIL missing KB tools: {sorted(names)}")
        if self.n == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        tool_name="search_knowledge",
                        args={"query": "ship review", "k": 2},
                    )
                ],
            )
        return ModelResponse(content="Review required before ship (policy).")


class _Coordinator:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="Team consensus: do not ship without review.")


async def main() -> None:
    docs = _seed()
    member = Agent(
        ModelSpec(provider="scripted", provider_impl=_Member()),
        role="Reviewer",
        use_llm_summarizer=False,
        max_tool_iterations=4,
    )
    team = Team(
        [member],
        model=ModelSpec(provider="scripted", provider_impl=_Coordinator()),
        mode="broadcast",
        knowledge_base=KnowledgeBase(sources=[docs], name="knowledge"),
    )
    built = member.build()
    assert "search_knowledge" in built.tool_runtime._tools
    result = await team.arun("Can we ship tonight?")
    print((result.output.text() or "").strip())


if __name__ == "__main__":
    asyncio.run(main())
