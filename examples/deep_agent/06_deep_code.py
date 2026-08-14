"""Deep code: index a repo (zvec) + coding skill + sandbox.

Run::

    python examples/deep_agent/06_deep_code.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loomable.agent import ModelSpec, create_deep_agent
from loomable.codeindex import CodeIndex
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall

ROOT = Path(__file__).resolve().parent / ".workspace_deep_code"
REPO = ROOT / "sample_repo"
ROOT.mkdir(parents=True, exist_ok=True)


def _ensure_repo() -> Path:
    pkg = REPO / "app"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "billing.py").write_text(
        "class Invoice:\n"
        "    def total(self, items: list[int]) -> int:\n"
        "        return sum(items)\n"
        "\n"
        "def apply_discount(amount: int, pct: int) -> int:\n"
        "    return amount - (amount * pct // 100)\n",
        encoding="utf-8",
    )
    return REPO


class _Scripted:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(id="1", tool_name="repo_map", args={})],
            )
        if self.n == 2:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="2",
                        tool_name="code_search",
                        args={"query": "discount invoice total", "limit": 3},
                    )
                ],
            )
        return ModelResponse(
            content="Deep code demo: located billing helpers via repo_map + code_search."
        )


async def main() -> None:
    repo = _ensure_repo()
    index = await CodeIndex.build(repo, persist_path=ROOT / "codeindex.zvec.json")
    print("indexed_chunks", index.size)
    print(index.repo_map(max_entries=20))

    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Scripted()),
        workspace=ROOT / "ws",
        profile="code",
        code_index=index,
        enable_task_tool=False,
        use_llm_summarizer=False,
    )
    result = await agent.arun("Where is discount logic?")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
