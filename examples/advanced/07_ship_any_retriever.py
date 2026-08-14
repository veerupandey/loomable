"""Ship any Retriever — the agent uses it as a named search tool.

Run::

    python examples/advanced/07_ship_any_retriever.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from loomable.agent import Agent, ModelSpec
from loomable.kernel.contracts import Retriever
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall


class AcmeCatalogRetriever(Retriever):
    """Custom retriever you ship — only needs ``name`` + ``retrieve``."""

    name = "search_acme"
    description = "Search the Acme product catalog for SKUs and pricing."

    def __init__(self) -> None:
        self._rows = [
            {"content": "SKU-100 widget — $9", "id": "1"},
            {"content": "SKU-200 sprocket — $12", "id": "2"},
        ]

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        q = (query or "").lower()
        hits = [r for r in self._rows if any(t in r["content"].lower() for t in q.split())]
        return (hits or self._rows)[: max(1, int(k))]


class _Scripted:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        tool_name="search_acme",
                        args={"query": "widget", "k": 1},
                    )
                ],
            )
        return ModelResponse(content="Acme catalog lookup done.")


async def main() -> None:
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_Scripted()),
        retrievers=[AcmeCatalogRetriever()],
        use_llm_summarizer=False,
    )
    result = await agent.arun("What is the widget SKU?")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
