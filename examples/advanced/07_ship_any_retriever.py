"""Ship any Retriever — high-level ``Agent(retrievers=[...])``.

Run::

    python examples/advanced/07_ship_any_retriever.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from loomable import Agent
from loomable.kernel.contracts import Retriever

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _offline import scripted_model  # noqa: E402


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


async def main() -> None:
    agent = Agent(
        model=scripted_model(
            [
                {"tool": "search_acme", "args": {"query": "widget", "k": 1}},
                "Acme catalog lookup done.",
            ]
        ),
        retrievers=[AcmeCatalogRetriever()],
        use_llm_summarizer=False,
    )
    result = await agent.arun("What is the widget SKU?")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
