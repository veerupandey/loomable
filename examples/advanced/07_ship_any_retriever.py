"""Ship any Retriever — live ``Agent(retrievers=[...])``.

Requires a real LLM key — see ``.env.example``.

Run::

    python examples/advanced/07_ship_any_retriever.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent
from loomable.kernel.contracts import Retriever

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402


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
        model=require_provider(),
        retrievers=[AcmeCatalogRetriever()],
        instructions="Use search_acme to look up SKUs before answering.",
        max_tool_iterations=4,
    )
    result = await agent.arun("What is the widget SKU and price?")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
