"""Pluggable agentic retriever → live ``Agent(retrievers=[...])``.

Prefer ``Agent(knowledge_base=...)`` when you only need vector-DB search.
Use this when you want rewrite / rerank / mode routing.

Requires a real LLM key — see ``.env.example``.

Run::

    python examples/advanced/06_agentic_retriever.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, open_vector_store
from loomable.retrieval import build_agentic_retriever, ingest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

ROOT = Path(__file__).resolve().parent / ".agentic_demo"
ROOT.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "auth.md").write_text(
        "# Authentication\n\nUse OAuth2 bearer tokens for API login.\n",
        encoding="utf-8",
    )
    (docs / "pricing.md").write_text(
        "# Pricing\n\nEnterprise plans include SSO and audit logs.\n",
        encoding="utf-8",
    )

    corpus = await ingest(
        [docs],
        name="docs",
        description="Product docs: auth and pricing",
        store=open_vector_store(engine="memory"),
        strategy="auto",
        base_mode="hybrid",
    )
    retriever = await build_agentic_retriever(
        corpus,
        name="search_docs",
        mode="auto",
        rewrite="off",
        rerank="mmr",
        compress="off",
    )
    agent = Agent(
        model=require_provider(),
        retrievers=[retriever],
        instructions="Use search_docs before answering.",
        max_tool_iterations=4,
    )
    result = await agent.arun("How do we authenticate?")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
