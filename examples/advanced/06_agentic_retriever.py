"""Pluggable agentic retriever → ``Agent(retrievers=[...])``.

Prefer ``Agent(knowledge_base=store_or_sources)`` when you only need a
vector-DB search tool. Use this when you want rewrite / rerank / mode routing.

Run::

    python examples/advanced/06_agentic_retriever.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from loomable import Agent, open_vector_store
from loomable.retrieval import build_agentic_retriever, ingest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _offline import scripted_model  # noqa: E402

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
        model=scripted_model(
            [
                {"tool": "search_docs", "args": {"query": "OAuth2 login", "k": 2}},
                "Agentic retrieve complete.",
            ]
        ),
        retrievers=[retriever],
        use_llm_summarizer=False,
    )
    result = await agent.arun("How do we authenticate?")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
