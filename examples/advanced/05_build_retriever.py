"""Build pluggable retrievers and attach via ``Agent(retrievers=...)``.

``Agent(knowledge_base=sources)`` is the usual path. Use ``build_retriever``
when you need an explicit hybrid/lexical/vector tool.

Offline in-memory store — swap ``engine=`` for zvec/faiss/chroma/milvus/postgres.

Run::

    python examples/advanced/05_build_retriever.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from loomable import Agent, open_vector_store
from loomable.retrieval import build_retriever, list_strategies

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _offline import scripted_model  # noqa: E402

ROOT = Path(__file__).resolve().parent / ".retrieval_demo"
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

    print("strategies", list_strategies())
    retriever = await build_retriever(
        [docs, {"id": "tip", "text": "Prefer hybrid mode for mixed corpora."}],
        name="search_docs",
        mode="hybrid",
        strategy="auto",
        store=open_vector_store(engine="memory"),
    )
    agent = Agent(
        model=scripted_model(
            [
                {"tool": "search_docs", "args": {"query": "OAuth2 login", "k": 2}},
                "Retriever demo complete.",
            ]
        ),
        retrievers=[retriever],
        use_llm_summarizer=False,
    )
    result = await agent.arun("How do we authenticate?")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
