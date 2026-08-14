"""Build pluggable retrievers and attach via ``Agent(retrievers=...)``.

Prefer the Stable path ``Agent(knowledge_base=...)`` when you only need vector-DB
search (see ``examples/agents/07_knowledge_base.py``). This example uses
experimental ``loomable.retrieval.build_retriever``.

Live model — the agent calls ``search_docs`` itself.
Requires a real LLM key — see ``.env.example``.

Run::

    python examples/advanced/05_build_retriever.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, open_vector_store
from loomable.retrieval import build_retriever, list_strategies

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

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
        model=require_provider(),
        retrievers=[retriever],
        instructions="Use search_docs before answering. Cite what you found.",
        max_tool_iterations=4,
    )
    result = await agent.arun("How do we authenticate to the API?")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
