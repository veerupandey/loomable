"""Pluggable agentic retriever: ingest → hybrid → auto-route → agent tool.

Run::

    python examples/advanced/03_agentic_retriever.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loomable.agent import Agent, ModelSpec
from loomable.kernel.long_term import open_vector_store
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.retrieval import build_agentic_retriever, ingest

ROOT = Path(__file__).resolve().parent / ".agentic_demo"
ROOT.mkdir(parents=True, exist_ok=True)


def _seed() -> Path:
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
    return docs


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
                        tool_name="docs",
                        args={"query": "OAuth2 login", "k": 2},
                    )
                ],
            )
        return ModelResponse(content="Agentic retrieve complete.")


async def main() -> None:
    docs = _seed()
    # Ingest (pluggable store — memory here; use open_vector_store(path=...) for zvec)
    corpus = await ingest(
        [docs],
        name="docs",
        description="Product docs: auth and pricing",
        store=open_vector_store(engine="memory"),
        strategy="auto",
        base_mode="hybrid",
    )
    # Agentic stack — every stage swappable
    retriever = await build_agentic_retriever(
        corpus,
        mode="auto",       # or "chunks" | "file" | custom ModeRouter
        rewrite="off",     # or "multi_query" / "hyde" with llm=
        rerank="score",    # or "llm" / custom Reranker
        compress="off",    # or "llm" / custom HitCompressor
    )
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_Scripted()),
        retrievers=[retriever],
        use_llm_summarizer=False,
    )
    result = await agent.arun("How do we authenticate?")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
