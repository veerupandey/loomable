"""Build pluggable retrievers for docs / markdown / code and attach to Agent.

``Agent(knowledge_base=sources)`` is the usual path. Use ``build_retriever``
when you need an explicit hybrid/lexical/vector tool to pass as ``retrievers=``.

This demo uses an in-memory vector store so it runs offline without optional
backends. Swap ``engine="memory"`` for ``"zvec"`` / ``"faiss"`` / ``"chroma"``
/ ``"milvus"`` / ``"postgres"`` when those extras are installed.

Run::

    python examples/advanced/05_build_retriever.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loomable.agent import Agent, ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.providers.vector_store import open_vector_store
from loomable.retrieval import build_retriever, list_strategies

ROOT = Path(__file__).resolve().parent / ".retrieval_demo"
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
                        tool_name="search_docs",
                        args={"query": "OAuth2 login", "k": 2},
                    )
                ],
            )
        return ModelResponse(content="Retriever demo complete.")


async def main() -> None:
    docs = _seed()
    print("strategies", list_strategies())
    store = open_vector_store(engine="memory")
    retriever = await build_retriever(
        [docs, {"id": "tip", "text": "Prefer hybrid mode for mixed corpora."}],
        name="search_docs",
        mode="hybrid",
        strategy="auto",
        store=store,
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
