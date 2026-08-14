"""Agentic + pluggable retrieval tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.providers.vector_store import open_vector_store
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.retrieval import (
    AgenticRetriever,
    CompositeRetriever,
    HeuristicModeRouter,
    IdentityRewriter,
    MultiQueryRewriter,
    ScoreReranker,
    build_agentic_retriever,
    ingest,
)


class _Noop:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


class _ScriptedRetrieve:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id="1", tool_name="docs", args={"query": "OAuth2 login", "k": 2})
                ],
            )
        return ModelResponse(content="retrieved")


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text

    async def __call__(self, prompt: str) -> str:
        return self.text


def _seed(tmp: Path) -> Path:
    docs = tmp / "docs"
    docs.mkdir()
    (docs / "auth.md").write_text(
        "# Auth\n\nUse OAuth2 bearer tokens for API login.\n", encoding="utf-8"
    )
    (docs / "billing.md").write_text(
        "# Billing\n\nInvoices and discounts for enterprise.\n", encoding="utf-8"
    )
    return docs


@pytest.mark.asyncio
async def test_ingest_and_agentic_hybrid(tmp_path: Path) -> None:
    docs = _seed(tmp_path)
    corpus = await ingest(
        [docs],
        name="docs",
        store=open_vector_store(engine="memory"),
        strategy="markdown",
        base_mode="hybrid",
    )
    assert corpus.chunks
    retriever = AgenticRetriever(
        corpus, mode="chunks", rewrite="off", rerank="score", compress="off"
    )
    hits = await retriever.retrieve("OAuth2 bearer tokens", k=3)
    assert hits
    assert any("oauth" in (h.get("content") or "").lower() for h in hits)
    assert hits[0].get("corpus") == "docs"


@pytest.mark.asyncio
async def test_file_mode_router(tmp_path: Path) -> None:
    docs = _seed(tmp_path)
    corpus = await ingest(
        [docs], name="docs", store=open_vector_store(engine="memory"), strategy="markdown"
    )
    retriever = AgenticRetriever(corpus, mode="auto", rewrite="off", rerank=False)
    hits = await retriever.retrieve("What does auth.md say about login?", k=3)
    assert hits
    assert any(h.get("retrieval_mode") == "file" for h in hits)


@pytest.mark.asyncio
async def test_multi_query_rewriter_pluggable() -> None:
    llm = _FakeLLM("OAuth tokens\nAPI authentication\nbearer login")
    rw = MultiQueryRewriter(llm, n=3, include_original=True)
    qs = await rw.rewrite("How do we log in?")
    assert qs[0] == "How do we log in?"
    assert len(qs) >= 2


@pytest.mark.asyncio
async def test_custom_plugins_injected(tmp_path: Path) -> None:
    docs = _seed(tmp_path)
    corpus = await ingest(
        [docs], name="docs", store=open_vector_store(engine="memory"), base_mode="vector"
    )

    class OnlyBillingRouter:
        name = "only_billing"

        async def choose_mode(self, query: str):
            return "chunks"

    class PrefPrefixRewriter:
        name = "prefix"

        async def rewrite(self, query: str) -> list[str]:
            return [f"billing {query}"]

    retriever = AgenticRetriever(
        corpus,
        mode=OnlyBillingRouter(),
        rewrite=PrefPrefixRewriter(),
        rerank=ScoreReranker(),
        compress="off",
    )
    hits = await retriever.retrieve("discounts", k=2)
    assert hits


@pytest.mark.asyncio
async def test_composite_router(tmp_path: Path) -> None:
    docs = _seed(tmp_path)
    a = await ingest(
        [docs / "auth.md"],
        name="auth",
        description="Authentication and OAuth",
        store=open_vector_store(engine="memory"),
    )
    b = await ingest(
        [docs / "billing.md"],
        name="billing",
        description="Invoices and pricing",
        store=open_vector_store(engine="memory"),
    )
    composite = CompositeRetriever([a, b], name="knowledge", corpus_router="all")
    hits = await composite.retrieve("OAuth2", k=3)
    assert hits


@pytest.mark.asyncio
async def test_build_agentic_retriever_and_agent(tmp_path: Path) -> None:
    docs = _seed(tmp_path)
    retriever = await build_agentic_retriever(
        [docs],
        name="docs",
        store=open_vector_store(engine="memory"),
        mode="chunks",
        rewrite=IdentityRewriter(),
        rerank=ScoreReranker(),
        base_mode="hybrid",
    )
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_ScriptedRetrieve()),
        retrievers=[retriever],
        use_llm_summarizer=False,
    )
    result = await agent.arun("How do we authenticate?")
    assert "retrieved" in (result.output.text() or "").lower()


@pytest.mark.asyncio
async def test_corpus_upsert_skips_unchanged(tmp_path: Path) -> None:
    docs = _seed(tmp_path)
    corpus = await ingest(
        [docs], name="docs", store=open_vector_store(engine="memory")
    )
    n1 = len(corpus.chunks)
    added = await corpus.upsert([docs], rebuild=False)
    assert added == 0
    assert len(corpus.chunks) == n1
    (docs / "auth.md").write_text("# Auth\n\nUpdated OAuth2 flow.\n", encoding="utf-8")
    added2 = await corpus.upsert([docs / "auth.md"], rebuild=False)
    assert added2 >= 1


@pytest.mark.asyncio
async def test_heuristic_mode_router() -> None:
    r = HeuristicModeRouter()
    assert await r.choose_mode("explain OAuth2") == "chunks"
    assert await r.choose_mode("open auth.md") == "file"
