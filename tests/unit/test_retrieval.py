"""Unit tests for loomable.retrieval (chunking, ingest, build_retriever)."""

from __future__ import annotations

from pathlib import Path

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.retrieval import (
    Document,
    build_retriever,
    chunk_documents,
    get_strategy,
    list_strategies,
    load_file,
    load_sources,
)


class _Noop:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


def test_list_strategies_includes_builtins() -> None:
    names = set(list_strategies())
    for required in ("auto", "text", "markdown", "code", "html", "pdf"):
        assert required in names


@pytest.mark.asyncio
async def test_markdown_and_code_chunkers(tmp_path: Path) -> None:
    md = Document(
        id="guide",
        text="# Intro\n\nHello\n\n## Details\n\nMore info about auth.\n",
        source="guide.md",
        media_type="text/markdown",
    )
    md_chunks = get_strategy("markdown").chunk(md)
    assert len(md_chunks) >= 2
    assert any(c.name == "Details" for c in md_chunks)

    code = Document(
        id="mod",
        text="class Foo:\n    def bar(self):\n        return 1\n",
        source="mod.py",
        media_type="text/x-python",
    )
    code_chunks = get_strategy("code").chunk(code)
    assert any(c.kind == "class" and c.name == "Foo" for c in code_chunks)


@pytest.mark.asyncio
async def test_load_sources_mixed(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nalpha\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    docs = load_sources(
        [
            tmp_path,
            {"id": "note", "text": "inline note about gamma"},
            "raw string document about delta",
        ]
    )
    ids = {d.id for d in docs}
    assert any("a.md" in i or i.endswith("a.md") for i in ids)
    assert "note" in ids
    assert len(docs) >= 3


@pytest.mark.asyncio
async def test_build_retriever_vector_and_lexical(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "auth.md").write_text(
        "# Auth\n\nUse OAuth2 for login tokens.\n", encoding="utf-8"
    )
    (docs_dir / "billing.md").write_text(
        "# Billing\n\nInvoices and discounts.\n", encoding="utf-8"
    )
    from loomable.kernel.long_term import LongTermStore
    from loomable.providers.vector_store import open_vector_store

    vector = await build_retriever(
        [docs_dir],
        name="docs",
        mode="vector",
        strategy="markdown",
        store=open_vector_store(engine="memory"),  # in-memory for unit speed
    )
    hits = await vector.retrieve("OAuth2 login", k=3)
    assert hits

    lexical = await build_retriever([docs_dir], name="lex", mode="lexical", strategy="auto")
    hits2 = await lexical.retrieve("invoices discounts", k=3)
    assert hits2
    assert "billing" in (hits2[0].get("path") or hits2[0].get("source") or "").lower() or (
        "discount" in hits2[0].get("content", "").lower()
        or "invoice" in hits2[0].get("content", "").lower()
    )


@pytest.mark.asyncio
async def test_build_retriever_alibaba_zvec(tmp_path: Path) -> None:
    pytest.importorskip("zvec")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "auth.md").write_text("# Auth\n\nOAuth2 tokens.\n", encoding="utf-8")
    persist = tmp_path / "docs_zvec"
    vector = await build_retriever(
        [docs_dir],
        name="docs",
        mode="vector",
        strategy="markdown",
        persist_path=persist,
    )
    hits = await vector.retrieve("OAuth2", k=2)
    assert hits
    assert persist.exists()


@pytest.mark.asyncio
async def test_hybrid_retriever_on_agent(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text(
        "# Product\n\nWidget factory APIs.\n", encoding="utf-8"
    )
    retriever = await build_retriever(
        [tmp_path / "readme.md"],
        name="knowledge_search",
        mode="hybrid",
        strategy="markdown",
    )
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_Noop()),
        retrievers=[retriever],
        use_llm_summarizer=False,
    )
    built = agent.build()
    assert "knowledge_search" in built.tool_runtime._tools
    result = await built.tool_runtime._tools["knowledge_search"].invoke(
        {"query": "Widget factory", "k": 2}
    )
    assert not result.is_error
    assert result.content


@pytest.mark.asyncio
async def test_chunk_documents_auto() -> None:
    docs = [
        Document(id="1", text="# X\n\nY", source="x.md", media_type="text/markdown"),
        Document(id="2", text="def z():\n    pass\n", source="z.py"),
    ]
    chunks = await chunk_documents(docs, strategy="auto")
    assert len(chunks) >= 2
