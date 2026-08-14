"""Rigorous tests for pluggable agentic retrieval.

Covers: ingest/upsert, all built-in plugins, resolve helpers, modes,
multi-corpus routing, store backends, agent tool wiring, and edge cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.providers.vector_store import open_vector_store
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.kernel.retrievers import RetrieverTool
from loomable.retrieval import (
    AgenticRetriever,
    AllCorporaRouter,
    CompositeRetriever,
    DescriptionCorpusRouter,
    FixedModeRouter,
    HeuristicModeRouter,
    HyDERewriter,
    IdentityCompressor,
    IdentityReranker,
    IdentityRewriter,
    LLMCompressor,
    LLMModeRouter,
    LLMReranker,
    MultiQueryRewriter,
    ScoreReranker,
    build_agentic_retriever,
    ingest,
)
from loomable.retrieval.plugins import (
    CorpusRouter,
    HitCompressor,
    ModeRouter,
    QueryRewriter,
    Reranker,
)
from loomable.retrieval.rerank import resolve_compressor, resolve_reranker
from loomable.retrieval.rewrite import resolve_rewriter
from loomable.retrieval.route import match_file_sources, resolve_corpus_router, resolve_mode_router


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Deterministic async LLM callable; optional per-call script."""

    def __init__(self, text: str = "", script: list[str] | None = None) -> None:
        self.text = text
        self.script = list(script or [])
        self.calls: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.script:
            return self.script.pop(0)
        return self.text


class _ScriptedAgent:
    def __init__(self, tool_name: str = "docs") -> None:
        self.n = 0
        self.tool_name = tool_name

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        tool_name=self.tool_name,
                        args={"query": "OAuth2", "k": 2},
                    )
                ],
            )
        return ModelResponse(content="done")


def _seed(tmp: Path) -> Path:
    docs = tmp / "docs"
    docs.mkdir()
    (docs / "auth.md").write_text(
        "# Auth\n\nUse OAuth2 bearer tokens for API login.\n\n## MFA\n\nEnable TOTP.\n",
        encoding="utf-8",
    )
    (docs / "billing.md").write_text(
        "# Billing\n\nInvoices and discounts for enterprise plans.\n",
        encoding="utf-8",
    )
    (docs / "readme.txt").write_text(
        "Project readme: deploy with docker compose.\n", encoding="utf-8"
    )
    return docs


async def _corpus(tmp: Path, **kwargs: Any):
    docs = _seed(tmp)
    opts = {
        "name": "docs",
        "description": "Product documentation",
        "store": open_vector_store(engine="memory"),
        "strategy": "auto",
        "base_mode": "hybrid",
    }
    opts.update(kwargs)
    return await ingest([docs], **opts), docs


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_builtin_plugins_satisfy_protocols() -> None:
    assert isinstance(IdentityRewriter(), QueryRewriter)
    assert isinstance(MultiQueryRewriter(_FakeLLM("a")), QueryRewriter)
    assert isinstance(HyDERewriter(_FakeLLM("a")), QueryRewriter)
    assert isinstance(IdentityReranker(), Reranker)
    assert isinstance(ScoreReranker(), Reranker)
    assert isinstance(LLMReranker(_FakeLLM("x")), Reranker)
    assert isinstance(IdentityCompressor(), HitCompressor)
    assert isinstance(LLMCompressor(_FakeLLM("x")), HitCompressor)
    assert isinstance(FixedModeRouter("chunks"), ModeRouter)
    assert isinstance(HeuristicModeRouter(), ModeRouter)
    assert isinstance(LLMModeRouter(_FakeLLM("chunks")), ModeRouter)
    assert isinstance(AllCorporaRouter(), CorpusRouter)
    assert isinstance(DescriptionCorpusRouter(_FakeLLM("a")), CorpusRouter)


# ---------------------------------------------------------------------------
# resolve_* helpers
# ---------------------------------------------------------------------------


def test_resolve_rewriter_variants() -> None:
    assert resolve_rewriter(None).name == "off"
    assert resolve_rewriter("off").name == "off"
    assert resolve_rewriter(False).name == "off"
    custom = IdentityRewriter()
    assert resolve_rewriter(custom) is custom
    with pytest.raises(ValueError, match="multi_query"):
        resolve_rewriter("multi_query")
    with pytest.raises(ValueError, match="hyde"):
        resolve_rewriter("hyde")
    mq = resolve_rewriter("multi_query", llm=_FakeLLM("q1"))
    assert mq.name == "multi_query"
    with pytest.raises(ValueError, match="unknown rewrite"):
        resolve_rewriter("nope")


def test_resolve_reranker_and_compressor() -> None:
    assert resolve_reranker(None).name == "off"
    assert resolve_reranker(False).name == "off"
    assert resolve_reranker(True).name == "mmr"
    assert resolve_reranker("score").name == "score"
    assert resolve_reranker("mmr").name == "mmr"
    assert resolve_reranker("llm", llm=_FakeLLM("id")).name == "llm"
    with pytest.raises(ValueError, match="llm"):
        resolve_reranker("llm")
    with pytest.raises(ValueError, match="unknown rerank"):
        resolve_reranker("magic")

    assert resolve_compressor(None).name == "off"
    assert resolve_compressor("llm", llm=_FakeLLM("x")).name == "llm"
    with pytest.raises(ValueError, match="llm"):
        resolve_compressor(True)
    with pytest.raises(ValueError, match="unknown compress"):
        resolve_compressor("gzip")


def test_resolve_routers() -> None:
    assert resolve_mode_router("chunks").mode == "chunks"
    assert resolve_mode_router("file").mode == "file"
    assert resolve_mode_router("auto").name == "heuristic"
    assert resolve_mode_router(None).name == "heuristic"
    assert resolve_mode_router("llm", llm=_FakeLLM("file")).name == "llm"
    with pytest.raises(ValueError):
        resolve_mode_router("llm")
    with pytest.raises(ValueError):
        resolve_mode_router("bogus")

    assert resolve_corpus_router(None).name == "all"
    assert resolve_corpus_router("all").name == "all"
    assert resolve_corpus_router("description", llm=_FakeLLM("a")).name == "description"
    with pytest.raises(ValueError):
        resolve_corpus_router("description")


# ---------------------------------------------------------------------------
# Rewriters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identity_and_empty_rewrite() -> None:
    assert await IdentityRewriter().rewrite("  hello ") == ["hello"]
    assert await IdentityRewriter().rewrite("   ") == []
    assert await IdentityRewriter().rewrite("") == []


@pytest.mark.asyncio
async def test_multi_query_strips_numbering_and_caps_n() -> None:
    llm = _FakeLLM("1. alpha\n2. beta\n3. gamma\n4. delta")
    rw = MultiQueryRewriter(llm, n=2, include_original=True)
    qs = await rw.rewrite("root question")
    assert qs[0] == "root question"
    assert qs[1:] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_hyde_includes_hypothetical() -> None:
    llm = _FakeLLM("Hypothetical answer about OAuth2 tokens.")
    qs = await HyDERewriter(llm, include_original=True).rewrite("How is auth done?")
    assert qs[0] == "How is auth done?"
    assert "OAuth2" in qs[1]


@pytest.mark.asyncio
async def test_multi_query_with_provider_complete() -> None:
    class Prov:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="alt one\nalt two")

    qs = await MultiQueryRewriter(Prov(), n=2).rewrite("main")
    assert "main" in qs
    assert "alt one" in qs


# ---------------------------------------------------------------------------
# Rerank / compress
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_and_identity_rerank_truncate() -> None:
    hits = [
        {"id": "a", "content": "a", "score": 0.1},
        {"id": "b", "content": "b", "score": 0.9},
        {"id": "c", "content": "c", "score": 0.5},
    ]
    ranked = await ScoreReranker().rerank("q", hits, top_n=2)
    assert [h["id"] for h in ranked] == ["b", "c"]
    ident = await IdentityReranker().rerank("q", hits, top_n=1)
    assert ident[0]["id"] == "a"


@pytest.mark.asyncio
async def test_llm_reranker_orders_by_ids() -> None:
    hits = [
        {"id": "low", "content": "irrelevant"},
        {"id": "high", "content": "perfect match"},
    ]
    llm = _FakeLLM("high\nlow")
    out = await LLMReranker(llm).rerank("q", hits, top_n=1)
    assert out[0]["id"] == "high"


@pytest.mark.asyncio
async def test_llm_compressor_drops_empty() -> None:
    llm = _FakeLLM(script=["relevant sentence about OAuth", "EMPTY"])
    hits = [
        {"id": "1", "content": "OAuth stuff"},
        {"id": "2", "content": "unrelated"},
    ]
    out = await LLMCompressor(llm).compress("oauth", hits)
    assert len(out) == 1
    assert out[0]["id"] == "1"
    assert out[0].get("compressed") is True


# ---------------------------------------------------------------------------
# Mode / file matching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mode_routers() -> None:
    assert await FixedModeRouter("chunks").choose_mode("auth.md") == "chunks"
    assert await FixedModeRouter("file").choose_mode("anything") == "file"
    h = HeuristicModeRouter()
    assert await h.choose_mode("explain discounts") == "chunks"
    assert await h.choose_mode("see billing.md please") == "file"
    assert await h.choose_mode("open the readme file") == "file"
    assert await LLMModeRouter(_FakeLLM("file")).choose_mode("q") == "file"
    assert await LLMModeRouter(_FakeLLM("chunks")).choose_mode("q") == "chunks"


def test_match_file_sources() -> None:
    sources = ["docs/auth.md", "docs/billing.md", "/abs/path/README.md"]
    assert match_file_sources("auth.md details", sources) == ["docs/auth.md"]
    assert "docs/billing.md" in match_file_sources("billing", sources)
    assert match_file_sources("nothing here", sources) == []


@pytest.mark.asyncio
async def test_forced_file_and_chunks_modes(tmp_path: Path) -> None:
    corpus, _ = await _corpus(tmp_path)
    file_r = AgenticRetriever(corpus, mode="file", rewrite="off", rerank=False)
    file_hits = await file_r.retrieve("What does auth.md cover?", k=5)
    assert file_hits
    assert all(h.get("retrieval_mode") == "file" for h in file_hits)

    chunk_r = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank="score")
    chunk_hits = await chunk_r.retrieve("OAuth2 bearer", k=3)
    assert chunk_hits
    assert all(h.get("retrieval_mode") == "chunks" for h in chunk_hits)


@pytest.mark.asyncio
async def test_file_mode_falls_back_to_chunks_without_match(tmp_path: Path) -> None:
    corpus, _ = await _corpus(tmp_path)
    r = AgenticRetriever(corpus, mode="file", rewrite="off", rerank=False)
    hits = await r.retrieve("totally unknown filename xyzzy.qqq", k=2)
    # falls back to chunk retrieve; still tagged file in _retrieve_file only when matched —
    # without match, _retrieve_chunks is used and retrieval_mode set to mode ("file") in retrieve()
    assert hits
    assert all(h.get("retrieval_mode") == "file" for h in hits)


# ---------------------------------------------------------------------------
# Base modes + k limits + metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("base_mode", ["hybrid", "vector", "lexical"])
async def test_base_modes_return_hits(tmp_path: Path, base_mode: str) -> None:
    corpus, _ = await _corpus(tmp_path, base_mode=base_mode, strategy="markdown")
    r = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank="score")
    hits = await r.retrieve("OAuth2", k=2)
    assert hits
    assert len(hits) <= 2
    for h in hits:
        assert "content" in h
        assert h.get("corpus") == "docs"


@pytest.mark.asyncio
async def test_k_zero_and_empty_query(tmp_path: Path) -> None:
    corpus, _ = await _corpus(tmp_path)
    r = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank=False)
    # k<=0 coerced to 1 in AgenticRetriever
    hits = await r.retrieve("OAuth2", k=0)
    assert len(hits) == 1
    empty = await r.retrieve("   ", k=3)
    # may be empty or weak lexical noise — must not crash
    assert isinstance(empty, list)


# ---------------------------------------------------------------------------
# Rewrite merge path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_query_merge_invokes_base_per_query(tmp_path: Path) -> None:
    corpus, _ = await _corpus(tmp_path, base_mode="lexical")
    seen: list[str] = []

    class SpyRewriter:
        name = "spy"

        async def rewrite(self, query: str) -> list[str]:
            return ["OAuth2", "bearer tokens"]

    class SpyCorpus:
        """Wrap corpus retriever to count calls."""

    original = corpus.retriever.retrieve

    async def wrapped(query: str, k: int):
        seen.append(query)
        return await original(query, k)

    corpus.retriever.retrieve = wrapped  # type: ignore[method-assign]
    r = AgenticRetriever(
        corpus, mode="chunks", rewrite=SpyRewriter(), rerank="score", fetch_k=5
    )
    hits = await r.retrieve("ignored", k=3)
    assert seen == ["OAuth2", "bearer tokens"]
    assert hits


# ---------------------------------------------------------------------------
# Upsert correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_replaces_content_for_recall(tmp_path: Path) -> None:
    corpus, docs = await _corpus(tmp_path, base_mode="hybrid", strategy="markdown")
    r = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank="score")
    before = await r.retrieve("bearer tokens", k=3)
    assert any("bearer" in (h.get("content") or "").lower() for h in before)

    (docs / "auth.md").write_text(
        "# Auth\n\nNow uses passkeys only. No OAuth2.\n", encoding="utf-8"
    )
    n = await corpus.upsert([docs / "auth.md"], rebuild=False)
    assert n >= 1
    after = await r.retrieve("passkeys", k=5)
    assert any("passkey" in (h.get("content") or "").lower() for h in after)


@pytest.mark.asyncio
async def test_upsert_noop_when_unchanged(tmp_path: Path) -> None:
    corpus, docs = await _corpus(tmp_path)
    n0 = len(corpus.chunks)
    assert await corpus.upsert([docs], rebuild=False) == 0
    assert len(corpus.chunks) == n0


# ---------------------------------------------------------------------------
# Multi-corpus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_composite_all_and_description_router(tmp_path: Path) -> None:
    docs = _seed(tmp_path)
    auth = await ingest(
        [docs / "auth.md"],
        name="auth",
        description="Authentication OAuth MFA",
        store=open_vector_store(engine="memory"),
        base_mode="lexical",
    )
    billing = await ingest(
        [docs / "billing.md"],
        name="billing",
        description="Invoices discounts pricing",
        store=open_vector_store(engine="memory"),
        base_mode="lexical",
    )

    all_r = CompositeRetriever([auth, billing], name="knowledge", corpus_router="all")
    hits = await all_r.retrieve("OAuth2", k=4)
    assert hits
    corpora = {h.get("corpus") for h in hits}
    # fan-out may still rank one corpus higher; at least one hit
    assert corpora & {"auth", "billing"}

    llm = _FakeLLM("billing")
    routed = CompositeRetriever(
        [auth, billing],
        name="knowledge",
        corpus_router=DescriptionCorpusRouter(llm),
        mode="chunks",
        rewrite="off",
        rerank=False,
    )
    billed = await routed.retrieve("enterprise discounts", k=3)
    assert billed
    assert all(h.get("corpus") == "billing" for h in billed)


@pytest.mark.asyncio
async def test_build_agentic_multi_corpus_list(tmp_path: Path) -> None:
    docs = _seed(tmp_path)
    a = await ingest(
        [docs / "auth.md"], name="auth", store=open_vector_store(engine="memory")
    )
    b = await ingest(
        [docs / "billing.md"], name="billing", store=open_vector_store(engine="memory")
    )
    r = await build_agentic_retriever([a, b], name="knowledge", corpus_router="all")
    assert isinstance(r, CompositeRetriever)
    assert await r.retrieve("invoice", k=2)


# ---------------------------------------------------------------------------
# Custom plugins end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_reranker_and_compressor(tmp_path: Path) -> None:
    corpus, _ = await _corpus(tmp_path, base_mode="lexical")

    class ReverseReranker:
        name = "reverse"

        async def rerank(self, query, hits, *, top_n):
            return list(reversed(list(hits)))[:top_n]

    class PrefixCompressor:
        name = "prefix"

        async def compress(self, query, hits):
            out = []
            for h in hits:
                row = dict(h)
                row["content"] = "KEEP:" + str(row.get("content") or "")[:80]
                out.append(row)
            return out

    r = AgenticRetriever(
        corpus,
        mode="chunks",
        rewrite="off",
        rerank=ReverseReranker(),
        compress=PrefixCompressor(),
    )
    hits = await r.retrieve("OAuth2", k=2)
    assert hits
    assert all(str(h["content"]).startswith("KEEP:") for h in hits)


# ---------------------------------------------------------------------------
# RetrieverTool + Agent wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retriever_tool_adapter_and_error(tmp_path: Path) -> None:
    corpus, _ = await _corpus(tmp_path)
    r = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank=False)
    tool = RetrieverTool(r)
    ok = await tool.invoke({"query": "OAuth2", "k": 2})
    assert ok.error is None
    assert ok.content
    assert ok.metadata["retriever_name"] == "search_docs"

    class Boom:
        name = "boom"

        async def retrieve(self, query: str, k: int):
            raise RuntimeError("explode")

    err = await RetrieverTool(Boom()).invoke({"query": "x"})  # type: ignore[arg-type]
    assert err.error and "boom" in err.error


@pytest.mark.asyncio
async def test_agent_tool_roundtrip(tmp_path: Path) -> None:
    docs = _seed(tmp_path)
    retriever = await build_agentic_retriever(
        [docs],
        name="docs",
        store=open_vector_store(engine="memory"),
        mode="chunks",
        rewrite="off",
        rerank="score",
        base_mode="hybrid",
    )
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_ScriptedAgent("search_docs")),
        retrievers=[retriever],
        use_llm_summarizer=False,
    )
    result = await agent.arun("How do we authenticate?")
    assert result.output.text() == "done"


# ---------------------------------------------------------------------------
# Optional backends: zvec + faiss
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agentic_over_alibaba_zvec(tmp_path: Path) -> None:
    pytest.importorskip("zvec")
    docs = _seed(tmp_path)
    path = tmp_path / "docs_zvec"
    corpus = await ingest(
        [docs],
        name="docs",
        store=open_vector_store(path=path, dimensions=256),
        base_mode="vector",
        strategy="markdown",
    )
    r = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank="score")
    hits = await r.retrieve("OAuth2", k=3)
    assert hits
    corpus.store.close()


@pytest.mark.asyncio
async def test_agentic_over_faiss(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    from loomable.codeindex.embedders import HashingEmbedder

    docs = _seed(tmp_path)
    emb = HashingEmbedder()
    # probe dims
    dims = len(await emb.embed("probe"))
    store = open_vector_store(
        engine="faiss", path=tmp_path / "faiss", dimensions=dims, device="cpu"
    )
    corpus = await ingest(
        [docs],
        name="docs",
        store=store,
        embedder=emb,
        base_mode="vector",
        strategy="markdown",
    )
    r = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank="score")
    hits = await r.retrieve("OAuth2", k=3)
    assert hits
    store.close()


# ---------------------------------------------------------------------------
# Inline sources + strategy override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inline_sources_and_strategy(tmp_path: Path) -> None:
    corpus = await ingest(
        [
            {"id": "n1", "text": "Alpha uses OAuth2 for login."},
            "Beta document about invoices and discounts.",
        ],
        name="mixed",
        store=open_vector_store(engine="memory"),
        strategy="text",
        base_mode="lexical",
    )
    assert len(corpus.documents) == 2
    r = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank=False)
    hits = await r.retrieve("OAuth2", k=2)
    assert hits
    assert any("oauth" in (h.get("content") or "").lower() for h in hits)
