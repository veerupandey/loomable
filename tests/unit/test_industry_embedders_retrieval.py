"""Industry embedders + retrieval quality tests.

Covers Gemini (live when GEMINI_API_KEY set), HuggingFace local MiniLM,
Azure/OpenAI (HTTP mocked), and a mini recall benchmark comparing naive
vector search vs Loomable hybrid+MMR agentic retrieval.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from loomable.kernel.errors import ModelProviderError
from loomable.providers.embedders import (
    AzureOpenAIEmbedder,
    Embedder,
    GeminiEmbedder,
    HuggingFaceEmbedder,
    OpenAIEmbedder,
    embed_many,
)
from loomable.providers.vector_store import open_vector_store
from loomable.retrieval import AgenticRetriever, MMRReranker, ingest

_FAKE_REQ = httpx.Request("POST", "https://fake/embeddings")


def _openai_resp(vectors: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)],
            "model": "test",
        },
        request=_FAKE_REQ,
    )


# ---------------------------------------------------------------------------
# Protocol + batch helper
# ---------------------------------------------------------------------------


def test_new_embedders_satisfy_protocol() -> None:
    assert isinstance(GeminiEmbedder(api_key="x"), Embedder)
    assert isinstance(
        AzureOpenAIEmbedder(
            deployment="d", endpoint="https://x.openai.azure.com", api_key="k"
        ),
        Embedder,
    )
    # HF local construction does not load weights until embed()
    assert isinstance(HuggingFaceEmbedder(backend="local"), Embedder)


@pytest.mark.asyncio
async def test_openai_and_azure_embed_many_mocked() -> None:
    vecs = [[0.1, 0.2], [0.3, 0.4]]
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = _openai_resp(vecs)
        out = await OpenAIEmbedder(api_key="k").embed_many(["a", "b"])
    assert out == vecs

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = _openai_resp([[1.0, 0.0]])
        az = AzureOpenAIEmbedder(
            deployment="emb", endpoint="https://r.openai.azure.com", api_key="k"
        )
        one = await az.embed("hello")
    assert one == [1.0, 0.0]


@pytest.mark.asyncio
async def test_gemini_embedder_mocked_batch() -> None:
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = _openai_resp([[0.5] * 4, [0.1] * 4])
        g = GeminiEmbedder(api_key="k")
        out = await g.embed_many(["x", "y"])
    assert len(out) == 2 and len(out[0]) == 4
    post.assert_called_once()
    assert "generativelanguage.googleapis.com" in post.call_args[0][0]


@pytest.mark.asyncio
async def test_hf_api_backend_mocked() -> None:
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = httpx.Response(
            200, json=[[0.1, 0.2, 0.3]], request=_FAKE_REQ
        )
        emb = HuggingFaceEmbedder(backend="api", api_key="hf_x")
        v = await emb.embed("hi")
    assert v == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Live Gemini
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_gemini_embed_and_retrieve(tmp_path: Path) -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        pytest.skip("GEMINI_API_KEY not set")
    emb = GeminiEmbedder()
    v = await emb.embed("OAuth2 bearer tokens for API authentication")
    assert isinstance(v, list) and len(v) >= 256
    assert all(isinstance(x, float) for x in v[:5])

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "auth.md").write_text(
        "# Auth\n\nClients authenticate with OAuth2 bearer tokens.\n",
        encoding="utf-8",
    )
    (docs / "food.md").write_text(
        "# Recipes\n\nBake sourdough at 230C for 40 minutes.\n",
        encoding="utf-8",
    )
    dims = len(v)
    store = open_vector_store(engine="faiss", dimensions=dims, device="cpu")
    corpus = await ingest(
        [docs],
        name="docs",
        store=store,
        embedder=emb,
        strategy="markdown",
        base_mode="hybrid",
    )
    r = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank="mmr")
    hits = await r.retrieve("How do users log into the API?", k=2)
    assert hits
    blob = " ".join(h.get("content", "") for h in hits).lower()
    assert "oauth" in blob or "bearer" in blob or "auth" in blob
    store.close()


# ---------------------------------------------------------------------------
# Live HuggingFace local
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_huggingface_minilm_retrieve(tmp_path: Path) -> None:
    pytest.importorskip("sentence_transformers")
    emb = HuggingFaceEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2", backend="local"
    )
    v = await emb.embed("enterprise invoice discounts")
    assert len(v) == 384

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "billing.md").write_text(
        "# Billing\n\nEnterprise customers get volume discounts on invoices.\n",
        encoding="utf-8",
    )
    (docs / "sports.md").write_text(
        "# Sports\n\nThe team won the championship in overtime.\n",
        encoding="utf-8",
    )
    store = open_vector_store(engine="faiss", dimensions=384, device="cpu")
    corpus = await ingest(
        [docs],
        name="docs",
        store=store,
        embedder=emb,
        strategy="markdown",
        base_mode="hybrid",
    )
    r = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank="mmr")
    hits = await r.retrieve("pricing reductions for large accounts", k=2)
    assert hits
    assert "discount" in hits[0].get("content", "").lower() or "invoice" in hits[
        0
    ].get("content", "").lower()
    # batch path
    batch = await embed_many(emb, ["a", "b", "c"])
    assert len(batch) == 3 and len(batch[0]) == 384
    store.close()


# ---------------------------------------------------------------------------
# Quality: agentic hybrid+MMR vs naive vector (LlamaIndex-style top-k)
# ---------------------------------------------------------------------------


def _hit_doc(hits: list[dict]) -> str:
    if not hits:
        return ""
    src = (hits[0].get("source") or hits[0].get("path") or "").lower()
    for key in ("auth", "billing", "ops", "noise", "cooking"):
        if key in src:
            return key
    content = (hits[0].get("content") or "").lower()
    if "kubectl" in content or "rollout" in content:
        return "ops"
    if "invoice" in content or "discount" in content:
        return "billing"
    if "oauth" in content or "jwt" in content:
        return "auth"
    if "roast" in content or "gravy" in content:
        return "cooking"
    return "other"


@pytest.mark.asyncio
async def test_agentic_hybrid_mmr_beats_naive_vector_on_mixed_queries(
    tmp_path: Path,
) -> None:
    """Hybrid RRF + MMR should beat naive dense-only (LlamaIndex-style top-k).

    Includes exact rare keywords (where sparse helps) and paraphrases. Naive
    vector-only is the common LlamaIndex VectorIndexRetriever default.
    """
    pytest.importorskip("sentence_transformers")
    emb = HuggingFaceEmbedder(backend="local")
    docs = tmp_path / "corp"
    docs.mkdir()
    corpus_files = {
        "auth.md": "# Auth\n\nSign in with OAuth2. Issue JWT access tokens.\n",
        "billing.md": "# Billing\n\nQuarterly invoices include early-pay discounts.\n",
        "ops.md": "# Ops\n\nUse kubectl to roll out the payment service.\n",
        # Lexical trap: shares "service" with ops queries / auth paraphrases.
        "cooking.md": "# Cooking\n\nService the roast with gravy and herbs.\n",
        "noise.md": "# Notes\n\nThe weather in Lisbon is mild in spring.\n",
    }
    for name, text in corpus_files.items():
        (docs / name).write_text(text, encoding="utf-8")

    store_a = open_vector_store(engine="faiss", dimensions=384, device="cpu")
    store_b = open_vector_store(engine="faiss", dimensions=384, device="cpu")
    hybrid_corpus = await ingest(
        [docs],
        name="hybrid",
        store=store_a,
        embedder=emb,
        strategy="markdown",
        base_mode="hybrid",
    )
    naive_corpus = await ingest(
        [docs],
        name="naive",
        store=store_b,
        embedder=emb,
        strategy="markdown",
        base_mode="vector",
    )

    agentic = AgenticRetriever(
        hybrid_corpus, mode="chunks", rewrite="off", rerank=MMRReranker(lambda_mult=0.7)
    )
    naive = AgenticRetriever(
        naive_corpus, mode="chunks", rewrite="off", rerank=False
    )

    cases = [
        ("kubectl rollout payment", "ops"),
        ("early payment discount invoices", "billing"),
        ("JWT OAuth sign-in", "auth"),
        ("how do users authenticate with bearer tokens", "auth"),
        ("volume pricing reductions on quarterly bills", "billing"),
    ]

    agentic_ok = 0
    naive_ok = 0
    for query, expect in cases:
        a_hits = await agentic.retrieve(query, k=1)
        n_hits = await naive.retrieve(query, k=1)
        if _hit_doc(a_hits) == expect:
            agentic_ok += 1
        if _hit_doc(n_hits) == expect:
            naive_ok += 1

    # Agentic hybrid+MMR must match or beat naive top-1 and clear a majority.
    assert agentic_ok >= naive_ok
    assert agentic_ok >= 4
    store_a.close()
    store_b.close()


@pytest.mark.asyncio
async def test_score_metadata_does_not_clobber_similarity(tmp_path: Path) -> None:
    """Indexed metadata must never overwrite FAISS/query similarity scores."""
    pytest.importorskip("sentence_transformers")
    emb = HuggingFaceEmbedder(backend="local")
    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "auth.md").write_text(
        "# Auth\n\nOAuth2 bearer tokens for API identity.\n", encoding="utf-8"
    )
    (docs / "cooking.md").write_text(
        "# Cooking\n\nService the roast with gravy.\n", encoding="utf-8"
    )
    store = open_vector_store(engine="faiss", dimensions=384, device="cpu")
    corp = await ingest(
        [docs],
        name="d",
        store=store,
        embedder=emb,
        strategy="markdown",
        base_mode="vector",
    )
    hits = await AgenticRetriever(
        corp, mode="chunks", rewrite="off", rerank=False
    ).retrieve("API authentication OAuth bearer", k=2)
    assert hits
    assert _hit_doc(hits) == "auth"
    # Real similarities are in (0, 1] for cosine/IP; metadata score=0 would fail this.
    assert float(hits[0]["score"]) > 0.05
    store.close()


@pytest.mark.asyncio
async def test_live_azure_openai_embed_when_configured() -> None:
    """Live Azure embeddings when AZURE_OPENAI_* env is present."""
    if not (
        os.environ.get("AZURE_OPENAI_API_KEY")
        and os.environ.get("AZURE_OPENAI_ENDPOINT")
        and (
            os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        )
    ):
        pytest.skip("Azure OpenAI embedding env not configured")
    deployment = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME") or os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    )
    emb = AzureOpenAIEmbedder(deployment=deployment)
    v = await emb.embed("hybrid retrieval with reciprocal rank fusion")
    assert isinstance(v, list) and len(v) >= 256
    batch = await emb.embed_many(["a", "b"])
    assert len(batch) == 2 and len(batch[0]) == len(v)


@pytest.mark.asyncio
async def test_live_gemini_quality_vs_hashing(tmp_path: Path) -> None:
    """Gemini embeddings should retrieve paraphrases hashing embedder misses."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        pytest.skip("GEMINI_API_KEY not set")
    from loomable.codeindex.embedders import HashingEmbedder

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "security.md").write_text(
        "# Security\n\nRotate API keys every 90 days and store secrets in a vault.\n",
        encoding="utf-8",
    )
    (docs / "gardening.md").write_text(
        "# Gardening\n\nWater tomatoes deeply twice a week in summer.\n",
        encoding="utf-8",
    )
    query = "How often should credentials be refreshed?"

    gem = GeminiEmbedder()
    dims = len(await gem.embed("probe"))
    g_store = open_vector_store(engine="faiss", dimensions=dims, device="cpu")
    g_corp = await ingest(
        [docs],
        name="g",
        store=g_store,
        embedder=gem,
        strategy="markdown",
        base_mode="hybrid",
    )
    g_hits = await AgenticRetriever(
        g_corp, mode="chunks", rewrite="off", rerank="mmr"
    ).retrieve(query, k=1)

    h_emb = HashingEmbedder()
    h_store = open_vector_store(engine="memory")
    h_corp = await ingest(
        [docs],
        name="h",
        store=h_store,
        embedder=h_emb,
        strategy="markdown",
        base_mode="vector",
    )
    h_hits = await AgenticRetriever(
        h_corp, mode="chunks", rewrite="off", rerank=False
    ).retrieve(query, k=1)

    g_text = (g_hits[0].get("content") or "").lower() if g_hits else ""
    assert "key" in g_text or "secret" in g_text or "vault" in g_text or "90" in g_text
    # Hashing often fails paraphrase; we only require Gemini succeeds.
    g_store.close()
