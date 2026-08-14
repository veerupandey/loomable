"""Citation metadata round-trip + retrieve filters."""

from __future__ import annotations

from pathlib import Path

import pytest

from loomable.providers.vector_store import open_vector_store
from loomable.retrieval import AgenticRetriever, ingest, matches_filters
from loomable.retrieval.metadata import shape_hit


def test_matches_filters_equality_and_tags() -> None:
    hit = {
        "page": 3,
        "media_type": "application/pdf",
        "tags": ["auth", "security"],
        "author": "alice",
    }
    assert matches_filters(hit, {"page": 3, "author": "alice"})
    assert matches_filters(hit, {"tags": ["auth"]})
    assert not matches_filters(hit, {"page": 12})
    # chroma-style JSON list
    assert matches_filters({"tags": '["auth"]'}, {"tags": "auth"})


@pytest.mark.asyncio
async def test_ingest_metadata_on_hits_and_filters(tmp_path: Path) -> None:
    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "auth.md").write_text("# Auth\n\nOAuth2 bearer tokens.\n", encoding="utf-8")
    (docs / "food.md").write_text("# Food\n\nRoast with gravy.\n", encoding="utf-8")

    corpus = await ingest(
        [docs],
        name="kb",
        store=open_vector_store(engine="memory"),
        strategy="markdown",
        base_mode="hybrid",
        metadata={"author": "alice", "department": "security", "tags": ["internal"]},
    )
    assert all(c.metadata.get("author") == "alice" for c in corpus.chunks)
    assert all(c.metadata.get("filename") for c in corpus.chunks)
    assert all(c.metadata.get("media_type") for c in corpus.chunks)

    rag = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank=False)
    hits = await rag.retrieve("OAuth2 bearer", k=3)
    assert hits
    top = hits[0]
    assert top.get("author") == "alice"
    assert top.get("department") == "security"
    assert top.get("filename")
    assert top.get("corpus") == "kb"
    assert top.get("source") or top.get("path")

    filtered = await rag.retrieve(
        "OAuth2 bearer", k=5, filters={"filename": "food.md"}
    )
    assert filtered
    assert all("food.md" in str(h.get("filename") or h.get("path") or "") for h in filtered)

    # per-file metadata via dict source
    extra = await ingest(
        [{"path": docs / "auth.md", "metadata": {"title": "Login Guide", "author": "bob"}}],
        name="one",
        store=open_vector_store(engine="memory"),
        strategy="markdown",
        base_mode="lexical",
    )
    assert extra.chunks[0].metadata.get("title") == "Login Guide"
    assert extra.chunks[0].metadata.get("author") == "bob"
    corpus.store.close()
    extra.store.close()


@pytest.mark.asyncio
async def test_metadata_roundtrip_across_file_engines(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# Note\n\nSKU-META-1 confidential.\n", encoding="utf-8")
    engines = []
    try:
        import faiss  # noqa: F401

        engines.append(
            (
                "faiss",
                lambda: open_vector_store(
                    engine="faiss", path=tmp_path / "f", dimensions=256
                ),
            )
        )
    except ImportError:
        pass
    try:
        import zvec  # noqa: F401

        engines.append(
            (
                "zvec",
                lambda: open_vector_store(
                    engine="zvec", path=tmp_path / "z", dimensions=256
                ),
            )
        )
    except ImportError:
        pass
    try:
        import chromadb  # noqa: F401

        engines.append(
            (
                "chroma",
                lambda: open_vector_store(
                    engine="chroma",
                    path=tmp_path / "c",
                    dimensions=256,
                    collection="meta",
                ),
            )
        )
    except ImportError:
        pass
    engines.append(("memory", lambda: open_vector_store(engine="memory")))
    assert engines
    from loomable.codeindex.embedders import HashingEmbedder

    failures: list[str] = []
    for name, factory in engines:
        store = factory()
        try:
            corpus = await ingest(
                [tmp_path / "note.md"],
                name="m",
                store=store,
                embedder=HashingEmbedder(),
                strategy="markdown",
                base_mode="vector",
                metadata={"classification": "confidential", "owner": "secops"},
            )
            rag = AgenticRetriever(corpus, mode="chunks", rewrite="off", rerank=False)
            hits = await rag.retrieve("SKU-META-1", k=3)
            if not hits:
                failures.append(f"{name}: no hits")
                continue
            hit = shape_hit(hits[0])
            if hit.get("classification") != "confidential":
                failures.append(f"{name}: missing classification {hit.keys()}")
            if hit.get("owner") != "secops":
                failures.append(f"{name}: missing owner")
            scoped = await rag.retrieve(
                "SKU-META-1", k=3, filters={"classification": "confidential"}
            )
            if not scoped:
                failures.append(f"{name}: filter returned empty")
            empty = await rag.retrieve(
                "SKU-META-1", k=3, filters={"classification": "public"}
            )
            if empty:
                failures.append(f"{name}: filter leaked public")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    assert not failures, "\n".join(failures)
