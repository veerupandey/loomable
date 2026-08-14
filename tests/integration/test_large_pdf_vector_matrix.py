"""Large PDF ingest + retrieval quality across vector engines.

The framework owns PDF handling: ``ingest([handbook.pdf])`` extracts pages and
chunks them. This file only asserts quality per store.

Engines: zvec, faiss, chroma, milvus-lite, postgres (Docker when POSTGRES_URL set).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import pytest

from loomable.providers.vector_store import open_vector_store
from loomable.retrieval import AgenticRetriever, ingest
from tests.helpers.pdf_fixture import large_handbook_pages, write_pdf

POSTGRES_URL = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")

# Unique plants the ingest pipeline must keep and retrieval must surface.
CASES = [
    ("AUTH-PLANT-A1 OAuth2 bearer", ["AUTH-PLANT-A1", "oauth"]),
    ("SKU-PDF-8821 enterprise audit add-on", ["SKU-PDF-8821"]),
    ("rotate KEY-WHSEC-9901 after webhook leak", ["KEY-WHSEC-9901"]),
    ("ap-south p99 latency gold tier", ["ap-south", "118"]),
]


def _embedder():
    try:
        from loomable.providers.embedders import HuggingFaceEmbedder

        import sentence_transformers  # noqa: F401

        return HuggingFaceEmbedder(backend="local"), 384
    except Exception:
        from loomable.codeindex.embedders import HashingEmbedder

        return HashingEmbedder(dim=256), 256


@pytest.fixture(scope="module")
def handbook_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("pypdf")
    root = tmp_path_factory.mktemp("large_pdf")
    return write_pdf(root / "handbook.pdf", large_handbook_pages(n_pages=80))


def _hit_blob(hits: list[dict[str, Any]]) -> str:
    return " ".join(str(h.get("content") or "") for h in hits).lower()


async def _eval_engine(
    *,
    name: str,
    store_factory: Callable[[], Any],
    pdf: Path,
    embedder: Any,
) -> list[str]:
    store = store_factory()
    failures: list[str] = []
    try:
        corpus = await ingest(
            [pdf],
            name="handbook",
            store=store,
            embedder=embedder,
            strategy="auto",  # framework chooses PDF page chunking
            base_mode="hybrid",
        )
        # Ingest must have kept buried SKU (would be lost if truncated at 8k).
        if not any("SKU-PDF-8821" in (c.text or "") for c in corpus.chunks):
            failures.append(f"{name}: ingest dropped SKU-PDF-8821")
        rag = AgenticRetriever(
            corpus, name="search_docs", mode="chunks", rewrite="off", rerank="mmr"
        )
        for query, needles in CASES:
            hits = await rag.retrieve(query, k=5)
            blob = _hit_blob(hits)
            missing = [n for n in needles if n.lower() not in blob]
            if missing:
                top = (hits[0].get("content") if hits else "")[:160]
                failures.append(f"{name}: {query!r} missing {missing} top={top!r}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
    finally:
        aclose = getattr(store, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass
        else:
            close = getattr(store, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    return failures


@pytest.mark.asyncio
async def test_large_pdf_ingest_retrieval_all_engines(
    handbook_pdf: Path, tmp_path: Path
) -> None:
    embedder, dims = _embedder()
    engines: list[tuple[str, Callable[[], Any]]] = []

    def _add(name: str, factory: Callable[[], Any], dep: str) -> None:
        try:
            __import__(dep)
        except ImportError:
            return
        engines.append((name, factory))

    _add(
        "zvec",
        lambda: open_vector_store(engine="zvec", path=tmp_path / "zvec", dimensions=dims),
        "zvec",
    )
    _add(
        "faiss",
        lambda: open_vector_store(
            engine="faiss", path=tmp_path / "faiss", dimensions=dims, device="cpu"
        ),
        "faiss",
    )
    _add(
        "chroma",
        lambda: open_vector_store(
            engine="chroma",
            path=tmp_path / "chroma",
            dimensions=dims,
            collection="pdf_chroma",
        ),
        "chromadb",
    )
    try:
        import milvus_lite  # noqa: F401
        from pymilvus import MilvusClient  # noqa: F401

        engines.append(
            (
                "milvus",
                lambda: open_vector_store(
                    engine="milvus",
                    path=tmp_path / "milvus.db",
                    dimensions=dims,
                    collection="pdf_milvus",
                ),
            )
        )
    except ImportError:
        pass
    if POSTGRES_URL:
        try:
            import asyncpg  # noqa: F401

            engines.append(
                (
                    "postgres",
                    lambda: open_vector_store(
                        postgres_url=POSTGRES_URL,
                        dimensions=dims,
                        user_id=f"pdf_{tmp_path.name}",
                    ),
                )
            )
        except ImportError:
            pass

    assert engines, "no vector backends installed"
    report: list[str] = []
    for name, factory in engines:
        failures = await _eval_engine(
            name=name, store_factory=factory, pdf=handbook_pdf, embedder=embedder
        )
        if failures:
            report.extend(failures)
        else:
            report.append(f"{name}: PASS")
    bad = [r for r in report if not r.endswith("PASS")]
    assert not bad, "large PDF quality failures:\n" + "\n".join(report)
