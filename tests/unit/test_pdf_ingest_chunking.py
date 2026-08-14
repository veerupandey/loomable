"""PDF ingest owns extraction + page chunking (no caller-side splitting)."""

from __future__ import annotations

from pathlib import Path

import pytest

from loomable.retrieval.chunking.html_pdf import PdfChunker
from loomable.retrieval.ingest import load_file
from loomable.retrieval.types import Document
from tests.helpers.pdf_fixture import large_handbook_pages, write_pdf


def test_pdf_chunker_does_not_truncate_oversized_page() -> None:
    plant = "SKU-PDF-8821-MUST-SURVIVE"
    filler = ("dense pdf paragraph. " * 80) * 8  # well over 8k
    doc = Document(
        id="handbook",
        text=f"--- Page 1 ---\nshort\n--- Page 2 ---\n{filler}\n{plant}\n",
        source="handbook.pdf",
        media_type="application/pdf",
    )
    chunks = PdfChunker(max_chars=2_500).chunk(doc)
    blob = " ".join(c.text for c in chunks)
    assert plant in blob
    assert all(c.metadata.get("page") in {1, 2} for c in chunks)
    page2 = [c for c in chunks if c.metadata.get("page") == 2]
    assert 2 <= len(page2) < 20
    assert any(plant in c.text for c in page2)


@pytest.mark.asyncio
async def test_ingest_auto_chunks_large_pdf(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    from loomable.retrieval import ingest
    from loomable.providers.vector_store import open_vector_store

    pdf = write_pdf(tmp_path / "handbook.pdf", large_handbook_pages(n_pages=80))
    # Caller only passes the file — ingest extracts + chunks.
    corpus = await ingest(
        [pdf],
        name="docs",
        store=open_vector_store(engine="memory"),
        strategy="auto",
        base_mode="lexical",
    )
    assert corpus.documents
    assert corpus.documents[0].kind_hint == "pdf"
    raw = corpus.documents[0].text
    assert "--- Page 12 ---" in raw
    assert "SKU-PDF-8821" in raw  # extract kept the buried token

    kinds = {c.kind for c in corpus.chunks}
    assert "page" in kinds
    pages = {c.metadata.get("page") for c in corpus.chunks}
    assert 3 in pages and 12 in pages and 41 in pages and 70 in pages
    blob = " ".join(c.text for c in corpus.chunks)
    for token in ("AUTH-PLANT-A1", "SKU-PDF-8821", "KEY-WHSEC-9901", "118"):
        assert token in blob, f"ingest dropped {token}"
    # Page 12 is oversized → more than one chunk tagged page=12
    assert sum(1 for c in corpus.chunks if c.metadata.get("page") == 12) >= 2
    # Long-page split must stay bounded (old overlap bug emitted 100+ shards).
    assert sum(1 for c in corpus.chunks if c.metadata.get("page") == 12) < 20
    corpus.store.close()
