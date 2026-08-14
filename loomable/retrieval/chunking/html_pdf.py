"""HTML and PDF chunk strategies (best-effort; optional deps)."""

from __future__ import annotations

import re

from loomable.retrieval.chunking.base import register_strategy
from loomable.retrieval.chunking.text import TextChunker
from loomable.retrieval.types import Document


class HtmlChunker:
    """Strip tags then text-chunk. Uses bs4 when installed."""

    name = "html"

    def __init__(self) -> None:
        self._text = TextChunker()

    def chunk(self, document: Document) -> list:
        raw = document.text or ""
        try:
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text("\n")
        except Exception:  # noqa: BLE001
            text = re.sub(r"<[^>]+>", " ", raw)
        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
        return self._text.chunk(
            Document(
                id=document.id,
                text=cleaned,
                source=document.source,
                media_type="text/plain",
                metadata=document.metadata,
            )
        )


class PdfChunker:
    """PDF text is expected already extracted on Document.text; page-split if markers."""

    name = "pdf"

    def __init__(self) -> None:
        self._text = TextChunker(max_chars=2_500)

    def chunk(self, document: Document) -> list:
        text = document.text or ""
        if "--- Page " in text:
            parts = re.split(r"\n?--- Page \d+ ---\n?", text)
            chunks = []
            from loomable.retrieval.types import Chunk, merge_metadata

            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                chunks.append(
                    Chunk(
                        id=f"{document.id}:page:{i}",
                        text=part[:8_000],
                        document_id=document.id,
                        start_line=1,
                        end_line=part.count("\n") + 1,
                        kind="page",
                        name=f"page-{i}",
                        metadata=merge_metadata(
                            document.metadata,
                            {"source": document.source, "path": document.source, "page": i},
                        ),
                    )
                )
            if chunks:
                return chunks
        return self._text.chunk(document)


class AutoChunker:
    """Dispatch to markdown / code / html / pdf / text from document hints."""

    name = "auto"

    def chunk(self, document: Document) -> list:
        from loomable.retrieval.chunking.base import get_strategy

        hint = document.kind_hint
        if hint in {"markdown", "code", "html", "pdf"}:
            return get_strategy(hint).chunk(document)
        return get_strategy("text").chunk(document)


register_strategy(HtmlChunker())
register_strategy(PdfChunker())
register_strategy(AutoChunker())
