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
    """Page-aware PDF chunking.

    Expects ``--- Page N ---`` markers from ingest. Each page becomes one
    chunk when it fits; oversized pages are split with overlap (no truncation).
    """

    name = "pdf"

    def __init__(self, *, max_chars: int = 2_500, overlap: int = 200) -> None:
        self.max_chars = max(200, int(max_chars))
        self._text = TextChunker(max_chars=self.max_chars, overlap=overlap)

    def chunk(self, document: Document) -> list:
        text = document.text or ""
        if "--- Page " not in text:
            return self._text.chunk(document)

        from loomable.retrieval.types import Chunk, merge_metadata

        parts = re.split(r"\n?--- Page (\d+) ---\n?", text)
        # re.split with a capture group: [preamble, num, body, num, body, ...]
        chunks: list = []
        # parts[0] is text before first marker (usually empty)
        i = 1
        while i + 1 < len(parts):
            try:
                page_no = int(parts[i])
            except ValueError:
                page_no = (i // 2) + 1
            body = (parts[i + 1] or "").strip()
            i += 2
            if not body:
                continue
            page_meta = merge_metadata(
                document.metadata,
                {
                    "source": document.source,
                    "path": document.source,
                    "page": page_no,
                },
            )
            if len(body) <= self.max_chars:
                chunks.append(
                    Chunk(
                        id=f"{document.id}:page:{page_no}",
                        text=body,
                        document_id=document.id,
                        start_line=1,
                        end_line=body.count("\n") + 1,
                        kind="page",
                        name=f"page-{page_no}",
                        metadata=page_meta,
                    )
                )
                continue
            subdoc = Document(
                id=f"{document.id}:page:{page_no}",
                text=body,
                source=document.source,
                media_type="text/plain",
                metadata=page_meta,
            )
            for j, sub in enumerate(self._text.chunk(subdoc)):
                sub.kind = "page"
                sub.name = f"page-{page_no}.{j}"
                sub.metadata = merge_metadata(page_meta, sub.metadata)
                chunks.append(sub)
        return chunks or self._text.chunk(document)


class AutoChunker:
    """Dispatch to markdown / code / html / pdf / json / csv / text from hints."""

    name = "auto"

    def chunk(self, document: Document) -> list:
        from loomable.retrieval.chunking.base import get_strategy

        hint = document.kind_hint
        if hint in {"markdown", "code", "html", "pdf", "json", "csv"}:
            return get_strategy(hint).chunk(document)
        return get_strategy("text").chunk(document)


register_strategy(HtmlChunker())
register_strategy(PdfChunker())
register_strategy(AutoChunker())
