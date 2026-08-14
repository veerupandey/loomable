"""JSON / CSV chunk strategies for structured sources."""

from __future__ import annotations

import csv
import io
import json

from loomable.retrieval.chunking.base import register_strategy
from loomable.retrieval.chunking.text import TextChunker
from loomable.retrieval.types import Chunk, Document, merge_metadata


class JsonChunker:
    """Chunk JSON objects/arrays into retrieval units (falls back to text)."""

    name = "json"

    def __init__(self, *, max_chars: int = 2_000) -> None:
        self.max_chars = max_chars
        self._text = TextChunker(max_chars=max_chars)

    def chunk(self, document: Document) -> list[Chunk]:
        raw = (document.text or "").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._text.chunk(document)

        units: list[tuple[str, str]] = []
        if isinstance(data, dict):
            for key, value in data.items():
                units.append((str(key), json.dumps({key: value}, ensure_ascii=False, indent=2)))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                units.append((f"item-{i}", json.dumps(item, ensure_ascii=False, indent=2)))
        else:
            return self._text.chunk(document)

        out: list[Chunk] = []
        for name, text in units:
            text = text[:8_000]
            out.append(
                Chunk(
                    id=f"{document.id}:json:{name}",
                    text=text,
                    document_id=document.id,
                    start_line=1,
                    end_line=text.count("\n") + 1,
                    kind="json",
                    name=name,
                    metadata=merge_metadata(
                        document.metadata,
                        {"source": document.source, "path": document.source},
                    ),
                )
            )
        return out or self._text.chunk(document)


class CsvChunker:
    """Chunk CSV/TSV into row batches."""

    name = "csv"

    def __init__(self, *, rows_per_chunk: int = 25) -> None:
        self.rows_per_chunk = max(1, int(rows_per_chunk))
        self._text = TextChunker()

    def chunk(self, document: Document) -> list[Chunk]:
        raw = document.text or ""
        if not raw.strip():
            return []
        dialect = csv.excel_tab if (document.source or "").lower().endswith(".tsv") else csv.excel
        try:
            reader = csv.reader(io.StringIO(raw), dialect=dialect)
            rows = list(reader)
        except csv.Error:
            return self._text.chunk(document)
        if not rows:
            return []
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        if not body:
            return self._text.chunk(document)

        out: list[Chunk] = []
        for start in range(0, len(body), self.rows_per_chunk):
            batch = body[start : start + self.rows_per_chunk]
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(header)
            writer.writerows(batch)
            text = buf.getvalue().strip()
            name = f"rows-{start + 1}-{start + len(batch)}"
            out.append(
                Chunk(
                    id=f"{document.id}:csv:{name}",
                    text=text[:8_000],
                    document_id=document.id,
                    start_line=start + 2,
                    end_line=start + 1 + len(batch),
                    kind="csv",
                    name=name,
                    metadata=merge_metadata(
                        document.metadata,
                        {"source": document.source, "path": document.source},
                    ),
                )
            )
        return out


register_strategy(JsonChunker())
register_strategy(CsvChunker())
