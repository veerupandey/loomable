"""Markdown heading-aware chunking."""

from __future__ import annotations

import re

from loomable.retrieval.chunking.base import register_strategy
from loomable.retrieval.chunking.text import TextChunker
from loomable.retrieval.types import Chunk, Document, merge_metadata

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class MarkdownChunker:
    """Split Markdown on AT1–ATX headings; fall back to text chunker."""

    name = "markdown"

    def __init__(self, *, max_chars: int = 3_000) -> None:
        self.max_chars = max_chars
        self._text = TextChunker(max_chars=max_chars)

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text or ""
        if not text.strip():
            return []
        matches = list(_HEADING.finditer(text))
        if not matches:
            return self._text.chunk(document)

        sections: list[tuple[str, str, int, int]] = []  # title, body, start, end
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            title = match.group(2).strip()
            body = text[start:end].strip()
            start_line = text[:start].count("\n") + 1
            end_line = start_line + body.count("\n")
            sections.append((title, body, start_line, end_line))

        # preface before first heading
        if matches[0].start() > 0:
            preface = text[: matches[0].start()].strip()
            if preface:
                sections.insert(
                    0,
                    (
                        "preface",
                        preface,
                        1,
                        preface.count("\n") + 1,
                    ),
                )

        chunks: list[Chunk] = []
        for i, (title, body, start_line, end_line) in enumerate(sections):
            if len(body) > self.max_chars:
                # sub-chunk oversized sections
                sub_doc = Document(
                    id=f"{document.id}#{title}",
                    text=body,
                    source=document.source,
                    media_type=document.media_type,
                    metadata=document.metadata,
                )
                for j, sub in enumerate(self._text.chunk(sub_doc)):
                    chunks.append(
                        Chunk(
                            id=f"{document.id}:md:{i}.{j}",
                            text=sub.text,
                            document_id=document.id,
                            start_line=start_line + sub.start_line - 1,
                            end_line=start_line + sub.end_line - 1,
                            kind="section",
                            name=title,
                            metadata=merge_metadata(
                                document.metadata,
                                {
                                    "source": document.source,
                                    "path": document.source,
                                    "heading": title,
                                },
                            ),
                        )
                    )
            else:
                chunks.append(
                    Chunk(
                        id=f"{document.id}:md:{i}:{start_line}-{end_line}",
                        text=body,
                        document_id=document.id,
                        start_line=start_line,
                        end_line=end_line,
                        kind="section",
                        name=title,
                        metadata=merge_metadata(
                            document.metadata,
                            {
                                "source": document.source,
                                "path": document.source,
                                "heading": title,
                            },
                        ),
                    )
                )
        return chunks


register_strategy(MarkdownChunker())
