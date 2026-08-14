"""Fixed / recursive text chunking."""

from __future__ import annotations

from loomable.retrieval.chunking.base import register_strategy
from loomable.retrieval.types import Chunk, Document, merge_metadata


class TextChunker:
    """Split plain text on paragraph boundaries with size limits."""

    name = "text"

    def __init__(
        self,
        *,
        max_chars: int = 2_000,
        overlap: int = 200,
    ) -> None:
        self.max_chars = max(200, int(max_chars))
        self.overlap = max(0, min(int(overlap), self.max_chars // 2))

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text or ""
        if not text.strip():
            return []
        paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
        paragraphs = [p for p in paragraphs if p]
        if not paragraphs:
            paragraphs = [text]

        blocks: list[str] = []
        buf = ""
        for para in paragraphs:
            candidate = f"{buf}\n\n{para}".strip() if buf else para
            if len(candidate) <= self.max_chars:
                buf = candidate
                continue
            if buf:
                blocks.append(buf)
            if len(para) <= self.max_chars:
                buf = para
            else:
                # hard-split long paragraph
                start = 0
                while start < len(para):
                    end = min(len(para), start + self.max_chars)
                    blocks.append(para[start:end])
                    start = max(end - self.overlap, start + 1)
                buf = ""
        if buf:
            blocks.append(buf)

        # apply overlap between blocks
        if self.overlap and len(blocks) > 1:
            overlapped: list[str] = [blocks[0]]
            for prev, cur in zip(blocks, blocks[1:]):
                tail = prev[-self.overlap :] if len(prev) > self.overlap else prev
                if not cur.startswith(tail):
                    overlapped.append(f"{tail}\n{cur}"[-self.max_chars :])
                else:
                    overlapped.append(cur)
            blocks = overlapped

        chunks: list[Chunk] = []
        cursor = 1
        for i, block in enumerate(blocks):
            n_lines = block.count("\n") + 1
            end = cursor + n_lines - 1
            chunks.append(
                Chunk(
                    id=f"{document.id}:text:{i}:{cursor}-{end}",
                    text=block,
                    document_id=document.id,
                    start_line=cursor,
                    end_line=end,
                    kind="text",
                    name=f"chunk-{i}",
                    metadata=merge_metadata(
                        document.metadata,
                        {"source": document.source, "path": document.source},
                    ),
                )
            )
            cursor = end + 1
        return chunks


register_strategy(TextChunker())
