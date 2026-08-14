"""Walk a repo and emit code chunks — delegates to ``loomable.retrieval``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from loomable.retrieval.chunking import get_strategy
from loomable.retrieval.ingest import load_directory
from loomable.retrieval.types import Chunk

_CODE_EXTS = (
    ".py",
    ".pyi",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".cs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sql",
    ".sh",
)


@dataclass(frozen=True)
class CodeChunk:
    """One indexable unit of a codebase (compat wrapper over retrieval Chunk)."""

    chunk_id: str
    path: str
    text: str
    start_line: int
    end_line: int
    kind: str
    name: str
    language: str

    @classmethod
    def from_retrieval(cls, chunk: Chunk, *, root: Path) -> "CodeChunk":
        path = str(chunk.metadata.get("path") or chunk.metadata.get("source") or "")
        try:
            rel = Path(path).resolve().relative_to(root.resolve()).as_posix()
        except Exception:  # noqa: BLE001
            rel = path or chunk.document_id
        lang = str(chunk.metadata.get("language") or Path(rel).suffix.lstrip(".") or "txt")
        return cls(
            chunk_id=chunk.id,
            path=rel,
            text=chunk.text,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            kind=chunk.kind,
            name=chunk.name,
            language=lang,
        )


def iter_code_chunks(
    root: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    max_files: int = 5_000,
) -> Iterator[CodeChunk]:
    """Yield :class:`CodeChunk` values for ``root`` via the shared code strategy."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return
    docs = load_directory(base, extensions=extensions or _CODE_EXTS, max_files=max_files)
    strategy = get_strategy("code")
    for doc in docs:
        # Prefer relative path as document source for nicer maps
        try:
            rel = Path(doc.source).resolve().relative_to(base).as_posix()
            doc.metadata.setdefault("language", Path(rel).suffix.lstrip("."))
            doc.source = str(Path(doc.source).resolve())
        except Exception:  # noqa: BLE001
            pass
        for chunk in strategy.chunk(doc):
            yield CodeChunk.from_retrieval(chunk, root=base)
