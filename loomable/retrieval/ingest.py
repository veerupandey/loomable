"""Load heterogeneous sources into :class:`~loomable.retrieval.types.Document`s."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Iterable, Sequence

from loomable.retrieval.types import Document

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".deep_workspace",
    ".sandbox",
    ".loomable",
    ".next",
    "target",
    "vendor",
}

_TEXT_EXTS = {
    ".txt",
    ".md",
    ".mdx",
    ".rst",
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
    ".html",
    ".htm",
    ".css",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".sql",
    ".sh",
    ".csv",
}


def _guess_media_type(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "text/plain"


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PDF ingest requires optional dependency 'pypdf' (pip install loomable[pdf])"
        ) from exc
    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        parts.append(f"--- Page {i + 1} ---\n{text}")
    return "\n".join(parts)


def load_file(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Load a single file into a Document (text/code/md/html/pdf)."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    media = _guess_media_type(p)
    if p.suffix.lower() == ".pdf":
        text = _read_pdf(p)
        media = "application/pdf"
    else:
        text = p.read_text(encoding="utf-8", errors="replace")
    return Document(
        id=doc_id or p.as_posix(),
        text=text,
        source=p.as_posix(),
        media_type=media,
        metadata={"filename": p.name, "suffix": p.suffix.lower()},
    )


def load_directory(
    root: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    max_files: int = 5_000,
) -> list[Document]:
    """Load text-like files under ``root`` (skips venv/git/build dirs)."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(str(base))
    exts = {
        e if e.startswith(".") else f".{e}"
        for e in (extensions or (*_TEXT_EXTS, ".pdf"))
    }
    out: list[Document] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(base).parts[:-1]
        if any(part in _SKIP_DIRS or part.startswith(".") for part in rel_parts):
            continue
        if path.suffix.lower() not in exts:
            continue
        try:
            if path.stat().st_size > 2_000_000 and path.suffix.lower() != ".pdf":
                continue
        except OSError:
            continue
        try:
            out.append(load_file(path))
        except Exception:  # noqa: BLE001 — skip unreadable
            continue
        if len(out) >= max_files:
            break
    return out


def coerce_source(source: Any) -> list[Document]:
    """Normalize a source spec into one or more Documents.

    Accepts:
    - :class:`Document`
    - ``Path`` / existing path string → file or directory
    - ``dict`` with ``text`` (+ optional ``id`` / ``source`` / ``media_type``)
    - raw ``str`` that is not a path → treated as inline text
    """
    if isinstance(source, Document):
        return [source]
    if isinstance(source, dict):
        text = str(source.get("text") or "")
        return [
            Document(
                id=str(source.get("id") or source.get("source") or f"doc-{hash(text) & 0xffff}"),
                text=text,
                source=str(source.get("source") or ""),
                media_type=str(source.get("media_type") or "text/plain"),
                metadata=dict(source.get("metadata") or {}),
            )
        ]
    if isinstance(source, Path) or (
        isinstance(source, str) and (Path(source).expanduser().exists())
    ):
        path = Path(source).expanduser()
        if path.is_dir():
            return load_directory(path)
        if path.is_file():
            return [load_file(path)]
    if isinstance(source, str):
        return [
            Document(
                id=f"inline-{hash(source) & 0xfffffff:x}",
                text=source,
                source="inline",
                media_type="text/plain",
            )
        ]
    raise TypeError(f"unsupported retrieval source type: {type(source)!r}")


def load_sources(sources: Iterable[Any]) -> list[Document]:
    """Flatten many source specs into Documents (dedupe by id)."""
    docs: list[Document] = []
    seen: set[str] = set()
    for raw in sources:
        for doc in coerce_source(raw):
            if doc.id in seen:
                continue
            seen.add(doc.id)
            docs.append(doc)
    return docs
