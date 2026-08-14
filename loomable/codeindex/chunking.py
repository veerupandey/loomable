"""Walk a repo and emit file / symbol chunks for indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

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

_CODE_EXTS = {
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
}

# Python / JS-ish symbol starts
_SYMBOL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<kind>class|def|async\s+def|function|export\s+(?:default\s+)?"
    r"(?:async\s+)?function|export\s+class|interface|type|fn|func|pub\s+(?:async\s+)?"
    r"fn|public\s+class|struct)\s+(?P<name>[A-Za-z_][\w]*)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class CodeChunk:
    """One indexable unit of a codebase."""

    chunk_id: str
    path: str
    text: str
    start_line: int
    end_line: int
    kind: str  # file | class | function | other
    name: str
    language: str


def _language_for(path: Path) -> str:
    return path.suffix.lstrip(".").lower() or "txt"


def _should_skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or name.startswith(".")


def iter_source_files(
    root: Path,
    *,
    extensions: Sequence[str] | None = None,
    max_files: int = 5_000,
) -> Iterator[Path]:
    exts = {e if e.startswith(".") else f".{e}" for e in (extensions or _CODE_EXTS)}
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(_should_skip_dir(p) for p in path.relative_to(root).parts[:-1]):
            continue
        if path.suffix.lower() not in exts:
            continue
        # skip huge / binary-ish
        try:
            if path.stat().st_size > 1_500_000:
                continue
        except OSError:
            continue
        yield path
        count += 1
        if count >= max_files:
            return


def _chunk_file(root: Path, path: Path, *, max_file_lines: int = 200) -> list[CodeChunk]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rel = path.relative_to(root).as_posix()
    lang = _language_for(path)
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []

    symbols: list[tuple[int, str, str]] = []  # start_idx, kind, name
    for match in _SYMBOL_RE.finditer(text):
        # line number 1-based
        start = text[: match.start()].count("\n") + 1
        kind_raw = re.sub(r"\s+", " ", match.group("kind")).strip().lower()
        kind = "class" if "class" in kind_raw or "interface" in kind_raw or "struct" in kind_raw else "function"
        symbols.append((start, kind, match.group("name")))

    chunks: list[CodeChunk] = []
    # Always keep a file-level chunk for small files (or when no symbols).
    if not symbols or n <= max_file_lines:
        chunks.append(
            CodeChunk(
                chunk_id=f"{rel}:1-{n}:file",
                path=rel,
                text=text if n <= 400 else "\n".join(lines[:400]),
                start_line=1,
                end_line=n,
                kind="file",
                name=path.name,
                language=lang,
            )
        )
    if not symbols:
        return chunks

    for i, (start, kind, name) in enumerate(symbols):
        end = (symbols[i + 1][0] - 1) if i + 1 < len(symbols) else n
        end = max(end, start)
        body = "\n".join(lines[start - 1 : end])
        if len(body) > 12_000:
            body = body[:12_000]
        chunks.append(
            CodeChunk(
                chunk_id=f"{rel}:{start}-{end}:{kind}:{name}",
                path=rel,
                text=body,
                start_line=start,
                end_line=end,
                kind=kind,
                name=name,
                language=lang,
            )
        )
    return chunks


def iter_code_chunks(
    root: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    max_files: int = 5_000,
) -> Iterator[CodeChunk]:
    """Yield :class:`CodeChunk` values for ``root``."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return
    for path in iter_source_files(base, extensions=extensions, max_files=max_files):
        yield from _chunk_file(base, path)
