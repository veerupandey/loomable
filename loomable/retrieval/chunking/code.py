"""Code-aware chunking (regex symbols; optional tree-sitter when installed)."""

from __future__ import annotations

import re
from typing import Any

from loomable.retrieval.chunking.base import register_strategy
from loomable.retrieval.chunking.text import TextChunker
from loomable.retrieval.types import Chunk, Document, merge_metadata

_SYMBOL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<kind>class|def|async\s+def|function|export\s+(?:default\s+)?"
    r"(?:async\s+)?function|export\s+class|interface|type|fn|func|pub\s+(?:async\s+)?"
    r"fn|public\s+class|struct)\s+(?P<name>[A-Za-z_][\w]*)",
    re.MULTILINE,
)


def _try_tree_sitter_chunk(document: Document) -> list[Chunk] | None:
    """Optional tree-sitter path — returns None if unavailable."""
    try:
        from tree_sitter_languages import get_parser  # type: ignore
    except Exception:  # noqa: BLE001
        return None

    lang = (document.metadata.get("language") or "").lower()
    src = (document.source or "").lower()
    lang_map = {
        "py": "python",
        "python": "python",
        "js": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "tsx": "tsx",
        "go": "go",
        "rs": "rust",
        "java": "java",
    }
    for ext, key in (
        (".py", "python"),
        (".js", "javascript"),
        (".jsx", "javascript"),
        (".ts", "typescript"),
        (".tsx", "tsx"),
        (".go", "go"),
        (".rs", "rust"),
        (".java", "java"),
    ):
        if src.endswith(ext):
            lang = key
            break
    parser_lang = lang_map.get(lang)
    if not parser_lang:
        return None
    try:
        parser = get_parser(parser_lang)
        tree = parser.parse(document.text.encode("utf-8"))
    except Exception:  # noqa: BLE001
        return None

    # Collect top-level definitions via simple node type names
    interesting = {
        "function_definition",
        "class_definition",
        "function_declaration",
        "class_declaration",
        "method_definition",
        "export_statement",
        "function_item",
        "impl_item",
        "struct_item",
    }
    text = document.text
    chunks: list[Chunk] = []
    root = tree.root_node

    def walk(node: Any, depth: int = 0) -> None:
        if node.type in interesting and depth <= 2:
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            body = text.encode("utf-8")[node.start_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            )
            name = ""
            for child in node.children:
                if "name" in child.type or child.type in {"identifier", "type_identifier"}:
                    name = text.encode("utf-8")[child.start_byte : child.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    break
            kind = "class" if "class" in node.type or "struct" in node.type else "function"
            chunks.append(
                Chunk(
                    id=f"{document.id}:ts:{start}-{end}:{kind}:{name or 'anon'}",
                    text=body[:12_000],
                    document_id=document.id,
                    start_line=start,
                    end_line=end,
                    kind=kind,
                    name=name or "anon",
                    metadata=merge_metadata(
                        document.metadata,
                        {
                            "source": document.source,
                            "path": document.source,
                            "parser": "tree-sitter",
                        },
                    ),
                )
            )
        for child in node.children:
            walk(child, depth + 1)

    walk(root)
    return chunks or None


class CodeChunker:
    """Symbol-oriented code chunker with tree-sitter upgrade when present."""

    name = "code"

    def __init__(self, *, max_file_lines: int = 200) -> None:
        self.max_file_lines = max_file_lines
        self._text = TextChunker(max_chars=4_000)

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text or ""
        if not text.strip():
            return []

        ts_chunks = _try_tree_sitter_chunk(document)
        if ts_chunks is not None:
            # Also keep a file overview for small files
            lines = text.splitlines()
            n = len(lines)
            if n <= self.max_file_lines:
                ts_chunks.insert(
                    0,
                    Chunk(
                        id=f"{document.id}:1-{n}:file",
                        text=text if n <= 400 else "\n".join(lines[:400]),
                        document_id=document.id,
                        start_line=1,
                        end_line=n,
                        kind="file",
                        name=document.metadata.get("filename")
                        or (document.source.split("/")[-1] if document.source else document.id),
                        metadata=merge_metadata(
                            document.metadata,
                            {"source": document.source, "path": document.source},
                        ),
                    ),
                )
            return ts_chunks

        lines = text.splitlines()
        n = len(lines)
        symbols: list[tuple[int, str, str]] = []
        for match in _SYMBOL_RE.finditer(text):
            start = text[: match.start()].count("\n") + 1
            kind_raw = re.sub(r"\s+", " ", match.group("kind")).strip().lower()
            kind = (
                "class"
                if any(x in kind_raw for x in ("class", "interface", "struct"))
                else "function"
            )
            symbols.append((start, kind, match.group("name")))

        meta = merge_metadata(
            document.metadata,
            {"source": document.source, "path": document.source},
        )
        chunks: list[Chunk] = []
        file_name = document.metadata.get("filename") or (
            document.source.split("/")[-1] if document.source else document.id
        )
        if not symbols or n <= self.max_file_lines:
            chunks.append(
                Chunk(
                    id=f"{document.id}:1-{n}:file",
                    text=text if n <= 400 else "\n".join(lines[:400]),
                    document_id=document.id,
                    start_line=1,
                    end_line=n,
                    kind="file",
                    name=str(file_name),
                    metadata=meta,
                )
            )
        if not symbols:
            return chunks or self._text.chunk(document)

        for i, (start, kind, name) in enumerate(symbols):
            end = (symbols[i + 1][0] - 1) if i + 1 < len(symbols) else n
            end = max(end, start)
            body = "\n".join(lines[start - 1 : end])
            if len(body) > 12_000:
                body = body[:12_000]
            chunks.append(
                Chunk(
                    id=f"{document.id}:{start}-{end}:{kind}:{name}",
                    text=body,
                    document_id=document.id,
                    start_line=start,
                    end_line=end,
                    kind=kind,
                    name=name,
                    metadata=meta,
                )
            )
        return chunks


register_strategy(CodeChunker())
