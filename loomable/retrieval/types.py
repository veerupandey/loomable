"""Shared document / chunk types for loomable retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class Document:
    """A source document before chunking."""

    id: str
    text: str
    source: str = ""
    media_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def kind_hint(self) -> str:
        """Best-effort kind from media type / source suffix."""
        mt = (self.media_type or "").lower()
        src = (self.source or self.id or "").lower()
        if "markdown" in mt or src.endswith((".md", ".mdx")):
            return "markdown"
        code_exts = (
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
            ".kts",
            ".c",
            ".h",
            ".cpp",
            ".hpp",
            ".cc",
            ".cxx",
            ".cs",
            ".rb",
            ".php",
            ".swift",
            ".scala",
            ".sc",
            ".m",
            ".mm",
            ".r",
            ".jl",
            ".lua",
            ".pl",
            ".ex",
            ".exs",
            ".erl",
            ".hs",
            ".dart",
            ".vue",
            ".svelte",
            ".tf",
            ".proto",
            ".graphql",
            ".gql",
            ".ipynb",
        )
        if any(src.endswith(ext) for ext in code_exts) or src.endswith("/dockerfile"):
            return "code"
        if "pdf" in mt or src.endswith(".pdf"):
            return "pdf"
        if src.endswith((".html", ".htm")) or "html" in mt:
            return "html"
        if src.endswith((".json", ".jsonl")) or "json" in mt:
            return "json"
        if src.endswith((".csv", ".tsv")) or "csv" in mt:
            return "csv"
        if src.endswith((".docx",)) or "wordprocessingml" in mt:
            return "text"
        if src.endswith((".pptx",)) or "presentationml" in mt:
            return "text"
        return "text"


@dataclass
class Chunk:
    """A retrieval unit derived from a :class:`Document`."""

    id: str
    text: str
    document_id: str
    start_line: int = 1
    end_line: int = 1
    kind: str = "text"  # text | section | class | function | page | …
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_result(self, *, score: float = 0.0) -> dict[str, Any]:
        """Shape expected by :class:`~loomable.kernel.contracts.Retriever`."""
        return {
            "id": self.id,
            "content": self.text,
            "score": score,
            "document_id": self.document_id,
            "source": self.metadata.get("source", ""),
            "path": self.metadata.get("path", self.metadata.get("source", "")),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "kind": self.kind,
            "name": self.name,
            **{
                k: v
                for k, v in self.metadata.items()
                if k not in {"source", "path"}
            },
        }


def merge_metadata(*parts: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in parts:
        if part:
            out.update(dict(part))
    return out
