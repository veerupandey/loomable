"""CodeIndex — embed + store codebase chunks in a pluggable vector backend."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from loomable.codeindex.chunking import CodeChunk, iter_code_chunks
from loomable.codeindex.embedders import HashingEmbedder
from loomable.kernel.contracts import VectorBackend
from loomable.kernel.long_term import LongTermStore, open_vector_store  # noqa: F401


@dataclass(frozen=True)
class CodeHit:
    """A ranked search hit from :meth:`CodeIndex.search`."""

    path: str
    score: float
    start_line: int
    end_line: int
    kind: str
    name: str
    language: str
    text: str
    chunk_id: str

    def preview(self, max_chars: int = 500) -> str:
        body = self.text if len(self.text) <= max_chars else self.text[:max_chars] + "\n…"
        return (
            f"{self.path}:{self.start_line}-{self.end_line} "
            f"[{self.kind} {self.name}] score={self.score:.3f}\n{body}"
        )


class CodeIndex:
    """Indexed view of a repository for agent code understanding.

    Default file store is **Alibaba zvec** (``pip install loomable[zvec]``)
    under ``persist_path`` or ``<repo>/.loomable/codeindex_zvec``. Pass
    ``store=`` / ``backend=`` for Postgres (:class:`PgVectorBackend`) or any
    :class:`~loomable.kernel.contracts.VectorBackend`.

    Usage::

        index = await CodeIndex.build("./repo")  # Alibaba zvec on disk
        index = await CodeIndex.build(
            "./repo",
            store=open_vector_store(postgres_url=DSN, dimensions=1536),
        )
    """

    def __init__(
        self,
        root: str | Path,
        *,
        store: LongTermStore | None = None,
        embedder: Any | None = None,
        backend: VectorBackend | None = None,
        persist_path: str | Path | None = None,
        chunks: list[CodeChunk] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.embedder = embedder or HashingEmbedder()
        if store is not None:
            self.store = store
        elif backend is not None:
            self.store = LongTermStore(backend=backend, backend_name="custom")
        elif persist_path is not None:
            self.store = LongTermStore(path=persist_path, backend_name="zvec")
        else:
            cache = self.root / ".loomable" / "codeindex_zvec"
            self.store = LongTermStore(path=cache, backend_name="zvec")
        self._chunks: list[CodeChunk] = list(chunks or [])
        self._by_id: dict[str, CodeChunk] = {c.chunk_id: c for c in self._chunks}

    @classmethod
    async def build(
        cls,
        root: str | Path,
        *,
        embedder: Any | None = None,
        store: LongTermStore | None = None,
        backend: VectorBackend | None = None,
        persist_path: str | Path | None = None,
        extensions: Sequence[str] | None = None,
        max_files: int = 5_000,
        rebuild: bool = True,
    ) -> "CodeIndex":
        """Walk ``root``, embed chunks, and index them into the vector store."""
        index = cls(
            root,
            embedder=embedder,
            store=store,
            backend=backend,
            persist_path=persist_path,
        )
        await index.areindex(
            extensions=extensions, max_files=max_files, rebuild=rebuild
        )
        return index

    @classmethod
    def build_sync(cls, root: str | Path, **kwargs: Any) -> "CodeIndex":
        """Sync wrapper around :meth:`build` for non-async callers."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(cls.build(root, **kwargs))
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(lambda: asyncio.run(cls.build(root, **kwargs))).result()

    async def areindex(
        self,
        *,
        extensions: Sequence[str] | None = None,
        max_files: int = 5_000,
        rebuild: bool = True,
    ) -> int:
        """(Re)index the repository. Returns number of chunks indexed."""
        chunks = list(
            iter_code_chunks(self.root, extensions=extensions, max_files=max_files)
        )
        self._chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        if rebuild:
            # Soft clear: overwrite ids; stale ids may linger in custom backends.
            pass
        for chunk in chunks:
            vector = await self.embedder.embed(
                f"{chunk.path} {chunk.kind} {chunk.name}\n{chunk.text}"
            )
            await self.store.index(
                id=chunk.chunk_id,
                vector=vector,
                metadata={
                    "path": chunk.path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "kind": chunk.kind,
                    "name": chunk.name,
                    "language": chunk.language,
                    "text": chunk.text[:8_000],
                    "source": "codeindex",
                },
            )
        # Side metadata for repo_map without vector query
        meta_path = self.root / ".loomable" / "codeindex.meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "path": c.path,
                        "kind": c.kind,
                        "name": c.name,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "language": c.language,
                    }
                    for c in chunks
                ]
            ),
            encoding="utf-8",
        )
        return len(chunks)

    async def search(self, query: str, *, k: int = 8) -> list[CodeHit]:
        """Semantic search over indexed chunks."""
        vector = await self.embedder.embed(query)
        raw = await self.store.query(vector, k=max(1, int(k)))
        hits: list[CodeHit] = []
        for row in raw:
            if row.get("source") not in (None, "codeindex"):
                # Allow rows without source (fresh zvec) but prefer ours
                if "path" not in row:
                    continue
            hits.append(
                CodeHit(
                    path=str(row.get("path") or ""),
                    score=float(row.get("score") or 0.0),
                    start_line=int(row.get("start_line") or 1),
                    end_line=int(row.get("end_line") or 1),
                    kind=str(row.get("kind") or "file"),
                    name=str(row.get("name") or ""),
                    language=str(row.get("language") or ""),
                    text=str(row.get("text") or ""),
                    chunk_id=str(row.get("id") or ""),
                )
            )
        return hits

    def find_symbol(self, name: str, *, limit: int = 20) -> list[CodeChunk]:
        """Exact / case-insensitive symbol name lookup from the chunk catalog."""
        needle = (name or "").strip().lower()
        if not needle:
            return []
        out = [c for c in self._chunks if c.name.lower() == needle]
        if not out:
            out = [c for c in self._chunks if needle in c.name.lower()]
        return out[:limit]

    def repo_map(self, *, max_entries: int = 80) -> str:
        """Compact outline of the repo (paths + top symbols) for the model."""
        if not self._chunks:
            return "(empty code index)"
        by_path: dict[str, list[CodeChunk]] = {}
        for c in self._chunks:
            by_path.setdefault(c.path, []).append(c)
        lines: list[str] = [f"Repo map for {self.root.name} ({len(by_path)} files):"]
        count = 0
        for path in sorted(by_path):
            syms = [
                c
                for c in by_path[path]
                if c.kind in {"class", "function"} and c.name
            ][:6]
            if syms:
                sym_txt = ", ".join(f"{c.kind}:{c.name}" for c in syms)
                lines.append(f"- {path} ({sym_txt})")
            else:
                lines.append(f"- {path}")
            count += 1
            if count >= max_entries:
                lines.append(f"… ({len(by_path) - max_entries} more files)")
                break
        return "\n".join(lines)

    def as_knowledge(self, *, max_chunks: int = 40, max_chars: int = 2_000) -> list[str]:
        """Flatten top chunks into strings for ``Agent(knowledge=[...])``."""
        docs: list[str] = []
        for c in self._chunks[:max_chunks]:
            body = c.text if len(c.text) <= max_chars else c.text[:max_chars] + "\n…"
            docs.append(f"{c.path}:{c.start_line} [{c.kind} {c.name}]\n{body}")
        return docs

    @property
    def size(self) -> int:
        return len(self._chunks)
