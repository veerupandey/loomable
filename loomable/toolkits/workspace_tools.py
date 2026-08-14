"""Virtual workspace filesystem for deep agents (LangGraph-style state FS).

Stores files in process memory (and optionally mirrors to disk under ``root``)
so agents can offload drafts/notes without stuffing the context window.
"""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Any

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit

__all__ = ["WorkspaceStore", "WorkspaceTools"]


class WorkspaceStore:
    """Path → text content map with optional disk mirror."""

    def __init__(self, root: str | Path | None = None, *, mirror_disk: bool = True) -> None:
        self._root = Path(root).resolve() if root else None
        self._mirror = bool(mirror_disk) and self._root is not None
        self._files: dict[str, str] = {}
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
            if self._mirror:
                self._hydrate_from_disk()

    def _norm(self, path: str) -> str | None:
        raw = (path or "").strip().replace("\\", "/")
        if not raw or raw.startswith("/"):
            raw = raw.lstrip("/")
        if ".." in raw.split("/"):
            return None
        return raw or ""

    def _hydrate_from_disk(self) -> None:
        assert self._root is not None
        for p in self._root.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(self._root)).replace("\\", "/")
                try:
                    self._files[rel] = p.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue

    def _mirror_write(self, path: str, content: str) -> None:
        if not self._mirror or self._root is None:
            return
        dest = self._root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    def _mirror_delete(self, path: str) -> None:
        if not self._mirror or self._root is None:
            return
        dest = self._root / path
        if dest.is_file():
            dest.unlink()

    def ls(self, path: str = "") -> list[str]:
        prefix = self._norm(path) or ""
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        names: set[str] = set()
        for key in self._files:
            if prefix and not key.startswith(prefix):
                continue
            rest = key[len(prefix) :] if prefix else key
            if not rest:
                continue
            first = rest.split("/", 1)[0]
            if "/" in rest:
                names.add(first + "/")
            else:
                names.add(first)
        return sorted(names)

    def read(self, path: str) -> str | None:
        key = self._norm(path)
        if key is None:
            return None
        return self._files.get(key)

    def write(self, path: str, content: str) -> str | None:
        key = self._norm(path)
        if key is None or key == "":
            return None
        self._files[key] = content
        self._mirror_write(key, content)
        return key

    def edit(self, path: str, old: str, new: str, *, replace_all: bool = False) -> tuple[str | None, str]:
        key = self._norm(path)
        if key is None:
            return None, "invalid path"
        cur = self._files.get(key)
        if cur is None:
            return None, "file not found"
        if old not in cur:
            return None, "old_string not found"
        count = cur.count(old)
        if count > 1 and not replace_all:
            return None, f"old_string matched {count} times; set replace_all=true or make it unique"
        updated = cur.replace(old, new) if replace_all else cur.replace(old, new, 1)
        self._files[key] = updated
        self._mirror_write(key, updated)
        return key, "ok"

    def delete(self, path: str) -> bool:
        key = self._norm(path)
        if key is None or key not in self._files:
            return False
        del self._files[key]
        self._mirror_delete(key)
        return True

    def glob(self, pattern: str) -> list[str]:
        pat = (pattern or "*").strip()
        return sorted(k for k in self._files if fnmatch.fnmatch(k, pat) or fnmatch.fnmatch(k.split("/")[-1], pat))

    def grep(self, query: str, *, path: str = "", max_hits: int = 40) -> list[dict[str, Any]]:
        prefix = self._norm(path) or ""
        hits: list[dict[str, Any]] = []
        try:
            rx = re.compile(query)
        except re.error:
            rx = None
        for key, content in sorted(self._files.items()):
            if prefix and not (key == prefix or key.startswith(prefix.rstrip("/") + "/")):
                continue
            for i, line in enumerate(content.splitlines(), start=1):
                matched = bool(rx.search(line)) if rx is not None else (query in line)
                if matched:
                    hits.append({"path": key, "line": i, "text": line[:240]})
                    if len(hits) >= max_hits:
                        return hits
        return hits


class WorkspaceTools(Toolkit):
    """Virtual FS tools: ``ls``, ``read_file``, ``write_file``, ``edit_file``, ``glob``, ``grep``.

    These intentionally reuse familiar names where possible so prompts transfer
    from LangGraph deep agents. Prefer this over :class:`FileTools` when the
    agent should offload context into a sandboxed workspace.
    """

    def __init__(
        self,
        *,
        store: WorkspaceStore | None = None,
        root: str | Path | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        self._store = store or WorkspaceStore(root)

    @property
    def store(self) -> WorkspaceStore:
        return self._store

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._ls, name="ls"),
            FunctionTool(self._read_file, name="read_file"),
            FunctionTool(self._write_file, name="write_file", idempotent=False),
            FunctionTool(self._edit_file, name="edit_file", idempotent=False),
            FunctionTool(self._glob, name="glob"),
            FunctionTool(self._grep, name="grep"),
        ]

    async def _ls(self, path: str = "") -> str:
        """List files and directories in the workspace (virtual filesystem)."""
        return json.dumps({"path": path or "/", "entries": self._store.ls(path)}, indent=2)

    async def _read_file(self, path: str) -> str:
        """Read a workspace file. Prefer this over stuffing long drafts into chat."""
        data = self._store.read(path)
        if data is None:
            return f"Error: File not found or invalid path: {path}"
        return data

    async def _write_file(self, path: str, content: str) -> str:
        """Write/overwrite a workspace file (create parents as needed)."""
        key = self._store.write(path, content)
        if key is None:
            return f"Error: Invalid path: {path}"
        return json.dumps({"ok": True, "path": key, "bytes": len(content.encode("utf-8"))})

    async def _edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Patch a file by replacing ``old_string`` with ``new_string``."""
        key, msg = self._store.edit(path, old_string, new_string, replace_all=replace_all)
        if key is None:
            return f"Error: {msg}"
        return json.dumps({"ok": True, "path": key, "detail": msg})

    async def _glob(self, pattern: str = "**/*") -> str:
        """Find workspace paths matching a glob pattern."""
        return json.dumps({"pattern": pattern, "matches": self._store.glob(pattern)}, indent=2)

    async def _grep(self, query: str, path: str = "", max_hits: int = 40) -> str:
        """Search workspace file contents (regex if valid, else substring)."""
        hits = self._store.grep(query, path=path, max_hits=int(max_hits) or 40)
        return json.dumps({"query": query, "hits": hits}, indent=2)
