"""Code navigation toolkit backed by :class:`~loomable.codeindex.CodeIndex`."""

from __future__ import annotations

import json
from typing import Any

from loomable.agent.tools import FunctionTool
from loomable.codeindex import CodeIndex
from loomable.toolkits._base import Toolkit


class CodeTools(Toolkit):
    """Tools for understanding a repository via a :class:`CodeIndex`.

    Usage::

        index = await CodeIndex.build("./repo")
        agent = Agent(model=..., tools=[CodeTools(index)], skills=["coding"])
    """

    def __init__(
        self,
        index: CodeIndex,
        *,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        self._index = index

    @property
    def index(self) -> CodeIndex:
        return self._index

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._repo_map, name="repo_map"),
            FunctionTool(self._code_search, name="code_search"),
            FunctionTool(self._find_symbol, name="find_symbol"),
        ]

    async def _repo_map(self, max_entries: int = 80) -> str:
        """Return a compact outline of the indexed repository (paths + symbols)."""
        return self._index.repo_map(max_entries=max(1, int(max_entries)))

    async def _code_search(self, query: str, limit: int = 8) -> str:
        """Semantic search over the code index. Returns ranked path/line snippets."""
        q = (query or "").strip()
        if not q:
            return "Error: query is required"
        hits = await self._index.search(q, k=max(1, int(limit)))
        if not hits:
            return "No matching code chunks."
        return "\n\n---\n\n".join(h.preview() for h in hits)

    async def _find_symbol(self, name: str, limit: int = 20) -> str:
        """Find definitions matching a symbol name (exact, then substring)."""
        matches = self._index.find_symbol(name, limit=max(1, int(limit)))
        if not matches:
            return f"No symbols matching {name!r}."
        payload = [
            {
                "path": c.path,
                "kind": c.kind,
                "name": c.name,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "language": c.language,
            }
            for c in matches
        ]
        return json.dumps({"symbols": payload}, ensure_ascii=False, indent=2)
