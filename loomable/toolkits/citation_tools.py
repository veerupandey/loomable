"""Citation / source registry for research deep agents.

Persists sources under ``{workspace}/sources.json`` so long-horizon agents can
cite URLs without stuffing the full page into the final answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit

__all__ = ["CitationStore", "CitationTools"]


class CitationStore:
    """JSON-backed list of research sources."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._sources: list[dict[str, Any]] = []
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._sources = [x for x in data if isinstance(x, dict)]
            except (OSError, json.JSONDecodeError, TypeError):
                self._sources = []

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._sources, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def register(
        self,
        *,
        url: str,
        title: str = "",
        summary: str = "",
        quote: str = "",
    ) -> dict[str, Any]:
        url = (url or "").strip()
        if not url:
            raise ValueError("url is required")
        # Upsert by URL
        for existing in self._sources:
            if str(existing.get("url") or "") == url:
                if title:
                    existing["title"] = title
                if summary:
                    existing["summary"] = summary
                if quote:
                    existing["quote"] = quote
                self._persist()
                return dict(existing)

        host = urlparse(url).netloc or ""
        entry = {
            "id": f"S{len(self._sources) + 1}",
            "url": url,
            "title": (title or host or url).strip(),
            "summary": (summary or "").strip(),
            "quote": (quote or "").strip(),
            "host": host,
        }
        self._sources.append(entry)
        self._persist()
        return dict(entry)

    def list(self) -> list[dict[str, Any]]:
        return [dict(x) for x in self._sources]

    def bibliography_markdown(self) -> str:
        if not self._sources:
            return "_No sources registered._"
        lines = ["## Sources"]
        for src in self._sources:
            sid = src.get("id") or "?"
            title = src.get("title") or src.get("url")
            url = src.get("url") or ""
            summary = src.get("summary") or ""
            quote = src.get("quote") or ""
            block = f"- **[{sid}]** [{title}]({url})"
            if summary:
                block += f" — {summary}"
            lines.append(block)
            if quote:
                lines.append(f"  > {quote}")
        return "\n".join(lines) + "\n"


class CitationTools(Toolkit):
    """Research citations: ``register_source``, ``list_sources``, ``format_bibliography``.

    Usage::

        from loomable.toolkits import CitationTools
        agent = Agent(model=..., tools=[CitationTools(workspace=\"./.deep_workspace\")])
    """

    def __init__(
        self,
        workspace: str | Path = "./.deep_workspace",
        *,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        root = Path(workspace)
        root.mkdir(parents=True, exist_ok=True)
        self._store = CitationStore(root / "sources.json")

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._register_source, name="register_source"),
            FunctionTool(self._list_sources, name="list_sources"),
            FunctionTool(self._format_bibliography, name="format_bibliography"),
        ]

    async def _register_source(
        self,
        url: str,
        title: str = "",
        summary: str = "",
        quote: str = "",
    ) -> str:
        """Register a web source for the final brief citations."""
        try:
            entry = self._store.register(
                url=url, title=title, summary=summary, quote=quote
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return json.dumps({"ok": True, "source": entry}, ensure_ascii=False)

    async def _list_sources(self) -> str:
        """List all registered research sources."""
        return json.dumps({"sources": self._store.list()}, indent=2, ensure_ascii=False)

    async def _format_bibliography(self) -> str:
        """Return a Markdown bibliography block for the deliverable."""
        return self._store.bibliography_markdown()
