"""Todo list toolkit — LangGraph-style ``write_todos`` planning for deep agents.

Keeps a structured checklist the model can rewrite as work progresses.
Optionally persists to ``{workspace}/todos.json`` so long runs survive rebuilds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from loomable.agent.tools import FunctionTool
from loomable.toolkits._base import Toolkit

TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]

__all__ = ["TodoTools", "TodoStore"]


class TodoStore:
    """In-memory todo list with optional JSON persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._items: list[dict[str, Any]] = []
        if self._path and self._path.is_file():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._items = [self._normalize(x) for x in data if isinstance(x, dict)]
            except (OSError, json.JSONDecodeError, TypeError):
                self._items = []

    @staticmethod
    def _normalize(item: dict[str, Any]) -> dict[str, Any]:
        content = str(item.get("content") or item.get("title") or "").strip()
        status = str(item.get("status") or "pending").strip().lower()
        if status not in ("pending", "in_progress", "completed", "cancelled"):
            status = "pending"
        return {"content": content, "status": status}

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._items, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def replace(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._items = [self._normalize(x) for x in items if str(x.get("content") or "").strip()]
        self._persist()
        return list(self._items)

    def list(self) -> list[dict[str, Any]]:
        return list(self._items)

    def update(self, index: int, *, status: str | None = None, content: str | None = None) -> dict[str, Any] | None:
        if index < 0 or index >= len(self._items):
            return None
        item = dict(self._items[index])
        if content is not None and content.strip():
            item["content"] = content.strip()
        if status is not None:
            item = self._normalize({**item, "status": status})
        self._items[index] = item
        self._persist()
        return item


class TodoTools(Toolkit):
    """Planning toolkit: ``write_todos``, ``read_todos``, ``update_todo``.

    Usage::

        from loomable.toolkits import TodoTools
        agent = Agent(model=..., tools=[TodoTools(workspace=\"./.deep_workspace\")])
    """

    def __init__(
        self,
        *,
        store: TodoStore | None = None,
        workspace: str | Path | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        if store is not None:
            self._store = store
        elif workspace is not None:
            self._store = TodoStore(Path(workspace) / "todos.json")
        else:
            self._store = TodoStore()

    @property
    def store(self) -> TodoStore:
        return self._store

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._write_todos, name="write_todos", idempotent=False),
            FunctionTool(self._read_todos, name="read_todos"),
            FunctionTool(self._update_todo, name="update_todo", idempotent=False),
        ]

    async def _write_todos(self, todos: str) -> str:
        """Replace the full todo list.

        Pass ``todos`` as a JSON array of objects:
        ``[{"content": "...", "status": "pending|in_progress|completed|cancelled"}, ...]``

        Prefer rewriting the whole list when the plan changes substantially.
        Keep at most one item ``in_progress`` at a time.
        """
        try:
            data = json.loads(todos)
        except json.JSONDecodeError as exc:
            return f"Error: todos must be JSON array: {exc}"
        if not isinstance(data, list):
            return "Error: todos must be a JSON array of {content, status} objects"
        items = self._store.replace(data)
        return json.dumps({"ok": True, "count": len(items), "todos": items}, indent=2)

    async def _read_todos(self) -> str:
        """Return the current todo list as JSON."""
        return json.dumps({"todos": self._store.list()}, indent=2)

    async def _update_todo(self, index: int, status: str = "", content: str = "") -> str:
        """Update one todo by 0-based index. Provide ``status`` and/or ``content``."""
        item = self._store.update(
            int(index),
            status=status or None,
            content=content or None,
        )
        if item is None:
            return json.dumps({"error": f"unknown todo index {index}"})
        return json.dumps({"ok": True, "todo": item, "todos": self._store.list()}, indent=2)
