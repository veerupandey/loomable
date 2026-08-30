"""loomable.agent.notes - Durable structured notes for cross-session memory.

Provides :class:`NoteStore` (structured, deduplicated notes over the kernel
:class:`~loomable.kernel.long_term.LongTermStore`) and :func:`make_memory_tool`
which exposes note operations as a single ``memory`` tool for the agent.

Design principle: "one lesson per file, update don't duplicate, delete when wrong."
"""

from __future__ import annotations

__all__ = ["Note", "NoteStore", "make_memory_tool"]

import json
from dataclasses import dataclass, field
from typing import Any

from loomable.kernel.long_term import LongTermStore
from loomable.providers.embedders import Embedder

from loomable.agent.tools import FunctionTool


@dataclass
class Note:
    """A single durable note — one lesson per file."""

    note_id: str
    text: str
    tags: list[str] = field(default_factory=list)


class NoteStore:
    """Structured, deduplicated notes over the existing kernel LongTermStore.

    'One lesson per file, update don't duplicate, delete when wrong.'

    Notes are persisted as vectors in the LongTermStore with metadata carrying
    the note text and tags. The embedder is used to vectorize note text for
    indexing and recall (vector search).
    """

    def __init__(self, long_term: LongTermStore, embedder: Embedder) -> None:
        self._store = long_term
        self._embedder = embedder

    async def write(self, note_id: str, text: str, tags: list[str] | tuple[str, ...] = ()) -> Note:
        """Upsert a note by id. Updates existing notes rather than duplicating."""
        tags_list = list(tags)
        vector = await self._embedder.embed(text)
        metadata = {
            "note_id": note_id,
            "text": text,
            "tags": json.dumps(tags_list),
        }
        await self._store.index(note_id, vector, metadata)
        return Note(note_id=note_id, text=text, tags=tags_list)

    async def read(self, note_id: str) -> Note | None:
        """Read a single note by id. Returns None if not found."""
        result = await self._store.get(note_id)
        if result is None:
            return None
        return self._result_to_note(result)

    async def list(self, tag: str | None = None) -> list[Note]:
        """List all notes, optionally filtered by tag."""
        results = await self._store.scan()
        notes: list[Note] = []
        for result in results:
            note = self._result_to_note(result)
            if tag is None or tag in note.tags:
                notes.append(note)
        return notes

    async def delete(self, note_id: str) -> None:
        """Delete a note by id."""
        await self._store.delete(note_id)

    async def recall(self, query: str, k: int = 3) -> list[Note]:
        """Vector search for notes most relevant to the query."""
        vector = await self._embedder.embed(query)
        results = await self._store.query(vector, k)
        return [self._result_to_note(r) for r in results]

    @staticmethod
    def _result_to_note(result: dict[str, Any]) -> Note:
        """Convert a backend result dict into a Note."""
        note_id = result.get("note_id", result.get("id", ""))
        text = result.get("text", "")
        tags_raw = result.get("tags", "[]")
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
            except (json.JSONDecodeError, TypeError):
                tags = []
        elif isinstance(tags_raw, list):
            tags = tags_raw
        else:
            tags = []
        return Note(note_id=note_id, text=text, tags=tags)


def make_memory_tool(store: NoteStore) -> FunctionTool:
    """Create a single FunctionTool 'memory' with action-based dispatch.

    The tool accepts an ``action`` parameter from {write, read, list, delete, recall}
    and routes to the appropriate NoteStore method, giving the agent cross-session
    note-taking.
    """

    async def memory(
        action: str,
        note_id: str = "",
        text: str = "",
        tags: str = "",
        query: str = "",
        k: int = 3,
    ) -> str:
        """Manage durable notes across sessions.

        Actions:
        - write: Upsert a note (requires note_id, text; optional tags as comma-separated).
        - read: Read a note by id (requires note_id).
        - list: List all notes (optional tags to filter by a single tag).
        - delete: Delete a note by id (requires note_id).
        - recall: Vector search for relevant notes (requires query; optional k).
        """
        if action == "write":
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            note = await store.write(note_id, text, tag_list)
            return json.dumps({"note_id": note.note_id, "text": note.text, "tags": note.tags})
        elif action == "read":
            note = await store.read(note_id)
            if note is None:
                return json.dumps({"error": f"Note '{note_id}' not found."})
            return json.dumps({"note_id": note.note_id, "text": note.text, "tags": note.tags})
        elif action == "list":
            tag_filter = tags.strip() if tags.strip() else None
            notes = await store.list(tag_filter)
            return json.dumps([
                {"note_id": n.note_id, "text": n.text, "tags": n.tags} for n in notes
            ])
        elif action == "delete":
            await store.delete(note_id)
            return json.dumps({"deleted": note_id})
        elif action == "recall":
            notes = await store.recall(query or text, k)
            return json.dumps([
                {"note_id": n.note_id, "text": n.text, "tags": n.tags} for n in notes
            ])
        else:
            return json.dumps({"error": f"Unknown action '{action}'. Use: write, read, list, delete, recall."})

    return FunctionTool(
        memory,
        name="memory",
        description=(
            "Manage durable notes across sessions. "
            "Actions: write (upsert note by id), read (get note by id), "
            "list (list notes, optionally filter by tag), delete (remove note by id), "
            "recall (vector search for relevant notes)."
        ),
    )
