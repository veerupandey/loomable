"""Compatibility shim — notes live in :mod:`loomable.memory.notes`.

Prefer::

    from loomable.memory import NoteStore, Note, make_memory_tool
"""

from __future__ import annotations

from loomable.memory.notes import Note, NoteStore, make_memory_tool

__all__ = ["Note", "NoteStore", "make_memory_tool"]
