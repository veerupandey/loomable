"""MemoryStore protocol and TieredMemoryStore default.

Provides a pluggable, shared, four-tier memory component (working, episodic,
semantic, procedural) behind one interface. Zero-dependency default
implementation included.

Tiers:
- WORKING: scratch-pad for the current run context (in-memory deque per session).
- EPISODIC: past events and turns (in-memory deque per session).
- SEMANTIC: long-term factual knowledge (delegates to kernel LongTermStore).
- PROCEDURAL: learned instructions/rules (reuses notes pattern, in-memory list).
"""

from __future__ import annotations

__all__ = ["MemoryStore", "Tier", "TieredMemoryStore"]

import time
from collections import deque
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Tier(str, Enum):
    """Functional memory tiers (Req 12.1)."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for a pluggable, tiered memory component (Req 12.1, 12.7).

    Any object implementing ``write`` and ``recall`` with these signatures
    can be used as a Flow's memory store, making the store fully swappable
    without changes to Flow, node, or engine code.
    """

    async def write(self, record: str, *, tier: Tier = Tier.EPISODIC, **meta: Any) -> None:
        """Store a record under the specified tier (default EPISODIC — Req 12.3)."""
        ...

    async def recall(
        self, query: str, *, tiers: list[Tier] | None = None, k: int = 5
    ) -> list[dict[str, Any]]:
        """Recall up to *k* records relevant to *query* across requested tiers (Req 12.4).

        If *tiers* is None, searches all tiers.
        """
        ...


# ---------------------------------------------------------------------------
# Session-scoped storage for working and episodic tiers (Req 12.5)
# ---------------------------------------------------------------------------

# Module-level registry keyed by session_id so that multiple instantiations of
# TieredMemoryStore with the same session_id share their working/episodic data.
_session_stores: dict[str, dict[str, deque[dict[str, Any]]]] = {}


def _get_session_deques(session_id: str) -> dict[str, deque[dict[str, Any]]]:
    """Return (or create) the per-session deque pair for working and episodic."""
    if session_id not in _session_stores:
        _session_stores[session_id] = {
            "working": deque(maxlen=100),
            "episodic": deque(maxlen=1000),
        }
    return _session_stores[session_id]


# ---------------------------------------------------------------------------
# TieredMemoryStore — zero-dependency default (Req 12.1)
# ---------------------------------------------------------------------------


class TieredMemoryStore:
    """Zero-dependency default MemoryStore (Req 12.1).

    - Working/Episodic: in-memory deques, persisted per session_id across
      instantiations of the same session (Req 12.5).
    - Semantic: delegates to the kernel LongTermStore when provided (Req 12.6).
      Does NOT modify the kernel — only reads/writes through its existing API.
    - Procedural: in-memory list of instructions/notes (reuses the NoteStore
      pattern conceptually; stored as dicts).

    Parameters
    ----------
    session_id:
        Scopes working and episodic tiers. If None, uses a private instance
        that does not persist across instantiations.
    long_term_store:
        An optional kernel LongTermStore instance for the semantic tier.
        When absent, semantic writes are stored in-memory (simple fallback).
    """

    def __init__(
        self,
        session_id: str | None = None,
        long_term_store: Any | None = None,
    ) -> None:
        self._session_id = session_id
        self._long_term_store = long_term_store

        # Working and episodic: session-scoped deques (Req 12.5)
        if session_id is not None:
            deques = _get_session_deques(session_id)
            self._working: deque[dict[str, Any]] = deques["working"]
            self._episodic: deque[dict[str, Any]] = deques["episodic"]
        else:
            self._working = deque(maxlen=100)
            self._episodic = deque(maxlen=1000)

        # Procedural: in-memory list of instructions/notes
        self._procedural: list[dict[str, Any]] = []

        # Semantic fallback (when no LongTermStore provided)
        self._semantic_fallback: list[dict[str, Any]] = []

    async def write(self, record: str, *, tier: Tier = Tier.EPISODIC, **meta: Any) -> None:
        """Store a record under the specified tier (default EPISODIC — Req 12.3).

        Parameters
        ----------
        record:
            The text content to store.
        tier:
            Which memory tier to write to. Defaults to EPISODIC.
        **meta:
            Arbitrary metadata stored alongside the record.
        """
        entry: dict[str, Any] = {
            "record": record,
            "tier": tier.value,
            "timestamp": time.time(),
            **meta,
        }

        if tier == Tier.WORKING:
            self._working.append(entry)
        elif tier == Tier.EPISODIC:
            self._episodic.append(entry)
        elif tier == Tier.SEMANTIC:
            if self._long_term_store is not None:
                # Delegate to kernel LongTermStore (Req 12.6)
                # Use a simple embedding placeholder — the LongTermStore's
                # index method requires a vector. We store the record text
                # as metadata and use a zero-vector as a simple key.
                import hashlib

                record_id = hashlib.sha1(record.encode()).hexdigest()[:16]
                # Store with metadata; vector is a placeholder (real embeddings
                # would be provided by a proper embedder in production)
                await self._long_term_store.index(
                    id=record_id,
                    vector=[0.0],
                    metadata={"record": record, "tier": tier.value, **meta},
                )
            else:
                self._semantic_fallback.append(entry)
        elif tier == Tier.PROCEDURAL:
            self._procedural.append(entry)

    async def recall(
        self, query: str, *, tiers: list[Tier] | None = None, k: int = 5
    ) -> list[dict[str, Any]]:
        """Recall up to *k* records relevant to *query* across requested tiers.

        If *tiers* is None, searches all tiers. Returns records most relevant
        to the query (simple substring match for in-memory tiers; vector search
        for semantic tier when LongTermStore is available).

        Parameters
        ----------
        query:
            The search query string.
        tiers:
            Which tiers to search. None means all tiers.
        k:
            Maximum number of records to return.

        Returns
        -------
        list[dict]:
            Records matching the query, each containing at minimum 'record'
            and 'tier' fields plus any stored metadata.
        """
        search_tiers = tiers if tiers is not None else list(Tier)
        results: list[dict[str, Any]] = []

        query_lower = query.lower()

        if Tier.WORKING in search_tiers:
            for entry in self._working:
                if query_lower in entry["record"].lower():
                    results.append(entry)

        if Tier.EPISODIC in search_tiers:
            for entry in self._episodic:
                if query_lower in entry["record"].lower():
                    results.append(entry)

        if Tier.SEMANTIC in search_tiers:
            if self._long_term_store is not None:
                # Query via the LongTermStore (Req 12.6)
                try:
                    lts_results = await self._long_term_store.query(vector=[0.0], k=k)
                    for r in lts_results:
                        if query_lower in r.get("record", "").lower():
                            results.append({
                                "record": r.get("record", ""),
                                "tier": Tier.SEMANTIC.value,
                                **{k_: v for k_, v in r.items() if k_ not in ("record", "tier", "id", "score")},
                            })
                except Exception:
                    pass  # Gracefully handle backend errors
            else:
                for entry in self._semantic_fallback:
                    if query_lower in entry["record"].lower():
                        results.append(entry)

        if Tier.PROCEDURAL in search_tiers:
            for entry in self._procedural:
                if query_lower in entry["record"].lower():
                    results.append(entry)

        # Sort by timestamp (most recent first) and limit to k
        results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return results[:k]
