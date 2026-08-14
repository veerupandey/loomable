"""Composable Agent memory — assemble layers, pass one object to ``Agent(memory=...)``.

Layers (any subset)::

    from loomable.memory import (
        Memory, ConversationMemory, UserMemory, KnowledgeMemory, MemoryScope,
        open_session_store,
    )

    memory = Memory.compose(
        conversation=ConversationMemory(
            store=open_session_store("postgres", url=DSN, user_id="alice"),
            window=8,
        ),
        user=UserMemory(
            note_store=notes,
            memory_tool=True,
            auto_extract=True,
        ),
    )

    agent = Agent(
        model=...,
        memory=memory,
        session_id="c1",
        user_id="alice",
        scopes={"claim_id": "CLM-4421"},  # any extra isolation keys
    )
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

__all__ = [
    "MemoryScope",
    "ConversationMemory",
    "UserMemory",
    "KnowledgeMemory",
    "WorkingMemory",
    "Memory",
    "ScopedNoteStore",
    "is_memory_bundle",
    "is_kernel_memory_manager",
    "extract_user_facts",
    "auto_extract_into_notes",
]


def is_kernel_memory_manager(value: Any) -> bool:
    """True for vestigial kernel :class:`~loomable.kernel.memory.MemoryManager`."""
    return (
        value is not None
        and hasattr(value, "record_turn")
        and hasattr(value, "l1")
        and hasattr(value, "l3")
        and not isinstance(value, Memory)
    )


def is_memory_bundle(value: Any) -> bool:
    return isinstance(value, Memory)


@dataclass(frozen=True)
class MemoryScope:
    """Arbitrary isolation keys for long-term (and tenant) memory.

    Use any business keys — not only ``user_id``::

        MemoryScope.of(user_id="alice", claim_id="CLM-4421")
        MemoryScope.of(policy_id="POL-9", lob="auto")
        MemoryScope.from_mapping({"user_id": "alice", "claim_id": "CLM-4421"})

    Notes are stored under a stable prefix and tagged ``scope:key=value`` so
    recall never leaks across claims/users/tenants sharing one vector store.
    """

    parts: tuple[tuple[str, str], ...] = ()

    @classmethod
    def of(cls, **kwargs: Any) -> MemoryScope:
        pairs = tuple(
            sorted(
                (str(k), str(v))
                for k, v in kwargs.items()
                if v is not None and str(v).strip() != ""
            )
        )
        return cls(parts=pairs)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> MemoryScope:
        if not mapping:
            return cls()
        return cls.of(**dict(mapping))

    def merge(self, other: MemoryScope | Mapping[str, Any] | None) -> MemoryScope:
        """Return a new scope; ``other`` overrides duplicate keys."""
        base = dict(self.parts)
        if other is None:
            return self
        if isinstance(other, MemoryScope):
            base.update(dict(other.parts))
        else:
            for k, v in other.items():
                if v is not None and str(v).strip() != "":
                    base[str(k)] = str(v)
        return MemoryScope.of(**base)

    def __bool__(self) -> bool:
        return bool(self.parts)

    @property
    def prefix(self) -> str:
        """Stable id prefix, e.g. ``claim_id=CLM-1|user_id=alice``."""
        return "|".join(f"{k}={v}" for k, v in self.parts)

    @property
    def tags(self) -> list[str]:
        return [f"scope:{k}={v}" for k, v in self.parts]

    def get(self, key: str, default: str | None = None) -> str | None:
        for k, v in self.parts:
            if k == key:
                return v
        return default

    def as_dict(self) -> dict[str, str]:
        return dict(self.parts)

    def tenant_key(self) -> str:
        """Single string suitable for Postgres ``user_id`` / tenant columns."""
        return self.prefix or "_"


@dataclass
class ConversationMemory:
    """Short-term / thread memory (L1 turns + L2 summaries).

    Parameters
    ----------
    store:
        Session store (``open_session_store(...)`` / ``SessionStore`` / file / …).
    backend:
        Alternative to ``store`` — any ``MemoryBackend`` (wrapped as session store).
    window:
        Max raw L1 turns replayed (``memory_window``).
    compaction_threshold:
        Spill oldest L1 → L2 when exceeded.
    use_llm_summarizer:
        Use model-based compaction when True.
    enabled:
        When False, Agent runs without conversation replay even if session_id set.

    Thread isolation uses ``session_id`` (e.g. ``session_id=f"claim:{claim_id}"``).
    Tenant / user scoping for notes belongs on ``UserMemory`` / ``Agent(user_id=)``.
    """

    store: Any | None = None
    backend: Any | None = None
    window: int | None = None
    compaction_threshold: int | None = None
    use_llm_summarizer: bool = False
    enabled: bool = True


@dataclass
class UserMemory:
    """Long-term scoped facts (L3 notes).

    Parameters
    ----------
    note_store / long_term / embedder:
        Provide a NoteStore, or enough pieces to build one.
    memory_tool:
        Expose agentic ``memory`` tool. Requires ``note_store`` or ``embedder``
        (raises ``AgentConfigError`` otherwise).
    auto_extract:
        Heuristic Always-mode: extract facts from user text after each turn.
        Requires a resolvable note store (same as ``memory_tool``).
    user_id:
        Convenience for ``scopes={"user_id": ...}``.
    scopes:
        Extra isolation keys (``claim_id``, ``policy_id``, ``case_id``, …).
    """

    note_store: Any | None = None
    long_term: Any | None = None
    embedder: Any | None = None
    memory_tool: bool = True
    auto_extract: bool = False
    user_id: str | None = None
    scopes: Mapping[str, Any] | MemoryScope | None = None

    def resolve_scope(self) -> MemoryScope:
        scope = MemoryScope.from_mapping(
            self.scopes if isinstance(self.scopes, Mapping) else None
        )
        if isinstance(self.scopes, MemoryScope):
            scope = self.scopes
        if self.user_id:
            scope = scope.merge({"user_id": self.user_id})
        return scope


@dataclass
class KnowledgeMemory:
    """RAG layer: short documents (auto-recall) and/or a vector-DB knowledge base.

    ``documents`` are embedded and injected into the prompt (passive recall).
    ``store`` / ``sources`` / ``knowledge_base`` become searchable ``search_*``
    tools on the Agent — same as ``Agent(knowledge_base=...)``.
    """

    documents: list[str] = field(default_factory=list)
    embedder: Any = None
    top_k: int = 3
    store: Any | None = None
    sources: list[Any] | None = None
    knowledge_base: Any | None = None


@dataclass
class WorkingMemory:
    """Workflow blackboard (:class:`~loomable.flow.memory.TieredMemoryStore`).

    Not valid inside ``Agent(memory=Memory.compose(...))`` — that raises.
    Use ``Workflow(memory=True)`` or ``Workflow(memory=WorkingMemory.tiered().store)``.
    """

    store: Any | None = None
    enabled: bool = True

    @classmethod
    def tiered(cls, session_id: str | None = None) -> WorkingMemory:
        from loomable.flow.memory import TieredMemoryStore

        return cls(store=TieredMemoryStore(session_id=session_id))


@dataclass
class Memory:
    """Composable memory bundle for ``Agent(memory=...)``."""

    conversation: ConversationMemory | None = None
    user: UserMemory | None = None
    knowledge: KnowledgeMemory | None = None
    working: WorkingMemory | None = None

    @classmethod
    def compose(
        cls,
        *,
        conversation: ConversationMemory | None = None,
        user: UserMemory | None = None,
        knowledge: KnowledgeMemory | None = None,
        working: WorkingMemory | None = None,
    ) -> Memory:
        """Assemble memory layers for ``Agent(memory=...)``."""
        return cls(
            conversation=conversation,
            user=user,
            knowledge=knowledge,
            working=working,
        )

    def with_scopes(
        self,
        *,
        user_id: str | None = None,
        scopes: Mapping[str, Any] | MemoryScope | None = None,
    ) -> Memory:
        """Stamp Agent-level ``user_id`` / ``scopes`` onto the user layer."""
        if self.user is None:
            return self
        merged = self.user.resolve_scope()
        if user_id:
            merged = merged.merge({"user_id": user_id})
        if scopes:
            merged = merged.merge(scopes)
        if not merged:
            return self
        return replace(
            self,
            user=replace(
                self.user,
                user_id=merged.get("user_id") or self.user.user_id,
                scopes=merged,
            ),
        )

    def resolve_note_store(self) -> Any | None:
        """Materialize a (possibly scoped) NoteStore from the user layer."""
        if self.user is None:
            return None
        store = self.user.note_store
        if store is None and self.user.long_term is not None and self.user.embedder is not None:
            from loomable.agent.notes import NoteStore

            store = NoteStore(long_term=self.user.long_term, embedder=self.user.embedder)
        elif store is None and self.user.embedder is not None:
            from loomable.agent.notes import NoteStore
            from loomable.kernel.long_term import LongTermStore

            store = NoteStore(long_term=LongTermStore(), embedder=self.user.embedder)
        if store is None:
            return None
        scope = self.user.resolve_scope()
        if scope:
            return ScopedNoteStore(store, scope=scope)
        return store

    def to_agent_kwargs(self) -> dict[str, Any]:
        """Flatten into Agent constructor store kwargs (no Nones)."""
        out: dict[str, Any] = {}
        if self.conversation is not None and self.conversation.enabled:
            c = self.conversation
            if c.store is not None:
                out["session_store"] = c.store
            if c.backend is not None:
                out["memory_backend"] = c.backend
            if c.window is not None:
                out["memory_window"] = c.window
            if c.compaction_threshold is not None:
                out["compaction_threshold"] = c.compaction_threshold
            if c.use_llm_summarizer:
                out["use_llm_summarizer"] = True
        elif self.conversation is not None and not self.conversation.enabled:
            out["use_memory"] = False

        note_store = self.resolve_note_store()
        if self.user is not None:
            wants_store = bool(self.user.memory_tool or self.user.auto_extract)
            if wants_store and note_store is None:
                from loomable.agent.errors import AgentConfigError

                raise AgentConfigError(
                    "UserMemory(memory_tool=True) and/or auto_extract=True require "
                    "note_store= or embedder= (to build a NoteStore). "
                    "Pass note_store=..., embedder=..., or set memory_tool=False "
                    "and auto_extract=False."
                )
        if note_store is not None:
            out["note_store"] = note_store
            if self.user and self.user.memory_tool:
                out["memory_tool"] = True

        if self.knowledge is not None:
            k = self.knowledge
            if k.documents and k.embedder is None:
                from loomable.agent.errors import AgentConfigError

                raise AgentConfigError(
                    "KnowledgeMemory(documents=...) requires embedder= for "
                    "passive RAG indexing. Pass embedder=..., or use "
                    "store=/sources=/knowledge_base= for search_* tools."
                )
            if k.documents:
                out["knowledge"] = list(k.documents)
            if k.embedder is not None:
                out["embedder"] = k.embedder
            out["knowledge_top_k"] = k.top_k
            kb = k.knowledge_base
            if kb is None and (k.store is not None or k.sources):
                from loomable.retrieval.knowledge import KnowledgeBase

                kb = KnowledgeBase(
                    store=k.store,
                    sources=k.sources,
                    embedder=k.embedder,
                )
            if kb is not None:
                out["knowledge_base"] = kb

        return out


class ScopedNoteStore:
    """Wrap a NoteStore so note ids/tags are namespaced by a :class:`MemoryScope`."""

    def __init__(
        self,
        inner: Any,
        *,
        scope: MemoryScope,
    ) -> None:
        if scope is None or not scope:
            raise ValueError("ScopedNoteStore requires a non-empty MemoryScope")
        self._inner = inner
        self._scope = scope
        self._prefix = scope.prefix
        self._tags = scope.tags

    @property
    def scope(self) -> MemoryScope:
        return self._scope

    @property
    def user_id(self) -> str | None:
        return self._scope.get("user_id")

    def _scoped_id(self, note_id: str) -> str:
        if note_id.startswith(f"{self._prefix}:"):
            return note_id
        return f"{self._prefix}:{note_id}"

    async def write(self, note_id: str, text: str, tags: list[str] | tuple[str, ...] = ()) -> Any:
        tag_list = list(tags)
        for t in self._tags:
            if t not in tag_list:
                tag_list.append(t)
        return await self._inner.write(self._scoped_id(note_id), text, tag_list)

    async def read(self, note_id: str) -> Any:
        return await self._inner.read(self._scoped_id(note_id))

    async def list(self, tag: str | None = None) -> list[Any]:
        notes = await self._inner.list(tag=tag)
        return [n for n in notes if self._owns(n)]

    async def delete(self, note_id: str) -> None:
        await self._inner.delete(self._scoped_id(note_id))

    async def recall(self, query: str, k: int = 3) -> list[Any]:
        hits = await self._inner.recall(query, k=max(k * 4, k))
        owned = [n for n in hits if self._owns(n)]
        return owned[:k]

    def _owns(self, note: Any) -> bool:
        nid = getattr(note, "note_id", "") or ""
        tags = list(getattr(note, "tags", None) or [])
        if nid.startswith(f"{self._prefix}:"):
            return True
        # Require ALL scope tags so claim_id isolation is strict.
        return all(t in tags for t in self._tags)


_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmy name is\s+(.+?)(?:[.!]|$)", re.I),
    re.compile(r"\bi(?:'m| am)\s+(.+?)(?:[.!]|$)", re.I),
    re.compile(r"\bi prefer\s+(.+?)(?:[.!]|$)", re.I),
    re.compile(r"\bi like\s+(.+?)(?:[.!]|$)", re.I),
    re.compile(r"\bmy favorite\s+(\w+)\s+is\s+(.+?)(?:[.!]|$)", re.I),
    re.compile(r"\bcall me\s+(.+?)(?:[.!]|$)", re.I),
)


def extract_user_facts(text: str, *, limit: int = 5) -> list[str]:
    """Heuristic fact extraction for ``UserMemory(auto_extract=True)``."""
    if not text or not text.strip():
        return []
    facts: list[str] = []
    for pat in _FACT_PATTERNS:
        for m in pat.finditer(text):
            if pat.pattern.startswith(r"\bmy favorite"):
                fact = f"Favorite {m.group(1).strip()}: {m.group(2).strip()}"
            else:
                fact = m.group(0).strip().rstrip(".!")
            if fact and fact not in facts:
                facts.append(fact)
            if len(facts) >= limit:
                return facts
    return facts


async def auto_extract_into_notes(
    note_store: Any,
    user_text: str,
    *,
    user_id: str | None = None,
    scope: MemoryScope | None = None,
) -> list[str]:
    """Write heuristic facts into ``note_store``. Returns texts written."""
    facts = extract_user_facts(user_text)
    written: list[str] = []
    extra_tags = list(scope.tags) if scope else []
    if user_id and f"scope:user_id={user_id}" not in extra_tags:
        extra_tags.append(f"user:{user_id}")
    for fact in facts:
        digest = hashlib.sha1(fact.lower().encode("utf-8")).hexdigest()[:12]
        note_id = f"auto:{digest}"
        tags = ["auto", "user_fact", *extra_tags]
        await note_store.write(note_id, fact, tags)
        written.append(fact)
    return written
