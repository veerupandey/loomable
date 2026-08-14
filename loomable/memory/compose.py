"""Composable Agent memory — assemble layers, pass one object to ``Agent(memory=...)``.

Layers (any subset)::

    from loomable.memory import Memory, ConversationMemory, UserMemory, KnowledgeMemory
    from loomable.memory import open_session_store
    from loomable.agent import NoteStore
    from loomable.kernel.long_term import LongTermStore

    memory = Memory.compose(
        conversation=ConversationMemory(
            store=open_session_store("postgres", url=DSN, user_id="alice"),
            window=8,
        ),
        user=UserMemory(
            note_store=NoteStore(LongTermStore(), embedder),
            memory_tool=True,       # agentic write/recall
            auto_extract=False,     # set True for Always-mode (heuristic)
        ),
        knowledge=KnowledgeMemory(documents=[...], embedder=embedder),
    )

    agent = Agent(model=..., memory=memory, session_id="c1", user_id="alice")

Aliases: ``short=`` → conversation, ``long=`` → user.
Legacy ``session_store=`` / ``note_store=`` / ``memory_backend=`` still work and
override the matching layer when both are set.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "ConversationMemory",
    "UserMemory",
    "KnowledgeMemory",
    "WorkingMemory",
    "Memory",
    "ScopedNoteStore",
    "is_memory_bundle",
    "is_kernel_memory_manager",
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
    """

    store: Any | None = None
    backend: Any | None = None
    window: int | None = None
    compaction_threshold: int | None = None
    use_llm_summarizer: bool = False
    enabled: bool = True


@dataclass
class UserMemory:
    """Long-term user-scoped facts (L3 notes).

    Parameters
    ----------
    note_store:
        Existing :class:`~loomable.agent.notes.NoteStore`.
    long_term / embedder:
        Build a NoteStore when ``note_store`` is omitted.
    memory_tool:
        Expose agentic ``memory`` tool (write/read/list/delete/recall).
    auto_extract:
        After each turn, heuristically extract user facts into notes
        (Agno Always-mode lite). Disabled when ``memory_tool`` path alone is enough.
        If both are True, agentic tool still wins for model writes; auto_extract
        only adds passive facts from user text (never both nested LLM calls).
    user_id:
        Scope note ids/tags. Falls back to ``Agent(user_id=...)``.
    """

    note_store: Any | None = None
    long_term: Any | None = None
    embedder: Any | None = None
    memory_tool: bool = True
    auto_extract: bool = False
    user_id: str | None = None


@dataclass
class KnowledgeMemory:
    """RAG documents injected into the prompt (separate from user notes)."""

    documents: list[str] = field(default_factory=list)
    embedder: Any = None
    top_k: int = 3


@dataclass
class WorkingMemory:
    """Flow/Workflow blackboard (:class:`~loomable.flow.memory.TieredMemoryStore`).

    Not applied to Agent chat by default — expose via ``.store`` for
    ``Workflow(memory=bundle.working.store)`` or ``Flow(memory=...)``.
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
        short: ConversationMemory | None = None,
        long: UserMemory | None = None,
    ) -> Memory:
        """Assemble layers. ``short``/``long`` are aliases for conversation/user."""
        return cls(
            conversation=conversation or short,
            user=user or long,
            knowledge=knowledge,
            working=working,
        )

    def with_user_id(self, user_id: str | None) -> Memory:
        """Return a copy that stamps ``user_id`` onto the user layer when missing."""
        if not user_id or self.user is None:
            return self
        if self.user.user_id:
            return self
        return replace(self, user=replace(self.user, user_id=user_id))

    def resolve_note_store(self) -> Any | None:
        """Materialize a (possibly user-scoped) NoteStore from the user layer."""
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
        uid = self.user.user_id
        if uid:
            return ScopedNoteStore(store, user_id=uid)
        return store

    def to_agent_kwargs(self) -> dict[str, Any]:
        """Flatten into legacy Agent constructor kwargs (no Nones)."""
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
        if note_store is not None:
            out["note_store"] = note_store
            if self.user and self.user.memory_tool:
                out["memory_tool"] = True

        if self.knowledge is not None and self.knowledge.documents:
            out["knowledge"] = list(self.knowledge.documents)
            if self.knowledge.embedder is not None:
                out["embedder"] = self.knowledge.embedder
            out["knowledge_top_k"] = self.knowledge.top_k

        return out


class ScopedNoteStore:
    """Wrap a NoteStore so note ids/tags are namespaced by ``user_id``."""

    def __init__(self, inner: Any, *, user_id: str) -> None:
        if not user_id:
            raise ValueError("ScopedNoteStore requires a non-empty user_id")
        self._inner = inner
        self._user_id = user_id
        self._tag = f"user:{user_id}"

    @property
    def user_id(self) -> str:
        return self._user_id

    def _scoped_id(self, note_id: str) -> str:
        if note_id.startswith(f"{self._user_id}:"):
            return note_id
        return f"{self._user_id}:{note_id}"

    async def write(self, note_id: str, text: str, tags: list[str] | tuple[str, ...] = ()) -> Any:
        tag_list = list(tags)
        if self._tag not in tag_list:
            tag_list.append(self._tag)
        return await self._inner.write(self._scoped_id(note_id), text, tag_list)

    async def read(self, note_id: str) -> Any:
        return await self._inner.read(self._scoped_id(note_id))

    async def list(self, tag: str | None = None) -> list[Any]:
        notes = await self._inner.list(tag=tag)
        return [n for n in notes if self._owns(n)]

    async def delete(self, note_id: str) -> None:
        await self._inner.delete(self._scoped_id(note_id))

    async def recall(self, query: str, k: int = 3) -> list[Any]:
        # Over-fetch then filter so vector neighbors from other users are dropped.
        hits = await self._inner.recall(query, k=max(k * 4, k))
        owned = [n for n in hits if self._owns(n)]
        return owned[:k]

    def _owns(self, note: Any) -> bool:
        nid = getattr(note, "note_id", "") or ""
        tags = list(getattr(note, "tags", None) or [])
        return nid.startswith(f"{self._user_id}:") or self._tag in tags


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
) -> list[str]:
    """Write heuristic facts into ``note_store``. Returns texts written."""
    facts = extract_user_facts(user_text)
    written: list[str] = []
    for fact in facts:
        digest = hashlib.sha1(fact.lower().encode("utf-8")).hexdigest()[:12]
        note_id = f"auto:{digest}"
        tags = ["auto", "user_fact"]
        if user_id:
            tags.append(f"user:{user_id}")
        await note_store.write(note_id, fact, tags)
        written.append(fact)
    return written
