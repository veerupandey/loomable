"""Shared Agent memory kwargs — Team / Case / Flow must accept the same set.

Conversation (L1/L2): ``session_id``, ``resume``, ``session_store`` / ``memory_backend``,
window / compaction knobs.

Long-term (L3): ``note_store``, ``memory_tool``, ``knowledge`` / ``embedder``.

Flow/Workflow ``memory=`` (TieredMemoryStore) and ``checkpointer`` are separate
concerns and are intentionally not in this set.
"""

from __future__ import annotations

from typing import Any

# Keys forwarded identically on Agent, Team (parent), and Case (role agents).
MEMORY_KEYS: tuple[str, ...] = (
    "memory",
    "session_id",
    "user_id",
    "scopes",
    "resume",
    "use_memory",
    "memory_window",
    "compaction_threshold",
    "use_llm_summarizer",
    "session_store",
    "memory_backend",
    "note_store",
    "memory_tool",
    "knowledge",
    "embedder",
    "knowledge_top_k",
    "retrievers",
)

# Attribute names on Agent for the corresponding kwargs.
_AGENT_ATTR: dict[str, str] = {
    "memory": "_memory_bundle",
    "session_id": "_session_id",
    "user_id": "_user_id",
    "scopes": "_scopes",
    "resume": "_resume",
    "use_memory": "_use_memory",
    "memory_window": "_memory_window",
    "compaction_threshold": "_compaction_threshold",
    "use_llm_summarizer": "_use_llm_summarizer",
    "session_store": "_session_store",
    "memory_backend": "_memory_backend",
    "note_store": "_note_store",
    "memory_tool": "_memory_tool",
    "knowledge": "_knowledge",
    "embedder": "_embedder",
    "knowledge_top_k": "_knowledge_top_k",
    "retrievers": "_retrievers",
}

__all__ = [
    "MEMORY_KEYS",
    "filter_memory_kwargs",
    "memory_kwargs_from_agent",
    "role_scoped_memory",
]


def filter_memory_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only Agent memory keys that were explicitly provided (not None)."""
    return {k: kwargs[k] for k in MEMORY_KEYS if k in kwargs and kwargs[k] is not None}


def memory_kwargs_from_agent(agent: Any) -> dict[str, Any]:
    """Extract memory configuration from an :class:`~loomable.agent.builder.Agent`."""
    out: dict[str, Any] = {}
    for key, attr in _AGENT_ATTR.items():
        if not hasattr(agent, attr):
            continue
        val = getattr(agent, attr)
        if val is None:
            continue
        # Skip False resume / memory_tool noise only when False is default-ish;
        # still forward explicit False for resume/use_memory/memory_tool.
        out[key] = val
    return out


def role_scoped_memory(
    memory: dict[str, Any],
    *,
    role: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Copy memory kwargs for a Case/Team role without colliding L1/L2 threads.

    Shared ``note_store`` / knowledge stay the same object. Conversation
    ``session_id`` becomes ``{base}:{role}`` so planner/worker/synth do not
    overwrite each other's chat turns on one store key.
    """
    out = dict(memory)
    base = session_id or out.get("session_id")
    if base and role:
        out["session_id"] = f"{base}:{role}"
        out.pop("resume", None)
    return out
