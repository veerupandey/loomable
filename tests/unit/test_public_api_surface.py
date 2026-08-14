"""Public surface freeze for beta — top-level __all__ must stay intentional."""

from __future__ import annotations

import loomable


# Documented stable + deprecated-alias exports (docs/STABILITY.md).
_EXPECTED = {
    "__version__",
    "Agent",
    "BuiltAgent",
    "Team",
    "Workflow",
    "Case",
    "Board",
    "WorkItem",
    "Step",
    "Loop",
    "Condition",
    "Parallel_Group",
    "tool",
    "RunResult",
    "ContextPolicy",
    "spawn_specialist",
    "create_deep_agent",
    "create_research_agent",
    "resolve_skills",
    "list_bundled_skills",
    "Memory",
    "MemoryScope",
    "ConversationMemory",
    "UserMemory",
    "KnowledgeMemory",
    "WorkingMemory",
    "open_session_store",
    "FlowPaused",
    "build_case_workflow",
    "map_specialists",
    "Verifier",
    "VerdictResult",
    "sequential",
    "parallel",
    "route",
    "coordinate",
    "plan_and_execute",
    "JsonFileCheckpointer",
    "SQLiteCheckpointer",
    "InMemoryCheckpointer",
    "PostgresCheckpointer",
    "Flow",
}


def test_version_matches_beta() -> None:
    assert loomable.__version__ == "0.2.0b0"
    assert "__version__" in loomable.__all__


def test_public_all_matches_stability_surface() -> None:
    actual = set(loomable.__all__)
    missing = _EXPECTED - actual
    extra = actual - _EXPECTED
    assert not missing, f"missing from __all__: {sorted(missing)}"
    assert not extra, f"unexpected __all__ entries (update STABILITY.md): {sorted(extra)}"


def test_stable_symbols_importable() -> None:
    for name in _EXPECTED:
        if name == "__version__":
            continue
        assert hasattr(loomable, name), name
