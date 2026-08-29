"""Public surface freeze for beta — top-level __all__ must stay intentional."""

from __future__ import annotations

import loomable


# Documented public exports (loomable.__all__).
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
    "StepFailed",
    "Loop",
    "Condition",
    "Parallel_Group",
    "tool",
    "RunResult",
    "ContextPolicy",
    "spawn_specialist",
    "create_deep_agent",
    "resolve_skills",
    "list_bundled_skills",
    "Memory",
    "MemoryScope",
    "ConversationMemory",
    "UserMemory",
    "KnowledgeMemory",
    "WorkingMemory",
    "open_session_store",
    "open_vector_store",
    "FlowPaused",
    "build_case_workflow",
    "map_specialists",
    "Verifier",
    "VerdictResult",
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
    assert not extra, f"unexpected __all__ entries (update this freeze): {sorted(extra)}"


def test_stable_symbols_importable() -> None:
    for name in _EXPECTED:
        if name == "__version__":
            continue
        assert hasattr(loomable, name), name


# Names removed in 0.2.0b0 (CHANGELOG / docs/API.md). No compatibility shims.
_REMOVED = {
    "sequential",
    "parallel",
    "route",
    "coordinate",
    "HITLPause",
    "Pipeline",
    "Orchestrator",
    "AutoPlan",
    "create_personalized_agent",
    "Map",
    "Router",
}


def test_removed_symbols_absent_from_top_level() -> None:
    for name in _REMOVED:
        assert name not in loomable.__all__, name
        assert not hasattr(loomable, name), name


def test_hitlpause_not_public_on_agent() -> None:
    import loomable.agent as agent_mod
    import loomable.agent.errors as errors_mod

    assert "HITLPause" not in agent_mod.__all__
    assert not hasattr(agent_mod, "HITLPause")
    assert not hasattr(errors_mod, "HITLPause")
