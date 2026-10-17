"""loomable — a lightweight Python framework for agents, specialist teams, and workflows.

High-level imports::

    from loomable import Agent, Team, Workflow, Case, Step, Loop, tool

Progressive disclosure:
  Agent → Team → Workflow (+ Step / Loop / branch / parallel)
  Case for goal + WorkItems board + dispatch + accept
  Flow / engines / edges for advanced graph control

Package tiers: facade (``loomable``) → product (``agent``, ``memory``, ``flow``,
``case``, …) → runtime (``kernel``, ``providers``, ``persist``). There is no
separate ``loomable.core`` — ``kernel`` is the runtime core."""
from __future__ import annotations

from importlib import import_module

__version__ = "0.2.0b0"

__all__ = ['__version__', 'Agent', 'BuiltAgent', 'Team', 'Workflow', 'Case', 'Board', 'WorkItem', 'Step', 'StepFailed', 'Command', 'Send', 'Route', 'Loop', 'Condition', 'Parallel_Group', 'tool', 'RunResult', 'ContextPolicy', 'spawn_specialist', 'create_deep_agent', 'resolve_skills', 'list_bundled_skills', 'Memory', 'MemoryScope', 'ConversationMemory', 'UserMemory', 'KnowledgeMemory', 'WorkingMemory', 'open_session_store', 'open_vector_store', 'FlowPaused', 'build_case_workflow', 'map_specialists', 'Verifier', 'VerdictResult', 'plan_and_execute', 'JsonFileCheckpointer', 'SQLiteCheckpointer', 'InMemoryCheckpointer', 'PostgresCheckpointer', 'Flow']

_EXPORTS = {
    'Agent': ('loomable.agent', 'Agent'),
    'BuiltAgent': ('loomable.agent', 'BuiltAgent'),
    'Team': ('loomable.agent', 'Team'),
    'Workflow': ('loomable.flow', 'Workflow'),
    'Case': ('loomable.case', 'Case'),
    'Board': ('loomable.case', 'Board'),
    'WorkItem': ('loomable.case', 'WorkItem'),
    'Step': ('loomable.flow', 'Step'),
    'StepFailed': ('loomable.flow', 'StepFailed'),
    'Command': ('loomable.flow', 'Command'),
    'Send': ('loomable.flow', 'Send'),
    'Route': ('loomable.flow', 'Route'),
    'Loop': ('loomable.flow', 'Loop'),
    'Condition': ('loomable.flow', 'Condition'),
    'Parallel_Group': ('loomable.flow', 'Parallel_Group'),
    'tool': ('loomable.agent', 'tool'),
    'RunResult': ('loomable.agent', 'RunResult'),
    'ContextPolicy': ('loomable.agent', 'ContextPolicy'),
    'spawn_specialist': ('loomable.agent', 'spawn_specialist'),
    'create_deep_agent': ('loomable.agent', 'create_deep_agent'),
    'resolve_skills': ('loomable.skills', 'resolve_skills'),
    'list_bundled_skills': ('loomable.skills', 'list_bundled_skills'),
    'Memory': ('loomable.memory', 'Memory'),
    'MemoryScope': ('loomable.memory', 'MemoryScope'),
    'ConversationMemory': ('loomable.memory', 'ConversationMemory'),
    'UserMemory': ('loomable.memory', 'UserMemory'),
    'KnowledgeMemory': ('loomable.memory', 'KnowledgeMemory'),
    'WorkingMemory': ('loomable.memory', 'WorkingMemory'),
    'open_session_store': ('loomable.memory', 'open_session_store'),
    'open_vector_store': ('loomable.providers.vector_store', 'open_vector_store'),
    'FlowPaused': ('loomable.flow', 'FlowPaused'),
    'build_case_workflow': ('loomable.case', 'build_case_workflow'),
    'map_specialists': ('loomable.case', 'map_specialists'),
    'Verifier': ('loomable.flow', 'Verifier'),
    'VerdictResult': ('loomable.flow', 'VerdictResult'),
    'plan_and_execute': ('loomable.flow', 'plan_and_execute'),
    'JsonFileCheckpointer': ('loomable.persist', 'JsonFileCheckpointer'),
    'SQLiteCheckpointer': ('loomable.persist', 'SQLiteCheckpointer'),
    'InMemoryCheckpointer': ('loomable.persist', 'InMemoryCheckpointer'),
    'PostgresCheckpointer': ('loomable.persist', 'PostgresCheckpointer'),
    'Flow': ('loomable.flow', 'Flow'),
}

def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
