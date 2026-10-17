"""loomable.agent - High-level Agent builder and runtime.

This package holds the ergonomic high-level API that composes existing kernel
primitives without modifying ``loomable.kernel``:

- ``Agent`` builder and ``BuiltAgent`` runtime wrapper
- ``ModelSpec`` declarative model configuration
- ``AgentConfigError`` for invalid/missing builder configuration

L3 ``NoteStore`` lives in :mod:`loomable.memory` (re-exported here for compatibility).
Multi-agent orchestration is handled via ``Team`` / ``Workflow`` (and low-level ``Flow``)."""
from __future__ import annotations

from importlib import import_module

__all__ = ['Agent', 'BuiltAgent', 'GatedDispatchResult', 'ModelSpec', 'Team', 'ToolHook', 'PostToolHook', 'AgentConfigError', 'UnsupportedModalityError', 'StructuredOutputError', 'InputValidationError', 'ToolHookRejection', 'RequireToolsError', 'RunResult', 'RunChunk', 'RunContext', 'StopReason', 'ContextPolicy', 'CompactionResult', 'AgentEvents', 'Event', 'NoOpEvents', 'JSONTracer', 'ComplexityRouter', 'RunStrategy', 'NoteStore', 'Note', 'make_memory_tool', 'make_delegation_tools', 'spawn_specialist', 'make_think_tool', 'make_plan_tool', 'create_deep_agent', 'make_task_tool', 'make_task_tools', 'make_research_accept', 'make_compact_conversation_tool', 'SpecialistSpec', 'LLMSummarizer', 'image', 'video', 'Image', 'Audio', 'Video', 'File', 'tool', 'FunctionTool', 'MCPTool']

_EXPORTS = {
    'ExecutionOptions': ('.options', 'ExecutionOptions'),
    'DurabilityOptions': ('.options', 'DurabilityOptions'),
    'AgentPaused': ('.durable', 'AgentPaused'),
    'UnknownToolOutcome': ('.durable', 'UnknownToolOutcome'),
    'Agent': ('.builder', 'Agent'),
    'BuiltAgent': ('.builder', 'BuiltAgent'),
    'GatedDispatchResult': ('.builder', 'GatedDispatchResult'),
    'ModelSpec': ('.builder', 'ModelSpec'),
    'Team': ('.team', 'Team'),
    'ToolHook': ('.builder', 'ToolHook'),
    'PostToolHook': ('.builder', 'PostToolHook'),
    'AgentConfigError': ('.errors', 'AgentConfigError'),
    'UnsupportedModalityError': ('.errors', 'UnsupportedModalityError'),
    'StructuredOutputError': ('.errors', 'StructuredOutputError'),
    'InputValidationError': ('.errors', 'InputValidationError'),
    'ToolHookRejection': ('.errors', 'ToolHookRejection'),
    'RequireToolsError': ('.errors', 'RequireToolsError'),
    'RunResult': ('.run', 'RunResult'),
    'RunChunk': ('.run', 'RunChunk'),
    'RunContext': ('.context', 'RunContext'),
    'StopReason': ('.context', 'StopReason'),
    'ContextPolicy': ('.context_policy', 'ContextPolicy'),
    'CompactionResult': ('.context_policy', 'CompactionResult'),
    'AgentEvents': ('.events', 'AgentEvents'),
    'Event': ('.events', 'Event'),
    'NoOpEvents': ('.events', 'NoOpEvents'),
    'JSONTracer': ('.events', 'JSONTracer'),
    'ComplexityRouter': ('.routing', 'ComplexityRouter'),
    'RunStrategy': ('.routing', 'RunStrategy'),
    'NoteStore': ('.notes', 'NoteStore'),
    'Note': ('.notes', 'Note'),
    'make_memory_tool': ('.notes', 'make_memory_tool'),
    'make_delegation_tools': ('.delegation', 'make_delegation_tools'),
    'spawn_specialist': ('.delegation', 'spawn_specialist'),
    'make_think_tool': ('.reasoning', 'make_think_tool'),
    'make_plan_tool': ('.reasoning', 'make_plan_tool'),
    'create_deep_agent': ('.deep', 'create_deep_agent'),
    'make_task_tool': ('.deep', 'make_task_tool'),
    'make_task_tools': ('.deep', 'make_task_tools'),
    'make_research_accept': ('.deep', 'make_research_accept'),
    'make_compact_conversation_tool': ('.deep', 'make_compact_conversation_tool'),
    'SpecialistSpec': ('.deep', 'SpecialistSpec'),
    'LLMSummarizer': ('.summarize', 'LLMSummarizer'),
    'image': ('.media', 'image'),
    'video': ('.media', 'video'),
    'Image': ('loomable.media', 'Image'),
    'Audio': ('loomable.media', 'Audio'),
    'Video': ('loomable.media', 'Video'),
    'File': ('loomable.media', 'File'),
    'tool': ('.tools', 'tool'),
    'FunctionTool': ('.tools', 'FunctionTool'),
    'MCPTool': ('.tools', 'MCPTool'),
}

__all__ += ["ExecutionOptions", "DurabilityOptions", "AgentPaused", "UnknownToolOutcome"]

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
