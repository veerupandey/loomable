"""loomable.kernel - The stable, generic core of the agent framework."""
from __future__ import annotations

from importlib import import_module

__all__ = ['AgentLoop', 'AdmissionResult', 'ContextManager', 'MemoryBackend', 'ModelProvider', 'Retriever', 'Skill', 'Tool', 'VectorBackend', 'LongTermStore', 'ZvecVectorBackend', 'InMemoryVectorBackend', 'DEFAULT_ZVEC_PATH', 'MemoryManager', 'Summarizer', 'ModelInterface', 'ModelRouter', 'TierSubstitution', 'RetrieverTool', 'ToolRuntime', 'AgentConfig', 'SQLITE_DEFAULT', 'ZVEC_DEFAULT', 'ExtensionMechanism', 'SUPPORTED_MECHANISMS', 'OnboardingRequest', 'ModelRequest', 'ModelResponse', 'StreamEvent', 'ToolCall', 'ToolError', 'ToolOutcome', 'ToolResult', 'ContextItem', 'ContextWindow', 'LoopPhase', 'LoopState', 'Session', 'StructuredSummary', 'Turn', 'APIToolError', 'APIToolTimeoutError', 'GuardrailViolation', 'LoomableError', 'MCPConnectionError', 'MCPToolError', 'MemoryBackendError', 'ModelProviderError', 'PlanningModelError', 'ScriptToolError', 'SessionNotFoundError', 'SkillLoadError', 'SubagentError', 'UnsupportedExtensionError']

_EXPORTS = {
    'AgentLoop': ('loomable.kernel.agent_loop', 'AgentLoop'),
    'AdmissionResult': ('loomable.kernel.context', 'AdmissionResult'),
    'ContextManager': ('loomable.kernel.context', 'ContextManager'),
    'MemoryBackend': ('loomable.kernel.contracts', 'MemoryBackend'),
    'ModelProvider': ('loomable.kernel.contracts', 'ModelProvider'),
    'Retriever': ('loomable.kernel.contracts', 'Retriever'),
    'Skill': ('loomable.kernel.contracts', 'Skill'),
    'Tool': ('loomable.kernel.contracts', 'Tool'),
    'VectorBackend': ('loomable.kernel.contracts', 'VectorBackend'),
    'LongTermStore': ('loomable.kernel.long_term', 'LongTermStore'),
    'ZvecVectorBackend': ('loomable.kernel.long_term', 'ZvecVectorBackend'),
    'InMemoryVectorBackend': ('loomable.kernel.long_term', 'InMemoryVectorBackend'),
    'DEFAULT_ZVEC_PATH': ('loomable.kernel.long_term', 'DEFAULT_ZVEC_PATH'),
    'MemoryManager': ('loomable.kernel.memory', 'MemoryManager'),
    'Summarizer': ('loomable.kernel.summarizer', 'Summarizer'),
    'ModelInterface': ('loomable.kernel.model_interface', 'ModelInterface'),
    'ModelRouter': ('loomable.kernel.model_router', 'ModelRouter'),
    'TierSubstitution': ('loomable.kernel.model_router', 'TierSubstitution'),
    'RetrieverTool': ('loomable.kernel.retrievers', 'RetrieverTool'),
    'ToolRuntime': ('loomable.kernel.tool_runtime', 'ToolRuntime'),
    'AgentConfig': ('loomable.kernel.models', 'AgentConfig'),
    'SQLITE_DEFAULT': ('loomable.kernel.models', 'SQLITE_DEFAULT'),
    'ZVEC_DEFAULT': ('loomable.kernel.models', 'ZVEC_DEFAULT'),
    'ExtensionMechanism': ('loomable.kernel.models', 'ExtensionMechanism'),
    'SUPPORTED_MECHANISMS': ('loomable.kernel.models', 'SUPPORTED_MECHANISMS'),
    'OnboardingRequest': ('loomable.kernel.models', 'OnboardingRequest'),
    'ModelRequest': ('loomable.kernel.models', 'ModelRequest'),
    'ModelResponse': ('loomable.kernel.models', 'ModelResponse'),
    'StreamEvent': ('loomable.kernel.models', 'StreamEvent'),
    'ToolCall': ('loomable.kernel.models', 'ToolCall'),
    'ToolError': ('loomable.kernel.models', 'ToolError'),
    'ToolOutcome': ('loomable.kernel.models', 'ToolOutcome'),
    'ToolResult': ('loomable.kernel.models', 'ToolResult'),
    'ContextItem': ('loomable.kernel.models', 'ContextItem'),
    'ContextWindow': ('loomable.kernel.models', 'ContextWindow'),
    'LoopPhase': ('loomable.kernel.models', 'LoopPhase'),
    'LoopState': ('loomable.kernel.models', 'LoopState'),
    'Session': ('loomable.kernel.models', 'Session'),
    'StructuredSummary': ('loomable.kernel.models', 'StructuredSummary'),
    'Turn': ('loomable.kernel.models', 'Turn'),
    'APIToolError': ('loomable.kernel.errors', 'APIToolError'),
    'APIToolTimeoutError': ('loomable.kernel.errors', 'APIToolTimeoutError'),
    'GuardrailViolation': ('loomable.kernel.errors', 'GuardrailViolation'),
    'LoomableError': ('loomable.kernel.errors', 'LoomableError'),
    'MCPConnectionError': ('loomable.kernel.errors', 'MCPConnectionError'),
    'MCPToolError': ('loomable.kernel.errors', 'MCPToolError'),
    'MemoryBackendError': ('loomable.kernel.errors', 'MemoryBackendError'),
    'ModelProviderError': ('loomable.kernel.errors', 'ModelProviderError'),
    'PlanningModelError': ('loomable.kernel.errors', 'PlanningModelError'),
    'ScriptToolError': ('loomable.kernel.errors', 'ScriptToolError'),
    'SessionNotFoundError': ('loomable.kernel.errors', 'SessionNotFoundError'),
    'SkillLoadError': ('loomable.kernel.errors', 'SkillLoadError'),
    'SubagentError': ('loomable.kernel.errors', 'SubagentError'),
    'UnsupportedExtensionError': ('loomable.kernel.errors', 'UnsupportedExtensionError'),
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
