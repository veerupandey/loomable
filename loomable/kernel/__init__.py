"""loomable.kernel - The stable, generic core of the agent framework."""

from loomable.kernel.agent_loop import AgentLoop
from loomable.kernel.context import AdmissionResult, ContextManager
from loomable.kernel.contracts import (
    MemoryBackend,
    ModelProvider,
    Retriever,
    Skill,
    Tool,
    VectorBackend,
)
from loomable.kernel.errors import (
    APIToolError,
    APIToolTimeoutError,
    GuardrailViolation,
    LoomableError,
    MCPConnectionError,
    MCPToolError,
    MemoryBackendError,
    ModelProviderError,
    PlanningModelError,
    ScriptToolError,
    SessionNotFoundError,
    SkillLoadError,
    SubagentError,
    UnsupportedExtensionError,
)
from loomable.kernel.long_term import LongTermStore, ZvecVectorBackend
from loomable.kernel.memory import MemoryManager
from loomable.kernel.summarizer import Summarizer
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.model_router import ModelRouter, TierSubstitution
from loomable.kernel.registry import (
    ExtensionHandle,
    ExtensionRegistry,
    ExtensionSpec,
    KernelData,
)
from loomable.kernel.retrievers import RetrieverTool
from loomable.kernel.tool_runtime import ToolRuntime
from loomable.kernel.models import (
    SQLITE_DEFAULT,
    SUPPORTED_MECHANISMS,
    ZVEC_DEFAULT,
    AgentConfig,
    ContextItem,
    ContextWindow,
    ExtensionMechanism,
    LoopPhase,
    LoopState,
    ModelRequest,
    ModelResponse,
    OnboardingRequest,
    Session,
    StreamEvent,
    StructuredSummary,
    ToolCall,
    ToolError,
    ToolOutcome,
    ToolResult,
    Turn,
)

__all__ = [
    # Agent Loop
    "AgentLoop",
    # Context
    "AdmissionResult",
    "ContextManager",
    # Contracts
    "MemoryBackend",
    "ModelProvider",
    "Retriever",
    "Skill",
    "Tool",
    "VectorBackend",
    # Long-Term Store
    "LongTermStore",
    "ZvecVectorBackend",
    # Memory Manager
    "MemoryManager",
    # Summarizer
    "Summarizer",
    # Model Interface
    "ModelInterface",
    # Model Router
    "ModelRouter",
    "TierSubstitution",
    # Registry
    "ExtensionHandle",
    "ExtensionRegistry",
    "ExtensionSpec",
    "KernelData",
    # Retrievers
    "RetrieverTool",
    # Tool Runtime
    "ToolRuntime",
    # Models - Configuration
    "AgentConfig",
    "SQLITE_DEFAULT",
    "ZVEC_DEFAULT",
    # Models - Extension
    "ExtensionMechanism",
    "SUPPORTED_MECHANISMS",
    "OnboardingRequest",
    # Models - Tools
    "ModelRequest",
    "ModelResponse",
    "StreamEvent",
    "ToolCall",
    "ToolError",
    "ToolOutcome",
    "ToolResult",
    # Models - Memory / Context
    "ContextItem",
    "ContextWindow",
    "LoopPhase",
    "LoopState",
    "Session",
    "StructuredSummary",
    "Turn",
    # Errors
    "APIToolError",
    "APIToolTimeoutError",
    "GuardrailViolation",
    "LoomableError",
    "MCPConnectionError",
    "MCPToolError",
    "MemoryBackendError",
    "ModelProviderError",
    "PlanningModelError",
    "ScriptToolError",
    "SessionNotFoundError",
    "SkillLoadError",
    "SubagentError",
    "UnsupportedExtensionError",
]
