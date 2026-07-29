"""loomable.agent - High-level, agno-style Agent builder and runtime.

This package holds the ergonomic high-level API that composes existing kernel
primitives without modifying ``loomable.kernel``:

- ``Agent`` builder and ``BuiltAgent`` runtime wrapper
- ``ModelSpec`` declarative model configuration
- ``OrchestrationMode`` multi-agent orchestration modes
- ``AgentConfigError`` for invalid/missing builder configuration

It depends on ``loomable.kernel`` and ``loomable.content``.
"""

from .builder import (
    Agent,
    BuiltAgent,
    GatedDispatchResult,
    ModelSpec,
    OrchestrationMode,
    PostToolHook,
    ToolHook,
)
from .context import RunContext, StopReason
from .errors import (
    AgentConfigError,
    InputValidationError,
    StructuredOutputError,
    ToolHookRejection,
    UnsupportedModalityError,
)
from .events import AgentEvents, Event, JSONTracer, NoOpEvents
from .media import image, video
from .notes import Note, NoteStore, make_memory_tool
from .orchestration import Orchestrator
from .reasoning import make_plan_tool, make_think_tool
from .routing import ComplexityRouter, RunStrategy
from .run import RunChunk, RunResult
from .summarize import LLMSummarizer
from .tools import FunctionTool, MCPTool, tool

__all__ = [
    "Agent",
    "BuiltAgent",
    "GatedDispatchResult",
    "ModelSpec",
    "OrchestrationMode",
    "Orchestrator",
    "ToolHook",
    "PostToolHook",
    "AgentConfigError",
    "UnsupportedModalityError",
    "StructuredOutputError",
    "InputValidationError",
    "ToolHookRejection",
    "RunResult",
    "RunChunk",
    "RunContext",
    "StopReason",
    "AgentEvents",
    "Event",
    "NoOpEvents",
    "JSONTracer",
    "ComplexityRouter",
    "RunStrategy",
    "NoteStore",
    "Note",
    "make_memory_tool",
    "make_think_tool",
    "make_plan_tool",
    "LLMSummarizer",
    "image",
    "video",
    "tool",
    "FunctionTool",
    "MCPTool",
]
