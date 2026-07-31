"""loomable.agent - High-level, agno-style Agent builder and runtime.

This package holds the ergonomic high-level API that composes existing kernel
primitives without modifying ``loomable.kernel``:

- ``Agent`` builder and ``BuiltAgent`` runtime wrapper
- ``ModelSpec`` declarative model configuration
- ``AgentConfigError`` for invalid/missing builder configuration

Multi-agent orchestration is now handled via ``loomable.flow.Flow`` (Req 14.4).
"""

from .builder import (
    Agent,
    BuiltAgent,
    GatedDispatchResult,
    ModelSpec,
    PostToolHook,
    ToolHook,
)
from .channels import Channel, ChannelMessage, InMemoryChannel
from .context import RunContext, StopReason
from .errors import (
    AgentConfigError,
    HITLPause,
    InputValidationError,
    StructuredOutputError,
    ToolHookRejection,
    UnsupportedModalityError,
)
from .events import AgentEvents, Event, JSONTracer, NoOpEvents
from .media import image, video
from .notes import Note, NoteStore, make_memory_tool
from .reasoning import make_plan_tool, make_think_tool
from .routing import AlwaysPlan, ComplexityRouter, RunStrategy, always_plan
from .run import RunChunk, RunResult
from .summarize import LLMSummarizer
from .tools import FunctionTool, MCPTool, tool

__all__ = [
    "Agent",
    "BuiltAgent",
    "GatedDispatchResult",
    "ModelSpec",
    "ToolHook",
    "PostToolHook",
    "AgentConfigError",
    "UnsupportedModalityError",
    "StructuredOutputError",
    "InputValidationError",
    "ToolHookRejection",
    "HITLPause",
    "Channel",
    "ChannelMessage",
    "InMemoryChannel",
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
    "AlwaysPlan",
    "always_plan",
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
