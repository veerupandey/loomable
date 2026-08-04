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
from .delegation import make_delegation_tools
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
from loomable.media import Image as Image, Audio, Video, File
from .notes import Note, NoteStore, make_memory_tool
from .reasoning import make_plan_tool, make_think_tool
from .routing import ComplexityRouter, RunStrategy
from .run import RunChunk, RunResult
from .summarize import LLMSummarizer
from .team import Team
from .tools import FunctionTool, MCPTool, tool

__all__ = [
    "Agent",
    "BuiltAgent",
    "GatedDispatchResult",
    "ModelSpec",
    "Team",
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
    "NoteStore",
    "Note",
    "make_memory_tool",
    "make_delegation_tools",
    "make_think_tool",
    "make_plan_tool",
    "LLMSummarizer",
    "image",
    "video",
    "Image",
    "Audio",
    "Video",
    "File",
    "tool",
    "FunctionTool",
    "MCPTool",
]
