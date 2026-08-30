"""loomable.agent - High-level Agent builder and runtime.

This package holds the ergonomic high-level API that composes existing kernel
primitives without modifying ``loomable.kernel``:

- ``Agent`` builder and ``BuiltAgent`` runtime wrapper
- ``ModelSpec`` declarative model configuration
- ``AgentConfigError`` for invalid/missing builder configuration

L3 ``NoteStore`` lives in :mod:`loomable.memory` (re-exported here for compatibility).
Multi-agent orchestration is handled via ``Team`` / ``Workflow`` (and low-level ``Flow``).
"""

from .builder import (
    Agent,
    BuiltAgent,
    GatedDispatchResult,
    ModelSpec,
    PostToolHook,
    ToolHook,
)
from .context import RunContext, StopReason
from .context_policy import CompactionResult, ContextPolicy
from .delegation import make_delegation_tools, spawn_specialist
from .errors import (
    AgentConfigError,
    InputValidationError,
    RequireToolsError,
    StructuredOutputError,
    ToolHookRejection,
    UnsupportedModalityError,
)
from .events import AgentEvents, Event, JSONTracer, NoOpEvents
from .media import image, video
from loomable.media import Image as Image, Audio, Video, File
from .notes import Note, NoteStore, make_memory_tool  # compat re-export; prefer loomable.memory
from .reasoning import make_plan_tool, make_think_tool
from .routing import ComplexityRouter, RunStrategy
from .run import RunChunk, RunResult
from .summarize import LLMSummarizer
from .team import Team
from .tools import FunctionTool, MCPTool, tool
from .deep import (
    SpecialistSpec,
    create_deep_agent,
    make_compact_conversation_tool,
    make_research_accept,
    make_task_tool,
    make_task_tools,
)

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
    "RequireToolsError",
    "RunResult",
    "RunChunk",
    "RunContext",
    "StopReason",
    "ContextPolicy",
    "CompactionResult",
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
    "spawn_specialist",
    "make_think_tool",
    "make_plan_tool",
    "create_deep_agent",
    "make_task_tool",
    "make_task_tools",
    "make_research_accept",
    "make_compact_conversation_tool",
    "SpecialistSpec",
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
