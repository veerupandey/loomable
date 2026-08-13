"""Core data models for the loomable agent framework.

This module defines the foundational data structures used throughout the Kernel:
configuration, tool calls/outcomes, memory/context items, session/loop state,
and the extension onboarding model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Type aliases for specs (structured dicts, refined later as needed)
# ---------------------------------------------------------------------------

ModelProviderSpec = dict[str, Any]
"""Configuration for a model provider (provider id, endpoint, credentials, etc.)."""

TierPolicy = dict[str, Any]
"""Policy definition for tiered model routing (cost/latency thresholds, rules)."""

BackendSpec = dict[str, Any]
"""Configuration for a pluggable storage backend."""

SkillSpec = dict[str, Any]
"""Configuration for a Skill (path, enabled flag, etc.)."""

MCPServerSpec = dict[str, Any]
"""Configuration for an MCP server (url, transport, auth, etc.)."""

APIToolSpec = dict[str, Any]
"""Configuration for an API tool (method, url, headers, timeout, etc.)."""

GuardrailRule = dict[str, Any]
"""A single guardrail rule definition (pattern, action, etc.)."""

GateSpec = dict[str, Any]
"""Verification gate specification for a given step."""

# Default backend specs
SQLITE_DEFAULT: BackendSpec = {"type": "sqlite", "path": ":memory:"}
ZVEC_DEFAULT: BackendSpec = {"type": "zvec"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    """Immutable agent configuration.

    Frozen because configuration is shared by reference across agent instances
    and must not be mutated after construction.
    """

    model: ModelProviderSpec
    planning_model: ModelProviderSpec | None
    tiers: dict[str, ModelProviderSpec]
    tier_policy: TierPolicy | None
    fallback_tiers: dict[str, str]
    token_budget: int
    checkpoint_interval: int
    short_term: BackendSpec = field(default_factory=lambda: dict(SQLITE_DEFAULT))
    long_term: BackendSpec = field(default_factory=lambda: dict(ZVEC_DEFAULT))
    skills: list[SkillSpec] = field(default_factory=list)
    mcp_servers: list[MCPServerSpec] = field(default_factory=list)
    api_tools: list[APIToolSpec] = field(default_factory=list)
    guardrails: list[GuardrailRule] = field(default_factory=list)
    verification_gates: dict[int, GateSpec] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Onboarding / Extension Model
# ---------------------------------------------------------------------------


class ExtensionMechanism(Enum):
    """Supported and rejected extension mechanisms."""

    SKILL = "skill"
    MCP_SERVER = "mcp_server"
    API_TOOL = "api_tool"
    KERNEL_MODIFICATION = "kernel_modification"  # always rejected


#: The set of mechanisms the framework accepts for onboarding.
SUPPORTED_MECHANISMS: frozenset[ExtensionMechanism] = frozenset(
    {ExtensionMechanism.SKILL, ExtensionMechanism.MCP_SERVER, ExtensionMechanism.API_TOOL}
)


@dataclass
class OnboardingRequest:
    """A request to onboard a new capability through an extension mechanism."""

    capability: str
    mechanism: ExtensionMechanism


# ---------------------------------------------------------------------------
# Tool Models
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A request to invoke a tool."""

    id: str
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    # Provider-specific extras (e.g. Gemini ``extra_content`` / thought signatures).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The successful result of a tool invocation."""

    content: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """True if this result represents an error."""
        return self.error is not None


@dataclass
class ToolError:
    """An error produced by a tool invocation."""

    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutcome:
    """The outcome of a single tool call - carries exactly one of result or error.

    Invariant: exactly one of ``result`` and ``error`` must be set (not None).
    This is enforced via __post_init__ validation.
    """

    call_id: str
    result: ToolResult | None = None
    error: ToolError | None = None

    def __post_init__(self) -> None:
        has_result = self.result is not None
        has_error = self.error is not None
        if has_result == has_error:
            raise ValueError(
                "ToolOutcome must carry exactly one of 'result' or 'error', "
                f"got result={has_result}, error={has_error}"
            )


# ---------------------------------------------------------------------------
# Memory / Context Models
# ---------------------------------------------------------------------------

#: Valid kinds for context items.
ContextItemKind = Literal["system", "schema", "turn", "summary", "recall"]

#: Kinds that are always pinned.
_PINNED_KINDS: frozenset[str] = frozenset({"system", "schema"})


@dataclass
class Turn:
    """A single conversational turn stored in L1 memory."""

    role: str
    content: str
    tokens: int
    step: int


@dataclass
class StructuredSummary:
    """A compressed summary produced by checkpoint summarization (L2)."""

    covers_steps: range
    objectives: list[str]
    decisions: list[str]
    text: str
    tokens: int


@dataclass
class ContextItem:
    """An item in the context window with priority and pinning metadata.

    Invariant: items with kind "system" or "schema" are always pinned.
    This is enforced via __post_init__.
    """

    kind: ContextItemKind
    tokens: int
    priority: int
    pinned: bool = False

    def __post_init__(self) -> None:
        # Enforce: system and schema items must be pinned
        if self.kind in _PINNED_KINDS and not self.pinned:
            self.pinned = True


@dataclass
class ContextWindow:
    """The assembled context window.

    Convention: items[0..n] = system items, then schema items, then the rest
    ordered by priority/recency.
    """

    items: list[ContextItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session / Loop State
# ---------------------------------------------------------------------------

#: Valid phases in the agent loop.
LoopPhase = Literal["perceive", "plan", "act", "observe"]


@dataclass
class Session:
    """A persistent agent session containing memory tiers and step counter."""

    session_id: str
    agent_config_ref: str
    l1: list[Turn] = field(default_factory=list)
    l2: list[StructuredSummary] = field(default_factory=list)
    step: int = 0


@dataclass
class LoopState:
    """Snapshot of loop execution state for resumability.

    Persisted after each step so an interrupted loop can resume from
    the last completed step and phase.
    """

    session_id: str
    step: int
    phase: LoopPhase
    pending: list[ToolCall] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider-agnostic model shapes (used by the Model Interface layer)
# ---------------------------------------------------------------------------


@dataclass
class ModelRequest:
    """Provider-agnostic model invocation request."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 1.0
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Provider-agnostic model invocation response."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Best-effort native reasoning / thinking segments from the provider.
    reasoning: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Streaming models
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """An incremental unit yielded during provider streaming.

    - kind="text": a text delta (``text`` is populated).
    - kind="tool_call": an assembled tool call delta (``tool_call`` is populated).
    - kind="end": the terminal event carrying final ``usage``.
    """

    kind: Literal["text", "tool_call", "end"]
    text: str = ""
    tool_call: ToolCall | None = None
    usage: dict[str, int] = field(default_factory=dict)
