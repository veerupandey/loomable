"""loomable.agent.errors - High-level API error taxonomy.

These errors extend the kernel :class:`~loomable.kernel.errors.LoomableError` base so
the high-level layer shares a single error hierarchy with the kernel, without
modifying ``loomable.kernel`` (the base class is imported, not changed).
"""

from __future__ import annotations

from loomable.kernel.errors import LoomableError


class AgentConfigError(LoomableError):
    """Raised when the Agent builder configuration is missing or invalid.

    Carries the name of the offending field so callers know exactly which
    configuration value must be supplied or corrected (Req 1.6).
    """

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"Invalid or missing agent configuration field: {field}")


class UnsupportedModalityError(LoomableError):
    """Raised when a run involves a modality the configured model does not support.

    Raised for input modalities not in ``capabilities.input`` (before the provider
    is invoked) and for output modalities not in ``capabilities.output`` (Req 4.4,
    5.4, 6.4). Carries the offending ``modality`` and the ``model`` identifier so
    callers know exactly what was rejected.
    """

    def __init__(self, modality: str, model: str) -> None:
        self.modality = modality
        self.model = model
        super().__init__(
            f"Model '{model}' does not support modality '{modality}'."
        )


class ToolHookRejection(LoomableError):
    """Raised by a pre-hook to reject a tool call before it executes.

    A pre-hook may signal rejection either by returning ``False`` or by raising
    this error. When a call is rejected, the :class:`~loomable.agent.builder.BuiltAgent`
    blocks it and records the rejection through the kernel guardrail harness without
    executing the tool (Req 14.3). Carries the offending ``tool_name`` and an optional
    human-readable ``reason`` so audits know exactly what was blocked and why.
    """

    def __init__(self, tool_name: str, reason: str | None = None) -> None:
        self.tool_name = tool_name
        self.reason = reason
        message = f"Tool call rejected by pre-hook: {tool_name}"
        if reason:
            message = f"{message} ({reason})"
        super().__init__(message)


class InputValidationError(LoomableError):
    """Raised when an agent input fails validation against a configured input schema.

    When an ``Agent`` is created with an ``input_schema`` (a Pydantic model or
    dataclass), dict/model inputs are validated/coerced against it before the run.
    Carries the human-readable ``reason`` naming the validation failure.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Input validation failed: {reason}")


class StructuredOutputError(LoomableError):
    """Raised when a structured-output run cannot produce the requested schema.

    Raised when the model's text output cannot be parsed as JSON, or when the parsed
    value fails validation/coercion into the requested ``output_schema`` (Req 13.3).
    Carries the human-readable ``reason`` naming the failure so callers know exactly
    what went wrong.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Structured output failed: {reason}")


class HITLPause(LoomableError):
    """Raised when a run pauses for human-in-the-loop approval.

    This is not a failure — it signals that the agent needs external approval
    before continuing. The pending actions are checkpointed so the run can
    resume from where it paused after approval is granted, even across
    process restarts.

    Attributes:
        pending_calls: The tool calls awaiting approval.
        thread_id: The thread identifier for resuming.
    """

    def __init__(self, pending_calls: list, thread_id: str = "") -> None:
        self.pending_calls = pending_calls
        self.thread_id = thread_id
        tool_names = [getattr(c, "tool_name", str(c)) for c in pending_calls]
        super().__init__(
            f"Run paused for approval: {', '.join(tool_names)}. "
            f"Resume with thread_id='{thread_id}' after approving."
        )
