"""loomable.kernel.errors - Error taxonomy for the loomable agent framework.

Each error identifies the responsible component and carries the fields
specified in the design Error Taxonomy. All errors extend LoomableError.
"""

from __future__ import annotations


class LoomableError(Exception):
    """Base class for all loomable framework errors."""


class UnsupportedExtensionError(LoomableError):
    """Raised when onboarding via an unsupported mechanism.

    Carries the list of supported mechanisms so the caller knows what is allowed.
    """

    def __init__(self, supported_mechanisms: list[str]) -> None:
        self.supported_mechanisms = supported_mechanisms
        super().__init__(
            f"Unsupported extension mechanism. "
            f"Supported mechanisms: {', '.join(supported_mechanisms)}"
        )


class ModelProviderError(LoomableError):
    """Raised when a configured model provider is unavailable."""

    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        super().__init__(f"Model provider unavailable: {provider_id}")


class MCPConnectionError(LoomableError):
    """Raised when an MCP server connection fails."""

    def __init__(self, server_id: str) -> None:
        self.server_id = server_id
        super().__init__(f"MCP server connection failed: {server_id}")


class MCPToolError(LoomableError):
    """Raised when an MCP tool invocation returns an error."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"MCP tool invocation failed: {tool_name}")


class SkillLoadError(LoomableError):
    """Raised when a Skill fails to load."""

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        super().__init__(f"Skill failed to load: {skill_name}")


class ScriptToolError(LoomableError):
    """Raised when a script tool execution fails."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Script tool execution failed: {tool_name}")


class APIToolError(LoomableError):
    """Raised when an API tool returns a non-2xx HTTP status."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"API tool returned HTTP {status_code}")


class APIToolTimeoutError(LoomableError):
    """Raised when an API tool exceeds its configured timeout."""

    def __init__(self, tool_name: str, timeout: float) -> None:
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(
            f"API tool '{tool_name}' timed out after {timeout}s"
        )


class MemoryBackendError(LoomableError):
    """Raised when a short-term or vector memory backend is unavailable."""

    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        super().__init__(f"Memory backend unavailable: {backend_id}")


class SessionNotFoundError(LoomableError):
    """Raised when resuming an unknown session id."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class PlanningModelError(LoomableError):
    """Raised when the planning model is unavailable."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Planning model unavailable: {model_id}")


class SubagentError(LoomableError):
    """Raised when a subagent fails."""

    def __init__(self, subagent_id: str) -> None:
        self.subagent_id = subagent_id
        super().__init__(f"Subagent failed: {subagent_id}")


class GuardrailViolation(LoomableError):
    """Raised when an action violates a guardrail rule."""

    def __init__(self, rule_id: str, action: str) -> None:
        self.rule_id = rule_id
        self.action = action
        super().__init__(
            f"Guardrail violation: rule '{rule_id}' blocked action '{action}'"
        )
