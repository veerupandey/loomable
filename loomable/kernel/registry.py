"""loomable.kernel.registry - Extension Registry with lazy resolution.

The ExtensionRegistry is the Kernel component responsible for onboarding
extensions (Skills, MCP servers, API tools), filtering by enabled status,
and lazily resolving tools on first use.

Immutable Kernel data (tool schemas, static system prompt, parsed config)
is shared by reference across agent instances through KernelData.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from loomable.kernel.contracts import Tool
from loomable.kernel.errors import UnsupportedExtensionError
from loomable.kernel.models import (
    AgentConfig,
    ExtensionMechanism,
    OnboardingRequest,
    SUPPORTED_MECHANISMS,
)


# ---------------------------------------------------------------------------
# Extension Handle and Spec (returned by onboarding / enabled_extensions)
# ---------------------------------------------------------------------------


@dataclass
class ExtensionHandle:
    """Handle returned after successful onboarding of an extension."""

    id: str
    capability: str
    mechanism: ExtensionMechanism


@dataclass
class ExtensionSpec:
    """Specification describing an onboarded, enabled extension."""

    id: str
    capability: str
    mechanism: ExtensionMechanism
    enabled: bool = True
    tool_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Immutable Kernel Data (shared by reference across agent instances)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelData:
    """Immutable Kernel data shared by reference across agent instances.

    This includes tool schemas, the static system prompt, and parsed config.
    Because the dataclass is frozen, it cannot be mutated after construction,
    ensuring safe sharing without copies.
    """

    tool_schemas: tuple[dict[str, Any], ...] = ()
    system_prompt: str = ""
    config: AgentConfig | None = None


# ---------------------------------------------------------------------------
# Extension Registry
# ---------------------------------------------------------------------------


class ExtensionRegistry:
    """Registry for onboarding, enabling, and lazily resolving extensions.

    Key behaviours:
    - Only SKILL, MCP_SERVER, and API_TOOL mechanisms are accepted.
    - Only extensions marked ``enabled`` are eligible for resolution.
    - Expensive resources (Skill body, MCP connections, HTTP clients) are
      materialized lazily on first ``resolve_tool()`` call, not at agent
      instantiation.
    - Immutable Kernel data is shared by reference across agent instances
      via the ``kernel_data`` attribute.
    """

    def __init__(self, kernel_data: KernelData | None = None) -> None:
        # Shared immutable kernel data — same reference across agent instances
        self.kernel_data: KernelData = kernel_data or KernelData()

        # Internal stores
        self._extensions: dict[str, ExtensionSpec] = {}
        self._tool_factories: dict[str, Callable[[], Tool]] = {}
        self._resolved_tools: dict[str, Tool] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Onboarding
    # ------------------------------------------------------------------

    def onboard(self, request: OnboardingRequest) -> ExtensionHandle:
        """Onboard a new extension.

        Raises:
            UnsupportedExtensionError: if the request's mechanism is not one
                of {SKILL, MCP_SERVER, API_TOOL}.
        """
        if request.mechanism not in SUPPORTED_MECHANISMS:
            supported_names = sorted(m.value for m in SUPPORTED_MECHANISMS)
            raise UnsupportedExtensionError(supported_names)

        ext_id = self._generate_id()
        spec = ExtensionSpec(
            id=ext_id,
            capability=request.capability,
            mechanism=request.mechanism,
            enabled=True,
        )
        self._extensions[ext_id] = spec

        handle = ExtensionHandle(
            id=ext_id,
            capability=request.capability,
            mechanism=request.mechanism,
        )
        return handle

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def enabled_extensions(self) -> list[ExtensionSpec]:
        """Return all currently enabled extensions.

        Only extensions with ``enabled=True`` are included. Disabled
        extensions are never materialized or made available.
        """
        return [ext for ext in self._extensions.values() if ext.enabled]

    # ------------------------------------------------------------------
    # Lazy Tool Resolution
    # ------------------------------------------------------------------

    def register_tool_factory(
        self, tool_name: str, factory: Callable[[], Tool]
    ) -> None:
        """Register a lazy factory for a tool.

        The factory is a zero-arg callable that materializes the expensive
        resource (Skill body, MCP connection, HTTP client) on first call.
        It will only be invoked when ``resolve_tool()`` is called for this
        tool name, and only if the owning extension is enabled.
        """
        self._tool_factories[tool_name] = factory

    def resolve_tool(self, name: str) -> Tool:
        """Resolve a tool by name, triggering lazy materialization on first use.

        Only tools belonging to enabled extensions are resolvable. The
        expensive resource behind the tool is created on the first call
        and cached for subsequent calls.

        Raises:
            KeyError: if the tool name is not registered or its owning
                extension is not enabled.
        """
        # Return cached if already resolved
        if name in self._resolved_tools:
            return self._resolved_tools[name]

        # Check factory exists
        if name not in self._tool_factories:
            raise KeyError(f"Tool not found: {name}")

        # Verify the tool belongs to an enabled extension
        if not self._is_tool_enabled(name):
            raise KeyError(
                f"Tool '{name}' belongs to a disabled extension"
            )

        # Lazy materialization: invoke the factory
        tool = self._tool_factories[name]()
        self._resolved_tools[name] = tool
        return tool

    # ------------------------------------------------------------------
    # Extension management helpers
    # ------------------------------------------------------------------

    def disable_extension(self, ext_id: str) -> None:
        """Mark an extension as disabled. Its tools become unresolvable."""
        if ext_id in self._extensions:
            # ExtensionSpec is mutable, just flip the flag
            self._extensions[ext_id].enabled = False
            # Evict any already-resolved tools for this extension
            for tool_name in list(self._resolved_tools):
                if self._tool_owner(tool_name) == ext_id:
                    del self._resolved_tools[tool_name]

    def enable_extension(self, ext_id: str) -> None:
        """Mark an extension as enabled."""
        if ext_id in self._extensions:
            self._extensions[ext_id].enabled = True

    def register_tool_for_extension(
        self, ext_id: str, tool_name: str, factory: Callable[[], Tool]
    ) -> None:
        """Register a tool under a specific extension with a lazy factory."""
        if ext_id in self._extensions:
            self._extensions[ext_id].tool_names.append(tool_name)
        self._tool_factories[tool_name] = factory

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_id(self) -> str:
        """Generate a unique extension id."""
        ext_id = f"ext-{self._next_id}"
        self._next_id += 1
        return ext_id

    def _is_tool_enabled(self, tool_name: str) -> bool:
        """Check whether a tool belongs to an enabled extension."""
        owner_id = self._tool_owner(tool_name)
        if owner_id is None:
            # Tool not associated with any extension — allow resolution
            return True
        return self._extensions[owner_id].enabled

    def _tool_owner(self, tool_name: str) -> str | None:
        """Find the extension that owns a given tool, or None."""
        for ext_id, spec in self._extensions.items():
            if tool_name in spec.tool_names:
                return ext_id
        return None
