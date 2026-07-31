"""loomable.toolkits._base - Abstract base class for all loomable toolkits.

A Toolkit groups related :class:`~loomable.agent.tools.FunctionTool` instances
and exposes them via the :meth:`tools` method.  The Agent builder flattens
toolkits transparently so they integrate with the ``Agent(tools=[...])`` API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from loomable.agent.tools import FunctionTool
from loomable.agent.errors import AgentConfigError


class Toolkit(ABC):
    """Base class for all loomable toolkits.

    Subclasses register tool methods and expose them as FunctionTool instances.
    Supports include/exclude filtering to let users cherry-pick tools.
    """

    def __init__(
        self,
        *,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        if include_tools is not None and exclude_tools is not None:
            raise AgentConfigError(
                "include_tools/exclude_tools: cannot specify both"
            )
        self._include_tools = include_tools
        self._exclude_tools = exclude_tools

    @abstractmethod
    def _register_tools(self) -> list[FunctionTool]:
        """Return all FunctionTool instances this toolkit provides."""
        ...

    def tools(self) -> list[FunctionTool]:
        """Return filtered list of FunctionTool instances."""
        all_tools = self._register_tools()
        if self._include_tools is not None:
            return [t for t in all_tools if t.name in self._include_tools]
        if self._exclude_tools is not None:
            return [t for t in all_tools if t.name not in self._exclude_tools]
        return all_tools
