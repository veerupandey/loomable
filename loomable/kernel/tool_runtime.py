"""loomable.kernel.tool_runtime - Concurrent tool dispatch with isolation.

The ToolRuntime dispatches a batch of ToolCalls concurrently using
asyncio.gather(..., return_exceptions=True). Each ToolOutcome carries the
originating tool_call_id with either a successful result or an isolated error.
One tool failure does NOT cancel siblings.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loomable.kernel.contracts import Tool
from loomable.kernel.models import ToolCall, ToolError, ToolOutcome, ToolResult


class ToolRuntime:
    """Dispatches tool calls concurrently with per-call isolation.

    The runtime resolves tool names via a registry (name -> Tool mapping)
    and executes all calls in parallel. Each call produces its own
    ToolOutcome regardless of whether siblings succeed or fail.
    """

    def __init__(self, tools: dict[str, Tool]) -> None:
        """Initialize with a mapping of tool name -> Tool instance.

        Args:
            tools: A dictionary mapping tool names to their Tool implementations.
        """
        self._tools = tools

    async def dispatch(self, calls: list[ToolCall]) -> list[ToolOutcome]:
        """Dispatch tool calls concurrently and return their outcomes.

        Uses asyncio.gather(..., return_exceptions=True) so that one failure
        does not cancel siblings. Each outcome carries the originating
        tool_call_id with either a result or an isolated error.

        Args:
            calls: List of ToolCall objects to execute concurrently.

        Returns:
            A list of ToolOutcome objects, one per input call, in the same
            order as the input calls. Each outcome's call_id matches the
            originating ToolCall.id.
        """
        if not calls:
            return []

        tasks = [self._invoke_one(call) for call in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outcomes: list[ToolOutcome] = []
        for call, result in zip(calls, results):
            if isinstance(result, ToolOutcome):
                outcomes.append(result)
            elif isinstance(result, BaseException):
                # An unexpected exception escaped _invoke_one; wrap it
                outcomes.append(
                    ToolOutcome(
                        call_id=call.id,
                        error=ToolError(
                            message=f"Unexpected error: {type(result).__name__}: {result}",
                            details={"exception_type": type(result).__name__},
                        ),
                    )
                )
            else:
                # Should not happen, but handle defensively
                outcomes.append(
                    ToolOutcome(
                        call_id=call.id,
                        error=ToolError(
                            message=f"Unexpected dispatch result type: {type(result).__name__}",
                        ),
                    )
                )

        return outcomes

    async def _invoke_one(self, call: ToolCall) -> ToolOutcome:
        """Invoke a single tool call and return its outcome.

        Catches exceptions from the tool invocation and wraps them
        into a ToolOutcome with a ToolError, ensuring isolation.
        """
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return ToolOutcome(
                call_id=call.id,
                error=ToolError(
                    message=f"Tool not found: {call.tool_name}",
                    details={"tool_name": call.tool_name},
                ),
            )

        try:
            result = await tool.invoke(call.args)
            return ToolOutcome(call_id=call.id, result=result)
        except Exception as exc:
            return ToolOutcome(
                call_id=call.id,
                error=ToolError(
                    message=f"{type(exc).__name__}: {exc}",
                    details={
                        "exception_type": type(exc).__name__,
                        "tool_name": call.tool_name,
                    },
                ),
            )
