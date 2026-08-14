# Feature: agent-ergonomics, Property 6
"""Property 6: Tool-use loop runs tools and terminates.

For any scripted sequence of model responses that request tool calls then stop,
the loop SHALL dispatch each requested tool call, feed results back, and terminate
with the final no-tool response as output, executing exactly the requested calls.
The loop terminates — bounded by max_tool_iterations.

**Validates: Requirements 3.1, 3.2, 3.3**
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.agent import Agent, ModelSpec, RunResult
from loomable.content import ModelCapabilities
from loomable.kernel.contracts import Tool
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: a tool name (valid identifier, short)
tool_names_st = st.from_regex(r"[a-z][a-z0-9_]{0,14}", fullmatch=True)

# Strategy: simple argument values
simple_values_st = st.one_of(
    st.text(min_size=0, max_size=20),
    st.integers(min_value=-1000, max_value=1000),
    st.booleans(),
)

# Strategy: a single tool call specification (tool_name, args dict)
# Each tool call includes a unique "_call_idx" arg to avoid triggering
# loop detection when the same tool is called with otherwise-identical args.
tool_call_spec_st = st.tuples(
    tool_names_st,
    st.dictionaries(
        st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True),
        simple_values_st,
        min_size=0,
        max_size=3,
    ),
)

# Strategy: a single "iteration" — a non-empty list of tool calls the model requests
iteration_st = st.lists(tool_call_spec_st, min_size=1, max_size=3)

# Strategy: a scripted sequence of iterations (1 to 4 iterations of tool calls,
# followed by an implicit final response with no tool calls)
scripted_sequence_st = st.lists(iteration_st, min_size=1, max_size=4)

# Strategy: the final answer text
# Non-whitespace: empty/whitespace finals are synthesized into a summary.
final_answer_st = st.text(min_size=1, max_size=50).filter(lambda s: bool(s.strip()))

# Strategy: max_tool_iterations (at least 1)
max_iterations_st = st.integers(min_value=1, max_value=10)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """A model provider that returns a scripted sequence of responses.

    Each response in the script is either a list of tool calls (the model wants
    to call tools) or a final text response (no tool calls, loop should stop).
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._call_index >= len(self._responses):
            # Safety fallback: return a no-tool response
            return ModelResponse(content="fallback", tool_calls=[])
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp


class RecordingTool(Tool):
    """A tool that records invocations and returns a fixed result."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Tool {name}"
        self.invocations: list[dict[str, Any]] = []

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(args))
        return ToolResult(content=f"result-from-{self.name}")


def _build_scripted_agent(
    scripted_sequence: list[list[tuple[str, dict[str, Any]]]],
    final_answer: str,
    max_tool_iterations: int,
) -> tuple[Agent, ScriptedProvider, dict[str, RecordingTool]]:
    """Build an agent with scripted model responses and recording tools.

    Each tool call gets a unique "_idx" arg injected to avoid triggering the
    RunContext loop-detection mechanism (which fires when the same tool+args
    signature repeats >= loop_repeat_threshold times).

    Returns the Agent (not yet built), the scripted provider, and the tool registry.
    """
    # Collect all unique tool names from the scripted sequence
    all_tool_names: set[str] = set()
    for iteration in scripted_sequence:
        for tool_name, _args in iteration:
            all_tool_names.add(tool_name)

    # Create recording tools for each unique name
    tools: dict[str, RecordingTool] = {}
    for name in all_tool_names:
        tools[name] = RecordingTool(name)

    # Build scripted model responses: each iteration becomes a ModelResponse with
    # tool_calls, followed by a final response with no tool_calls.
    # Inject a unique "_idx" per call to avoid loop detection.
    responses: list[ModelResponse] = []
    call_counter = 0
    for iteration in scripted_sequence:
        tool_calls = []
        for tn, args in iteration:
            unique_args = dict(args)
            unique_args["_idx"] = call_counter
            call_counter += 1
            tool_calls.append(
                ToolCall(id=str(uuid.uuid4()), tool_name=tn, args=unique_args)
            )
        responses.append(ModelResponse(content="", tool_calls=tool_calls))

    # Final response (no tool calls)
    responses.append(ModelResponse(content=final_answer, tool_calls=[]))

    provider = ScriptedProvider(responses)

    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        capabilities=ModelCapabilities(),
        tools=list(tools.values()),
    )

    return agent, provider, tools


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestToolLoopExecutionAndTermination:
    """Property 6: Tool-use loop runs tools and terminates."""

    @settings(max_examples=100, deadline=None)
    @given(
        scripted_sequence=scripted_sequence_st,
        final_answer=final_answer_st,
    )
    @pytest.mark.asyncio
    async def test_loop_dispatches_all_tool_calls_and_terminates(
        self,
        scripted_sequence: list[list[tuple[str, dict[str, Any]]]],
        final_answer: str,
    ) -> None:
        """The loop dispatches each tool call in the scripted sequence and
        terminates with the final no-tool response as its output."""
        # Use a max_tool_iterations high enough to not cut off the sequence
        max_iter = len(scripted_sequence) + 2

        agent, provider, tools = _build_scripted_agent(
            scripted_sequence, final_answer, max_iter
        )

        # Build the agent with sufficient iterations
        built = agent.build()
        built.max_tool_iterations = max_iter

        result = await built.arun("test input")

        # (1) The loop terminated and returned the final answer
        assert isinstance(result, RunResult)
        assert result.output.text() == final_answer

        # (2) Every requested tool call was dispatched exactly once
        # Count expected invocations per tool (args include injected _idx)
        expected_invocations: dict[str, int] = {}
        for iteration in scripted_sequence:
            for tool_name, _args in iteration:
                expected_invocations[tool_name] = expected_invocations.get(tool_name, 0) + 1

        for tool_name, expected_count in expected_invocations.items():
            recording_tool = tools[tool_name]
            assert len(recording_tool.invocations) == expected_count, (
                f"Tool '{tool_name}' expected {expected_count} invocations, "
                f"got {len(recording_tool.invocations)}"
            )
            # Verify the original args (minus _idx) match
            call_idx = 0
            for iteration in scripted_sequence:
                for tn, args in iteration:
                    if tn == tool_name:
                        actual = dict(recording_tool.invocations[call_idx])
                        actual.pop("_idx", None)
                        assert actual == args
                        call_idx += 1

        # (3) The model was called exactly len(scripted_sequence) + 1 times
        # (one per tool-iteration + one final)
        assert provider._call_index == len(scripted_sequence) + 1

    @settings(max_examples=100, deadline=None)
    @given(
        scripted_sequence=scripted_sequence_st,
        final_answer=final_answer_st,
    )
    @pytest.mark.asyncio
    async def test_loop_feeds_tool_results_back_to_model(
        self,
        scripted_sequence: list[list[tuple[str, dict[str, Any]]]],
        final_answer: str,
    ) -> None:
        """After dispatching tools, the loop feeds results back as tool messages
        in the next model request."""
        max_iter = len(scripted_sequence) + 2

        agent, provider, tools = _build_scripted_agent(
            scripted_sequence, final_answer, max_iter
        )

        built = agent.build()
        built.max_tool_iterations = max_iter

        await built.arun("test input")

        # For each iteration after the first model call, the request should
        # contain tool result messages from the previous dispatch.
        for i in range(1, len(provider.requests)):
            request = provider.requests[i]
            # The messages should contain at least one "tool" role message
            tool_messages = [
                m for m in request.messages if m.get("role") == "tool"
            ]
            # There should be tool result messages from previous iterations
            assert len(tool_messages) > 0, (
                f"Request {i} should contain tool result messages"
            )

    @settings(max_examples=100, deadline=None)
    @given(
        max_iter=st.integers(min_value=1, max_value=5),
        extra_iterations=st.integers(min_value=1, max_value=5),
        final_answer=final_answer_st,
    )
    @pytest.mark.asyncio
    async def test_loop_bounded_by_max_tool_iterations(
        self,
        max_iter: int,
        extra_iterations: int,
        final_answer: str,
    ) -> None:
        """The loop terminates even when the model keeps requesting tool calls,
        bounded by max_tool_iterations. The model is nudged for a final answer."""
        # Create a sequence longer than max_tool_iterations
        total_iterations = max_iter + extra_iterations
        # Each iteration requests one tool call
        scripted_sequence = [
            [("bounded_tool", {"step": i})] for i in range(total_iterations)
        ]

        # Build responses: all tool-call responses followed by a final answer
        tool_responses: list[ModelResponse] = []
        for i in range(total_iterations):
            tool_responses.append(
                ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=str(uuid.uuid4()),
                            tool_name="bounded_tool",
                            args={"step": i},
                        )
                    ],
                )
            )
        # The nudge response (final answer after being told to stop)
        tool_responses.append(ModelResponse(content=final_answer, tool_calls=[]))

        provider = ScriptedProvider(tool_responses)
        recording_tool = RecordingTool("bounded_tool")

        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            capabilities=ModelCapabilities(),
            tools=[recording_tool],
        )
        built = agent.build()
        built.max_tool_iterations = max_iter

        result = await built.arun("test input")

        # The loop terminated
        assert isinstance(result, RunResult)

        # The tool was invoked at most max_iter times (the loop stops dispatching
        # once max_tool_iterations is reached)
        assert len(recording_tool.invocations) <= max_iter

        # The stop reason indicates max iterations was reached
        assert result.metadata.get("stop_reason") == "max_iterations"

    @settings(max_examples=100, deadline=None)
    @given(final_answer=final_answer_st)
    @pytest.mark.asyncio
    async def test_single_iteration_loop_dispatches_and_terminates(
        self,
        final_answer: str,
    ) -> None:
        """A single iteration with tool calls followed by a no-tool response
        terminates correctly."""
        scripted_sequence = [[("simple_tool", {"key": "value"})]]

        agent, provider, tools = _build_scripted_agent(
            scripted_sequence, final_answer, max_tool_iterations=10
        )
        built = agent.build()
        built.max_tool_iterations = 10

        result = await built.arun("test input")

        assert result.output.text() == final_answer
        assert len(tools["simple_tool"].invocations) == 1
        # The invocation includes the injected _idx; verify the original args
        actual = dict(tools["simple_tool"].invocations[0])
        actual.pop("_idx", None)
        assert actual == {"key": "value"}
