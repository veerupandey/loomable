"""Unit tests for RunContext threading and loop control (task 7.1).

Covers:
- Loop detection stops with LOOP_DETECTED when a tool call repeats `loop_repeat_threshold` times.
- Max iterations stops with MAX_ITERATIONS and re-invokes the model for a final answer.
- Cooperative cancellation stops with CANCELLED and issues no further calls.
- Step budget stops with STEP_BUDGET when exhausted.
- Token budget stops with TOKEN_BUDGET when exceeded.
- Stop reason is recorded in RunResult.metadata["stop_reason"].
- A loop_stop event is emitted.
- Non-idempotent tools are excluded from re-dispatch.
"""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.agent.context import RunContext, StopReason
from loomable.agent.events import Event, NoOpEvents
from loomable.agent.tools import FunctionTool, tool
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _RecordingEvents:
    """Records emitted events for assertion."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


class _ToolLoopProvider:
    """A provider that always requests the same tool call, for loop detection tests."""

    def __init__(self, tool_name: str = "get_data", args: dict | None = None,
                 final_text: str = "final answer"):
        self._tool_name = tool_name
        self._args = args or {"q": "hello"}
        self._final_text = final_text
        self.call_count = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        # If tools are removed (nudge call), return a text-only response.
        if not request.tools:
            return ModelResponse(
                content=self._final_text,
                usage={"input_tokens": 5, "output_tokens": 3},
            )
        return ModelResponse(
            content="",
            tool_calls=[ToolCall(id=f"call_{self.call_count}", tool_name=self._tool_name, args=self._args)],
            usage={"input_tokens": 10, "output_tokens": 5},
        )


class _MaxIterProvider:
    """A provider that always requests tools until tools are removed (nudge)."""

    def __init__(self):
        self.call_count = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        if not request.tools:
            return ModelResponse(
                content="forced final answer",
                usage={"input_tokens": 5, "output_tokens": 3},
            )
        return ModelResponse(
            content="",
            tool_calls=[ToolCall(id=f"call_{self.call_count}", tool_name="fetch", args={"url": f"http://example.com/{self.call_count}"})],
            usage={"input_tokens": 10, "output_tokens": 5},
        )


@tool
def get_data(q: str) -> str:
    """Fetch some data."""
    return f"data for {q}"


@tool
def fetch(url: str) -> str:
    """Fetch a URL."""
    return f"content of {url}"


@tool(idempotent=False)
def send_email(to: str, body: str) -> str:
    """Send an email (side-effecting)."""
    return f"sent to {to}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_loop_detection_stops_with_loop_detected() -> None:
    """Repeating the same (tool, args) `loop_repeat_threshold` times stops the loop."""
    provider = _ToolLoopProvider("get_data", {"q": "hello"})
    agent = Agent(model=ModelSpec(provider="test", provider_impl=provider), tools=[get_data])
    built = agent.build()

    events = _RecordingEvents()
    ctx = RunContext(events=events, loop_repeat_threshold=3, max_steps=10)

    result = await built._run_tool_loop(
        built._coerce_input("test"),
        ctx=ctx,
    )

    assert result.metadata["stop_reason"] == StopReason.LOOP_DETECTED
    # The tool should NOT have been dispatched on the 3rd repeat.
    # Provider is called: iter1 (dispatches), iter2 (dispatches), iter3 (loop detected, break)
    # So provider.call_count == 3 (3 model calls, but 3rd triggers loop detection before dispatch)
    assert provider.call_count == 3
    # A loop_stop event should have been emitted.
    loop_events = [e for e in events.events if e.kind == "loop_stop"]
    assert len(loop_events) == 1
    assert loop_events[0].attributes["stop_reason"] == "loop_detected"


async def test_max_iterations_stops_with_nudge() -> None:
    """Reaching max_tool_iterations stops with MAX_ITERATIONS and nudges the model."""
    provider = _MaxIterProvider()
    agent = Agent(model=ModelSpec(provider="test", provider_impl=provider), tools=[fetch])
    built = agent.build()
    built.max_tool_iterations = 3

    events = _RecordingEvents()
    ctx = RunContext(events=events, max_steps=20)

    result = await built._run_tool_loop(
        built._coerce_input("test"),
        ctx=ctx,
    )

    assert result.metadata["stop_reason"] == StopReason.MAX_ITERATIONS
    # The final output should be the nudge response.
    assert result.output.text() == "forced final answer"
    # Provider called: 3 iterations + 1 nudge = 4.
    assert provider.call_count == 4
    # A loop_stop event should have been emitted.
    loop_events = [e for e in events.events if e.kind == "loop_stop"]
    assert len(loop_events) == 1
    assert loop_events[0].attributes["stop_reason"] == "max_iterations"


async def test_cancellation_stops_immediately() -> None:
    """Setting cancel before the first iteration stops with CANCELLED."""
    provider = _ToolLoopProvider()
    agent = Agent(model=ModelSpec(provider="test", provider_impl=provider), tools=[get_data])
    built = agent.build()

    events = _RecordingEvents()
    ctx = RunContext(events=events, max_steps=10)
    ctx.cancel()  # Cancel before the run starts.

    result = await built._run_tool_loop(
        built._coerce_input("test"),
        ctx=ctx,
    )

    assert result.metadata["stop_reason"] == StopReason.CANCELLED
    # Provider should never have been called.
    assert provider.call_count == 0
    # A loop_stop event should have been emitted.
    loop_events = [e for e in events.events if e.kind == "loop_stop"]
    assert len(loop_events) == 1
    assert loop_events[0].attributes["stop_reason"] == "cancelled"


async def test_step_budget_stops_when_exhausted() -> None:
    """Step budget exhaustion stops with STEP_BUDGET."""
    provider = _MaxIterProvider()
    agent = Agent(model=ModelSpec(provider="test", provider_impl=provider), tools=[fetch])
    built = agent.build()
    built.max_tool_iterations = 20  # High to avoid hitting it.

    events = _RecordingEvents()
    ctx = RunContext(events=events, max_steps=2)

    result = await built._run_tool_loop(
        built._coerce_input("test"),
        ctx=ctx,
    )

    assert result.metadata["stop_reason"] == StopReason.STEP_BUDGET
    # Provider called twice (steps 1 and 2), then step 3 fails budget check.
    assert provider.call_count == 2
    loop_events = [e for e in events.events if e.kind == "loop_stop"]
    assert len(loop_events) == 1
    assert loop_events[0].attributes["stop_reason"] == "step_budget"


async def test_token_budget_stops_when_exceeded() -> None:
    """Token budget exhaustion stops with TOKEN_BUDGET."""
    provider = _MaxIterProvider()
    agent = Agent(model=ModelSpec(provider="test", provider_impl=provider), tools=[fetch])
    built = agent.build()
    built.max_tool_iterations = 20

    events = _RecordingEvents()
    # Set a very low token budget that will be exceeded after 1 model call.
    ctx = RunContext(events=events, max_steps=20, token_budget=10)

    result = await built._run_tool_loop(
        built._coerce_input("test"),
        ctx=ctx,
    )

    assert result.metadata["stop_reason"] == StopReason.TOKEN_BUDGET
    # Provider called once (15 tokens used exceeds budget of 10).
    assert provider.call_count == 1
    loop_events = [e for e in events.events if e.kind == "loop_stop"]
    assert len(loop_events) == 1
    assert loop_events[0].attributes["stop_reason"] == "token_budget"


async def test_max_run_tokens_zero_is_unbounded() -> None:
    """max_run_tokens=0 disables cumulative spend stop (deep-agent default)."""
    from loomable.agent.context import RunContext

    ctx = RunContext(token_budget=10, max_run_tokens=0)
    ctx.add_tokens(10_000)
    assert ctx.token_budget_exceeded() is False


async def test_max_run_tokens_overrides_token_budget() -> None:
    ctx = RunContext(token_budget=10, max_run_tokens=100)
    ctx.add_tokens(50)
    assert ctx.token_budget_exceeded() is False
    ctx.add_tokens(60)
    assert ctx.token_budget_exceeded() is True


async def test_non_idempotent_tool_excluded_from_re_dispatch() -> None:
    """A non-idempotent tool is not re-dispatched on a second identical call."""
    call_count = 0

    @tool(idempotent=False)
    def side_effect_tool(x: str) -> str:
        """A side-effecting tool."""
        nonlocal call_count
        call_count += 1
        return f"done {x}"

    invocation = 0

    class _RepeatingProvider:
        """Calls the side-effecting tool twice with the same args, then stops."""
        def __init__(self):
            self.call_count = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.call_count += 1
            if self.call_count <= 2:
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(id=f"call_{self.call_count}", tool_name="side_effect_tool", args={"x": "hello"})],
                    usage={"input_tokens": 5, "output_tokens": 3},
                )
            return ModelResponse(
                content="all done",
                usage={"input_tokens": 5, "output_tokens": 3},
            )

    provider = _RepeatingProvider()
    agent = Agent(model=ModelSpec(provider="test", provider_impl=provider), tools=[side_effect_tool])
    built = agent.build()

    events = _RecordingEvents()
    ctx = RunContext(events=events, max_steps=10, loop_repeat_threshold=5)

    result = await built._run_tool_loop(
        built._coerce_input("test"),
        ctx=ctx,
    )

    # The side-effecting tool should only have been invoked once (the second
    # identical call should be excluded from dispatch).
    assert call_count == 1
    assert result.metadata["stop_reason"] == StopReason.FINAL


async def test_normal_run_produces_final_stop_reason() -> None:
    """A normal run with no tool calls produces FINAL stop reason."""
    class _SimpleProvider:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="hello", usage={"input_tokens": 5, "output_tokens": 3})

    agent = Agent(model=ModelSpec(provider="test", provider_impl=_SimpleProvider()), tools=[get_data])
    built = agent.build()

    events = _RecordingEvents()
    ctx = RunContext(events=events, max_steps=10)

    result = await built._run_tool_loop(
        built._coerce_input("test"),
        ctx=ctx,
    )

    assert result.metadata["stop_reason"] == StopReason.FINAL
    assert result.output.text() == "hello"
    loop_events = [e for e in events.events if e.kind == "loop_stop"]
    assert len(loop_events) == 1
    assert loop_events[0].attributes["stop_reason"] == "final"


async def test_run_single_accepts_ctx_parameter() -> None:
    """_run_single accepts an optional ctx parameter without error."""
    class _SimpleProvider:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="hello", usage={"input_tokens": 5, "output_tokens": 3})

    agent = Agent(model=ModelSpec(provider="test", provider_impl=_SimpleProvider()))
    built = agent.build()

    ctx = RunContext()
    result = await built._run_single(built._coerce_input("test"), ctx=ctx)

    assert result.output.text() == "hello"
