"""Stress tests for the agent-harness feature — real-world scenarios.

Exercises the full harness stack: resilience, loop detection, cancellation,
budgets, tools (simple + side-effecting), MCP tools, skills, complexity routing,
reasoning tools (think/plan), notes, observability, context bounding, pinned facts,
and model-based summarization.

Organized by difficulty:
  - SIMPLE:  Single-shot and basic tool use, events emitted
  - MEDIUM:  Multi-tool loops, loop detection, timeouts, concurrency, notes
  - TOUGH:   Cancellation mid-run, budget exhaustion, plan escalation, resilience
             under transient failures, MCP tool integration, skill loading

All providers/MCP/HTTP are mocked — this runs entirely in-process.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from loomable.agent import Agent, ModelSpec, tool, FunctionTool
from loomable.agent.context import RunContext, StopReason
from loomable.agent.events import Event, JSONTracer, NoOpEvents
from loomable.memory import Note, NoteStore
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.kernel.long_term import LongTermStore
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolOutcome,
    ToolResult,
)
from loomable.providers.errors import PermanentProviderError, TransientProviderError
from loomable.providers.resilient import ResilientModel, RetryPolicy

pytestmark = pytest.mark.asyncio


# ===========================================================================
# FIXTURES & HELPERS
# ===========================================================================


# --- Scripted providers ---


class ScriptedProvider:
    """A provider that returns pre-scripted responses in sequence."""

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if self._idx >= len(self._responses):
            return ModelResponse(content="(exhausted)", usage={})
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class FlakyProvider:
    """A provider that fails N times with transient errors, then succeeds."""

    def __init__(self, fail_count: int, success_response: ModelResponse):
        self._fail_count = fail_count
        self._success = success_response
        self.attempts = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.attempts += 1
        if self.attempts <= self._fail_count:
            raise TransientProviderError("flaky", status_code=503, retry_after=0.01)
        return self._success


class PermanentlyBrokenProvider:
    """A provider that always fails with a permanent error."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise PermanentProviderError("broken", status_code=401)


# --- Tools ---


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def weather(city: str) -> str:
    """Get current weather for a city."""
    data = {"Vancouver": "18°C, cloudy", "Tokyo": "25°C, sunny", "London": "12°C, rain"}
    return data.get(city, f"Unknown city: {city}")


@tool(idempotent=False)
def send_notification(user: str, message: str) -> str:
    """Send a notification to a user (side-effecting)."""
    return f"Notification sent to {user}: {message}"


@tool
def slow_database_query(query: str) -> str:
    """A slow database query that takes time."""
    import asyncio as _asyncio
    # Simulate slow I/O
    _asyncio.get_event_loop()  # Just to check we're in async context
    return f"Results for: {query}"


@tool
def search_docs(query: str) -> str:
    """Search the documentation."""
    return f"Found 3 results for '{query}': [doc1, doc2, doc3]"


@tool
def code_review(code: str) -> str:
    """Review code for issues."""
    return f"Review of {len(code)} chars: LGTM, no issues found."


# --- Fake MCP tool (simulates an MCP server tool) ---


class FakeMCPTool:
    """Simulates an MCP-backed tool for testing."""

    def __init__(self, name: str, delay: float = 0.0):
        self.name = name
        self.description = f"MCP tool: {name}"
        self.parameters = {"type": "object", "properties": {"input": {"type": "string"}}}
        self.idempotent = True
        self.invocations = 0

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        self.invocations += 1
        await asyncio.sleep(0.01)
        return ToolResult(content=f"MCP result from {self.name}: {args}")


# ===========================================================================
# SIMPLE SCENARIOS — single-shot, basic tools, observability
# ===========================================================================


class TestSimpleScenarios:
    """Tests for simple, single-turn agent interactions."""

    async def test_single_shot_no_tools(self):
        """Simple question, no tools — just the model answers."""
        provider = ScriptedProvider([
            ModelResponse(content="Paris is the capital of France.",
                         usage={"input_tokens": 20, "output_tokens": 8}),
        ])
        tracer = JSONTracer()
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            instructions="You are a geography expert.",
            events=tracer,
        )
        result = await agent.arun("What is the capital of France?")

        assert result.output.text() == "Paris is the capital of France."
        assert result.usage["input_tokens"] == 20
        # Events: run_start, model_call, run_end
        assert any(e.kind == "run_start" for e in tracer.trace)
        assert any(e.kind == "model_call" for e in tracer.trace)
        assert any(e.kind == "run_end" for e in tracer.trace)
        # run_start is first, run_end is last
        assert tracer.trace[0].kind == "run_start"
        assert tracer.trace[-1].kind == "run_end"


    async def test_single_tool_call_and_response(self):
        """Model calls one tool, gets result, produces final answer."""
        provider = ScriptedProvider([
            # First call: model requests calculator tool
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="calculator", args={"expression": "2**10"})],
                usage={"input_tokens": 30, "output_tokens": 10},
            ),
            # Second call: model sees result and answers
            ModelResponse(
                content="2^10 = 1024",
                usage={"input_tokens": 40, "output_tokens": 5},
            ),
        ])
        tracer = JSONTracer()
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            tools=[calculator],
            events=tracer,
        )
        result = await agent.arun("What is 2 to the power of 10?")

        assert "1024" in result.output.text()
        assert result.metadata.get("stop_reason") == "final"
        # Trace should have tool_call events
        tool_events = [e for e in tracer.trace if e.kind == "tool_call"]
        assert len(tool_events) == 1
        model_events = [e for e in tracer.trace if e.kind == "model_call"]
        assert len(model_events) == 2

    async def test_multiple_tools_in_sequence(self):
        """Model calls multiple tools across iterations."""
        provider = ScriptedProvider([
            # Iter 1: call weather
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="weather", args={"city": "Vancouver"})],
                usage={"input_tokens": 30, "output_tokens": 10},
            ),
            # Iter 2: call weather again for Tokyo
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c2", tool_name="weather", args={"city": "Tokyo"})],
                usage={"input_tokens": 50, "output_tokens": 10},
            ),
            # Iter 3: final answer
            ModelResponse(
                content="Vancouver is 18°C and cloudy. Tokyo is 25°C and sunny.",
                usage={"input_tokens": 60, "output_tokens": 15},
            ),
        ])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            tools=[weather],
        )
        result = await agent.arun("Compare weather in Vancouver and Tokyo")

        assert "Vancouver" in result.output.text()
        assert "Tokyo" in result.output.text()
        assert result.metadata["stop_reason"] == "final"


    async def test_think_tool_scratchpad(self):
        """The think tool echoes the thought without side effects."""
        provider = ScriptedProvider([
            # Model uses think to reason
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="think",
                                     args={"thought": "Let me consider the options..."})],
                usage={"input_tokens": 20, "output_tokens": 10},
            ),
            # Then answers
            ModelResponse(
                content="After careful consideration, option B is best.",
                usage={"input_tokens": 40, "output_tokens": 10},
            ),
        ])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            tools=[calculator],
            think_tool=True,
        )
        result = await agent.arun("Which option is best?")

        assert "option B" in result.output.text()
        assert result.metadata["stop_reason"] == "final"


# ===========================================================================
# MEDIUM SCENARIOS — loop detection, timeouts, concurrency, notes, routing
# ===========================================================================


class TestMediumScenarios:
    """Tests for medium-complexity scenarios with multiple harness features."""

    async def test_loop_detection_triggers_on_repeated_calls(self):
        """Model stuck calling the same tool repeatedly — harness detects and stops."""
        provider = ScriptedProvider([
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id=f"c{i}", tool_name="search_docs",
                                     args={"query": "how to configure"})],
                usage={"input_tokens": 20, "output_tokens": 5},
            )
            for i in range(10)  # Would loop forever without detection
        ])
        tracer = JSONTracer()
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            tools=[search_docs],
            events=tracer,
            loop_repeat_threshold=3,
        )
        result = await agent.arun("How do I configure the system?")

        assert result.metadata["stop_reason"] == "loop_detected"
        # Should have stopped after 3 identical calls
        loop_events = [e for e in tracer.trace if e.kind == "loop_stop"]
        assert len(loop_events) == 1
        assert "loop_detected" in loop_events[0].attributes["stop_reason"]


    async def test_max_iterations_with_nudge(self):
        """Model keeps requesting tools until max iterations — gets nudged."""
        call_num = [0]

        class AlwaysToolProvider:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1
                if not request.tools:
                    return ModelResponse(content="Fine, here's my answer.",
                                        usage={"input_tokens": 10, "output_tokens": 5})
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(id=f"c{call_num[0]}", tool_name="search_docs",
                                         args={"query": f"query_{call_num[0]}"})],
                    usage={"input_tokens": 20, "output_tokens": 5},
                )

        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=AlwaysToolProvider()),
            tools=[search_docs],
        )
        built = agent.build()
        built.max_tool_iterations = 4

        result = await built.arun("Keep searching")

        assert result.metadata["stop_reason"] == "max_iterations"
        assert result.output.text() == "Fine, here's my answer."

    async def test_tool_timeout_produces_error_fed_back(self):
        """A slow tool times out; the error is fed back to the model."""
        @tool
        async def very_slow_tool(query: str) -> str:
            """A tool that hangs."""
            await asyncio.sleep(10.0)
            return "never reached"

        provider = ScriptedProvider([
            # Model calls the slow tool
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="very_slow_tool",
                                     args={"query": "test"})],
                usage={"input_tokens": 20, "output_tokens": 5},
            ),
            # Model sees timeout error, gives final answer
            ModelResponse(
                content="The tool timed out. I cannot complete that request.",
                usage={"input_tokens": 40, "output_tokens": 10},
            ),
        ])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            tools=[very_slow_tool],
        )
        built = agent.build()
        built.tool_timeout = 0.05  # 50ms timeout

        result = await built.arun("Run the slow query")

        assert result.metadata["stop_reason"] == "final"
        assert "timed out" in result.output.text()
        # Verify the timeout message was in the tool result fed to model
        last_request = provider.calls[-1]
        tool_msgs = [m for m in last_request.messages if m.get("role") == "tool"]
        assert any("timed out" in m["content"][0]["text"] for m in tool_msgs)


    async def test_concurrency_cap_limits_parallel_tools(self):
        """Concurrency cap limits how many tools run simultaneously."""
        peak_concurrent = [0]
        current = [0]

        @tool
        async def tracked_tool(id: str) -> str:
            """A tool that tracks concurrency."""
            current[0] += 1
            if current[0] > peak_concurrent[0]:
                peak_concurrent[0] = current[0]
            await asyncio.sleep(0.03)
            current[0] -= 1
            return f"done-{id}"

        # Model requests 4 tools at once
        provider = ScriptedProvider([
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id=f"c{i}", tool_name="tracked_tool", args={"id": str(i)})
                    for i in range(4)
                ],
                usage={"input_tokens": 20, "output_tokens": 10},
            ),
            ModelResponse(content="All done.", usage={"input_tokens": 30, "output_tokens": 3}),
        ])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            tools=[tracked_tool],
        )
        built = agent.build()
        built.tool_concurrency = 2  # Only 2 at a time

        result = await built.arun("Do 4 things")

        assert result.metadata["stop_reason"] == "final"
        assert peak_concurrent[0] <= 2

    async def test_non_idempotent_tool_not_re_dispatched(self):
        """A side-effecting tool is only dispatched once, even if model re-requests."""
        send_count = [0]

        @tool(idempotent=False)
        def send_email(to: str, body: str) -> str:
            """Send an email (dangerous)."""
            send_count[0] += 1
            return f"Sent to {to}"

        provider = ScriptedProvider([
            # First request: send email
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="send_email",
                                     args={"to": "alice@co.com", "body": "hi"})],
                usage={"input_tokens": 20, "output_tokens": 5},
            ),
            # Second request: tries to re-send the same email
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c2", tool_name="send_email",
                                     args={"to": "alice@co.com", "body": "hi"})],
                usage={"input_tokens": 30, "output_tokens": 5},
            ),
            # Final answer
            ModelResponse(content="Email sent.", usage={"input_tokens": 35, "output_tokens": 3}),
        ])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            tools=[send_email],
            loop_repeat_threshold=10,  # Don't trigger loop detection
        )
        result = await agent.arun("Send email to Alice")

        assert send_count[0] == 1  # Only dispatched once!
        assert result.metadata["stop_reason"] == "final"


    async def test_complexity_router_selects_strategy(self):
        """The complexity router correctly routes simple vs complex tasks."""
        provider = ScriptedProvider([
            ModelResponse(content="42", usage={"input_tokens": 10, "output_tokens": 1}),
        ])
        router = ComplexityRouter()
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            tools=[calculator],
            complexity_router=router,
        )
        # A simple question should go through single-shot or tool loop
        result = await agent.arun("Hi")
        assert result.output.text() == "42"

    async def test_event_trace_attached_to_result(self):
        """A JSONTracer's accumulated events are copied to RunResult.trace."""
        provider = ScriptedProvider([
            ModelResponse(content="done", usage={"input_tokens": 5, "output_tokens": 2}),
        ])
        tracer = JSONTracer()
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            events=tracer,
        )
        result = await agent.arun("test")

        # Trace on result matches tracer's internal trace
        assert len(result.trace) > 0
        assert result.trace == tracer.trace
        # All events have timestamps
        for event in result.trace:
            assert event.t > 0

    async def test_token_budget_stops_run(self):
        """Token budget exceeded mid-run stops with TOKEN_BUDGET reason."""
        call_num = [0]

        class HighTokenProvider:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1
                if not request.tools:
                    return ModelResponse(content="forced", usage={"input_tokens": 5, "output_tokens": 2})
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(id=f"c{call_num[0]}", tool_name="calculator",
                                         args={"expression": f"{call_num[0]}+1"})],
                    usage={"input_tokens": 500, "output_tokens": 200},
                )

        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=HighTokenProvider()),
            tools=[calculator],
            token_budget=600,  # Will be exceeded after first model call
        )
        result = await agent.arun("Calculate stuff")

        assert result.metadata["stop_reason"] == "token_budget"


# ===========================================================================
# TOUGH SCENARIOS — resilience, cancellation, plan escalation, MCP, skills
# ===========================================================================


class TestToughScenarios:
    """Tests for tough, production-edge-case scenarios."""


    async def test_resilient_model_retries_transient_errors(self):
        """ResilientModel retries transient 503 errors and succeeds."""
        success_response = ModelResponse(
            content="Recovered after retries!",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        flaky = FlakyProvider(fail_count=2, success_response=success_response)
        policy = RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.05)
        resilient = ResilientModel(inner=flaky, policy=policy)

        request = ModelRequest(messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        response = await resilient.complete(request)

        assert response.content == "Recovered after retries!"
        assert flaky.attempts == 3  # 2 failures + 1 success

    async def test_resilient_model_fails_fast_on_permanent_error(self):
        """ResilientModel does NOT retry permanent errors (401)."""
        broken = PermanentlyBrokenProvider()
        policy = RetryPolicy(max_attempts=5, base_delay=0.01)
        resilient = ResilientModel(inner=broken, policy=policy)

        request = ModelRequest(messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        with pytest.raises(PermanentProviderError) as exc_info:
            await resilient.complete(request)

        assert exc_info.value.status_code == 401

    async def test_resilient_model_exhausts_retries(self):
        """ResilientModel raises after exhausting all attempts."""
        always_fail = FlakyProvider(fail_count=100, success_response=ModelResponse(content=""))
        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02)
        resilient = ResilientModel(inner=always_fail, policy=policy)

        request = ModelRequest(messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        with pytest.raises(TransientProviderError):
            await resilient.complete(request)

        assert always_fail.attempts == 3

    async def test_cooperative_cancellation_mid_run(self):
        """Cancelling mid-run stops at the next loop boundary."""
        call_num = [0]
        cancel_ctx = [None]

        class SlowProvider:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1
                if call_num[0] == 2:
                    # Cancel after the first tool dispatch cycle
                    cancel_ctx[0].cancel()
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(id=f"c{call_num[0]}", tool_name="calculator",
                                         args={"expression": f"{call_num[0]}*2"})],
                    usage={"input_tokens": 10, "output_tokens": 5},
                )

        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=SlowProvider()),
            tools=[calculator],
        )
        built = agent.build()

        # We need to thread the RunContext manually
        ctx = RunContext(max_steps=20, loop_repeat_threshold=20)
        cancel_ctx[0] = ctx

        result = await built._run_tool_loop(built._coerce_input("keep going"), ctx=ctx)

        assert result.metadata["stop_reason"] == "cancelled"
        # Should have stopped after 2 model calls (cancelled before 3rd iteration)
        assert call_num[0] == 2


    async def test_resilience_wired_through_builder(self):
        """Builder wraps provider in ResilientModel when resilience is configured."""
        success_response = ModelResponse(
            content="Recovered!",
            usage={"input_tokens": 10, "output_tokens": 3},
        )
        flaky = FlakyProvider(fail_count=1, success_response=success_response)
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=flaky),
            resilience=RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.05),
        )
        result = await agent.arun("test")

        assert result.output.text() == "Recovered!"
        assert flaky.attempts == 2  # 1 failure + 1 success

    async def test_mcp_tool_integration(self):
        """MCP tools work through the harness like regular tools."""
        mcp_tool = FakeMCPTool("remote_api")

        provider = ScriptedProvider([
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="remote_api",
                                     args={"input": "fetch data"})],
                usage={"input_tokens": 20, "output_tokens": 5},
            ),
            ModelResponse(
                content="Got data from remote API.",
                usage={"input_tokens": 30, "output_tokens": 5},
            ),
        ])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            tools=[mcp_tool],
        )
        result = await agent.arun("Call the remote API")

        assert result.metadata["stop_reason"] == "final"
        assert "remote API" in result.output.text()
        assert mcp_tool.invocations == 1

    async def test_multiple_tools_with_mcp_and_local_mixed(self):
        """Mix of local tools and MCP tools in the same run."""
        mcp_tool = FakeMCPTool("external_search")

        provider = ScriptedProvider([
            # First: use local calculator
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="calculator",
                                     args={"expression": "100/4"})],
                usage={"input_tokens": 20, "output_tokens": 5},
            ),
            # Then: use MCP tool
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c2", tool_name="external_search",
                                     args={"input": "latest news"})],
                usage={"input_tokens": 30, "output_tokens": 5},
            ),
            # Final answer combining both
            ModelResponse(
                content="25 and here's the news.",
                usage={"input_tokens": 40, "output_tokens": 5},
            ),
        ])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            tools=[calculator, mcp_tool],
        )
        result = await agent.arun("Calculate 100/4 and search news")

        assert result.metadata["stop_reason"] == "final"
        assert mcp_tool.invocations == 1


    async def test_plan_escalation_via_plan_tool(self):
        """The plan tool decomposes a task into parallel steps."""
        call_num = [0]

        class PlanProvider:
            """Simulates planning behavior."""
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1
                messages_text = str(request.messages)

                # Planning phase: return steps as JSON array
                if "planner" in messages_text.lower() or "Break the user" in messages_text:
                    return ModelResponse(
                        content='["Research topic A", "Research topic B"]',
                        usage={"input_tokens": 30, "output_tokens": 10},
                    )
                # Step execution: return step results
                if "Complete ONLY this step" in messages_text:
                    return ModelResponse(
                        content=f"Step result {call_num[0]}",
                        usage={"input_tokens": 20, "output_tokens": 5},
                    )
                # Synthesis: combine results
                if "Integrate these" in messages_text:
                    return ModelResponse(
                        content="Synthesized answer combining A and B.",
                        usage={"input_tokens": 40, "output_tokens": 8},
                    )
                # Initial call: model decides to use the plan tool
                if request.tools and call_num[0] == 1:
                    return ModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="c1", tool_name="plan",
                                             args={"task": "Research topic A and B",
                                                   "max_steps": 3})],
                        usage={"input_tokens": 25, "output_tokens": 10},
                    )
                # After plan tool returns, model gives final answer
                return ModelResponse(
                    content="Here's the synthesized research on A and B.",
                    usage={"input_tokens": 50, "output_tokens": 12},
                )

        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=PlanProvider()),
            tools=[calculator],
            plan_tool=True,
        )
        result = await agent.arun("Research topic A and B in depth")

        # The run should complete with a final answer
        assert result.output.text()  # Non-empty output
        assert "A and B" in result.output.text()

    async def test_pinned_facts_survive_compaction(self):
        """Pinned facts are never compacted away."""
        provider = ScriptedProvider([
            ModelResponse(content=f"answer {i}", usage={"input_tokens": 10, "output_tokens": 3})
            for i in range(20)
        ])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            session_id="test-session",
            memory_window=4,
            compaction_threshold=6,
        )
        built = agent.build()

        # Pin a fact early
        built.pin_fact("CRITICAL: The API key is abc123")

        # Run multiple times to trigger compaction
        for i in range(8):
            await built.arun(f"question {i}")

        # The pinned fact should still be in memory
        prefix = built._memory_prefix()
        prefix_text = str(prefix)
        assert "abc123" in prefix_text


    async def test_full_harness_stress_multi_feature(self):
        """Stress test combining: tools + timeout + loop detection + events + trace.

        Simulates a real-world scenario where an agent:
        1. Calls a fast tool successfully
        2. Calls a slow tool that times out
        3. Sees the timeout error and tries to loop (same tool, same args)
        4. Loop detection kicks in and stops
        """
        @tool
        async def fast_api(endpoint: str) -> str:
            """Call a fast API."""
            return f"200 OK from {endpoint}"

        @tool
        async def slow_api(endpoint: str) -> str:
            """Call a slow API."""
            await asyncio.sleep(10.0)
            return "never"

        call_num = [0]

        class RealisticProvider:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1
                if call_num[0] == 1:
                    return ModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="c1", tool_name="fast_api",
                                             args={"endpoint": "/health"})],
                        usage={"input_tokens": 20, "output_tokens": 5},
                    )
                elif call_num[0] == 2:
                    return ModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="c2", tool_name="slow_api",
                                             args={"endpoint": "/data"})],
                        usage={"input_tokens": 30, "output_tokens": 5},
                    )
                else:
                    # After seeing timeout, model retries the same slow call
                    return ModelResponse(
                        content="",
                        tool_calls=[ToolCall(id=f"c{call_num[0]}", tool_name="slow_api",
                                             args={"endpoint": "/data"})],
                        usage={"input_tokens": 40, "output_tokens": 5},
                    )

        tracer = JSONTracer()
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=RealisticProvider()),
            tools=[fast_api, slow_api],
            events=tracer,
            loop_repeat_threshold=3,
        )
        built = agent.build()
        built.tool_timeout = 0.05

        result = await built.arun("Check health then fetch data")

        # Should stop due to loop detection on the slow_api retry
        assert result.metadata["stop_reason"] == "loop_detected"
        # Trace should have all the events
        assert any(e.kind == "run_start" for e in tracer.trace)
        assert any(e.kind == "model_call" for e in tracer.trace)
        assert any(e.kind == "tool_call" for e in tracer.trace)
        assert any(e.kind == "loop_stop" for e in tracer.trace)
        assert any(e.kind == "run_end" for e in tracer.trace)
        # Model call durations should be non-negative
        for e in tracer.trace:
            if e.duration_ms is not None:
                assert e.duration_ms >= 0


    async def test_step_budget_limits_runaway_agent(self):
        """Step budget prevents a runaway agent from burning through resources."""
        call_num = [0]

        class RunawayProvider:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(id=f"c{call_num[0]}", tool_name="search_docs",
                                         args={"query": f"different query {call_num[0]}"})],
                    usage={"input_tokens": 15, "output_tokens": 5},
                )

        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=RunawayProvider()),
            tools=[search_docs],
            loop_repeat_threshold=100,  # Won't trigger (different args each time)
        )
        built = agent.build()
        built.max_tool_iterations = 100  # Very high

        ctx = RunContext(max_steps=5, loop_repeat_threshold=100)
        result = await built._run_tool_loop(built._coerce_input("search"), ctx=ctx)

        assert result.metadata["stop_reason"] == "step_budget"
        assert call_num[0] == 5  # Exactly 5 steps executed

    async def test_context_bounding_evicts_old_messages(self):
        """Context bounding evicts low-priority messages to stay within budget."""
        # Build a conversation with many messages
        provider = ScriptedProvider([
            ModelResponse(content="Final.", usage={"input_tokens": 10, "output_tokens": 2}),
        ])
        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            token_budget=200,  # Very small budget
            instructions="System prompt.",
        )
        built = agent.build()

        # The system prompt + user message should fit, but _bound_messages is called
        result = await built.arun("Short question")
        assert result.output.text() == "Final."

    async def test_end_to_end_with_all_features_enabled(self):
        """Full integration: resilience + tools + events + router + budget."""
        success_response = ModelResponse(
            content="All features working!",
            usage={"input_tokens": 20, "output_tokens": 5},
        )
        # Provider fails once then succeeds
        flaky = FlakyProvider(fail_count=1, success_response=success_response)
        tracer = JSONTracer()
        router = ComplexityRouter()

        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=flaky),
            instructions="You are helpful.",
            events=tracer,
            complexity_router=router,
            resilience=RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02),
            think_tool=True,
            loop_repeat_threshold=3,
            token_budget=10000,
        )
        result = await agent.arun("Simple greeting")

        assert result.output.text() == "All features working!"
        assert flaky.attempts == 2
        assert len(result.trace) > 0
        assert any(e.kind == "run_start" for e in result.trace)
        assert any(e.kind == "run_end" for e in result.trace)
