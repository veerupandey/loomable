"""Tests for channels, pipeline, and universal memory."""
from __future__ import annotations

import asyncio

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.agent.channels import ChannelMessage, InMemoryChannel
from loomable.agent.pipeline import Pipeline
from loomable.kernel.models import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider:
    """Provider that returns configurable responses."""
    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses) if responses else ["default response"]
        self._call_count = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return ModelResponse(content=self._responses[idx], usage={"input_tokens": 5, "output_tokens": 5})


class ApprovalProvider:
    """Provider that says APPROVED on second call."""
    def __init__(self):
        self._call_count = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self._call_count += 1
        if self._call_count >= 2:
            return ModelResponse(content="APPROVED: looks great", usage={"input_tokens": 5, "output_tokens": 5})
        return ModelResponse(content="Needs more detail on section 2", usage={"input_tokens": 5, "output_tokens": 5})


# ---------------------------------------------------------------------------
# Channel tests
# ---------------------------------------------------------------------------


class TestInMemoryChannel:
    """Test InMemoryChannel send/receive/peek/clear."""

    async def test_send_and_receive(self):
        ch = InMemoryChannel(name="test")
        msg = ChannelMessage(sender="agent-a", content="hello")
        await ch.send(msg)
        received = await ch.receive(timeout=1.0)
        assert received is not None
        assert received.sender == "agent-a"
        assert received.content == "hello"

    async def test_receive_timeout_returns_none(self):
        ch = InMemoryChannel(name="test")
        result = await ch.receive(timeout=0.05)
        assert result is None

    async def test_peek_returns_history(self):
        ch = InMemoryChannel(name="test")
        await ch.send(ChannelMessage(sender="a", content="msg1"))
        await ch.send(ChannelMessage(sender="b", content="msg2"))
        history = await ch.peek()
        assert len(history) == 2
        assert history[0].content == "msg1"
        assert history[1].content == "msg2"

    async def test_clear(self):
        ch = InMemoryChannel(name="test")
        await ch.send(ChannelMessage(sender="a", content="msg1"))
        await ch.clear()
        history = await ch.peek()
        assert len(history) == 0

    async def test_name_property(self):
        ch = InMemoryChannel(name="feedback")
        assert ch.name == "feedback"

    async def test_fifo_order(self):
        ch = InMemoryChannel(name="test")
        for i in range(5):
            await ch.send(ChannelMessage(sender="a", content=f"msg-{i}"))
        for i in range(5):
            msg = await ch.receive(timeout=1.0)
            assert msg.content == f"msg-{i}"


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestPipeline:
    """Test Pipeline sequential execution."""

    async def test_single_step_pipeline(self):
        agent = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider(["hello world"])))
        pipeline = Pipeline(steps=[agent])
        result = await pipeline.run("test input")
        assert "hello world" in result.output.text()

    async def test_multi_step_pipeline(self):
        """Each step's output becomes the next step's input."""
        step1 = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider(["step 1 output"])))
        step2 = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider(["final output"])))
        pipeline = Pipeline(steps=[step1, step2])
        result = await pipeline.run("initial input")
        assert "final output" in result.output.text()

    async def test_pipeline_with_session_memory(self):
        """Pipeline should remember across runs when session_id is set."""
        provider = FakeProvider(["I remember everything"])
        agent = Agent(model=ModelSpec(provider="t", provider_impl=provider))
        pipeline = Pipeline(steps=[agent], session_id="session-1")

        await pipeline.run("My name is Alice")
        # Second run should include history context
        result = await pipeline.run("What's my name?")
        # The pipeline should have built history context (we can't test LLM output
        # but we can verify the session was maintained)
        assert pipeline._session is not None
        assert len(pipeline._session.l1) == 4  # 2 user + 2 assistant turns

    async def test_pipeline_without_session_is_stateless(self):
        """Without session_id, each run is independent."""
        provider = FakeProvider(["stateless response"])
        agent = Agent(model=ModelSpec(provider="t", provider_impl=provider))
        pipeline = Pipeline(steps=[agent])

        await pipeline.run("first run")
        await pipeline.run("second run")
        assert pipeline._session is None


# ---------------------------------------------------------------------------
# Pipeline iterative refinement tests
# ---------------------------------------------------------------------------


class TestPipelineRefinement:
    """Test Pipeline with feedback loops."""

    async def test_stops_on_approval(self):
        """Pipeline should stop iterating when stop_condition is met."""
        writer = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider(["draft article"])))
        critic = Agent(model=ModelSpec(provider="t", provider_impl=ApprovalProvider()))
        feedback = InMemoryChannel(name="feedback")

        pipeline = Pipeline(
            steps=[writer, critic],
            feedback_channel=feedback,
            max_iterations=5,
        )
        result = await pipeline.run("Write an article")
        # Should have stopped (APPROVED in output)
        assert "APPROVED" in result.output.text()

    async def test_respects_max_iterations(self):
        """Pipeline should stop at max_iterations even without approval."""
        # Provider that never says APPROVED
        never_approves = FakeProvider(["needs more work"] * 10)
        writer = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider(["draft"])))
        critic = Agent(model=ModelSpec(provider="t", provider_impl=never_approves))
        feedback = InMemoryChannel(name="feedback")

        pipeline = Pipeline(
            steps=[writer, critic],
            feedback_channel=feedback,
            max_iterations=2,
        )
        result = await pipeline.run("Write something")
        # Should complete without hanging
        assert result is not None

    async def test_custom_stop_condition(self):
        """Custom stop condition should be respected."""
        provider = FakeProvider(["DONE: finished"])
        agent = Agent(model=ModelSpec(provider="t", provider_impl=provider))
        feedback = InMemoryChannel(name="fb")

        pipeline = Pipeline(
            steps=[agent],
            feedback_channel=feedback,
            max_iterations=5,
            stop_condition=lambda text: "DONE" in text,
        )
        result = await pipeline.run("test")
        assert "DONE" in result.output.text()


# ---------------------------------------------------------------------------
# Universal memory tests (agent + pipeline share session pattern)
# ---------------------------------------------------------------------------


class TestUniversalMemory:
    """Test that memory works uniformly across agents and pipelines."""

    async def test_agent_session_persists_turns(self):
        """Agent with session_id should accumulate turns."""
        provider = FakeProvider(["response 1", "response 2"])
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=provider),
            session_id="s1",
        )
        await agent.arun("hello")
        await agent.arun("world")
        built = agent._get_built()
        # Should have accumulated turns in session
        assert built.session.step >= 1

    async def test_pipeline_session_persists_turns(self):
        """Pipeline with session_id should accumulate turns."""
        provider = FakeProvider(["pipeline response"])
        agent = Agent(model=ModelSpec(provider="t", provider_impl=provider))
        pipeline = Pipeline(steps=[agent], session_id="p1")

        await pipeline.run("first")
        await pipeline.run("second")
        assert pipeline._session is not None
        assert pipeline._session.step == 2
        assert len(pipeline._session.l1) == 4  # 2 runs x (user + assistant)
