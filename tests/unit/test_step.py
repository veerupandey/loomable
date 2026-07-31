"""Tests for loomable.flow.step.Step (Task 1.1).

Validates:
- Req 1.1: Step accepts name, agent, description, deps
- Req 1.2: Step implements Runnable protocol by delegating arun
- Req 1.3: Non-Runnable callables wrapped in FunctionRunnable
- Req 1.6: Empty/None name raises ValueError
- Req 1.7: deps injected into RunContext, overriding flow-level deps
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def echo_fn(input):
    """Simple function that echoes input as a string."""
    return f"echo:{input}"


async def async_echo_fn(input):
    """Async function that echoes input."""
    return f"async_echo:{input}"


def fn_with_deps(input, *, deps=None):
    """Function that returns deps info."""
    return f"deps={deps}"


class MockRunnable:
    """A minimal Runnable implementation for testing."""

    async def arun(self, input, *, context=None):
        from loomable.content import AgentOutput, MediaPart, Modality

        text = f"mock:{input}"
        output = AgentOutput(
            parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=text.encode())]
        )
        return RunResult(output=output, session_id="test")


# ---------------------------------------------------------------------------
# Construction validation (Req 1.6)
# ---------------------------------------------------------------------------


class TestStepConstruction:
    """Step construction validates name and wraps callables."""

    def test_empty_name_raises_valueerror(self):
        """Empty string name raises ValueError."""
        with pytest.raises(ValueError, match="Step name is required"):
            Step("", echo_fn)

    def test_none_name_raises_valueerror(self):
        """None name raises ValueError."""
        with pytest.raises(ValueError, match="Step name is required"):
            Step(None, echo_fn)

    def test_valid_name_is_stored(self):
        """A valid non-empty name is accessible via property."""
        s = Step("my_step", echo_fn)
        assert s.name == "my_step"

    def test_description_default_empty(self):
        """Description defaults to empty string."""
        s = Step("step1", echo_fn)
        assert s.description == ""

    def test_description_stored(self):
        """Provided description is accessible via property."""
        s = Step("step1", echo_fn, description="Does echoing")
        assert s.description == "Does echoing"

    def test_callable_wrapped_in_function_runnable(self):
        """A plain callable is internally wrapped in FunctionRunnable."""
        s = Step("step1", echo_fn)
        assert isinstance(s._agent, FunctionRunnable)

    def test_runnable_not_wrapped(self):
        """A Runnable instance is stored directly without wrapping."""
        r = MockRunnable()
        s = Step("step1", r)
        assert s._agent is r

    def test_non_callable_raises_typeerror(self):
        """Passing a non-callable, non-Runnable raises TypeError."""
        with pytest.raises(TypeError, match="agent must be a Runnable or callable"):
            Step("step1", 42)


# ---------------------------------------------------------------------------
# Runnable protocol (Req 1.2)
# ---------------------------------------------------------------------------


class TestStepRunnable:
    """Step satisfies the Runnable protocol."""

    def test_step_is_runnable(self):
        """Step instances pass isinstance check for Runnable."""
        s = Step("step1", echo_fn)
        assert isinstance(s, Runnable)

    @pytest.mark.asyncio
    async def test_arun_with_callable(self):
        """arun delegates to wrapped callable and returns RunResult."""
        s = Step("step1", echo_fn)
        result = await s.arun("hello")
        assert isinstance(result, RunResult)
        assert "echo:hello" in result.output.text()

    @pytest.mark.asyncio
    async def test_arun_with_async_callable(self):
        """arun delegates to async callable correctly."""
        s = Step("step1", async_echo_fn)
        result = await s.arun("world")
        assert isinstance(result, RunResult)
        assert "async_echo:world" in result.output.text()

    @pytest.mark.asyncio
    async def test_arun_with_runnable(self):
        """arun delegates to a Runnable agent directly."""
        r = MockRunnable()
        s = Step("step1", r)
        result = await s.arun("test")
        assert isinstance(result, RunResult)
        assert "mock:test" in result.output.text()


# ---------------------------------------------------------------------------
# Dependency injection (Req 1.7)
# ---------------------------------------------------------------------------


class TestStepDepsInjection:
    """Step injects deps into RunContext, overriding flow-level deps."""

    @pytest.mark.asyncio
    async def test_deps_injected_when_no_context(self):
        """When no context is provided, step creates one with its deps."""
        s = Step("step1", fn_with_deps, deps="my_deps")
        result = await s.arun("input")
        assert "deps=my_deps" in result.output.text()

    @pytest.mark.asyncio
    async def test_deps_override_flow_level_deps(self):
        """Step-level deps override the deps on a provided context."""
        flow_ctx = RunContext(deps="flow_deps")
        s = Step("step1", fn_with_deps, deps="step_deps")
        result = await s.arun("input", context=flow_ctx)
        assert "deps=step_deps" in result.output.text()

    @pytest.mark.asyncio
    async def test_no_deps_passes_context_through(self):
        """When step has no deps, the original context deps are used."""
        flow_ctx = RunContext(deps="flow_deps")
        s = Step("step1", fn_with_deps)
        result = await s.arun("input", context=flow_ctx)
        assert "deps=flow_deps" in result.output.text()

    @pytest.mark.asyncio
    async def test_no_deps_no_context(self):
        """When step has no deps and no context, deps is None."""
        s = Step("step1", fn_with_deps)
        result = await s.arun("input")
        assert "deps=None" in result.output.text()


# ---------------------------------------------------------------------------
# Properties (Req 1.1)
# ---------------------------------------------------------------------------


class TestStepProperties:
    """Step exposes name and description as read-only properties."""

    def test_name_property(self):
        """name property returns the construction-time name."""
        s = Step("research", echo_fn)
        assert s.name == "research"

    def test_description_property(self):
        """description property returns the construction-time description."""
        s = Step("research", echo_fn, description="Do research")
        assert s.description == "Do research"
