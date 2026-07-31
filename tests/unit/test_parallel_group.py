"""Tests for loomable.flow.parallel_group.Parallel_Group (Task 3.1).

Validates:
- Req 3.1: Parallel_Group accepts one or more steps and optional name
- Req 3.2: Steps execute concurrently using ParallelEngine
- Req 3.3: Implements Runnable protocol
- Req 3.4: Outputs merged into SharedState keyed by step name
- Req 3.6: Zero steps raises ValueError
- Req 3.7: Auto-generates name from step names when not provided
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.parallel_group import Parallel_Group
from loomable.flow.runnable import Runnable
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_step(name: str, output: str | None = None) -> Step:
    """Create a simple Step that returns a predictable output."""
    text = output or f"output_from_{name}"

    def fn(input):  # noqa: A002
        return text

    return Step(name, fn)


def make_failing_step(name: str) -> Step:
    """Create a Step that always raises an exception."""

    def fn(input):  # noqa: A002
        raise RuntimeError(f"{name} failed!")

    return Step(name, fn)


async def async_fn_a(input):  # noqa: A002
    return "result_a"


async def async_fn_b(input):  # noqa: A002
    return "result_b"


# ---------------------------------------------------------------------------
# Construction validation (Req 3.6)
# ---------------------------------------------------------------------------


class TestParallelGroupConstruction:
    """Parallel_Group construction validates inputs and sets properties."""

    def test_zero_steps_raises_valueerror(self):
        """No steps raises ValueError."""
        with pytest.raises(ValueError, match="At least one step is required"):
            Parallel_Group()

    def test_single_step_valid(self):
        """A single step is accepted."""
        step = make_step("research")
        pg = Parallel_Group(step)
        assert len(pg.steps) == 1

    def test_multiple_steps_valid(self):
        """Multiple steps are accepted."""
        s1 = make_step("research")
        s2 = make_step("analysis")
        pg = Parallel_Group(s1, s2)
        assert len(pg.steps) == 2

    def test_explicit_name_stored(self):
        """An explicit name is used when provided."""
        step = make_step("research")
        pg = Parallel_Group(step, name="my_parallel")
        assert pg.name == "my_parallel"


# ---------------------------------------------------------------------------
# Name auto-generation (Req 3.7)
# ---------------------------------------------------------------------------


class TestParallelGroupNameGeneration:
    """Parallel_Group auto-generates name from step names."""

    def test_auto_name_single_step(self):
        """Auto-generated name includes the step name."""
        step = make_step("research")
        pg = Parallel_Group(step)
        assert pg.name == "parallel_research"

    def test_auto_name_multiple_steps(self):
        """Auto-generated name combines all step names."""
        s1 = make_step("research")
        s2 = make_step("analysis")
        pg = Parallel_Group(s1, s2)
        assert pg.name == "parallel_research_analysis"

    def test_auto_name_three_steps(self):
        """Auto-generated name includes all three step names."""
        s1 = make_step("fetch")
        s2 = make_step("parse")
        s3 = make_step("summarize")
        pg = Parallel_Group(s1, s2, s3)
        assert pg.name == "parallel_fetch_parse_summarize"


# ---------------------------------------------------------------------------
# Runnable protocol (Req 3.3)
# ---------------------------------------------------------------------------


class TestParallelGroupRunnable:
    """Parallel_Group satisfies the Runnable protocol."""

    def test_is_runnable(self):
        """Parallel_Group instances pass isinstance check for Runnable."""
        step = make_step("research")
        pg = Parallel_Group(step)
        assert isinstance(pg, Runnable)

    @pytest.mark.asyncio
    async def test_arun_returns_run_result(self):
        """arun returns a RunResult."""
        step = make_step("research")
        pg = Parallel_Group(step)
        result = await pg.arun("hello")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_arun_with_context(self):
        """arun accepts an optional context parameter."""
        step = make_step("research")
        pg = Parallel_Group(step)
        ctx = RunContext()
        result = await pg.arun("hello", context=ctx)
        assert isinstance(result, RunResult)


# ---------------------------------------------------------------------------
# Parallel execution and output merging (Req 3.2, 3.4)
# ---------------------------------------------------------------------------


class TestParallelGroupExecution:
    """Parallel_Group executes steps concurrently and merges outputs."""

    @pytest.mark.asyncio
    async def test_single_step_output(self):
        """Single step produces output in result."""
        step = make_step("research", "research_result")
        pg = Parallel_Group(step)
        result = await pg.arun("input")
        assert result is not None
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_multiple_steps_all_execute(self):
        """All steps execute and produce outputs."""
        s1 = make_step("step_a", "output_a")
        s2 = make_step("step_b", "output_b")
        pg = Parallel_Group(s1, s2)
        result = await pg.arun("input")

        # Both steps should have produced sub_results
        assert result.sub_results is not None
        assert "step_a" in result.sub_results
        assert "step_b" in result.sub_results

    @pytest.mark.asyncio
    async def test_outputs_keyed_by_step_name(self):
        """Step outputs are accessible in sub_results keyed by name."""
        s1 = make_step("research", "found_data")
        s2 = make_step("analysis", "analyzed_data")
        pg = Parallel_Group(s1, s2)
        result = await pg.arun("input")

        assert result.sub_results is not None
        # Each step's result should contain its output
        assert "found_data" in result.sub_results["research"].output.text()
        assert "analyzed_data" in result.sub_results["analysis"].output.text()

    @pytest.mark.asyncio
    async def test_three_steps_all_execute(self):
        """Three steps all produce outputs."""
        s1 = make_step("a", "out_a")
        s2 = make_step("b", "out_b")
        s3 = make_step("c", "out_c")
        pg = Parallel_Group(s1, s2, s3)
        result = await pg.arun("input")

        assert result.sub_results is not None
        assert "a" in result.sub_results
        assert "b" in result.sub_results
        assert "c" in result.sub_results

    @pytest.mark.asyncio
    async def test_async_steps_execute(self):
        """Async step functions execute correctly in parallel."""
        s1 = Step("step_a", async_fn_a)
        s2 = Step("step_b", async_fn_b)
        pg = Parallel_Group(s1, s2)
        result = await pg.arun("input")

        assert result.sub_results is not None
        assert "result_a" in result.sub_results["step_a"].output.text()
        assert "result_b" in result.sub_results["step_b"].output.text()


# ---------------------------------------------------------------------------
# Fault isolation (Req 3.5)
# ---------------------------------------------------------------------------


class TestParallelGroupFaultIsolation:
    """Parallel_Group isolates individual step failures."""

    @pytest.mark.asyncio
    async def test_one_failure_doesnt_block_others(self):
        """When one step fails, other steps still complete."""
        good_step = make_step("good", "success")
        bad_step = make_failing_step("bad")
        pg = Parallel_Group(good_step, bad_step)
        result = await pg.arun("input")

        # The good step should still have produced a result
        assert result.sub_results is not None
        assert "good" in result.sub_results
        # The good step's output should be present
        assert "success" in result.sub_results["good"].output.text()

    @pytest.mark.asyncio
    async def test_multiple_successes_with_one_failure(self):
        """Multiple good steps succeed even when one fails."""
        s1 = make_step("alpha", "out_alpha")
        s2 = make_failing_step("failing")
        s3 = make_step("gamma", "out_gamma")
        pg = Parallel_Group(s1, s2, s3)
        result = await pg.arun("input")

        assert result.sub_results is not None
        assert "alpha" in result.sub_results
        assert "gamma" in result.sub_results
        assert "out_alpha" in result.sub_results["alpha"].output.text()
        assert "out_gamma" in result.sub_results["gamma"].output.text()


# ---------------------------------------------------------------------------
# Compilation internals
# ---------------------------------------------------------------------------


class TestParallelGroupCompilation:
    """Parallel_Group compiles to a Flow with engine='parallel'."""

    def test_compiled_flow_uses_parallel_engine(self):
        """The internal flow uses engine='parallel'."""
        step = make_step("research")
        pg = Parallel_Group(step)
        assert pg._compiled_flow._engine == "parallel"

    def test_compiled_flow_has_no_edges(self):
        """The internal flow has no edges (all nodes independent)."""
        s1 = make_step("a")
        s2 = make_step("b")
        pg = Parallel_Group(s1, s2)
        assert pg._compiled_flow._edges == []

    def test_compiled_flow_nodes_match_step_names(self):
        """The internal flow's node_ids match the step names."""
        s1 = make_step("research")
        s2 = make_step("analysis")
        pg = Parallel_Group(s1, s2)
        node_ids = set(pg._compiled_flow._nodes.keys())
        assert node_ids == {"research", "analysis"}


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestParallelGroupRepr:
    """Parallel_Group has useful repr."""

    def test_repr_shows_name_and_count(self):
        """repr includes name and step count."""
        s1 = make_step("a")
        s2 = make_step("b")
        pg = Parallel_Group(s1, s2, name="my_pg")
        r = repr(pg)
        assert "my_pg" in r
        assert "2" in r
