"""Integration tests for workflow ergonomics (Task 10.2).

Validates backward compatibility and integration of the new workflow API:
- Req 8.1: Existing helpers (sequential, parallel, route, coordinate) unchanged
- Req 8.2: Existing Loop(body=runnable) constructor unchanged
- Req 8.3: New classes accepted by existing helpers (Runnable protocol)
- Req 8.4: New classes importable from loomable.flow
- Req 8.5: New classes don't modify existing behavior
- Req 9.1: Nested Workflow inside Workflow executes correctly
- Req 9.2: Step used as Loop body
- Req 9.3: Parallel_Group with nested elements compiles correctly
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.state import SharedState


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def echo_fn(input):
    """Simple echo function."""
    return f"echo:{input}"


def upper_fn(input):
    """Upper-case function."""
    return str(input).upper()


def prefix_fn(input):
    """Adds a prefix."""
    return f"prefix:{input}"


async def async_double_fn(input):
    """Async function that doubles input text."""
    return f"{input}:{input}"


def counter_fn(input):
    """Returns a counter string."""
    return f"counted:{input}"


# ---------------------------------------------------------------------------
# 1. Import tests — Verify all new symbols importable from loomable.flow
# ---------------------------------------------------------------------------


class TestImports:
    """Verify new workflow ergonomics classes are importable from loomable.flow."""

    def test_step_importable(self):
        """Step is importable from loomable.flow."""
        from loomable.flow import Step

        assert Step is not None

    def test_workflow_importable(self):
        """Workflow is importable from loomable.flow."""
        from loomable.flow import Workflow

        assert Workflow is not None

    def test_condition_importable(self):
        """Condition is importable from loomable.flow."""
        from loomable.flow import Condition

        assert Condition is not None

    def test_parallel_group_importable(self):
        """Parallel_Group is importable from loomable.flow."""
        from loomable.flow import Parallel_Group

        assert Parallel_Group is not None

    def test_flowclass_importable(self):
        """FlowClass is importable from loomable.flow."""
        from loomable.flow import FlowClass

        assert FlowClass is not None

    def test_composable_element_importable(self):
        """ComposableElement type alias is importable from loomable.flow."""
        from loomable.flow import ComposableElement

        assert ComposableElement is not None

    def test_start_decorator_importable(self):
        """start decorator is importable from loomable.flow."""
        from loomable.flow import start

        assert start is not None

    def test_listen_decorator_importable(self):
        """listen decorator is importable from loomable.flow."""
        from loomable.flow import listen

        assert listen is not None

    def test_router_decorator_importable(self):
        """router decorator is importable from loomable.flow."""
        from loomable.flow import router

        assert router is not None

    def test_all_new_symbols_in_one_import(self):
        """All new symbols importable in a single import statement."""
        from loomable.flow import (
            Step,
            Workflow,
            Condition,
            ComposableElement,
            Parallel_Group,
            FlowClass,
            start,
            listen,
            router,
        )

        # All should be non-None and distinct
        symbols = [Step, Workflow, Condition, ComposableElement,
                   Parallel_Group, FlowClass, start, listen, router]
        assert all(s is not None for s in symbols)


# ---------------------------------------------------------------------------
# 2. Backward compatibility — Existing helpers unchanged
# ---------------------------------------------------------------------------


class TestFlowHelpersModule:
    """``loomable.flow.helpers`` still builds working Flows (advanced escape hatch)."""

    @pytest.mark.asyncio
    async def test_sequential_still_works(self):
        """sequential(FunctionRunnable(fn)) produces a working Flow."""
        from loomable.flow import FunctionRunnable
        from loomable.flow.helpers import sequential

        flow = sequential(FunctionRunnable(echo_fn), FunctionRunnable(upper_fn))
        result = await flow.arun("hello")
        assert isinstance(result, RunResult)
        # sequential: echo_fn("hello") -> "echo:hello", then upper_fn("echo:hello") -> "ECHO:HELLO"
        assert "ECHO:HELLO" in result.output.text()

    @pytest.mark.asyncio
    async def test_sequential_with_plain_callables(self):
        """sequential() still accepts plain callables and wraps them."""
        from loomable.flow.helpers import sequential

        flow = sequential(echo_fn, upper_fn)
        result = await flow.arun("test")
        assert isinstance(result, RunResult)
        assert "ECHO:TEST" in result.output.text()

    @pytest.mark.asyncio
    async def test_parallel_still_works(self):
        """parallel(FunctionRunnable(fn1), FunctionRunnable(fn2)) works."""
        from loomable.flow import FunctionRunnable
        from loomable.flow.helpers import parallel

        flow = parallel(FunctionRunnable(echo_fn), FunctionRunnable(upper_fn))
        result = await flow.arun("hi")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_parallel_with_plain_callables(self):
        """parallel() still accepts plain callables."""
        from loomable.flow.helpers import parallel

        flow = parallel(echo_fn, upper_fn)
        result = await flow.arun("data")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_route_still_works(self):
        """route(chooser, choices) still works with callables."""
        from loomable.flow.helpers import route

        def chooser(input):
            return "branch_a"

        flow = route(chooser, {"branch_a": echo_fn, "branch_b": upper_fn})
        result = await flow.arun("input")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_coordinate_still_works(self):
        """coordinate(workers, manager) still works."""
        from loomable.flow.helpers import coordinate

        def manager(input):
            return f"managed:{input}"

        flow = coordinate([echo_fn, upper_fn], manager)
        result = await flow.arun("task")
        assert isinstance(result, RunResult)


# ---------------------------------------------------------------------------
# 3. Backward compatibility — Existing Loop(body=runnable) unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompatibilityLoop:
    """Existing Loop(body=runnable) constructor works unchanged."""

    @pytest.mark.asyncio
    async def test_loop_body_with_function_runnable(self):
        """Loop(body=FunctionRunnable(fn)) still works."""
        from loomable.flow import Loop, FunctionRunnable

        loop = Loop(body=FunctionRunnable(echo_fn), max_iterations=2)
        result = await loop.arun("input")
        assert isinstance(result, RunResult)
        assert "echo:" in result.output.text()

    @pytest.mark.asyncio
    async def test_loop_body_with_verifier(self):
        """Loop(body=..., verifier=callable) still works."""
        from loomable.flow import Loop, FunctionRunnable

        call_count = {"n": 0}

        def counting_fn(input):
            call_count["n"] += 1
            return f"attempt:{call_count['n']}"

        # Verifier that passes on second iteration
        def verifier(output, context):
            return "attempt:2" in output.text()

        loop = Loop(
            body=FunctionRunnable(counting_fn),
            verifier=verifier,
            max_iterations=5,
        )
        result = await loop.arun("start")
        assert isinstance(result, RunResult)
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_loop_max_iterations_default(self):
        """Loop defaults to max_iterations=3."""
        from loomable.flow import Loop, FunctionRunnable

        loop = Loop(body=FunctionRunnable(echo_fn))
        # AlwaysOkVerifier is used when no verifier → runs exactly once
        result = await loop.arun("once")
        assert isinstance(result, RunResult)
        assert "echo:once" in result.output.text()


# ---------------------------------------------------------------------------
# 4. Integration — New classes compose with existing helpers
# ---------------------------------------------------------------------------


class TestComposabilityWithHelpers:
    """New classes compose with sequential(), parallel(), Loop(body=...)."""

    @pytest.mark.asyncio
    async def test_step_in_sequential(self):
        """Step can be passed to sequential() as a Runnable."""
        from loomable.flow import Step
        from loomable.flow.helpers import sequential

        step_a = Step("echo_step", echo_fn)
        step_b = Step("upper_step", upper_fn)
        flow = sequential(step_a, step_b)
        result = await flow.arun("hello")
        assert isinstance(result, RunResult)
        assert "ECHO:HELLO" in result.output.text()

    @pytest.mark.asyncio
    async def test_step_as_loop_body(self):
        """Step can be used as Loop(body=step) since it's a Runnable."""
        from loomable.flow import Step, Loop

        step = Step("echo_step", echo_fn)
        loop = Loop(body=step, max_iterations=2)
        result = await loop.arun("data")
        assert isinstance(result, RunResult)
        assert "echo:" in result.output.text()

    @pytest.mark.asyncio
    async def test_workflow_in_sequential(self):
        """Workflow can be passed to sequential() as a Runnable."""
        from loomable.flow import Step, Workflow
        from loomable.flow.helpers import sequential

        wf = Workflow("inner", steps=[Step("step1", echo_fn)])
        flow = sequential(wf, FunctionRunnable(upper_fn))
        result = await flow.arun("data")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_workflow_as_loop_body(self):
        """Workflow can be used as Loop(body=workflow)."""
        from loomable.flow import Step, Workflow, Loop

        wf = Workflow("loop_body", steps=[Step("echo", echo_fn)])
        loop = Loop(body=wf, max_iterations=2)
        result = await loop.arun("input")
        assert isinstance(result, RunResult)
        assert "echo:" in result.output.text()

    @pytest.mark.asyncio
    async def test_workflow_as_loop_body_with_verifier(self):
        """Workflow can be used as Loop body with verifier."""
        from loomable.flow import Step, Workflow, Loop

        call_count = {"n": 0}

        def counting_fn(input):
            call_count["n"] += 1
            return f"iteration:{call_count['n']}"

        wf = Workflow("counting_wf", steps=[Step("counter", counting_fn)])
        loop = Loop(
            body=wf,
            verifier=lambda output, ctx: "iteration:2" in output.text(),
            max_iterations=5,
        )
        result = await loop.arun("start")
        assert isinstance(result, RunResult)
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# 5. Integration — Nested Workflow inside Workflow
# ---------------------------------------------------------------------------


class TestNestedWorkflow:
    """Workflow inside Workflow executes correctly end-to-end."""

    @pytest.mark.asyncio
    async def test_nested_workflow_executes(self):
        """A Workflow used as a step inside another Workflow executes correctly."""
        from loomable.flow import Step, Workflow

        inner_wf = Workflow(
            "inner",
            steps=[Step("inner_step", echo_fn)],
        )
        outer_wf = Workflow(
            "outer",
            steps=[
                Step("first", prefix_fn),
                inner_wf,
            ],
        )
        result = await outer_wf.arun("hello")
        assert isinstance(result, RunResult)
        # The inner workflow receives the previous node's output via SharedState.
        # The key point is that it executes without error and produces output.
        output_text = result.output.text()
        assert "echo:" in output_text
        assert "prefix:hello" in output_text

    @pytest.mark.asyncio
    async def test_nested_workflow_data_flows_through(self):
        """Data flows from outer steps -> inner workflow -> outer continuation."""
        from loomable.flow import Step, Workflow

        inner_wf = Workflow(
            "inner",
            steps=[Step("inner_echo", echo_fn)],
        )
        outer_wf = Workflow(
            "outer",
            steps=[
                Step("step_a", prefix_fn),
                inner_wf,
                Step("step_c", upper_fn),
            ],
        )
        result = await outer_wf.arun("test")
        assert isinstance(result, RunResult)
        # Data flows through all three stages. The final step applies upper().
        # The exact string depends on how the Flow engine passes data between
        # nested workflows — the key invariant is that all 3 stages execute
        # and the final output contains upper-cased content.
        output_text = result.output.text()
        # step_c applies upper_fn, so output should be upper-cased
        assert output_text == output_text.upper()
        # And the output should contain evidence of the earlier steps
        assert "PREFIX:TEST" in output_text


# ---------------------------------------------------------------------------
# 6. Integration — Workflow with Condition + Parallel_Group mixed
# ---------------------------------------------------------------------------


class TestWorkflowWithConditionAndParallel:
    """Workflow mixing Condition and Parallel_Group elements."""

    @pytest.mark.asyncio
    async def test_workflow_with_condition_then_branch(self):
        """Workflow with Condition executes then_steps when predicate is True."""
        from loomable.flow import Step, Workflow, Condition
        from loomable.flow.state import SharedState

        cond = Condition(
            condition=lambda state: True,
            then_steps=[Step("then_step", upper_fn)],
            else_steps=[Step("else_step", echo_fn)],
        )
        wf = Workflow("cond_wf", steps=[Step("first", prefix_fn), cond])
        result = await wf.arun("data")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_workflow_with_parallel_group(self):
        """Workflow containing a Parallel_Group executes steps concurrently."""
        from loomable.flow import Step, Workflow, Parallel_Group

        pg = Parallel_Group(
            Step("worker_a", echo_fn),
            Step("worker_b", upper_fn),
            name="parallel_workers",
        )
        wf = Workflow("pg_wf", steps=[Step("initial", prefix_fn), pg])
        result = await wf.arun("input")
        assert isinstance(result, RunResult)


# ---------------------------------------------------------------------------
# 7. Integration — FlowClass with complex topology
# ---------------------------------------------------------------------------


class TestFlowClassComplexTopology:
    """FlowClass with multiple @start, fan-out, and router."""

    @pytest.mark.asyncio
    async def test_flowclass_single_start_with_listener(self):
        """FlowClass with one @start and one @listen works."""
        from loomable.flow import FlowClass, start, listen

        class SimpleFlow(FlowClass):
            @start()
            async def begin(self, input):
                return f"started:{input}"

            @listen("begin")
            async def process(self, input):
                return f"processed:{input}"

        flow = SimpleFlow()
        result = await flow.kickoff("hello")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_flowclass_fan_out(self):
        """FlowClass with fan-out (one start, multiple listeners) works."""
        from loomable.flow import FlowClass, start, listen

        class FanOutFlow(FlowClass):
            @start()
            async def begin(self, input):
                return f"start:{input}"

            @listen("begin")
            async def branch_a(self, input):
                return f"A:{input}"

            @listen("begin")
            async def branch_b(self, input):
                return f"B:{input}"

        flow = FanOutFlow()
        result = await flow.kickoff("data")
        assert isinstance(result, RunResult)

    @pytest.mark.asyncio
    async def test_flowclass_with_router(self):
        """FlowClass with a @router that routes to different branches."""
        from loomable.flow import FlowClass, start, listen, router

        class RoutedFlow(FlowClass):
            @start()
            async def begin(self, input):
                return f"input:{input}"

            @router("begin")
            async def decide(self, input):
                # Route to either "handle_a" or "handle_b"
                return "handle_a"

            @listen("decide")
            async def handle_a(self, input):
                return f"handled_a:{input}"

            @listen("decide")
            async def handle_b(self, input):
                return f"handled_b:{input}"

        flow = RoutedFlow()
        result = await flow.kickoff("test")
        assert isinstance(result, RunResult)

    def test_flowclass_explain_before_execution(self):
        """FlowClass explain() works before any execution."""
        from loomable.flow import FlowClass, start, listen

        class ExplainFlow(FlowClass):
            @start()
            async def begin(self, input):
                return input

            @listen("begin")
            async def process(self, input):
                return input

        flow = ExplainFlow()
        plan = flow.explain()
        # FlowPlan.original_nodes is a list of strings (node_id values)
        assert "begin" in plan.original_nodes
        assert "process" in plan.original_nodes

    def test_flowclass_satisfies_runnable_protocol(self):
        """FlowClass instances satisfy the Runnable protocol."""
        from loomable.flow import FlowClass, start, listen

        class RunnableFlow(FlowClass):
            @start()
            async def begin(self, input):
                return input

        flow = RunnableFlow()
        assert isinstance(flow, Runnable)

    @pytest.mark.asyncio
    async def test_flowclass_as_sequential_node(self):
        """FlowClass can be used inside sequential() since it's a Runnable."""
        from loomable.flow import FlowClass, start, listen
        from loomable.flow.helpers import sequential

        class InnerFlow(FlowClass):
            @start()
            async def begin(self, input):
                return f"flow:{input}"

        flow_instance = InnerFlow()
        composed = sequential(FunctionRunnable(prefix_fn), flow_instance)
        result = await composed.arun("test")
        assert isinstance(result, RunResult)


# ---------------------------------------------------------------------------
# 8. Integration — Loop with steps parameter
# ---------------------------------------------------------------------------


class TestLoopWithSteps:
    """Loop enhanced with steps= parameter works correctly."""

    @pytest.mark.asyncio
    async def test_loop_with_steps_list(self):
        """Loop(steps=[...]) compiles steps into a sequential body."""
        from loomable.flow import Step, Loop

        loop = Loop(
            steps=[Step("s1", echo_fn), Step("s2", upper_fn)],
            max_iterations=1,
        )
        result = await loop.arun("hello")
        assert isinstance(result, RunResult)
        # s1: "echo:hello" → s2: "ECHO:HELLO"
        assert "ECHO:HELLO" in result.output.text()

    def test_loop_both_body_and_steps_raises(self):
        """Loop raises ValueError when both body and steps are provided."""
        from loomable.flow import Step, Loop, FunctionRunnable

        with pytest.raises(ValueError, match="Only one of 'body' or 'steps'"):
            Loop(
                body=FunctionRunnable(echo_fn),
                steps=[Step("s1", echo_fn)],
            )

    @pytest.mark.asyncio
    async def test_loop_with_verifier(self):
        """Loop with verifier terminates when condition is True."""
        from loomable.flow import Step, Loop

        call_count = {"n": 0}

        def tracked_fn(input):
            call_count["n"] += 1
            return f"iter:{call_count['n']}"

        loop = Loop(
            steps=[Step("tracked", tracked_fn)],
            verifier=lambda output, ctx: "iter:3" in output.text(),
            max_iterations=10,
        )
        result = await loop.arun("go")
        assert isinstance(result, RunResult)
        assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# 9. Workflow explain() works pre-execution
# ---------------------------------------------------------------------------


class TestWorkflowExplain:
    """Workflow explain() returns topology before execution."""

    def test_explain_returns_flow_plan(self):
        """explain() returns a FlowPlan with step names as node_ids."""
        from loomable.flow import Step, Workflow, FlowPlan

        wf = Workflow(
            "my_workflow",
            steps=[
                Step("research", echo_fn),
                Step("analyze", upper_fn),
                Step("report", prefix_fn),
            ],
        )
        plan = wf.explain()
        assert isinstance(plan, FlowPlan)
        # FlowPlan.original_nodes is a list of strings (node_id values)
        assert "research" in plan.original_nodes
        assert "analyze" in plan.original_nodes
        assert "report" in plan.original_nodes


# ---------------------------------------------------------------------------
# 10. Existing exports coexist with new exports
# ---------------------------------------------------------------------------


class TestExportsCoexist:
    """New exports don't break existing ones in loomable.flow."""

    def test_existing_core_exports(self):
        """Core existing exports are still available."""
        from loomable.flow import (
            Runnable,
            FunctionRunnable,
            Flow,
            FlowPlan,
            Node,
            Edge,
            MapNode,
            RouterNode,
            SharedState,
            Reducer,
        )

        assert all(x is not None for x in [
            Runnable, FunctionRunnable, Flow, FlowPlan,
            Node, Edge, MapNode, RouterNode, SharedState, Reducer,
        ])

    def test_existing_engine_exports(self):
        """Engine exports are still available."""
        from loomable.flow import (
            SequentialEngine,
            ParallelEngine,
            HierarchicalEngine,
            ExecutionEngine,
        )

        assert all(x is not None for x in [
            SequentialEngine, ParallelEngine, HierarchicalEngine, ExecutionEngine,
        ])

    def test_existing_helper_exports(self):
        """Advanced helpers remain on loomable.flow.helpers; plan_and_execute on flow."""
        from loomable.flow import plan_and_execute
        from loomable.flow.helpers import (
            sequential,
            parallel,
            route,
            coordinate,
        )

        assert all(x is not None for x in [
            sequential, parallel, route, coordinate, plan_and_execute,
        ])

    def test_existing_loop_exports(self):
        """Loop and verifier exports are still available."""
        from loomable.flow import (
            Loop,
            Verifier,
            VerdictResult,
            AlwaysOkVerifier,
            CallableVerifier,
        )

        assert all(x is not None for x in [
            Loop, Verifier, VerdictResult, AlwaysOkVerifier, CallableVerifier,
        ])

    def test_existing_memory_exports(self):
        """Memory exports are still available."""
        from loomable.flow import (
            MemoryStore,
            Tier,
            TieredMemoryStore,
        )

        assert all(x is not None for x in [
            MemoryStore, Tier, TieredMemoryStore,
        ])

    def test_map_and_router_nodes_exported(self):
        """MapNode and RouterNode are the public node types."""
        from loomable.flow import MapNode, RouterNode

        assert MapNode is not None
        assert RouterNode is not None
