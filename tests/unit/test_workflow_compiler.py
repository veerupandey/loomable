"""Unit tests for WorkflowCompiler.

Verifies that the compiler correctly translates step lists into Flow graphs
with proper nodes and edges.
"""

import asyncio

import pytest

from loomable.flow.compiler import WorkflowCompiler
from loomable.flow.condition import Condition
from loomable.flow.flow import Flow
from loomable.flow.loop import Loop
from loomable.flow.nodes import Edge, Node, RouterNode
from loomable.flow.parallel_group import Parallel_Group
from loomable.flow.runnable import FunctionRunnable
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Helper functions / agents
# ---------------------------------------------------------------------------


async def _echo(input, *, context=None):
    """Simple echo agent for testing."""
    from loomable.agent.run import RunResult
    from loomable.content import AgentOutput, MediaPart, Modality

    text = str(input)
    output = AgentOutput(
        parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=text.encode())]
    )
    return RunResult(output=output, session_id="")


async def _uppercase(input, *, context=None):
    """Uppercase agent for testing."""
    from loomable.agent.run import RunResult
    from loomable.content import AgentOutput, MediaPart, Modality

    text = str(input).upper()
    output = AgentOutput(
        parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=text.encode())]
    )
    return RunResult(output=output, session_id="")


# ---------------------------------------------------------------------------
# Tests: Basic Step compilation
# ---------------------------------------------------------------------------


class TestStepCompilation:
    """Test that Steps are compiled into Nodes with correct node_ids."""

    def test_single_step_creates_one_node(self):
        """A single Step compiles to a Flow with one node."""
        step = Step("greet", _echo)
        flow = WorkflowCompiler.compile([step], name="test_workflow")

        assert isinstance(flow, Flow)
        assert "greet" in flow.nodes
        assert len(flow.nodes) == 1
        assert len(flow.edges) == 0

    def test_two_steps_connected_by_edge(self):
        """Two sequential Steps produce one edge connecting them."""
        step_a = Step("fetch", _echo)
        step_b = Step("process", _uppercase)
        flow = WorkflowCompiler.compile([step_a, step_b], name="test_workflow")

        assert "fetch" in flow.nodes
        assert "process" in flow.nodes
        assert len(flow.nodes) == 2
        assert len(flow.edges) == 1

        edge = flow.edges[0]
        assert edge.source == "fetch"
        assert edge.target == "process"

    def test_three_steps_produce_two_edges_in_order(self):
        """Three sequential Steps produce two edges in declaration order."""
        steps = [
            Step("a", _echo),
            Step("b", _echo),
            Step("c", _echo),
        ]
        flow = WorkflowCompiler.compile(steps, name="test_workflow")

        assert len(flow.nodes) == 3
        assert len(flow.edges) == 2

        assert flow.edges[0].source == "a"
        assert flow.edges[0].target == "b"
        assert flow.edges[1].source == "b"
        assert flow.edges[1].target == "c"

    def test_step_name_used_as_node_id(self):
        """Step's name attribute is used as the node_id in the compiled Flow."""
        step = Step("my_custom_name", _echo)
        flow = WorkflowCompiler.compile([step], name="test_workflow")

        assert "my_custom_name" in flow.nodes


# ---------------------------------------------------------------------------
# Tests: Parallel_Group compilation
# ---------------------------------------------------------------------------


class TestParallelGroupCompilation:
    """Test that Parallel_Groups compile as single nodes."""

    def test_parallel_group_becomes_single_node(self):
        """A Parallel_Group compiles to a single node in the flow."""
        step_a = Step("research", _echo)
        step_b = Step("analysis", _echo)
        pg = Parallel_Group(step_a, step_b, name="parallel_work")

        flow = WorkflowCompiler.compile([pg], name="test_workflow")

        assert "parallel_work" in flow.nodes
        assert len(flow.nodes) == 1

    def test_parallel_group_between_steps(self):
        """A Parallel_Group between Steps is connected with edges."""
        step_before = Step("prepare", _echo)
        pg = Parallel_Group(Step("r1", _echo), Step("r2", _echo), name="parallel_phase")
        step_after = Step("summarize", _echo)

        flow = WorkflowCompiler.compile(
            [step_before, pg, step_after], name="test_workflow"
        )

        assert len(flow.nodes) == 3
        assert len(flow.edges) == 2

        # prepare -> parallel_phase -> summarize
        assert flow.edges[0].source == "prepare"
        assert flow.edges[0].target == "parallel_phase"
        assert flow.edges[1].source == "parallel_phase"
        assert flow.edges[1].target == "summarize"


# ---------------------------------------------------------------------------
# Tests: Condition compilation
# ---------------------------------------------------------------------------


class TestConditionCompilation:
    """Test that Conditions compile to RouterNode + branch nodes + join."""

    def test_condition_creates_router_and_branches(self):
        """A Condition compiles to a router node, branch nodes, and a join node."""
        cond = Condition(
            condition=lambda state: True,
            then_steps=[Step("do_work", _echo)],
            else_steps=[Step("skip", _echo)],
        )

        flow = WorkflowCompiler.compile([cond], name="test_workflow")

        # Should have: router, then branch, else branch, join = 4 nodes
        assert len(flow.nodes) == 4

        # Router node should exist
        router_id = "_condition_0_router"
        assert router_id in flow.nodes

    def test_condition_without_else(self):
        """A Condition without else_steps still compiles correctly."""
        cond = Condition(
            condition=lambda state: True,
            then_steps=[Step("do_work", _echo)],
        )

        flow = WorkflowCompiler.compile([cond], name="test_workflow")

        # Should have: router, then branch, join = 3 nodes
        assert len(flow.nodes) == 3

    def test_condition_between_steps_connected(self):
        """A Condition between Steps is properly connected with edges."""
        step_before = Step("prep", _echo)
        cond = Condition(
            condition=lambda state: True,
            then_steps=[Step("branch_a", _echo)],
            else_steps=[Step("branch_b", _echo)],
        )
        step_after = Step("finish", _echo)

        flow = WorkflowCompiler.compile(
            [step_before, cond, step_after], name="test_workflow"
        )

        # Nodes: prep, router, then, else, join, finish = 6
        assert len(flow.nodes) == 6

        # Check sequential connection edges exist
        edge_pairs = [(e.source, e.target) for e in flow.edges]

        # prep -> router (sequential)
        assert ("prep", "_condition_1_router") in edge_pairs
        # join -> finish (sequential)
        assert ("_condition_1_join", "finish") in edge_pairs


# ---------------------------------------------------------------------------
# Tests: Loop compilation
# ---------------------------------------------------------------------------


class TestLoopCompilation:
    """Test that Loops compile as single nodes."""

    def test_loop_becomes_single_node(self):
        """A Loop compiles to a single node in the flow."""
        loop = Loop(body=FunctionRunnable(_echo), max_iterations=3)
        flow = WorkflowCompiler.compile([loop], name="test_workflow")

        assert len(flow.nodes) == 1
        # Loop nodes get a generated name
        node_id = list(flow.nodes.keys())[0]
        assert node_id.startswith("_loop_")

    def test_loop_between_steps(self):
        """A Loop between Steps is connected with edges."""
        step_a = Step("start", _echo)
        loop = Loop(body=FunctionRunnable(_echo), max_iterations=2)
        step_b = Step("end", _echo)

        flow = WorkflowCompiler.compile([step_a, loop, step_b], name="test_workflow")

        assert len(flow.nodes) == 3
        assert len(flow.edges) == 2

        # start -> _loop_1 -> end
        assert flow.edges[0].source == "start"
        assert flow.edges[0].target == "_loop_1"
        assert flow.edges[1].source == "_loop_1"
        assert flow.edges[1].target == "end"


# ---------------------------------------------------------------------------
# Tests: Nested Workflow compilation (mock since Workflow doesn't exist yet)
# ---------------------------------------------------------------------------


class TestNestedWorkflowCompilation:
    """Test that nested Runnables with a name are treated as single nodes."""

    def test_named_runnable_as_single_node(self):
        """Any named Runnable compiles to a single node using its name."""

        class FakeWorkflow:
            """Simulate a nested Workflow with a name and arun."""

            def __init__(self, name: str):
                self.name = name

            async def arun(self, input, *, context=None):
                pass

        nested = FakeWorkflow("sub_workflow")
        step_a = Step("before", _echo)
        step_b = Step("after", _echo)

        flow = WorkflowCompiler.compile(
            [step_a, nested, step_b], name="test_workflow"
        )

        assert "sub_workflow" in flow.nodes
        assert len(flow.nodes) == 3
        assert len(flow.edges) == 2

        assert flow.edges[0].source == "before"
        assert flow.edges[0].target == "sub_workflow"
        assert flow.edges[1].source == "sub_workflow"
        assert flow.edges[1].target == "after"


# ---------------------------------------------------------------------------
# Tests: Parameters passthrough
# ---------------------------------------------------------------------------


class TestCompilerParameters:
    """Test that deps, memory, session_id, and reducers are passed to the Flow."""

    def test_deps_passed_to_flow(self):
        """The deps parameter is passed through to the compiled Flow."""
        step = Step("work", _echo)
        deps = {"api_key": "test"}
        flow = WorkflowCompiler.compile([step], name="test", deps=deps)

        # Flow stores deps internally
        assert flow._deps == deps

    def test_memory_passed_to_flow(self):
        """The memory parameter is passed through to the compiled Flow."""
        from loomable.flow.memory import TieredMemoryStore

        step = Step("work", _echo)
        mem = TieredMemoryStore(session_id="test_session")
        flow = WorkflowCompiler.compile([step], name="test", memory=mem)

        assert flow._memory == mem

    def test_session_id_passed_to_flow(self):
        """The session_id parameter is passed through to the compiled Flow."""
        step = Step("work", _echo)
        flow = WorkflowCompiler.compile([step], name="test", session_id="sess_123")

        assert flow._session_id == "sess_123"

    def test_reducers_passed_to_flow(self):
        """The reducers parameter is passed through to the compiled Flow."""
        from loomable.flow.state import append

        step = Step("work", _echo)
        reducers = {"history": append}
        flow = WorkflowCompiler.compile([step], name="test", reducers=reducers)

        assert flow._reducers == reducers


# ---------------------------------------------------------------------------
# Tests: Mixed composition
# ---------------------------------------------------------------------------


class TestMixedComposition:
    """Test compilation with multiple element types in sequence."""

    def test_step_parallel_condition_loop_sequence(self):
        """All element types can be mixed in a single steps list."""
        step = Step("init", _echo)
        pg = Parallel_Group(Step("p1", _echo), Step("p2", _echo), name="parallel")
        cond = Condition(
            condition=lambda state: True,
            then_steps=[Step("yes", _echo)],
        )
        loop = Loop(body=FunctionRunnable(_echo), max_iterations=2)
        final = Step("done", _echo)

        flow = WorkflowCompiler.compile(
            [step, pg, cond, loop, final], name="test_workflow"
        )

        # All elements should produce nodes
        assert "init" in flow.nodes
        assert "parallel" in flow.nodes
        assert "done" in flow.nodes

        # Flow should be constructable without errors
        assert isinstance(flow, Flow)
