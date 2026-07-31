"""Tests for FlowClassCompiler (Task 8.2).

Validates:
- Req 6.2: Compilation of decorated methods into Flow graph
- Req 6.3: @start() methods become entry-point nodes
- Req 6.4: @listen(source) creates edges
- Req 6.5: @router(source) creates RouterNode
- Req 6.8: Multiple listeners on same source → parallel fan-out
- Req 6.9: Invalid source references raise FlowConfigError
- Req 10.3: Cycle detection raises FlowConfigError
- Req 10.4: Missing @start raises FlowConfigError
"""

from __future__ import annotations

import pytest

from loomable.flow.compiler import FlowClassCompiler
from loomable.flow.flow_class import listen, router, start, _StartMeta, _ListenMeta, _RouterMeta
from loomable.flow.nodes import FlowConfigError, RouterNode, Node


# ---------------------------------------------------------------------------
# Helper: Simple FlowClass-like objects for testing the compiler
# ---------------------------------------------------------------------------


class SimpleLinearFlow:
    """start → process → finalize"""

    @start()
    async def begin(self, input):
        return f"started:{input}"

    @listen("begin")
    async def process(self, input):
        return f"processed:{input}"

    @listen("process")
    async def finalize(self, input):
        return f"final:{input}"


class FanOutFlow:
    """One source with multiple listeners (parallel fan-out)."""

    @start()
    async def begin(self, input):
        return input

    @listen("begin")
    async def branch_a(self, input):
        return f"a:{input}"

    @listen("begin")
    async def branch_b(self, input):
        return f"b:{input}"

    @listen("begin")
    async def branch_c(self, input):
        return f"c:{input}"


class RouterFlow:
    """A flow with a router that routes based on return value."""

    @start()
    async def begin(self, input):
        return input

    @router("begin")
    async def route_decision(self, input):
        if input == "fast":
            return "fast_path"
        return "slow_path"

    @listen("route_decision")
    async def fast_path(self, input):
        return f"fast:{input}"

    @listen("route_decision")
    async def slow_path(self, input):
        return f"slow:{input}"


class NoStartFlow:
    """A flow with no @start method — should fail validation."""

    @listen("nonexistent")
    async def process(self, input):
        return input


class InvalidSourceFlow:
    """A flow with @listen referencing a non-existent method."""

    @start()
    async def begin(self, input):
        return input

    @listen("does_not_exist")
    async def process(self, input):
        return input


class CyclicFlow:
    """A flow with a cycle: A → B → C → A."""

    @start()
    async def entry(self, input):
        return input

    @listen("entry")
    async def step_a(self, input):
        return input

    @listen("step_a")
    async def step_b(self, input):
        return input

    @listen("step_b")
    async def step_a_again(self, input):
        # This creates a reference back, but the cycle requires
        # step_a_again -> step_a. Let's use a direct cycle.
        return input


class DirectCycleFlow:
    """A flow with a direct cycle: a → b → a."""

    @start()
    async def begin(self, input):
        return input

    @listen("begin")
    async def step_a(self, input):
        return input

    @listen("step_a")
    async def step_b(self, input):
        return input

    @listen("step_b")
    async def step_a(self, input):  # noqa: F811
        # This won't work with class method redefinition.
        # We need to approach this differently.
        return input


class MultiStartFlow:
    """A flow with multiple @start methods."""

    @start()
    async def begin_one(self, input):
        return f"one:{input}"

    @start()
    async def begin_two(self, input):
        return f"two:{input}"

    @listen("begin_one")
    async def merge(self, input):
        return f"merged:{input}"


class InvalidRouterSourceFlow:
    """A flow with @router referencing a non-existent method."""

    @start()
    async def begin(self, input):
        return input

    @router("ghost_method")
    async def route(self, input):
        return "somewhere"


# ---------------------------------------------------------------------------
# Tests: Simple linear chain
# ---------------------------------------------------------------------------


class TestLinearChainCompilation:
    """A simple linear chain: start → listen → listen."""

    def test_compiles_to_flow(self):
        """Compiler returns a Flow object."""
        instance = SimpleLinearFlow()
        flow = FlowClassCompiler.compile(instance)
        assert flow is not None

    def test_contains_all_nodes(self):
        """All decorated methods become nodes in the compiled Flow."""
        instance = SimpleLinearFlow()
        flow = FlowClassCompiler.compile(instance)
        node_ids = set(flow.nodes.keys())
        assert "begin" in node_ids
        assert "process" in node_ids
        assert "finalize" in node_ids

    def test_edges_connect_in_order(self):
        """Edges follow the listen chain: begin→process, process→finalize."""
        instance = SimpleLinearFlow()
        flow = FlowClassCompiler.compile(instance)
        edge_tuples = [(e.source, e.target) for e in flow.edges]
        assert ("begin", "process") in edge_tuples
        assert ("process", "finalize") in edge_tuples

    def test_node_count(self):
        """Three decorated methods → three nodes."""
        instance = SimpleLinearFlow()
        flow = FlowClassCompiler.compile(instance)
        assert len(flow.nodes) == 3

    def test_edge_count(self):
        """Two @listen decorators → two edges."""
        instance = SimpleLinearFlow()
        flow = FlowClassCompiler.compile(instance)
        assert len(flow.edges) == 2


# ---------------------------------------------------------------------------
# Tests: Fan-out (multiple listeners on same source)
# ---------------------------------------------------------------------------


class TestFanOutCompilation:
    """One source with multiple listeners → parallel fan-out."""

    def test_all_branches_have_nodes(self):
        """Each listener method becomes a node."""
        instance = FanOutFlow()
        flow = FlowClassCompiler.compile(instance)
        node_ids = set(flow.nodes.keys())
        assert "begin" in node_ids
        assert "branch_a" in node_ids
        assert "branch_b" in node_ids
        assert "branch_c" in node_ids

    def test_multiple_edges_from_source(self):
        """Multiple edges originate from the same source (fan-out)."""
        instance = FanOutFlow()
        flow = FlowClassCompiler.compile(instance)
        begin_edges = [e for e in flow.edges if e.source == "begin"]
        assert len(begin_edges) == 3

    def test_fan_out_targets(self):
        """Fan-out edges target all listener methods."""
        instance = FanOutFlow()
        flow = FlowClassCompiler.compile(instance)
        begin_targets = {e.target for e in flow.edges if e.source == "begin"}
        assert begin_targets == {"branch_a", "branch_b", "branch_c"}


# ---------------------------------------------------------------------------
# Tests: Router
# ---------------------------------------------------------------------------


class TestRouterCompilation:
    """A router method creates a RouterNode in the compiled Flow."""

    def test_router_node_created(self):
        """The router method becomes a node whose runnable is a RouterNode."""
        instance = RouterFlow()
        flow = FlowClassCompiler.compile(instance)
        route_node = flow.nodes.get("route_decision")
        assert route_node is not None
        # Flow wraps RouterNode in a Node; the inner runnable is the RouterNode
        assert isinstance(route_node.runnable, RouterNode)

    def test_router_has_choices(self):
        """The RouterNode has downstream methods as choices."""
        instance = RouterFlow()
        flow = FlowClassCompiler.compile(instance)
        route_node = flow.nodes["route_decision"]
        router_inner = route_node.runnable
        assert isinstance(router_inner, RouterNode)
        assert "fast_path" in router_inner.choices
        assert "slow_path" in router_inner.choices

    def test_router_edge_from_source(self):
        """An edge connects from router's source to the router node."""
        instance = RouterFlow()
        flow = FlowClassCompiler.compile(instance)
        edge_tuples = [(e.source, e.target) for e in flow.edges]
        assert ("begin", "route_decision") in edge_tuples

    def test_downstream_listener_edges(self):
        """Edges from router to downstream listener nodes exist."""
        instance = RouterFlow()
        flow = FlowClassCompiler.compile(instance)
        edge_tuples = [(e.source, e.target) for e in flow.edges]
        assert ("route_decision", "fast_path") in edge_tuples
        assert ("route_decision", "slow_path") in edge_tuples


# ---------------------------------------------------------------------------
# Tests: Missing @start validation
# ---------------------------------------------------------------------------


class TestMissingStartValidation:
    """FlowConfigError raised when no @start method exists."""

    def test_raises_flow_config_error(self):
        """Compiling a flow with no @start raises FlowConfigError."""
        instance = NoStartFlow()
        with pytest.raises(FlowConfigError) as exc_info:
            FlowClassCompiler.compile(instance)
        assert "At least one @start method is required" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: Invalid source reference validation
# ---------------------------------------------------------------------------


class TestInvalidSourceValidation:
    """FlowConfigError raised when @listen references unknown source."""

    def test_listen_invalid_source(self):
        """@listen referencing non-existent method raises FlowConfigError."""
        instance = InvalidSourceFlow()
        with pytest.raises(FlowConfigError) as exc_info:
            FlowClassCompiler.compile(instance)
        assert "@listen/@router references unknown source: 'does_not_exist'" in str(
            exc_info.value
        )

    def test_router_invalid_source(self):
        """@router referencing non-existent method raises FlowConfigError."""
        instance = InvalidRouterSourceFlow()
        with pytest.raises(FlowConfigError) as exc_info:
            FlowClassCompiler.compile(instance)
        assert "@listen/@router references unknown source: 'ghost_method'" in str(
            exc_info.value
        )


# ---------------------------------------------------------------------------
# Tests: Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    """FlowConfigError raised when cycles exist in the listener graph."""

    def test_self_referencing_cycle(self):
        """A method listening to itself is a cycle."""

        class SelfCycle:
            @start()
            async def begin(self, input):
                return input

            @listen("begin")
            async def step_a(self, input):
                return input

            @listen("step_a")
            async def step_b(self, input):
                return input

            @listen("step_b")
            async def step_a(self, input):  # noqa: F811
                return input

        # Python class definition won't allow duplicate method names,
        # so we need to construct the cycle differently.
        # Instead, we'll create a mock-like class with _flow_meta set manually.

        class CycleByMeta:
            pass

        instance = CycleByMeta()

        # Manually create methods with _flow_meta
        async def begin(self_ref, input):
            return input
        begin._flow_meta = _StartMeta()

        async def step_a(self_ref, input):
            return input
        step_a._flow_meta = _ListenMeta(source="begin")

        async def step_b(self_ref, input):
            return input
        step_b._flow_meta = _ListenMeta(source="step_a")

        async def step_c(self_ref, input):
            return input
        step_c._flow_meta = _ListenMeta(source="step_b")

        # Now create the cycle: step_a also listens to step_c
        # But we can't have two methods with same name. Let's make
        # step_c listen to step_b, and a different method "back_to_a" listen to step_c
        # with step_a listening to "back_to_a" — but that's complex.
        # Simpler: make step_c point back to step_a.
        # Actually the cycle is: step_a -> step_b -> step_c -> step_a
        # This means step_a is a listener of step_c, creating a cycle.
        # But step_a already listens to "begin". We can't have dual metadata.

        # Let's use a fresh approach: build a simpler cycle.
        class SimpleCycleFlow:
            @start()
            async def begin(self, input):
                return input

            @listen("begin")
            async def alpha(self, input):
                return input

            @listen("alpha")
            async def beta(self, input):
                return input

            @listen("beta")
            async def gamma(self, input):
                return input

        # This is NOT a cycle (it's a chain: begin → alpha → beta → gamma).
        # For a real cycle, we need gamma to point back to alpha.
        # We'll monkey-patch the metadata.
        instance2 = SimpleCycleFlow()
        # Get the raw function and change its source to create a cycle
        # gamma's meta says source="beta", let's also add alpha listening to gamma.
        # But we can't add two @listen on one method.
        # Instead, let's do it by creating a class with a proper cycle:

        class RealCycleFlow:
            """alpha → beta → alpha (cycle)."""
            pass

        real_instance = RealCycleFlow()

        # Attach methods manually
        async def m_begin(input):
            return input
        m_begin._flow_meta = _StartMeta()
        real_instance.begin = m_begin

        async def m_alpha(input):
            return input
        m_alpha._flow_meta = _ListenMeta(source="begin")
        real_instance.alpha = m_alpha

        async def m_beta(input):
            return input
        m_beta._flow_meta = _ListenMeta(source="alpha")
        real_instance.beta = m_beta

        async def m_gamma(input):
            return input
        m_gamma._flow_meta = _ListenMeta(source="beta")
        real_instance.gamma = m_gamma

        # Now make alpha listen to gamma to create the cycle
        # But alpha already listens to begin. We need a method that listens to gamma
        # and creates a back-edge to alpha.
        # Actually, for a cycle: alpha → beta → gamma → alpha
        # gamma listens to beta (done), and something listens to gamma pointing to alpha.
        # The simplest approach: make gamma's source be "alpha" (gamma listens to alpha),
        # and alpha listens to gamma — but we'd need two methods or change the source.

        # Let's just set gamma's source to point back, creating: begin → alpha → beta → alpha
        # Meaning beta listens to alpha, and alpha also listens to beta => cycle.
        # Simplest: alpha listens to beta, beta listens to alpha.

        class ProperCycleFlow:
            pass

        cycle_instance = ProperCycleFlow()

        async def c_begin(input):
            return input
        c_begin._flow_meta = _StartMeta()
        cycle_instance.begin = c_begin

        async def c_alpha(input):
            return input
        c_alpha._flow_meta = _ListenMeta(source="beta")
        cycle_instance.alpha = c_alpha

        async def c_beta(input):
            return input
        c_beta._flow_meta = _ListenMeta(source="alpha")
        cycle_instance.beta = c_beta

        with pytest.raises(FlowConfigError) as exc_info:
            FlowClassCompiler.compile(cycle_instance)
        assert "Cycle detected in listener graph" in str(exc_info.value)

    def test_longer_cycle(self):
        """A longer cycle (A → B → C → A) is detected."""

        class LongerCycle:
            pass

        instance = LongerCycle()

        async def m_start(input):
            return input
        m_start._flow_meta = _StartMeta()
        instance.begin = m_start

        async def m_a(input):
            return input
        m_a._flow_meta = _ListenMeta(source="begin")
        instance.step_a = m_a

        async def m_b(input):
            return input
        m_b._flow_meta = _ListenMeta(source="step_a")
        instance.step_b = m_b

        async def m_c(input):
            return input
        m_c._flow_meta = _ListenMeta(source="step_b")
        instance.step_c = m_c

        # Create the back-edge: step_a listens to step_c → cycle
        # But step_a already listens to begin. We need another method.
        async def m_back(input):
            return input
        m_back._flow_meta = _ListenMeta(source="step_c")
        instance.step_a_back = m_back

        # This isn't actually a cycle because step_a_back doesn't feed into step_a.
        # For a real cycle: step_c → step_a (but step_a listens to begin, not step_c)
        # We need: begin → A → B → C → A (C points to A)
        # So let's make A listen to BOTH begin and C? No, one method = one meta.
        # The simplest true cycle: B listens to C, C listens to B.

        class TrueLongerCycle:
            pass

        inst = TrueLongerCycle()

        async def s(input):
            return input
        s._flow_meta = _StartMeta()
        inst.begin = s

        async def a(input):
            return input
        a._flow_meta = _ListenMeta(source="begin")
        inst.step_a = a

        async def b(input):
            return input
        b._flow_meta = _ListenMeta(source="step_c")
        inst.step_b = b

        async def c(input):
            return input
        c._flow_meta = _ListenMeta(source="step_a")
        inst.step_c = c

        # Cycle: step_a → step_c → step_b → step_c (no, step_b listens to step_c,
        # step_c listens to step_a — no cycle: begin → step_a → step_c → step_b (chain)
        # Actually: step_c source is step_a, step_b source is step_c.
        # Graph: begin → step_a, step_a → step_c, step_c → step_b. This is a chain, not cycle.

        # For a real cycle: step_b → step_a and step_a → step_b
        class RealLongerCycle:
            pass

        inst2 = RealLongerCycle()

        async def s2(input):
            return input
        s2._flow_meta = _StartMeta()
        inst2.begin = s2

        async def a2(input):
            return input
        a2._flow_meta = _ListenMeta(source="begin")
        inst2.node_a = a2

        async def b2(input):
            return input
        b2._flow_meta = _ListenMeta(source="node_a")
        inst2.node_b = b2

        async def c2(input):
            return input
        c2._flow_meta = _ListenMeta(source="node_b")
        inst2.node_c = c2

        async def d2(input):
            return input
        d2._flow_meta = _ListenMeta(source="node_c")
        inst2.node_a = d2  # This overwrites node_a, making it listen to node_c.
        # Now: begin → (nothing targets node_a since we overwrote)
        # Actually after overwrite: node_a listens to node_c
        # node_b listens to node_a, node_c listens to node_b
        # So: node_a → node_b → node_c → node_a => CYCLE!

        with pytest.raises(FlowConfigError) as exc_info:
            FlowClassCompiler.compile(inst2)
        assert "Cycle detected in listener graph" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: Multiple @start methods
# ---------------------------------------------------------------------------


class TestMultipleStartMethods:
    """Multiple @start methods create multiple entry-point nodes."""

    def test_both_start_methods_become_nodes(self):
        """Both @start methods are included as nodes."""
        instance = MultiStartFlow()
        flow = FlowClassCompiler.compile(instance)
        node_ids = set(flow.nodes.keys())
        assert "begin_one" in node_ids
        assert "begin_two" in node_ids

    def test_listener_edges_correct(self):
        """The merge method listens to begin_one."""
        instance = MultiStartFlow()
        flow = FlowClassCompiler.compile(instance)
        edge_tuples = [(e.source, e.target) for e in flow.edges]
        assert ("begin_one", "merge") in edge_tuples


# ---------------------------------------------------------------------------
# Tests: Node types are correct
# ---------------------------------------------------------------------------


class TestNodeTypes:
    """Verify correct node types are created."""

    def test_start_method_is_regular_node(self):
        """@start methods become regular Node instances."""
        instance = SimpleLinearFlow()
        flow = FlowClassCompiler.compile(instance)
        begin_node = flow.nodes["begin"]
        assert isinstance(begin_node, Node)

    def test_listen_method_is_regular_node(self):
        """@listen methods become regular Node instances."""
        instance = SimpleLinearFlow()
        flow = FlowClassCompiler.compile(instance)
        process_node = flow.nodes["process"]
        assert isinstance(process_node, Node)

    def test_router_method_is_router_node(self):
        """@router methods become nodes wrapping a RouterNode."""
        instance = RouterFlow()
        flow = FlowClassCompiler.compile(instance)
        route_node = flow.nodes["route_decision"]
        assert isinstance(route_node, Node)
        assert isinstance(route_node.runnable, RouterNode)
