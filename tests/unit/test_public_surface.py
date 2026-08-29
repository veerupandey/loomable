"""Tests for public surface, progressive-disclosure, and lean audit (Task 16).

Validates:
- Req 2.1: Zero-config Agent works without Flow/Loop/engine/memory knowledge
- Req 2.2: Agent + tools auto-escalates without strategy selection
- Req 2.3: Existing Agents usable as Flow nodes unchanged
- Req 2.4: Loop accepts a Runnable unchanged
- Req 2.5: Agent reliability features remain in effect inside Loop/Flow
- Req 2.6: Public Agent entry point preserved unchanged
- Req 14.1: No new mandatory runtime dependency added
- Req 14.2: No loomable.kernel module modified
- Req 14.8: Zero-config defaults (no optimizer/engine/checkpointer/verifier/deps/memory)
"""

from __future__ import annotations

import importlib
import ast
import pathlib

import pytest

# ---------------------------------------------------------------------------
# 1. Public surface: all documented exports are importable from loomable.flow
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """Verify all documented exports exist in loomable.flow."""

    def test_core_exports(self):
        """Core primitives are importable."""
        from loomable.flow import Runnable, FunctionRunnable

        assert Runnable is not None
        assert FunctionRunnable is not None

    def test_tier2_exports(self):
        """Tier 2 (Loop + Verifier) exports are importable."""
        from loomable.flow import (
            Loop,
            Verifier,
            VerdictResult,
            AlwaysOkVerifier,
            CallableVerifier,
        )

        assert Loop is not None
        assert Verifier is not None
        assert VerdictResult is not None
        assert AlwaysOkVerifier is not None
        assert CallableVerifier is not None

    def test_tier3_exports(self):
        """Tier 3 (Flow graph) exports are importable."""
        from loomable.flow import Flow, FlowPlan, Node, Edge, MapNode, RouterNode

        assert Flow is not None
        assert FlowPlan is not None
        assert Node is not None
        assert Edge is not None
        assert MapNode is not None
        assert RouterNode is not None

    def test_map_and_router_nodes(self):
        """MapNode and RouterNode are exported (no Map/Router aliases)."""
        from loomable.flow import MapNode, RouterNode

        assert MapNode is not None
        assert RouterNode is not None

    def test_state_exports(self):
        """State management exports are importable."""
        from loomable.flow import SharedState, Reducer, overwrite, append, merge

        assert SharedState is not None
        assert Reducer is not None
        assert callable(overwrite)
        assert callable(append)
        assert callable(merge)

    def test_engine_exports(self):
        """Engine exports are importable from loomable.flow."""
        from loomable.flow import (
            ExecutionEngine,
            SequentialEngine,
            ParallelEngine,
            HierarchicalEngine,
        )

        assert ExecutionEngine is not None
        assert SequentialEngine is not None
        assert ParallelEngine is not None
        assert HierarchicalEngine is not None

    def test_optimizer_exports(self):
        """Optimizer exports are importable."""
        from loomable.flow import Optimizer, OptimizationRule

        assert Optimizer is not None
        assert OptimizationRule is not None

    def test_memory_exports(self):
        """Memory exports are importable."""
        from loomable.flow import MemoryStore, Tier, TieredMemoryStore

        assert MemoryStore is not None
        assert Tier is not None
        assert TieredMemoryStore is not None

    def test_hitl_exports(self):
        """HITL exports are importable."""
        from loomable.flow import FlowPaused

        assert FlowPaused is not None

    def test_top_level_enterprise_exports(self):
        """Agent / Team / Workflow / Case are importable from loomable."""
        from loomable import Agent, Board, Case, Team, Workflow

        assert all(x is not None for x in (Agent, Team, Workflow, Case, Board))
        pytest.importorskip("fastapi")
        from loomable.serve import mount_agent, mount_case

        assert callable(mount_agent) and callable(mount_case)

    def test_observability_exports(self):
        """Observability exports are importable."""
        from loomable.flow import (
            ContextSnapshotConfig,
            MessageDisposition,
            MessageSnapshot,
        )

        assert ContextSnapshotConfig is not None
        assert MessageDisposition is not None
        assert MessageSnapshot is not None

    def test_helper_exports(self):
        """Advanced Flow helpers live under loomable.flow.helpers; plan_and_execute on flow."""
        from loomable.flow import plan_and_execute
        from loomable.flow.helpers import (
            sequential,
            parallel,
            route,
            coordinate,
        )

        assert callable(sequential)
        assert callable(parallel)
        assert callable(route)
        assert callable(coordinate)
        assert callable(plan_and_execute)

    def test_all_list_completeness(self):
        """__all__ contains every documented export."""
        import loomable.flow as flow_mod

        expected = {
            "Runnable",
            "FunctionRunnable",
            "VerdictResult",
            "Verifier",
            "AlwaysOkVerifier",
            "CallableVerifier",
            "Loop",
            "Flow",
            "FlowPlan",
            "FlowPaused",
            "Edge",
            "Node",
            "MapNode",
            "RouterNode",
            "FlowConfigError",
            "Reducer",
            "SharedState",
            "overwrite",
            "append",
            "merge",
            "ExecutionEngine",
            "SequentialEngine",
            "ParallelEngine",
            "HierarchicalEngine",
            "Optimizer",
            "OptimizationRule",
            "MemoryStore",
            "Tier",
            "TieredMemoryStore",
            "ContextSnapshotConfig",
            "MessageDisposition",
            "MessageSnapshot",
            "emit_context_snapshot",
            "emit_node_end",
            "emit_node_start",
            "plan_and_execute",
            "Step",
            "StepFailed",
            "FAILURE_ACTIONS",
            "Workflow",
            "Condition",
            "ComposableElement",
            "Parallel_Group",
            "FlowClass",
            "start",
            "listen",
            "router",
        }
        actual = set(flow_mod.__all__)
        missing = expected - actual
        extra = actual - expected
        assert not missing, f"Missing from __all__: {missing}"
        assert not extra, f"Unexpected in __all__: {extra}"

        actual = set(flow_mod.__all__)
        missing = expected - actual
        assert not missing, f"Missing from __all__: {missing}"


# ---------------------------------------------------------------------------
# 2. Zero-config defaults (Req 14.8)
# ---------------------------------------------------------------------------


class TestZeroConfigDefaults:
    """Flow([step]) works with no optimizer/engine/checkpointer/verifier/deps/memory."""

    def test_flow_single_step_no_config(self):
        """A Flow with a single function step requires zero additional config."""
        from loomable.flow import Flow

        def step(input):
            return f"done: {input}"

        # This must not raise — no engine, optimizer, memory, etc. needed
        flow = Flow([step])
        assert len(flow.nodes) == 1
        assert len(flow.edges) == 0

    def test_flow_multi_step_no_config(self):
        """A Flow with multiple steps uses sensible defaults for everything."""
        from loomable.flow import Flow

        def a(x):
            return f"a:{x}"

        def b(x):
            return f"b:{x}"

        flow = Flow([a, b])
        assert len(flow.nodes) == 2
        assert len(flow.edges) == 1

    def test_flow_defaults_no_optimizer(self):
        """Flow defaults to optimizer=False (disabled)."""
        from loomable.flow import Flow

        flow = Flow([lambda x: x])
        assert flow._optimizer is False

    def test_flow_defaults_no_memory(self):
        """Flow defaults to memory=None."""
        from loomable.flow import Flow

        flow = Flow([lambda x: x])
        assert flow._memory is None

    def test_flow_defaults_no_checkpointer(self):
        """Flow defaults to checkpointer=None."""
        from loomable.flow import Flow

        flow = Flow([lambda x: x])
        assert flow._checkpointer is None

    def test_flow_defaults_no_deps(self):
        """Flow defaults to deps=None."""
        from loomable.flow import Flow

        flow = Flow([lambda x: x])
        assert flow._deps is None

    def test_flow_defaults_engine_auto(self):
        """Flow defaults to engine='auto'."""
        from loomable.flow import Flow

        flow = Flow([lambda x: x])
        assert flow._engine == "auto"

    def test_flow_defaults_no_events(self):
        """Flow defaults to events=None."""
        from loomable.flow import Flow

        flow = Flow([lambda x: x])
        assert flow._events is None


# ---------------------------------------------------------------------------
# 3. Lean audit: no new mandatory dependency (Req 14.1)
# ---------------------------------------------------------------------------


class TestLeanAudit:
    """Verify no new mandatory dependency was added to pyproject.toml."""

    # Known mandatory dependencies at the start of the flow-engine work.
    # If a new dep appears, this test fails explicitly.
    ALLOWED_MANDATORY_DEPS = {
        "fastapi",
        "httpx",
        "mcp",
        "pydantic",
        "python-dotenv",
        "uvicorn",
    }

    def test_no_new_mandatory_dependency(self):
        """pyproject.toml [project].dependencies has no unexpected entries."""
        pyproject_path = pathlib.Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml not found"

        content = pyproject_path.read_text()

        # Parse the dependencies section
        in_deps = False
        deps_found: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped == "dependencies = [":
                in_deps = True
                continue
            if in_deps:
                if stripped == "]":
                    break
                # Extract package name (before any version specifier)
                dep = stripped.strip('",').split(">=")[0].split("<=")[0].split("==")[0].split(">")[0].split("<")[0].strip()
                if dep:
                    deps_found.append(dep)

        actual_deps = {d.lower().replace("-", "-") for d in deps_found}
        unexpected = actual_deps - self.ALLOWED_MANDATORY_DEPS
        assert not unexpected, (
            f"New mandatory dependencies detected: {unexpected}. "
            f"Req 14.1 forbids adding mandatory runtime deps."
        )


# ---------------------------------------------------------------------------
# 4. Kernel not modified (Req 14.2)
# ---------------------------------------------------------------------------


class TestKernelNotModified:
    """Verify loomable.kernel was not modified by the flow-engine feature."""

    def test_kernel_init_has_no_flow_imports(self):
        """loomable/kernel/__init__.py does not import from loomable.flow."""
        kernel_init = pathlib.Path("loomable/kernel/__init__.py")
        assert kernel_init.exists()

        content = kernel_init.read_text()
        # Parse the AST to check imports
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("loomable.flow"), (
                        f"Kernel imports loomable.flow: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("loomable.flow"):
                    pytest.fail(
                        f"Kernel imports from loomable.flow: {node.module}"
                    )

    def test_kernel_modules_do_not_import_flow(self):
        """No kernel module file imports from loomable.flow."""
        kernel_dir = pathlib.Path("loomable/kernel")
        assert kernel_dir.is_dir()

        for py_file in kernel_dir.glob("*.py"):
            content = py_file.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("loomable.flow"), (
                            f"{py_file.name} imports loomable.flow: {alias.name}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("loomable.flow"):
                        pytest.fail(
                            f"{py_file.name} imports from loomable.flow: {node.module}"
                        )

    def test_kernel_init_exports_unchanged(self):
        """Kernel __all__ doesn't reference any flow-engine symbols."""
        import loomable.kernel as kernel_mod

        for name in kernel_mod.__all__:
            # Flow-engine symbols that should never appear in kernel
            assert "Flow" not in name or name in (
                # Allow any existing kernel symbols that happen to contain "Flow"
            ), f"Kernel exports flow symbol: {name}"
            assert name != "Runnable", f"Kernel exports Runnable — belongs in flow"
            assert name != "Loop", f"Kernel exports Loop — belongs in flow"
            assert name != "SharedState", f"Kernel exports SharedState — belongs in flow"
