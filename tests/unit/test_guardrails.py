"""Unit tests for loomable.kernel.guardrails module."""

import pytest

from loomable.kernel.errors import GuardrailViolation
from loomable.kernel.guardrails import GuardrailHarness, VerificationGate
from loomable.kernel.models import ToolCall


# ---------------------------------------------------------------------------
# GuardrailHarness tests
# ---------------------------------------------------------------------------


class TestGuardrailHarness:
    """Tests for the GuardrailHarness class."""

    def test_no_rules_allows_all_actions(self):
        """With no rules configured, all actions are allowed."""
        harness = GuardrailHarness(rules=[])
        actions = [
            ToolCall(id="1", tool_name="read_file", args={"path": "/tmp/x"}),
            ToolCall(id="2", tool_name="write_file", args={"path": "/tmp/y"}),
        ]
        allowed, violations = harness.evaluate(actions)
        assert allowed == actions
        assert violations == []

    def test_blocked_tools_blocks_matching_action(self):
        """A tool in blocked_tools is blocked and recorded."""
        rules = [
            {"rule_id": "no-delete", "blocked_tools": ["delete_file"]}
        ]
        harness = GuardrailHarness(rules=rules)
        actions = [
            ToolCall(id="1", tool_name="read_file", args={}),
            ToolCall(id="2", tool_name="delete_file", args={"path": "/important"}),
        ]
        allowed, violations = harness.evaluate(actions)
        assert len(allowed) == 1
        assert allowed[0].tool_name == "read_file"
        assert len(violations) == 1
        assert violations[0].rule_id == "no-delete"
        assert violations[0].action == "delete_file"

    def test_blocked_tools_multiple_rules(self):
        """Multiple rules are evaluated; first matching rule blocks."""
        rules = [
            {"rule_id": "no-write", "blocked_tools": ["write_file"]},
            {"rule_id": "no-exec", "blocked_tools": ["exec_command"]},
        ]
        harness = GuardrailHarness(rules=rules)
        actions = [
            ToolCall(id="1", tool_name="write_file", args={}),
            ToolCall(id="2", tool_name="exec_command", args={}),
            ToolCall(id="3", tool_name="read_file", args={}),
        ]
        allowed, violations = harness.evaluate(actions)
        assert len(allowed) == 1
        assert allowed[0].tool_name == "read_file"
        assert len(violations) == 2
        assert violations[0].rule_id == "no-write"
        assert violations[1].rule_id == "no-exec"

    def test_blocked_patterns_matches_tool_name(self):
        """A regex pattern matching tool_name blocks the action."""
        rules = [
            {"rule_id": "no-danger", "blocked_patterns": [r"^danger_"]}
        ]
        harness = GuardrailHarness(rules=rules)
        actions = [
            ToolCall(id="1", tool_name="danger_rm", args={}),
            ToolCall(id="2", tool_name="safe_read", args={}),
        ]
        allowed, violations = harness.evaluate(actions)
        assert len(allowed) == 1
        assert allowed[0].tool_name == "safe_read"
        assert len(violations) == 1
        assert violations[0].rule_id == "no-danger"

    def test_blocked_patterns_matches_args(self):
        """A regex pattern matching stringified args blocks the action."""
        rules = [
            {"rule_id": "no-secrets", "blocked_patterns": [r"/etc/passwd"]}
        ]
        harness = GuardrailHarness(rules=rules)
        actions = [
            ToolCall(id="1", tool_name="read_file", args={"path": "/etc/passwd"}),
            ToolCall(id="2", tool_name="read_file", args={"path": "/tmp/safe"}),
        ]
        allowed, violations = harness.evaluate(actions)
        assert len(allowed) == 1
        assert allowed[0].args["path"] == "/tmp/safe"
        assert len(violations) == 1
        assert violations[0].rule_id == "no-secrets"

    def test_empty_actions_returns_empty(self):
        """Evaluating an empty list returns empty allowed and violations."""
        rules = [{"rule_id": "r1", "blocked_tools": ["x"]}]
        harness = GuardrailHarness(rules=rules)
        allowed, violations = harness.evaluate([])
        assert allowed == []
        assert violations == []

    def test_violation_is_guardrail_violation_instance(self):
        """Violations are GuardrailViolation instances with correct fields."""
        rules = [{"rule_id": "block-all", "blocked_patterns": [r".*"]}]
        harness = GuardrailHarness(rules=rules)
        actions = [ToolCall(id="1", tool_name="anything", args={})]
        _, violations = harness.evaluate(actions)
        assert len(violations) == 1
        v = violations[0]
        assert isinstance(v, GuardrailViolation)
        assert v.rule_id == "block-all"
        assert v.action == "anything"

    def test_rule_without_rule_id_defaults_to_unknown(self):
        """A rule without rule_id uses 'unknown' as the identifier."""
        rules = [{"blocked_tools": ["bad_tool"]}]
        harness = GuardrailHarness(rules=rules)
        actions = [ToolCall(id="1", tool_name="bad_tool", args={})]
        _, violations = harness.evaluate(actions)
        assert violations[0].rule_id == "unknown"


# ---------------------------------------------------------------------------
# VerificationGate tests
# ---------------------------------------------------------------------------


class TestVerificationGate:
    """Tests for the VerificationGate class."""

    def test_no_gate_configured_returns_true(self):
        """If no gate is configured for a step, check returns True."""
        gate = VerificationGate(gates={})
        assert gate.check(step=0, context={}) is True
        assert gate.check(step=5, context={}) is True

    def test_always_pass_condition(self):
        """A gate with condition 'always_pass' returns True."""
        gate = VerificationGate(gates={1: {"condition": "always_pass"}})
        assert gate.check(step=1, context={}) is True

    def test_always_fail_required_returns_false(self):
        """A required gate with condition 'always_fail' returns False."""
        gate = VerificationGate(
            gates={2: {"condition": "always_fail", "required": True}}
        )
        assert gate.check(step=2, context={}) is False

    def test_always_fail_not_required_returns_true(self):
        """A non-required gate with condition 'always_fail' returns True."""
        gate = VerificationGate(
            gates={2: {"condition": "always_fail", "required": False}}
        )
        assert gate.check(step=2, context={}) is True

    def test_callable_condition_passes(self):
        """A callable condition that returns True allows advancement."""
        gate = VerificationGate(
            gates={3: {"condition": "my_check", "required": True}}
        )
        context = {"my_check": lambda step, ctx: True}
        assert gate.check(step=3, context=context) is True

    def test_callable_condition_fails(self):
        """A callable condition that returns False blocks advancement."""
        gate = VerificationGate(
            gates={3: {"condition": "my_check", "required": True}}
        )
        context = {"my_check": lambda step, ctx: False}
        assert gate.check(step=3, context=context) is False

    def test_callable_condition_raises_required(self):
        """A callable that raises on a required gate returns False."""
        def bad_check(step, ctx):
            raise RuntimeError("oops")

        gate = VerificationGate(
            gates={4: {"condition": "bad_check", "required": True}}
        )
        context = {"bad_check": bad_check}
        assert gate.check(step=4, context=context) is False

    def test_callable_condition_raises_not_required(self):
        """A callable that raises on a non-required gate returns True."""
        def bad_check(step, ctx):
            raise RuntimeError("oops")

        gate = VerificationGate(
            gates={4: {"condition": "bad_check", "required": False}}
        )
        context = {"bad_check": bad_check}
        assert gate.check(step=4, context=context) is True

    def test_missing_callable_required_returns_false(self):
        """A condition naming a callable not in context, required, returns False."""
        gate = VerificationGate(
            gates={5: {"condition": "missing_fn", "required": True}}
        )
        assert gate.check(step=5, context={}) is False

    def test_missing_callable_not_required_returns_true(self):
        """A condition naming a callable not in context, not required, returns True."""
        gate = VerificationGate(
            gates={5: {"condition": "missing_fn", "required": False}}
        )
        assert gate.check(step=5, context={}) is True

    def test_default_required_is_true(self):
        """If 'required' is not specified, it defaults to True."""
        gate = VerificationGate(
            gates={6: {"condition": "always_fail"}}
        )
        assert gate.check(step=6, context={}) is False

    def test_multiple_steps_independent(self):
        """Gates for different steps are independent."""
        gate = VerificationGate(
            gates={
                1: {"condition": "always_pass"},
                2: {"condition": "always_fail", "required": True},
                3: {"condition": "always_pass"},
            }
        )
        assert gate.check(step=1, context={}) is True
        assert gate.check(step=2, context={}) is False
        assert gate.check(step=3, context={}) is True
        assert gate.check(step=4, context={}) is True  # no gate configured
