"""loomable.kernel.guardrails - Guardrail harness and verification gates.

The GuardrailHarness evaluates configured guardrail rules before any tool
dispatch: a violating action is never dispatched and is recorded as a
GuardrailViolation, while non-violating actions pass through.

The VerificationGate checks configurable per-step gates that block loop
advancement unless they pass.
"""

from __future__ import annotations

import re
from typing import Any

from loomable.kernel.errors import GuardrailViolation
from loomable.kernel.models import GuardrailRule, GateSpec, ToolCall


class GuardrailHarness:
    """Evaluates guardrail rules against proposed tool actions.

    Rules are dicts that can contain:
    - rule_id (str): identifier for the rule
    - blocked_tools (list[str]): tool names to block
    - blocked_patterns (list[str]): regex patterns matched against tool_name
      or stringified args
    """

    def __init__(self, rules: list[GuardrailRule]) -> None:
        """Initialize with configured guardrail rules.

        Args:
            rules: List of guardrail rule dictionaries.
        """
        self._rules = rules

    def evaluate(
        self, actions: list[ToolCall]
    ) -> tuple[list[ToolCall], list[GuardrailViolation]]:
        """Evaluate each action against configured rules.

        Returns a tuple of (allowed_actions, violations). A violating action
        is never in allowed_actions and is recorded as a GuardrailViolation.

        Args:
            actions: List of proposed tool calls to evaluate.

        Returns:
            A tuple of (allowed_actions, violations).
        """
        allowed: list[ToolCall] = []
        violations: list[GuardrailViolation] = []

        for action in actions:
            violation = self._check_action(action)
            if violation is not None:
                violations.append(violation)
            else:
                allowed.append(action)

        return allowed, violations

    def _check_action(self, action: ToolCall) -> GuardrailViolation | None:
        """Check a single action against all rules.

        Returns a GuardrailViolation if the action violates any rule,
        or None if the action is allowed.
        """
        for rule in self._rules:
            rule_id = rule.get("rule_id", "unknown")

            # Check blocked_tools list
            blocked_tools = rule.get("blocked_tools", [])
            if action.tool_name in blocked_tools:
                return GuardrailViolation(
                    rule_id=rule_id, action=action.tool_name
                )

            # Check blocked_patterns (regex against tool_name or args)
            blocked_patterns = rule.get("blocked_patterns", [])
            for pattern in blocked_patterns:
                if re.search(pattern, action.tool_name):
                    return GuardrailViolation(
                        rule_id=rule_id, action=action.tool_name
                    )
                # Also check against stringified args
                args_str = str(action.args)
                if re.search(pattern, args_str):
                    return GuardrailViolation(
                        rule_id=rule_id, action=action.tool_name
                    )

        return None


class VerificationGate:
    """Configurable per-step verification gates.

    Gates block loop advancement unless they pass. A GateSpec can contain:
    - condition (str): "always_pass", "always_fail", or a callable name
    - required (bool): whether the gate is required (default True)
    """

    def __init__(self, gates: dict[int, GateSpec]) -> None:
        """Initialize with gate specs per step.

        Args:
            gates: Mapping of step number to GateSpec dict.
        """
        self._gates = gates

    def check(self, step: int, context: dict[str, Any]) -> bool:
        """Check whether the gate passes for a given step.

        Returns True if the gate passes for this step (or no gate is
        configured for this step). Returns False if a gate is configured
        and it fails.

        Args:
            step: The current step number.
            context: A dict of contextual information available to the gate.

        Returns:
            True if the step may proceed, False if blocked.
        """
        gate_spec = self._gates.get(step)
        if gate_spec is None:
            return True

        condition = gate_spec.get("condition", "always_pass")
        required = gate_spec.get("required", True)

        if condition == "always_pass":
            return True
        elif condition == "always_fail":
            return not required
        else:
            # Treat condition as a callable name in context
            gate_fn = context.get(condition)
            if gate_fn is not None and callable(gate_fn):
                try:
                    result = gate_fn(step, context)
                    return bool(result)
                except Exception:
                    # Gate function raised: treat as failure
                    return not required
            # No callable found: if required, fail; otherwise pass
            return not required
