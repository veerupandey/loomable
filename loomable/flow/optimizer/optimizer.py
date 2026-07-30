"""Optimizer: applies enabled rules to produce a rewritten Flow + explain report.

The Optimizer is opt-in and a no-op when not enabled (Req 10.1). It applies
only individually enabled rules in a fixed order (Req 10.7) and captures
both original and rewritten plans for inspection (Req 10.8).
"""

from __future__ import annotations

__all__ = ["Optimizer"]

from typing import TYPE_CHECKING

from loomable.flow.optimizer.rules import OptimizationRule

if TYPE_CHECKING:
    from loomable.flow.flow import Flow


class Optimizer:
    """Opt-in optimization pass that rewrites a Flow into an equivalent plan.

    The Optimizer applies each enabled OptimizationRule in fixed order
    (the order they were provided at construction). Each rule is individually
    toggleable via `enable_rule` / `disable_rule`.

    When disabled (the default for a Flow without an explicit optimizer),
    `optimize()` returns the flow unchanged — it is a complete no-op.

    Parameters
    ----------
    rules:
        The optimization rules to apply, in order. If ``None``, defaults
        to an empty list (no rules applied). Actual rules are registered
        via tasks 10.2 and 10.3.
    enabled:
        Whether the optimizer is active. When ``False``, `optimize()`
        is a no-op regardless of rules (Req 10.1).
    """

    def __init__(
        self,
        rules: list[OptimizationRule] | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._rules: list[OptimizationRule] = list(rules) if rules else []
        self._enabled = enabled
        # Track which rules are individually enabled (all enabled by default)
        self._rule_enabled: dict[str, bool] = {
            rule.name: True for rule in self._rules
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the optimizer is globally enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def rules(self) -> list[OptimizationRule]:
        """The list of registered rules (in application order)."""
        return list(self._rules)

    def enable_rule(self, name: str) -> None:
        """Enable an individual rule by name.

        Raises KeyError if no rule with the given name is registered.
        """
        if name not in self._rule_enabled:
            raise KeyError(
                f"No rule named {name!r} registered. "
                f"Available: {sorted(self._rule_enabled.keys())}"
            )
        self._rule_enabled[name] = True

    def disable_rule(self, name: str) -> None:
        """Disable an individual rule by name.

        Raises KeyError if no rule with the given name is registered.
        """
        if name not in self._rule_enabled:
            raise KeyError(
                f"No rule named {name!r} registered. "
                f"Available: {sorted(self._rule_enabled.keys())}"
            )
        self._rule_enabled[name] = False

    def is_rule_enabled(self, name: str) -> bool:
        """Check if a specific rule is currently enabled."""
        return self._rule_enabled.get(name, False)

    def optimize(self, flow: "Flow") -> tuple["Flow", list[str]]:
        """Apply enabled rules in fixed order and return the rewritten flow.

        Parameters
        ----------
        flow:
            The Flow to optimize.

        Returns
        -------
        tuple[Flow, list[str]]:
            A tuple of (optimized_flow, applied_rule_names).
            If the optimizer is disabled or no rules fire, returns
            (flow, []) — the original flow unchanged.
        """
        if not self._enabled:
            return flow, []

        applied: list[str] = []
        current = flow

        for rule in self._rules:
            if not self._rule_enabled.get(rule.name, False):
                continue
            rewritten = rule.apply(current)
            # If the rule produced a different flow, record it
            if rewritten is not current:
                applied.append(rule.name)
                current = rewritten

        return current, applied

    def __repr__(self) -> str:
        n_rules = len(self._rules)
        n_enabled = sum(1 for v in self._rule_enabled.values() if v)
        state = "enabled" if self._enabled else "disabled"
        return f"Optimizer({state}, rules={n_rules}, active={n_enabled})"
