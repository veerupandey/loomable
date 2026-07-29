"""Per-run control and observability seam for the agent harness.

This is an **edge** module — distinct from the kernel ``ContextManager``.
It carries the event emitter, step/token budgets, cooperative cancel flag,
and tool-call-signature history that every harness workstream reads from.

A fresh ``RunContext`` is created per ``arun`` invocation.  Existing callers
that do not supply one get a default with ``NoOpEvents`` so behavior is
unchanged.
"""

from __future__ import annotations

__all__ = ["RunContext", "StopReason"]

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import ClassVar

from .events import AgentEvents, NoOpEvents


def _signature(tool_name: str, args: dict) -> str:
    """Produce a stable, canonical hash for a (tool_name, args) pair.

    Canonicalizes *args* with ``json.dumps(args, sort_keys=True, default=str)``
    and hashes ``f"{tool_name}:{canonical}"`` with SHA-1.  The hex digest is
    returned as the signature string.
    """
    canonical = json.dumps(args, sort_keys=True, default=str)
    raw = f"{tool_name}:{canonical}"
    return hashlib.sha1(raw.encode()).hexdigest()


@dataclass
class StopReason:
    """Why the harness loop terminated.

    ``kind`` is one of the ``STOP_*`` class-level constants.  ``detail``
    carries optional human-readable context (e.g. which tool was looping).
    """

    # --- constants (ClassVar — not dataclass fields) ---
    FINAL: ClassVar[str] = "final"
    MAX_ITERATIONS: ClassVar[str] = "max_iterations"
    LOOP_DETECTED: ClassVar[str] = "loop_detected"
    CANCELLED: ClassVar[str] = "cancelled"
    STEP_BUDGET: ClassVar[str] = "step_budget"
    TOKEN_BUDGET: ClassVar[str] = "token_budget"
    ERROR: ClassVar[str] = "error"

    # --- instance fields ---
    kind: str
    detail: str = ""


@dataclass
class RunContext:
    """Per-run control and observability seam threaded through the run path.

    Threaded once through ``_run_single`` / ``_run_tool_loop``; tracing,
    loop-detection, budgets, and cancellation all read from it.  Kept out
    of the kernel entirely.
    """

    events: AgentEvents = field(default_factory=NoOpEvents)
    max_steps: int = 6
    token_budget: int | None = None
    loop_repeat_threshold: int = 3

    # --- internal state (not part of the public constructor) ---
    _cancelled: bool = field(default=False, init=False, repr=False)
    _steps_used: int = field(default=0, init=False, repr=False)
    _tokens_used: int = field(default=0, init=False, repr=False)
    _call_history: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _t0: float = field(default_factory=time.monotonic, init=False, repr=False)

    # ------------------------------------------------------------------
    # Cancellation (cooperative)
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Set the cooperative cancel flag.

        Checked at each loop boundary; once set the harness will stop and
        issue no further model or tool calls.
        """
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        """Whether the cooperative cancel flag has been set."""
        return self._cancelled

    # ------------------------------------------------------------------
    # Budgets
    # ------------------------------------------------------------------

    def tick_step(self) -> bool:
        """Increment steps used and return ``True`` while within ``max_steps``.

        Returns ``False`` once the step budget is exhausted.
        """
        self._steps_used += 1
        return self._steps_used <= self.max_steps

    def add_tokens(self, n: int) -> None:
        """Add *n* tokens to the cumulative usage counter."""
        self._tokens_used += n

    def token_budget_exceeded(self) -> bool:
        """Return ``True`` when cumulative tokens have reached the budget.

        If ``token_budget`` is ``None`` (unbounded), always returns ``False``.
        """
        if self.token_budget is None:
            return False
        return self._tokens_used >= self.token_budget

    # ------------------------------------------------------------------
    # Loop / no-progress detection
    # ------------------------------------------------------------------

    def record_call(self, tool_name: str, args: dict) -> int:
        """Record a tool call and return the updated repeat count.

        Hashes ``(tool_name, canonicalized args)`` into a signature, bumps
        its count in the history, and returns the new count.
        """
        sig = _signature(tool_name, args)
        self._call_history[sig] = self._call_history.get(sig, 0) + 1
        return self._call_history[sig]

    def is_looping(self, tool_name: str, args: dict) -> bool:
        """Return ``True`` when the call count would reach ``loop_repeat_threshold``.

        This checks the *current* count — i.e. the signature has already been
        recorded enough times to trigger the threshold.
        """
        sig = _signature(tool_name, args)
        count = self._call_history.get(sig, 0)
        return count >= self.loop_repeat_threshold

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def elapsed(self) -> float:
        """Return seconds elapsed since this context was created."""
        return time.monotonic() - self._t0
