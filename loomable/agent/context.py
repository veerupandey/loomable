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
from typing import Any, ClassVar

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
    #: Cumulative spend stop for the run. ``None`` falls back to ``token_budget``
    #: (legacy). ``0`` or negative means unbounded spend (industry deep agents).
    max_run_tokens: int | None = None
    loop_repeat_threshold: int = 3

    # --- Flow-engine extensions (additive, defaults preserve existing behavior) ---

    #: Typed dependency injection object (Req 3). Any Python object (dataclass,
    #: Pydantic model, or plain value) made available to tools and nodes that
    #: declare they accept it. ``None`` when not supplied (the default).
    deps: Any = None

    #: SharedState handle set when running inside a Flow. Allows nodes and tools
    #: to read/write the flow's shared state. ``None`` outside a flow context.
    shared_state: Any = None

    #: Shared MemoryStore instance (Req 12.2). When a Flow has a memory store
    #: attached, it is set here so every node that accepts it can read/write
    #: memory. ``None`` when no memory is configured.
    memory: Any = None

    # --- internal state (not part of the public constructor) ---
    # Shared list so Step.fork() / cloned contexts observe the same cancel.
    _cancel_flag: list[bool] = field(default_factory=lambda: [False], init=False, repr=False)
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

        Checked at each loop / Workflow step boundary; once set the harness
        will stop and issue no further model or tool calls.
        """
        self._cancel_flag[0] = True

    @property
    def cancelled(self) -> bool:
        """Whether the cooperative cancel flag has been set."""
        return bool(self._cancel_flag[0])

    def fork(self, *, deps: Any | None = None) -> "RunContext":
        """Clone this context, sharing the cancel flag with the parent."""
        ctx = RunContext(
            events=self.events,
            max_steps=self.max_steps,
            token_budget=self.token_budget,
            max_run_tokens=self.max_run_tokens,
            loop_repeat_threshold=self.loop_repeat_threshold,
            deps=self.deps if deps is None else deps,
            shared_state=self.shared_state,
            memory=self.memory,
        )
        ctx._cancel_flag = self._cancel_flag
        ctx._steps_used = self._steps_used
        ctx._tokens_used = self._tokens_used
        ctx._call_history = self._call_history
        return ctx

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
        """Return ``True`` when cumulative spend has reached the run budget.

        Spend limit resolution:
        - ``max_run_tokens > 0`` → that hard cap
        - ``max_run_tokens <= 0`` → unbounded (never exceeded)
        - ``max_run_tokens is None`` → fall back to ``token_budget`` (legacy)
        - both unset → unbounded
        """
        if self.max_run_tokens is not None:
            if self.max_run_tokens <= 0:
                return False
            return self._tokens_used >= self.max_run_tokens
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
