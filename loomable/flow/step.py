"""Step — Named wrapper around a Runnable or callable.

A Step is the atomic building block of Workflows. It carries a name
(used as node_id when compiled into a Flow), an optional description,
and optional per-step dependency injection.

The Step satisfies the :class:`Runnable` protocol by delegating ``arun``
to the wrapped agent (or a :class:`FunctionRunnable` adapter for plain
callables).

Graph-engineering extensions (aligned with node contracts / local failure):

- ``on_failure`` — local failure policy (``raise`` / ``retry`` / ``skip`` /
  ``fallback`` / ``stop``)
- ``reads`` — SharedState key this step consumes (edge data contract)
- ``complexity`` — cost hint (``"low"`` / ``"high"``) for model-tier routing
"""

from __future__ import annotations

__all__ = ["Step", "StepFailed", "FAILURE_ACTIONS"]

from typing import Any, Callable, Literal

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.kernel.errors import LoomableError

FAILURE_ACTIONS = ("raise", "retry", "skip", "fallback", "stop")
FailureAction = Literal["raise", "retry", "skip", "fallback", "stop"]


class StepFailed(LoomableError):
    """Raised when a Step's ``on_failure="stop"`` policy fires (or retries exhaust).

    Attributes
    ----------
    step_name:
        The Step that failed.
    action:
        The failure policy that produced this error.
    attempts:
        Number of primary-agent attempts before giving up.
    """

    def __init__(
        self,
        step_name: str,
        *,
        action: str = "stop",
        attempts: int = 1,
        cause: BaseException | None = None,
    ) -> None:
        self.step_name = step_name
        self.action = action
        self.attempts = attempts
        self.cause = cause
        detail = f"Step {step_name!r} failed after {attempts} attempt(s) (on_failure={action!r})"
        if cause is not None:
            detail = f"{detail}: {cause}"
        super().__init__(detail)


class Step:
    """Named wrapper around a Runnable or callable.

    Parameters
    ----------
    name:
        A non-empty string that uniquely identifies this step within a
        Workflow. Used as ``node_id`` in the compiled Flow graph.
    agent:
        The execution unit — either an object satisfying the :class:`Runnable`
        protocol or a plain sync/async callable (which is adapted via
        :class:`FunctionRunnable`).
    description:
        Optional human-readable description of what this step does.
    deps:
        Optional dependency object injected into :class:`RunContext` when
        this step executes, overriding any flow-level deps for this step only.
    require_confirmation:
        When True, the compiled Flow pauses before this step (HITL) until
        ``Workflow.approve(name)`` is called and the run is resumed.
    on_failure:
        Local failure policy when the primary agent raises:

        - ``"raise"`` (default) — propagate the exception
        - ``"retry"`` — retry up to ``max_retries`` times, then raise
        - ``"skip"`` — return an empty result with ``metadata["step_skipped"]=True``
        - ``"fallback"`` — run ``fallback`` instead (requires ``fallback=``)
        - ``"stop"`` — raise :class:`StepFailed` to halt the graph
    max_retries:
        Extra attempts after the first failure when ``on_failure="retry"``.
        Defaults to ``0`` (no retries) unless ``on_failure="retry"``, in which
        case it defaults to ``2``.
    fallback:
        Alternate Runnable/callable used when ``on_failure="fallback"``.
    reads:
        Optional SharedState key this step consumes. Compiled as an edge
        ``payload_key`` so the engine feeds ``state[reads]`` instead of the
        previous node's ambient output.
    complexity:
        Cost hint (``"low"`` / ``"high"``). Low-complexity steps can be
        marked for cheaper model tiers by the flow optimizer.
    """

    def __init__(
        self,
        name: str,
        agent: Runnable | Callable[..., Any],
        *,
        description: str = "",
        deps: Any = None,
        require_confirmation: bool = False,
        confirm: bool | None = None,
        on_failure: FailureAction = "raise",
        max_retries: int | None = None,
        fallback: Runnable | Callable[..., Any] | None = None,
        reads: str | None = None,
        complexity: Literal["low", "high"] | None = None,
    ) -> None:
        if not name:
            raise ValueError("Step name is required")
        if on_failure not in FAILURE_ACTIONS:
            raise ValueError(
                f"on_failure must be one of {FAILURE_ACTIONS}, got {on_failure!r}"
            )
        if on_failure == "fallback" and fallback is None:
            raise ValueError('on_failure="fallback" requires fallback=')
        if complexity is not None and complexity not in ("low", "high"):
            raise ValueError(
                f"complexity must be 'low', 'high', or None, got {complexity!r}"
            )

        self._name = name
        self._description = description
        self._deps = deps
        if confirm is not None:
            require_confirmation = confirm
        self.require_confirmation = bool(require_confirmation)
        self.on_failure: FailureAction = on_failure
        if max_retries is None:
            self.max_retries = 2 if on_failure == "retry" else 0
        else:
            if max_retries < 0:
                raise ValueError("max_retries must be >= 0")
            self.max_retries = max_retries
        self.reads = reads
        self.complexity = complexity

        # Wrap plain callables in FunctionRunnable to satisfy the Runnable protocol.
        self._agent: Runnable = self._as_runnable(agent, label="agent")
        self._fallback: Runnable | None = None
        if fallback is not None:
            self._fallback = self._as_runnable(fallback, label="fallback")

    @staticmethod
    def _as_runnable(
        value: Runnable | Callable[..., Any], *, label: str
    ) -> Runnable:
        if isinstance(value, Runnable):
            return value
        if callable(value):
            return FunctionRunnable(value)
        raise TypeError(
            f"{label} must be a Runnable or callable, got {type(value).__name__}"
        )

    async def arun(
        self, input: Any, *, context: RunContext | None = None  # noqa: A002
    ) -> RunResult:
        """Execute the wrapped agent, applying the local failure policy.

        If this Step has ``deps`` set, it overrides the ``deps`` on the
        RunContext for this execution only. A new or cloned RunContext is
        used to avoid mutating a shared context object.
        """
        if self._deps is not None:
            # Inject step-level deps without dropping the parent's cancel flag.
            if context is None:
                context = RunContext(deps=self._deps)
            else:
                context = context.fork(deps=self._deps)

        attempts = 0
        last_exc: BaseException | None = None
        max_attempts = 1 + (self.max_retries if self.on_failure == "retry" else 0)

        while attempts < max_attempts:
            attempts += 1
            try:
                result = await self._agent.arun(input, context=context)
                if attempts > 1:
                    result.metadata["failure_retries"] = attempts - 1
                    result.metadata["failure_policy"] = self.on_failure
                return result
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                last_exc = exc
                if self.on_failure == "retry" and attempts < max_attempts:
                    continue
                break

        assert last_exc is not None
        return await self._apply_failure_policy(last_exc, attempts, input, context)

    async def _apply_failure_policy(
        self,
        exc: BaseException,
        attempts: int,
        input: Any,  # noqa: A002
        context: RunContext | None,
    ) -> RunResult:
        """Handle a failed primary attempt according to ``on_failure``."""
        if self.on_failure == "skip":
            from loomable.content import AgentOutput, MediaPart, Modality

            output = AgentOutput(
                parts=[
                    MediaPart(
                        modality=Modality.TEXT,
                        media_type="text/plain",
                        data=b"",
                    )
                ]
            )
            return RunResult(
                output=output,
                session_id="",
                metadata={
                    "step_skipped": True,
                    "failure_policy": "skip",
                    "failure_error": str(exc),
                    "failure_attempts": attempts,
                },
            )

        if self.on_failure == "fallback":
            assert self._fallback is not None
            result = await self._fallback.arun(input, context=context)
            result.metadata["failure_policy"] = "fallback"
            result.metadata["failure_error"] = str(exc)
            result.metadata["failure_attempts"] = attempts
            result.metadata["fallback_used"] = True
            return result

        if self.on_failure == "stop":
            raise StepFailed(
                self._name, action="stop", attempts=attempts, cause=exc
            ) from exc

        # "raise" and exhausted "retry"
        if self.on_failure == "retry":
            raise StepFailed(
                self._name, action="retry", attempts=attempts, cause=exc
            ) from exc
        raise exc

    @property
    def name(self) -> str:
        """The step's unique identifier (used as node_id in compiled Flows)."""
        return self._name

    @property
    def description(self) -> str:
        """Optional human-readable description of this step."""
        return self._description
