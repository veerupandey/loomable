"""Loop primitive (Tier 2) and Verifier protocol.

The Loop repeats a body Runnable until a Verifier reports success or an
iteration cap is reached, whichever comes first. It is itself a Runnable so
it can be used standalone or as a node inside a Flow.

The ``Verifier`` protocol is defined here and is usable by Agent, Loop, and
Flow node without redefinition (Req 4.5).
"""

from __future__ import annotations

__all__ = [
    "VerdictResult",
    "Verifier",
    "AlwaysOkVerifier",
    "CallableVerifier",
    "Loop",
]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput

if TYPE_CHECKING:
    from loomable.flow.runnable import Runnable


# ---------------------------------------------------------------------------
# VerdictResult
# ---------------------------------------------------------------------------


@dataclass
class VerdictResult:
    """Result of a verification check.

    Attributes
    ----------
    ok:
        Whether the output passed the verifier.
    detail:
        Optional human-readable detail. On failure, this is fed to the next
        iteration (reflexion) so the body can self-correct.
    """

    ok: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Verifier protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Verifier(Protocol):
    """Pluggable output-verification protocol (Req 4.1, 4.5).

    A Verifier inspects a produced output together with the run context and
    returns a :class:`VerdictResult` indicating success or failure with
    optional detail.

    The same protocol is usable by Agent (output guardrail), Loop (exit
    condition), and any Flow node — one definition, three consumers.
    """

    def check(self, output: AgentOutput, context: RunContext) -> VerdictResult:
        """Evaluate *output* and return a verdict."""
        ...


# ---------------------------------------------------------------------------
# Default verifier: always ok
# ---------------------------------------------------------------------------


class AlwaysOkVerifier:
    """A verifier that always reports success.

    This is the default when no verifier is configured:
    - A Loop with no verifier runs its body exactly once (Req 5.4).
    - An Agent with no verifier is unchanged in behavior (Req 4.4).
    """

    def check(self, output: AgentOutput, context: RunContext) -> VerdictResult:
        """Always return ok=True."""
        return VerdictResult(ok=True)


# ---------------------------------------------------------------------------
# Callable → Verifier adapter
# ---------------------------------------------------------------------------


class CallableVerifier:
    """Adapt a plain callable ``(output, context) -> bool`` to the Verifier protocol.

    This provides ergonomic shorthand so users can supply a simple boolean
    predicate instead of implementing the full Verifier protocol::

        loop = Loop(body, verifier=lambda output, ctx: "done" in output.text)

    The callable receives the :class:`AgentOutput` and :class:`RunContext` and
    must return a truthy/falsy value. On falsy, a ``VerdictResult(ok=False)``
    is returned with an empty detail string (the callable has no way to
    provide detail).
    """

    def __init__(self, fn: Callable[[AgentOutput, RunContext], bool]) -> None:
        self._fn = fn

    def check(self, output: AgentOutput, context: RunContext) -> VerdictResult:
        """Delegate to the wrapped callable and wrap into a VerdictResult."""
        result = self._fn(output, context)
        if isinstance(result, VerdictResult):
            return result
        return VerdictResult(ok=bool(result))


# ---------------------------------------------------------------------------
# Loop — Tier 2 Runnable
# ---------------------------------------------------------------------------


class Loop:
    """Repeat a body Runnable until a Verifier passes or a cap is reached.

    The Loop is itself a Runnable (Req 5.6) so it can be used standalone or
    as a node inside a Flow with identical behavior.

    Parameters
    ----------
    body:
        The Runnable to execute on each iteration. Mutually exclusive with
        ``steps``.
    steps:
        A list of composable elements (Step, Condition, Parallel_Group,
        Workflow) that are compiled into a sequential Workflow and used as
        the loop body. Mutually exclusive with ``body``.
    verifier:
        A Verifier, a callable ``(output, context) -> bool``, or ``None``.
        When ``None``, an :class:`AlwaysOkVerifier` is used — the body runs
        exactly once (Req 5.4). A callable is automatically adapted via
        :class:`CallableVerifier`.
    max_iterations:
        Maximum number of iterations before the loop stops regardless of
        the verifier outcome (Req 5.3). Defaults to 3.
    """

    def __init__(
        self,
        body: "Runnable | None" = None,
        *,
        steps: "list[Any] | None" = None,
        verifier: "Verifier | Callable[[AgentOutput, RunContext], bool] | None" = None,
        max_iterations: int = 3,
    ) -> None:
        # --- Resolve body vs steps (Req 5.1, 5.2, 5.3) ---
        if body is not None and steps is not None:
            raise ValueError("Only one of 'body' or 'steps' may be specified")

        if steps is not None:
            # Compile steps list into a sequential Workflow and use as body
            from loomable.flow.workflow import Workflow

            self._body: "Runnable" = Workflow(
                name="_loop_body",
                steps=steps,
            )
        elif body is not None:
            # Accept plain callables the same way Step does.
            from loomable.flow.runnable import FunctionRunnable, Runnable

            if isinstance(body, Runnable):
                self._body = body
            elif callable(body):
                self._body = FunctionRunnable(body)
            else:
                raise TypeError(
                    f"Loop body must be a Runnable or callable, got {type(body).__name__}"
                )
        else:
            raise ValueError("Either 'body' or 'steps' must be provided")

        self._max_iterations = max_iterations

        # Propagate edge data contracts from a Step body so the compiler can
        # stamp ``payload_key`` on the edge into this Loop node.
        self.reads = getattr(body, "reads", None) if body is not None else None
        if self.reads is None and steps:
            # First Step in a steps= list may declare reads=
            for el in steps:
                reads = getattr(el, "reads", None)
                if reads:
                    self.reads = reads
                    break

        if verifier is None:
            self._verifier: Verifier = AlwaysOkVerifier()
        elif callable(verifier) and not isinstance(verifier, Verifier):
            self._verifier = CallableVerifier(verifier)
        else:
            self._verifier = verifier  # type: ignore[assignment]

    async def arun(
        self, input: Any, *, context: RunContext | None = None  # noqa: A002
    ) -> RunResult:
        """Execute the loop body repeatedly until verified or capped.

        On each iteration:
        1. Run the body with the current input/context.
        2. Check the verifier against the body's output.
        3. If ok → stop and return the body's output (Req 5.2).
        4. If not ok and iterations remain → feed failure detail forward
           so the body can self-correct (Req 5.5).
        5. If the iteration cap is reached without success → stop and
           record ``metadata["loop_stop"] = "max_iterations"`` (Req 5.3).
        """
        ctx = context or RunContext()
        last_result: RunResult | None = None
        current_input = input
        failure_detail: str = ""

        for iteration in range(1, self._max_iterations + 1):
            # Prepare input for body — on subsequent iterations, append
            # failure detail so the body can self-correct (Req 5.5).
            body_input = current_input
            if failure_detail:
                # Augment the input with the verifier's failure detail.
                # If input is a string, append detail. Otherwise, wrap in a
                # dict structure the body can inspect.
                if isinstance(current_input, str):
                    body_input = (
                        f"{current_input}\n\n"
                        f"[Previous attempt failed: {failure_detail}]"
                    )
                else:
                    body_input = {
                        "input": current_input,
                        "feedback": failure_detail,
                    }

            # Run the body
            if ctx.cancelled:
                break
            last_result = await self._body.arun(body_input, context=ctx)

            # Verify the output
            verdict = self._verifier.check(last_result.output, ctx)

            if verdict.ok:
                # Success — expose last body output as result (Req 5.2)
                last_result.metadata["loop_iterations"] = iteration
                last_result.metadata["loop_verified"] = True
                return last_result

            # Failure — capture detail for next iteration (Req 5.5)
            failure_detail = verdict.detail

        # Iteration cap reached without success (Req 5.3), or cancelled.
        if last_result is None:
            from loomable.content import AgentOutput, Text

            last_result = RunResult(
                output=AgentOutput(parts=[Text("")]), session_id=""
            )
        if ctx.cancelled:
            last_result.metadata["loop_stop"] = "cancelled"
            last_result.metadata["stop_reason"] = "cancelled"
        else:
            last_result.metadata["loop_stop"] = "max_iterations"
            last_result.metadata["loop_iterations"] = self._max_iterations
        last_result.metadata["loop_verified"] = False
        return last_result
