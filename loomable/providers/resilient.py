"""loomable.providers.resilient - Retry wrapper with backoff + jitter + timeout.

``ResilientModel`` wraps any ``ModelProvider`` to add:

- Per-call timeout (``asyncio.wait_for``)
- Exponential backoff with full jitter on transient failures
- Fail-fast on permanent provider errors (no retry)
- Honoring ``Retry-After`` headers when larger than computed backoff

This is **transport resilience for model calls only** — it never touches tools
(replan, don't retry).
"""

from __future__ import annotations

__all__ = ["RetryPolicy", "ResilientModel"]

import asyncio
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loomable.kernel.contracts import ModelProvider
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.providers.errors import PermanentProviderError, TransientProviderError

if TYPE_CHECKING:
    from loomable.agent.events import AgentEvents


@dataclass
class RetryPolicy:
    """Configuration controlling ResilientModel retry behavior.

    Attributes
    ----------
    max_attempts:
        Total tries (1 initial + N-1 retries). Default 3.
    base_delay:
        Base delay in seconds for backoff computation. Default 0.5.
    max_delay:
        Ceiling for the computed backoff delay. Default 20.0.
    multiplier:
        Exponential factor applied per attempt. Default 2.0.
    jitter:
        Full-jitter fraction (kept for documentation; the actual jitter
        range is ``[0, ceiling]`` per AWS best practice). Default 0.5.
    per_call_timeout:
        Seconds to bound each individual attempt via ``asyncio.wait_for``.
        Default 60.0.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 20.0
    multiplier: float = 2.0
    jitter: float = 0.5
    per_call_timeout: float = 60.0


def _backoff_delay(attempt: int, policy: RetryPolicy, retry_after: float | None) -> float:
    """Compute the backoff delay for a given attempt using full jitter.

    Parameters
    ----------
    attempt:
        0-based attempt index (0 = first retry, after the initial failure).
    policy:
        The retry policy providing base_delay, multiplier, and max_delay.
    retry_after:
        Server-provided Retry-After value in seconds, or None.

    Returns
    -------
    float
        A delay in seconds: ``random.uniform(0, min(max_delay, base * mult**attempt))``,
        but never below ``retry_after`` when provided.
    """
    ceiling = min(policy.max_delay, policy.base_delay * (policy.multiplier ** attempt))
    delay = random.uniform(0.0, ceiling)
    if retry_after is not None:
        return max(delay, retry_after)
    return delay


class ResilientModel:
    """Wraps a ModelProvider, adding per-call timeout + backoff-with-jitter retry.

    Retries ONLY transient errors (``TransientProviderError`` / timeout). Fails fast
    on ``PermanentProviderError`` (4xx/auth/policy). This is transport resilience for
    MODEL CALLS ONLY — it never touches tools (replan, don't retry).

    Implements the kernel ``ModelProvider`` protocol so it drops in wherever a
    provider is used.
    """

    def __init__(
        self,
        inner: ModelProvider,
        policy: RetryPolicy | None = None,
        events: "AgentEvents | None" = None,
    ) -> None:
        self._inner = inner
        self._policy = policy if policy is not None else RetryPolicy()
        self._events = events

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Attempt up to ``policy.max_attempts`` calls to the inner provider.

        Each attempt is bounded by ``asyncio.wait_for(per_call_timeout)``.
        Transient failures trigger backoff + jitter sleep and retry.
        Permanent failures are raised immediately after one attempt.
        On success, the inner provider's response is returned unchanged.
        """
        policy = self._policy
        last_error: Exception | None = None

        for attempt in range(policy.max_attempts):
            try:
                response = await asyncio.wait_for(
                    self._inner.complete(request),
                    timeout=policy.per_call_timeout,
                )
                return response

            except PermanentProviderError:
                # Fail fast — no retry for permanent errors
                raise

            except TransientProviderError as exc:
                last_error = exc
                if attempt < policy.max_attempts - 1:
                    delay = _backoff_delay(attempt, policy, exc.retry_after)
                    await asyncio.sleep(delay)

            except asyncio.TimeoutError:
                # Timeout is treated as a transient error
                last_error = TransientProviderError(
                    "timeout", status_code=None, retry_after=None
                )
                if attempt < policy.max_attempts - 1:
                    delay = _backoff_delay(attempt, policy, None)
                    await asyncio.sleep(delay)

        # All attempts exhausted — raise the last transient error
        if last_error is not None:
            raise last_error
        # Should not reach here, but satisfy type checker
        raise TransientProviderError("unknown", status_code=None, retry_after=None)  # pragma: no cover
