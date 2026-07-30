"""Unit tests for Agent output verification (Task 3, Req 4.2–4.4).

Covers:
- Verify pass is recorded on RunResult.verification (Req 4.2)
- Verify failure is recorded on RunResult.verification (Req 4.2)
- Retry re-runs with failure detail appended when retry_on_failure=True (Req 4.3)
- Retry stops after max_verify_retries exhausted (Req 4.3)
- Absence of verifier = unchanged behavior, verification is None (Req 4.4)
- Callable verifier (output, context) -> bool is accepted and adapted (Req 4.5)
"""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec, RunResult
from loomable.agent.context import RunContext
from loomable.content import AgentOutput, Modality
from loomable.flow.loop import VerdictResult, Verifier
from loomable.kernel.models import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EchoProvider:
    """A provider that echoes the first user text part back."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        self.last_requests.append(request)
        text = ""
        for message in request.messages:
            if message["role"] == "user":
                for part in message["content"]:
                    if part.get("type") == "text":
                        text = part["text"]
                        break
                if text:
                    break
        return ModelResponse(
            content=f"echo: {text}",
            usage={"input_tokens": 3, "output_tokens": 2},
        )


class AlwaysPassVerifier:
    """A verifier that always reports success."""

    def check(self, output: AgentOutput, context: RunContext) -> VerdictResult:
        return VerdictResult(ok=True, detail="all good")


class AlwaysFailVerifier:
    """A verifier that always reports failure."""

    def check(self, output: AgentOutput, context: RunContext) -> VerdictResult:
        return VerdictResult(ok=False, detail="output not acceptable")


class PassOnNthCallVerifier:
    """A verifier that passes on the Nth check."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._count = 0

    def check(self, output: AgentOutput, context: RunContext) -> VerdictResult:
        self._count += 1
        if self._count >= self._n:
            return VerdictResult(ok=True, detail="passed on retry")
        return VerdictResult(ok=False, detail=f"failed attempt {self._count}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_verifier_unchanged_behavior():
    """Req 4.4: When no verifier is configured, behavior is unchanged and
    verification is None."""
    provider = EchoProvider()
    agent = Agent(model=ModelSpec(provider="echo", provider_impl=provider))

    result = await agent.arun("hello")

    assert isinstance(result, RunResult)
    assert result.output.text() == "echo: hello"
    assert result.verification is None
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_verifier_pass_recorded():
    """Req 4.2: When verifier passes, the VerdictResult is recorded on RunResult."""
    provider = EchoProvider()
    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        verifier=AlwaysPassVerifier(),
    )

    result = await agent.arun("hello")

    assert result.verification is not None
    assert result.verification.ok is True
    assert result.verification.detail == "all good"
    assert provider.call_count == 1  # No retry


@pytest.mark.asyncio
async def test_verifier_fail_recorded_no_retry():
    """Req 4.2: When verifier fails and retry is off, failure is recorded but no retry."""
    provider = EchoProvider()
    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        verifier=AlwaysFailVerifier(),
        retry_on_failure=False,
    )

    result = await agent.arun("hello")

    assert result.verification is not None
    assert result.verification.ok is False
    assert result.verification.detail == "output not acceptable"
    assert provider.call_count == 1  # No retry because retry_on_failure=False


@pytest.mark.asyncio
async def test_verifier_fail_with_retry_reruns():
    """Req 4.3: When verifier fails and retry is enabled, the agent re-runs with
    failure detail appended to context."""
    provider = EchoProvider()
    # Passes on the 2nd verification check (i.e., after one retry)
    verifier = PassOnNthCallVerifier(n=2)
    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        verifier=verifier,
        retry_on_failure=True,
        max_verify_retries=1,
    )

    result = await agent.arun("hello")

    # The agent ran twice: original + 1 retry
    assert provider.call_count == 2
    assert result.verification is not None
    assert result.verification.ok is True
    # The retry input should contain the failure detail
    retry_request = provider.last_requests[1]
    user_texts = []
    for msg in retry_request.messages:
        if msg["role"] == "user":
            for part in msg["content"]:
                if part.get("type") == "text":
                    user_texts.append(part["text"])
    retry_text = " ".join(user_texts)
    assert "Verification failed" in retry_text
    assert "failed attempt 1" in retry_text


@pytest.mark.asyncio
async def test_verifier_retry_exhausted():
    """Req 4.3: When retries are exhausted and verifier still fails, the last
    failed verification is recorded."""
    provider = EchoProvider()
    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        verifier=AlwaysFailVerifier(),
        retry_on_failure=True,
        max_verify_retries=2,
    )

    result = await agent.arun("hello")

    # Original run + 2 retries = 3 model calls
    assert provider.call_count == 3
    assert result.verification is not None
    assert result.verification.ok is False
    assert result.verification.detail == "output not acceptable"


@pytest.mark.asyncio
async def test_callable_verifier_accepted():
    """Req 4.5: A callable (output, context) -> bool is adapted to Verifier."""
    provider = EchoProvider()
    # This callable checks if "echo" is in the output text — should pass
    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        verifier=lambda output, ctx: "echo" in output.text(),
    )

    result = await agent.arun("hello")

    assert result.verification is not None
    assert result.verification.ok is True
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_callable_verifier_fail():
    """A callable verifier that returns False records failure."""
    provider = EchoProvider()
    # This callable always fails
    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        verifier=lambda output, ctx: False,
    )

    result = await agent.arun("hello")

    assert result.verification is not None
    assert result.verification.ok is False
    assert provider.call_count == 1  # No retry (retry_on_failure defaults to False)


@pytest.mark.asyncio
async def test_callable_verifier_with_retry():
    """Callable verifier combined with retry works correctly."""
    provider = EchoProvider()
    call_count = [0]

    def check_fn(output: AgentOutput, ctx: RunContext) -> bool:
        call_count[0] += 1
        # Pass on the 2nd verification check (after one retry)
        return call_count[0] >= 2

    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        verifier=check_fn,
        retry_on_failure=True,
        max_verify_retries=1,
    )

    result = await agent.arun("hello")

    assert provider.call_count == 2  # Original + 1 retry
    assert result.verification is not None
    assert result.verification.ok is True


@pytest.mark.asyncio
async def test_max_verify_retries_default_is_one():
    """max_verify_retries defaults to 1."""
    provider = EchoProvider()
    # Verifier never passes
    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        verifier=AlwaysFailVerifier(),
        retry_on_failure=True,
        # max_verify_retries not specified, defaults to 1
    )

    result = await agent.arun("hello")

    # Original run + 1 retry (default) = 2 calls
    assert provider.call_count == 2
    assert result.verification is not None
    assert result.verification.ok is False
