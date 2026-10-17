"""Observable guarantees shared by local and streamed agent execution."""
import asyncio
from typing import Literal

import pytest
from pydantic import BaseModel

from loomable import Agent, tool
from loomable.agent import ModelSpec, StructuredOutputError
from loomable.kernel.models import ModelResponse, StreamEvent
from loomable.providers.resilient import RetryPolicy
from loomable.providers.errors import TransientProviderError


class Answer(BaseModel):
    value: int


class Provider:
    def __init__(self, text='{"value": 7}'):
        self.text = text
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, request):
        self.complete_calls += 1
        return ModelResponse(content=self.text, usage={"input_tokens": 2, "output_tokens": 3})

    async def stream(self, request):
        self.stream_calls += 1
        yield StreamEvent(kind="text", text=self.text)
        yield StreamEvent(kind="end", usage={"input_tokens": 2, "output_tokens": 3})


def agent(provider=None, **kwargs):
    return Agent(model=ModelSpec(provider="test", provider_impl=provider or Provider()), **kwargs)


@pytest.mark.parametrize("surface", ["arun", "astream", "astream_events"])
async def test_invalid_structured_output_never_reports_success(surface):
    a = agent(Provider("invalid"), response_model=Answer)
    if surface == "arun":
        with pytest.raises(StructuredOutputError):
            await a.arun("question")
    elif surface == "astream":
        with pytest.raises(StructuredOutputError):
            _ = [chunk async for chunk in a.astream("question")]
    else:
        events = [event async for event in a.astream_events("question")]
        assert "RUN_ERROR" in [e.type for e in events]
        assert "RUN_FINISHED" not in [e.type for e in events]


@pytest.mark.parametrize("surface", ["astream", "astream_events"])
async def test_success_is_emitted_after_history_and_verification(surface):
    verified = []
    a = agent(session_id="history", verifier=lambda output, ctx: verified.append(output.text()) or True)
    stream = getattr(a, surface)("question")
    async for event in stream:
        terminal = event.done if surface == "astream" else event.type == "RUN_FINISHED"
        if terminal:
            assert verified
            assert len(a._get_built().session.l1) == 2
            break
    await stream.aclose()


async def test_retry_wrapper_keeps_native_streaming():
    provider = Provider()
    a = agent(provider, resilience=RetryPolicy())
    chunks = [c async for c in a.astream("question")]
    assert chunks[-1].done
    assert provider.stream_calls == 1
    assert provider.complete_calls == 0


async def test_partial_stream_is_never_retried():
    class Broken(Provider):
        async def stream(self, request):
            self.stream_calls += 1
            yield StreamEvent(kind="text", text="partial")
            raise TransientProviderError("disconnected")

    provider = Broken()
    a = agent(provider, resilience=RetryPolicy(base_delay=0, max_delay=0))
    with pytest.raises(TransientProviderError):
        _ = [c async for c in a.astream("question")]
    assert provider.stream_calls == 1


async def test_closing_stream_awaits_provider_cleanup():
    closed = asyncio.Event()

    class Slow(Provider):
        async def stream(self, request):
            try:
                yield StreamEvent(kind="text", text="started")
                await asyncio.Event().wait()
            finally:
                closed.set()

    a = agent(Slow())
    stream = a.astream("question")
    await anext(stream)
    await stream.aclose()
    assert closed.is_set()


async def test_tool_schema_preserves_nested_types_and_rejects_invalid_args():
    executed = []

    @tool
    def typed(count: int | None, mode: Literal["fast", "slow"], tags: list[str]):
        executed.append(True)
        return count

    props = typed.parameters["properties"]
    assert props["count"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert props["mode"]["enum"] == ["fast", "slow"]
    assert props["tags"]["items"] == {"type": "string"}
    result = await typed.invoke({"count": 1, "mode": "invalid", "tags": ["a"]})
    assert result.is_error
    assert not executed


async def test_independent_per_call_sessions_do_not_mix_history():
    from loomable.kernel.stores import SessionStore

    class Echo(Provider):
        async def complete(self, request):
            await asyncio.sleep(0)
            return ModelResponse(content=str(request.messages))

    store = SessionStore()
    a = agent(Echo(), session_store=store)
    x, y = await asyncio.gather(a.arun("alice-only", session_id="alice"),
                                a.arun("bob-only", session_id="bob"))
    assert x.session_id == "alice" and y.session_id == "bob"
    assert "bob-only" not in x.text and "alice-only" not in y.text
    followup = await a.arun("remember", session_id="alice")
    assert "alice-only" in followup.text and "bob-only" not in followup.text


async def test_same_runtime_overlapping_run_is_rejected():
    entered = asyncio.Event()
    release = asyncio.Event()

    class Blocking(Provider):
        async def complete(self, request):
            entered.set()
            await release.wait()
            return ModelResponse(content="done")

    a = agent(Blocking())
    task = asyncio.create_task(a.arun("first"))
    await entered.wait()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await asyncio.wait_for(a.arun("second"), timeout=0.2)
    finally:
        release.set()
        await task


def test_forked_context_accounts_for_child_tokens():
    from loomable.agent import RunContext
    parent = RunContext(max_run_tokens=10)
    child = parent.fork()
    child.add_tokens(10)
    assert parent.token_budget_exceeded()
