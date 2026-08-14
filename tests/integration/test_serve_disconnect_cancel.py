"""Serve SSE/stream disconnect triggers cooperative cancel."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from loomable.agent import Agent, ModelSpec
from loomable.agent.context import RunContext, StopReason
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.serve.fastapi_adapter import (
    MediaPartModel,
    MessageModel,
    RunRequestModel,
    _register_agent_routes,
)
from loomable.stream import RUN_STARTED, StreamEvent


class _NeverFinalProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        await asyncio.sleep(0.02)
        return ModelResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=str(self.calls),
                    tool_name="think",
                    args={"thought": f"step-{self.calls}"},
                )
            ],
        )


@pytest.mark.asyncio
async def test_disconnect_helper_cancels_built_agent() -> None:
    """Simulate serve cancel resolution → BuiltAgent.cancel() (cooperative)."""
    from loomable.agent.reasoning import make_think_tool

    provider = _NeverFinalProvider()
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        tools=[make_think_tool()],
        think_tool=False,
        max_tool_iterations=20,
        use_llm_summarizer=False,
    )
    built = agent.build()
    ctx = RunContext(max_steps=20)

    async def _run():
        return await built.arun("go", context=ctx)

    task = asyncio.create_task(_run())
    for _ in range(100):
        if built._active_ctx is ctx and provider.calls >= 1:
            break
        await asyncio.sleep(0.01)

    # Mirror fastapi_adapter._cancel_agent resolution order.
    for target in (built, getattr(built, "_built", None), getattr(built, "_agent", None)):
        if target is None:
            continue
        cancel = getattr(target, "cancel", None)
        if callable(cancel):
            assert cancel() is True
            break

    result = await task
    assert ctx.cancelled is True
    assert (result.metadata or {}).get("stop_reason") == StopReason.CANCELLED


@pytest.mark.asyncio
async def test_run_events_checks_disconnect_and_cancels() -> None:
    """Registered SSE route cancels when request.is_disconnected becomes True."""

    cancel_hits: list[bool] = []

    class _StubAgent:
        def __init__(self) -> None:
            self._built = self

        def cancel(self) -> bool:
            cancel_hits.append(True)
            return True

        async def astream_events(self, *_a, **_k):
            yield StreamEvent(type=RUN_STARTED, run_id="r1", data={})
            await asyncio.sleep(0.05)
            yield StreamEvent(type=RUN_STARTED, run_id="r2", data={})

    app = FastAPI()
    _register_agent_routes(app, _StubAgent(), prefix="/agent")
    route = next(r for r in app.routes if getattr(r, "path", "") == "/agent/run/events")
    endpoint = route.endpoint

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/agent/run/events",
        "raw_path": b"/agent/run/events",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 50000),
        "server": ("test", 80),
    }
    request = Request(scope)
    # First event: connected; second check (before 2nd yield): disconnected.
    request.is_disconnected = AsyncMock(side_effect=[False, True, True])  # type: ignore[method-assign]

    payload = RunRequestModel(
        messages=[
            MessageModel(
                role="user",
                parts=[MediaPartModel(modality="text", text="hi")],
            )
        ]
    )

    response = await endpoint(request, payload)
    assert response.media_type == "text/event-stream"
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else str(chunk).encode())
        if cancel_hits:
            break

    assert cancel_hits, "expected cancel after is_disconnected became True"
    assert chunks, "expected at least one SSE frame before cancel"
