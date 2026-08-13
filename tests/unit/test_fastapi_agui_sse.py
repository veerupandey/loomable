"""FastAPI AG-UI SSE tests for Agent."""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.serve import FastAPIAdapter, mount_agent


class _Echo:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="sse-hi", usage={"input_tokens": 1, "output_tokens": 1})


@pytest.fixture
def agent() -> Agent:
    return Agent(
        model=ModelSpec(provider="scripted", provider_impl=_Echo()),
        modalities="text",
    )


@pytest.mark.asyncio
async def test_mount_agent_agui_sse(agent: Agent) -> None:
    fastapi = pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI

    app = FastAPI()
    mount_agent(app, agent, prefix="/agent")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/agent/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        body = {
            "messages": [
                {"role": "user", "parts": [{"modality": "text", "text": "hello"}]}
            ]
        }
        resp = await client.post("/agent/run/events", json=body)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        text = resp.text
        assert "event: RUN_STARTED" in text
        assert "event: TEXT_MESSAGE_CONTENT" in text or "sse-hi" in text
        assert "event: RUN_FINISHED" in text


@pytest.mark.asyncio
async def test_fastapi_adapter_root_and_agent_prefix(agent: Agent) -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")

    app = FastAPIAdapter(agent).app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.get("/health")
        r2 = await client.get("/agent/health")
        assert r1.status_code == 200 and r2.status_code == 200
        body = {
            "messages": [
                {"role": "user", "parts": [{"modality": "text", "text": "hi"}]}
            ]
        }
        run = await client.post("/run", json=body)
        assert run.status_code == 200
        assert "sse-hi" in run.text


@pytest.mark.asyncio
async def test_mount_case_agui_sse() -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI
    from loomable import Case
    from loomable.serve import mount_case

    case = Case(
        model=ModelSpec(provider="scripted", provider_impl=_Echo()),
        goal="triage",
        board=True,
        dispatch="reuse",
        max_rounds=1,
        max_steps=1,
        modalities="text",
    )
    app = FastAPI()
    mount_case(app, case, prefix="/cases")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "messages": [
                {"role": "user", "parts": [{"modality": "text", "text": "hello"}]}
            ]
        }
        resp = await client.post("/cases/run/events", json=body)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        assert "event: RUN_STARTED" in resp.text
        assert "event: RUN_FINISHED" in resp.text
