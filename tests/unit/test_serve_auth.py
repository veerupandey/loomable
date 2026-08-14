"""Serve mount api_key auth baseline."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from loomable.agent import Agent, ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.serve import mount_agent


class _OkProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="hello")


@pytest.fixture
def agent() -> Agent:
    return Agent(
        model=ModelSpec(provider="scripted", provider_impl=_OkProvider()),
        use_llm_summarizer=False,
    )


def test_mount_without_api_key_allows_anonymous(agent: Agent) -> None:
    app = FastAPI()
    mount_agent(app, agent, prefix="/agent")
    client = TestClient(app)
    assert client.get("/agent/health").status_code == 200


def test_mount_api_key_rejects_anonymous(agent: Agent) -> None:
    app = FastAPI()
    mount_agent(app, agent, prefix="/agent", api_key="secret-key")
    client = TestClient(app)
    r = client.get("/agent/health")
    assert r.status_code == 401
    assert r.json()["detail"] == "unauthorized"


def test_mount_api_key_accepts_bearer(agent: Agent) -> None:
    app = FastAPI()
    mount_agent(app, agent, prefix="/agent", api_key="secret-key")
    client = TestClient(app)
    r = client.get(
        "/agent/health", headers={"Authorization": "Bearer secret-key"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_mount_api_key_accepts_x_api_key_and_run(agent: Agent) -> None:
    app = FastAPI()
    mount_agent(app, agent, prefix="/agent", api_key="secret-key")
    client = TestClient(app)
    r = client.post(
        "/agent/run",
        headers={"X-API-Key": "secret-key"},
        json={"messages": [{"role": "user", "parts": [{"modality": "text", "text": "hi"}]}]},
    )
    assert r.status_code == 200
