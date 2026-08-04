"""Integration tests for the FastAPI transport adapter (task 5.1).

Drives a real ``BuiltAgent`` (built with a fake in-process ``ModelProvider``)
through a ``fastapi.testclient.TestClient`` and asserts:

- ``GET /health`` reports readiness (Req 7.4).
- ``POST /run`` returns a well-formed ``RunResult`` for a text input (Req 7.2).
- ``POST /run/stream`` streams chunks (Req 7.3).
- a malformed / unsupported-modality payload yields a 4xx (Req 7.6).

Feature: agent-api
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from loomable.agent import Agent, ModelSpec
from loomable.content import Modality, ModelCapabilities
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.serve import FastAPIAdapter


class EchoProvider:
    """Fake in-process ModelProvider echoing the last text part it received."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        last_text = ""
        for message in request.messages:
            for part in message.get("content", []):
                if part.get("type") == "text":
                    last_text = part.get("text", "")
        return ModelResponse(content=f"echo: {last_text}", usage={"input": 1, "output": 2})


def _build_client(*, capabilities: ModelCapabilities | None = None) -> TestClient:
    agent = Agent(
        model=ModelSpec(provider="fake", provider_impl=EchoProvider()),
        capabilities=capabilities,
    ).build()
    app = FastAPIAdapter(agent).app()
    return TestClient(app)


@pytest.mark.integration
def test_health_reports_ready() -> None:
    """GET /health returns 200 and a readiness status (Req 7.4)."""
    client = _build_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_run_returns_well_formed_run_result() -> None:
    """POST /run returns a RunResult body for a text input (Req 7.2)."""
    client = _build_client()
    payload = {
        "messages": [{"role": "user", "parts": [{"modality": "text", "text": "hello"}]}]
    }
    response = client.post("/run", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["usage"] == {"input": 1, "output": 2}
    assert len(body["output"]) == 1
    part = body["output"][0]
    assert part["modality"] == "text"
    assert part["text"] == "echo: hello"


@pytest.mark.integration
def test_run_stream_yields_chunks() -> None:
    """POST /run/stream streams newline-delimited RunChunks (Req 7.3)."""
    client = _build_client()
    payload = {
        "messages": [{"role": "user", "parts": [{"modality": "text", "text": "hi"}]}]
    }
    with client.stream("POST", "/run/stream", json=payload) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    assert lines
    import json

    chunks = [json.loads(line) for line in lines]
    assert chunks[-1]["done"] is True
    assert chunks[-1]["delta"]["text"] == "echo: hi"


@pytest.mark.integration
def test_unsupported_modality_yields_4xx() -> None:
    """An image input to a text-only model yields a 400 naming the problem (Req 7.6)."""
    client = _build_client(
        capabilities=ModelCapabilities(
            input=frozenset({Modality.TEXT}), output=frozenset({Modality.TEXT})
        )
    )
    data_b64 = base64.b64encode(b"\x89PNG").decode("ascii")
    payload = {
        "messages": [
            {
                "role": "user",
                "parts": [
                    {"modality": "image", "media_type": "image/png", "data_base64": data_b64}
                ],
            }
        ]
    }
    response = client.post("/run", json=payload)

    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


@pytest.mark.integration
def test_malformed_part_yields_4xx() -> None:
    """A part with an unknown modality yields a 422 (Req 7.6)."""
    client = _build_client()
    payload = {
        "messages": [{"role": "user", "parts": [{"modality": "hologram", "text": "x"}]}]
    }
    response = client.post("/run", json=payload)

    assert response.status_code == 422
    assert "hologram" in response.json()["detail"].lower()
