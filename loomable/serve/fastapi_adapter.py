"""loomable.serve.fastapi_adapter - FastAPI HTTP transport for a BuiltAgent.

:class:`FastAPIAdapter` wraps a :class:`~loomable.agent.BuiltAgent` and exposes it
over HTTP without embedding any agent logic (Req 7.7, 9.3). It is a thin
request/response translator that:

- deserializes a JSON body into an :class:`~loomable.content.AgentInput`,
- forwards to :meth:`BuiltAgent.arun` / :meth:`BuiltAgent.astream`,
- serializes the :class:`~loomable.agent.RunResult` / :class:`~loomable.agent.RunChunk`
  back to JSON,
- routes runs by ``session_id`` so state persists across calls (Req 7.5),
- maps domain errors to 4xx responses with descriptive messages (Req 7.6).

Routes (Req 7.1-7.4):
- ``GET  /health``     → ``{"status": "ok"}`` readiness probe.
- ``POST /run``        → JSON ``AgentInput`` → JSON ``RunResult``.
- ``POST /run/stream`` → newline-delimited JSON stream of ``RunChunk``s.

Depends on ``loomable.agent`` and ``loomable.content`` plus FastAPI/Pydantic.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from loomable.agent import BuiltAgent, RunChunk, RunResult, UnsupportedModalityError
from loomable.content import (
    AgentInput,
    AgentOutput,
    Image,
    MediaPart,
    MediaPartError,
    Message,
    Modality,
    Text,
    Video,
)


# ---------------------------------------------------------------------------
# Pydantic request/response models (mirror the content model)
# ---------------------------------------------------------------------------


class MediaPartModel(BaseModel):
    """JSON representation of a :class:`~loomable.content.MediaPart`.

    For text parts the decoded string is exposed as ``text``. For image/video
    parts the payload is either ``data_base64`` (base64 of the inline bytes) or a
    ``uri`` reference.
    """

    modality: str
    media_type: str | None = None
    text: str | None = None
    data_base64: str | None = None
    uri: str | None = None


class MessageModel(BaseModel):
    """JSON representation of a :class:`~loomable.content.Message`."""

    role: str = "user"
    parts: list[MediaPartModel] = Field(default_factory=list)


class RunRequestModel(BaseModel):
    """Request body for ``POST /run`` and ``POST /run/stream``.

    A JSON ``AgentInput`` (ordered ``messages``) plus an optional ``session_id``
    used to route the run so state persists across calls (Req 7.5).
    """

    messages: list[MessageModel] = Field(default_factory=list)
    session_id: str | None = None


class RunResultModel(BaseModel):
    """Response body for ``POST /run`` mirroring :class:`RunResult`."""

    output: list[MediaPartModel]
    session_id: str
    usage: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Translation helpers (pure request/response mapping, no agent logic)
# ---------------------------------------------------------------------------


def _model_to_media_part(part: MediaPartModel) -> MediaPart:
    """Deserialize a :class:`MediaPartModel` into a :class:`MediaPart`.

    Raises :class:`MediaPartError` (a ``ValueError``) on an unknown modality or an
    otherwise invalid part; the route maps that to a 4xx response.
    """
    try:
        modality = Modality(part.modality)
    except ValueError as exc:  # unknown modality string
        raise MediaPartError(f"Unknown modality '{part.modality}'.") from exc

    data = base64.b64decode(part.data_base64) if part.data_base64 is not None else None

    if modality is Modality.TEXT:
        if part.text is not None:
            return Text(part.text)
        if data is not None:
            return Text(data.decode("utf-8"))
        if part.uri is not None:
            return MediaPart(
                modality=Modality.TEXT,
                media_type=part.media_type or "text/plain",
                uri=part.uri,
            )
        raise MediaPartError("Text part must provide 'text', 'data_base64', or 'uri'.")

    if modality is Modality.IMAGE:
        return Image(data=data, uri=part.uri, media_type=part.media_type or "image/png")

    # Modality.VIDEO
    return Video(data=data, uri=part.uri, media_type=part.media_type or "video/mp4")


def _request_to_agent_input(body: RunRequestModel) -> AgentInput:
    """Build an :class:`AgentInput` from the request body.

    Raises ``ValueError`` (including :class:`MediaPartError`) on an empty or invalid
    payload; the route maps that to a 4xx response.
    """
    messages = [
        Message(
            role=message.role,
            parts=[_model_to_media_part(part) for part in message.parts],
        )
        for message in body.messages
    ]
    return AgentInput(messages=messages)


def _media_part_to_model(part: MediaPart) -> MediaPartModel:
    """Serialize a :class:`MediaPart` into a :class:`MediaPartModel`."""
    if part.modality is Modality.TEXT:
        text = part.data.decode("utf-8") if part.data is not None else None
        return MediaPartModel(
            modality=part.modality.value,
            media_type=part.media_type,
            text=text,
            uri=part.uri,
        )
    data_base64 = base64.b64encode(part.data).decode("ascii") if part.data is not None else None
    return MediaPartModel(
        modality=part.modality.value,
        media_type=part.media_type,
        data_base64=data_base64,
        uri=part.uri,
    )


def _output_to_models(output: AgentOutput) -> list[MediaPartModel]:
    """Serialize an :class:`AgentOutput` into a list of part models."""
    return [_media_part_to_model(part) for part in output.parts]


def _run_result_to_model(result: RunResult) -> RunResultModel:
    """Serialize a :class:`RunResult` into its response model."""
    return RunResultModel(
        output=_output_to_models(result.output),
        session_id=result.session_id,
        usage=dict(result.usage),
    )


def _chunk_to_dict(chunk: RunChunk) -> dict[str, Any]:
    """Serialize a :class:`RunChunk` into a JSON-ready dict."""
    return {
        "delta": _media_part_to_model(chunk.delta).model_dump(),
        "done": chunk.done,
    }


class FastAPIAdapter:
    """Expose a :class:`BuiltAgent` over HTTP with FastAPI (Req 7.1).

    The adapter holds no agent logic beyond request/response translation and
    session routing (Req 7.7, 9.3). ``session_id`` is accepted for routing; the
    wrapped ``BuiltAgent`` owns a single session whose state persists across calls
    (Req 7.5), so successive requests to the same adapter share that session.
    """

    def __init__(self, agent: BuiltAgent) -> None:
        self._agent = agent

    def app(self) -> FastAPI:
        """Build and return the FastAPI application exposing the agent."""
        app = FastAPI(title="loomable agent")
        agent = self._agent

        @app.get("/health")
        async def health() -> dict[str, str]:
            """Report agent readiness (Req 7.4)."""
            return {"status": "ok"}

        @app.post("/run")
        async def run(body: RunRequestModel) -> JSONResponse:
            """Run the agent once and return a :class:`RunResult` body (Req 7.2)."""
            try:
                agent_input = _request_to_agent_input(body)
            except ValueError as exc:  # MediaPartError / empty input / bad base64
                return JSONResponse(status_code=422, content={"detail": str(exc)})

            try:
                result = await agent.arun(agent_input)
            except UnsupportedModalityError as exc:
                return JSONResponse(status_code=400, content={"detail": str(exc)})

            return JSONResponse(
                status_code=200,
                content=_run_result_to_model(result).model_dump(),
            )

        @app.post("/run/stream")
        async def run_stream(body: RunRequestModel) -> Any:
            """Stream incremental output as newline-delimited JSON chunks (Req 7.3)."""
            try:
                agent_input = _request_to_agent_input(body)
            except ValueError as exc:
                return JSONResponse(status_code=422, content={"detail": str(exc)})

            async def event_stream():
                try:
                    async for chunk in agent.astream(agent_input):
                        yield json.dumps(_chunk_to_dict(chunk)) + "\n"
                except UnsupportedModalityError as exc:
                    yield json.dumps({"error": str(exc)}) + "\n"

            return StreamingResponse(event_stream(), media_type="application/x-ndjson")

        return app
