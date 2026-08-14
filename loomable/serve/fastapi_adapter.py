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

import asyncio
import base64
import json
import logging
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


def _extract_request_api_key(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip() or None
    xkey = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if xkey:
        return xkey.strip() or None
    return None


def _supports_ndjson_stream(agent: Any) -> bool:
    """True when ``POST /run/stream`` can call ``astream`` without a guaranteed fail."""
    if not hasattr(agent, "astream"):
        return False
    # Agent(mode="case") defines astream but raises — same class of lie as Case.
    if getattr(agent, "_mode", None) == "case":
        return False
    return True


def _register_agent_routes(
    app: FastAPI,
    agent: Any,
    *,
    prefix: str = "",
    api_key: str | None = None,
) -> None:
    """Register health / run / NDJSON stream / AG-UI SSE routes on ``app``."""
    p = prefix.rstrip("/")
    expected_key = (api_key or "").strip() or None

    def _auth_or_401(request: Request) -> JSONResponse | None:
        if expected_key is None:
            return None
        provided = _extract_request_api_key(request)
        if provided is None or provided != expected_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "unauthorized"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None

    def _cancel_agent() -> None:
        for target in (agent, getattr(agent, "_built", None), getattr(agent, "_agent", None)):
            if target is None:
                continue
            cancel = getattr(target, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception as exc:  # noqa: BLE001
                    logging.getLogger("loomable.serve").debug(
                        "cancel() failed on %s: %s", type(target).__name__, exc
                    )

    def _apply_session(body: RunRequestModel) -> None:
        sid = body.session_id
        if not sid:
            return
        # Prefer bind_session so Agent L1/L2 and Case checkpoints stay aligned.
        if hasattr(agent, "bind_session") and callable(getattr(agent, "bind_session")):
            try:
                agent.bind_session(sid)
                return
            except Exception as exc:  # noqa: BLE001
                logging.getLogger("loomable.serve").warning(
                    "bind_session(%r) failed; falling back to session_id assign: %s",
                    sid,
                    exc,
                )
        if hasattr(agent, "session_id"):
            try:
                agent.session_id = sid  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logging.getLogger("loomable.serve").debug(
                    "session_id assign failed: %s", exc
                )
        if hasattr(agent, "_session_id"):
            try:
                agent._session_id = sid  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logging.getLogger("loomable.serve").debug(
                    "_session_id assign failed: %s", exc
                )
        # Case (direct or cached under Agent mode=case) must bind checkpoint thread.
        case = agent if type(agent).__name__ == "Case" else getattr(agent, "_case", None)
        if case is not None and hasattr(case, "bind_session"):
            case.bind_session(sid)
        elif case is not None and hasattr(case, "session_id"):
            case.session_id = sid
            if hasattr(case, "_kwargs") and isinstance(case._kwargs, dict):
                case._kwargs["session_id"] = sid
                wf = getattr(case, "_workflow", None)
                if wf is not None:
                    wf._session_id = sid

    async def _invoke_arun(agent_input: AgentInput, body: RunRequestModel) -> RunResult:
        _apply_session(body)
        # Case / Workflow-style runnables prefer plain text
        if type(agent).__name__ == "Case" or getattr(agent, "_mode", None) == "case":
            from loomable.agent.builder import _input_text

            return await agent.arun(_input_text(agent_input))
        return await agent.arun(agent_input)

    async def _invoke_astream_events(agent_input: AgentInput, body: RunRequestModel):
        _apply_session(body)
        kwargs: dict[str, Any] = {}
        if body.session_id:
            kwargs["session_id"] = body.session_id
        if type(agent).__name__ == "Case" or getattr(agent, "_mode", None) == "case":
            from loomable.agent.builder import _input_text

            text = _input_text(agent_input)
            if "session_id" in kwargs:
                async for event in agent.astream_events(text, session_id=kwargs["session_id"]):
                    yield event
            else:
                async for event in agent.astream_events(text):
                    yield event
            return
        async for event in agent.astream_events(agent_input):
            yield event

    @app.get(f"{p}/health")
    async def health(request: Request) -> Any:
        denied = _auth_or_401(request)
        if denied is not None:
            return denied
        return {"status": "ok"}

    @app.post(f"{p}/run")
    async def run(request: Request, body: RunRequestModel) -> JSONResponse:
        denied = _auth_or_401(request)
        if denied is not None:
            return denied
        try:
            agent_input = _request_to_agent_input(body)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        try:
            result = await _invoke_arun(agent_input, body)
        except UnsupportedModalityError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return JSONResponse(
            status_code=200,
            content=_run_result_to_model(result).model_dump(),
        )

    # NDJSON token chunks require a usable ``astream`` (Agent / BuiltAgent without
    # mode="case"). Case / Team / Workflow / case-mode Agent expose AG-UI via
    # ``astream_events`` only — do not register a lying route.
    if _supports_ndjson_stream(agent):

        @app.post(f"{p}/run/stream")
        async def run_stream(request: Request, body: RunRequestModel) -> Any:
            denied = _auth_or_401(request)
            if denied is not None:
                return denied
            try:
                agent_input = _request_to_agent_input(body)
            except ValueError as exc:
                return JSONResponse(status_code=422, content={"detail": str(exc)})
            _apply_session(body)

            async def event_stream():
                try:
                    async for chunk in agent.astream(agent_input):
                        if await request.is_disconnected():
                            _cancel_agent()
                            break
                        yield json.dumps(_chunk_to_dict(chunk)) + "\n"
                except UnsupportedModalityError as exc:
                    yield json.dumps({"error": str(exc)}) + "\n"
                except asyncio.CancelledError:
                    _cancel_agent()
                    raise
                except Exception as exc:  # noqa: BLE001
                    yield json.dumps({"error": str(exc)}) + "\n"
                finally:
                    if await request.is_disconnected():
                        _cancel_agent()

            return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @app.post(f"{p}/run/events")
    async def run_events(request: Request, body: RunRequestModel) -> Any:
        """AG-UI-compatible Server-Sent Events stream."""
        from loomable.stream import sse_encode

        denied = _auth_or_401(request)
        if denied is not None:
            return denied
        try:
            agent_input = _request_to_agent_input(body)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})

        if not hasattr(agent, "astream_events"):
            return JSONResponse(
                status_code=501,
                content={"detail": "agent does not support astream_events"},
            )

        async def sse_stream():
            from loomable.stream import RUN_ERROR, StreamEvent

            try:
                async for event in _invoke_astream_events(agent_input, body):
                    if await request.is_disconnected():
                        _cancel_agent()
                        break
                    yield sse_encode(event)
            except UnsupportedModalityError as exc:
                yield sse_encode(
                    StreamEvent(
                        type=RUN_ERROR,
                        run_id="error",
                        data={"message": str(exc)},
                    )
                )
            except asyncio.CancelledError:
                _cancel_agent()
                raise
            except Exception as exc:  # noqa: BLE001
                yield sse_encode(
                    StreamEvent(
                        type=RUN_ERROR,
                        run_id="error",
                        data={"message": str(exc), "error_type": type(exc).__name__},
                    )
                )
            finally:
                if await request.is_disconnected():
                    _cancel_agent()

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


def mount_agent(
    app: FastAPI,
    agent: Any,
    *,
    prefix: str = "/agent",
    api_key: str | None = None,
) -> FastAPI:
    """Mount Agent AG-UI routes on an existing FastAPI app.

    Routes: ``{prefix}/health``, ``{prefix}/run``, ``{prefix}/run/events`` (SSE).
    ``{prefix}/run/stream`` (NDJSON) is registered only when the target exposes
    ``astream`` (Agent / BuiltAgent).

    When ``api_key`` is set, require ``Authorization: Bearer <key>`` or
    ``X-API-Key: <key>`` on all routes (including health).
    """
    # High-level Agent → BuiltAgent when needed for legacy adapter compat
    target = agent
    if hasattr(agent, "build") and not hasattr(agent, "astream_events"):
        target = agent.build()
    elif hasattr(agent, "_get_built") and not hasattr(agent, "astream_events"):
        target = agent._get_built()
    _register_agent_routes(app, target, prefix=prefix, api_key=api_key)
    return app


def mount_case(
    app: FastAPI,
    case: Any,
    *,
    prefix: str = "/cases",
    api_key: str | None = None,
) -> FastAPI:
    """Mount Case routes: health, ``/run``, AG-UI ``/run/events``.

    NDJSON ``/run/stream`` is **not** registered — Case has no ``astream``.
    Prefer ``POST {prefix}/run/events`` (SSE).
    """
    _register_agent_routes(app, case, prefix=prefix, api_key=api_key)
    return app


class FastAPIAdapter:
    """Expose a :class:`BuiltAgent` or :class:`Agent` over HTTP with FastAPI.

    Routes (Req 7.1-7.4 + AG-UI SSE):
    - ``GET  /health``
    - ``POST /run``
    - ``POST /run/stream`` (NDJSON)
    - ``POST /run/events`` (``text/event-stream`` AG-UI events)

    Optional ``api_key`` enables Bearer / ``X-API-Key`` auth on all routes.
    """

    def __init__(self, agent: Any, *, api_key: str | None = None) -> None:
        # Accept BuiltAgent or high-level Agent
        if hasattr(agent, "astream_events"):
            self._agent = agent
        elif hasattr(agent, "_get_built"):
            self._agent = agent  # Agent has astream_events after our change
        elif hasattr(agent, "build"):
            self._agent = agent.build()
        else:
            self._agent = agent
        self._api_key = api_key

    def app(self) -> FastAPI:
        """Build and return the FastAPI application exposing the agent."""
        app = FastAPI(title="loomable agent")
        _register_agent_routes(app, self._agent, prefix="", api_key=self._api_key)
        # Also expose under /agent for enterprise convention
        _register_agent_routes(app, self._agent, prefix="/agent", api_key=self._api_key)
        return app
