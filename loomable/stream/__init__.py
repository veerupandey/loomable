"""AG-UI-compatible realtime stream protocol.

Zero FastAPI dependency. Transport adapters (SSE) live in ``loomable.serve``.

AG-UI clients consume these event types over ``text/event-stream`` without a
Loomable-specific client.
"""

from __future__ import annotations

__all__ = [
    "StreamEvent",
    "StreamEventType",
    "AsyncStreamBus",
    "StreamBridge",
    "sse_encode",
    "RUN_STARTED",
    "RUN_FINISHED",
    "RUN_ERROR",
    "TEXT_MESSAGE_START",
    "TEXT_MESSAGE_CONTENT",
    "TEXT_MESSAGE_END",
    "TOOL_CALL_START",
    "TOOL_CALL_ARGS",
    "TOOL_CALL_END",
    "TOOL_CALL_RESULT",
    "NODE_STARTED",
    "NODE_FINISHED",
    "STATE_SNAPSHOT",
    "STATE_DELTA",
    "THINKING_CONTENT",
]

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Literal

from loomable.agent.events import AgentEvents, Event, NoOpEvents

# AG-UI-aligned type constants
RUN_STARTED = "RUN_STARTED"
RUN_FINISHED = "RUN_FINISHED"
RUN_ERROR = "RUN_ERROR"
TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
TOOL_CALL_START = "TOOL_CALL_START"
TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
TOOL_CALL_END = "TOOL_CALL_END"
TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
NODE_STARTED = "NODE_STARTED"
NODE_FINISHED = "NODE_FINISHED"
STATE_SNAPSHOT = "STATE_SNAPSHOT"
STATE_DELTA = "STATE_DELTA"
THINKING_CONTENT = "THINKING_CONTENT"

StreamEventType = Literal[
    "RUN_STARTED",
    "RUN_FINISHED",
    "RUN_ERROR",
    "TEXT_MESSAGE_START",
    "TEXT_MESSAGE_CONTENT",
    "TEXT_MESSAGE_END",
    "TOOL_CALL_START",
    "TOOL_CALL_ARGS",
    "TOOL_CALL_END",
    "TOOL_CALL_RESULT",
    "NODE_STARTED",
    "NODE_FINISHED",
    "STATE_SNAPSHOT",
    "STATE_DELTA",
    "THINKING_CONTENT",
]


@dataclass
class StreamEvent:
    """One AG-UI-compatible event in a realtime agent stream."""

    type: str
    run_id: str
    session_id: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "data": self.data,
        }


def sse_encode(event: StreamEvent) -> bytes:
    """Encode a :class:`StreamEvent` as one SSE frame (``event:`` + ``data:``)."""
    payload = json.dumps(event.to_dict(), default=str, ensure_ascii=False)
    return f"event: {event.type}\ndata: {payload}\n\n".encode("utf-8")


class AsyncStreamBus:
    """Async multicast queue for :class:`StreamEvent` frames."""

    def __init__(self, run_id: str = "", session_id: str = "") -> None:
        self.run_id = run_id
        self.session_id = session_id
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._closed = False

    async def publish(self, event: StreamEvent) -> None:
        if self._closed:
            return
        await self._queue.put(event)

    async def emit(self, event: StreamEvent) -> None:
        """Alias for :meth:`publish`."""
        await self.publish(event)

    def publish_nowait(self, event: StreamEvent) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def emit_sync(self, event: StreamEvent) -> None:
        """Sync publish (for tool callbacks / AgentEvents bridges)."""
        self.publish_nowait(event)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)

    async def close(self) -> None:
        """Close the bus (async). Also callable as fire-and-forget from sync contexts."""
        await self.aclose()

    def close_sync(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def events(self) -> AsyncIterator[StreamEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self.events()


class StreamBridge:
    """Adapt sync :class:`AgentEvents` into AG-UI :class:`StreamEvent` publishes.

    Wraps an inner tracer (optional) so existing observability keeps working.
    """

    def __init__(
        self,
        bus: AsyncStreamBus,
        *,
        run_id: str,
        session_id: str = "",
        inner: AgentEvents | None = None,
    ) -> None:
        self._bus = bus
        self._run_id = run_id
        self._session_id = session_id
        self._inner = inner or NoOpEvents()
        self.trace: list[dict[str, Any]] = getattr(self._inner, "trace", [])

    def emit(self, event: Event) -> None:
        self._inner.emit(event)
        mapped = self._map(event)
        for se in mapped:
            self._bus.publish_nowait(se)

    def publish(self, type_: str, data: dict[str, Any] | None = None) -> None:
        self._bus.publish_nowait(
            StreamEvent(
                type=type_,
                run_id=self._run_id,
                session_id=self._session_id,
                data=data or {},
            )
        )

    def _map(self, event: Event) -> list[StreamEvent]:
        attrs = dict(event.attributes or {})
        base = {
            "run_id": self._run_id,
            "session_id": self._session_id,
        }
        if event.kind == "run_start":
            return [
                StreamEvent(
                    type=RUN_STARTED,
                    **base,
                    data={"attributes": attrs},
                )
            ]
        if event.kind == "run_end":
            # Caller emits RUN_FINISHED after final TEXT_* frames.
            return []
        if event.kind == "tool_call":
            # Prefer structured per-call payloads when present (START/ARGS/RESULT/END
            # already published explicitly around dispatch).
            if attrs.get("agui_skip"):
                return []
            calls = attrs.get("tool_calls")
            if isinstance(calls, list) and calls:
                out: list[StreamEvent] = []
                results_by_id = {
                    str(r.get("call_id")): r
                    for r in (attrs.get("results") or [])
                    if isinstance(r, dict)
                }
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    cid = str(call.get("id") or "")
                    name = str(call.get("tool_name") or call.get("name") or "tool")
                    args = call.get("args") if isinstance(call.get("args"), dict) else {}
                    out.append(
                        StreamEvent(
                            type=TOOL_CALL_START,
                            **base,
                            data={"tool_call_id": cid, "tool_name": name},
                        )
                    )
                    out.append(
                        StreamEvent(
                            type=TOOL_CALL_ARGS,
                            **base,
                            data={"tool_call_id": cid, "tool_name": name, "args": args},
                        )
                    )
                    res = results_by_id.get(cid)
                    if res is not None:
                        out.append(
                            StreamEvent(
                                type=TOOL_CALL_RESULT,
                                **base,
                                data={
                                    "tool_call_id": cid,
                                    "tool_name": name,
                                    "content": res.get("content", ""),
                                    "is_error": bool(res.get("is_error")),
                                },
                            )
                        )
                    out.append(
                        StreamEvent(
                            type=TOOL_CALL_END,
                            **base,
                            data={
                                "tool_call_id": cid,
                                "tool_name": name,
                                "duration_ms": event.duration_ms,
                            },
                        )
                    )
                return out
            # Legacy collapsed form (name list only)
            names = str(attrs.get("gen_ai.tool.name") or attrs.get("tool_count") or "")
            tool_names = [n for n in names.split(",") if n]
            out = []
            for name in tool_names or ["tool"]:
                out.append(
                    StreamEvent(
                        type=TOOL_CALL_START,
                        **base,
                        data={"tool_name": name, "attributes": attrs},
                    )
                )
                out.append(
                    StreamEvent(
                        type=TOOL_CALL_END,
                        **base,
                        data={
                            "tool_name": name,
                            "duration_ms": event.duration_ms,
                            "attributes": attrs,
                        },
                    )
                )
            return out
        if event.kind == "model_call":
            return []  # text streaming handled separately when available
        if event.kind == "node_start":
            return [
                StreamEvent(
                    type=NODE_STARTED,
                    **base,
                    data={"node_id": attrs.get("node_id"), "attributes": attrs},
                )
            ]
        if event.kind == "node_end":
            return [
                StreamEvent(
                    type=NODE_FINISHED,
                    **base,
                    data={
                        "node_id": attrs.get("node_id"),
                        "duration_ms": event.duration_ms,
                        "attributes": attrs,
                    },
                )
            ]
        return []
