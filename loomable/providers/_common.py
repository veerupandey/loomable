"""loomable.providers._common - Shared translation helpers for HTTP providers.

These helpers convert the provider-agnostic :class:`~loomable.kernel.models.ModelRequest`
message shape (produced by ``loomable.content.to_model_request``) into the concrete
wire formats used by OpenAI-compatible and Anthropic APIs, and parse responses back
into :class:`~loomable.kernel.models.ModelResponse`. Keeping the translation here lets
the OpenAI, Azure, and Anthropic providers stay small and share one implementation.
"""

from __future__ import annotations

import base64
import json
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from loomable.kernel.errors import ModelProviderError
from loomable.kernel.models import ModelResponse, StreamEvent, ToolCall

from .errors import PermanentProviderError, TransientProviderError


# ---------------------------------------------------------------------------
# HTTP error classification helpers
# ---------------------------------------------------------------------------


def _parse_retry_after(header: str | None) -> float | None:
    """Parse a Retry-After header (integer seconds or HTTP date) into seconds.

    Returns ``None`` when the header is absent or unparseable.
    """
    if header is None:
        return None
    # Try integer seconds first.
    try:
        value = int(header)
        return float(value) if value >= 0 else None
    except ValueError:
        pass
    # Try float seconds.
    try:
        value_f = float(header)
        return value_f if value_f >= 0 else None
    except ValueError:
        pass
    # Try HTTP-date format (e.g. "Fri, 31 Dec 2021 23:59:59 GMT").
    try:
        from datetime import datetime, timezone

        target = parsedate_to_datetime(header)
        now = datetime.now(timezone.utc)
        delta = (target - now).total_seconds()
        return max(delta, 0.0)
    except (ValueError, TypeError, OverflowError):
        return None


def _classify_http_error(provider_id: str, exc: httpx.HTTPError) -> ModelProviderError:
    """Map an httpx error to a Transient/Permanent provider error carrying status.

    - Timeouts, ConnectError, ReadError, RemoteProtocolError → TransientProviderError(status_code=None)
    - HTTPStatusError with 429 or 5xx → TransientProviderError with status_code and parsed retry_after
    - HTTPStatusError with other 4xx → PermanentProviderError with status_code
    - Unknown httpx errors → TransientProviderError(status_code=None)
    """
    if isinstance(
        exc,
        (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError),
    ):
        return TransientProviderError(provider_id, status_code=None)
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429 or 500 <= code < 600:
            retry_after = _parse_retry_after(exc.response.headers.get("retry-after"))
            return TransientProviderError(provider_id, status_code=code, retry_after=retry_after)
        return PermanentProviderError(provider_id, status_code=code)  # 4xx / auth / policy
    return TransientProviderError(provider_id, status_code=None)  # unknown network error


# ---------------------------------------------------------------------------
# Content-part helpers
# ---------------------------------------------------------------------------


def _content_is_all_text(content: Any) -> bool:
    """True when a message ``content`` is a list of only text parts."""
    return isinstance(content, list) and all(
        isinstance(p, dict) and p.get("type") == "text" for p in content
    )


def _join_text(content: list[dict[str, Any]]) -> str:
    """Concatenate the ``text`` fields of a content-part list."""
    return "".join(part.get("text", "") for part in content)


def _parse_data_uri(url: str) -> tuple[str, bytes] | None:
    """Parse a ``data:<media_type>;base64,<payload>`` URI into (media_type, bytes).

    Returns ``None`` when ``url`` is not a base64 data URI (e.g. a normal https URL).
    """
    if not url.startswith("data:"):
        return None
    try:
        header, encoded = url.split(",", 1)
        media_type = header[len("data:") :].split(";", 1)[0] or "application/octet-stream"
        return media_type, base64.b64decode(encoded)
    except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
        return None


# ---------------------------------------------------------------------------
# OpenAI-compatible translation
# ---------------------------------------------------------------------------


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate provider-agnostic messages to OpenAI chat-completions messages.

    - A message whose ``content`` is a plain string is passed through.
    - A message whose ``content`` is a list of only text parts is flattened to a
      single string (maximally compatible across OpenAI-compatible servers).
    - A message with mixed/multimodal parts keeps the content-array form, retaining
      ``text`` and ``image_url`` parts (the types the OpenAI schema supports) and
      dropping any unsupported part types (e.g. ``video_url``).
    - Tool-related fields (``tool_calls`` on assistant messages and ``tool_call_id``
      on tool messages) are preserved so the tool-use loop works end-to-end.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        # Build the base translated message.
        if isinstance(content, str):
            translated: dict[str, Any] = {"role": role, "content": content}
        elif _content_is_all_text(content):
            translated = {"role": role, "content": _join_text(content)}
        else:
            # Mixed content: keep text and image_url parts, drop unsupported types.
            kept = [
                part
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "image_url"}
            ]
            translated = {"role": role, "content": kept}

        # Preserve tool_calls on assistant messages (required for the tool-use loop).
        if "tool_calls" in message:
            raw_calls = message["tool_calls"]
            # Normalize to OpenAI wire format: each call has id, type, function.
            openai_calls = []
            for tc in raw_calls:
                if "function" in tc:
                    # Already in OpenAI format.
                    openai_calls.append(tc)
                else:
                    # Internal format: {id, tool_name, args} -> OpenAI format.
                    openai_calls.append({
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("tool_name", ""),
                            "arguments": json.dumps(tc.get("args", {})),
                        },
                    })
            translated["tool_calls"] = openai_calls

        # Preserve tool_call_id on tool-role messages (required for the tool-use loop).
        if "tool_call_id" in message:
            translated["tool_call_id"] = message["tool_call_id"]

        out.append(translated)
    return out


def parse_openai_response(data: dict[str, Any]) -> ModelResponse:
    """Parse an OpenAI/Azure chat-completions response into a ``ModelResponse``."""
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}

    tool_calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function", {})
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": args_raw}
        tool_calls.append(
            ToolCall(id=raw.get("id", ""), tool_name=fn.get("name", ""), args=args)
        )

    usage = data.get("usage", {}) or {}
    return ModelResponse(
        content=message.get("content") or "",
        tool_calls=tool_calls,
        usage={
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        metadata={"model": data.get("model", "")},
    )


# ---------------------------------------------------------------------------
# Anthropic translation
# ---------------------------------------------------------------------------


def split_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split provider-agnostic messages into an Anthropic ``(system, messages)`` pair.

    Anthropic takes the system prompt as a separate top-level parameter rather than a
    message role, so system messages are extracted and concatenated. The remaining
    user/assistant messages are converted to Anthropic content blocks: ``text`` parts
    become ``{"type": "text", ...}`` and ``image_url`` parts become an ``image`` block
    using a base64 ``source`` (for data URIs) or a ``url`` ``source`` (for http URLs).
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            system_parts.append(content if isinstance(content, str) else _text_of(content))
            continue

        converted.append({"role": role, "content": _to_anthropic_blocks(content)})

    return "\n".join(p for p in system_parts if p), converted


def _text_of(content: Any) -> str:
    """Best-effort text extraction from a string or content-part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _join_text([p for p in content if isinstance(p, dict)])
    return ""


def _to_anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    """Convert message content into a list of Anthropic content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    blocks: list[dict[str, Any]] = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            parsed = _parse_data_uri(url)
            if parsed is not None:
                media_type, raw = parsed
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(raw).decode("ascii"),
                        },
                    }
                )
            elif url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    # Anthropic requires at least one block per message.
    return blocks or [{"type": "text", "text": ""}]


def parse_anthropic_response(data: dict[str, Any]) -> ModelResponse:
    """Parse an Anthropic Messages API response into a ``ModelResponse``."""
    text = "".join(
        block.get("text", "")
        for block in data.get("content", []) or []
        if isinstance(block, dict) and block.get("type") == "text"
    )
    usage = data.get("usage", {}) or {}
    return ModelResponse(
        content=text,
        usage={
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
        metadata={"model": data.get("model", "")},
    )


# ---------------------------------------------------------------------------
# SSE streaming helpers (OpenAI / Azure)
# ---------------------------------------------------------------------------


def parse_openai_sse_line(line: str) -> dict[str, Any] | None:
    """Parse a single SSE data line from an OpenAI streaming response.

    Returns the parsed JSON dict, or None for non-data lines and [DONE].
    """
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None


def iter_openai_stream_events(
    chunks: list[dict[str, Any]],
) -> list[StreamEvent]:
    """Convert a sequence of parsed SSE chunks into StreamEvents.

    Accumulates tool_call fragments across chunks and emits assembled tool calls.
    """
    events: list[StreamEvent] = []
    # Accumulate tool call fragments: {index: {id, name, args_str}}
    tool_call_acc: dict[int, dict[str, str]] = {}

    for chunk in chunks:
        choices = chunk.get("choices", [])
        if not choices:
            # Check for usage-only final chunk
            usage = chunk.get("usage")
            if usage:
                events.append(StreamEvent(
                    kind="end",
                    usage={
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
                ))
            continue

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Text delta
        content = delta.get("content")
        if content:
            events.append(StreamEvent(kind="text", text=content))

        # Tool call fragments
        tc_list = delta.get("tool_calls")
        if tc_list:
            for tc in tc_list:
                idx = tc.get("index", 0)
                if idx not in tool_call_acc:
                    tool_call_acc[idx] = {
                        "id": tc.get("id", ""),
                        "name": "",
                        "args": "",
                    }
                fn = tc.get("function", {})
                if fn.get("name"):
                    tool_call_acc[idx]["name"] = fn["name"]
                if fn.get("arguments"):
                    tool_call_acc[idx]["args"] += fn["arguments"]

        # Emit assembled tool calls on finish
        if finish_reason and tool_call_acc:
            for _idx, acc in sorted(tool_call_acc.items()):
                try:
                    args = json.loads(acc["args"]) if acc["args"] else {}
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": acc["args"]}
                events.append(StreamEvent(
                    kind="tool_call",
                    tool_call=ToolCall(
                        id=acc["id"],
                        tool_name=acc["name"],
                        args=args,
                    ),
                ))
            tool_call_acc.clear()

        # Terminal usage from the chunk (OpenAI includes it when stream_options.include_usage=true)
        usage = chunk.get("usage")
        if usage and finish_reason:
            events.append(StreamEvent(
                kind="end",
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
            ))

    # If no usage event was emitted, emit an end with empty usage
    if not any(e.kind == "end" for e in events):
        events.append(StreamEvent(kind="end"))

    return events


# ---------------------------------------------------------------------------
# SSE streaming helpers (Anthropic)
# ---------------------------------------------------------------------------


def parse_anthropic_sse_line(line: str) -> tuple[str, dict[str, Any]] | None:
    """Parse an Anthropic SSE event line pair.

    Anthropic SSE has 'event: <type>' followed by 'data: <json>'.
    Returns (event_type, data_dict) or None.
    """
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    try:
        data = json.loads(payload)
        return data.get("type", ""), data
    except (json.JSONDecodeError, TypeError):
        return None


def iter_anthropic_stream_events(
    events_data: list[tuple[str, dict[str, Any]]],
) -> list[StreamEvent]:
    """Convert parsed Anthropic SSE events into StreamEvents."""
    stream_events: list[StreamEvent] = []
    usage: dict[str, int] = {}

    for event_type, data in events_data:
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    stream_events.append(StreamEvent(kind="text", text=text))

        elif event_type == "message_delta":
            msg_usage = data.get("usage", {})
            if msg_usage.get("output_tokens"):
                usage["output_tokens"] = msg_usage["output_tokens"]

        elif event_type == "message_start":
            msg = data.get("message", {})
            msg_usage = msg.get("usage", {})
            if msg_usage.get("input_tokens"):
                usage["input_tokens"] = msg_usage["input_tokens"]

        elif event_type == "message_stop":
            pass  # handled below

    # Emit terminal event
    stream_events.append(StreamEvent(kind="end", usage=usage))
    return stream_events
