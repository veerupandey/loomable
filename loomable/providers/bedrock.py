"""loomable.providers.bedrock - Amazon Bedrock provider (Converse API).

:class:`BedrockProvider` targets the Amazon Bedrock **Converse API**, a single
unified surface that works across Bedrock model families (Anthropic Claude,
Amazon Nova, Meta Llama, Mistral, ...) and supports tool use, system prompts,
and multimodal image input. It implements the kernel ``ModelProvider`` protocol
(``async complete(request) -> ModelResponse``) plus best-effort streaming.

Authentication uses the standard AWS credential chain via ``boto3`` — environment
variables, shared config/credentials files, SSO profiles, or an assumed role.
Pass ``profile_name=`` / ``region_name=`` to select a profile/region, or hand in
a pre-built ``bedrock-runtime`` client via ``client=``.

``boto3`` is an optional dependency::

    pip install "loomable[bedrock]"

The synchronous boto3 calls are executed in a worker thread (``asyncio.to_thread``)
so the provider never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
from typing import Any

from loomable.kernel.models import ModelRequest, ModelResponse, StreamEvent, ToolCall

from .errors import PermanentProviderError, TransientProviderError

_DEFAULT_TIMEOUT = 60.0
#: Bedrock requires an explicit max_tokens for most models; used when unset.
_DEFAULT_MAX_TOKENS = 1024

#: Image media types Converse accepts, mapped to its short ``format`` token.
_IMAGE_FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}


def _parse_data_uri(url: str) -> tuple[str, bytes] | None:
    """Parse ``data:<media_type>;base64,<payload>`` into (media_type, bytes)."""
    if not url.startswith("data:"):
        return None
    try:
        header, encoded = url.split(",", 1)
        media_type = header[len("data:"):].split(";", 1)[0] or "application/octet-stream"
        return media_type, base64.b64decode(encoded)
    except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
        return None


class BedrockProvider:
    """A ``ModelProvider`` for Amazon Bedrock via the Converse API.

    Parameters
    ----------
    model:
        The Bedrock model id or inference-profile id (e.g.
        ``"anthropic.claude-3-haiku-20240307-v1:0"`` or a cross-region profile
        such as ``"us.anthropic.claude-sonnet-4-5-20250929-v1:0"``).
    region_name:
        AWS region. Defaults to ``AWS_REGION`` / ``AWS_DEFAULT_REGION``.
    profile_name:
        Shared-config/SSO profile name. Defaults to ``AWS_PROFILE``.
    aws_access_key_id / aws_secret_access_key / aws_session_token:
        Explicit static credentials (optional). When omitted, the standard boto3
        credential chain is used (env, profile, SSO, instance role, ...).
    client:
        A pre-built boto3 ``bedrock-runtime`` client. When provided, the
        region/profile/credential arguments are ignored.
    max_tokens:
        Default ``max_tokens`` used when a request does not specify one.
    timeout:
        Per-request read timeout in seconds.
    additional_model_request_fields:
        Optional dict forwarded as Converse ``additionalModelRequestFields``
        (e.g. Anthropic ``{"thinking": {...}}`` or Nova-specific knobs).
    """

    def __init__(
        self,
        model: str,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        client: Any | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: float = _DEFAULT_TIMEOUT,
        additional_model_request_fields: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._extra_fields = dict(additional_model_request_fields or {})
        self._region = region_name or os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION"
        )
        self._profile = profile_name or os.environ.get("AWS_PROFILE")
        self._static_creds = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
        }
        self._client = client
        self._client_lock = threading.Lock()

    @property
    def _provider_id(self) -> str:
        return f"bedrock:{self.model}"

    # ------------------------------------------------------------------ client
    def _get_client(self) -> Any:
        """Lazily build (and cache) a thread-safe bedrock-runtime client."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import boto3  # type: ignore
                from botocore.config import Config  # type: ignore
            except ImportError as exc:  # pragma: no cover - env dependent
                raise PermanentProviderError(self._provider_id) from exc

            session_kwargs: dict[str, Any] = {}
            if self._profile:
                session_kwargs["profile_name"] = self._profile
            if self._region:
                session_kwargs["region_name"] = self._region
            creds = {k: v for k, v in self._static_creds.items() if v}
            session = boto3.Session(**session_kwargs, **creds)
            cfg = Config(
                read_timeout=self._timeout,
                connect_timeout=self._timeout,
                retries={"max_attempts": 2, "mode": "standard"},
            )
            self._client = session.client("bedrock-runtime", config=cfg)
            return self._client

    # ------------------------------------------------------------- translation
    def _build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        system, messages = self._to_converse_messages(request.messages)
        inference: dict[str, Any] = {
            "maxTokens": int(request.max_tokens or self._max_tokens),
        }
        if request.temperature is not None:
            inference["temperature"] = float(request.temperature)

        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": inference,
        }
        if system:
            kwargs["system"] = system
        tool_config = self._to_tool_config(request.tools)
        if tool_config:
            kwargs["toolConfig"] = tool_config
        if self._extra_fields:
            kwargs["additionalModelRequestFields"] = self._extra_fields
        return kwargs

    def _to_tool_config(self, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Translate OpenAI-style function tools into a Converse ``toolConfig``."""
        specs: list[dict[str, Any]] = []
        for tool in tools or []:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            name = fn.get("name")
            if not name:
                continue
            schema = fn.get("parameters") or {"type": "object", "properties": {}}
            spec: dict[str, Any] = {
                "name": name,
                "inputSchema": {"json": schema},
            }
            desc = fn.get("description")
            if desc:
                spec["description"] = desc
            specs.append({"toolSpec": spec})
        if not specs:
            return None
        return {"tools": specs}

    def _to_converse_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split agnostic messages into Converse ``(system, messages)``.

        - system messages are collected into the top-level ``system`` list.
        - user/assistant/tool messages become Converse content blocks.
        - ``tool`` role results are emitted as ``toolResult`` blocks inside a
          ``user`` message (Converse requirement).
        - Consecutive messages that map to the same Converse role are merged so
          the transcript strictly alternates user/assistant.
        """
        system: list[dict[str, Any]] = []
        out: list[dict[str, Any]] = []

        def _append(role: str, blocks: list[dict[str, Any]]) -> None:
            if not blocks:
                return
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                text = content if isinstance(content, str) else _text_of(content)
                if text:
                    system.append({"text": text})
                continue

            if role == "tool":
                block = self._tool_result_block(
                    msg.get("tool_call_id", ""), content
                )
                _append("user", [block])
                continue

            if role == "assistant":
                blocks = self._content_blocks(content)
                for call in msg.get("tool_calls", []) or []:
                    blocks.append(self._tool_use_block(call))
                # Converse forbids empty assistant content.
                if not blocks:
                    blocks = [{"text": " "}]
                _append("assistant", blocks)
                continue

            # user (and any other) role
            blocks = self._content_blocks(content) or [{"text": " "}]
            _append("user", blocks)

        # Converse requires the transcript to begin with a user turn.
        if out and out[0]["role"] != "user":
            out.insert(0, {"role": "user", "content": [{"text": " "}]})
        return system, out

    def _content_blocks(self, content: Any) -> list[dict[str, Any]]:
        """Convert message content (str or part-list) into Converse blocks."""
        if isinstance(content, str):
            return [{"text": content}] if content else []
        blocks: list[dict[str, Any]] = []
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text", "")
                if text:
                    blocks.append({"text": text})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                parsed = _parse_data_uri(url)
                if parsed is not None:
                    media_type, raw = parsed
                    fmt = _IMAGE_FORMATS.get(media_type.lower())
                    if fmt:
                        blocks.append(
                            {"image": {"format": fmt, "source": {"bytes": raw}}}
                        )
                # Converse image blocks require inline bytes; remote http URLs
                # are not directly supported and are skipped.
        return blocks

    @staticmethod
    def _tool_use_block(call: dict[str, Any]) -> dict[str, Any]:
        """Build a Converse ``toolUse`` block from an agnostic tool call."""
        # Support both internal ({id, tool_name, args}) and OpenAI
        # ({id, function:{name, arguments}}) shapes.
        if "function" in call:
            fn = call.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", {})
        else:
            name = call.get("tool_name", "")
            raw_args = call.get("args", {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                raw_args = {"_raw": raw_args}
        return {
            "toolUse": {
                "toolUseId": call.get("id") or name,
                "name": name,
                "input": raw_args if isinstance(raw_args, dict) else {"value": raw_args},
            }
        }

    @staticmethod
    def _tool_result_block(tool_use_id: str, content: Any) -> dict[str, Any]:
        """Build a Converse ``toolResult`` block from a tool-role message."""
        text = _text_of(content)
        return {
            "toolResult": {
                "toolUseId": tool_use_id,
                "content": [{"text": text if text else " "}],
            }
        }

    # -------------------------------------------------------------- invocation
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send a Converse request and return a ``ModelResponse``.

        Raises a classified :class:`TransientProviderError` /
        :class:`PermanentProviderError` naming the provider on failure.
        """
        kwargs = self._build_kwargs(request)
        try:
            client = await asyncio.to_thread(self._get_client)
            data = await asyncio.to_thread(lambda: client.converse(**kwargs))
        except Exception as exc:  # noqa: BLE001 - classified below
            raise self._classify(exc) from exc
        return _parse_converse_response(data)

    async def stream(self, request: ModelRequest):
        """Stream a Converse completion, yielding ``StreamEvent`` objects.

        Bridges boto3's synchronous ``converse_stream`` event iterator onto the
        event loop via a background thread and an ``asyncio.Queue``.
        """
        kwargs = self._build_kwargs(request)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        _SENTINEL = object()

        def _worker() -> None:
            try:
                client = self._get_client()
                resp = client.converse_stream(**kwargs)
                for event in _iter_converse_stream(resp.get("stream", [])):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:  # noqa: BLE001 - surfaced to consumer
                loop.call_soon_threadsafe(queue.put_nowait, self._classify(exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    # ------------------------------------------------------------ error mapping
    def _classify(self, exc: Exception) -> Exception:
        """Map botocore/boto3 errors to Transient/Permanent provider errors."""
        # Import lazily; botocore is present whenever boto3 is.
        try:
            from botocore.exceptions import (  # type: ignore
                ClientError,
                ConnectionError as BotoConnectionError,
                ConnectTimeoutError,
                EndpointConnectionError,
                ReadTimeoutError,
            )
        except ImportError:
            return TransientProviderError(self._provider_id, status_code=None)

        if isinstance(exc, (PermanentProviderError, TransientProviderError)):
            return exc
        if isinstance(
            exc,
            (
                EndpointConnectionError,
                BotoConnectionError,
                ConnectTimeoutError,
                ReadTimeoutError,
            ),
        ):
            return TransientProviderError(self._provider_id, status_code=None)
        if isinstance(exc, ClientError):
            meta = exc.response.get("ResponseMetadata", {}) if hasattr(exc, "response") else {}
            code = meta.get("HTTPStatusCode")
            err_code = exc.response.get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
            transient_codes = {
                "ThrottlingException",
                "TooManyRequestsException",
                "ModelTimeoutException",
                "ServiceUnavailableException",
                "InternalServerException",
                "ModelNotReadyException",
            }
            if err_code in transient_codes or (isinstance(code, int) and (code == 429 or 500 <= code < 600)):
                return TransientProviderError(self._provider_id, status_code=code)
            return PermanentProviderError(self._provider_id, status_code=code)
        # Unknown → treat as transient network-ish error.
        return TransientProviderError(self._provider_id, status_code=None)


# ---------------------------------------------------------------------------
# Response / stream parsing helpers (module-level for testability)
# ---------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    """Best-effort text extraction from a string or content-part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, dict) and "text" in p and "type" not in p:
                parts.append(p.get("text", ""))
        return "".join(parts)
    return ""


def _parse_converse_response(data: dict[str, Any]) -> ModelResponse:
    """Parse a Bedrock Converse response into a ``ModelResponse``."""
    message = (data.get("output") or {}).get("message") or {}
    blocks = message.get("content") or []

    text_parts: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                ToolCall(
                    id=tu.get("toolUseId", ""),
                    tool_name=tu.get("name", ""),
                    args=tu.get("input", {}) or {},
                )
            )
        elif "reasoningContent" in block:
            rc = block["reasoningContent"] or {}
            rtext = (rc.get("reasoningText") or {}).get("text")
            if isinstance(rtext, str) and rtext.strip():
                reasoning.append(rtext.strip())

    usage_raw = data.get("usage", {}) or {}
    return ModelResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        usage={
            "input_tokens": usage_raw.get("inputTokens", 0),
            "output_tokens": usage_raw.get("outputTokens", 0),
        },
        metadata={"stop_reason": data.get("stopReason", "")},
        reasoning=reasoning,
    )


def _iter_converse_stream(stream: Any) -> list[StreamEvent]:
    """Convert a Converse stream event iterator into ``StreamEvent`` objects.

    Handled events: ``contentBlockStart`` (toolUse begin), ``contentBlockDelta``
    (text or tool input JSON deltas), ``contentBlockStop`` (emit assembled tool
    call), and ``metadata`` (terminal usage).
    """
    events: list[StreamEvent] = []
    # Accumulate tool-use fragments keyed by content block index.
    tool_acc: dict[int, dict[str, str]] = {}
    usage: dict[str, int] = {}

    for event in stream:
        if not isinstance(event, dict):
            continue

        if "contentBlockStart" in event:
            start = event["contentBlockStart"]
            idx = start.get("contentBlockIndex", 0)
            tu = (start.get("start") or {}).get("toolUse") or {}
            if tu:
                tool_acc[idx] = {
                    "id": tu.get("toolUseId", ""),
                    "name": tu.get("name", ""),
                    "args": "",
                }

        elif "contentBlockDelta" in event:
            delta_wrap = event["contentBlockDelta"]
            idx = delta_wrap.get("contentBlockIndex", 0)
            delta = delta_wrap.get("delta", {}) or {}
            if "text" in delta and delta["text"]:
                events.append(StreamEvent(kind="text", text=delta["text"]))
            elif "toolUse" in delta:
                frag = (delta["toolUse"] or {}).get("input", "")
                if idx in tool_acc:
                    tool_acc[idx]["args"] += frag or ""

        elif "contentBlockStop" in event:
            idx = event["contentBlockStop"].get("contentBlockIndex", 0)
            acc = tool_acc.pop(idx, None)
            if acc is not None:
                try:
                    args = json.loads(acc["args"]) if acc["args"] else {}
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": acc["args"]}
                events.append(
                    StreamEvent(
                        kind="tool_call",
                        tool_call=ToolCall(
                            id=acc["id"], tool_name=acc["name"], args=args
                        ),
                    )
                )

        elif "metadata" in event:
            u = event["metadata"].get("usage", {}) or {}
            if u:
                usage = {
                    "input_tokens": u.get("inputTokens", 0),
                    "output_tokens": u.get("outputTokens", 0),
                }

    events.append(StreamEvent(kind="end", usage=usage))
    return events
