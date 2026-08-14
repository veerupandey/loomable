"""Shared offline helpers for Loomable examples.

Keeps demos on the high-level Agent / Team / create_deep_agent API while
still runnable without an LLM key. Prefer a live provider in production::

    from loomable.providers.gemini import GeminiProvider
    Agent(model=GeminiProvider(), knowledge_base=...)
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from loomable.agent import ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall

Step = dict[str, Any] | str | Callable[[ModelRequest, int], ModelResponse]


def tool_names(request: ModelRequest) -> set[str]:
    """Names advertised to the model on this turn."""
    names: set[str] = set()
    for t in request.tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = (fn or {}).get("name") or t.get("name")
        if name:
            names.add(str(name))
    return names


def _last_user_text(request: ModelRequest) -> str:
    """Best-effort extract of the latest user turn (for offline echo scripts)."""
    for message in reversed(request.messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text") or ""))
                elif isinstance(part, str):
                    chunks.append(part)
            return "".join(chunks)
    return ""


def scripted_model(
    steps: Sequence[Step],
    *,
    provider: str = "scripted",
) -> ModelSpec:
    """Build a ``ModelSpec`` that walks a short script of tool calls / answers.

    Each step is either:
      - ``str`` — final text response
      - ``{"tool": name, "args": {...}}`` — one tool call
      - ``{"echo": "APPROVED: {input}"}`` — format prior agent/user text
        (``{input}`` is replaced with the latest user turn — what Workflow
        already passed from the previous Agent)
      - ``callable(request, n) -> ModelResponse`` — custom turn
    """

    class _Scripted:
        def __init__(self) -> None:
            self.n = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.n += 1
            idx = self.n - 1
            if idx >= len(steps):
                last = steps[-1] if steps else "done"
                if isinstance(last, str):
                    return ModelResponse(content=last)
                return ModelResponse(content="done")
            step = steps[idx]
            if callable(step) and not isinstance(step, type):
                return step(request, self.n)
            if isinstance(step, str):
                return ModelResponse(content=step)
            if isinstance(step, dict) and "echo" in step:
                template = str(step["echo"])
                prior = _last_user_text(request)
                return ModelResponse(content=template.replace("{input}", prior))
            if isinstance(step, dict) and "tool" in step:
                return ModelResponse(
                    content=str(step.get("content") or ""),
                    tool_calls=[
                        ToolCall(
                            id=str(step.get("id") or self.n),
                            tool_name=str(step["tool"]),
                            args=dict(step.get("args") or {}),
                        )
                    ],
                )
            raise TypeError(f"Unsupported scripted step: {step!r}")

    return ModelSpec(provider=provider, provider_impl=_Scripted())
