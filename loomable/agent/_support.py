"""Shared configuration, input coercion, and validation helpers."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loomable.content import (
    AgentInput,
    MediaPart,
    Modality,
    ModelCapabilities,
)
from loomable.kernel.contracts import ModelProvider
from loomable.kernel.errors import GuardrailViolation
from loomable.kernel.models import (
    ToolCall,
    ToolOutcome,
)

from .errors import (
    StructuredOutputError,
)
from .run import RunResult, extract_plan_steps, extract_thoughts

if TYPE_CHECKING:  # pragma: no cover - typing only

    pass


#: Default provider id used when a bare ``ModelProvider`` is supplied as the model.
_DEFAULT_PROVIDER_ID = "default"


# ---------------------------------------------------------------------------
# Tool hooks / Human-in-the-loop types (Req 14)
# ---------------------------------------------------------------------------

#: A pre-hook: ``(tool_name, call, args) -> decision``. Returning ``False`` (or raising
#: :class:`~loomable.agent.errors.ToolHookRejection`) rejects the call before it runs
#: (Req 14.1/14.3). Any other return value allows the call.
ToolHook = Callable[[str, "ToolCall", dict], object]

#: A post-hook: ``(tool_name, call, outcome) -> outcome | None``. Runs after execution
#: to observe or transform the result; returning a :class:`ToolOutcome` replaces the
#: original outcome, any other value leaves it unchanged (Req 14.1/14.2).
PostToolHook = Callable[[str, "ToolCall", "ToolOutcome"], object]

#: An approver callback for tools that require confirmation: ``(call) -> bool``. Returns
#: ``True`` to approve execution, ``False`` to deny (Req 14.4).
Approver = Callable[["ToolCall"], bool]

#: Rule id recorded when a pre-hook rejects a tool call.
_HOOK_REJECTION_RULE_ID = "tool-hook-rejection"

#: Rule id recorded when a confirmation-required tool is not approved.
_CONFIRMATION_RULE_ID = "require-confirmation"


def _deny_all_approver(call: "ToolCall") -> bool:
    """Default approver: deny every confirmation-required tool (headless-safe, Req 14.4).

    In a headless run there is no human to approve, so the safe default is to deny.
    Callers inject their own approver (e.g. a prompt or an auto-approve callback) to
    grant execution.
    """
    return False


@dataclass
class GatedDispatchResult:
    """The result of a gated tool-dispatch step (Req 14).

    Carries the outcomes of the tool calls that actually executed alongside the
    guardrail violations recorded for calls that were blocked — whether by a pre-hook
    rejection (Req 14.3) or a denied confirmation (Req 14.4). Blocked calls never
    reach the tool runtime.
    """

    outcomes: list[ToolOutcome]
    blocked: list[GuardrailViolation]


def _input_text(agent_input: AgentInput) -> str:
    """Concatenate the decoded UTF-8 text of every TEXT part across all messages.

    :class:`~loomable.content.AgentInput` has no ``text()`` helper of its own, so this
    mirrors :meth:`AgentOutput.text` for the input side when recording a user turn for
    session persistence (Req 15.2). Non-text parts contribute nothing.
    """
    pieces: list[str] = []
    for message in agent_input.messages:
        for part in message.parts:
            if part.modality is Modality.TEXT and part.data is not None:
                pieces.append(part.data.decode("utf-8"))
    return "".join(pieces)


def _is_run_context(value: Any) -> bool:
    """True when ``value`` is a Flow/Workflow :class:`~loomable.agent.context.RunContext`."""
    if value is None or isinstance(value, dict):
        return False
    # Duck-type to avoid import cycles with flow engines.
    return hasattr(value, "shared_state") and hasattr(value, "memory")


# ---------------------------------------------------------------------------
# Media coercion dispatch (Req 3.1–3.5, 4.1–4.2)
# ---------------------------------------------------------------------------

#: Mapping from target modality name to the high-level Media_Class constructor.
_MODALITY_CLASS_MAP: dict[str, type] = {}  # populated lazily to avoid circular imports

#: Default MIME types for each modality when no format/extension can be inferred.
_MODALITY_FALLBACK_MIME: dict[str, str] = {
    "image": "image/png",
    "video": "video/mp4",
    "audio": "audio/wav",
}


def _get_modality_class(target_modality: str):
    """Return the Media_Class for the given modality string, importing lazily."""
    if not _MODALITY_CLASS_MAP:
        from loomable.media import Audio as _Audio, Image as _Image, Video as _Video

        _MODALITY_CLASS_MAP["image"] = _Image
        _MODALITY_CLASS_MAP["video"] = _Video
        _MODALITY_CLASS_MAP["audio"] = _Audio
    return _MODALITY_CLASS_MAP[target_modality]


def _coerce_media_item(item: "Any", target_modality: str) -> "MediaPart":
    """Dispatch a single media input item to a :class:`MediaPart`.

    Handles the following input types:

    - ``_MediaBase`` instance → call ``.to_media_part()``
    - ``MediaPart`` → pass-through (already the target type)
    - ``str`` starting with ``http://`` or ``https://`` → construct Media_Class(url=item)
    - ``str`` (other) → construct Media_Class(filepath=item)
    - ``bytes`` → construct Media_Class(content=item)

    Args:
        item: The input media item to coerce.
        target_modality: One of ``"image"``, ``"video"``, ``"audio"`` — determines
            which high-level Media_Class to use for string/bytes inputs.

    Returns:
        A ``MediaPart`` ready for inclusion in an ``AgentInput`` message.
    """
    from loomable.media.types import _MediaBase

    # Already a MediaPart: pass through unchanged.
    if isinstance(item, MediaPart):
        return item

    # A high-level Media_Class instance (Image, Audio, Video, File): convert.
    if isinstance(item, _MediaBase):
        return item.to_media_part()

    # Determine the appropriate Media_Class for string/bytes construction.
    media_cls = _get_modality_class(target_modality)
    fallback_mime = _MODALITY_FALLBACK_MIME.get(target_modality)

    if isinstance(item, str):
        if item.startswith("http://") or item.startswith("https://"):
            return media_cls(url=item).to_media_part()
        else:
            return media_cls(filepath=item).to_media_part()

    if isinstance(item, bytes):
        # For raw bytes with no extension hint, supply a fallback mime_type
        # consistent with the target modality to satisfy MediaPart validation.
        return media_cls(content=item, mime_type=fallback_mime).to_media_part()

    # Fallback for Path objects.
    if isinstance(item, Path):
        return media_cls(filepath=item).to_media_part()

    raise TypeError(
        f"Cannot coerce {type(item).__name__!r} to a MediaPart for modality "
        f"'{target_modality}'. Expected str, bytes, Path, MediaPart, or a "
        f"Media class instance."
    )


def _synthesize_tool_activity_summary(tool_activity: list[Any]) -> str:
    """Build a short user-facing summary when the model ends with empty text."""
    lines: list[str] = ["Completed tool actions:"]
    for outcome in tool_activity[-12:]:
        name = ""
        detail = ""
        if getattr(outcome, "result", None) is not None:
            name = str((outcome.result.metadata or {}).get("tool_name") or "")
            content = outcome.result.content
            if isinstance(content, str) and content.strip():
                detail = content.strip().replace("\n", " ")
                if len(detail) > 160:
                    detail = detail[:157] + "..."
        if not name:
            name = "tool"
        if detail:
            lines.append(f"- {name}: {detail}")
        else:
            lines.append(f"- {name}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _schema_instruction(output_schema: type) -> str:
    """Build a lightweight, provider-agnostic instruction asking for JSON output.

    Names the target schema when it can be identified (Req 13.1). This is only a hint
    to the model; validation is enforced separately after the response arrives.
    """
    name = getattr(output_schema, "__name__", "the requested schema")
    details = ""
    try:
        import pydantic

        if isinstance(output_schema, type) and issubclass(output_schema, pydantic.BaseModel):
            schema = output_schema.model_json_schema()
            props = list((schema.get("properties") or {}).keys())
            if props:
                details = f" Use exactly these keys: {', '.join(props)}."
            details += f" JSON Schema: {json.dumps(schema)}"
    except Exception:
        pass
    return (
        "Respond ONLY with a single JSON object that matches "
        f"{name}.{details} Do not include any prose, markdown, or code fences."
    )


def _strip_json_fences(text: str) -> str:
    """Remove common markdown code fences around JSON payloads."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _path_constraint_met(path_arg: str, required: str) -> bool:
    """Return True when ``path_arg`` satisfies a require_tools path constraint.

    Matches:
    - exact path
    - path ending with ``/<required>``
    - directory / prefix constraints when *required* ends with ``/``
      (``reports/`` matches ``reports/brief.md``)
    - bare directory name without slash (``reports`` matches ``reports/x.md``)

    Substring matches (e.g. ``myoutput/brief.md`` for ``output/brief.md``)
    do **not** count.
    """
    from pathlib import Path

    a = Path(str(path_arg).replace("\\", "/")).as_posix().lstrip("./")
    r = str(required).replace("\\", "/").lstrip("./")
    if not r:
        return True
    # Directory prefix: "reports/" → reports/brief.md
    if r.endswith("/"):
        prefix = r.rstrip("/")
        return a == prefix or a.startswith(prefix + "/")
    r_norm = Path(r).as_posix().lstrip("./")
    if a == r_norm:
        return True
    if a.endswith("/" + r_norm):
        return True
    # Bare directory name: "reports" → reports/x.md
    if "/" not in r_norm and a.startswith(r_norm + "/"):
        return True
    return False


def _parse_require_tool_specs(specs: list[str]) -> list[tuple[str, str | None]]:
    """Parse ``require_tools`` entries into ``(name, path_constraint|None)``.

    Supports plain names (``\"write_file\"``) and path constraints
    (``\"write_file:output/brief.md\"``) so scribes cannot "satisfy" the
    contract by writing to an arbitrary path. A trailing ``*`` matches a tool
    name prefix.
    """
    parsed: list[tuple[str, str | None]] = []
    for raw in specs:
        spec = (raw or "").strip()
        if not spec:
            continue
        if ":" in spec:
            name, _, path = spec.partition(":")
            name, path = name.strip(), path.strip()
            if name:
                parsed.append((name, path or None))
            continue
        parsed.append((spec, None))
    return parsed


def _missing_require_tool_specs(
    specs: list[tuple[str, str | None]],
    satisfied: set[tuple[str, str | None]],
) -> list[str]:
    """Return human-readable missing require_tools specs."""
    missing: list[str] = []
    for name, path in specs:
        if (name, path) in satisfied:
            continue
        missing.append(f"{name}:{path}" if path else name)
    return missing


def _format_require_tools_nudge(missing: list[str]) -> str:
    parts: list[str] = []
    for item in missing:
        if ":" in item:
            name, _, path = item.partition(":")
            parts.append(
                f"call {name} with path matching {path!r} (exact or ending with /{path})"
            )
        else:
            parts.append(f"call a tool whose name starts with {item[:-1]}" if item.endswith("*") else f"call {item}")
    return (
        "You finished without satisfying required tools: "
        + "; ".join(parts)
        + ". Do those tool calls now (exact required paths), then give your final answer."
    )


# Todo / private-think tools that do not advance the user-facing deliverable.
# After require_tools are satisfied, repeated rounds of only these tools are
# treated as "done — emit final answer" instead of burning max_tool_iterations.
_BOOKKEEPING_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "write_todos",
        "read_todos",
        "update_todo",
        "think",
        "search_skills",
        "load_skill",
        "search_tools",
        "search_mcp",
        "search_namespaces",
        "activate_tool",
        "activate_mcp_server",
        "refresh_capabilities",
        "list_skill_resources",
        "read_skill_resource",
    }
)

_DELIVERABLE_COMPLETE_NUDGE = (
    "Required deliverables are already complete (required tools succeeded). "
    "Stop calling tools now — do not keep updating todos. "
    "Provide your final answer summarizing what you delivered and where."
)


def _is_bookkeeping_only(tool_names: set[str]) -> bool:
    return bool(tool_names) and tool_names.issubset(_BOOKKEEPING_TOOL_NAMES)


def _extract_json_object(text: str) -> str:
    """Return the first JSON object substring, or the original text."""
    cleaned = _strip_json_fences(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _validate_structured(text: str, output_schema: type) -> object:
    """Parse ``text`` as JSON and validate/coerce it into ``output_schema``.

    Handles ``pydantic`` models, dataclasses, and generic callables (Req 13.2). Any
    parse or validation failure is surfaced as :class:`StructuredOutputError` naming
    the failure (Req 13.3). ``pydantic`` is imported lazily so the module still
    imports when only dataclasses are used.
    """
    text = _extract_json_object(text)
    # pydantic BaseModel: prefer its own JSON validation (handles parse + validate).
    try:  # lazy/defensive import — pydantic may be absent for dataclass-only users.
        import pydantic

        if isinstance(output_schema, type) and issubclass(output_schema, pydantic.BaseModel):
            try:
                return output_schema.model_validate_json(text)
            except pydantic.ValidationError as exc:
                raise StructuredOutputError(
                    f"response does not match schema '{output_schema.__name__}': {exc}"
                ) from exc
            except ValueError as exc:
                raise StructuredOutputError(
                    f"response is not valid JSON for '{output_schema.__name__}': {exc}"
                ) from exc
    except ImportError:
        pass

    # Everything else parses JSON first, then coerces into the schema.
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise StructuredOutputError(f"response is not valid JSON: {exc}") from exc

    # Dataclass: construct from a mapping of fields.
    if dataclasses.is_dataclass(output_schema) and isinstance(output_schema, type):
        if not isinstance(data, dict):
            raise StructuredOutputError(
                f"expected a JSON object for dataclass "
                f"'{output_schema.__name__}', got {type(data).__name__}"
            )
        try:
            return output_schema(**data)
        except TypeError as exc:
            raise StructuredOutputError(
                f"response does not match dataclass '{output_schema.__name__}': {exc}"
            ) from exc

    # Generic callable schema: pass the parsed value through the constructor.
    if callable(output_schema):
        try:
            if isinstance(data, dict):
                return output_schema(**data)
            return output_schema(data)
        except (TypeError, ValueError) as exc:
            name = getattr(output_schema, "__name__", "schema")
            raise StructuredOutputError(
                f"response could not be coerced into '{name}': {exc}"
            ) from exc

    # Not callable: store the parsed value directly.
    return data


def _finalize_run_result(
    result: RunResult,
    *,
    provider_reasoning: list[str] | None = None,
) -> RunResult:
    """Attach thoughts / plan / reasoning derived from the run."""
    thoughts = extract_thoughts(result.tool_activity)
    plan = extract_plan_steps(result.tool_activity)
    reasoning = list(provider_reasoning or [])
    # Fall back to think-tool contents only when no native reasoning was exposed.
    if not reasoning and thoughts:
        reasoning = list(thoughts)
        result.metadata.setdefault("reasoning_source", "think_tool")
    elif reasoning:
        result.metadata.setdefault("reasoning_source", "provider")
    result.thoughts = thoughts
    result.plan = plan
    result.reasoning = reasoning
    return result





@dataclass
class ModelSpec:
    """Declarative model configuration for the builder.

    Attributes
    ----------
    provider:
        The provider identifier this model is registered under.
    provider_impl:
        The concrete :class:`ModelProvider` implementation. May be ``None`` when a
        provider is registered elsewhere; a runnable agent requires an impl.
    capabilities:
        The declared input/output modalities. Defaults to text+image+video input,
        text output (multimodal-by-default).
    """

    provider: str
    provider_impl: ModelProvider | None = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
