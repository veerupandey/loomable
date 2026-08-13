"""loomable.agent.builder - High-level, agno-style Agent builder.

This module implements the ergonomic high-level entry point (:class:`Agent`) that
assembles a runnable :class:`BuiltAgent` from a compact configuration, composing the
existing kernel primitives without modifying ``loomable.kernel`` (Req 1.7, 2.4).

``Agent(model=...)`` is enough to produce a runnable agent: :meth:`Agent.build`
constructs default implementations for every kernel subsystem that was not supplied
(Req 1.2) and uses any pre-constructed primitive that *was* supplied (Req 2.2/2.3).
Missing or invalid required fields raise :class:`AgentConfigError` naming the field
before any run is attempted (Req 1.6).

Multi-agent orchestration is now handled via ``loomable.flow.Flow``. The agent
retains only single-agent auto-escalation: single-shot → tool-loop → self-plan
(via the Flow engine) (Req 14.4, 17.3).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loomable.content import (
    AgentInput,
    AgentOutput,
    MediaPart,
    Message,
    Modality,
    ModelCapabilities,
    from_model_response,
    to_agent_input,
    to_model_request,
)
from loomable.kernel.context import ContextManager
from loomable.kernel.contracts import ModelProvider, Retriever, Tool
from loomable.kernel.errors import GuardrailViolation, MCPConnectionError, SkillLoadError
from loomable.kernel.guardrails import GuardrailHarness
from loomable.kernel.long_term import LongTermStore
from loomable.kernel.mcp_client import MCPClient, MCPSession
from loomable.kernel.memory import MemoryManager
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.model_router import ModelRouter, TierSubstitution
from loomable.kernel.models import (
    AgentConfig,
    ContextItem,
    Session,
    TierPolicy,
    ToolCall,
    ToolError,
    ToolOutcome,
    Turn,
)
from loomable.kernel.planner import Planner
from loomable.kernel.retrievers import RetrieverTool
from loomable.kernel.skills import SkillLoader
from loomable.kernel.stores import SessionStore
from loomable.kernel.summarizer import Summarizer
from loomable.kernel.tool_runtime import ToolRuntime

from .errors import (
    AgentConfigError,
    InputValidationError,
    StructuredOutputError,
    ToolHookRejection,
    UnsupportedModalityError,
)
from .context import RunContext, StopReason, _signature as _ctx_signature
from .events import AgentEvents, Event, NoOpEvents
from .run import RunChunk, RunResult, extract_plan_steps, extract_thoughts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator

    from loomable.kernel.agent_loop import AgentLoop
    from loomable.providers.resilient import RetryPolicy
    from loomable.toolkits._base import Toolkit
    from .notes import NoteStore
    from .routing import ComplexityRouter


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


def _parse_require_tool_specs(specs: list[str]) -> list[tuple[str, str | None]]:
    """Parse ``require_tools`` entries into ``(name, path_substr|None)``.

    Supports plain names (``\"write_file\"``) and path constraints
    (``\"write_file:output/brief.md\"``) so scribes cannot "satisfy" the
    contract by writing to an arbitrary path.
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
                f"call {name} with path containing {path!r}"
            )
        else:
            parts.append(f"call {item}")
    return (
        "You finished without satisfying required tools: "
        + "; ".join(parts)
        + ". Do those tool calls now (exact required paths), then give your final answer."
    )


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


@dataclass
class BuiltAgent:
    """A runnable agent produced by :meth:`Agent.build`.

    Exposes read access to the composed kernel subsystems (Req 2.1) and runs the
    agent with modality capability gating (Req 4/5/6).
    """

    model_interface: ModelInterface
    memory: MemoryManager
    tool_runtime: ToolRuntime
    session: Session
    capabilities: ModelCapabilities

    loop: AgentLoop | None = None
    """The kernel AgentLoop.

    ``None`` for the high-level harness path (the production path).  The kernel
    ``AgentLoop`` remains available for autonomous/batch use but is not implicitly
    constructed by the high-level Builder.
    """
    # When True, arun persists session state after each run via session_store (Req 15.2).
    # Enabled by build() only when the developer supplied an explicit session_id, so
    # runs with an auto-generated session id do not incur persistence.
    persist_session: bool = False
    # Retained for later tasks (run flow, hooks/HITL, sessions):
    instructions: str | None = None
    harness: GuardrailHarness | None = None
    planner: Planner | None = None
    session_store: SessionStore | None = None

    # Conversational memory (Req 15): when the agent has a session, recent turns are
    # injected into each request so it remembers across calls. ``memory_window`` caps
    # how many recent turns are replayed (0 = all). Set ``use_memory=False`` to disable.
    use_memory: bool = True
    memory_window: int = 8
    # Kernel Summarizer instance used for automatic memory compaction (Req 6.1–6.5).
    summarizer: Summarizer | None = None
    # Optional Pydantic/dataclass schema used to validate dict/model inputs before a
    # run (agno-style structured input). ``None`` disables input validation.
    input_schema: type | None = None
    # Tool hooks / human-in-the-loop (task 9.2, Req 14):
    #   - tool_hooks: pre-hooks (a hook with ``phase == "post"`` is treated as a
    #     post-hook); post_tool_hooks: additional post-hooks (injectable after build).
    #   - require_confirmation: tool names that need approval before executing.
    #   - approver: injectable approval callback (default denies — headless-safe).
    tool_hooks: list[ToolHook] = field(default_factory=list)
    post_tool_hooks: list[PostToolHook] = field(default_factory=list)
    require_confirmation: list[str] = field(default_factory=list)
    approver: Approver = _deny_all_approver

    # Maximum iterations for the tool-use loop (Req 3.2).
    max_tool_iterations: int = 12

    # When True, re-prompt once if the final model text is empty after tools.
    require_final_text: bool = True

    # Tool names that MUST be called at least once before the run finishes.
    # Missing tools trigger re-prompts with tools still enabled.
    # Entries may be plain names ("write_file") or path constraints
    # ("write_file:output/brief.md") matched against the tool's path argument.
    require_tools: list[str] = field(default_factory=list)

    # Per-tool timeout and concurrency cap for gated dispatch (Req 2.1–2.4).
    # When set, each tool call in a batch is bounded by asyncio.wait_for and
    # parallelism is limited by an asyncio.Semaphore.
    tool_timeout: float | None = None
    tool_concurrency: int | None = None

    # Compaction threshold: when len(session.l1) exceeds this value, the oldest
    # turns beyond the retained window are summarized into session.l2 and dropped
    # from session.l1 (Req 6.1–6.5). Default is 2× memory_window.
    compaction_threshold: int = 16

    # Unified auto-compaction / spill policy (enterprise long-run safety).
    context_policy: Any | None = None

    # Tiered model routing (Req 7): when configured, model calls are routed through
    # this kernel ModelRouter instead of the model_interface directly.
    router: ModelRouter | None = None

    # Knowledge / RAG (Req 8.2–8.5): when knowledge docs are attached, the
    # LongTermStore holds their indexed embeddings and the embedder produces query
    # vectors at run time for recall.
    long_term: LongTermStore | None = None
    embedder: Any = None  # Embedder protocol instance (or None)
    # Number of top-k results to recall from the LongTermStore per run.
    knowledge_top_k: int = 3

    # --- Harness features (Req 4.5, 10–12) ---

    # Structured observability event emitter (Req 11.1–11.6). Defaults to NoOpEvents
    # so existing callers incur no recording overhead.
    events: AgentEvents = field(default_factory=NoOpEvents)

    # Opt-in complexity router (Req 10.2/10.3): when set, arun consults it before
    # mode selection (SINGLE→_run_single, TOOL_LOOP→_run_tool_loop, PLAN→_run_plan).
    complexity_router: "ComplexityRouter | None" = None

    # Durable note store for the memory tool (Req 7 notes).
    note_store: "NoteStore | None" = None

    # Loop-repeat threshold for no-progress detection (Req 3.1/3.2).
    loop_repeat_threshold: int = 3

    # --- Output verification (Req 4.2–4.4) ---
    # When a Verifier is configured, it is evaluated against the final output.
    # The VerdictResult is recorded on RunResult.verification.
    # When retry_on_failure is True and verification fails, the agent re-runs with
    # the failure detail appended to context, up to max_verify_retries times.
    verifier: Any = None  # Verifier | Callable | None
    retry_on_failure: bool = False
    max_verify_retries: int = 1

    # Transport resilience config (stored for reference; wrapping happens at build time).
    resilience: "RetryPolicy | None" = None

    # Token budget for context bounding (Req 13.1–13.4): when set, _bound_messages
    # applies evict-then-admit against this budget before each model call.
    _token_budget: int | None = None

    # Skill load errors: isolated failures from skill loading (Req 4.3). Each
    # entry identifies a Skill that failed to load while others succeeded.
    skill_errors: list["SkillLoadError"] = field(default_factory=list)

    # MCP connection errors: isolated failures from MCP server connections (Req 5.3).
    # Each entry identifies a server that failed to connect while others succeeded.
    mcp_errors: list["MCPConnectionError"] = field(default_factory=list)

    # Pinned facts (Req 6.1–6.4): steps whose turns are never eligible for compaction
    # and are always replayed in the memory prefix regardless of the rolling window.
    pinned_steps: set[int] = field(default_factory=set)

    # Multimodal feedback (Req 7.5): when True, tool-generated media is injected
    # into the conversation so the model can reason about it in subsequent turns.
    _feedback_media: bool = True

    @property
    def _model_id(self) -> str:
        """The model identifier used in capability-error messages."""
        return self.model_interface.default_provider

    def pin_fact(self, text: str) -> None:
        """Append a pinned assistant turn and record its step as never compactable (Req 6.1).

        The turn is appended to ``session.l1`` with the current step, that step is
        added to ``pinned_steps``, and ``session.step`` is advanced. Pinned turns are
        excluded from compaction overflow and always replayed in ``_memory_prefix``.
        """
        self.session.l1.append(
            Turn(
                role="assistant",
                content=text,
                tokens=0,
                step=self.session.step,
            )
        )
        self.pinned_steps.add(self.session.step)
        self.session.step += 1

    async def arun(
        self,
        input: "AgentInput | str",  # noqa: A002
        *,
        images: "list[str | Path | MediaPart] | None" = None,
        videos: "list[str | Path | MediaPart] | None" = None,
        audio: "list[str | Path | MediaPart] | None" = None,
        output_schema: type | None = None,
        context: "RunContext | None" = None,
    ) -> RunResult:
        """Run the agent once and return a :class:`RunResult`.

        The flow (Req 1.4, 4.1–4.4, 5.1/5.4, 6.3/6.4, 14.4, 17.3):

        1. Wrap a bare-string ``input`` via :meth:`AgentInput.from_text`.
        2. Validate every input modality is declared in ``capabilities.input``
           *before* touching the provider; raise :class:`UnsupportedModalityError`
           naming the modality and model otherwise (no provider call).
        3. Route by complexity: single-shot, tool-loop, or self-plan (via Flow).
        4. When the complexity router selects PLAN, the agent builds and runs a
           plan→map→synthesize Flow using the flow engine.
        5. Validate every output modality is declared in ``capabilities.output``;
           raise :class:`UnsupportedModalityError` otherwise.

        Parameters
        ----------
        input:
            The user input — a plain string or a structured :class:`AgentInput`.
        images:
            Optional list of images to include with the input. Each item can be:
            - A file path (str or Path) — will be read and sent as inline bytes.
            - A :class:`MediaPart` constructed via ``image(path=...)`` or ``Image(...)``.
        videos:
            Optional list of videos to include with the input. Same format as images.
        audio:
            Optional list of audio files to include with the input. Each item can be:
            - A file path (str or Path) — will be read and sent as inline bytes.
            - A :class:`MediaPart` constructed via ``Audio(...)`` or similar.
        output_schema:
            Optional Pydantic/dataclass schema for structured output validation.
        context:
            Optional :class:`RunContext` for flow-engine integration (Req 1.2).
            When ``None`` (the default), a fresh context is created internally so
            existing callers are unaffected.
        """
        # Early capability gating for audio (Req 4.4): raise before coercion/model call.
        if audio and Modality.AUDIO not in self.capabilities.input:
            raise UnsupportedModalityError("audio", self._model_id)

        agent_input = self._coerce_input(input, images=images, videos=videos, audio=audio)

        # --- Build a RunContext per run (Req 4.5, 11.1) ---
        # When a context is supplied externally (flow-engine integration), use it
        # so deps/shared_state propagate. Otherwise create a fresh one internally.
        if context is not None:
            ctx = context
        else:
            ctx = RunContext(
                events=self.events,
                max_steps=self.max_tool_iterations,
                token_budget=self._token_budget,
                loop_repeat_threshold=self.loop_repeat_threshold,
            )

        # --- Emit run_start event (Req 11.1) ---
        ctx.events.emit(Event(
            kind="run_start",
            t=time.monotonic(),
            attributes={"gen_ai.operation.name": "chat"},
        ))

        # Route: single-shot / tool-loop / self-plan. Multi-agent orchestration is
        # now handled via Flow (Req 14.4, 17.3). The agent keeps only single-agent
        # auto-escalation.
        # --- Consult the complexity_router before mode selection (Req 10.2/10.3) ---
        if self.complexity_router is not None:
            from .routing import RunStrategy

            has_tools = bool(self.tool_runtime._tools)
            strategy = self.complexity_router.classify(agent_input, has_tools=has_tools)

            if strategy == RunStrategy.SINGLE:
                result = await self._run_single(agent_input, output_schema=output_schema, ctx=ctx)
            elif strategy == RunStrategy.PLAN:
                result = await self._run_plan(agent_input, output_schema=output_schema, ctx=ctx)
            else:
                # TOOL_LOOP (default)
                result = await self._run_tool_loop(agent_input, output_schema=output_schema, ctx=ctx)
        else:
            # Default behavior (Req 10.3): tool-loop if tools exist, else single-shot.
            if self.tool_runtime._tools:
                result = await self._run_tool_loop(agent_input, output_schema=output_schema, ctx=ctx)
            else:
                result = await self._run_single(agent_input, output_schema=output_schema, ctx=ctx)

        # --- Emit run_end event (Req 11.1) ---
        ctx.events.emit(Event(
            kind="run_end",
            t=time.monotonic(),
            duration_ms=ctx.elapsed() * 1000,
            attributes={"gen_ai.operation.name": "chat"},
        ))

        # --- Copy trace onto RunResult (Req 12.1) ---
        if hasattr(self.events, "trace"):
            result.trace = self.events.trace

        # --- Output verification (Req 4.2–4.4) ---
        # When a verifier is configured, evaluate it against the final output and
        # record the verdict on the RunResult. When retry_on_failure is enabled and
        # the verifier reports failure, re-run with the failure detail appended to
        # context up to max_verify_retries times.
        if self.verifier is not None:
            from loomable.flow.loop import CallableVerifier, Verifier, VerdictResult

            # Resolve the verifier: callable → CallableVerifier adapter
            resolved_verifier: Verifier
            if callable(self.verifier) and not isinstance(self.verifier, Verifier):
                resolved_verifier = CallableVerifier(self.verifier)
            else:
                resolved_verifier = self.verifier

            verdict = resolved_verifier.check(result.output, ctx)
            result.verification = verdict

            # Retry on failure when enabled (Req 4.3)
            if not verdict.ok and self.retry_on_failure:
                retries_left = self.max_verify_retries
                while not verdict.ok and retries_left > 0:
                    retries_left -= 1
                    # Append the failure detail to the input for self-correction
                    retry_input_text = (
                        f"{_input_text(agent_input)}\n\n"
                        f"[Verification failed: {verdict.detail}]"
                    )
                    retry_agent_input = self._coerce_input(retry_input_text)

                    # Re-run via the same routing logic
                    if self.complexity_router is not None:
                        from .routing import RunStrategy

                        has_tools = bool(self.tool_runtime._tools)
                        strategy = self.complexity_router.classify(retry_agent_input, has_tools=has_tools)
                        if strategy == RunStrategy.SINGLE:
                            result = await self._run_single(retry_agent_input, output_schema=output_schema, ctx=ctx)
                        elif strategy == RunStrategy.PLAN:
                            result = await self._run_plan(retry_agent_input, output_schema=output_schema, ctx=ctx)
                        else:
                            result = await self._run_tool_loop(retry_agent_input, output_schema=output_schema, ctx=ctx)
                    else:
                        if self.tool_runtime._tools:
                            result = await self._run_tool_loop(retry_agent_input, output_schema=output_schema, ctx=ctx)
                        else:
                            result = await self._run_single(retry_agent_input, output_schema=output_schema, ctx=ctx)

                    # Re-verify
                    verdict = resolved_verifier.check(result.output, ctx)
                    result.verification = verdict

        # Persist conversational + session state after the run so it survives across
        # calls via the kernel SessionStore (Req 15.2): the input is recorded as a
        # user turn and the output as an assistant turn, the step counter advances,
        # then the session is saved.
        self._persist_session(_input_text(agent_input), result.output.text())
        return result

    def _coerce_input(
        self,
        value: Any,
        *,
        images: list | None = None,
        videos: list | None = None,
        audio: list | None = None,
    ) -> AgentInput:
        """Normalize any supported input into an :class:`AgentInput` (agno-style).

        Accepts a plain string, an :class:`AgentInput`, a Pydantic model, a dataclass
        instance, or a ``dict``. When an ``input_schema`` is configured, dict/model
        inputs are validated/coerced against it first (plain strings and existing
        :class:`AgentInput` values pass through unchanged). Validation failures raise
        :class:`~loomable.agent.errors.InputValidationError`.

        When ``images``, ``videos``, or ``audio`` are provided, they are appended as
        additional media parts to the user message, enabling multimodal input with a
        simple API. Each item is dispatched via :func:`_coerce_media_item`.
        """
        if self.input_schema is not None and not isinstance(value, (str, AgentInput)):
            value = self._validate_against_schema(value, self.input_schema)
        agent_input = to_agent_input(value)

        # Append images/videos/audio as additional media parts to the last user message.
        if images or videos or audio:
            extra_parts: list = []
            if images:
                for img in images:
                    extra_parts.append(_coerce_media_item(img, "image"))
            if videos:
                for vid in videos:
                    extra_parts.append(_coerce_media_item(vid, "video"))
            if audio:
                for aud in audio:
                    extra_parts.append(_coerce_media_item(aud, "audio"))

            if extra_parts and agent_input.messages:
                # Append to the last user message
                last_msg = agent_input.messages[-1]
                agent_input = AgentInput(
                    messages=agent_input.messages[:-1] + [
                        Message(role=last_msg.role, parts=last_msg.parts + extra_parts)
                    ]
                )

        return agent_input

    def _validate_against_schema(self, value: Any, schema: type) -> Any:
        """Validate/coerce ``value`` against a Pydantic or dataclass ``schema``."""
        # Pydantic model schema (lazy import so pydantic stays optional).
        try:
            import pydantic

            if isinstance(schema, type) and issubclass(schema, pydantic.BaseModel):
                if isinstance(value, schema):
                    return value
                try:
                    if isinstance(value, pydantic.BaseModel):
                        return schema.model_validate(value.model_dump())
                    if isinstance(value, dict):
                        return schema.model_validate(value)
                except pydantic.ValidationError as exc:
                    raise InputValidationError(
                        f"input does not match schema '{schema.__name__}': {exc}"
                    ) from exc
                raise InputValidationError(
                    f"cannot validate {type(value).__name__} against '{schema.__name__}'"
                )
        except ImportError:
            pass

        # Dataclass schema.
        if dataclasses.is_dataclass(schema) and isinstance(schema, type):
            if isinstance(value, schema):
                return value
            if isinstance(value, dict):
                try:
                    return schema(**value)
                except TypeError as exc:
                    raise InputValidationError(
                        f"input does not match dataclass '{schema.__name__}': {exc}"
                    ) from exc

        raise InputValidationError(
            f"unsupported input for schema '{getattr(schema, '__name__', schema)}'"
        )

    def _memory_prefix(self, include_history: bool = True) -> list[dict]:
        """Return recent conversation turns as leading messages for context (Req 15).

        Conversation memory is active only when the agent has a session
        (``persist_session``) and ``use_memory`` is enabled, so stateless agents and
        transport reuse are unaffected. The window caps how many recent turns are
        replayed to bound token usage. Turns are returned oldest-first so the model
        sees the conversation in order, ahead of the current input.

        When L2 summaries exist (produced by compaction), they are prepended as
        system messages ahead of the retained raw turns so the model has compressed
        context of earlier conversation (Req 6.3).
        """
        if not (include_history and self.use_memory and self.persist_session):
            return []

        messages: list[dict] = []

        # Prepend L2 summaries as system/context messages (Req 6.3).
        for summary in self.session.l2:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": summary.text}]}
            )

        # Append the retained recent raw turns from L1.
        turns = self.session.l1
        if self.memory_window and len(turns) > self.memory_window:
            turns = turns[-self.memory_window :]

        # Collect the step values of windowed turns so we can detect which pinned
        # turns are already included (avoid duplicates).
        windowed_steps = {t.step for t in turns}

        # Include pinned turns that are outside the window (Req 6.4): iterate all of
        # session.l1 and pick those whose step is pinned but not already windowed.
        pinned_outside_window = [
            t
            for t in self.session.l1
            if t.step in self.pinned_steps and t.step not in windowed_steps
        ]

        # Merge: pinned turns (oldest-first) followed by windowed turns so the model
        # sees the conversation in chronological order.
        all_turns = sorted(pinned_outside_window + list(turns), key=lambda t: t.step)

        messages.extend(
            {"role": turn.role, "content": [{"type": "text", "text": turn.content}]}
            for turn in all_turns
            if turn.content
        )
        return messages

    async def _recall_knowledge(self, agent_input: AgentInput) -> list[dict]:
        """Recall relevant knowledge snippets for the given input (Req 8.3).

        Embeds the input text, queries the LongTermStore for top-k results, and
        returns system messages containing the retrieved snippets to be prepended
        to the model context. Returns an empty list when knowledge is not configured.
        """
        if self.long_term is None or self.embedder is None:
            return []

        # Extract input text for embedding.
        input_text = _input_text(agent_input)
        if not input_text:
            return []

        # Embed the query and recall top-k from the LongTermStore (Req 8.3/8.5).
        query_vector = await self.embedder.embed(input_text)
        results = await self.long_term.query(query_vector, self.knowledge_top_k)

        # Build context messages from recalled snippets.
        messages: list[dict] = []
        for result in results:
            text = result.get("text", "")
            if text:
                messages.append(
                    {"role": "system", "content": [{"type": "text", "text": text}]}
                )
        return messages

    def _bound_messages(self, messages: list[dict], budget: int) -> list[dict]:
        """Feed messages through the kernel ContextManager to bound token usage (Req 13.1–13.4).

        Each message is mapped to a :class:`ContextItem` with a cheap token estimate
        (``len(json.dumps(msg)) // 4``). System/instructions messages and tool-schema
        messages are pinned (always retained). Messages from pinned steps (tracked via
        ``self.pinned_steps``) are also pinned. Other messages get ``kind="turn"``
        with priority based on their index (more recent = higher priority).

        After all items are admitted into a fresh ``ContextManager(budget)``, the
        assembled window determines which messages survive. The kept messages are
        returned in their original order.

        Parameters
        ----------
        messages:
            The full list of messages about to be sent to the model.
        budget:
            The token budget for context bounding.

        Returns
        -------
        list[dict]
            The subset of messages that fit within the budget, in original order.
        """
        if not messages:
            return messages

        from loomable.agent.context_policy import ContextPolicy

        policy = self.context_policy or ContextPolicy(
            memory_window=self.memory_window,
            compaction_threshold=self.compaction_threshold,
            token_budget=budget,
        )
        # Hard spill of bulky tool payloads before ContextManager eviction.
        messages = policy.spill_bulky_tool_messages(list(messages))

        # Build ContextItems, one per message, tracking the original index.
        items: list[tuple[int, ContextItem]] = []
        for idx, msg in enumerate(messages):
            # Cheap token estimate: character count of JSON serialization / 4.
            tokens = len(json.dumps(msg, default=str)) // 4

            role = msg.get("role", "")
            # Determine kind and pinning.
            if role == "system":
                kind: str = "system"
                pinned = True
            elif "tool_calls" not in msg and role == "assistant" and self._is_tool_schema_msg(msg):
                kind = "schema"
                pinned = True
            else:
                kind = "turn"
                # Pin messages corresponding to pinned steps.
                pinned = self._is_pinned_message(msg)

            # Priority: higher index = more recent = higher priority.
            priority = idx

            items.append((idx, ContextItem(kind=kind, tokens=tokens, priority=priority, pinned=pinned)))

        # Also pin tool-schema messages (those injected via request.tools are not in
        # messages; but tool_call_id messages and tool-result messages stay as turns).

        # Admit all items into a fresh ContextManager.
        cm = ContextManager(budget)
        for _idx, item in items:
            cm.admit(item)

        # Assemble the window: get the set of kept items.
        window = cm.assemble()
        kept_ids = {id(item) for item in window.items}

        # Map back to original messages: retain those whose ContextItem survived.
        result: list[dict] = []
        for idx, item in items:
            if id(item) in kept_ids:
                result.append(messages[idx])

        return result

    @staticmethod
    def _is_tool_schema_msg(msg: dict) -> bool:
        """Detect if a message is a tool schema definition.

        Tool schemas are typically embedded as system messages containing function
        definitions. This heuristic checks for common schema patterns.
        """
        # Tool schemas are advertised via request.tools, not as messages.
        # However, some structured-output hints or schema messages may look like schemas.
        # For now, only system messages with "function" schema content qualify.
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if '"type": "function"' in text and '"parameters"' in text:
                        return True
        return False

    def _is_pinned_message(self, msg: dict) -> bool:
        """Check if a message corresponds to a pinned step (Req 6.4, 13.3).

        Uses the ``pinned_steps`` set to determine if a turn's content matches a
        pinned fact. Since messages don't carry step metadata directly, we compare
        the message content against the content of turns at pinned steps in
        ``session.l1``.
        """
        if not self.pinned_steps:
            return False
        # Collect the content strings of all pinned turns.
        pinned_contents: set[str] = set()
        for turn in self.session.l1:
            if turn.step in self.pinned_steps and turn.content:
                pinned_contents.add(turn.content)
        if not pinned_contents:
            return False
        # Extract text content from the message for comparison.
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if text in pinned_contents:
                        return True
        elif isinstance(content, str) and content in pinned_contents:
            return True
        return False

    async def _run_plan(
        self,
        agent_input: AgentInput,
        *,
        output_schema: type | None = None,
        ctx: RunContext | None = None,
    ) -> RunResult:
        """Execute the self-plan strategy via the Flow engine (Req 17.2, 17.3).

        Builds a plan→map→synthesize Flow using :func:`plan_and_execute` from
        ``loomable.flow.helpers``, replacing the removed ``AutoPlan`` class. The
        planner, worker, and synthesizer are all backed by this agent's single-shot
        path so the agent's session/tools/knowledge remain available.
        """
        from loomable.flow.helpers import plan_and_execute

        task_text = _input_text(agent_input)

        async def _planner(input: Any, **kwargs: Any) -> dict:
            """Ask the model for a concise plan; return steps in shared state."""
            import json as _json

            plan_prompt = (
                "You are a planner. Break the user's task into at most 5 concrete, "
                "independent, actionable steps. Return ONLY a JSON array of short "
                "imperative step strings (e.g. [\"Do X\", \"Do Y\"]). "
                "No prose, no markdown, no code fences.\n\n"
                f"Task: {task_text}"
            )
            result = await self._run_single(
                AgentInput.from_text(plan_prompt), include_history=False, ctx=ctx
            )
            # Parse the plan response into a list of steps.
            text = result.output.text().strip()
            # Strip code fences if present.
            if text.startswith("```"):
                text = text.split("\n", 1)[-1] if "\n" in text else text
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if text.startswith("json"):
                    text = text[len("json"):].strip()
            try:
                steps = _json.loads(text)
                if not isinstance(steps, list):
                    steps = [text]
            except (ValueError, _json.JSONDecodeError):
                # Fallback: split on newlines and strip bullets
                steps = [
                    line.strip().lstrip("-*•0123456789.) ")
                    for line in text.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
            return {"plan_steps": steps[:5]}

        async def _worker(input: Any, **kwargs: Any) -> str:
            """Run a single plan step through the agent's single-shot path."""
            step = input if isinstance(input, str) else str(input)
            prompt = (
                f"Overall task:\n{task_text}\n\n"
                f"Complete ONLY this step, concisely and concretely:\n{step}"
            )
            result = await self._run_single(
                AgentInput.from_text(prompt), include_history=False, ctx=ctx
            )
            return result.output.text()

        async def _synthesizer(input: Any, *, context: Any = None, **kwargs: Any) -> str:
            """Combine step results into a final cohesive answer."""
            pieces: list[Any] = []
            if context is not None and getattr(context, "shared_state", None) is not None:
                raw = context.shared_state.get("map")
                if isinstance(raw, list):
                    pieces = raw
            if not pieces and isinstance(input, dict):
                pieces = input.get("map", []) or []
            if not pieces:
                pieces = [str(input)]
            combined = "\n".join(f"- {p}" for p in pieces)
            prompt = (
                f"Original task:\n{task_text}\n\n"
                f"Results from the planned steps:\n{combined}\n\n"
                "Integrate these into one cohesive, well-structured final answer."
            )
            result = await self._run_single(
                AgentInput.from_text(prompt),
                output_schema=output_schema,
                include_history=False,
                ctx=ctx,
            )
            return result.output.text()

        # Build and run the plan→map→synthesize flow.
        flow = plan_and_execute(
            planner=_planner,
            workers=_worker,
            synthesizer=_synthesizer,
            session_id=self.session.session_id,
        )
        flow_result = await flow.arun(AgentInput.from_text(task_text))

        # Wrap the flow result back into a RunResult with this agent's session.
        from loomable.content import AgentOutput, Text

        output_text = flow_result.output.text() if flow_result.output else ""
        output = AgentOutput(parts=[Text(output_text)])

        # Structured output validation if requested.
        structured: object | None = None
        if output_schema is not None:
            structured = _validate_structured(output_text, output_schema)

        return _finalize_run_result(
            RunResult(
                output=output,
                session_id=self.session.session_id,
                usage=flow_result.usage,
                tool_activity=[],
                structured=structured,
            )
        )

    async def _run_single(
        self,
        agent_input: AgentInput,
        *,
        output_schema: type | None = None,
        include_history: bool = True,
        ctx: RunContext | None = None,
    ) -> RunResult:
        """Execute one ordinary agent run (no orchestration, no persistence).

        This is the core single-turn path shared by :meth:`arun` and the autonomous
        planner's step/synthesis runs. It performs input capability gating, prepends
        the agent's recent conversation memory (Req 15) when enabled, invokes the
        model, rebuilds the multimodal output, performs output capability gating, and
        applies structured-output validation. Persistence and mode-routing are handled
        by :meth:`arun` so this method stays reusable and side-effect free.

        Set ``include_history=False`` to run without conversation memory — used for
        focused, self-contained subagent steps that carry their own explicit context.
        """
        if ctx is None:
            ctx = RunContext()
        # (2) Input capability gating — before any provider invocation (Req 6.3/6.4).
        for modality in agent_input.modalities():
            if modality not in self.capabilities.input:
                raise UnsupportedModalityError(modality.value, self._model_id)

        # (3) Assemble the request: [system instructions] + [knowledge context] +
        #     [conversation memory] + [current input], then optionally a trailing
        #     structured-output hint.
        request = to_model_request(agent_input)
        prefix: list[dict] = []
        if self.instructions:
            prefix.append(
                {"role": "system", "content": [{"type": "text", "text": self.instructions}]}
            )
        # Knowledge recall: embed the input and prepend top-k snippets (Req 8.3).
        knowledge_snippets = await self._recall_knowledge(agent_input)
        prefix.extend(knowledge_snippets)
        prefix.extend(self._memory_prefix(include_history))
        if prefix:
            request.messages = prefix + request.messages
        # Structured output: hint the model to emit JSON matching the schema (Req 13.1).
        if output_schema is not None:
            request.messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": _schema_instruction(output_schema)}
                    ],
                }
            )

        # (3b) Context bounding: evict low-priority messages to fit within the
        #      token budget before the model call (Req 13.1–13.4).
        effective_budget = ctx.token_budget or self._token_budget
        if effective_budget is not None:
            request.messages = self._bound_messages(request.messages, effective_budget)

        # Route through the tiered ModelRouter when configured (Req 7.1–7.3),
        # otherwise use the single model interface unchanged (Req 7.4).
        tier_substitution: TierSubstitution | None = None
        _model_call_t0 = time.monotonic()
        if self.router is not None:
            response, tier_substitution = await self.router.route(request)
        else:
            response = await self.model_interface.invoke(request)
        _model_call_duration = (time.monotonic() - _model_call_t0) * 1000

        # Track token usage from model response (Req 13.4).
        usage = response.usage if hasattr(response, "usage") and response.usage else {}
        tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if tokens_used:
            ctx.add_tokens(tokens_used)

        # Emit model_call event (Req 11.2).
        ctx.events.emit(Event(
            kind="model_call",
            t=time.monotonic(),
            duration_ms=_model_call_duration,
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            attributes={
                "gen_ai.request.model": self._model_id,
                "gen_ai.usage.input_tokens": usage.get("input_tokens", 0),
                "gen_ai.usage.output_tokens": usage.get("output_tokens", 0),
            },
        ))

        # Emit tier_substitution event if the router substituted a tier (Req 11).
        if tier_substitution is not None:
            ctx.events.emit(Event(
                kind="tier_substitution",
                t=time.monotonic(),
                attributes={
                    "gen_ai.request.model": str(tier_substitution),
                    "loomable.tier.requested": str(tier_substitution),
                },
            ))

        # (4) Rebuild the multimodal output.
        output = from_model_response(response)

        # (5) Output capability gating (Req 5.4).
        for modality in output.modalities():
            if modality not in self.capabilities.output:
                raise UnsupportedModalityError(modality.value, self._model_id)

        # (6) Structured output: parse/validate the text into the schema (Req 13.2/13.3).
        structured: object | None = None
        if output_schema is not None:
            structured = _validate_structured(output.text(), output_schema)

        # (7) Build metadata: record tier substitution if any (Req 7.2/7.3).
        metadata: dict[str, Any] = {}
        if tier_substitution is not None:
            metadata["tier_substitution"] = tier_substitution

        return _finalize_run_result(
            RunResult(
                output=output,
                session_id=self.session.session_id,
                usage=response.usage,
                tool_activity=[],
                structured=structured,
                metadata=metadata,
            ),
            provider_reasoning=list(getattr(response, "reasoning", None) or []),
        )

    async def _run_tool_loop(
        self,
        agent_input: AgentInput,
        *,
        output_schema: type | None = None,
        include_history: bool = True,
        ctx: RunContext | None = None,
    ) -> RunResult:
        """Execute the model→dispatch→feed-back loop when tools are present (Req 3).

        Advertises tool schemas via ``ModelRequest.tools``, and while the model
        returns tool calls (and iterations < ``max_tool_iterations``), dispatches
        them through the gated path (hooks/guardrails apply), appends assistant
        tool-call messages and tool-result messages to the conversation, and
        re-invokes the model. When the model returns no tool calls, returns the
        final response as the run output. Tool outcomes are collected into
        ``RunResult.tool_activity`` (Req 3.4).

        If the first model response contains no tool calls, behaves identically
        to the single-shot path (Req 3.6).

        Threading ``RunContext`` enables loop detection, cooperative cancellation,
        and step/token budget enforcement (Req 3.1–3.5, 4.1–4.4).
        """
        from .tools import FunctionTool, MCPTool

        if ctx is None:
            ctx = RunContext()

        # (1) Input capability gating — before any provider invocation (Req 6.3/6.4).
        for modality in agent_input.modalities():
            if modality not in self.capabilities.input:
                raise UnsupportedModalityError(modality.value, self._model_id)

        # (2) Build tool schemas to advertise to the model.
        tool_schemas: list[dict] = []
        for tool_obj in self.tool_runtime._tools.values():
            if isinstance(tool_obj, (FunctionTool, MCPTool)):
                tool_schemas.append(tool_obj.schema())
            else:
                # Generic schema for non-FunctionTool tools (e.g. RetrieverTool).
                tool_schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool_obj.name,
                        "description": getattr(tool_obj, "description", ""),
                        "parameters": getattr(tool_obj, "parameters", {"type": "object", "properties": {}}),
                    },
                })

        # (3) Assemble the initial request: [system] + [knowledge] + [memory] + [input] + tools.
        request = to_model_request(agent_input)
        prefix: list[dict] = []
        if self.instructions:
            prefix.append(
                {"role": "system", "content": [{"type": "text", "text": self.instructions}]}
            )
        # Knowledge recall: embed the input and prepend top-k snippets (Req 8.3).
        knowledge_snippets = await self._recall_knowledge(agent_input)
        prefix.extend(knowledge_snippets)
        prefix.extend(self._memory_prefix(include_history))
        if prefix:
            request.messages = prefix + request.messages
        # Structured output hint (Req 13.1).
        if output_schema is not None:
            request.messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": _schema_instruction(output_schema)}
                    ],
                }
            )
        request.tools = tool_schemas

        # (4) Tool-use loop: invoke, dispatch, feed back, repeat.
        tool_activity: list[ToolOutcome] = []
        called_tool_names: set[str] = set()
        write_json_payloads: list[str] = []
        require_tool_specs = _parse_require_tool_specs(list(self.require_tools))
        satisfied_require_tools: set[tuple[str, str | None]] = set()
        tier_substitutions: list[TierSubstitution] = []
        iterations = 0
        stop_reason: StopReason | None = None
        response = None  # May remain None if we break before the first model call.
        require_tools_nudges = 0
        has_path_constraints = any(path for _, path in require_tool_specs)
        max_require_tools_nudges = (
            max(1, len(require_tool_specs) * 2)
            if has_path_constraints
            else max(1, len(require_tool_specs))
        ) if require_tool_specs else 0

        while True:
            # --- Check cooperative cancellation at each loop boundary (Req 4.1) ---
            if ctx.cancelled:
                stop_reason = StopReason(kind=StopReason.CANCELLED)
                break

            # --- Check step budget at each loop boundary (Req 4.2/4.3) ---
            if not ctx.tick_step():
                stop_reason = StopReason(kind=StopReason.STEP_BUDGET)
                break

            # --- Check token budget at each loop boundary (Req 4.4) ---
            if ctx.token_budget_exceeded():
                stop_reason = StopReason(kind=StopReason.TOKEN_BUDGET)
                break

            # Context bounding: evict low-priority messages before model call (Req 13.1–13.4).
            effective_budget = ctx.token_budget or self._token_budget
            if effective_budget is not None:
                request.messages = self._bound_messages(request.messages, effective_budget)

            # Route through the tiered ModelRouter when configured (Req 7.1–7.3),
            # otherwise use the single model interface unchanged (Req 7.4).
            _loop_model_t0 = time.monotonic()
            if self.router is not None:
                response, tier_sub = await self.router.route(request)
                if tier_sub is not None:
                    tier_substitutions.append(tier_sub)
            else:
                response = await self.model_interface.invoke(request)
            _loop_model_duration = (time.monotonic() - _loop_model_t0) * 1000

            # Track token usage from model response (Req 4.4, 13.4).
            usage = response.usage if hasattr(response, "usage") and response.usage else {}
            tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            if tokens_used:
                ctx.add_tokens(tokens_used)

            # Emit model_call event (Req 11.2).
            ctx.events.emit(Event(
                kind="model_call",
                t=time.monotonic(),
                duration_ms=_loop_model_duration,
                tokens_in=usage.get("input_tokens", 0),
                tokens_out=usage.get("output_tokens", 0),
                attributes={
                    "gen_ai.request.model": self._model_id,
                    "gen_ai.usage.input_tokens": usage.get("input_tokens", 0),
                    "gen_ai.usage.output_tokens": usage.get("output_tokens", 0),
                },
            ))

            # Emit tier_substitution event if the router substituted a tier.
            if self.router is not None and tier_substitutions:
                latest_sub = tier_substitutions[-1]
                if latest_sub is tier_sub and tier_sub is not None:
                    ctx.events.emit(Event(
                        kind="tier_substitution",
                        t=time.monotonic(),
                        attributes={
                            "gen_ai.request.model": str(tier_sub),
                            "loomable.tier.requested": str(tier_sub),
                        },
                    ))

            iterations += 1

            # No tool calls: candidate final answer — but required tools may still
            # be missing (e.g. scribe returns JSON without write_file/write_json).
            if not response.tool_calls:
                missing_required = _missing_require_tool_specs(
                    require_tool_specs, satisfied_require_tools
                )
                if (
                    missing_required
                    and require_tools_nudges < max_require_tools_nudges
                    and not ctx.cancelled
                    and iterations < self.max_tool_iterations
                ):
                    require_tools_nudges += 1
                    prior_text = ""
                    if getattr(response, "content", None):
                        prior_text = str(response.content)
                    assistant_text = prior_text.strip() or "(attempted to finish)"
                    request.messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": assistant_text}],
                        }
                    )
                    request.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": _format_require_tools_nudge(missing_required),
                                }
                            ],
                        }
                    )
                    # Keep tools enabled so the model can satisfy the contract.
                    continue

                stop_reason = StopReason(kind=StopReason.FINAL)
                break

            # Max iterations reached: stop with MAX_ITERATIONS and re-invoke the
            # model once with a "you must answer now, no tools" system nudge so it
            # produces a non-tool final response (Req 3.3).
            if iterations >= self.max_tool_iterations:
                stop_reason = StopReason(kind=StopReason.MAX_ITERATIONS)
                # Re-invoke the model with a no-tools nudge for a final answer.
                request.messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": (
                        "You have reached the maximum number of tool iterations. "
                        "You must provide your final answer now. Do not request any "
                        "tool calls."
                    )}],
                })
                request.tools = []  # Remove tools so model cannot call them.
                # Context bounding before the nudge call (Req 13.1–13.4).
                if effective_budget is not None:
                    request.messages = self._bound_messages(request.messages, effective_budget)
                _nudge_t0 = time.monotonic()
                if self.router is not None:
                    response, tier_sub = await self.router.route(request)
                    if tier_sub is not None:
                        tier_substitutions.append(tier_sub)
                else:
                    response = await self.model_interface.invoke(request)
                _nudge_duration = (time.monotonic() - _nudge_t0) * 1000
                # Track tokens from the nudge call.
                nudge_usage = response.usage if hasattr(response, "usage") and response.usage else {}
                nudge_tokens = nudge_usage.get("input_tokens", 0) + nudge_usage.get("output_tokens", 0)
                if nudge_tokens:
                    ctx.add_tokens(nudge_tokens)
                # Emit model_call event for the nudge call (Req 11.2).
                ctx.events.emit(Event(
                    kind="model_call",
                    t=time.monotonic(),
                    duration_ms=_nudge_duration,
                    tokens_in=nudge_usage.get("input_tokens", 0),
                    tokens_out=nudge_usage.get("output_tokens", 0),
                    attributes={
                        "gen_ai.request.model": self._model_id,
                        "gen_ai.usage.input_tokens": nudge_usage.get("input_tokens", 0),
                        "gen_ai.usage.output_tokens": nudge_usage.get("output_tokens", 0),
                    },
                ))
                break

            # --- Loop detection: record each proposed call signature (Req 3.1/3.2) ---
            loop_detected = False
            for tc in response.tool_calls:
                count = ctx.record_call(tc.tool_name, tc.args)
                if count >= ctx.loop_repeat_threshold:
                    stop_reason = StopReason(
                        kind=StopReason.LOOP_DETECTED,
                        detail=f"tool '{tc.tool_name}' repeated {count} times",
                    )
                    loop_detected = True
                    break
            if loop_detected:
                break

            # --- Exclude non-idempotent tools from dispatch when they have already
            # been dispatched once (Req 3.5). Non-idempotent tools are never auto-
            # retried: if the model proposes a repeat of an idempotent=False tool,
            # exclude it from the batch dispatched to the runtime. ---
            calls_to_dispatch = []
            for tc in response.tool_calls:
                tool_obj = self.tool_runtime._tools.get(tc.tool_name)
                if tool_obj is not None and not getattr(tool_obj, "idempotent", True):
                    # A non-idempotent tool with a repeat count > 1 means it was
                    # already dispatched once; exclude it from re-dispatch.
                    sig = _ctx_signature(tc.tool_name, tc.args)
                    if ctx._call_history.get(sig, 0) > 1:
                        continue  # Skip re-dispatch of non-idempotent tool
                calls_to_dispatch.append(tc)

            # Dispatch tool calls through the gated path (hooks/guardrails, Req 3.5/3.7).
            _tool_dispatch_t0 = time.monotonic()
            gated_result = await self.dispatch_tools_gated(calls_to_dispatch)
            _tool_dispatch_duration = (time.monotonic() - _tool_dispatch_t0) * 1000

            # Annotate outcomes with tool_name for display/access convenience.
            call_name_map = {tc.id: tc.tool_name for tc in calls_to_dispatch}
            for outcome in gated_result.outcomes:
                if outcome.result is not None and "tool_name" not in outcome.result.metadata:
                    outcome.result.metadata["tool_name"] = call_name_map.get(outcome.call_id, "")

            tool_activity.extend(gated_result.outcomes)
            for tc in calls_to_dispatch:
                called_tool_names.add(tc.tool_name)
                path_arg = str(tc.args.get("path") or "")
                for name, path_sub in require_tool_specs:
                    if tc.tool_name != name:
                        continue
                    if path_sub is None or path_sub in path_arg:
                        satisfied_require_tools.add((name, path_sub))
                if tc.tool_name == "write_json":
                    outcome = next(
                        (o for o in gated_result.outcomes if o.call_id == tc.id),
                        None,
                    )
                    ok = (
                        outcome is not None
                        and outcome.result is not None
                        and not outcome.result.is_error
                    )
                    raw = tc.args.get("content")
                    if ok and raw is not None:
                        if isinstance(raw, str):
                            write_json_payloads.append(raw)
                        else:
                            write_json_payloads.append(json.dumps(raw))

            # Emit tool_call event (Req 11.3).
            tool_names = [tc.tool_name for tc in calls_to_dispatch]
            ctx.events.emit(Event(
                kind="tool_call",
                t=time.monotonic(),
                duration_ms=_tool_dispatch_duration,
                attributes={
                    "gen_ai.tool.name": ",".join(tool_names) if tool_names else "",
                    "tool_count": len(calls_to_dispatch),
                },
            ))

            # Append assistant message with tool calls to the conversation.
            assistant_tool_calls = []
            for tc in response.tool_calls:
                entry: dict[str, Any] = {
                    "id": tc.id,
                    "tool_name": tc.tool_name,
                    "args": tc.args,
                }
                if tc.metadata:
                    entry["metadata"] = tc.metadata
                    # Flatten Gemini thought signature for OpenAI-compat replay.
                    if "extra_content" in tc.metadata:
                        entry["extra_content"] = tc.metadata["extra_content"]
                assistant_tool_calls.append(entry)
            request.messages.append({
                "role": "assistant",
                "content": [],
                "tool_calls": assistant_tool_calls,
            })

            # Build a mapping from tool_name to call_id(s) for correlating blocked calls.
            name_to_call_ids: dict[str, list[str]] = {}
            executed_call_ids = {o.call_id for o in gated_result.outcomes}
            for tc in response.tool_calls:
                name_to_call_ids.setdefault(tc.tool_name, []).append(tc.id)

            # Append tool result messages for each executed outcome.
            for outcome in gated_result.outcomes:
                if outcome.result is not None:
                    result_content = str(outcome.result.content) if outcome.result.content is not None else ""
                    if outcome.result.is_error:
                        result_content = outcome.result.error or ""
                else:
                    result_content = outcome.error.message if outcome.error else "unknown error"

                request.messages.append({
                    "role": "tool",
                    "content": [{"type": "text", "text": result_content}],
                    "tool_call_id": outcome.call_id,
                })

            # --- Feedback injection: inject tool-generated media into the
            # conversation so the model can reason about it (Req 7.1–7.5). ---
            # IMPORTANT: append media as a follow-up *user* message. Attaching
            # image_url parts onto role=tool messages breaks Gemini's OpenAI-
            # compatible endpoint ("Invalid content part type: image_url").
            if self._feedback_media:
                from loomable.content.capabilities import _part_to_content

                feedback_parts: list[dict[str, Any]] = []
                for outcome in gated_result.outcomes:
                    if outcome.result is None:
                        continue
                    media_items = outcome.result.metadata.get("media", [])
                    if not media_items:
                        continue
                    for media_item in media_items:
                        modality = getattr(media_item, "_modality", None)
                        if modality is not None and modality in self.capabilities.input:
                            part = media_item.to_media_part()
                            content_entry = _part_to_content(part)
                            feedback_parts.append(content_entry)
                if feedback_parts:
                    request.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "[System: the previous tool call produced "
                                        "media. Inspect it before continuing.]"
                                    ),
                                },
                                *feedback_parts,
                            ],
                        }
                    )

            # Append error messages for blocked calls (guardrail rejections).
            # Map blocked tool_names back to their call_ids from the original request.
            for violation in gated_result.blocked:
                blocked_name = violation.action
                call_ids = name_to_call_ids.get(blocked_name, [])
                for call_id in call_ids:
                    if call_id not in executed_call_ids:
                        request.messages.append({
                            "role": "tool",
                            "content": [{"type": "text", "text": f"Tool call blocked: {violation.rule_id}"}],
                            "tool_call_id": call_id,
                        })
                        executed_call_ids.add(call_id)  # prevent duplicate messages

            # Append skipped non-idempotent tool messages.
            dispatched_ids = {tc.id for tc in calls_to_dispatch}
            for tc in response.tool_calls:
                if tc.id not in dispatched_ids and tc.id not in executed_call_ids:
                    request.messages.append({
                        "role": "tool",
                        "content": [{"type": "text", "text": (
                            f"Tool '{tc.tool_name}' was not re-executed because it is "
                            "non-idempotent and has already been dispatched."
                        )}],
                        "tool_call_id": tc.id,
                    })

        # If no explicit stop reason was set (shouldn't happen, but defensive).
        if stop_reason is None:
            stop_reason = StopReason(kind=StopReason.FINAL)

        # Emit a loop_stop event (Req 3.4, 11).
        ctx.events.emit(Event(
            kind="loop_stop",
            t=time.monotonic(),
            attributes={"stop_reason": stop_reason.kind, "detail": stop_reason.detail},
        ))

        # (5) Rebuild the multimodal output from the final response.
        # If we broke before any model call (early cancel/budget), produce an empty
        # output so the result is still valid.
        if response is not None:
            output = from_model_response(response)
        else:
            # Early exit (cancel/budget) before any model call — produce a minimal
            # empty-text output so the result is still valid.
            from loomable.content.parts import MediaPart
            output = AgentOutput(
                parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=b"")]
            )

        # (5b) Empty-final recovery: some providers return empty content after a
        # tool-only turn. When require_final_text or an output schema is set,
        # re-prompt once without tools — but only on a normal/max-iter finish,
        # not after loop/budget/cancel stops.
        final_text_reprompted = False
        structured_from_write_json = False
        stop_kind = stop_reason.kind if stop_reason is not None else StopReason.FINAL
        allow_empty_reprompt = stop_kind in (StopReason.FINAL, StopReason.MAX_ITERATIONS)
        needs_text = (
            allow_empty_reprompt
            and (self.require_final_text or output_schema is not None)
            and response is not None
            and not (output.text() or "").strip()
            and not ctx.cancelled
        )
        if needs_text:
            if output_schema is not None:
                nudge = (
                    "Your previous reply had no text. "
                    + _schema_instruction(output_schema)
                )
            else:
                nudge = (
                    "Your previous reply had no text. "
                    "Provide a brief final answer confirming what you did "
                    "(or the answer to the user). Do not call tools."
                )
            request.messages.append({"role": "user", "content": nudge})
            request.tools = []
            response = await self.model_interface.invoke(request)
            output = from_model_response(response)
            final_text_reprompted = True

        # (6) Output capability gating (Req 5.4).
        for modality in output.modalities():
            if modality not in self.capabilities.output:
                raise UnsupportedModalityError(modality.value, self._model_id)

        # (7) Structured output: parse/validate (Req 13.2/13.3).
        # Prefer final text; if empty/invalid after tools, reuse a successful
        # write_json payload (common when the model treats the tool as the answer).
        structured: object | None = None
        if output_schema is not None and response is not None:
            final_text = (output.text() or "").strip()
            if final_text:
                try:
                    structured = _validate_structured(final_text, output_schema)
                except StructuredOutputError:
                    structured = None
            if structured is None and write_json_payloads:
                for payload in reversed(write_json_payloads):
                    try:
                        structured = _validate_structured(payload, output_schema)
                    except StructuredOutputError:
                        continue
                    structured_from_write_json = True
                    if not final_text:
                        from loomable.content.parts import MediaPart as _MP

                        output = AgentOutput(
                            parts=[
                                _MP(
                                    modality=Modality.TEXT,
                                    media_type="text/plain",
                                    data=_extract_json_object(payload).encode("utf-8"),
                                )
                            ]
                        )
                    break
            if structured is None:
                # Preserve prior behavior: raise with schema error on final text.
                structured = _validate_structured(output.text() or "", output_schema)

        # (8) Build metadata: record tier substitutions if any (Req 7.2/7.3)
        #     and stop reason (Req 3.4).
        metadata: dict[str, Any] = {}
        if tier_substitutions:
            metadata["tier_substitution"] = tier_substitutions[-1]
            metadata["tier_substitutions"] = tier_substitutions
        metadata["stop_reason"] = stop_reason.kind
        if final_text_reprompted:
            metadata["final_text_reprompted"] = True
        if require_tools_nudges:
            metadata["require_tools_nudged"] = True
            metadata["require_tools_nudges"] = require_tools_nudges
        if structured_from_write_json:
            metadata["structured_from_write_json"] = True
        still_missing = _missing_require_tool_specs(
            require_tool_specs, satisfied_require_tools
        )
        if still_missing:
            metadata["required_tools_missing"] = still_missing

        provider_reasoning: list[str] = []
        if response is not None:
            provider_reasoning = list(getattr(response, "reasoning", None) or [])

        return _finalize_run_result(
            RunResult(
                output=output,
                session_id=self.session.session_id,
                usage=response.usage if response is not None else {},
                tool_activity=tool_activity,
                structured=structured,
                metadata=metadata,
            ),
            provider_reasoning=provider_reasoning,
        )

    def _persist_session(self, input_text: str, output_text: str) -> None:
        """Record the run's turns into the session and persist it (Req 15.2).

        Appends a user turn for the input and an assistant turn for the output to
        ``session.l1``, advances ``session.step``, and saves the session through the
        kernel :class:`~loomable.kernel.stores.SessionStore`. A no-op when no session
        store is configured.

        After recording, if ``len(session.l1)`` exceeds ``compaction_threshold``, the
        oldest turns beyond the retained window are summarized via the kernel
        :class:`~loomable.kernel.summarizer.Summarizer` into a
        :class:`~loomable.kernel.models.StructuredSummary`, stored in ``session.l2``,
        and dropped from ``session.l1`` (Req 6.1–6.5).
        """
        if not self.persist_session or self.session_store is None:
            return
        self.session.l1.append(
            Turn(
                role="user",
                content=input_text,
                tokens=0,
                step=self.session.step,
            )
        )
        self.session.l1.append(
            Turn(
                role="assistant",
                content=output_text,
                tokens=0,
                step=self.session.step,
            )
        )
        self.session.step += 1

        # --- Automatic memory compaction via ContextPolicy ---
        from loomable.agent.context_policy import ContextPolicy

        policy = self.context_policy or ContextPolicy(
            memory_window=self.memory_window,
            compaction_threshold=self.compaction_threshold,
            token_budget=self._token_budget or 8192,
        )
        if self.summarizer is not None and policy.should_compact_turns(len(self.session.l1)):
            new_l1, summaries, outcome = policy.compact_turns(
                self.session.l1,
                pinned_steps=self.pinned_steps,
                summarizer=self.summarizer,
            )
            if outcome.compacted:
                self.session.l1 = new_l1
                self.session.l2.extend(summaries)
                self.events.emit(Event(
                    kind="compaction",
                    t=time.monotonic(),
                    attributes={
                        "turns_compacted": outcome.turns_before - outcome.turns_after,
                        "summary_tokens": getattr(summaries[0], "tokens", 0) if summaries else 0,
                        "reason": outcome.reason,
                    },
                ))

        self.session_store.save(self.session)

    async def astream(
        self,
        input: AgentInput | str,  # noqa: A002
        *,
        output_schema: type | None = None,
    ) -> "AsyncIterator[RunChunk]":
        """Stream incremental output as :class:`RunChunk`s (Req 1.5).

        When the active provider implements ``stream()``, real token-level deltas
        are yielded as they arrive. Otherwise, falls back to running ``arun()`` and
        chunking its output (preserving pre-feature behavior).

        The same context assembly (instructions, knowledge, memory prefix, token
        bounding) and capability gating apply as in the non-streaming path.
        Session state is persisted identically to ``arun`` so streamed and
        non-streamed runs leave the same durable state.
        """
        from loomable.content import Text as _Text

        agent_input = self._coerce_input(input)

        # Resolve the provider for streaming detection
        provider = self.model_interface._providers.get(self.model_interface.default_provider)

        if provider is not None and hasattr(provider, "stream"):
            # --- Real streaming path ---
            # (1) Input capability gating
            for modality in agent_input.modalities():
                if modality not in self.capabilities.input:
                    raise UnsupportedModalityError(modality.value, self._model_id)

            # (2) Assemble the request (same as _run_single)
            request = to_model_request(agent_input)
            prefix: list[dict] = []
            if self.instructions:
                prefix.append(
                    {"role": "system", "content": [{"type": "text", "text": self.instructions}]}
                )
            knowledge_snippets = await self._recall_knowledge(agent_input)
            prefix.extend(knowledge_snippets)
            prefix.extend(self._memory_prefix())
            if prefix:
                request.messages = prefix + request.messages
            if output_schema is not None:
                request.messages.append(
                    {"role": "system", "content": [{"type": "text", "text": _schema_instruction(output_schema)}]}
                )
            if self._token_budget is not None:
                request.messages = self._bound_messages(request.messages, self._token_budget)

            # (3) Stream from provider
            accumulated_text = ""
            async for event in provider.stream(request):
                if event.kind == "text" and event.text:
                    accumulated_text += event.text
                    yield RunChunk(delta=_Text(event.text))
                elif event.kind == "end":
                    break

            # (4) Terminal chunk
            yield RunChunk(delta=_Text(""), done=True)

            # (5) Persist session state (same as arun)
            self._persist_session(_input_text(agent_input), accumulated_text)
        else:
            # --- Fallback: run then chunk ---
            result = await self.arun(input, output_schema=output_schema)
            parts = result.output.parts
            last_index = len(parts) - 1
            for index, part in enumerate(parts):
                yield RunChunk(delta=part, done=index == last_index)

    async def astream_events(
        self,
        input: AgentInput | str,  # noqa: A002
        *,
        images: "list[str | Path | MediaPart] | None" = None,
        videos: "list[str | Path | MediaPart] | None" = None,
        audio: "list[str | Path | MediaPart] | None" = None,
        output_schema: type | None = None,
        context: "RunContext | None" = None,
    ) -> "AsyncIterator[Any]":
        """Yield AG-UI-compatible :class:`~loomable.stream.StreamEvent` frames.

        Covers the full tool-loop ``arun`` path (lifecycle + tools + final text).
        When the provider supports token streaming and the run is single-shot
        (no tools), text deltas are streamed as ``TEXT_MESSAGE_CONTENT``.
        """
        import uuid
        from loomable.stream import (
            RUN_ERROR,
            RUN_FINISHED,
            TEXT_MESSAGE_CONTENT,
            TEXT_MESSAGE_END,
            TEXT_MESSAGE_START,
            AsyncStreamBus,
            StreamBridge,
            StreamEvent,
        )

        run_id = uuid.uuid4().hex
        session_id = getattr(self.session, "session_id", "") or ""
        bus = AsyncStreamBus()
        bridge = StreamBridge(
            bus,
            run_id=run_id,
            session_id=session_id,
            inner=self.events,
        )
        original_events = self.events
        self.events = bridge  # type: ignore[assignment]

        async def _runner() -> None:
            try:
                # Prefer token streaming for tool-less single-shot when possible.
                has_tools = bool(getattr(self.tool_runtime, "_tools", None))
                provider = self.model_interface._providers.get(
                    self.model_interface.default_provider
                )
                if (
                    not has_tools
                    and provider is not None
                    and hasattr(provider, "stream")
                    and images is None
                    and videos is None
                    and audio is None
                ):
                    from loomable.agent.events import Event
                    import time as _time

                    bridge.emit(
                        Event(
                            kind="run_start",
                            t=_time.monotonic(),
                            attributes={"gen_ai.operation.name": "chat"},
                        )
                    )
                    accumulated = ""
                    bridge.publish(TEXT_MESSAGE_START, {"role": "assistant"})
                    async for chunk in self.astream(input, output_schema=output_schema):
                        if chunk.delta is not None:
                            try:
                                delta_text = (
                                    chunk.delta.data.decode("utf-8")
                                    if chunk.delta.data
                                    else ""
                                )
                            except Exception:  # noqa: BLE001
                                delta_text = ""
                            if delta_text:
                                accumulated += delta_text
                                bridge.publish(
                                    TEXT_MESSAGE_CONTENT, {"delta": delta_text}
                                )
                        if chunk.done:
                            break
                    bridge.publish(TEXT_MESSAGE_END, {"text": accumulated})
                    bridge.publish(RUN_FINISHED, {"text": accumulated})
                else:
                    result = await self.arun(
                        input,
                        images=images,
                        videos=videos,
                        audio=audio,
                        output_schema=output_schema,
                        context=context,
                    )
                    text = (result.output.text() or "") if result.output else ""
                    if text:
                        bridge.publish(TEXT_MESSAGE_START, {"role": "assistant"})
                        bridge.publish(TEXT_MESSAGE_CONTENT, {"delta": text})
                        bridge.publish(TEXT_MESSAGE_END, {"text": text})
                    bridge.publish(RUN_FINISHED, {"text": text})
            except Exception as exc:  # noqa: BLE001
                bridge.publish(
                    RUN_ERROR,
                    {"message": str(exc), "error_type": type(exc).__name__},
                )
            finally:
                self.events = original_events
                await bus.close()

        task = asyncio.create_task(_runner())
        try:
            async for event in bus.events():
                yield event
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            else:
                exc = task.exception()
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    raise exc

    async def dispatch_tools(self, calls: list[ToolCall]) -> list[ToolOutcome]:
        """Dispatch multiple tool calls concurrently (Req 12.1–12.4).

        This is the explicit high-level entry point for a multi-tool step: when a
        step yields several independent tool calls, they are dispatched through the
        kernel :class:`~loomable.kernel.tool_runtime.ToolRuntime` unchanged (Req 12.4).
        The kernel runtime runs the calls concurrently (Req 12.1), matches each
        :class:`~loomable.kernel.models.ToolOutcome` to its originating call via
        ``call_id`` (Req 12.2), and isolates faults so that one failing tool yields an
        error outcome while its siblings still return results (Req 12.3).

        This is a thin surface over the kernel runtime — it holds no dispatch logic of
        its own and simply delegates to ``self.tool_runtime.dispatch``.

        Parameters
        ----------
        calls:
            The independent tool calls produced by a single agent step. Each
            :class:`~loomable.kernel.models.ToolCall` carries a distinct ``id``.

        Returns
        -------
        list[ToolOutcome]
            One outcome per input call, in the same order, each carrying the
            originating ``call_id`` with either a ``result`` or an isolated ``error``.
        """
        return await self.tool_runtime.dispatch(calls)

    # ------------------------------------------------------------------
    # Tool hooks / Human-in-the-loop gated dispatch (Req 14)
    # ------------------------------------------------------------------

    def _pre_hooks(self) -> list[ToolHook]:
        """The registered pre-hooks (hooks not explicitly marked ``phase == "post"``)."""
        return [h for h in self.tool_hooks if getattr(h, "phase", "pre") != "post"]

    def _post_hooks(self) -> list[PostToolHook]:
        """The registered post-hooks: ``phase == "post"`` entries plus ``post_tool_hooks``."""
        marked = [h for h in self.tool_hooks if getattr(h, "phase", "pre") == "post"]
        return [*marked, *self.post_tool_hooks]

    # ------------------------------------------------------------------
    # Per-tool timeout + concurrency cap (Req 2.1–2.4)
    # ------------------------------------------------------------------

    async def _dispatch_with_limits(
        self, calls: list[ToolCall]
    ) -> list[ToolOutcome]:
        """Dispatch tool calls with per-tool timeout and concurrency cap.

        Wraps each kernel :class:`~loomable.kernel.tool_runtime.ToolRuntime` call
        with ``asyncio.wait_for(per_tool_timeout)`` and caps parallelism with an
        ``asyncio.Semaphore(tool_concurrency)``. A timeout produces a
        :class:`~loomable.kernel.models.ToolOutcome` carrying a
        :class:`~loomable.kernel.models.ToolError` naming the tool — the call is
        NOT retried, just fed back to the model so it can replan.

        Sibling calls still complete even when one times out (Req 2.2).

        Parameters
        ----------
        calls:
            The approved tool calls to dispatch.

        Returns
        -------
        list[ToolOutcome]
            One outcome per input call, preserving order. Timed-out calls carry a
            ``ToolError`` with a descriptive message.
        """
        if not calls:
            return []

        timeout = self.tool_timeout
        semaphore = (
            asyncio.Semaphore(self.tool_concurrency)
            if self.tool_concurrency is not None
            else None
        )

        async def _run_one(call: ToolCall) -> ToolOutcome:
            """Run a single tool call, respecting timeout and concurrency."""
            async def _invoke() -> ToolOutcome:
                results = await self.tool_runtime.dispatch([call])
                return results[0]

            if semaphore is not None:
                async with semaphore:
                    if timeout is not None:
                        try:
                            return await asyncio.wait_for(_invoke(), timeout=timeout)
                        except asyncio.TimeoutError:
                            return ToolOutcome(
                                call_id=call.id,
                                error=ToolError(
                                    message=f"tool '{call.tool_name}' timed out after {timeout}s",
                                    details={"tool_name": call.tool_name, "timeout": timeout},
                                ),
                            )
                    else:
                        return await _invoke()
            else:
                if timeout is not None:
                    try:
                        return await asyncio.wait_for(_invoke(), timeout=timeout)
                    except asyncio.TimeoutError:
                        return ToolOutcome(
                            call_id=call.id,
                            error=ToolError(
                                message=f"tool '{call.tool_name}' timed out after {timeout}s",
                                details={"tool_name": call.tool_name, "timeout": timeout},
                            ),
                        )
                else:
                    return await _invoke()

        # Run all calls concurrently (bounded by semaphore when configured).
        # asyncio.gather ensures sibling calls complete even if one times out.
        tasks = [_run_one(call) for call in calls]
        return list(await asyncio.gather(*tasks))

    def _rejected_by_pre_hooks(self, call: ToolCall) -> bool:
        """Run pre-hooks for a single call; return True if any hook rejects it.

        A hook rejects by returning ``False`` or by raising
        :class:`~loomable.agent.errors.ToolHookRejection` (Req 14.3). Any other return
        value allows the call. The first rejecting hook short-circuits the rest.
        """
        for hook in self._pre_hooks():
            try:
                decision = hook(call.tool_name, call, call.args)
            except ToolHookRejection:
                return True
            if decision is False:
                return True
        return False

    async def dispatch_tools_gated(
        self, calls: list[ToolCall]
    ) -> GatedDispatchResult:
        """Dispatch tool calls through the hook + confirmation gate (Req 14).

        The gated path layers human-in-the-loop controls over the kernel tool runtime
        without modifying ``loomable.kernel`` (Req 14.5):

        1. Any configured static :class:`~loomable.kernel.guardrails.GuardrailHarness`
           is applied first; blocked calls are recorded and never executed.
        2. Pre-hooks run before dispatch. A call rejected by a pre-hook is expressed as
           a guardrail rule and evaluated through the kernel ``GuardrailHarness`` so the
           block + record reuses the kernel, and the call is not executed (Req
           14.1/14.2/14.3/14.5).
        3. For tools listed in ``require_confirmation``, the injectable ``approver`` is
           consulted; the tool executes only when approval is granted, otherwise the
           denial is recorded (default deny — headless-safe) (Req 14.4).
        4. Surviving/approved calls are dispatched concurrently through the kernel
           :class:`~loomable.kernel.tool_runtime.ToolRuntime`, then post-hooks run over
           the outcomes and may observe or transform each one (Req 14.1/14.2).

        Parameters
        ----------
        calls:
            The tool calls proposed by a single agent step.

        Returns
        -------
        GatedDispatchResult
            The executed ``outcomes`` and the ``blocked`` guardrail violations for calls
            that a pre-hook rejected or that failed confirmation.
        """
        blocked: list[GuardrailViolation] = []

        # (1) Existing static guardrails: block + record via the kernel harness.
        if self.harness is not None:
            surviving, violations = self.harness.evaluate(list(calls))
            blocked.extend(violations)
        else:
            surviving = list(calls)

        # (2) Pre-hooks: express rejection as a guardrail rule so the kernel harness
        #     performs the block + record (Req 14.3/14.5). Per-call decisions are made
        #     here; the kernel harness produces the violation records.
        hook_rejected = [c for c in surviving if self._rejected_by_pre_hooks(c)]
        if hook_rejected:
            rejection_harness = GuardrailHarness(
                [
                    {
                        "rule_id": _HOOK_REJECTION_RULE_ID,
                        "blocked_tools": [c.tool_name for c in hook_rejected],
                    }
                ]
            )
            _, hook_violations = rejection_harness.evaluate(hook_rejected)
            blocked.extend(hook_violations)
            rejected_ids = {id(c) for c in hook_rejected}
            surviving = [c for c in surviving if id(c) not in rejected_ids]

        # (3) Confirmation gate: execute only when the approver grants approval (Req 14.4).
        approved: list[ToolCall] = []
        for call in surviving:
            if self.require_confirmation and call.tool_name in self.require_confirmation:
                if self.approver(call):
                    approved.append(call)
                else:
                    blocked.append(
                        GuardrailViolation(
                            rule_id=_CONFIRMATION_RULE_ID, action=call.tool_name
                        )
                    )
            else:
                approved.append(call)

        # (4) Dispatch the surviving/approved calls through the kernel runtime.
        #     Route through _dispatch_with_limits when per-tool timeout or concurrency
        #     cap is configured (Req 2.1–2.4); otherwise dispatch directly.
        if self.tool_timeout is not None or self.tool_concurrency is not None:
            outcomes = await self._dispatch_with_limits(approved)
        else:
            outcomes = await self.tool_runtime.dispatch(approved)

        # (5) Post-hooks: observe or transform each outcome (Req 14.2).
        post_hooks = self._post_hooks()
        if post_hooks:
            call_by_id = {c.id: c for c in approved}
            transformed: list[ToolOutcome] = []
            for outcome in outcomes:
                call = call_by_id.get(outcome.call_id)
                tool_name = call.tool_name if call is not None else ""
                current = outcome
                for hook in post_hooks:
                    replacement = hook(tool_name, call, current)
                    if isinstance(replacement, ToolOutcome):
                        current = replacement
                transformed.append(current)
            outcomes = transformed

        return GatedDispatchResult(outcomes=outcomes, blocked=blocked)


class Agent:
    """High-level agent builder.

    The constructor is cheap: it only records configuration. Call :meth:`build` to
    validate the configuration and construct the runnable :class:`BuiltAgent`.
    """

    def __init__(
        self,
        model: "ModelProvider | ModelSpec | str",
        *,
        name: str = "",
        description: str = "",
        role: str = "",
        goal: str = "",
        instructions: str | None = None,
        tools: "list[Tool | Toolkit] | None" = None,
        subagents: "list[Agent] | None" = None,
        skills: list[Path] | None = None,
        mcp_servers: list[Any] | None = None,
        capabilities: ModelCapabilities | str | list[str] | None = None,
        modalities: str | list[str] | None = None,
        text_only: bool = False,
        multimodal: bool = False,
        token_budget: int = 8192,
        checkpoint_interval: int = 5,
        session_id: str | None = None,
        user_id: str | None = None,
        resume: bool = False,
        use_memory: bool = True,
        memory_window: int = 8,
        compaction_threshold: int = 16,
        input_schema: type | None = None,
        response_model: type | None = None,
        # knowledge / RAG:
        retrievers: list[Retriever] | None = None,
        knowledge: list[str] | None = None,
        embedder: Any = None,
        knowledge_top_k: int = 3,
        # tool hooks / HITL:
        tool_hooks: list[Any] | None = None,
        require_confirmation: list[str] | None = None,
        # harness knobs (avoids needing build() for common config):
        tool_timeout: float | None = None,
        tool_concurrency: int | None = None,
        max_tool_iterations: int | None = None,
        require_final_text: bool = True,
        require_tools: list[str] | None = None,
        # Tiered model routing:
        tiers: dict[str, Any] | None = None,
        tier_policy: dict[str, Any] | None = None,
        fallback_tiers: dict[str, str] | None = None,
        # low-level overrides:
        context_manager: ContextManager | None = None,
        memory: MemoryManager | None = None,
        tool_runtime: ToolRuntime | None = None,
        harness: GuardrailHarness | None = None,
        planner: Planner | None = None,
        session_store: SessionStore | None = None,
        # Harness features:
        events: AgentEvents | None = None,
        complexity_router: "ComplexityRouter | None" = None,
        note_store: "NoteStore | None" = None,
        loop_repeat_threshold: int = 3,
        resilience: "RetryPolicy | None" = None,
        # Output verification (Req 4.2–4.4):
        verifier: Any = None,
        retry_on_failure: bool = False,
        max_verify_retries: int = 1,
        use_llm_summarizer: bool = False,
        think_tool: bool = False,
        plan_tool: bool = False,
        memory_tool: bool = False,
        # Case mode (plan → dispatch → synthesize → accept):
        mode: str | None = None,
        dispatch: str = "reuse",
        accept: Any = None,
        board: bool = True,
        max_rounds: int | None = None,
        max_plan_steps: int = 5,
        # Multimodal feedback (Req 7.5):
        feedback_media: bool = True,
        # Developer experience:
        debug: bool = False,
        # Lifecycle callbacks:
        on_tool_call: Any = None,
        on_complete: Any = None,
    ) -> None:
        # --- Resolve model string shorthand (e.g. "openai:gpt-4o-mini") ---
        if isinstance(model, str):
            from loomable.providers.resolver import resolve_model
            resolved_provider = resolve_model(model)
            model = ModelSpec(provider=model.split(":")[0] if ":" in model else "openai",
                             provider_impl=resolved_provider)

        # Req 1.1/1.6: a model is required.
        if model is None:
            raise AgentConfigError("model")

        self._model = model
        self._name = name
        self._description = description
        self._role = role
        self._goal = goal
        self._instructions = instructions
        self._tools = tools
        self._subagents = subagents
        self._skills = skills
        self._mcp_servers = mcp_servers
        # High-level modality DX: modalities="text" / text_only=True preferred over
        # constructing ModelCapabilities with frozensets.
        from loomable.content.capabilities import capabilities_for

        self._modalities_raw = modalities if isinstance(modalities, str) else None
        if text_only:
            self._modalities_raw = "text"

        if text_only and (modalities is not None or capabilities is not None):
            raise AgentConfigError("text_only")
        if modalities is not None and capabilities is not None:
            raise AgentConfigError("modalities")
        if text_only:
            self._capabilities = capabilities_for("text")
        elif modalities is not None:
            self._capabilities = capabilities_for(modalities)
        elif isinstance(capabilities, ModelCapabilities):
            self._capabilities = capabilities
        elif capabilities is not None:
            self._capabilities = capabilities_for(capabilities)
        else:
            self._capabilities = None
        # multimodal=True is a deprecated no-op alias: media is allowed by default.
        _ = multimodal  # retained for back-compat; default capabilities already include media
        self._token_budget = token_budget
        self._checkpoint_interval = checkpoint_interval
        self._session_id = session_id
        self._user_id = user_id
        self._resume = resume
        self._response_model = response_model

        self._use_memory = use_memory
        self._memory_window = memory_window
        self._compaction_threshold = compaction_threshold
        self._input_schema = input_schema
        self._retrievers = retrievers
        self._knowledge = knowledge
        self._embedder = embedder
        self._knowledge_top_k = knowledge_top_k
        self._tool_hooks = tool_hooks
        self._require_confirmation = require_confirmation
        self._tool_timeout = tool_timeout
        self._tool_concurrency = tool_concurrency
        self._max_tool_iterations = max_tool_iterations
        self._require_final_text = require_final_text
        self._require_tools = list(require_tools) if require_tools else []

        # Tiered model routing.
        self._tiers = tiers
        self._tier_policy = tier_policy
        self._fallback_tiers = fallback_tiers

        # Low-level overrides.
        self._context_manager = context_manager
        self._memory = memory
        self._tool_runtime = tool_runtime
        self._harness = harness
        self._planner = planner
        self._session_store = session_store

        # Harness features.
        self._events = events
        self._complexity_router = complexity_router
        self._note_store = note_store
        self._loop_repeat_threshold = loop_repeat_threshold
        self._resilience = resilience
        self._verifier = verifier
        self._retry_on_failure = retry_on_failure
        self._max_verify_retries = max_verify_retries
        self._use_llm_summarizer = use_llm_summarizer
        self._think_tool = think_tool
        self._plan_tool = plan_tool
        self._memory_tool = memory_tool
        self._mode = (mode or "").strip().lower() or None
        self._dispatch = dispatch if dispatch in ("reuse", "spawn") else "reuse"
        self._accept = accept
        self._board = board
        self._max_rounds = max_rounds
        self._max_plan_steps = max_plan_steps
        self._feedback_media = feedback_media
        if accept is not None and self._verifier is None:
            self._verifier = accept

        # Developer experience.
        self._debug = debug
        self._on_tool_call = on_tool_call
        self._on_complete = on_complete

        # Cached BuiltAgent so repeated run calls reuse one runtime/session.
        self._built: BuiltAgent | None = None
        # Cached Case for mode="case" so board/state survive across arun calls.
        self._case: Any | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> BuiltAgent:
        """Validate configuration and construct a runnable :class:`BuiltAgent`.

        Constructs default kernel subsystems for anything not supplied (Req 1.2) and
        uses supplied primitives when present (Req 2.2/2.3). Raises
        :class:`AgentConfigError` naming the offending field on invalid config (Req 1.6).
        """
        # --- Validate required fields (Req 1.6) ---
        if self._model is None:
            raise AgentConfigError("model")
        if not isinstance(self._token_budget, int) or self._token_budget <= 0:
            raise AgentConfigError("token_budget")
        if not isinstance(self._checkpoint_interval, int) or self._checkpoint_interval <= 0:
            raise AgentConfigError("checkpoint_interval")

        # --- Resolve model interface + effective capabilities ---
        model_interface, provider_id, model_capabilities = self._build_model_interface()
        capabilities = self._resolve_capabilities(model_capabilities)

        # --- Build kernel config ---
        config = AgentConfig(
            model={"provider": provider_id},
            planning_model=None,
            tiers={},
            tier_policy=None,
            fallback_tiers={},
            token_budget=self._token_budget,
            checkpoint_interval=self._checkpoint_interval,
        )

        # --- Construct or reuse subsystems (Req 1.2 defaults, 2.2/2.3 overrides) ---
        context_manager = self._context_manager or ContextManager(self._token_budget)
        memory = self._memory or MemoryManager()
        tool_registry, skill_errors = self._build_tool_registry()
        # --- MCP servers: connect and enumerate tools (Req 5.1–5.4) ---
        mcp_tools, mcp_errors = self._connect_mcp_servers_sync()
        tool_registry.update(mcp_tools)
        tool_runtime = self._tool_runtime or ToolRuntime(tool_registry)
        harness = self._harness or GuardrailHarness([])
        planner = self._planner or Planner(model_interface)
        session_store = self._session_store or SessionStore()

        # Summarizer is used for compaction (memory management) in the high-level
        # run path.  The kernel AgentLoop is NOT constructed here — the high-level
        # harness (_run_single / _run_tool_loop) IS the production loop.
        summarizer: Any = Summarizer(self._checkpoint_interval)

        # --- Resolve events emitter (Req 11.1–11.5) ---
        events: AgentEvents = self._events if self._events is not None else NoOpEvents()

        # --- Resilience (Req 4.5): wrap provider in ResilientModel when configured ---
        if self._resilience is not None:
            from loomable.providers.resilient import ResilientModel

            # Wrap every registered provider impl in ResilientModel.
            for name, provider in list(model_interface._providers.items()):
                model_interface._providers[name] = ResilientModel(
                    inner=provider, policy=self._resilience, events=events
                )

        # --- LLM Summarizer (Req 5.1–5.3): when requested, use model-based compaction ---
        if self._use_llm_summarizer:
            from .summarize import LLMSummarizer

            # Use the default provider for summarization.
            default_provider = model_interface._providers.get(
                model_interface.default_provider
            )
            if default_provider is not None:
                summarizer = LLMSummarizer(default_provider)

        # --- Reasoning tools (Req 8, 9): register think/plan/memory tools ---
        if self._think_tool:
            from .reasoning import make_think_tool

            think = make_think_tool()
            tool_registry[think.name] = think
            # Rebuild tool_runtime if we're using the default (not user-supplied).
            if self._tool_runtime is None:
                tool_runtime = ToolRuntime(tool_registry)

        if self._memory_tool and self._note_store is not None:
            from .notes import make_memory_tool

            mem_tool = make_memory_tool(self._note_store)
            tool_registry[mem_tool.name] = mem_tool
            if self._tool_runtime is None:
                tool_runtime = ToolRuntime(tool_registry)

        # --- Subagent delegation tools (Proposal §2): register each subagent as
        # a tool so the parent LLM can delegate tasks at runtime ---
        if self._subagents:
            from .delegation import make_delegation_tools

            delegation_tools = make_delegation_tools(self._subagents)
            for dt in delegation_tools:
                tool_registry[dt.name] = dt
            if self._tool_runtime is None:
                tool_runtime = ToolRuntime(tool_registry)

        # --- Resolve session (create new, or resume from the session_store) ---
        # session_store is resolved above so _build_session can restore persisted
        # state from the same store instance used for save() after each run (Req 15).
        session = self._build_session(config, session_store)

        # --- Tiered model routing (Req 7): construct router if tiers configured ---
        router: ModelRouter | None = None
        if self._tiers:
            router = ModelRouter(
                model_interface=model_interface,
                tiers=self._tiers,
                tier_policy=self._tier_policy,
                fallback_tiers=self._fallback_tiers,
            )

        # --- Knowledge / RAG (Req 8.2–8.5): embed + index knowledge docs ---
        long_term: LongTermStore | None = None
        embedder_instance = self._embedder
        if self._knowledge and self._embedder is not None:
            long_term = LongTermStore()
            self._index_knowledge_sync(long_term, self._knowledge, self._embedder)

        built = BuiltAgent(
            loop=None,
            model_interface=model_interface,
            memory=memory,
            tool_runtime=tool_runtime,
            session=session,
            capabilities=capabilities,
            persist_session=self._session_id is not None,
            instructions=self._assemble_system_prompt(),
            harness=harness,
            planner=planner,
            session_store=session_store,
            use_memory=self._use_memory,
            memory_window=self._memory_window,
            compaction_threshold=self._compaction_threshold,
            context_policy=None,  # filled below from knobs
            input_schema=self._input_schema,
            tool_hooks=list(self._tool_hooks) if self._tool_hooks else [],
            require_confirmation=(
                list(self._require_confirmation) if self._require_confirmation else []
            ),
            skill_errors=skill_errors,
            mcp_errors=mcp_errors,
            summarizer=summarizer,
            router=router,
            long_term=long_term,
            embedder=embedder_instance,
            knowledge_top_k=self._knowledge_top_k,
            _token_budget=self._token_budget,
            events=events,
            complexity_router=self._complexity_router,
            note_store=self._note_store,
            loop_repeat_threshold=self._loop_repeat_threshold,
            resilience=self._resilience,
            verifier=self._verifier,
            retry_on_failure=self._retry_on_failure,
            max_verify_retries=self._max_verify_retries,
            _feedback_media=self._feedback_media,
        )

        from loomable.agent.context_policy import ContextPolicy

        built.context_policy = ContextPolicy(
            memory_window=self._memory_window,
            compaction_threshold=self._compaction_threshold,
            token_budget=self._token_budget,
        )

        # --- Wire harness knobs from Agent constructor (avoids build() boilerplate) ---
        if self._tool_timeout is not None:
            built.tool_timeout = self._tool_timeout
        if self._tool_concurrency is not None:
            built.tool_concurrency = self._tool_concurrency
        if self._max_tool_iterations is not None:
            built.max_tool_iterations = self._max_tool_iterations
        built.require_final_text = self._require_final_text
        built.require_tools = list(self._require_tools)

        # --- Wire debug mode: use a console-friendly tracer ---
        if self._debug and self._events is None:
            from .events import JSONTracer
            import sys
            built.events = JSONTracer(stream=sys.stderr)

        # --- Wire lifecycle callbacks as tool hooks ---
        if self._on_tool_call is not None:
            def _pre_hook(tool_name, call, args, _cb=self._on_tool_call):
                _cb(tool_name, args)
                return True  # allow execution
            built.tool_hooks.append(_pre_hook)

        # --- Store metadata on the built agent ---
        built.name = self._name or None  # type: ignore[attr-defined]
        built.description = self._description or None  # type: ignore[attr-defined]
        built.role = self._role or None  # type: ignore[attr-defined]
        built.goal = self._goal or None  # type: ignore[attr-defined]
        built.user_id = self._user_id or None  # type: ignore[attr-defined]

        # --- Plan tool (Req 9): must be registered post-build because it references
        # the BuiltAgent itself ---
        if self._plan_tool:
            from .reasoning import make_plan_tool

            plan = make_plan_tool(built)
            built.tool_runtime._tools[plan.name] = plan

        return built

    # ------------------------------------------------------------------
    # Run flow (high-level wrappers delegating to BuiltAgent)
    # ------------------------------------------------------------------

    def _get_built(self) -> BuiltAgent:
        """Build the agent once and cache it for subsequent runs."""
        if self._built is None:
            self._built = self.build()
        return self._built

    async def arun(
        self,
        input: AgentInput | str,  # noqa: A002
        *,
        images: "list[str | Path | Any] | None" = None,
        videos: "list[str | Path | Any] | None" = None,
        audio: "list[str | Path | Any] | None" = None,
        output_schema: type | None = None,
        context: dict[str, Any] | None = None,
    ) -> RunResult:
        """Build (once) and run the agent, returning a :class:`RunResult` (Req 1.4).

        Parameters
        ----------
        input:
            The user input (string or AgentInput).
        images:
            Optional list of images (file paths or MediaPart instances).
        videos:
            Optional list of videos (file paths or MediaPart instances).
        audio:
            Optional list of audio files (file paths or MediaPart instances).
        output_schema:
            Optional per-call structured output schema (overrides response_model).
        context:
            Optional runtime context dict accessible during the run.
        """
        # Case mode: plan → dispatch → synthesize → accept.
        if self._mode == "case":
            case = self._get_case()
            text = self._coerce_run_text(input)
            result = await case.arun(text)
            if self._on_complete is not None:
                self._on_complete(result)
            return result

        built = self._get_built()
        # Use response_model as default output_schema when not overridden per-call
        schema = output_schema or self._response_model
        result = await built.arun(input, images=images, videos=videos, audio=audio, output_schema=schema)
        # Lifecycle callback: on_complete
        if self._on_complete is not None:
            self._on_complete(result)
        return result

    def run(
        self,
        input: AgentInput | str,  # noqa: A002
        *,
        images: "list[str | Path | Any] | None" = None,
        videos: "list[str | Path | Any] | None" = None,
        audio: "list[str | Path | Any] | None" = None,
        output_schema: type | None = None,
        context: dict[str, Any] | None = None,
    ) -> RunResult:
        """Synchronous wrapper around :meth:`arun` (Req 1.4).

        Uses :func:`asyncio.run`, which requires that no event loop is already
        running on the calling thread. In an async context, call :meth:`arun`.
        """
        return asyncio.run(self.arun(input, images=images, videos=videos, audio=audio, output_schema=output_schema, context=context))

    async def astream(
        self,
        input: AgentInput | str,  # noqa: A002
        *,
        output_schema: type | None = None,
    ) -> "AsyncIterator[RunChunk]":
        """Build (once) and stream incremental output as :class:`RunChunk`s (Req 1.5)."""
        built = self._get_built()
        async for chunk in built.astream(input, output_schema=output_schema):
            yield chunk

    async def astream_events(
        self,
        input: AgentInput | str,  # noqa: A002
        *,
        images: "list[str | Path | Any] | None" = None,
        videos: "list[str | Path | Any] | None" = None,
        audio: "list[str | Path | Any] | None" = None,
        output_schema: type | None = None,
        context: dict[str, Any] | None = None,
    ) -> "AsyncIterator[Any]":
        """Yield AG-UI-compatible stream events (lifecycle, tools, text).

        When ``mode="case"``, streams Case pipeline events.
        """
        if self._mode == "case":
            case = self._get_case()
            text = self._coerce_run_text(input)
            async for event in case.astream_events(text):
                yield event
            return

        built = self._get_built()
        schema = output_schema or self._response_model
        async for event in built.astream_events(
            input,
            images=images,
            videos=videos,
            audio=audio,
            output_schema=schema,
        ):
            yield event

    def _coerce_run_text(self, value: Any) -> str:
        """Normalize string / AgentInput into plain text for Case pipelines."""
        if isinstance(value, str):
            return value
        if isinstance(value, AgentInput):
            return _input_text(value)
        if hasattr(value, "messages"):
            try:
                return _input_text(value)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                pass
        return str(value)

    def _get_case(self) -> Any:
        """Lazily build and cache a Case so board state survives across calls."""
        if self._case is None:
            from loomable.case import Case

            self._case = Case.from_agent(self)
        return self._case

    # ------------------------------------------------------------------
    # System prompt assembly
    # ------------------------------------------------------------------

    def _assemble_system_prompt(self) -> str | None:
        """Assemble role + goal + instructions into the system prompt.

        Produces a prompt like:
            You are a Senior Security Reviewer.
            Your goal: Identify vulnerabilities and suggest fixes.

            Focus on OWASP Top 10. Be specific about affected code.

        Returns None when none of role/goal/instructions is set.
        """
        parts: list[str] = []
        if self._role:
            parts.append(f"You are a {self._role}.")
        if self._goal:
            parts.append(f"Your goal: {self._goal}")
        if self._instructions:
            if parts:
                parts.append("")  # blank line separator
            parts.append(self._instructions)
        if not parts:
            return None
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_model_interface(
        self,
    ) -> tuple[ModelInterface, str, ModelCapabilities | None]:
        """Build a :class:`ModelInterface` from the configured model.

        Returns the interface, the provider id, and any capabilities declared by a
        :class:`ModelSpec` (``None`` for a bare provider).

        When ``tiers`` are configured, each tier's value is expected to be a
        ``ModelProvider`` instance; the tier providers are registered in the
        ``ModelInterface`` under their tier names so that the kernel ``ModelRouter``
        can route to them via ``model_interface.invoke(request, tier=tier_name)``.
        """
        if isinstance(self._model, ModelSpec):
            provider_id = self._model.provider
            if not provider_id:
                raise AgentConfigError("model")
            providers: dict[str, ModelProvider] = {}
            if self._model.provider_impl is not None:
                providers[provider_id] = self._model.provider_impl
            # Register tier providers (Req 7.1/7.5).
            if self._tiers:
                for tier_name, tier_value in self._tiers.items():
                    if hasattr(tier_value, "complete"):
                        providers[tier_name] = tier_value
            interface = ModelInterface(providers=providers, default_provider=provider_id)
            return interface, provider_id, self._model.capabilities

        # Bare ModelProvider: register under the default provider id.
        provider_id = _DEFAULT_PROVIDER_ID
        providers = {provider_id: self._model}
        # Register tier providers (Req 7.1/7.5).
        if self._tiers:
            for tier_name, tier_value in self._tiers.items():
                if hasattr(tier_value, "complete"):
                    providers[tier_name] = tier_value
        interface = ModelInterface(
            providers=providers,
            default_provider=provider_id,
        )
        return interface, provider_id, None

    def _resolve_capabilities(
        self, spec_capabilities: ModelCapabilities | None
    ) -> ModelCapabilities:
        """Resolve effective capabilities: explicit arg, else spec, else multimodal default."""
        if self._capabilities is not None:
            return self._capabilities
        if spec_capabilities is not None:
            return spec_capabilities
        return ModelCapabilities()

    def _build_tool_registry(self) -> tuple[dict[str, Tool], list[SkillLoadError]]:
        """Build the name→Tool mapping for the default ToolRuntime.

        Explicit ``tools`` are registered by their ``name`` (Req 1.3). Configured
        ``skills`` are discovered and loaded through the kernel :class:`SkillLoader`;
        each loaded Skill's script tools are registered by name (Req 4.1/4.2). A Skill
        that fails to load is isolated: its :class:`SkillLoadError` is captured and
        collected while other Skills continue loading (Req 4.3). No kernel code is
        modified (Req 4.4).

        Attached ``retrievers`` are then wrapped with the kernel :class:`RetrieverTool`
        adapter and registered under each retriever's ``name`` so retrieval is invocable
        as a normal tool — Agentic RAG at the edge (Req 16.1/16.2/16.3). This reuses
        the kernel retriever-as-tool mechanism unchanged (Req 16.4).

        Name collisions are surfaced eagerly: if a retriever's name matches an explicit
        tool's name (or another retriever's name), :class:`AgentConfigError` is raised
        naming the conflicting ``retrievers`` entry rather than silently shadowing a
        tool. This keeps configuration errors visible at build time.

        Note: retrievers are wired only into the *default* ``ToolRuntime`` built here.
        When a caller supplies their own ``tool_runtime`` override, that runtime is used
        verbatim and attached retrievers are not auto-registered into it — the caller is
        responsible for exposing retrieval through their custom runtime.

        Returns
        -------
        tuple[dict[str, Tool], list[SkillLoadError]]
            The tool registry and any skill load errors that were isolated.
        """
        registry: dict[str, Tool] = {}
        skill_errors: list[SkillLoadError] = []

        if self._tools:
            from loomable.toolkits._base import Toolkit
            for item in self._tools:
                if isinstance(item, Toolkit):
                    for ft in item.tools():
                        registry[ft.name] = ft
                else:
                    registry[item.name] = item

        # --- Skills: discover + load via the kernel SkillLoader (Req 4.1–4.4) ---
        if self._skills:
            loader = SkillLoader()
            manifests = loader.discover(self._skills)
            for manifest in manifests:
                try:
                    loaded_skill = loader.load(manifest)
                    for script_tool in loaded_skill.get_tools():
                        registry[script_tool.name] = script_tool
                except SkillLoadError as err:
                    skill_errors.append(err)

        if self._retrievers:
            for retriever in self._retrievers:
                if retriever.name in registry:
                    raise AgentConfigError(
                        f"retrievers[{retriever.name!r}] "
                        f"(name collides with an existing tool)"
                    )
                registry[retriever.name] = RetrieverTool(retriever)

        return registry, skill_errors

    def _connect_mcp_servers_sync(
        self,
    ) -> tuple[dict[str, Tool], list[MCPConnectionError]]:
        """Connect to configured MCP servers synchronously (Req 5.1–5.4).

        Wraps the async :meth:`_connect_mcp_servers` for use from the synchronous
        :meth:`build`. Returns the MCP tool registry and any connection errors.
        """
        if not self._mcp_servers:
            return {}, []
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already inside an event loop — create a new loop in a thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(asyncio.run, self._connect_mcp_servers())
                return future.result()
        return asyncio.run(self._connect_mcp_servers())

    async def _connect_mcp_servers(
        self,
    ) -> tuple[dict[str, Tool], list[MCPConnectionError]]:
        """Connect to configured MCP servers and enumerate their tools (Req 5.1–5.4).

        For each MCP server specification in ``self._mcp_servers``, the kernel
        :class:`~loomable.kernel.mcp_client.MCPClient` is used to connect and
        enumerate tools. Each discovered tool is wrapped as an :class:`MCPTool`
        (a kernel :class:`Tool`) whose ``invoke`` delegates to
        ``MCPClient.call_tool``.

        A failed connection yields an :class:`MCPConnectionError` for that server
        while other servers continue (Req 5.3). No kernel code is modified (Req 5.4).

        Returns
        -------
        tuple[dict[str, Tool], list[MCPConnectionError]]
            The MCP tool registry and any connection errors that were isolated.
        """
        from .tools import MCPTool

        registry: dict[str, Tool] = {}
        errors: list[MCPConnectionError] = []
        client = MCPClient()

        for spec in self._mcp_servers:
            try:
                session = await client.connect(spec)
                capabilities = await client.list_capabilities(session)
                for tool_info in capabilities.tools:
                    tool_name = tool_info.get("name", "")
                    if not tool_name:
                        continue
                    description = tool_info.get("description", "")
                    parameters = tool_info.get("parameters", {
                        "type": "object",
                        "properties": {},
                    })
                    mcp_tool = MCPTool(
                        name=tool_name,
                        description=description,
                        parameters=parameters,
                        mcp_client=client,
                        session=session,
                    )
                    registry[tool_name] = mcp_tool
            except MCPConnectionError as err:
                errors.append(err)

        return registry, errors

    def _build_session(
        self, config: AgentConfig, session_store: SessionStore
    ) -> Session:
        """Create a new session, or resume a persisted one (Req 15.1/15.3/15.4).

        When ``resume=True`` and a ``session_id`` is supplied, the session is restored
        from ``session_store`` so prior turns are available (Req 15.3); an unknown id
        propagates :class:`~loomable.kernel.errors.SessionNotFoundError` naming the id
        (Req 15.4). Otherwise a fresh :class:`Session` is created, using the supplied
        id when present (Req 15.1).
        """
        if self._resume and self._session_id is not None:
            # Propagate SessionNotFoundError for unknown ids (Req 15.4).
            return session_store.resume(self._session_id)

        session_id = self._session_id or f"session-{uuid.uuid4().hex}"
        return Session(
            session_id=session_id,
            agent_config_ref=config.model.get("provider", "default"),
        )

    # ------------------------------------------------------------------
    # Knowledge indexing helpers (Req 8.2)
    # ------------------------------------------------------------------

    def _index_knowledge_sync(
        self,
        long_term: LongTermStore,
        knowledge: list[str],
        embedder: Any,
    ) -> None:
        """Embed and index each knowledge document synchronously (for use from build).

        Wraps the async embedding/indexing coroutine so it can be called from the
        synchronous :meth:`build` method regardless of whether an event loop is
        already running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(
                    asyncio.run, self._index_knowledge(long_term, knowledge, embedder)
                )
                future.result()
        else:
            asyncio.run(self._index_knowledge(long_term, knowledge, embedder))

    async def _index_knowledge(
        self,
        long_term: LongTermStore,
        knowledge: list[str],
        embedder: Any,
    ) -> None:
        """Embed and index each knowledge document into the LongTermStore (Req 8.2).

        Each document is embedded via the configured :class:`Embedder` and indexed
        in the kernel :class:`LongTermStore` with the document text as metadata.
        """
        for idx, doc in enumerate(knowledge):
            vector = await embedder.embed(doc)
            await long_term.index(
                id=f"knowledge-{idx}",
                vector=vector,
                metadata={"text": doc},
            )
