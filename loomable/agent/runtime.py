"""Production single-agent runtime for model calls, tools, and finalization."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import dataclasses
import json
import logging
import time
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
    Text,
    from_model_response,
    to_agent_input,
    to_model_request,
)
from loomable.kernel.context import ContextManager
from loomable.kernel.errors import GuardrailViolation, MCPConnectionError, SkillLoadError
from loomable.kernel.guardrails import GuardrailHarness
from loomable.kernel.long_term import LongTermStore
from loomable.kernel.memory import MemoryManager
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.model_router import ModelRouter, TierSubstitution
from loomable.kernel.models import (
    ContextItem,
    ModelRequest,
    ModelResponse,
    Session,
    ToolCall,
    ToolError,
    ToolOutcome,
    Turn,
)
from loomable.kernel.planner import Planner
from loomable.kernel.stores import SessionStore
from loomable.kernel.summarizer import Summarizer
from loomable.kernel.tool_runtime import ToolRuntime

from .errors import (
    InputValidationError,
    StructuredOutputError,
    ToolHookRejection,
    UnsupportedModalityError,
)
from .context import RunContext, StopReason, _signature as _ctx_signature
from .events import AgentEvents, Event, NoOpEvents
from .run import RunResult

if TYPE_CHECKING:  # pragma: no cover - typing only

    from loomable.kernel.agent_loop import AgentLoop
    from loomable.providers.resilient import RetryPolicy
    from loomable.memory import NoteStore
    from .routing import ComplexityRouter


from ._support import (
    ToolHook,
    PostToolHook,
    Approver,
    _HOOK_REJECTION_RULE_ID,
    _CONFIRMATION_RULE_ID,
    _deny_all_approver,
    GatedDispatchResult,
    _input_text,
    _coerce_media_item,
    _synthesize_tool_activity_summary,
    _schema_instruction,
    _path_constraint_met,
    _parse_require_tool_specs,
    _missing_require_tool_specs,
    _format_require_tools_nudge,
    _DELIVERABLE_COMPLETE_NUDGE,
    _is_bookkeeping_only,
    _extract_json_object,
    _validate_structured,
    _finalize_run_result,
)

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
    on_complete: Any = field(default=None, kw_only=True, repr=False)
    durability: Any = field(default=None, kw_only=True)
    _journal: Any = field(default=None, init=False, repr=False)
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
    # run (structured dict/model input). ``None`` disables input validation.
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
    # When True, raise RequireToolsError if required tools are still missing
    # after nudges (fail-closed). Default is best-effort (metadata only).
    strict_require_tools: bool = False

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

    # Composable Memory(auto_extract=...) — heuristic fact write after each turn.
    memory_auto_extract: bool = False
    # Inject recalled / cached user facts into the prompt prefix.
    memory_auto_recall: bool = True
    _cached_user_facts: list[str] = field(default_factory=list)
    _memory_user_id: str | None = None
    _memory_scope: Any | None = None

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
    # Cumulative spend stop for the run (0 = unbounded; None = fall back to token_budget).
    _max_run_tokens: int | None = None

    # Skill load errors: isolated failures from skill loading (Req 4.3). Each
    # entry identifies a Skill that failed to load while others succeeded.
    skill_errors: list["SkillLoadError"] = field(default_factory=list)

    # MCP connection errors: isolated failures from MCP server connections (Req 5.3).
    # Each entry identifies a server that failed to connect while others succeeded.
    mcp_errors: list["MCPConnectionError"] = field(default_factory=list)

    # Runtime capability discovery (search_skills / load_skill / search_tools /
    # search_mcp / activate_tool). None when discovery is disabled.
    discovery: Any | None = None

    # Pinned facts (Req 6.1–6.4): steps whose turns are never eligible for compaction
    # and are always replayed in the memory prefix regardless of the rolling window.
    pinned_steps: set[int] = field(default_factory=set)

    # Multimodal feedback (Req 7.5): when True, tool-generated media is injected
    # into the conversation so the model can reason about it in subsequent turns.
    _feedback_media: bool = True

    # Active RunContext for cooperative cancel (BuiltAgent.cancel / SSE disconnect).
    _active_ctx: Any | None = field(default=None, init=False, repr=False)

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
        _resume: tuple | None = None,
        _once: bool = False,
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
        if self._active_ctx is not None:
            if _once:
                from loomable.persist.checkpoint import CheckpointBusy
                raise CheckpointBusy("Checkpoint thread already has an owner")
            raise RuntimeError("Agent runtime is already running; use a separate session_id.")

        # Early capability gating for audio (Req 4.4): raise before coercion/model call.
        if audio and Modality.AUDIO not in self.capabilities.input:
            raise UnsupportedModalityError("audio", self._model_id)


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
                max_run_tokens=self._max_run_tokens,
                loop_repeat_threshold=self.loop_repeat_threshold,
            )

        self._active_ctx = ctx
        ctx._usage.clear()
        try:
            async with AsyncExitStack() as resources:
                if self.durability is not None:
                    if images or videos or audio:
                        raise ValueError("Durable agents currently accept text only")
                    from .durable import ExecutionJournal, session_key
                    store = self.durability.checkpointer
                    key = session_key(self.session.session_id)
                    await resources.enter_async_context(store.claim(key))
                    checkpoint = await store.get(key)
                    if _once and checkpoint is not None:
                        saved = checkpoint.session_state
                        if saved["input"] != input:
                            raise ValueError("Durable session already has a run with different input")
                        if checkpoint.complete:
                            return self._recorded_result(saved, output_schema)
                        ctx.run_id = saved["run_id"]
                        _resume = (ctx.run_id, None, None)
                    if _resume is None:
                        self._journal = await ExecutionJournal.begin(self, input, ctx, checkpoint, once=_once)
                    else:
                        self._journal = await ExecutionJournal.load(self, checkpoint, *_resume)
                        input = self._journal.state["input"]
                agent_input = self._coerce_input(input, images=images, videos=videos, audio=audio)
                if self.note_store is not None and self.memory_auto_recall:
                    await self._refresh_user_memory_context(_input_text(agent_input))
                return await self._arun_with_context(agent_input, output_schema=output_schema, ctx=ctx)
        finally:
            self._journal = None
            if self._active_ctx is ctx:
                self._active_ctx = None

    async def acontinue_run(self, run_id, *, approvals=None, outcomes=None, output_schema=None):
        if self.durability is None:
            raise ValueError("acontinue_run requires durability=DurabilityOptions(...)")
        ctx = RunContext(events=self.events, max_steps=self.max_tool_iterations,
                         token_budget=self._token_budget, max_run_tokens=self._max_run_tokens,
                         loop_repeat_threshold=self.loop_repeat_threshold, run_id=run_id)
        return await self.arun("", context=ctx, output_schema=output_schema,
                               _resume=(run_id, approvals, outcomes))

    async def arun_once(self, input: str, *, output_schema: type | None = None) -> RunResult:
        """Start or recover a durable text run keyed by its dedicated session ID."""
        if self.durability is None or not isinstance(input, str):
            raise ValueError("arun_once requires durability and plain text input")
        if self.capabilities.output != {Modality.TEXT}:
            raise ValueError("arun_once requires a text-output model")
        return await self.arun(input, output_schema=output_schema, _once=True)

    def _recorded_result(self, saved, output_schema):
        response = saved.get("response")
        if response is None:
            # Older journals stored the output only as an assistant turn.
            turns = saved["final_session"]["l1"]
            if not turns or turns[-1]["role"] != "assistant":
                raise ValueError("Completed checkpoint has no recorded response")
            response = {"text": turns[-1]["content"], "usage": {}}
        content = response["text"]
        return RunResult(
            output=AgentOutput(parts=[Text(content)]),
            session_id=saved["session_id"],
            usage=dict(response["usage"]),
            structured=_validate_structured(content, output_schema) if output_schema else None,
            metadata={"run_id": saved["run_id"], "replayed": True},
        )

    def cancel(self) -> bool:
        """Request cooperative cancellation of the in-flight run.

        Returns True if an active :class:`RunContext` was marked cancelled.
        Cancellation is observed at tool-loop / step boundaries (not a hard
        abort of an in-flight provider HTTP call).
        """
        ctx = self._active_ctx
        if ctx is None:
            return False
        ctx.cancel()
        return True

    async def _arun_with_context(
        self,
        agent_input: AgentInput,
        *,
        output_schema: type | None,
        ctx: RunContext,
    ) -> RunResult:
        # --- Emit run_start event (Req 11.1) ---
        ctx.events.emit(Event(
            kind="run_start",
            t=time.monotonic(),
            attributes={"gen_ai.operation.name": "chat", "run_id": ctx.run_id},
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

        # --- Output verification (Req 4.2–4.4) ---
        # When a verifier is configured, evaluate it against the final output and
        # record the verdict on the RunResult. When retry_on_failure is enabled and
        # the verifier reports failure, re-run with the failure detail appended to
        # context up to max_verify_retries times.
        if self.verifier is not None:
            from loomable.flow.loop import CallableVerifier, Verifier

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
        if ctx._usage:
            result.usage = dict(ctx._usage)
        if self._journal is not None:
            await self._journal.finish(self.session, response={
                "text": result.output.text(), "usage": dict(result.usage),
            })
        result.metadata["run_id"] = ctx.run_id
        ctx.events.emit(Event(kind="run_end", t=time.monotonic(),
                              duration_ms=ctx.elapsed() * 1000,
                              attributes={"gen_ai.operation.name": "chat"}))
        if hasattr(ctx.events, "trace"):
            result.trace = list(ctx.events.trace)
        if self.on_complete is not None:
            self.on_complete(result)
        return result

    def _coerce_input(
        self,
        value: Any,
        *,
        images: list | None = None,
        videos: list | None = None,
        audio: list | None = None,
    ) -> AgentInput:
        """Normalize any supported input into an :class:`AgentInput`.

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

        # User-scoped long-term facts (from Memory.compose / NoteStore cache).
        if self.memory_auto_recall and self._cached_user_facts:
            blob = "\n".join(f"- {f}" for f in self._cached_user_facts[:12])
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Known facts about this user:\n{blob}",
                        }
                    ],
                }
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
        ``loomable.flow.helpers``. When ``self.planner`` is set, planning uses the
        kernel :class:`~loomable.kernel.planner.Planner`; otherwise a JSON plan prompt.
        Workers use the tool loop when tools are registered so session/tools/knowledge
        remain available during each step.
        """
        from loomable.flow.helpers import plan_and_execute

        task_text = _input_text(agent_input)

        async def _planner(input: Any, **kwargs: Any) -> dict:
            """Produce plan steps — kernel Planner when set, else JSON prompt."""
            if self.planner is not None:
                from loomable.kernel.planner import TaskContext

                plan = await self.planner.plan(TaskContext(task=task_text))
                steps = [str(s).strip() for s in plan.steps if str(s).strip()][:5]
                if not steps:
                    steps = [task_text]
                return {"plan_steps": steps}

            from loomable.plan_parse import parse_plan_steps

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
            steps = parse_plan_steps(result.output.text(), max_steps=5)
            return {"plan_steps": steps}

        plan_tool_activity: list[Any] = []
        plan_sub_results: dict[str, Any] = {}
        plan_step_index = {"n": 0}

        async def _worker(input: Any, **kwargs: Any) -> str:
            """Run a single plan step with the agent's full tool loop when tools exist."""
            step = input if isinstance(input, str) else str(input)
            prompt = (
                f"Overall task:\n{task_text}\n\n"
                f"Complete ONLY this step, concisely and concretely:\n{step}"
            )
            step_input = AgentInput.from_text(prompt)
            if self.tool_runtime._tools:
                result = await self._run_tool_loop(
                    step_input,
                    include_history=False,
                    ctx=ctx,
                    exclude_tools=frozenset({"plan"}),
                )
            else:
                result = await self._run_single(
                    step_input, include_history=False, ctx=ctx
                )
            if result.tool_activity:
                plan_tool_activity.extend(result.tool_activity)
            idx = plan_step_index["n"]
            plan_step_index["n"] = idx + 1
            plan_sub_results[f"step_{idx}"] = result
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
                tool_activity=plan_tool_activity,
                sub_results=plan_sub_results or None,
                structured=structured,
            )
        )

    async def _emit_astream_delta(self, text: str) -> None:
        """Forward provider text deltas to an active :meth:`astream` consumer."""
        if not text:
            return
        queue = getattr(self._active_ctx, "_delta_queue", None)
        if queue is not None:
            await queue.put(text)

    def _record_usage(self, response):
        ctx = self._active_ctx
        if ctx is not None:
            for key, value in (response.usage or {}).items():
                if isinstance(value, int):
                    ctx._usage[key] = ctx._usage.get(key, 0) + value

    async def _invoke_model_request(self, request):
        if self._journal is None:
            response, substitution = await self._invoke_model_request_uncached(request)
        else:
            async def invoke():
                response, _ = await self._invoke_model_request_uncached(request)
                return response
            response = await self._journal.model(request, invoke)
            substitution = None
        self._record_usage(response)
        return response, substitution

    async def _invoke_model_request_uncached(
        self,
        request: "ModelRequest",
    ) -> tuple["ModelResponse", "TierSubstitution | None"]:
        """Invoke the model, streaming token deltas when ``astream`` is active."""
        tier_substitution: TierSubstitution | None = None
        if self.router is not None:
            response, tier_substitution = await self.router.route(request)
            return response, tier_substitution

        provider = self.model_interface._providers.get(
            self.model_interface.default_provider
        )
        queue = getattr(self._active_ctx, "_delta_queue", None)
        # Tool-advertised requests must use complete() so providers whose stream()
        # omits tool_call events cannot bypass the tool loop (honesty guard).
        if (
            queue is not None
            and provider is not None
            and hasattr(provider, "stream")
            and not self.retry_on_failure
            and not request.tools
        ):
            accumulated = ""
            tool_calls: list[ToolCall] = []
            usage: dict[str, int] = {}
            source = provider.stream(request)
            try:
                async for event in source:
                    if event.kind == "text" and event.text:
                        accumulated += event.text
                        await self._emit_astream_delta(event.text)
                    elif event.kind == "tool_call" and event.tool_call is not None:
                        tool_calls.append(event.tool_call)
                    elif event.kind == "end":
                        usage.update(event.usage or {})
            finally:
                close = getattr(source, "aclose", None)
                if close is not None:
                    await close()
            response = ModelResponse(content=accumulated, tool_calls=tool_calls, usage=usage)
            return response, None

        return await self.model_interface.invoke(request), None

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
            if hasattr(output_schema, "model_json_schema"):
                request.metadata["output_schema"] = {
                    "name": output_schema.__name__,
                    "schema": output_schema.model_json_schema(),
                }
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
        response, tier_substitution = await self._invoke_model_request(request)
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
        exclude_tools: frozenset[str] | None = None,
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

        def _collect_tool_schemas() -> list[dict]:
            schemas: list[dict] = []
            for name, tool_obj in self.tool_runtime._tools.items():
                if exclude_tools and name in exclude_tools:
                    continue
                if isinstance(tool_obj, (FunctionTool, MCPTool)):
                    schemas.append(tool_obj.schema())
                else:
                    schemas.append({
                        "type": "function",
                        "function": {
                            "name": tool_obj.name,
                            "description": getattr(tool_obj, "description", ""),
                            "parameters": getattr(
                                tool_obj,
                                "parameters",
                                {"type": "object", "properties": {}},
                            ),
                        },
                    })
            return schemas

        # (2) Build tool schemas to advertise to the model (refreshed each iteration
        # so activate_tool / load_skill can extend the surface mid-run).
        tool_schemas = _collect_tool_schemas()
        tools_enabled = True

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
            if hasattr(output_schema, "model_json_schema"):
                request.metadata["output_schema"] = {
                    "name": output_schema.__name__,
                    "schema": output_schema.model_json_schema(),
                }
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
        bookkeeping_after_deliverable = 0
        deliverable_complete_nudges = 0
        deliverable_complete_forced = False

        while True:
            # Refresh advertised tools each turn (discovery activate/load).
            if tools_enabled:
                request.tools = _collect_tool_schemas()

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
            response, tier_sub = await self._invoke_model_request(request)
            if tier_sub is not None:
                tier_substitutions.append(tier_sub)
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
                if missing_required:
                    discovery_rt = getattr(self, "discovery", None)
                    if discovery_rt is not None and hasattr(
                        discovery_rt, "ensure_tools_activated"
                    ):
                        missing_tool_names = sorted(
                            {m.split(":", 1)[0] for m in missing_required if m}
                        )
                        try:
                            discovery_rt.ensure_tools_activated(missing_tool_names)
                        except Exception as exc:  # noqa: BLE001 — activation is best-effort
                            logging.getLogger("loomable.agent").warning(
                                "discovery ensure_tools_activated failed for %s: %s",
                                missing_tool_names,
                                exc,
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
                tools_enabled = False
                # Context bounding before the nudge call (Req 13.1–13.4).
                if effective_budget is not None:
                    request.messages = self._bound_messages(request.messages, effective_budget)
                _nudge_t0 = time.monotonic()
                response, tier_sub = await self._invoke_model_request(request)
                if tier_sub is not None:
                    tier_substitutions.append(tier_sub)
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

            # Emit AG-UI tool frames around the actual dispatch (START/ARGS before,
            # RESULT/END after) when the event sink supports publish().
            _agui = getattr(ctx.events, "publish", None)
            if callable(_agui):
                from loomable.stream import (
                    TOOL_CALL_ARGS,
                    TOOL_CALL_END,
                    TOOL_CALL_RESULT,
                    TOOL_CALL_START,
                )

                for tc in calls_to_dispatch:
                    _agui(TOOL_CALL_START, {"tool_call_id": tc.id, "tool_name": tc.tool_name})
                    _agui(
                        TOOL_CALL_ARGS,
                        {
                            "tool_call_id": tc.id,
                            "tool_name": tc.tool_name,
                            "args": dict(tc.args or {}),
                        },
                    )

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
            outcome_by_id = {o.call_id: o for o in gated_result.outcomes}
            for tc in calls_to_dispatch:
                called_tool_names.add(tc.tool_name)
                path_arg = str(tc.args.get("path") or "")
                outcome = outcome_by_id.get(tc.id)
                ok = (
                    outcome is not None
                    and outcome.result is not None
                    and not outcome.result.is_error
                    and not str(outcome.result.content or "").startswith("Error:")
                )
                for name, path_sub in require_tool_specs:
                    if tc.tool_name != name:
                        continue
                    if not ok:
                        continue
                    if path_sub is None or _path_constraint_met(path_arg, path_sub):
                        satisfied_require_tools.add((name, path_sub))
                if tc.tool_name == "write_json":
                    raw = tc.args.get("content")
                    if ok and raw is not None:
                        if isinstance(raw, str):
                            write_json_payloads.append(raw)
                        else:
                            write_json_payloads.append(json.dumps(raw))

            if callable(_agui):
                from loomable.stream import TOOL_CALL_END, TOOL_CALL_RESULT

                outcome_by_id = {o.call_id: o for o in gated_result.outcomes}
                for tc in calls_to_dispatch:
                    outcome = outcome_by_id.get(tc.id)
                    content = ""
                    is_error = False
                    if outcome is not None and outcome.result is not None:
                        is_error = bool(outcome.result.is_error)
                        if is_error:
                            content = outcome.result.error or ""
                        else:
                            content = (
                                str(outcome.result.content)
                                if outcome.result.content is not None
                                else ""
                            )
                    _agui(
                        TOOL_CALL_RESULT,
                        {
                            "tool_call_id": tc.id,
                            "tool_name": tc.tool_name,
                            "content": content[:4000],
                            "is_error": is_error,
                        },
                    )
                    _agui(
                        TOOL_CALL_END,
                        {
                            "tool_call_id": tc.id,
                            "tool_name": tc.tool_name,
                            "duration_ms": _tool_dispatch_duration,
                        },
                    )

            # Emit tool_call event (Req 11.3) for observability; AG-UI already published.
            tool_names = [tc.tool_name for tc in calls_to_dispatch]
            ctx.events.emit(Event(
                kind="tool_call",
                t=time.monotonic(),
                duration_ms=_tool_dispatch_duration,
                attributes={
                    "gen_ai.tool.name": ",".join(tool_names) if tool_names else "",
                    "tool_count": len(calls_to_dispatch),
                    "agui_skip": bool(callable(_agui)),
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
                "content": (
                    [{"type": "text", "text": str(response.content)}]
                    if (getattr(response, "content", None) or "").strip()
                    else []
                ),
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
                    content = outcome.result.content
                    if outcome.result.is_error:
                        result_content = outcome.result.error or ""
                    elif isinstance(content, (dict, list, tuple)):
                        result_content = json.dumps(content, ensure_ascii=False, default=str)
                    else:
                        result_content = str(content) if content is not None else ""
                else:
                    result_content = outcome.error.message if outcome.error else "unknown error"

                request.messages.append({
                    "role": "tool",
                    "content": [{"type": "text", "text": result_content}],
                    "tool_call_id": outcome.call_id,
                })

            # Mid-run skill loads: inject skill bodies as system messages so later
            # turns follow the skill without relying only on the tool result text.
            discovery_rt = getattr(self, "discovery", None)
            if discovery_rt is not None and hasattr(discovery_rt, "drain_prompt_injections"):
                for skill_name, body in discovery_rt.drain_prompt_injections():
                    if not (body or "").strip():
                        continue
                    request.messages.append(
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"## Skill: {skill_name}\n{body.strip()}",
                                }
                            ],
                        }
                    )

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

            # Deliverable-complete finish: once require_tools succeed, do not burn
            # the iteration budget on todo/think spam — nudge once, then force FINAL.
            if require_tool_specs and not _missing_require_tool_specs(
                require_tool_specs, satisfied_require_tools
            ):
                round_names = {tc.tool_name for tc in calls_to_dispatch}
                if _is_bookkeeping_only(round_names):
                    bookkeeping_after_deliverable += 1
                else:
                    bookkeeping_after_deliverable = 0

                if (
                    bookkeeping_after_deliverable >= 1
                    and deliverable_complete_nudges < 1
                    and iterations < self.max_tool_iterations
                    and not ctx.cancelled
                ):
                    deliverable_complete_nudges += 1
                    request.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": _DELIVERABLE_COMPLETE_NUDGE,
                                }
                            ],
                        }
                    )
                elif (
                    bookkeeping_after_deliverable >= 2
                    and iterations < self.max_tool_iterations
                    and not ctx.cancelled
                ):
                    deliverable_complete_forced = True
                    stop_reason = StopReason(
                        kind=StopReason.FINAL,
                        detail="deliverable_complete",
                    )
                    request.messages.append(
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Required deliverables are complete. "
                                        "Provide your final answer now. "
                                        "Do not request any tool calls."
                                    ),
                                }
                            ],
                        }
                    )
                    request.tools = []
                    tools_enabled = False
                    if effective_budget is not None:
                        request.messages = self._bound_messages(
                            request.messages, effective_budget
                        )
                    _done_t0 = time.monotonic()
                    if self.router is not None:
                        response, tier_sub = await self.router.route(request)
                        if tier_sub is not None:
                            tier_substitutions.append(tier_sub)
                    else:
                        response = await self.model_interface.invoke(request)
                    _done_duration = (time.monotonic() - _done_t0) * 1000
                    done_usage = (
                        response.usage
                        if hasattr(response, "usage") and response.usage
                        else {}
                    )
                    done_tokens = done_usage.get("input_tokens", 0) + done_usage.get(
                        "output_tokens", 0
                    )
                    if done_tokens:
                        ctx.add_tokens(done_tokens)
                    ctx.events.emit(
                        Event(
                            kind="model_call",
                            t=time.monotonic(),
                            duration_ms=_done_duration,
                            tokens_in=done_usage.get("input_tokens", 0),
                            tokens_out=done_usage.get("output_tokens", 0),
                            attributes={
                                "gen_ai.request.model": self._model_id,
                                "gen_ai.usage.input_tokens": done_usage.get(
                                    "input_tokens", 0
                                ),
                                "gen_ai.usage.output_tokens": done_usage.get(
                                    "output_tokens", 0
                                ),
                                "loomable.deliverable_complete": True,
                            },
                        )
                    )
                    break

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
            try:
                response = await self.model_interface.invoke(request)
                output = from_model_response(response)
            except Exception:  # noqa: BLE001 — provider may reject tool-free nudge
                pass
            final_text_reprompted = True

        # If the model still produced no text after tools (common with some Gemini
        # OpenAI-compat turns), synthesize a short confirmation from tool activity
        # so callers aren't left with an empty RunResult.
        if not (output.text() or "").strip() and tool_activity:
            summary = _synthesize_tool_activity_summary(tool_activity)
            if summary:
                from loomable.content.parts import MediaPart as _MP

                output = AgentOutput(
                    parts=[
                        _MP(
                            modality=Modality.TEXT,
                            media_type="text/plain",
                            data=summary.encode("utf-8"),
                        )
                    ]
                )

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
        if deliverable_complete_nudges:
            metadata["deliverable_complete_nudged"] = True
            metadata["deliverable_complete_nudges"] = deliverable_complete_nudges
        if deliverable_complete_forced:
            metadata["deliverable_complete_forced"] = True
        if structured_from_write_json:
            metadata["structured_from_write_json"] = True
        still_missing = _missing_require_tool_specs(
            require_tool_specs, satisfied_require_tools
        )
        if still_missing:
            metadata["required_tools_missing"] = still_missing
            if self.strict_require_tools:
                from loomable.agent.errors import RequireToolsError

                raise RequireToolsError(still_missing)

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

        # Passive Always-mode lite: heuristic user-fact extraction into NoteStore.
        if self.memory_auto_extract and self.note_store is not None and input_text:
            try:
                from loomable.kernel.stores import _run_sync
                from loomable.memory.compose import auto_extract_into_notes

                written = _run_sync(
                    auto_extract_into_notes(
                        self.note_store,
                        input_text,
                        user_id=self._memory_user_id,
                        scope=self._memory_scope,
                    )
                )
                if written:
                    merged = list(dict.fromkeys([*written, *self._cached_user_facts]))
                    self._cached_user_facts = merged[:20]
            except Exception as exc:  # noqa: BLE001 — never break the run for memory hygiene
                self.events.emit(Event(
                    kind="memory_error",
                    t=time.monotonic(),
                    attributes={
                        "op": "auto_extract",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    },
                ))

    async def _refresh_user_memory_context(self, query: str) -> None:
        """Recall user notes into ``_cached_user_facts`` for the next prompt prefix."""
        if not self.memory_auto_recall or self.note_store is None:
            return
        try:
            hits = await self.note_store.recall(query or "user preferences identity", k=5)
            if hits:
                self._cached_user_facts = list(
                    dict.fromkeys([*(h.text for h in hits), *self._cached_user_facts])
                )[:20]
        except Exception as exc:  # noqa: BLE001
            self.events.emit(Event(
                kind="memory_error",
                t=time.monotonic(),
                attributes={
                    "op": "user_recall",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            ))

    async def astream(self, input, *, output_schema=None, context=None,
                      images=None, videos=None, audio=None):
        """Project provisional deltas and a validated, persisted terminal result."""
        from contextlib import aclosing
        from .execution import stream_chunks
        async with aclosing(stream_chunks(
            self, input, output_schema=output_schema, context=context,
            images=images, videos=videos, audio=audio,
        )) as chunks:
            async for chunk in chunks:
                yield chunk

    async def astream_events(self, input, *, output_schema=None, context=None,
                             images=None, videos=None, audio=None):
        """Project AG-UI events over the same lifecycle as arun and astream."""
        from contextlib import aclosing
        from .execution import stream_events
        async with aclosing(stream_events(
            self, input, output_schema=output_schema, context=context,
            images=images, videos=videos, audio=audio,
        )) as events:
            async for event in events:
                yield event

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

    async def dispatch_tools_gated(self, calls):
        if self._journal is not None:
            return await self._journal.dispatch(self, calls, self._dispatch_tools_gated_uncached)
        return await self._dispatch_tools_gated_uncached(calls)

    async def _dispatch_tools_gated_uncached(
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
        if self._journal is not None:
            from copy import deepcopy
            approved_proposals = {
                call.id: deepcopy(call) for call in surviving
                if call.tool_name in self.require_confirmation
            }
        else:
            approved_proposals = {}
        hook_rejected = [c for c in surviving if self._rejected_by_pre_hooks(c)]
        if approved_proposals and any(
            original != next((call for call in surviving if call.id == call_id), None)
            for call_id, original in approved_proposals.items()
        ):
            raise ValueError("Tool hooks cannot change a durable approval proposal")
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
                decision = (self._journal.state["approvals"].get(call.id, False)
                            if self._journal is not None else self.approver(call))
                if decision:
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
