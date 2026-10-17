"""loomable.agent.builder - High-level Agent builder.

This module implements the ergonomic high-level entry point (:class:`Agent`) that
assembles a runnable :class:`BuiltAgent` from a compact configuration, composing the
existing kernel primitives without modifying ``loomable.kernel`` (Req 1.7, 2.4).

``Agent(model=...)`` is enough to produce a runnable agent: :meth:`Agent.build`
constructs default implementations for every kernel subsystem that was not supplied
(Req 1.2) and uses any pre-constructed primitive that *was* supplied (Req 2.2/2.3).
Missing or invalid required fields raise :class:`AgentConfigError` naming the field
before any run is attempted (Req 1.6).

Multi-agent orchestration is handled via ``Team`` / ``Workflow`` (and low-level
``Flow``). The agent retains single-agent auto-escalation: single-shot →
tool-loop → self-plan (Req 14.4, 17.3).
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from weakref import WeakValueDictionary
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from loomable.content import (
    AgentInput,
    ModelCapabilities,
)
from loomable.kernel.context import ContextManager
from loomable.kernel.contracts import ModelProvider, Retriever, Tool
from loomable.kernel.errors import MCPConnectionError, SessionNotFoundError, SkillLoadError
from loomable.kernel.guardrails import GuardrailHarness
from loomable.kernel.long_term import LongTermStore
from loomable.kernel.mcp_client import MCPClient
from loomable.kernel.memory import MemoryManager
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.model_router import ModelRouter
from loomable.kernel.models import (
    AgentConfig,
    Session,
)
from loomable.kernel.planner import Planner
from loomable.kernel.retrievers import RetrieverTool
from loomable.kernel.skills import SkillLoader
from loomable.kernel.stores import SessionStore
from loomable.kernel.summarizer import Summarizer
from loomable.kernel.tool_runtime import ToolRuntime

from .errors import (
    AgentConfigError,
)
from .events import AgentEvents, NoOpEvents
from .run import RunResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator

    from loomable.providers.resilient import RetryPolicy
    from loomable.toolkits._base import Toolkit
    from loomable.memory import NoteStore
    from .routing import ComplexityRouter


from ._support import (
    _DEFAULT_PROVIDER_ID,
    ToolHook as ToolHook,
    PostToolHook as PostToolHook,
    GatedDispatchResult as GatedDispatchResult,
    _input_text,
    _is_run_context,
    _coerce_media_item as _coerce_media_item,
    _path_constraint_met as _path_constraint_met,
    ModelSpec,
)

from .runtime import BuiltAgent


class Agent:
    """High-level agent builder.

    The constructor is cheap: it only records configuration. Call :meth:`build` to
    validate the configuration and construct the runnable :class:`BuiltAgent`.
    """

    def __init__(
        self,
        model: "ModelProvider | ModelSpec | str",
        *,
        execution: "Any | None" = None,
        durability: "Any | None" = None,
        name: str = "",
        description: str = "",
        role: str = "",
        goal: str = "",
        instructions: str | None = None,
        tools: "list[Tool | Toolkit] | None" = None,
        subagents: "list[Agent] | None" = None,
        skills: list[Path] | list[str] | list[Path | str] | None = None,
        mcp_servers: list[Any] | None = None,
        capabilities: ModelCapabilities | str | list[str] | None = None,
        modalities: str | list[str] | None = None,
        text_only: bool = False,
        token_budget: int = 8192,
        max_run_tokens: int | None = None,
        checkpoint_interval: int = 5,
        session_id: str | None = None,
        max_cached_sessions: int = 128,
        user_id: str | None = None,
        scopes: dict[str, Any] | None = None,
        resume: bool = False,
        use_memory: bool = True,
        memory_window: int = 8,
        compaction_threshold: int = 16,
        input_schema: type | None = None,
        response_model: type | None = None,
        # knowledge / RAG:
        retrievers: list[Retriever] | None = None,
        knowledge_base: Any = None,
        knowledge: list[str] | None = None,
        embedder: Any = None,
        knowledge_top_k: int = 3,
        # tool hooks / HITL:
        tool_hooks: list[Any] | None = None,
        require_confirmation: list[str] | None = None,
        approver: Any | None = None,
        # harness knobs (avoids needing build() for common config):
        tool_timeout: float | None = None,
        tool_concurrency: int | None = None,
        max_tool_iterations: int | None = None,
        require_final_text: bool = True,
        require_tools: list[str] | None = None,
        strict_require_tools: bool = False,
        max_delegations: int | None = None,
        max_depth: int = 4,
        # Tiered model routing:
        tiers: dict[str, Any] | None = None,
        tier_policy: dict[str, Any] | None = None,
        fallback_tiers: dict[str, str] | None = None,
        # low-level overrides:
        context_manager: ContextManager | None = None,
        memory: Any = None,
        kernel_memory: MemoryManager | None = None,
        tool_runtime: ToolRuntime | None = None,
        harness: GuardrailHarness | None = None,
        planner: Planner | None = None,
        planning_model: str | None = None,
        session_store: Any | None = None,
        memory_backend: Any | None = None,
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
        memory_auto_extract: bool = False,
        # Case mode (plan → dispatch → synthesize → accept):
        mode: str | None = None,
        dispatch: str = "reuse",
        accept: Any = None,
        board: bool = True,
        max_rounds: int | None = None,
        max_plan_steps: int = 5,
        checkpointer: Any = None,
        # Multimodal feedback (Req 7.5):
        feedback_media: bool = True,
        # Developer experience:
        debug: bool = False,
        # Lifecycle callbacks:
        on_tool_call: Any = None,
        on_complete: Any = None,
        # Progressive capability discovery (skills / tools / MCP):
        discovery: bool = False,
        skill_catalog: list[Path] | list[str] | list[Path | str] | None = None,
        defer_mcp: bool | None = None,
        defer_local_tools: bool | None = None,
        discovery_core_tools: Sequence[str] | None = None,
        eager_skills: bool | Sequence[str] | None = None,
        lazy_mcp: bool | None = None,
        activation_allowlist: Sequence[str] | None = None,
        activation_denylist: Sequence[str] | None = None,
        tool_namespaces: list[dict[str, Any]] | None = None,
    ) -> None:
        if execution is not None:
            from .options import ExecutionOptions
            if not isinstance(execution, ExecutionOptions):
                raise AgentConfigError("execution must be ExecutionOptions")
            flat = dict(token_budget=token_budget, max_run_tokens=max_run_tokens,
                        max_tool_iterations=max_tool_iterations, tool_timeout=tool_timeout,
                        tool_concurrency=tool_concurrency, loop_repeat_threshold=loop_repeat_threshold)
            defaults = dict(token_budget=8192, max_run_tokens=None, max_tool_iterations=None,
                            tool_timeout=None, tool_concurrency=None, loop_repeat_threshold=3)
            conflicts = [k for k, v in flat.items() if v != defaults[k]]
            if conflicts:
                raise AgentConfigError("execution conflicts with flat options: " + ", ".join(conflicts))
            token_budget, max_run_tokens = execution.token_budget, execution.max_run_tokens
            max_tool_iterations = execution.max_tool_iterations
            tool_timeout, tool_concurrency = execution.tool_timeout, execution.tool_concurrency
            loop_repeat_threshold = execution.loop_repeat_threshold
        if durability is not None:
            from .options import DurabilityOptions
            if not isinstance(durability, DurabilityOptions) or not session_id:
                raise AgentConfigError("durability requires DurabilityOptions and an explicit session_id")
            if mode or discovery or subagents or complexity_router or plan_tool or planner or mcp_servers:
                raise AgentConfigError("durability currently supports static single-agent tools only")
        self._durability = durability

        # --- Resolve model string shorthand (e.g. "openai:gpt-4o-mini") ---
        if isinstance(model, str):
            from loomable.providers.resolver import resolve_model
            resolved_provider = resolve_model(model)
            model = ModelSpec(provider=model.split(":")[0] if ":" in model else "openai",
                             provider_impl=resolved_provider,
                             capabilities=self._provider_capabilities(resolved_provider) or ModelCapabilities())

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
        self._skill_bodies: list[tuple[str, str]] = []
        self._mcp_servers = mcp_servers
        self._discovery = bool(discovery)
        self._skill_catalog = skill_catalog
        # Default: defer MCP tool advertisement when discovery is on.
        self._defer_mcp = bool(discovery) if defer_mcp is None else bool(defer_mcp)
        # Default: defer non-core local tools when discovery is on (schema budget).
        self._defer_local_tools = (
            bool(discovery) if defer_local_tools is None else bool(defer_local_tools)
        )
        self._discovery_core_tools = (
            [str(x) for x in discovery_core_tools] if discovery_core_tools else None
        )
        # None + discovery → progressive (metadata only). True → eager all skills=.
        # Sequence → eager those names only. False → never eager-load bodies.
        self._eager_skills = eager_skills
        # Lazy MCP: catalog server specs without connecting until activate_mcp_server.
        # Defaults to True when discovery is on (industry pattern: connect on demand).
        self._lazy_mcp = bool(discovery) if lazy_mcp is None else bool(lazy_mcp)
        self._activation_allowlist = (
            [str(x) for x in activation_allowlist] if activation_allowlist else None
        )
        self._activation_denylist = (
            [str(x) for x in activation_denylist] if activation_denylist else None
        )
        self._tool_namespaces = (
            [dict(ns) for ns in tool_namespaces] if tool_namespaces else None
        )
        # High-level modality DX: modalities="text" / text_only=True preferred over
        # constructing ModelCapabilities with frozensets.
        from loomable.content.capabilities import capabilities_for

        self._modalities_raw = modalities if isinstance(modalities, str) else None
        if text_only:
            self._modalities_raw = "text"

        if text_only and (modalities is not None or capabilities is not None):
            raise AgentConfigError(
                "text_only (mutually exclusive with modalities= and capabilities=; "
                "use modalities='text' instead)"
            )
        if modalities is not None and capabilities is not None:
            raise AgentConfigError(
                "modalities (mutually exclusive with capabilities=; pick one)"
            )
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
        self._token_budget = token_budget
        # None → spend uses token_budget. Deep agents pass 0 for unbounded.
        self._max_run_tokens = max_run_tokens
        self._checkpoint_interval = checkpoint_interval
        self._session_id = session_id
        self._user_id = user_id
        self._scopes = dict(scopes) if scopes else {}
        self._resume = resume
        self._response_model = response_model

        self._use_memory = use_memory
        self._memory_window = memory_window
        self._compaction_threshold = compaction_threshold
        self._input_schema = input_schema
        self._retrievers = retrievers
        self._knowledge_base = knowledge_base
        self._knowledge_retrievers: list[Retriever] | None = None
        self._knowledge = knowledge
        self._embedder = embedder
        self._knowledge_top_k = knowledge_top_k
        self._tool_hooks = tool_hooks
        self._require_confirmation = require_confirmation
        self._approver = approver
        self._tool_timeout = tool_timeout
        self._tool_concurrency = tool_concurrency
        self._max_tool_iterations = max_tool_iterations
        self._require_final_text = require_final_text
        self._require_tools = list(require_tools) if require_tools else []
        self._strict_require_tools = bool(strict_require_tools)
        self._max_delegations = max_delegations
        self._max_depth = max_depth

        # Tiered model routing.
        self._tiers = tiers
        self._tier_policy = tier_policy
        self._fallback_tiers = fallback_tiers

        # Low-level overrides.
        self._context_manager = context_manager
        self._memory_bundle = None
        self._memory_auto_extract = False
        # ``memory=`` accepts :class:`~loomable.memory.Memory` from Memory.compose.
        from loomable.memory.compose import is_kernel_memory_manager, is_memory_bundle

        if is_memory_bundle(memory):
            if getattr(memory, "working", None) is not None and memory.working is not None:
                raise AgentConfigError(
                    "WorkingMemory is for Workflow blackboards — use "
                    "Workflow(..., memory=True) or Workflow(memory=working.store). "
                    "Do not pass working= inside Agent(memory=Memory.compose(...))."
                )
            flat_conflict = [
                name
                for name, value in (
                    ("session_store", session_store),
                    ("memory_backend", memory_backend),
                    ("note_store", note_store),
                )
                if value is not None
            ]
            if flat_conflict:
                raise AgentConfigError(
                    "memory="
                    + f" already sets store layers; do not also pass {', '.join(flat_conflict)}. "
                    "Put stores only inside Memory.compose(...)."
                )
            bundle = memory.with_scopes(user_id=user_id, scopes=scopes)
            self._memory_bundle = bundle
            composed = bundle.to_agent_kwargs()
            if "session_store" in composed:
                session_store = composed["session_store"]
            if "memory_backend" in composed:
                memory_backend = composed["memory_backend"]
            if "note_store" in composed:
                note_store = composed["note_store"]
            if composed.get("memory_tool"):
                memory_tool = True
            if knowledge is None and "knowledge" in composed:
                knowledge = composed["knowledge"]
            if knowledge_base is None and composed.get("knowledge_base") is not None:
                knowledge_base = composed["knowledge_base"]
            if retrievers is None and composed.get("retrievers") is not None:
                retrievers = composed["retrievers"]
            if embedder is None and "embedder" in composed:
                embedder = composed["embedder"]
            if "knowledge_top_k" in composed and knowledge_top_k == 3:
                knowledge_top_k = composed["knowledge_top_k"]
            if "memory_window" in composed and memory_window == 8:
                memory_window = composed["memory_window"]
            if "compaction_threshold" in composed and compaction_threshold == 16:
                compaction_threshold = composed["compaction_threshold"]
            if composed.get("use_llm_summarizer"):
                use_llm_summarizer = True
            if composed.get("use_memory") is False:
                use_memory = False
            self._memory_auto_extract = bool(
                bundle.user is not None and bundle.user.auto_extract
            )
            self._memory = kernel_memory
        elif is_kernel_memory_manager(memory):
            raise AgentConfigError(
                "Agent(memory=...) expects Memory.compose(...); "
                "kernel MemoryManager is internal."
            )
        else:
            self._memory = kernel_memory or memory
            # Even without a Memory bundle, scopes+user_id can wrap an explicit note_store.
            if note_store is not None and (user_id or scopes):
                from loomable.memory.compose import MemoryScope, ScopedNoteStore

                scope = MemoryScope.of(**({**(scopes or {}), **({"user_id": user_id} if user_id else {})}))
                if scope and not isinstance(note_store, ScopedNoteStore):
                    note_store = ScopedNoteStore(note_store, scope=scope)
            self._memory_auto_extract = bool(memory_auto_extract)
        self._tool_runtime = tool_runtime
        self._harness = harness
        self._planner = planner
        self._planning_model = planning_model
        self._session_store = session_store
        self._memory_backend = memory_backend

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
        # Re-apply knobs that Memory.compose may have updated above
        self._use_memory = use_memory
        self._memory_window = memory_window
        self._compaction_threshold = compaction_threshold
        self._knowledge = knowledge
        self._knowledge_base = knowledge_base
        self._retrievers = retrievers
        self._embedder = embedder
        self._knowledge_top_k = knowledge_top_k
        self._mode = (mode or "").strip().lower() or None
        if self._mode is not None and self._mode != "case":
            raise AgentConfigError(
                f"Agent(mode={mode!r}) is unsupported; use mode='case' or omit mode."
            )
        if dispatch not in ("reuse", "spawn"):
            raise AgentConfigError(
                f"dispatch must be 'reuse' or 'spawn', got {dispatch!r}"
            )
        if self._mode != "case":
            if checkpointer is not None:
                raise AgentConfigError(
                    "checkpointer= only applies with Agent(mode='case') "
                    "(or construct Case/Workflow directly)."
                )
            if max_rounds is not None:
                raise AgentConfigError(
                    "max_rounds= only applies with Agent(mode='case')."
                )
        if self._mode != "case" and not board:
            raise AgentConfigError(
                "board= only applies with Agent(mode='case'). Omit board= or set mode='case'."
            )
        self._dispatch = dispatch
        self._accept = accept
        self._board = board
        self._max_rounds = max_rounds
        self._max_plan_steps = max_plan_steps
        self._checkpointer = checkpointer
        self._feedback_media = feedback_media
        if accept is not None and self._verifier is None:
            self._verifier = accept

        # Developer experience.
        self._debug = debug
        self._on_tool_call = on_tool_call
        self._on_complete = on_complete

        # Cached BuiltAgent so repeated run calls reuse one runtime/session.
        self._built: BuiltAgent | None = None
        if max_cached_sessions < 1:
            raise ValueError("max_cached_sessions must be positive")
        self._max_cached_sessions = max_cached_sessions
        self._session_agents: OrderedDict[str, Agent] = OrderedDict()
        self._live_session_agents: WeakValueDictionary[str, Agent] = WeakValueDictionary()
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
            planning_model=(
                {"provider": self._planning_model} if self._planning_model else None
            ),
            tiers={},
            tier_policy=None,
            fallback_tiers={},
            token_budget=self._token_budget,
            checkpoint_interval=self._checkpoint_interval,
        )

        # --- Construct or reuse subsystems ---
        memory = self._memory or MemoryManager()
        self._materialize_knowledge_base()
        tool_registry, skill_errors = self._build_tool_registry()
        # --- MCP servers: connect and enumerate tools (Req 5.1–5.4) ---
        # When discovery+defer_mcp: catalog MCP tools without advertising them until
        # activate_tool (progressive disclosure).
        mcp_tools, mcp_errors, mcp_stubs, mcp_server_stubs = self._connect_mcp_servers_sync(
            defer=self._discovery and self._defer_mcp,
            occupied=set(tool_registry),
            lazy=self._discovery and self._lazy_mcp,
        )
        tool_registry.update(mcp_tools)
        tool_runtime = self._tool_runtime or ToolRuntime(tool_registry)
        harness = self._harness or GuardrailHarness([])
        planner = self._planner
        session_store = self._resolve_session_store()

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
            from loomable.memory import make_memory_tool

            mem_tool = make_memory_tool(self._note_store)
            tool_registry[mem_tool.name] = mem_tool
            if self._tool_runtime is None:
                tool_runtime = ToolRuntime(tool_registry)

        # --- Subagent delegation tools (Proposal §2): register each subagent as
        # a tool so the parent LLM can delegate tasks at runtime ---
        if self._subagents:
            from .delegation import make_delegation_tools

            chain_max = getattr(self, "_delegation_max_depth", None)
            delegation_tools = make_delegation_tools(
                self._subagents,
                max_delegations=self._max_delegations,
                max_depth=chain_max if chain_max is not None else self._max_depth,
                depth=int(getattr(self, "_delegation_depth", 0) or 0),
            )
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
        if self._knowledge:
            if self._embedder is None:
                raise AgentConfigError(
                    "knowledge= requires embedder= (passive RAG indexes documents "
                    "at build time). Pass embedder=..., or use knowledge_base= for "
                    "search_* tools without passive injection."
                )
            long_term = LongTermStore()
            self._index_knowledge_sync(long_term, self._knowledge, self._embedder)

        # memory_tool without a resolvable note store was a silent no-op — fail loud.
        if self._memory_tool and self._note_store is None:
            raise AgentConfigError(
                "memory_tool=True requires note_store= (or UserMemory with "
                "note_store=/embedder= inside Memory.compose). "
                "Pass note_store=... or set memory_tool=False."
            )

        from loomable.memory.compose import MemoryScope

        if (
            self._memory_bundle is not None
            and self._memory_bundle.user is not None
        ):
            _resolved_memory_scope = self._memory_bundle.user.resolve_scope() or None
        elif self._scopes or self._user_id:
            _resolved_memory_scope = MemoryScope.of(
                **{
                    **self._scopes,
                    **({"user_id": self._user_id} if self._user_id else {}),
                }
            )
        else:
            _resolved_memory_scope = None

        built = BuiltAgent(
            loop=None,
            durability=self._durability,
            on_complete=self._on_complete,
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
            _max_run_tokens=self._max_run_tokens,
            events=events,
            complexity_router=self._complexity_router,
            note_store=self._note_store,
            memory_auto_extract=self._memory_auto_extract,
            memory_auto_recall=True,
            _memory_user_id=self._user_id,
            _memory_scope=_resolved_memory_scope,
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

        # --- Progressive discovery (skills / tools / MCP) ---
        if self._discovery:
            self._wire_discovery(
                built, mcp_stubs=mcp_stubs, mcp_server_stubs=mcp_server_stubs
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
        built.strict_require_tools = self._strict_require_tools
        if self._approver is not None:
            built.approver = self._approver

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

    def bind_session(self, session_id: str | None, *, resume: bool | None = None) -> None:
        """Bind a client ``session_id`` for L1/L2 (and Case checkpoints).

        Same idea as :meth:`loomable.case.Case.bind_session`: HTTP/stream callers
        change the conversation thread without reconstructing the Agent.

        When ``resume`` is omitted, auto-resumes if ``session_store`` /
        ``memory_backend`` already has that id; otherwise starts a fresh session
        on the next :meth:`build`. Requires an explicit durable store for chat
        history to survive across process requests (default in-memory SQLite is
        per build).
        """
        if not session_id:
            return
        changed = session_id != self._session_id
        self._session_id = session_id
        if self._case is not None and hasattr(self._case, "bind_session"):
            self._case.bind_session(session_id)

        if resume is not None:
            self._resume = bool(resume)
        else:
            self._resume = False
            try:
                store = self._resolve_session_store()
                store.resume(session_id)
                self._resume = True
            except Exception:  # noqa: BLE001 — missing session → create on build
                self._resume = False

        if changed or self._built is not None:
            self._built = None

    def _get_built(self) -> BuiltAgent:
        """Build the agent once and cache it for subsequent runs."""
        if self._built is None:
            self._built = self.build()
        return self._built

    def _for_session(self, session_id):
        if not session_id or session_id == self._session_id:
            return self
        scoped = self._live_session_agents.get(session_id)
        if scoped is None:
            import copy
            scoped = copy.copy(self)
            scoped._session_id = session_id
            scoped._resume = True
            scoped._built = None
            scoped._case = None
            scoped._session_agents = OrderedDict()
            scoped._live_session_agents = WeakValueDictionary()
            scoped._session_store = self._resolve_session_store()
            try:
                scoped._session_store.resume(session_id)
            except SessionNotFoundError:
                scoped._resume = False
            self._live_session_agents[session_id] = scoped
        self._session_agents[session_id] = scoped
        self._session_agents.move_to_end(session_id)
        if len(self._session_agents) > self._max_cached_sessions:
            self._session_agents.popitem(last=False)
        return scoped

    async def arun(
        self,
        input: AgentInput | str,  # noqa: A002
        *,
        images: "list[str | Path | Any] | None" = None,
        videos: "list[str | Path | Any] | None" = None,
        audio: "list[str | Path | Any] | None" = None,
        output_schema: type | None = None,
        context: "Any | None" = None,
        session_id: str | None = None,
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
            Optional :class:`~loomable.agent.context.RunContext` (Flow/Workflow)
            or a plain dict. RunContext is forwarded so Agent-in-Flow behaves
            like a standalone BuiltAgent run.
        """
        if session_id and session_id != self._session_id:
            return await self._for_session(session_id).arun(
                input, images=images, videos=videos, audio=audio,
                output_schema=output_schema, context=context,
            )
        # Case mode: plan → dispatch → synthesize → accept.
        if self._mode == "case":
            unsupported = []
            if images is not None:
                unsupported.append("images")
            if videos is not None:
                unsupported.append("videos")
            if audio is not None:
                unsupported.append("audio")
            if output_schema is not None:
                unsupported.append("output_schema")
            if context is not None:
                unsupported.append("context")
            if unsupported:
                raise AgentConfigError(
                    f"Agent(mode='case') does not accept per-call "
                    f"{', '.join(unsupported)}. Use plain text input via arun()."
                )
            case = self._get_case()
            text = self._coerce_run_text(input)
            result = await case.arun(text)
            if self._on_complete is not None:
                self._on_complete(result)
            return result

        built = self._get_built()
        # Use response_model as default output_schema when not overridden per-call
        schema = output_schema or self._response_model
        run_ctx = context if _is_run_context(context) else None
        result = await built.arun(
            input,
            images=images,
            videos=videos,
            audio=audio,
            output_schema=schema,
            context=run_ctx,
        )
        return result

    async def arun_once(self, input: str, *, session_id: str | None = None,
                        output_schema: type | None = None) -> RunResult:
        """Start, resume, or read a durable text run in a dedicated session."""
        if self._durability is None:
            raise ValueError("arun_once requires durability=DurabilityOptions(...)")
        scoped = self._for_session(session_id)
        return await scoped._get_built().arun_once(input, output_schema=output_schema or self._response_model)

    async def acontinue_run(self, run_id, *, approvals=None, outcomes=None, session_id=None,
                            output_schema=None):
        """Continue an opt-in durable run using its original recorded input."""
        scoped = self._for_session(session_id)
        built = scoped._get_built()
        return await built.acontinue_run(run_id, approvals=approvals, outcomes=outcomes,
                                         output_schema=output_schema or self._response_model)

    def cancel(self, *, session_id: str | None = None) -> bool:
        """Request cooperative cancellation of the active run.

        ``mode="case"`` cancels the underlying Case/Workflow. Otherwise cancels
        the BuiltAgent tool loop.
        """
        if session_id and session_id != self._session_id:
            scoped = self._live_session_agents.get(session_id)
            return scoped.cancel() if scoped is not None else False
        hit = False
        if self._mode == "case" and self._case is not None:
            cancel = getattr(self._case, "cancel", None)
            if callable(cancel):
                hit = bool(cancel()) or hit
        if self._built is not None:
            hit = self._built.cancel() or hit
        return hit

    def run(
        self,
        input: AgentInput | str,  # noqa: A002
        *,
        images: "list[str | Path | Any] | None" = None,
        videos: "list[str | Path | Any] | None" = None,
        audio: "list[str | Path | Any] | None" = None,
        output_schema: type | None = None,
        context: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """Synchronous wrapper around :meth:`arun` (Req 1.4).

        Uses :func:`asyncio.run`, which requires that no event loop is already
        running on the calling thread. In an async context, call :meth:`arun`.
        """
        return asyncio.run(self.arun(input, images=images, videos=videos, audio=audio, output_schema=output_schema, context=context, session_id=session_id))

    async def astream(self, input, *, output_schema=None, context=None,
                      images=None, videos=None, audio=None, session_id=None):
        """Stream the ordinary execution lifecycle with response_model defaults."""
        from contextlib import aclosing
        scoped = self._for_session(session_id)
        if scoped._mode == "case":
            raise AgentConfigError("Agent(mode='case') does not support astream(); use astream_events().")
        async with aclosing(scoped._get_built().astream(
            input, output_schema=output_schema or self._response_model,
            context=context if _is_run_context(context) else None,
            images=images, videos=videos, audio=audio,
        )) as chunks:
            async for chunk in chunks:
                yield chunk

    async def astream_events(
        self,
        input: AgentInput | str,  # noqa: A002
        *,
        images: "list[str | Path | Any] | None" = None,
        videos: "list[str | Path | Any] | None" = None,
        audio: "list[str | Path | Any] | None" = None,
        output_schema: type | None = None,
        context: "Any | None" = None,
        session_id: str | None = None,
    ) -> "AsyncIterator[Any]":
        """Yield AG-UI-compatible stream events (lifecycle, tools, text).

        When ``mode="case"``, streams Case pipeline events.
        """
        if session_id and session_id != self._session_id:
            from contextlib import aclosing
            async with aclosing(self._for_session(session_id).astream_events(
                input, images=images, videos=videos, audio=audio,
                output_schema=output_schema, context=context,
            )) as events:
                async for event in events:
                    yield event
            return

        if self._mode == "case":
            case = self._get_case()
            text = self._coerce_run_text(input)
            async for event in case.astream_events(text, session_id=session_id):
                yield event
            return

        built = self._get_built()
        schema = output_schema or self._response_model
        kwargs: dict[str, Any] = {
            "images": images,
            "videos": videos,
            "audio": audio,
            "output_schema": schema,
        }
        if _is_run_context(context):
            kwargs["context"] = context
        from contextlib import aclosing
        async with aclosing(built.astream_events(input, **kwargs)) as events:
            async for event in events:
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
        if self._description:
            parts.append(self._description)
        if self._instructions:
            if parts:
                parts.append("")  # blank line separator
            parts.append(self._instructions)
        # Progressive skills: inject loaded SKILL.md bodies into the prompt.
        if self._skill_bodies:
            if parts:
                parts.append("")
            for skill_name, body in self._skill_bodies:
                title = skill_name or "skill"
                parts.append(f"## Skill: {title}\n{body.strip()}")
        # Ship-any-retriever: tell the model which search tools are attached.
        attached = self._attached_retrievers()
        if attached:
            if parts:
                parts.append("")
            lines = [
                "Knowledge search tools (call when facts from the knowledge base are needed):"
            ]
            for r in attached:
                desc = (getattr(r, "description", None) or "").strip()
                if desc:
                    lines.append(f"- `{r.name}`: {desc}")
                else:
                    lines.append(f"- `{r.name}`: search this knowledge base")
            parts.append("\n".join(lines))
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
        :class:`ModelSpec` or a bare provider.

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
            if self._planning_model and self._planning_model not in providers:
                default_impl = providers.get(provider_id)
                if default_impl is not None:
                    providers[self._planning_model] = default_impl
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
        if self._planning_model and self._planning_model not in providers:
            default_impl = providers.get(provider_id)
            if default_impl is not None:
                providers[self._planning_model] = default_impl
        return interface, provider_id, self._provider_capabilities(self._model)

    @staticmethod
    def _provider_capabilities(provider: ModelProvider | None) -> ModelCapabilities | None:
        capabilities = getattr(provider, "capabilities", None)
        return capabilities if isinstance(capabilities, ModelCapabilities) else None

    def _resolve_capabilities(
        self, spec_capabilities: ModelCapabilities | None
    ) -> ModelCapabilities:
        """Apply agent policy within the input formats the provider can transport."""
        configured = self._capabilities or spec_capabilities or ModelCapabilities()
        provider = self._model.provider_impl if isinstance(self._model, ModelSpec) else self._model
        transport = self._provider_capabilities(provider)
        if transport is None:
            return configured
        return ModelCapabilities(input=configured.input & transport.input, output=configured.output)

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
                        if ft.name in registry:
                            import warnings

                            warnings.warn(
                                f"Tool name {ft.name!r} already registered; "
                                "later registration wins (check toolkit collisions).",
                                stacklevel=2,
                            )
                        registry[ft.name] = ft
                else:
                    if item.name in registry:
                        import warnings

                        warnings.warn(
                            f"Tool name {item.name!r} already registered; "
                            "later registration wins (check toolkit collisions).",
                            stacklevel=2,
                        )
                    registry[item.name] = item

        # --- Skills: discover + load via the kernel SkillLoader (Req 4.1–4.4) ---
        skill_bodies: list[tuple[str, str]] = []
        if self._skills:
            from loomable.skills import resolve_skills

            loader = SkillLoader()
            skill_roots = resolve_skills(self._skills)
            manifests = loader.discover(skill_roots)
            eager_names = self._resolve_eager_skill_names(
                [m.name for m in manifests]
            )
            for manifest in manifests:
                # Progressive disclosure: under discovery, skip full body/script
                # registration unless the skill is in the eager set.
                if self._discovery and manifest.name not in eager_names:
                    continue
                try:
                    loaded_skill = loader.load(manifest)
                    body = (getattr(loaded_skill, "body", None) or "").strip()
                    if body:
                        skill_bodies.append(
                            (getattr(loaded_skill, "name", None) or manifest.name, body)
                        )
                    for script_tool in loaded_skill.get_tools():
                        registry[script_tool.name] = script_tool
                except SkillLoadError as err:
                    skill_errors.append(err)
        self._skill_bodies = skill_bodies

        if self._attached_retrievers():
            for retriever in self._attached_retrievers():
                if retriever.name in registry:
                    raise AgentConfigError(
                        f"retrievers[{retriever.name!r}] "
                        f"(name collides with an existing tool)"
                    )
                desc = (getattr(retriever, "description", None) or "").strip() or None
                registry[retriever.name] = RetrieverTool(retriever, description=desc)

        return registry, skill_errors

    def _resolve_eager_skill_names(self, discovered_names: list[str]) -> set[str]:
        """Which configured skills get full body injection at build time.

        Under ``discovery=True`` the default is progressive (empty eager set):
        only metadata is catalogued and the model calls ``load_skill``. Pass
        ``eager_skills=True`` to restore legacy full injection, or a name list
        for selective eagerness (e.g. research profile can opt in).
        """
        if not self._discovery:
            return set(discovered_names)
        policy = self._eager_skills
        if policy is True:
            return set(discovered_names)
        if policy is False or policy is None:
            return set()
        wanted = {str(x).strip().lower() for x in policy if str(x).strip()}
        return {n for n in discovered_names if n.lower() in wanted}

    def _discovery_core_name_set(self) -> set[str]:
        """Always-advertised tool names when local deferral is on."""
        from .discovery import DISCOVERY_META_TOOL_NAMES

        core: set[str] = set(DISCOVERY_META_TOOL_NAMES)
        if self._discovery_core_tools:
            core.update(self._discovery_core_tools)
        # require_tools must remain callable without activate
        for raw in self._require_tools or []:
            name = (raw or "").split(":", 1)[0].strip()
            if name:
                core.add(name)
        # Knowledge-base / retriever tools must stay advertised — never defer them.
        for retriever in self._attached_retrievers():
            name = getattr(retriever, "name", "") or ""
            if name:
                core.add(name)
        return core

    def _connect_mcp_servers_sync(
        self,
        *,
        defer: bool = False,
        lazy: bool = False,
        occupied: set[str] | None = None,
    ) -> tuple[dict[str, Tool], list[MCPConnectionError], list[Any], list[Any]]:
        """Connect to configured MCP servers synchronously (Req 5.1–5.4).

        Wraps the async :meth:`_connect_mcp_servers` for use from the synchronous
        :meth:`build`. Returns the MCP tool registry, connection errors, tool
        catalog stubs (populated when ``defer=True`` or for discovery indexing),
        and server catalog stubs (:class:`~loomable.agent.discovery.ServerStub`).
        """
        if not self._mcp_servers:
            return {}, [], [], []
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already inside an event loop — create a new loop in a thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(
                    asyncio.run, self._connect_mcp_servers(defer=defer, lazy=lazy, occupied=occupied)
                )
                return future.result()
        return asyncio.run(self._connect_mcp_servers(defer=defer, lazy=lazy, occupied=occupied))

    async def _connect_mcp_servers(
        self,
        *,
        defer: bool = False,
        lazy: bool = False,
        occupied: set[str] | None = None,
    ) -> tuple[dict[str, Tool], list[MCPConnectionError], list[Any], list[Any]]:
        """Connect to configured MCP servers and enumerate their tools (Req 5.1–5.4).

        For each MCP server specification in ``self._mcp_servers``, the kernel
        :class:`~loomable.kernel.mcp_client.MCPClient` is used to connect and
        enumerate tools. Each discovered tool is wrapped as an :class:`MCPTool`
        (a kernel :class:`Tool`) whose ``invoke`` delegates to
        ``MCPClient.call_tool``.

        When ``defer=True``, tools are returned as catalog stubs only (not
        registered) so the model must ``activate_tool`` before calling them.

        When ``lazy=True`` (industry "search-then-connect" pattern), servers are
        **not connected at all** during build. Each configured server is
        catalogued as an unconnected :class:`~loomable.agent.discovery.ServerStub`
        so the model can discover it via ``search_mcp`` / ``search_namespaces``
        and connect on demand with ``activate_mcp_server``. ``lazy`` takes
        precedence over ``defer`` (nothing to defer if nothing connected).

        A failed connection yields an :class:`MCPConnectionError` for that server
        while other servers continue (Req 5.3). No kernel code is modified (Req 5.4).

        Returns
        -------
        tuple[dict[str, Tool], list[MCPConnectionError], list, list]
            The MCP tool registry, isolated connection errors, tool ToolStub
            list, and server ServerStub list.
        """
        from .discovery import ServerStub, ToolStub
        from .tools import MCPTool, mcp_tool_name

        occupied = set(occupied or ())
        registry: dict[str, Tool] = {}
        errors: list[MCPConnectionError] = []
        stubs: list[Any] = []
        servers: list[Any] = []

        def _resolve_server_id(spec: Any, idx: int) -> str:
            server_id = getattr(spec, "name", None) or getattr(spec, "id", None)
            if not server_id:
                if isinstance(spec, dict):
                    server_id = (
                        spec.get("server_id") or spec.get("name") or spec.get("id")
                    )
                if not server_id:
                    server_id = f"mcp-{idx}"
            return str(server_id)

        if lazy:
            # Catalog-only: do not open any transport. activate_mcp_server()
            # connects on demand from the stored spec.
            for spec_idx, spec in enumerate(self._mcp_servers):
                server_id = _resolve_server_id(spec, spec_idx)
                description = (
                    spec.get("description", "") if isinstance(spec, dict) else ""
                )
                servers.append(
                    ServerStub(
                        server_id=server_id,
                        description=description,
                        connected=False,
                        spec=spec,
                    )
                )
            return registry, errors, stubs, servers

        client = MCPClient()

        for spec_idx, spec in enumerate(self._mcp_servers):
            server_id = _resolve_server_id(spec, spec_idx)
            description = (
                spec.get("description", "") if isinstance(spec, dict) else ""
            )
            try:
                session = await client.connect({**spec, "server_id": server_id})
                try:
                    capabilities = await client.list_capabilities(session)
                finally:
                    if getattr(session, "spec", None) is not None:
                        await client.close(session)
                server_tool_names: list[str] = []
                for tool_info in capabilities.tools:
                    remote_name = tool_info.get("name", "")
                    if not remote_name:
                        continue
                    tool_name = mcp_tool_name(server_id, remote_name, occupied)
                    occupied.add(tool_name)
                    tool_description = tool_info.get("description", "")
                    parameters = tool_info.get("parameters", {
                        "type": "object",
                        "properties": {},
                    })
                    stub = ToolStub(
                        name=tool_name,
                        description=tool_description,
                        source="mcp",
                        server_id=server_id,
                        parameters=parameters if isinstance(parameters, dict) else {},
                        activated=not defer,
                        mcp_client=client,
                        mcp_session=session,
                        remote_name=remote_name,
                        namespace=f"mcp:{server_id}",
                    )
                    stubs.append(stub)
                    server_tool_names.append(tool_name)
                    if defer:
                        continue
                    mcp_tool = MCPTool(
                        name=tool_name,
                        description=tool_description,
                        parameters=parameters,
                        mcp_client=client,
                        session=session,
                        remote_name=remote_name,
                    )
                    registry[tool_name] = mcp_tool
                servers.append(
                    ServerStub(
                        server_id=server_id,
                        description=description,
                        connected=True,
                        spec=spec,
                        tool_count=len(server_tool_names),
                    )
                )
            except MCPConnectionError as err:
                errors.append(err)
                servers.append(
                    ServerStub(
                        server_id=server_id,
                        description=description,
                        connected=False,
                        spec=spec,
                    )
                )

        return registry, errors, stubs, servers

    def _wire_discovery(
        self,
        built: BuiltAgent,
        *,
        mcp_stubs: list[Any] | None = None,
        mcp_server_stubs: list[Any] | None = None,
    ) -> None:
        """Register search/load/activate meta-tools and build the capability catalog."""
        from loomable.skills import bundled_skills_root, resolve_skills

        from .discovery import (
            DISCOVERY_META_TOOL_NAMES,
            DISCOVERY_SYSTEM_NOTE,
            CapabilityCatalog,
            DiscoveryRuntime,
            NamespaceStub,
            SkillStub,
            ToolStub,
            catalog_from_skill_manifests,
            format_skill_catalog_for_prompt,
            make_discovery_tools,
        )

        catalog = CapabilityCatalog()
        loader = SkillLoader()

        # Skills already eagerly loaded via skills=
        loaded_names: set[str] = set()
        for name, _body in self._skill_bodies:
            loaded_names.add(name)

        skill_roots: list[Path] = []
        if self._skills:
            skill_roots.extend(resolve_skills(self._skills))
        if self._skill_catalog:
            skill_roots.extend(resolve_skills(self._skill_catalog))
        # Always index bundled package skills for search_skills.
        bundled = bundled_skills_root()
        if bundled not in skill_roots:
            skill_roots.append(bundled)

        seen_skill: set[str] = set()
        for root in skill_roots:
            try:
                manifests = loader.discover([root])
            except Exception:  # noqa: BLE001
                continue
            for stub in catalog_from_skill_manifests(manifests):
                if stub.name in seen_skill:
                    continue
                seen_skill.add(stub.name)
                stub.loaded = stub.name in loaded_names
                catalog.skills.append(stub)

        # Also mark loaded skills that may not have been rediscovered
        for name in loaded_names:
            if name not in seen_skill:
                catalog.skills.append(
                    SkillStub(name=name, description="", path=Path("."), loaded=True)
                )
                seen_skill.add(name)

        def _on_skill_body(skill_name: str, body: str) -> None:
            block = f"## Skill: {skill_name}\n{body.strip()}"
            if built.instructions:
                built.instructions = f"{built.instructions.rstrip()}\n\n{block}"
            else:
                built.instructions = block

        # Local tool → namespace map (for tools declared via tool_namespaces=).
        namespace_by_tool: dict[str, str] = {}
        for ns_spec in self._tool_namespaces or []:
            ns_name = str(ns_spec.get("name") or "").strip()
            if not ns_name:
                continue
            for tool_name in ns_spec.get("tools") or []:
                namespace_by_tool[str(tool_name)] = ns_name

        runtime = DiscoveryRuntime(
            catalog,
            skill_loader=loader,
            tool_runtime=built.tool_runtime,
            on_skill_body=_on_skill_body,
            skill_roots=skill_roots,
            activation_allowlist=self._activation_allowlist,
            activation_denylist=self._activation_denylist,
            lazy_mcp=self._discovery and self._lazy_mcp,
        )

        # Schema budget: move non-core local tools into deferred catalog.
        core_names = self._discovery_core_name_set()
        if self._defer_local_tools:
            for name, tool_obj in list(built.tool_runtime._tools.items()):
                if name in core_names or name in DISCOVERY_META_TOOL_NAMES:
                    continue
                runtime.register_pending_local(name, tool_obj)
                del built.tool_runtime._tools[name]
                catalog.tools.append(
                    ToolStub(
                        name=name,
                        description=getattr(tool_obj, "description", "") or "",
                        source="local",
                        activated=False,
                        parameters=getattr(tool_obj, "parameters", None) or {},
                        namespace=namespace_by_tool.get(name),
                    )
                )

        # Remaining (core) tools — searchable and already activated
        for tool_obj in built.tool_runtime._tools.values():
            name = getattr(tool_obj, "name", "") or ""
            if not name:
                continue
            if catalog.tool_by_name(name) is not None:
                continue
            catalog.tools.append(
                ToolStub(
                    name=name,
                    description=getattr(tool_obj, "description", "") or "",
                    source="local",
                    activated=True,
                    parameters=getattr(tool_obj, "parameters", None) or {},
                    namespace=namespace_by_tool.get(name),
                )
            )

        # Deferred / eager MCP stubs
        for stub in mcp_stubs or []:
            existing = catalog.tool_by_name(stub.name)
            if existing:
                existing.source = "mcp"
                existing.server_id = stub.server_id
                existing.mcp_client = stub.mcp_client
                existing.mcp_session = stub.mcp_session
                existing.parameters = stub.parameters
                existing.activated = stub.activated
                existing.description = stub.description or existing.description
                existing.namespace = existing.namespace or stub.namespace
            else:
                catalog.tools.append(stub)

        # MCP server catalog (connected or lazily deferred pending activate_mcp_server).
        for server in mcp_server_stubs or []:
            if catalog.server_by_id(server.server_id) is None:
                catalog.servers.append(server)

        # Namespaces: explicit tool_namespaces= groups + one auto mcp:<server_id>
        # namespace per MCP server (connected or lazy) grouping its tools.
        for ns_spec in self._tool_namespaces or []:
            ns_name = str(ns_spec.get("name") or "").strip()
            if not ns_name or catalog.namespace_by_name(ns_name) is not None:
                continue
            catalog.namespaces.append(
                NamespaceStub(
                    name=ns_name,
                    description=str(ns_spec.get("description") or ""),
                    tools=[str(t) for t in ns_spec.get("tools") or []],
                )
            )
        for server in mcp_server_stubs or []:
            ns_name = f"mcp:{server.server_id}"
            if catalog.namespace_by_name(ns_name) is not None:
                continue
            tools_in_ns = [
                t.name for t in catalog.tools if t.server_id == server.server_id
            ]
            catalog.namespaces.append(
                NamespaceStub(
                    name=ns_name,
                    description=server.description or f"MCP server {server.server_id}",
                    tools=tools_in_ns,
                )
            )

        for meta in make_discovery_tools(runtime):
            built.tool_runtime._tools[meta.name] = meta
            if catalog.tool_by_name(meta.name) is None:
                catalog.tools.append(
                    ToolStub(
                        name=meta.name,
                        description=meta.description or "",
                        source="local",
                        activated=True,
                    )
                )

        note_parts = [DISCOVERY_SYSTEM_NOTE.strip()]
        skill_note = format_skill_catalog_for_prompt(catalog.skills)
        if skill_note:
            note_parts.append(skill_note)
        if self._defer_local_tools:
            deferred = [t.name for t in catalog.tools if not t.activated]
            if deferred:
                note_parts.append(
                    "Deferred tools (search_tools / activate_tool): "
                    + ", ".join(sorted(deferred)[:40])
                    + ("…" if len(deferred) > 40 else "")
                )
        unconnected_servers = [s.server_id for s in catalog.servers if not s.connected]
        if unconnected_servers:
            note_parts.append(
                "Unconnected MCP servers (activate_mcp_server to connect + catalog "
                "their tools): " + ", ".join(sorted(unconnected_servers))
            )
        note = "\n\n".join(note_parts)
        if built.instructions:
            built.instructions = f"{built.instructions.rstrip()}\n\n{note}"
        else:
            built.instructions = note
        built.discovery = runtime

    def _resolve_session_store(self) -> Any:
        """Resolve L1/L2 persistence: ``session_store`` or ``memory_backend`` wrap."""
        if self._session_store is not None:
            return self._session_store
        if self._memory_backend is not None:
            from loomable.kernel.stores import BackendSessionStore

            return BackendSessionStore(self._memory_backend)
        self._session_store = SessionStore(max_sessions=self._max_cached_sessions)
        return self._session_store

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

    def _attached_retrievers(self) -> list[Retriever]:
        """Explicit ``retrievers=`` plus tools materialized from ``knowledge_base=``."""
        out: list[Retriever] = []
        seen: set[str] = set()
        for retriever in list(self._retrievers or []) + list(self._knowledge_retrievers or []):
            name = getattr(retriever, "name", "") or ""
            if name in seen:
                raise AgentConfigError(
                    f"retrievers[{name!r}] (duplicate knowledge_base/retrievers name)"
                )
            if name:
                seen.add(name)
            out.append(retriever)
        return out

    def _materialize_knowledge_base(self) -> None:
        """Resolve ``knowledge_base=`` (vector DB / sources) into retriever tools."""
        if self._knowledge_retrievers is not None:
            return
        spec = self._knowledge_base
        if not spec:
            self._knowledge_retrievers = []
            return
        from loomable.retrieval.knowledge import resolve_knowledge_base, run_sync

        self._knowledge_retrievers = list(
            run_sync(
                resolve_knowledge_base(
                    spec, embedder=self._embedder, user_id=self._user_id
                )
            )
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
