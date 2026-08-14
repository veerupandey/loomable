"""Deep Agent harness — loomable-native long-horizon agent.

Planning, workspace files, subagents, and research tools on the same Agent
surface (TodoTools, WorkspaceTools, task/task_batch, citations, accept gates).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from loomable.agent.builder import Agent
from loomable.agent.delegation import spawn_specialist
from loomable.agent.offload import make_workspace_offload_hook
from loomable.agent.tools import FunctionTool
from loomable.flow.loop import VerdictResult
from loomable.providers.resilient import RetryPolicy

__all__ = [
    "DEEP_DISCOVERY_CORE_TOOLS",
    "DEEP_DISCOVERY_CORE_SLIM",
    "DEEP_DISCOVERY_CORE_CODE",
    "SpecialistSpec",
    "create_deep_agent",
    "make_compact_conversation_tool",
    "make_research_accept",
    "make_task_tool",
    "make_task_tools",
]

logger = logging.getLogger("loomable.agent.deep")

# Always-advertised tools under discovery schema budget. Everything else is
# searchable via search_tools / activate_tool (images, pdf, code_exec, …).
DEEP_DISCOVERY_CORE_TOOLS: frozenset[str] = frozenset(
    {
        # Planning
        "write_todos",
        "read_todos",
        "update_todo",
        # Workspace
        "write_file",
        "read_file",
        "edit_file",
        "ls",
        "grep",
        "glob",
        "delete_file",
        # Research essentials
        "web_search",
        "fetch_url",
        "extract_text",
        "register_source",
        "verify_source",
        "register_claim",
        "list_sources",
        "format_bibliography",
        # Delegation / context
        "task",
        "task_batch",
        "think",
        "compact_conversation",
        "memory",
        "plan",
    }
)

# Schema-budget profile: keep planning + workspace + search/fetch + register_source.
# Activate verify/claim/bibliography/images/pdf via search_tools when needed.
DEEP_DISCOVERY_CORE_SLIM: frozenset[str] = frozenset(
    {
        "write_todos",
        "read_todos",
        "update_todo",
        "write_file",
        "read_file",
        "edit_file",
        "ls",
        "grep",
        "glob",
        "delete_file",
        "web_search",
        "fetch_url",
        "register_source",
        "task",
        "think",
        "compact_conversation",
    }
)

# Deep-code profile: planning + workspace + code nav + sandbox exec.
DEEP_DISCOVERY_CORE_CODE: frozenset[str] = frozenset(
    {
        "write_todos",
        "read_todos",
        "update_todo",
        "write_file",
        "read_file",
        "edit_file",
        "ls",
        "grep",
        "glob",
        "delete_file",
        "repo_map",
        "code_search",
        "find_symbol",
        "run_python",
        "run_python_file",
        "run_shell",
        "task",
        "task_batch",
        "think",
        "compact_conversation",
    }
)


def _resolve_discovery_core(
    discovery_core: str | Sequence[str] | None,
) -> list[str]:
    if discovery_core is None or discovery_core == "research":
        return list(DEEP_DISCOVERY_CORE_TOOLS)
    if discovery_core == "research-slim":
        return list(DEEP_DISCOVERY_CORE_SLIM)
    if discovery_core == "code":
        return list(DEEP_DISCOVERY_CORE_CODE)
    return [str(x) for x in discovery_core]


DEEP_AGENT_INSTRUCTIONS = """\
You are a loomable deep agent: a long-horizon agent built solely on loomable
(not LangGraph). You beat generic deep-agent scaffolds by planning, offloading
evidence to a shared workspace, verifying sources, and delegating in parallel.

Operating rules:
1. Start by calling write_todos with a concrete checklist for the user goal.
   Keep exactly one item in_progress. Mark items completed as you finish them.
2. Prefer writing intermediate notes, research dumps, and drafts to workspace
   files (write_file / edit_file). Do not paste huge blobs into chat.
3. Use read_file / grep / glob to pull back only the slices you need.
   For large/offloaded files use read_file(path, offset=N, limit=80) or grep —
   never reload an entire .offload dump into chat.
4. Use web_search until you have 3–5 solid primary sources (or the budget is
   tight), then fetch_url / extract_text for full pages. Large tool results may
   be offloaded to .offload/ — read those files in slices. Prefer primary docs
   over search-fallback snippets.
5. Register important sources with register_source; call verify_source on key
   URLs; link important findings with register_claim(claim, source_id, quote).
   End deliverables with format_bibliography (or paste its output into the report).
6. For visual evidence: search_tools/activate_tool for image tools, then
   discover_images on a source page, fetch_image + analyze_image; store notes
   under images/.
7. Use task / task_batch to spawn specialists for isolated research or drafting
   when that work would bloat your own context. Pass a crisp task description.
   Prefer task_batch when researching multiple angles in parallel.
   Use subagent_type to pick a named specialist when available; otherwise
   general-purpose. Specialists share this workspace — they can write files
   you can read.
8. Use think for brief private reasoning; use memory when durable facts should
   survive beyond this session (if the memory tool is available).
9. When chat context feels heavy, call compact_conversation with a short
   checkpoint summary so the workspace remains the source of truth.
10. Skills: if a matching skill is listed (e.g. research), call load_skill(name)
   before following its workflow. Use search_skills when unsure.
11. Finish by writing the user-facing deliverable under reports/ with write_file,
   then mark todos completed in at most one update_todo call and STOP. Do not
   keep updating todos after the report exists — emit your final answer
   summarizing the deliverable path. Do not stop after format_bibliography alone —
   the Markdown report file under reports/ is required.

Quality bar: be concrete, cite verified sources, and verify the deliverable
against the original goal before stopping.
"""


def make_compact_conversation_tool(
    workspace: str | Path,
    *,
    store: Any | None = None,
) -> FunctionTool:
    """Tool that archives a context checkpoint into the shared workspace."""

    root = Path(workspace)

    async def compact_conversation(summary: str) -> str:
        """Save a context checkpoint to the workspace and continue from files.

        Call when the conversation is getting long. Pass a short summary of
        decisions, open questions, and key file paths. Prefer workspace files
        over replaying chat history after this.
        """
        text = (summary or "").strip()
        if not text:
            return "Error: summary is required"
        stamp = time.strftime("%Y%m%dT%H%M%S")
        rel = f".offload/context_checkpoint_{stamp}.md"
        body = (
            f"# Context checkpoint ({stamp})\n\n"
            f"{text}\n\n"
            "_Continue from workspace files; do not reload full chat history._\n"
        )
        if store is not None and hasattr(store, "write"):
            written = store.write(rel, body)
            if written is None:
                (root / ".offload").mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(body, encoding="utf-8")
        else:
            (root / ".offload").mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(body, encoding="utf-8")
        return (
            f"Checkpoint saved to workspace:{rel}. "
            "Treat workspace files as source of truth; keep subsequent replies short."
        )

    return FunctionTool(
        compact_conversation,
        name="compact_conversation",
        description=(
            "Archive a short context checkpoint to .offload/ and continue from "
            "workspace files when the chat is getting long."
        ),
        idempotent=False,
    )


def _load_memory_files(paths: Sequence[str | Path] | None) -> str:
    """Load always-on project memory files (e.g. AGENTS.md) into instructions."""
    if not paths:
        return ""
    chunks: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            chunks.append(f"### {path.name}\n{text}")
    if not chunks:
        return ""
    return "Project memory (always follow):\n\n" + "\n\n".join(chunks)


@dataclass
class SpecialistSpec:
    """Named specialist available via ``task(subagent_type=...)``."""

    name: str
    description: str = ""
    instructions: str = ""
    role: str = ""
    tools: list[Any] | None = None
    model: Any = None
    modalities: str | None = None
    max_tool_iterations: int | None = None


def make_research_accept(
    workspace: str | Path,
    *,
    min_sources: int = 1,
    report_dir: str = "reports",
) -> Any:
    """Verifier: report under ``reports/`` + at least ``min_sources`` citations."""

    root = Path(workspace)
    report_dir = (report_dir or "reports").strip().strip("/") or "reports"
    min_sources = max(0, int(min_sources))

    def _check(output: Any, ctx: Any) -> VerdictResult:  # noqa: ANN401, ARG001
        reports = root / report_dir
        has_report = False
        if reports.is_dir():
            has_report = any(p.is_file() for p in reports.rglob("*"))
        sources: list[Any] = []
        src_path = root / "sources.json"
        if src_path.is_file():
            try:
                data = json.loads(src_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    sources = data
                elif isinstance(data, dict) and isinstance(data.get("sources"), list):
                    sources = data["sources"]
            except (OSError, json.JSONDecodeError, TypeError):
                sources = []
        missing: list[str] = []
        if not has_report:
            missing.append(f"write a Markdown report under {report_dir}/")
        if len(sources) < min_sources:
            missing.append(f"register at least {min_sources} source(s) via register_source")
        if missing:
            return VerdictResult(
                ok=False,
                detail="Research incomplete: " + "; ".join(missing),
            )
        return VerdictResult(ok=True)

    return _check


def make_task_tools(
    *,
    model: Any,
    tools: list[Any] | None = None,
    modalities: str | None = "text",
    default_role: str = "Specialist",
    max_tool_iterations: int = 20,
    token_budget: int = 64_000,
    max_run_tokens: int = 0,
    specialists: dict[str, SpecialistSpec] | None = None,
    skills: list[Any] | None = None,
    tool_hooks: list[Any] | None = None,
    memory: Any | None = None,
    note_store: Any | None = None,
    memory_tool: bool = False,
    think_tool: bool = False,
    resilience: Any | None = None,
    tool_timeout: float | None = None,
    tool_concurrency: int | None = None,
    enable_batch: bool = True,
    discovery: bool | None = None,
    discovery_core_tools: Sequence[str] | None = None,
    defer_local_tools: bool | None = None,
    lazy_mcp: bool | None = None,
    activation_allowlist: Sequence[str] | None = None,
    activation_denylist: Sequence[str] | None = None,
) -> list[FunctionTool]:
    """Build ``task`` (+ optional ``task_batch``) tools for deep agents.

    Passing ``discovery=True`` (as :func:`create_deep_agent` does by default)
    wires progressive capability discovery into every spawned specialist too,
    so a large shared toolset doesn't blow the specialist's schema budget.
    """

    registry: dict[str, SpecialistSpec | None] = {"general-purpose": None}
    for key, spec in (specialists or {}).items():
        slug = (key or "").strip().lower().replace(" ", "-")
        if not slug:
            continue
        registry[slug] = spec

    roster = ", ".join(sorted(registry.keys()))

    async def _run_one(
        description: str,
        *,
        role: str = "",
        subagent_type: str = "general-purpose",
    ) -> str:
        key = (subagent_type or "general-purpose").strip().lower().replace(" ", "-")
        if key not in registry:
            return (
                f"Unknown subagent_type={subagent_type!r}. "
                f"Available: {roster}"
            )
        spec = registry[key]
        use_model = model
        use_tools = list(tools or [])
        use_modalities = modalities
        use_iters = max_tool_iterations
        instructions = None
        use_role = (role or default_role).strip() or default_role
        if spec is not None:
            use_model = spec.model or model
            if spec.tools is not None:
                use_tools = list(spec.tools)
            if spec.modalities is not None:
                use_modalities = spec.modalities
            if spec.max_tool_iterations is not None:
                use_iters = spec.max_tool_iterations
            if spec.instructions:
                instructions = spec.instructions
            if not role:
                use_role = (spec.role or spec.name or default_role).strip() or default_role
            if spec.description and instructions is None:
                instructions = (
                    f"You are {use_role}. Specialty: {spec.description}. "
                    "Be concise and factual. Write artifacts into the shared workspace."
                )
        return await spawn_specialist(
            model=use_model,
            role=use_role,
            task=description,
            instructions=instructions,
            tools=use_tools,
            modalities=use_modalities,
            skills=skills,
            tool_hooks=tool_hooks,
            memory=memory,
            note_store=note_store,
            memory_tool=memory_tool,
            think_tool=think_tool,
            resilience=resilience,
            tool_timeout=tool_timeout,
            tool_concurrency=tool_concurrency,
            max_tool_iterations=use_iters,
            token_budget=token_budget,
            max_run_tokens=max_run_tokens,
            discovery=discovery,
            discovery_core_tools=list(discovery_core_tools) if discovery_core_tools else None,
            defer_local_tools=defer_local_tools,
            lazy_mcp=lazy_mcp,
            activation_allowlist=list(activation_allowlist) if activation_allowlist else None,
            activation_denylist=list(activation_denylist) if activation_denylist else None,
        )

    async def task(
        description: str,
        role: str = "",
        subagent_type: str = "general-purpose",
    ) -> str:
        """Delegate an isolated sub-task to a fresh specialist agent.

        Use for research, drafting, or analysis that should not pollute the
        parent context. ``description`` must be self-contained. Optional
        ``role`` names the specialist. ``subagent_type`` selects a named
        specialist from the registry (default: general-purpose).
        """
        return await _run_one(
            description, role=role, subagent_type=subagent_type
        )

    out: list[FunctionTool] = [
        FunctionTool(
            task,
            name="task",
            description=(
                "Spawn a specialist for isolated work. "
                f"subagent_type one of: {roster}."
            ),
            idempotent=False,
        )
    ]

    if enable_batch:

        async def task_batch(tasks_json: str) -> str:
            """Run multiple specialist tasks in parallel and return labeled results.

            ``tasks_json`` is a JSON list of objects with keys:
            ``description`` (required), ``role``, ``subagent_type``.
            """
            try:
                items = json.loads(tasks_json or "[]")
            except json.JSONDecodeError as exc:
                return f"Error: invalid tasks_json: {exc}"
            if not isinstance(items, list) or not items:
                return "Error: tasks_json must be a non-empty JSON list"
            if len(items) > 8:
                return "Error: task_batch supports at most 8 tasks"

            async def _one(idx: int, item: Any) -> dict[str, Any]:
                if not isinstance(item, dict):
                    return {"index": idx, "ok": False, "error": "item must be object"}
                desc = str(item.get("description") or "").strip()
                if not desc:
                    return {"index": idx, "ok": False, "error": "description required"}
                try:
                    text = await _run_one(
                        desc,
                        role=str(item.get("role") or ""),
                        subagent_type=str(
                            item.get("subagent_type") or "general-purpose"
                        ),
                    )
                    return {"index": idx, "ok": True, "text": text}
                except Exception as exc:  # noqa: BLE001
                    return {"index": idx, "ok": False, "error": str(exc)}

            results = await asyncio.gather(
                *[_one(i, item) for i, item in enumerate(items)]
            )
            return json.dumps({"results": list(results)}, ensure_ascii=False)

        out.append(
            FunctionTool(
                task_batch,
                name="task_batch",
                description=(
                    "Fan out multiple specialist tasks in parallel. "
                    "Pass JSON list of {description, role?, subagent_type?}."
                ),
                idempotent=False,
            )
        )

    return out


def make_task_tool(
    *,
    model: Any,
    tools: list[Any] | None = None,
    modalities: str | None = "text",
    default_role: str = "Specialist",
    max_tool_iterations: int = 20,
    token_budget: int = 64_000,
    max_run_tokens: int = 0,
    **kwargs: Any,
) -> FunctionTool:
    """LangGraph-style ``task`` tool — spawn an ephemeral specialist and return text."""
    tools_list = make_task_tools(
        model=model,
        tools=tools,
        modalities=modalities,
        default_role=default_role,
        max_tool_iterations=max_tool_iterations,
        token_budget=token_budget,
        max_run_tokens=max_run_tokens,
        enable_batch=False,
        **kwargs,
    )
    return tools_list[0]


def _modalities_include_image(modalities: str | None) -> bool:
    if not modalities:
        return False
    parts = {
        p.strip().lower()
        for p in modalities.replace(",", "+").replace("|", "+").split("+")
        if p.strip()
    }
    return "image" in parts


def create_deep_agent(
    model: Any,
    *,
    tools: Sequence[Any] | None = None,
    instructions: str | None = None,
    workspace: str | Path = "./.deep_workspace",
    subagents: list[Any] | None = None,
    specialists: dict[str, SpecialistSpec] | dict[str, Any] | None = None,
    session_id: str | None = None,
    session_store: Any | None = None,
    memory_backend: Any | None = None,
    note_store: Any | None = None,
    memory_tool: bool = False,
    memory: Any = None,
    knowledge: list[str] | None = None,
    retrievers: Sequence[Any] | None = None,
    knowledge_base: Any = None,
    embedder: Any = None,
    skills: Sequence[str | Path] | None = None,
    profile: str = "general",
    repo: str | Path | None = None,
    code_index: Any | None = None,
    user_id: str | None = None,
    scopes: dict[str, str] | None = None,
    require_confirmation: list[str] | None = None,
    require_tools: list[str] | None = None,
    strict_require_tools: bool = False,
    web_search: bool = True,
    search_provider: str = "duckduckgo",
    search_api_key: str | None = None,
    url_fetch: bool = True,
    url_max_length: int = 8_000,
    citations: bool = True,
    images: bool | None = None,
    documents: bool = False,
    code_exec: bool = False,
    shell: bool = False,
    sandbox: Any | None = None,
    sandbox_backend: str = "subprocess",
    offload_large_tools: bool = True,
    offload_threshold_tokens: int = 3_000,
    memory_files: Sequence[str | Path] | None = None,
    think_tool: bool = True,
    plan_tool: bool = False,
    enable_task_tool: bool = True,
    enable_task_batch: bool = True,
    task_tools: Sequence[Any] | None = None,
    discovery: bool = True,
    discovery_core: str | Sequence[str] | None = "research",
    mode: str | None = None,
    dispatch: str = "reuse",
    accept: Any = None,
    board: bool = True,
    max_rounds: int | None = None,
    max_plan_steps: int = 8,
    checkpointer: Any = None,
    max_tool_iterations: int = 40,
    memory_window: int = 16,
    compaction_threshold: int = 32,
    use_llm_summarizer: bool = True,
    loop_repeat_threshold: int = 6,
    token_budget: int = 128_000,
    max_run_tokens: int = 0,
    tool_concurrency: int = 4,
    tool_timeout: float = 60.0,
    resilience: Any | None = ...,  # type: ignore[assignment]
    max_delegations: int | None = 12,
    max_depth: int = 4,
    name: str = "deep-agent",
    goal: str = "Complete hard, long-horizon tasks with planning and delegation",
    modalities: str = "text+image",
    debug: bool = False,
    **agent_kwargs: Any,
) -> Agent:
    """Build a deep agent harness on top of :class:`~loomable.agent.builder.Agent`.

    ``profile="research"`` loads the bundled research skill (any topic) and
    turns on research kits + deliverable gates. Prefer that over a second
    factory — research is a skill, not a separate agent type.

    ``profile="code"`` loads the ``coding`` skill, indexes ``repo=`` (zvec
    :class:`~loomable.codeindex.CodeIndex`), attaches :class:`~loomable.toolkits.CodeTools`,
    and enables sandbox ``code_exec`` / ``shell``. Pass a prebuilt
    ``code_index=`` or a custom embedder/store via :meth:`CodeIndex.build`.

    ``discovery_core`` controls the always-advertised tool allowlist when
    ``discovery=True`` (default): ``"research"`` (correctness-first),
    ``"research-slim"`` (smaller schema budget), ``"code"`` (nav + sandbox),
    or an explicit name sequence. Workspace FS is local-only in the beta cut.

    ``code_exec`` / ``shell`` attach :class:`~loomable.toolkits.PythonTools` /
    :class:`~loomable.toolkits.ShellTools` on a shared sandbox (default
    subprocess under ``workspace/.sandbox``). Pass ``sandbox=`` or
    ``sandbox_backend="docker"`` for stronger isolation. Browser automation is
    via MCP (e.g. Lightpanda) + the bundled ``browser`` skill — not a built-in
    CDP client.

    ``knowledge_base`` is a vector store (or sources ingested into one) and
    becomes ``search_*`` tools on this Agent. ``retrievers=`` attaches extra
    search tools. Same kwargs as :class:`~loomable.agent.builder.Agent`.
    """
    from loomable.skills import resolve_skills
    from loomable.toolkits.citation_tools import CitationTools
    from loomable.toolkits.todo_tools import TodoTools
    from loomable.toolkits.workspace_tools import WorkspaceTools

    profile_key = (profile or "general").strip().lower()
    if profile_key not in {"general", "research", "code"}:
        raise ValueError(
            f"profile must be 'general', 'research', or 'code', got {profile!r}"
        )

    skill_list = list(skills or [])
    if profile_key == "research" and "research" not in {
        str(s).strip().lower() if not isinstance(s, Path) else s.name.lower()
        for s in skill_list
    }:
        skill_list.append("research")
    if profile_key == "code" and "coding" not in {
        str(s).strip().lower() if not isinstance(s, Path) else s.name.lower()
        for s in skill_list
    }:
        skill_list.append("coding")
    resolved_skills = resolve_skills(skill_list) or None

    # Research profile defaults (caller kwargs still win via explicit args below).
    if profile_key == "research":
        if require_tools is None:
            require_tools = ["write_file:reports/", "register_source"]
        if accept is None:
            accept = make_research_accept(workspace, min_sources=1)
        agent_kwargs.setdefault("retry_on_failure", True)
        agent_kwargs.setdefault("max_verify_retries", 1)
        if name == "deep-agent":
            name = "research-agent"
        if goal == "Complete hard, long-horizon tasks with planning and delegation":
            goal = (
                "Research any topic thoroughly: search, fetch, cite, "
                "analyze images, deliver briefs under reports/"
            )

    if profile_key == "code":
        # Deep-code defaults: sandbox on, code discovery core, lighter research kits.
        code_exec = True
        shell = True
        if discovery_core == "research":
            discovery_core = "code"
        web_search = False if web_search is True else web_search
        url_fetch = False if url_fetch is True else url_fetch
        citations = False if citations is True else citations
        if images is None:
            images = False
        if name == "deep-agent":
            name = "code-agent"
        if goal == "Complete hard, long-horizon tasks with planning and delegation":
            goal = (
                "Understand and change the target codebase: map, search, "
                "edit, and verify with sandboxed tests"
            )
        modalities = "text" if modalities == "text+image" else modalities

    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)

    todo_kit = TodoTools(workspace=root)
    workspace_kit = WorkspaceTools(root=root)
    citation_kit = CitationTools(workspace=root) if citations else None

    bundled: list[Any] = [todo_kit, workspace_kit]
    if citation_kit is not None:
        bundled.append(citation_kit)

    shared_for_task: list[Any] = [todo_kit, workspace_kit]
    if citation_kit is not None:
        shared_for_task.append(citation_kit)

    missing: list[str] = []

    # Deep code: index repo (zvec by default) and attach CodeTools.
    resolved_index = code_index
    if resolved_index is None and repo is not None:
        try:
            from loomable.codeindex import CodeIndex

            persist = root / ".loomable" / "codeindex_zvec"
            resolved_index = CodeIndex.build_sync(
                repo,
                embedder=embedder,
                persist_path=persist,
            )
        except Exception as exc:  # noqa: BLE001
            missing.append(f"code_index ({exc})")
            logger.warning("create_deep_agent: CodeIndex build failed: %s", exc)
            resolved_index = None
    if resolved_index is not None:
        try:
            from loomable.toolkits.code_tools import CodeTools

            code_kit = CodeTools(resolved_index)
            bundled.append(code_kit)
            shared_for_task.append(CodeTools(resolved_index))
            # Seed knowledge with a compact map if caller did not pass knowledge.
            # Agent.build() requires embedder= whenever knowledge= is set.
            if knowledge is None:
                knowledge = [
                    resolved_index.repo_map(max_entries=60),
                    *resolved_index.as_knowledge(max_chunks=12),
                ]
                if embedder is None:
                    embedder = resolved_index.embedder
        except Exception as exc:  # noqa: BLE001
            missing.append(f"code_tools ({exc})")
            logger.warning("create_deep_agent: CodeTools unavailable: %s", exc)

    if web_search:
        try:
            from loomable.toolkits.web_search import WebSearchTools

            search_kwargs: dict[str, Any] = {"provider": search_provider}
            if search_api_key:
                search_kwargs["api_key"] = search_api_key
            search_kit = WebSearchTools(**search_kwargs)
            bundled.append(search_kit)
            shared_for_task.append(WebSearchTools(**search_kwargs))
        except Exception as exc:  # noqa: BLE001
            missing.append(f"web_search ({exc})")
            logger.warning("create_deep_agent: web_search unavailable: %s", exc)

    if url_fetch:
        try:
            from loomable.toolkits.url_tools import URLTools

            url_kit = URLTools(max_length=url_max_length)
            bundled.append(url_kit)
            shared_for_task.append(URLTools(max_length=url_max_length))
        except Exception as exc:  # noqa: BLE001
            missing.append(f"url_fetch ({exc})")
            logger.warning("create_deep_agent: url_fetch unavailable: %s", exc)

    use_images = _modalities_include_image(modalities) if images is None else bool(images)
    if use_images:
        try:
            from loomable.toolkits.image_tools import ImageTools

            image_kit = ImageTools(workspace=root, model=model)
            bundled.append(image_kit)
            shared_for_task.append(ImageTools(workspace=root, model=model))
        except Exception as exc:  # noqa: BLE001
            missing.append(f"images ({exc})")
            logger.warning("create_deep_agent: ImageTools unavailable: %s", exc)

    if documents:
        try:
            from loomable.toolkits.pdf_tools import PDFTools

            pdf_kit = PDFTools()
            bundled.append(pdf_kit)
            shared_for_task.append(PDFTools())
        except Exception as exc:  # noqa: BLE001
            missing.append(f"documents ({exc})")
            logger.warning("create_deep_agent: PDFTools unavailable: %s", exc)

    hitl = list(require_confirmation) if require_confirmation else []

    # Shared execution sandbox for Python / shell (soft isolation by default).
    exec_sandbox = sandbox
    if exec_sandbox is None and (code_exec or shell):
        from loomable.sandbox import make_sandbox

        try:
            exec_sandbox = make_sandbox(
                str(root / ".sandbox"),
                timeout=30.0,
                backend=sandbox_backend,
            )
        except Exception as exc:  # noqa: BLE001
            missing.append(f"sandbox ({exc})")
            logger.warning("create_deep_agent: sandbox unavailable: %s", exc)
            exec_sandbox = None

    if code_exec:
        try:
            from loomable.toolkits.python_tools import PythonTools

            py_kit = PythonTools(sandbox=exec_sandbox, working_dir=str(root), timeout=30)
            bundled.append(py_kit)
            shared_for_task.append(
                PythonTools(sandbox=exec_sandbox, working_dir=str(root), timeout=30)
            )
            for name_ in ("run_python", "run_python_file"):
                if name_ not in hitl:
                    hitl.append(name_)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"code_exec ({exc})")
            logger.warning("create_deep_agent: PythonTools unavailable: %s", exc)

    if shell:
        try:
            from loomable.toolkits.shell_tools import ShellTools

            sh_kit = ShellTools(sandbox=exec_sandbox, working_dir=str(root), timeout=30)
            bundled.append(sh_kit)
            shared_for_task.append(
                ShellTools(sandbox=exec_sandbox, working_dir=str(root), timeout=30)
            )
            if "run_shell" not in hitl:
                hitl.append("run_shell")
        except Exception as exc:  # noqa: BLE001
            missing.append(f"shell ({exc})")
            logger.warning("create_deep_agent: ShellTools unavailable: %s", exc)

    if task_tools is None:
        task_tools = list(shared_for_task)

    # Normalize specialist registry
    specialist_registry: dict[str, SpecialistSpec] = {}
    for key, value in (specialists or {}).items():
        if isinstance(value, SpecialistSpec):
            specialist_registry[key] = value
        elif isinstance(value, dict):
            specialist_registry[key] = SpecialistSpec(name=key, **value)
        else:
            logger.warning("create_deep_agent: ignoring specialist %r", key)

    if resilience is ...:
        resilience = RetryPolicy(max_attempts=3, per_call_timeout=60.0)

    # Merge offload post-hook with any caller tool_hooks (needed before task tools).
    existing_hooks = list(agent_kwargs.pop("tool_hooks", None) or [])
    if offload_large_tools:
        existing_hooks.append(
            make_workspace_offload_hook(
                root,
                threshold_tokens=offload_threshold_tokens,
                store=workspace_kit.store,
            )
        )

    bundled.append(
        make_compact_conversation_tool(root, store=workspace_kit.store)
    )

    if enable_task_tool:
        bundled.extend(
            make_task_tools(
                model=model,
                tools=list(task_tools) if task_tools is not None else None,
                modalities=modalities,
                max_tool_iterations=min(20, max_tool_iterations),
                token_budget=min(token_budget, 64_000),
                max_run_tokens=max_run_tokens,
                specialists=specialist_registry or None,
                skills=list(resolved_skills) if resolved_skills else None,
                tool_hooks=list(existing_hooks) if existing_hooks else None,
                memory=memory,
                note_store=note_store,
                memory_tool=memory_tool,
                think_tool=False,
                resilience=resilience,
                tool_timeout=tool_timeout,
                tool_concurrency=min(tool_concurrency, 4) if tool_concurrency else None,
                enable_batch=enable_task_batch,
                # Specialists share the same (large) research/toolkit surface as
                # the parent — wire discovery so their schema budget stays small.
                discovery=discovery if discovery else None,
                discovery_core_tools=(
                    _resolve_discovery_core(discovery_core) if discovery else None
                ),
                defer_local_tools=True if discovery else None,
                lazy_mcp=agent_kwargs.get("lazy_mcp"),
                activation_allowlist=agent_kwargs.get("activation_allowlist"),
                activation_denylist=agent_kwargs.get("activation_denylist"),
            )
        )

    if tools:
        bundled.extend(list(tools))

    prompt = DEEP_AGENT_INSTRUCTIONS
    memory_block = _load_memory_files(memory_files)
    if memory_block:
        prompt = f"{prompt}\n\n{memory_block}"
    if specialist_registry:
        lines = [
            f"- {k}: {v.description or v.role or v.name}"
            for k, v in specialist_registry.items()
        ]
        prompt = (
            f"{prompt}\n\nNamed specialists (pass subagent_type):\n"
            + "\n".join(lines)
        )
    if instructions:
        prompt = f"{prompt}\n\nAdditional instructions:\n{instructions.strip()}"
    if missing:
        prompt = (
            f"{prompt}\n\nRuntime note: some optional toolkits failed to load: "
            + ", ".join(missing)
            + "."
        )

    kwargs: dict[str, Any] = dict(
        model=model,
        name=name,
        role="Deep Agent",
        goal=goal,
        instructions=prompt,
        tools=bundled,
        subagents=subagents,
        skills=resolved_skills,
        discovery=discovery,
        discovery_core_tools=_resolve_discovery_core(discovery_core),
        # Progressive skills: metadata in prompt; load_skill for full body.
        # Pass eager_skills=True via agent_kwargs to inject full skill bodies up front.
        eager_skills=agent_kwargs.pop("eager_skills", None),
        # Lazy MCP / activation policy: Agent already defaults lazy_mcp=True
        # when discovery is on; these just let callers override explicitly.
        lazy_mcp=agent_kwargs.pop("lazy_mcp", None),
        activation_allowlist=agent_kwargs.pop("activation_allowlist", None),
        activation_denylist=agent_kwargs.pop("activation_denylist", None),
        tool_namespaces=agent_kwargs.pop("tool_namespaces", None),
        session_id=session_id,
        session_store=session_store,
        memory_backend=memory_backend,
        note_store=note_store,
        memory_tool=memory_tool,
        memory=memory,
        user_id=user_id,
        scopes=scopes,
        knowledge=knowledge,
        retrievers=list(retrievers) if retrievers else None,
        knowledge_base=knowledge_base,
        embedder=embedder,
        think_tool=think_tool,
        plan_tool=plan_tool,
        require_confirmation=hitl or None,
        require_tools=require_tools,
        strict_require_tools=strict_require_tools or None,
        tool_hooks=existing_hooks or None,
        max_tool_iterations=max_tool_iterations,
        memory_window=memory_window,
        compaction_threshold=compaction_threshold,
        use_llm_summarizer=use_llm_summarizer,
        loop_repeat_threshold=loop_repeat_threshold,
        token_budget=token_budget,
        max_run_tokens=max_run_tokens,
        tool_concurrency=tool_concurrency,
        tool_timeout=tool_timeout,
        resilience=resilience,
        max_delegations=max_delegations,
        max_depth=max_depth,
        modalities=modalities,
        debug=debug,
        mode=mode,
        accept=accept,
    )
    if mode != "case":
        case_only = []
        if dispatch != "reuse":
            case_only.append("dispatch")
        if max_rounds is not None:
            case_only.append("max_rounds")
        if max_plan_steps != 8:
            case_only.append("max_plan_steps")
        if checkpointer is not None:
            case_only.append("checkpointer")
        if case_only:
            from loomable.agent.errors import AgentConfigError

            raise AgentConfigError(
                f"{', '.join(case_only)} only apply with "
                "create_deep_agent(..., mode='case') (or construct Case directly)."
            )
    else:
        kwargs["dispatch"] = dispatch
        kwargs["board"] = board
        kwargs["max_rounds"] = max_rounds
        kwargs["max_plan_steps"] = max_plan_steps
        kwargs["checkpointer"] = checkpointer
    kwargs.update(agent_kwargs)
    # Drop Nones so Agent defaults apply cleanly
    return Agent(**{k: v for k, v in kwargs.items() if v is not None})
