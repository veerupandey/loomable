"""Deep Agent harness — LangGraph-style long-horizon agent on loomable.

Four pillars (same idea as langchain-ai/deepagents), plus research wins:

1. **Planning** — ``TodoTools`` (``write_todos`` / ``read_todos`` / ``update_todo``)
2. **Workspace FS** — ``WorkspaceTools`` (ls/read/write/edit/glob/grep) to offload context
3. **Subagents** — ``task`` tool (+ optional named ``subagents=`` / Case spawn)
4. **Context engineering** — think/plan tools, memory, compaction, large-tool offload

Research defaults also bundle web search, URL fetch, image analyze, and citations.

Usage::

    from loomable.agent.deep import create_deep_agent, create_research_agent

    agent = create_research_agent(model=provider, workspace=\"./.deep_workspace\")
    result = await agent.arun(\"Research X and write a brief to /reports/x.md\")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from loomable.agent.builder import Agent
from loomable.agent.delegation import spawn_specialist
from loomable.agent.offload import make_workspace_offload_hook
from loomable.agent.tools import FunctionTool

__all__ = [
    "DEEP_AGENT_INSTRUCTIONS",
    "create_deep_agent",
    "create_research_agent",
    "make_task_tool",
]

logger = logging.getLogger("loomable.agent.deep")

DEEP_AGENT_INSTRUCTIONS = """\
You are a deep agent: you solve hard, long-horizon tasks by planning, offloading
context to the workspace filesystem, and delegating focused sub-work.

Operating rules:
1. Start by calling write_todos with a concrete checklist for the user goal.
   Keep exactly one item in_progress. Mark items completed as you finish them.
2. Prefer writing intermediate notes, research dumps, and drafts to workspace
   files (write_file / edit_file). Do not paste huge blobs into chat.
3. Use read_file / grep / glob to pull back only the slices you need.
   For large/offloaded files use read_file(path, offset=N, limit=80) or grep —
   never reload an entire .offload dump into chat.
4. Use web_search to find sources (at most 2 searches), then fetch_url /
   extract_text for full pages. Large tool results may be offloaded to .offload/
   — read those files in slices. Do not keep searching once you have usable URLs.
5. Register important sources with register_source; end deliverables with
   format_bibliography (or paste its output into the report).
6. For visual evidence: discover_images on a source page, then fetch_image +
   analyze_image; store notes under images/.
7. Use the task tool to spawn a specialist for isolated research or drafting
   when that work would bloat your own context. Pass a crisp task description.
   Specialists share this workspace — they can write files you can read.
8. Use think for brief private reasoning; use memory when durable facts should
   survive beyond this session (if the memory tool is available).
9. Finish by writing the user-facing deliverable with write_file (e.g. reports/…),
   then update todos to completed. Do not stop after format_bibliography alone —
   the Markdown report file is required.

Quality bar: be concrete, cite sources from tools when available, and verify
the deliverable against the original goal before stopping.
"""


def make_task_tool(
    *,
    model: Any,
    tools: list[Any] | None = None,
    modalities: str | None = "text",
    default_role: str = "Specialist",
    max_tool_iterations: int = 20,
    token_budget: int = 64_000,
    max_run_tokens: int = 0,
) -> FunctionTool:
    """LangGraph-style ``task`` tool — spawn an ephemeral specialist and return text."""

    async def task(description: str, role: str = "") -> str:
        """Delegate an isolated sub-task to a fresh specialist agent.

        Use for research, drafting, or analysis that should not pollute the
        parent context. ``description`` must be self-contained. Optional
        ``role`` names the specialist (e.g. \"Web Researcher\").
        """
        return await spawn_specialist(
            model=model,
            role=(role or default_role).strip() or default_role,
            task=description,
            tools=list(tools or []),
            modalities=modalities,
            max_tool_iterations=max_tool_iterations,
            token_budget=token_budget,
            max_run_tokens=max_run_tokens,
        )

    return FunctionTool(task, name="task", idempotent=False)


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
    session_id: str | None = None,
    session_store: Any | None = None,
    memory_backend: Any | None = None,
    note_store: Any | None = None,
    memory_tool: bool = False,
    memory: Any = None,
    knowledge: list[str] | None = None,
    embedder: Any = None,
    skills: list[Path] | None = None,
    user_id: str | None = None,
    scopes: dict[str, str] | None = None,
    require_confirmation: list[str] | None = None,
    require_tools: list[str] | None = None,
    web_search: bool = True,
    search_provider: str = "duckduckgo",
    search_api_key: str | None = None,
    url_fetch: bool = True,
    url_max_length: int = 8_000,
    citations: bool = True,
    images: bool | None = None,
    offload_large_tools: bool = True,
    offload_threshold: int = 12_000,
    think_tool: bool = True,
    plan_tool: bool = False,
    enable_task_tool: bool = True,
    task_tools: Sequence[Any] | None = None,
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
    name: str = "deep-agent",
    goal: str = "Complete hard, long-horizon tasks with planning and delegation",
    modalities: str = "text+image",
    debug: bool = False,
    **agent_kwargs: Any,
) -> Agent:
    """Build a deep agent harness on top of :class:`~loomable.agent.builder.Agent`.

    Parameters
    ----------
    model:
        Provider / ModelSpec / ``\"provider:model\"`` string.
    tools:
        Extra tools/toolkits merged after the deep defaults.
    workspace:
        Directory for virtual FS + persisted todos (created if missing).
    web_search / search_provider / search_api_key:
        Bundle web search (DuckDuckGo default, or Tavily with API key).
    url_fetch / url_max_length:
        Bundle URL fetch/extract with a safe length cap (marked truncation).
    citations:
        Include :class:`~loomable.toolkits.citation_tools.CitationTools`.
    images:
        Include :class:`~loomable.toolkits.image_tools.ImageTools` when modalities
        include image (default: auto).
    offload_large_tools:
        Post-hook that saves oversized tool results under ``.offload/``.
    require_tools:
        Tools that must succeed before the run may finish (e.g. ``write_file``).
    token_budget:
        Context window estimate for compaction / bounding (default 128k).
    max_run_tokens:
        Cumulative spend stop. ``0`` = unbounded (default for deep agents).
    enable_task_tool:
        Register the general-purpose ``task`` subagent tool (shares workspace tools).
    mode:
        Pass ``\"case\"`` for Case plan→dispatch→synthesize→accept spine.
    max_tool_iterations:
        Deep work needs a higher ceiling than the Agent default.
    """
    from loomable.toolkits.citation_tools import CitationTools
    from loomable.toolkits.todo_tools import TodoTools
    from loomable.toolkits.workspace_tools import WorkspaceTools

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

    if task_tools is None:
        task_tools = list(shared_for_task)

    if enable_task_tool:
        bundled.append(
            make_task_tool(
                model=model,
                tools=list(task_tools) if task_tools is not None else None,
                modalities=modalities,
                max_tool_iterations=min(20, max_tool_iterations),
                token_budget=min(token_budget, 64_000),
                max_run_tokens=max_run_tokens,
            )
        )

    if tools:
        bundled.extend(list(tools))

    prompt = DEEP_AGENT_INSTRUCTIONS
    if instructions:
        prompt = f"{prompt}\n\nAdditional instructions:\n{instructions.strip()}"
    if missing:
        prompt = (
            f"{prompt}\n\nRuntime note: some optional toolkits failed to load: "
            + ", ".join(missing)
            + "."
        )

    # Merge offload post-hook with any caller tool_hooks.
    existing_hooks = list(agent_kwargs.pop("tool_hooks", None) or [])
    if offload_large_tools:
        existing_hooks.append(
            make_workspace_offload_hook(
                root,
                threshold=offload_threshold,
                store=workspace_kit.store,
            )
        )

    kwargs: dict[str, Any] = dict(
        model=model,
        name=name,
        role="Deep Agent",
        goal=goal,
        instructions=prompt,
        tools=bundled,
        subagents=subagents,
        skills=skills,
        session_id=session_id,
        session_store=session_store,
        memory_backend=memory_backend,
        note_store=note_store,
        memory_tool=memory_tool,
        memory=memory,
        user_id=user_id,
        scopes=scopes,
        knowledge=knowledge,
        embedder=embedder,
        think_tool=think_tool,
        plan_tool=plan_tool,
        require_confirmation=require_confirmation,
        require_tools=require_tools,
        tool_hooks=existing_hooks or None,
        max_tool_iterations=max_tool_iterations,
        memory_window=memory_window,
        compaction_threshold=compaction_threshold,
        use_llm_summarizer=use_llm_summarizer,
        loop_repeat_threshold=loop_repeat_threshold,
        token_budget=token_budget,
        max_run_tokens=max_run_tokens,
        modalities=modalities,
        debug=debug,
        mode=mode,
        dispatch=dispatch,
        accept=accept,
        board=board,
        max_rounds=max_rounds,
        max_plan_steps=max_plan_steps,
        checkpointer=checkpointer,
    )
    kwargs.update(agent_kwargs)
    # Drop Nones so Agent defaults apply cleanly
    return Agent(**{k: v for k, v in kwargs.items() if v is not None})


def create_research_agent(
    model: Any,
    *,
    workspace: str | Path = "./.deep_workspace",
    session_id: str | None = "research",
    memory: Any = None,
    skills: list[Path] | None = None,
    **kwargs: Any,
) -> Agent:
    """Opinionated research deep agent — search, fetch, cite, vision, memory.

    Thin wrapper around :func:`create_deep_agent` with research-ready defaults.
    """
    require_tools = kwargs.pop("require_tools", None)
    if require_tools is None:
        require_tools = ["write_file"]
    return create_deep_agent(
        model,
        workspace=workspace,
        session_id=session_id,
        memory=memory,
        skills=skills,
        web_search=kwargs.pop("web_search", True),
        url_fetch=kwargs.pop("url_fetch", True),
        citations=kwargs.pop("citations", True),
        images=kwargs.pop("images", True),
        modalities=kwargs.pop("modalities", "text+image"),
        use_llm_summarizer=kwargs.pop("use_llm_summarizer", True),
        name=kwargs.pop("name", "research-agent"),
        goal=kwargs.pop(
            "goal",
            "Research topics thoroughly: search, fetch, cite, analyze images, deliver briefs",
        ),
        require_tools=require_tools,
        max_run_tokens=kwargs.pop("max_run_tokens", 0),
        **kwargs,
    )
