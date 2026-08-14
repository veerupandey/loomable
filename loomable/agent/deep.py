"""Deep Agent harness — LangGraph-style long-horizon agent on loomable.

Four pillars (same idea as langchain-ai/deepagents):

1. **Planning** — ``TodoTools`` (``write_todos`` / ``read_todos`` / ``update_todo``)
2. **Workspace FS** — ``WorkspaceTools`` (ls/read/write/edit/glob/grep) to offload context
3. **Subagents** — ``task`` tool (+ optional named ``subagents=`` / Case spawn)
4. **Context engineering** — think/plan tools, memory, compaction, long tool loops

Usage::

    from loomable.agent.deep import create_deep_agent

    agent = create_deep_agent(model=provider, workspace=\"./.deep_workspace\")
    result = await agent.arun(\"Research X and write a brief to /reports/x.md\")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from loomable.agent.builder import Agent
from loomable.agent.delegation import spawn_specialist
from loomable.agent.tools import FunctionTool

__all__ = [
    "DEEP_AGENT_INSTRUCTIONS",
    "create_deep_agent",
    "make_task_tool",
]

DEEP_AGENT_INSTRUCTIONS = """\
You are a deep agent: you solve hard, long-horizon tasks by planning, offloading
context to the workspace filesystem, and delegating focused sub-work.

Operating rules:
1. Start by calling write_todos with a concrete checklist for the user goal.
   Keep exactly one item in_progress. Mark items completed as you finish them.
2. Prefer writing intermediate notes, research dumps, and drafts to workspace
   files (write_file / edit_file). Do not paste huge blobs into chat.
3. Use read_file / grep / glob to pull back only the slices you need.
4. Use the task tool to spawn a specialist for isolated research or drafting
   when that work would bloat your own context. Pass a crisp task description.
5. Use think for brief private reasoning; use memory when durable facts should
   survive beyond this session (if the memory tool is available).
6. Finish by producing the user-facing deliverable (and update todos to completed).

Quality bar: be concrete, cite sources from tools when available, and verify
the deliverable against the original goal before stopping.
"""


def make_task_tool(
    *,
    model: Any,
    tools: list[Any] | None = None,
    modalities: str | None = "text",
    default_role: str = "Specialist",
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
        )

    return FunctionTool(task, name="task", idempotent=False)


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
    knowledge: list[str] | None = None,
    embedder: Any = None,
    web_search: bool = True,
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
    use_llm_summarizer: bool = False,
    name: str = "deep-agent",
    goal: str = "Complete hard, long-horizon tasks with planning and delegation",
    modalities: str = "text",
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
    web_search:
        Include :class:`~loomable.toolkits.web_search.WebSearchTools` when available.
    enable_task_tool:
        Register the general-purpose ``task`` subagent tool.
    mode:
        Pass ``\"case\"`` for Case plan→dispatch→synthesize→accept spine.
    max_tool_iterations:
        Deep work needs a higher ceiling than the Agent default.
    """
    from loomable.toolkits.todo_tools import TodoTools
    from loomable.toolkits.workspace_tools import WorkspaceTools

    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)

    bundled: list[Any] = [
        TodoTools(workspace=root),
        WorkspaceTools(root=root),
    ]

    if web_search:
        try:
            from loomable.toolkits.web_search import WebSearchTools

            bundled.append(WebSearchTools())
            if task_tools is None:
                task_tools = [WebSearchTools()]
        except Exception:  # noqa: BLE001 — optional deps / network backends
            pass

    if enable_task_tool:
        bundled.append(
            make_task_tool(
                model=model,
                tools=list(task_tools) if task_tools is not None else None,
                modalities=modalities,
            )
        )

    if tools:
        bundled.extend(list(tools))

    prompt = DEEP_AGENT_INSTRUCTIONS
    if instructions:
        prompt = f"{prompt}\n\nAdditional instructions:\n{instructions.strip()}"

    kwargs: dict[str, Any] = dict(
        model=model,
        name=name,
        role="Deep Agent",
        goal=goal,
        instructions=prompt,
        tools=bundled,
        subagents=subagents,
        session_id=session_id,
        session_store=session_store,
        memory_backend=memory_backend,
        note_store=note_store,
        memory_tool=memory_tool,
        knowledge=knowledge,
        embedder=embedder,
        think_tool=think_tool,
        plan_tool=plan_tool,
        max_tool_iterations=max_tool_iterations,
        memory_window=memory_window,
        compaction_threshold=compaction_threshold,
        use_llm_summarizer=use_llm_summarizer,
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
