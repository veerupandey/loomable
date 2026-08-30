"""loomable.agent.team - Team convenience wrapper for explicit orchestration modes.

:class:`Team` builds a parent :class:`~loomable.agent.builder.Agent` with
auto-generated coordination instructions and ``subagents=members``.

Hard modes ``broadcast`` and ``sequential`` bypass the LLM coordinator and
run members deterministically (enterprise predictability). Soft modes
``coordinate`` and ``route`` keep LLM-driven delegation.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from .builder import Agent
from .errors import AgentConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from loomable.agent.run import RunResult
    from loomable.content import AgentInput
    from loomable.kernel.contracts import ModelProvider

    from .builder import ModelSpec

TeamMode = Literal["coordinate", "route", "broadcast", "sequential", "tasks"]

__all__ = ["Team", "TeamMode"]

_MODE_INSTRUCTIONS: dict[str, str] = {
    "coordinate": (
        "You are a team coordinator. Delegate the task to ALL team members below, "
        "then synthesize their responses into one unified answer.\n\n"
        "Process:\n"
        "1. Call each delegate_to_* tool with the full task (or a focused portion).\n"
        "2. Wait for all members to respond.\n"
        "3. Synthesize the combined insights into a single coherent response."
    ),
    "route": (
        "You are a team router. Select the SINGLE best team member for this task "
        "and delegate only to them.\n\n"
        "Process:\n"
        "1. Analyze the task and pick the best-matched member.\n"
        "2. Call only that member's delegate_to_* tool.\n"
        "3. Return their output (lightly edited if needed)."
    ),
    "broadcast": (
        "You are a team broadcast coordinator. Send the SAME input to ALL team "
        "members and merge their labeled results.\n\n"
        "Process:\n"
        "1. Call every delegate_to_* tool with the same task.\n"
        "2. Present each member's response clearly labeled by role."
    ),
    "sequential": (
        "You are a team pipeline coordinator. Process the task through team members "
        "IN ORDER. Each member builds on the previous member's output.\n\n"
        "Process:\n"
        "1. Call the first member's delegate_to_* tool with the original task.\n"
        "2. Pass each subsequent member the prior member's output as context.\n"
        "3. Return the final member's output (optionally summarize the chain)."
    ),
    "tasks": (
        "You are a team task leader. Maintain a shared task list and execute until "
        "the goal is complete (Agno TeamMode.tasks parity).\n\n"
        "Process:\n"
        "1. Call write_todos with a concrete checklist for the goal "
        "(keep exactly one item in_progress).\n"
        "2. For each pending task, delegate to the best-matched member via "
        "delegate_to_*.\n"
        "3. Mark todos completed as members finish; replan or add tasks if blocked.\n"
        "4. When all todos are completed or cancelled, synthesize one final answer.\n"
        "5. Stop when the goal is done (or you hit the tool-iteration budget)."
    ),
}


def _assemble_team_instructions(
    mode: str,
    members: list[Agent],
    extra: str | None,
) -> str:
    """Build the parent agent instructions for a team mode."""
    if mode not in _MODE_INSTRUCTIONS:
        valid = ", ".join(sorted(_MODE_INSTRUCTIONS))
        raise ValueError(f"Unknown team mode {mode!r}. Valid modes: {valid}")

    from .delegation import format_member_roster

    parts = [_MODE_INSTRUCTIONS[mode], "", "Team members:", format_member_roster(members)]
    if extra:
        parts.extend(["", "Additional instructions:", extra.strip()])
    return "\n".join(parts)


def _member_label(member: Agent, index: int) -> str:
    return getattr(member, "_role", None) or getattr(member, "_name", None) or f"member_{index}"


def _input_as_text(value: Any) -> str:
    """Coerce Workflow/Agent prior output into plain text for Team hard modes.

    Matches :func:`loomable.content.to_agent_input` so ``AgentOutput`` /
    ``RunResult`` chain seamlessly — callers must not parse manually.
    """
    from loomable.content import to_agent_input

    if isinstance(value, str):
        return value
    agent_input = to_agent_input(value)
    chunks: list[str] = []
    for message in agent_input.messages:
        for part in message.parts:
            if part.data is not None:
                try:
                    chunks.append(part.data.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    continue
    return "".join(chunks)


def _coerce_member(member: "Agent | Team", index: int) -> Agent:
    """Normalize Agent or nested Team into a nestable Agent."""
    if isinstance(member, Team):
        return member.as_agent(index=index)
    return member


class Team:
    """Explicit multi-agent orchestration.

    Soft modes (``coordinate``, ``route``, ``tasks``): LLM parent + ``delegate_to_*``
    tools (``tasks`` also attaches :class:`~loomable.toolkits.todo_tools.TodoTools`).
    Hard modes (``broadcast``, ``sequential``): deterministic fan-out / pipeline
    without relying on the coordinator LLM to call tools correctly.

    Members may be :class:`Agent` instances or nested :class:`Team` instances
    (nested teams become a thin Agent wrapper via :meth:`as_agent`).
    """

    def __init__(
        self,
        members: list["Agent | Team"],
        model: "ModelProvider | ModelSpec | str",
        *,
        mode: TeamMode = "coordinate",
        instructions: str | None = None,
        session_id: str | None = None,
        max_delegations: int | None = None,
        max_depth: int = 4,
        max_iterations: int | None = None,
        hard: bool | None = None,
        # Same memory kwargs as Agent (L1/L2 + L3) — applied to the coordinator.
        resume: bool = False,
        use_memory: bool = True,
        memory_window: int = 8,
        compaction_threshold: int = 16,
        use_llm_summarizer: bool = False,
        session_store: Any | None = None,
        memory_backend: Any | None = None,
        note_store: Any | None = None,
        memory_tool: bool = False,
        knowledge: list[str] | None = None,
        knowledge_base: Any = None,
        embedder: Any = None,
        knowledge_top_k: int = 3,
        retrievers: list[Any] | None = None,
        user_id: str | None = None,
        # Composable memory bundle (same as Agent(memory=...))
        memory: Any | None = None,
    ) -> None:
        if not members:
            raise ValueError("Team requires at least one member")

        from .memory_opts import filter_memory_kwargs

        self._raw_members = list(members)
        self._members = [_coerce_member(m, i) for i, m in enumerate(members)]
        self._model = model
        self._mode = mode
        self._instructions = instructions
        self._session_id = session_id
        self._max_delegations = max_delegations
        self._max_depth = max_depth
        self._max_iterations = max_iterations
        # Hard by default for broadcast/sequential; soft for coordinate/route/tasks.
        # hard=True on soft modes used to be a silent no-op — reject it.
        soft_modes = ("coordinate", "route", "tasks")
        if hard is True and mode not in ("broadcast", "sequential"):
            raise AgentConfigError(
                f"Team(hard=True) only applies to mode='broadcast' or "
                f"'sequential' (got mode={mode!r}). Omit hard= for "
                f"{'/'.join(soft_modes)} (soft LLM orchestration)."
            )
        self._hard = (mode in ("broadcast", "sequential")) if hard is None else bool(hard)

        extra_tools: list[Any] = []
        if mode == "tasks":
            from loomable.toolkits.todo_tools import TodoTools

            extra_tools.append(TodoTools())

        agent_kwargs: dict[str, Any] = {
            "model": model,
            "role": "Team Coordinator",
            "goal": f"Coordinate the team in {mode} mode",
            "instructions": _assemble_team_instructions(mode, self._members, instructions),
            "subagents": self._members,
            "tools": extra_tools or None,
            **filter_memory_kwargs(
                {
                    "memory": memory,
                    "session_id": session_id,
                    "user_id": user_id,
                    "resume": resume,
                    "use_memory": use_memory,
                    "memory_window": memory_window,
                    "compaction_threshold": compaction_threshold,
                    "use_llm_summarizer": use_llm_summarizer,
                    "session_store": session_store,
                    "memory_backend": memory_backend,
                    "note_store": note_store,
                    "memory_tool": memory_tool,
                    "knowledge": knowledge,
                    "knowledge_base": knowledge_base,
                    "embedder": embedder,
                    "knowledge_top_k": knowledge_top_k,
                    "retrievers": retrievers,
                }
            ),
        }
        if max_iterations is not None:
            # Tasks loop: each iteration may be write_todos + delegate + update.
            agent_kwargs["max_tool_iterations"] = max(
                12, int(max_iterations) * 4
            )
        # resume=False is meaningful; filter drops None only — force session_id through
        if session_id is not None:
            agent_kwargs["session_id"] = session_id
        if resume:
            agent_kwargs["resume"] = True
        if memory is not None:
            agent_kwargs["memory"] = memory
        self._agent = Agent(**{k: v for k, v in agent_kwargs.items() if v is not None})
        # Stash budgets for build-time wiring (delegation tools rebuilt in arun soft path)
        self._agent._max_delegations = max_delegations  # type: ignore[attr-defined]
        self._agent._max_depth = max_depth  # type: ignore[attr-defined]
        from .memory_opts import apply_knowledge_base

        apply_knowledge_base(
            self._members,
            knowledge_base=knowledge_base,
            retrievers=retrievers,
            embedder=embedder,
        )
        # Soft coordinate: nudge the LLM to call every member (WR-020).
        if mode == "coordinate" and not self._hard and not self._agent._require_tools:
            from .delegation import delegation_tool_names

            self._agent._require_tools = [
                name for _, name in delegation_tool_names(self._members)
            ]
        # Soft tasks: nudge write_todos + at least one delegation.
        if mode == "tasks" and not self._agent._require_tools:
            from .delegation import delegation_tool_names

            names = [name for _, name in delegation_tool_names(self._members)]
            self._agent._require_tools = ["write_todos", *names[:1]]

    def as_agent(self, *, index: int = 0) -> Agent:
        """Expose this Team as a single Agent for nesting in another Team.

        The wrapper Agent's only tool is ``run_nested_team``, which forwards to
        :meth:`arun`. Used automatically when a :class:`Team` is passed as a member.
        """
        from loomable.agent.tools import FunctionTool

        team = self
        label = f"NestedTeam_{self._mode}_{index}"

        async def run_nested_team(task: str) -> str:
            """Run the nested team on a task and return its final text."""
            result = await team.arun(task)
            return result.output.text()

        tool = FunctionTool(
            run_nested_team,
            name="run_nested_team",
            description=(
                f"Run nested team ({self._mode} mode) on a task and return "
                "the synthesized result."
            ),
        )
        return Agent(
            model=self._model,
            name=label,
            role=label,
            goal=f"Execute nested {self._mode} team",
            instructions=(
                f"You wrap a nested Team (mode={self._mode}). "
                "Always call run_nested_team with the full task, then return its output."
            ),
            tools=[tool],
            require_tools=["run_nested_team"],
            max_tool_iterations=4,
        )

    def bind_session(self, session_id: str | None, *, resume: bool | None = None) -> None:
        """Bind HTTP/stream session id — same semantics as :meth:`Agent.bind_session`."""
        self._session_id = session_id or self._session_id
        self._agent.bind_session(session_id, resume=resume)

    @property
    def agent(self) -> Agent:
        """The underlying parent :class:`Agent` builder."""
        return self._agent

    @property
    def mode(self) -> str:
        """The team's orchestration mode."""
        return self._mode

    @property
    def members(self) -> list[Agent]:
        """The team member agents."""
        return list(self._members)

    async def _run_broadcast(
        self,
        task: str,
        *,
        images: "list[str | Any] | None" = None,
        videos: "list[str | Any] | None" = None,
        audio: "list[str | Any] | None" = None,
        output_schema: type | None = None,
    ) -> "RunResult":
        from loomable.agent.run import RunResult
        from loomable.content import AgentOutput, Text

        async def _one(member: Agent, index: int) -> tuple[str, str]:
            label = _member_label(member, index)
            try:
                result = await member.arun(
                    task,
                    images=images,
                    videos=videos,
                    audio=audio,
                    output_schema=output_schema,
                )
                return label, result.output.text()
            except Exception as exc:  # noqa: BLE001
                return label, f"ERROR: {exc}"

        pairs = await asyncio.gather(
            *[_one(m, i) for i, m in enumerate(self._members)]
        )
        lines = [f"## {label}\n{text}" for label, text in pairs]
        merged = "\n\n".join(lines)
        return RunResult(
            output=AgentOutput(parts=[Text(merged)]),
            session_id=self._session_id or "",
            metadata={"team_mode": "broadcast", "hard": True},
        )

    async def _run_sequential(
        self,
        task: str,
        *,
        images: "list[str | Any] | None" = None,
        videos: "list[str | Any] | None" = None,
        audio: "list[str | Any] | None" = None,
        output_schema: type | None = None,
    ) -> "RunResult":
        from loomable.agent.run import RunResult
        from loomable.content import AgentOutput, Text

        current = task
        trail: list[str] = []
        for i, member in enumerate(self._members):
            label = _member_label(member, i)
            prompt = current if i == 0 else (
                f"Prior output from previous team member:\n{current}\n\n"
                f"Continue the pipeline for the original task:\n{task}"
            )
            try:
                result = await member.arun(
                    prompt,
                    images=images if i == 0 else None,
                    videos=videos if i == 0 else None,
                    audio=audio if i == 0 else None,
                    output_schema=output_schema if i == len(self._members) - 1 else None,
                )
                current = result.output.text()
            except Exception as exc:  # noqa: BLE001
                current = f"ERROR from {label}: {exc}"
            trail.append(f"[{label}] {current[:500]}")
        return RunResult(
            output=AgentOutput(parts=[Text(current)]),
            session_id=self._session_id or "",
            metadata={
                "team_mode": "sequential",
                "hard": True,
                "trail": trail,
            },
        )

    async def arun(
        self,
        input: "AgentInput | str",  # noqa: A002
        *,
        images: "list[str | Any] | None" = None,
        videos: "list[str | Any] | None" = None,
        audio: "list[str | Any] | None" = None,
        output_schema: type | None = None,
        context: dict[str, Any] | None = None,
    ) -> "RunResult":
        """Run the team and return a :class:`~loomable.agent.run.RunResult`."""
        if self._hard and self._mode == "broadcast":
            text = _input_as_text(input)
            return await self._run_broadcast(
                text,
                images=images,
                videos=videos,
                audio=audio,
                output_schema=output_schema,
            )
        if self._hard and self._mode == "sequential":
            text = _input_as_text(input)
            return await self._run_sequential(
                text,
                images=images,
                videos=videos,
                audio=audio,
                output_schema=output_schema,
            )

        # Soft path: rebuild delegation tools with budgets if needed
        if self._max_delegations is not None or self._max_depth != 4:
            from .delegation import make_delegation_tools

            built = self._agent.build()
            # Replace delegation tools with budgeted versions
            budgeted = make_delegation_tools(
                self._members,
                max_delegations=self._max_delegations,
                max_depth=self._max_depth,
                depth=0,
            )
            for t in budgeted:
                built.tool_runtime._tools[t.name] = t
            result = await built.arun(
                input,
                images=images,
                videos=videos,
                audio=audio,
                output_schema=output_schema,
                context=context,
            )
        else:
            result = await self._agent.arun(
                input,
                images=images,
                videos=videos,
                audio=audio,
                output_schema=output_schema,
                context=context,
            )
        if self._mode == "coordinate" and not self._hard:
            result = await self._coordinate_fallback(_input_as_text(input), result)
        return result

    async def _coordinate_fallback(self, task: str, result: "RunResult") -> "RunResult":
        """Run members the coordinator never delegated to (WR-020)."""
        from .delegation import delegation_tool_names

        roster = delegation_tool_names(self._members)
        missing = {
            spec.split(":", 1)[0]
            for spec in (result.metadata or {}).get("required_tools_missing") or []
        }
        skipped = [(member, name) for member, name in roster if name in missing]
        if not skipped and not (result.tool_activity or []) and roster:
            skipped = list(roster)
        if not skipped:
            return result

        extras: list[tuple[str, str]] = []
        for index, (member, _name) in enumerate(skipped):
            label = _member_label(member, index)
            try:
                extra = await member.arun(task)
                extras.append((label, extra.output.text()))
            except Exception as exc:  # noqa: BLE001
                extras.append((label, f"ERROR: {exc}"))
        if not extras:
            return result

        from loomable.agent.run import RunResult
        from loomable.content import AgentOutput, Text

        extra_text = "\n\n".join(f"## {label} (fallback)\n{text}" for label, text in extras)
        merged = (result.output.text() or "").rstrip()
        if extra_text:
            merged = f"{merged}\n\n{extra_text}" if merged else extra_text
        meta = dict(result.metadata or {})
        meta["team_coordinate_fallback"] = [name for _, name in skipped]
        return RunResult(
            output=AgentOutput(parts=[Text(merged)]),
            session_id=result.session_id,
            usage=result.usage,
            tool_activity=list(result.tool_activity or []),
            structured=result.structured,
            metadata=meta,
        )

    def cancel(self) -> bool:
        """Cancel the coordinator and any in-flight members."""
        hit = False
        if self._agent.cancel():
            hit = True
        for member in self._members:
            cancel = getattr(member, "cancel", None)
            if callable(cancel) and cancel():
                hit = True
        return hit

    def run(
        self,
        input: "AgentInput | str",  # noqa: A002
        *,
        images: "list[str | Any] | None" = None,
        videos: "list[str | Any] | None" = None,
        audio: "list[str | Any] | None" = None,
        output_schema: type | None = None,
        context: dict[str, Any] | None = None,
    ) -> "RunResult":
        """Synchronous wrapper around :meth:`arun`."""
        return asyncio.run(
            self.arun(
                input,
                images=images,
                videos=videos,
                audio=audio,
                output_schema=output_schema,
                context=context,
            )
        )

    async def astream(
        self,
        input: "AgentInput | str",  # noqa: A002
        *,
        images: "list[str | Any] | None" = None,
        videos: "list[str | Any] | None" = None,
        audio: "list[str | Any] | None" = None,
        output_schema: type | None = None,
    ):
        """Stream NDJSON chunks from the coordinator agent (soft or hard modes).

        Hard ``broadcast`` / ``sequential`` fall back to ``arun`` then chunk
        (deterministic fan-out has no token stream).
        """
        from loomable.agent.run import RunChunk

        if self._hard and self._mode in ("broadcast", "sequential"):
            result = await self.arun(
                input,
                images=images,
                videos=videos,
                audio=audio,
                output_schema=output_schema,
            )
            parts = result.output.parts
            last = len(parts) - 1
            for index, part in enumerate(parts):
                yield RunChunk(delta=part, done=index == last)
            return

        async for chunk in self._agent.astream(input, output_schema=output_schema):
            yield chunk

    async def astream_events(
        self,
        input: "AgentInput | str",  # noqa: A002
        *,
        images: "list[str | Any] | None" = None,
        videos: "list[str | Any] | None" = None,
        audio: "list[str | Any] | None" = None,
        output_schema: type | None = None,
        context: dict[str, Any] | None = None,
        session_id: str | None = None,
    ):
        """Yield AG-UI :class:`~loomable.stream.StreamEvent` frames for the team.

        Soft modes (``coordinate`` / ``route``) delegate to the coordinator
        agent's ``astream_events``. Hard modes (``broadcast`` / ``sequential``)
        emit RUN_* lifecycle plus NODE_* frames per member.
        """
        # Soft / LLM-coordinated path: reuse Agent SSE (tools include delegates).
        if not (self._hard and self._mode in ("broadcast", "sequential")):
            if session_id:
                self.bind_session(session_id)
            from loomable.stream import (
                RUN_FINISHED,
                TEXT_MESSAGE_CONTENT,
                TEXT_MESSAGE_END,
                TEXT_MESSAGE_START,
                TOOL_CALL_START,
                StreamEvent,
            )

            called: set[str] = set()
            finished: Any = None
            async for ev in self._agent.astream_events(
                input,
                images=images,
                videos=videos,
                audio=audio,
                output_schema=output_schema,
                context=context,
                session_id=session_id,
            ):
                if getattr(ev, "type", None) == TOOL_CALL_START:
                    data = getattr(ev, "data", None) or {}
                    name = data.get("tool_name") or data.get("name") or ""
                    if name:
                        called.add(str(name))
                if (
                    getattr(ev, "type", None) == RUN_FINISHED
                    and self._mode == "coordinate"
                    and not self._hard
                ):
                    finished = ev
                    continue
                yield ev
            if finished is not None:
                from loomable.agent.run import RunResult
                from loomable.content import AgentOutput, Text

                from .delegation import delegation_tool_names

                roster = delegation_tool_names(self._members)
                missing = [name for _, name in roster if name not in called]
                # Skip fallback when every delegate was already called. An empty
                # synthetic RunResult has no tool_activity, which would otherwise
                # make _coordinate_fallback re-run the full roster.
                if missing:
                    fake = RunResult(
                        output=AgentOutput(parts=[Text("")]),
                        session_id=getattr(finished, "session_id", "")
                        or session_id
                        or "",
                        metadata={"required_tools_missing": missing},
                    )
                    extra = await self._coordinate_fallback(
                        _input_as_text(input), fake
                    )
                    extra_text = extra.output.text() if extra.output is not None else ""
                else:
                    extra = None
                    extra_text = ""
                if extra is not None and extra.metadata.get("team_coordinate_fallback") and extra_text:
                    rid = getattr(finished, "run_id", "") or ""
                    sid = getattr(finished, "session_id", "") or session_id or ""
                    yield StreamEvent(
                        type=TEXT_MESSAGE_START,
                        run_id=rid,
                        session_id=sid,
                        data={"role": "assistant"},
                    )
                    yield StreamEvent(
                        type=TEXT_MESSAGE_CONTENT,
                        run_id=rid,
                        session_id=sid,
                        data={"delta": extra_text},
                    )
                    yield StreamEvent(
                        type=TEXT_MESSAGE_END,
                        run_id=rid,
                        session_id=sid,
                        data={},
                    )
                yield finished
            return

        import uuid

        from loomable.stream import (
            RUN_ERROR,
            RUN_FINISHED,
            RUN_STARTED,
            TEXT_MESSAGE_CONTENT,
            TEXT_MESSAGE_END,
            TEXT_MESSAGE_START,
            AsyncStreamBus,
            StreamBridge,
        )

        text = _input_as_text(input)
        rid = uuid.uuid4().hex
        sid = session_id or self._session_id or ""
        bus = AsyncStreamBus(run_id=rid, session_id=sid)
        bridge = StreamBridge(bus, run_id=rid, session_id=sid)

        async def _runner() -> None:
            try:
                bridge.publish(RUN_STARTED, {"input": text[:500], "team_mode": self._mode})
                if self._mode == "broadcast":
                    result = await self._run_broadcast_streaming(
                        text,
                        bridge,
                        images=images,
                        videos=videos,
                        audio=audio,
                        output_schema=output_schema,
                    )
                else:
                    result = await self._run_sequential_streaming(
                        text,
                        bridge,
                        images=images,
                        videos=videos,
                        audio=audio,
                        output_schema=output_schema,
                    )
                out = result.output.text() if result.output is not None else ""
                bridge.publish(TEXT_MESSAGE_START, {"role": "assistant"})
                if out:
                    bridge.publish(TEXT_MESSAGE_CONTENT, {"delta": out})
                bridge.publish(TEXT_MESSAGE_END, {})
                bridge.publish(RUN_FINISHED, {"text": out[:2000], "team_mode": self._mode})
            except Exception as exc:  # noqa: BLE001
                bridge.publish(
                    RUN_ERROR,
                    {"message": str(exc), "error_type": type(exc).__name__},
                )
            finally:
                await bus.close()

        task = asyncio.create_task(_runner())
        try:
            async for event in bus:
                yield event
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _run_broadcast_streaming(
        self,
        task: str,
        bridge: Any,
        *,
        images: "list[str | Any] | None" = None,
        videos: "list[str | Any] | None" = None,
        audio: "list[str | Any] | None" = None,
        output_schema: type | None = None,
    ) -> "RunResult":
        import time

        from loomable.agent.run import RunResult
        from loomable.content import AgentOutput, Text
        from loomable.stream import NODE_FINISHED, NODE_STARTED

        async def _one(member: Agent, index: int) -> tuple[str, str]:
            label = _member_label(member, index)
            t0 = time.monotonic()
            bridge.publish(NODE_STARTED, {"node_id": label, "role": label})
            try:
                result = await member.arun(
                    task,
                    images=images,
                    videos=videos,
                    audio=audio,
                    output_schema=output_schema,
                )
                text = result.output.text()
                bridge.publish(
                    NODE_FINISHED,
                    {
                        "node_id": label,
                        "duration_ms": int((time.monotonic() - t0) * 1000),
                        "success": True,
                    },
                )
                return label, text
            except Exception as exc:  # noqa: BLE001
                bridge.publish(
                    NODE_FINISHED,
                    {
                        "node_id": label,
                        "duration_ms": int((time.monotonic() - t0) * 1000),
                        "success": False,
                        "error": str(exc),
                    },
                )
                return label, f"ERROR: {exc}"

        pairs = await asyncio.gather(
            *[_one(m, i) for i, m in enumerate(self._members)]
        )
        lines = [f"## {label}\n{text}" for label, text in pairs]
        merged = "\n\n".join(lines)
        return RunResult(
            output=AgentOutput(parts=[Text(merged)]),
            session_id=self._session_id or "",
            metadata={"team_mode": "broadcast", "hard": True},
        )

    async def _run_sequential_streaming(
        self,
        task: str,
        bridge: Any,
        *,
        images: "list[str | Any] | None" = None,
        videos: "list[str | Any] | None" = None,
        audio: "list[str | Any] | None" = None,
        output_schema: type | None = None,
    ) -> "RunResult":
        import time

        from loomable.agent.run import RunResult
        from loomable.content import AgentOutput, Text
        from loomable.stream import NODE_FINISHED, NODE_STARTED

        current = task
        trail: list[str] = []
        for i, member in enumerate(self._members):
            label = _member_label(member, i)
            prompt = current if i == 0 else (
                f"Prior output from previous team member:\n{current}\n\n"
                f"Continue the pipeline for the original task:\n{task}"
            )
            t0 = time.monotonic()
            bridge.publish(NODE_STARTED, {"node_id": label, "role": label})
            try:
                result = await member.arun(
                    prompt,
                    images=images if i == 0 else None,
                    videos=videos if i == 0 else None,
                    audio=audio if i == 0 else None,
                    output_schema=output_schema if i == len(self._members) - 1 else None,
                )
                current = result.output.text()
                bridge.publish(
                    NODE_FINISHED,
                    {
                        "node_id": label,
                        "duration_ms": int((time.monotonic() - t0) * 1000),
                        "success": True,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                current = f"ERROR from {label}: {exc}"
                bridge.publish(
                    NODE_FINISHED,
                    {
                        "node_id": label,
                        "duration_ms": int((time.monotonic() - t0) * 1000),
                        "success": False,
                        "error": str(exc),
                    },
                )
            trail.append(f"[{label}] {current[:500]}")
        return RunResult(
            output=AgentOutput(parts=[Text(current)]),
            session_id=self._session_id or "",
            metadata={
                "team_mode": "sequential",
                "hard": True,
                "trail": trail,
            },
        )
