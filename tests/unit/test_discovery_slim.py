"""Slim discovery_core profile + specialist discovery inheritance."""

from __future__ import annotations

from loomable.agent import ModelSpec, create_deep_agent
from loomable.agent.deep import (
    DEEP_DISCOVERY_CORE_SLIM,
    DEEP_DISCOVERY_CORE_TOOLS,
    _resolve_discovery_core,
    make_task_tools,
)
from loomable.kernel.models import ModelRequest, ModelResponse


class _Noop:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="done")


def test_resolve_discovery_core_profiles() -> None:
    full = set(_resolve_discovery_core("research"))
    slim = set(_resolve_discovery_core("research-slim"))
    assert full == set(DEEP_DISCOVERY_CORE_TOOLS)
    assert slim == set(DEEP_DISCOVERY_CORE_SLIM)
    assert len(slim) < len(full)
    custom = _resolve_discovery_core(["write_file", "think"])
    assert custom == ["write_file", "think"]


def test_slim_advertises_fewer_tools_than_research(tmp_path) -> None:
    model = ModelSpec(provider="scripted", provider_impl=_Noop())
    common = dict(
        workspace=tmp_path,
        web_search=False,
        url_fetch=False,
        citations=True,
        images=False,
        enable_task_tool=True,
        think_tool=True,
        modalities="text",
        use_llm_summarizer=False,
        discovery=True,
    )
    full = create_deep_agent(model, discovery_core="research", **common).build()
    slim = create_deep_agent(
        model, workspace=tmp_path / "slim", discovery_core="research-slim", **{
            k: v for k, v in common.items() if k != "workspace"
        }
    ).build()

    full_names = set(full.tool_runtime._tools)
    slim_names = set(slim.tool_runtime._tools)
    assert len(slim_names) < len(full_names)
    # Citation verify tools stay deferred under slim, advertised under research core.
    assert "verify_source" in full_names or (
        full.discovery.catalog.tool_by_name("verify_source") is not None
        and full.discovery.catalog.tool_by_name("verify_source").activated
    )
    assert "verify_source" not in slim_names
    stub = slim.discovery.catalog.tool_by_name("verify_source")
    assert stub is not None and stub.activated is False
    assert "write_file" in slim_names
    assert "search_tools" in slim_names


def test_task_specialists_inherit_discovery(tmp_path) -> None:
    """create_deep_agent wires discovery=True into make_task_tools for specialists."""
    model = ModelSpec(provider="scripted", provider_impl=_Noop())
    agent = create_deep_agent(
        model,
        workspace=tmp_path,
        web_search=False,
        url_fetch=False,
        citations=False,
        images=False,
        enable_task_tool=True,
        think_tool=True,
        modalities="text",
        use_llm_summarizer=False,
        discovery=True,
        discovery_core="research-slim",
    )
    built = agent.build()
    assert "task" in built.tool_runtime._tools

    # Direct factory: specialists get discovery kwargs when enabled.
    tools = make_task_tools(
        model=model,
        tools=[],
        discovery=True,
        discovery_core_tools=_resolve_discovery_core("research-slim"),
        defer_local_tools=True,
        enable_batch=False,
    )
    assert tools and tools[0].name == "task"
    # Closure chain: task → _run_one captures discovery=True for spawn_specialist.
    task_fn = tools[0]._func
    task_free = {
        name: cell.cell_contents
        for name, cell in zip(
            task_fn.__code__.co_freevars,
            task_fn.__closure__ or (),
            strict=True,
        )
    }
    run_one = task_free.get("_run_one")
    assert run_one is not None
    free = {
        name: cell.cell_contents
        for name, cell in zip(
            run_one.__code__.co_freevars,
            run_one.__closure__ or (),
            strict=True,
        )
    }
    assert free.get("discovery") is True
    assert set(free.get("discovery_core_tools") or []) == set(DEEP_DISCOVERY_CORE_SLIM)
