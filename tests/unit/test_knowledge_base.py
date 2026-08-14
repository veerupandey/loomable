"""Knowledge base is a vector DB; retrievers are extra search tools.

Agent / create_deep_agent / Team / Workflow share the same knowledge_base=
surface. There is no separate personalized-agent factory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomable.agent import Agent, ModelSpec, create_deep_agent
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.kernel.retrievers import RetrieverTool
from loomable.providers.vector_store import open_vector_store
from loomable.retrieval import KnowledgeBase, ingest


def _seed(root: Path) -> tuple[Path, Path]:
    personal = root / "personal"
    company = root / "company"
    personal.mkdir()
    company.mkdir()
    (personal / "prefs.md").write_text(
        "# Avery preferences\n\n"
        "Never commit secrets. No API tokens in git, including .env. "
        "Timezone Asia/Kolkata. Allergic to shellfish.\n",
        encoding="utf-8",
    )
    (company / "policy.md").write_text(
        "# Credential policy\n\n"
        "Staging credentials MAY be stored in a committed internal .env file.\n",
        encoding="utf-8",
    )
    (company / "runbook.md").write_text(
        "# Webhooks\n\n"
        "Current webhook signing secret is KEY-WHSEC-4419. Rotate after any leak.\n",
        encoding="utf-8",
    )
    return personal, company


def _tool_names(request: ModelRequest) -> set[str]:
    names: set[str] = set()
    for t in request.tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = (fn or {}).get("name") or t.get("name")
        if name:
            names.add(str(name))
    return names


class _ConflictSolver:
    """Calls both KBs then answers — fails if tools were deferred or empty."""

    def __init__(self) -> None:
        self.n = 0
        self.advertised: set[str] = set()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        names = _tool_names(request)
        if self.n == 1:
            self.advertised = names
            if "search_personal" not in names or "search_company" not in names:
                return ModelResponse(
                    content=f"FAIL: search tools not advertised: {sorted(names)}"
                )
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        tool_name="search_personal",
                        args={"query": "commit secrets API tokens git", "k": 3},
                    )
                ],
            )
        if self.n == 2:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="2",
                        tool_name="search_company",
                        args={"query": "staging .env credentials webhook KEY-WHSEC", "k": 5},
                    )
                ],
            )
        blob = str(request.messages).lower()
        has_personal = "never commit" in blob or "no api tokens" in blob
        has_key = "key-whsec-4419" in blob
        has_policy = ".env" in blob or "staging" in blob
        if has_personal and has_key:
            return ModelResponse(
                content=(
                    "Do not commit STAGING_API_TOKEN. Avery's personal notes forbid "
                    "secrets in git, which is stricter than company .env policy. "
                    "Rotate the webhook with KEY-WHSEC-4419 "
                    "(sources: prefs.md, runbook.md)."
                )
            )
        return ModelResponse(
            content=(
                f"FAIL evidence personal={has_personal} key={has_key} policy={has_policy}"
            )
        )


def test_personalized_factory_is_not_public() -> None:
    import loomable
    import loomable.agent as agent_mod

    assert "create_personalized_agent" not in loomable.__all__
    assert not hasattr(loomable, "create_personalized_agent")
    assert "create_personalized_agent" not in agent_mod.__all__
    assert not hasattr(agent_mod, "create_personalized_agent")


@pytest.mark.asyncio
async def test_agent_knowledge_base_named_collections(tmp_path: Path) -> None:
    personal, company = _seed(tmp_path)
    solver = _ConflictSolver()
    agent = Agent(
        ModelSpec(provider="scripted", provider_impl=solver),
        user_id="avery",
        knowledge_base={"personal": [personal], "company": [company]},
        max_tool_iterations=8,
        use_llm_summarizer=False,
    )
    built = agent.build()
    assert "search_personal" in built.tool_runtime._tools
    assert "search_company" in built.tool_runtime._tools
    assert isinstance(built.tool_runtime._tools["search_personal"], RetrieverTool)
    result = await agent.arun(
        "Can I commit STAGING_API_TOKEN=sk-live-99 per policy? "
        "Also what is the webhook signing secret? Cite sources."
    )
    text = (result.output.text() or "").lower()
    assert text.startswith("do not commit"), text
    assert "key-whsec-4419" in text
    assert "prefs.md" in text or "runbook.md" in text


@pytest.mark.asyncio
async def test_deep_agent_knowledge_base_not_deferred(tmp_path: Path) -> None:
    personal, company = _seed(tmp_path)
    solver = _ConflictSolver()
    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=solver),
        user_id="avery",
        knowledge_base={"personal": [personal], "company": [company]},
        discovery=True,
        workspace=tmp_path / "ws",
        max_tool_iterations=8,
        web_search=False,
        url_fetch=False,
        citations=False,
        enable_task_tool=False,
        enable_task_batch=False,
        think_tool=False,
        board=False,
        use_llm_summarizer=False,
    )
    built = agent.build()
    assert "search_personal" in built.tool_runtime._tools
    assert "search_company" in built.tool_runtime._tools
    result = await agent.arun(
        "Can I commit STAGING_API_TOKEN=sk-live-99? What is the webhook key?"
    )
    text = (result.output.text() or "").lower()
    assert "do not commit" in text, text
    assert "key-whsec-4419" in text
    assert "search_personal" in solver.advertised
    assert "search_company" in solver.advertised


@pytest.mark.asyncio
async def test_knowledge_base_is_vector_store(tmp_path: Path) -> None:
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "a.md").write_text("# Auth\n\nOAuth2 only. Client id is app-77.\n", encoding="utf-8")
    store = open_vector_store(engine="memory")
    await ingest(
        [docs],
        name="kb",
        store=store,
        strategy="markdown",
        base_mode="vector",
    )

    class _Ping:
        n = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.n += 1
            names = _tool_names(request)
            if self.n == 1:
                if "search_knowledge" not in names:
                    return ModelResponse(content=f"FAIL deferred {sorted(names)}")
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="1",
                            tool_name="search_knowledge",
                            args={"query": "OAuth2 client id", "k": 3},
                        )
                    ],
                )
            blob = str(request.messages).lower()
            if "app-77" in blob or "oauth2" in blob:
                return ModelResponse(content="OAuth2 only; client id app-77")
            return ModelResponse(content=f"FAIL {blob[:400]}")

    agent = Agent(
        ModelSpec(provider="scripted", provider_impl=_Ping()),
        knowledge_base=store,
        max_tool_iterations=6,
        use_llm_summarizer=False,
    )
    built = agent.build()
    assert "search_knowledge" in built.tool_runtime._tools
    result = await agent.arun("What auth protocol?")
    assert "oauth2" in (result.output.text() or "").lower()


@pytest.mark.asyncio
async def test_knowledge_base_and_retriever_together(tmp_path: Path) -> None:
    from loomable.retrieval import AgenticRetriever

    notes = tmp_path / "notes.md"
    notes.write_text("# Notes\n\nPrefer dark mode.\n", encoding="utf-8")
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "sku.md").write_text("# Catalog\n\nSKU-4419 is the widget.\n", encoding="utf-8")
    corpus = await ingest(
        [extra],
        name="catalog",
        store=open_vector_store(engine="memory"),
        strategy="markdown",
        base_mode="lexical",
    )
    rag = AgenticRetriever(corpus, name="search_catalog", rewrite="off", rerank=False)

    class _Both:
        n = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.n += 1
            names = _tool_names(request)
            if self.n == 1:
                assert "search_notes" in names
                assert "search_catalog" in names
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="1",
                            tool_name="search_notes",
                            args={"query": "dark mode", "k": 2},
                        )
                    ],
                )
            return ModelResponse(content="ok")

    agent = Agent(
        ModelSpec(provider="scripted", provider_impl=_Both()),
        knowledge_base={"notes": [notes]},
        retrievers=[rag],
        max_tool_iterations=4,
        use_llm_summarizer=False,
    )
    built = agent.build()
    assert "search_notes" in built.tool_runtime._tools
    assert "search_catalog" in built.tool_runtime._tools
    result = await agent.arun("prefs")
    assert "ok" in (result.output.text() or "")


@pytest.mark.asyncio
async def test_workflow_inherits_knowledge_base(tmp_path: Path) -> None:
    from loomable.flow.workflow import Workflow

    docs = tmp_path / "policy.md"
    docs.write_text("# Policy\n\nNever ship without review.\n", encoding="utf-8")

    class _Ping:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            names = _tool_names(request)
            if "search_knowledge" not in names:
                return ModelResponse(content=f"FAIL {sorted(names)}")
            return ModelResponse(content="kb visible")

    worker = Agent(
        ModelSpec(provider="scripted", provider_impl=_Ping()),
        role="Worker",
        use_llm_summarizer=False,
        max_tool_iterations=2,
    )
    wf = Workflow(
        "kb-share",
        knowledge_base=KnowledgeBase(sources=[docs], name="knowledge"),
    ).step("work", worker)
    built = worker.build()
    assert "search_knowledge" in built.tool_runtime._tools
    result = await wf.arun("ping")
    assert "visible" in (result.output.text() or "")


@pytest.mark.asyncio
async def test_create_deep_agent_keeps_custom_retriever_in_core(tmp_path: Path) -> None:
    from loomable.retrieval import AgenticRetriever

    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nOAuth2 only.\n", encoding="utf-8")
    corpus = await ingest(
        [docs],
        name="kb",
        store=open_vector_store(engine="memory"),
        strategy="markdown",
        base_mode="lexical",
    )
    rag = AgenticRetriever(corpus, name="search_kb", rewrite="off", rerank=False)

    class _Ping:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            names = _tool_names(request)
            if "search_kb" not in names:
                return ModelResponse(content=f"FAIL deferred {sorted(names)}")
            return ModelResponse(content="search_kb visible")

    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Ping()),
        retrievers=[rag],
        discovery=True,
        workspace=tmp_path / "ws2",
        web_search=False,
        url_fetch=False,
        citations=False,
        enable_task_tool=False,
        think_tool=False,
        board=False,
        use_llm_summarizer=False,
        max_tool_iterations=2,
    )
    built = agent.build()
    assert "search_kb" in built.tool_runtime._tools
    result = await agent.arun("ping")
    assert "visible" in (result.output.text() or "")


@pytest.mark.asyncio
async def test_team_inherits_knowledge_base(tmp_path: Path) -> None:
    from loomable.agent.team import Team

    docs = tmp_path / "policy.md"
    docs.write_text("# Policy\n\nNever ship without review.\n", encoding="utf-8")

    class _Ping:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            names = _tool_names(request)
            if "search_knowledge" not in names:
                return ModelResponse(content=f"FAIL {sorted(names)}")
            return ModelResponse(content="kb visible")

    member = Agent(
        ModelSpec(provider="scripted", provider_impl=_Ping()),
        role="Reviewer",
        use_llm_summarizer=False,
        max_tool_iterations=2,
    )
    Team(
        [member],
        model=ModelSpec(provider="scripted", provider_impl=_Ping()),
        mode="broadcast",
        knowledge_base=[docs],
    )
    built = member.build()
    assert "search_knowledge" in built.tool_runtime._tools
