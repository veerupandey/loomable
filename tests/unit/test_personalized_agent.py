"""Tough personalized-KB task: policy vs personal secret-handling conflict.

Company policy allows committed .env tokens. Avery's personal notes forbid
any secrets in git. Runbook has KEY-WHSEC-4419. The agent must search both
KBs, refuse the commit, and cite the webhook key.

Also proves deep-agent discovery does not defer search_* tools.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomable.agent import ModelSpec, create_deep_agent, create_personalized_agent
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.kernel.retrievers import RetrieverTool


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


@pytest.mark.asyncio
async def test_personalized_agent_resolves_secret_policy_conflict(tmp_path: Path) -> None:
    personal, company = _seed(tmp_path)
    solver = _ConflictSolver()
    agent = await create_personalized_agent(
        ModelSpec(provider="scripted", provider_impl=solver),
        user_id="avery",
        personal=[personal],
        knowledge=[company],
        deep=False,
        max_tool_iterations=8,
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
async def test_deep_personalized_agent_search_tools_not_deferred(tmp_path: Path) -> None:
    personal, company = _seed(tmp_path)
    solver = _ConflictSolver()
    agent = await create_personalized_agent(
        ModelSpec(provider="scripted", provider_impl=solver),
        user_id="avery",
        personal=[personal],
        knowledge=[company],
        deep=True,
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
async def test_create_deep_agent_keeps_custom_retriever_in_core(tmp_path: Path) -> None:
    from loomable.providers.vector_store import open_vector_store
    from loomable.retrieval import AgenticRetriever, ingest

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
