"""CodeIndex + CodeTools + deep profile=code tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from loomable.agent import ModelSpec, create_deep_agent
from loomable.codeindex import CodeIndex, HashingEmbedder
from loomable.kernel.long_term import LongTermStore, ZvecVectorBackend
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.skills import list_bundled_skills
from loomable.toolkits import CodeTools


class _Noop:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="done")


def _write_mini_repo(root: Path) -> None:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "auth.py").write_text(
        "class AuthService:\n"
        "    def login(self, user: str) -> bool:\n"
        "        return True\n"
        "\n"
        "def verify_token(token: str) -> bool:\n"
        "    return token == 'ok'\n",
        encoding="utf-8",
    )
    (root / "pkg" / "api.py").write_text(
        "from pkg.auth import AuthService\n"
        "\n"
        "def handle_login(user: str) -> str:\n"
        "    ok = AuthService().login(user)\n"
        "    return 'yes' if ok else 'no'\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_zvec_persist_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "store.json"
    store = LongTermStore(path=path)
    await store.index("a", [1.0, 0.0], {"text": "hello"})
    assert path.is_file()
    store2 = LongTermStore(path=path)
    hits = await store2.query([1.0, 0.0], k=1)
    assert hits and hits[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_codeindex_build_search_and_map(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_mini_repo(repo)
    persist = tmp_path / "idx.json"
    index = await CodeIndex.build(repo, persist_path=persist, embedder=HashingEmbedder())
    assert index.size >= 2
    assert persist.is_file()
    mapping = index.repo_map()
    assert "auth.py" in mapping
    hits = await index.search("AuthService login verify token", k=5)
    assert hits
    assert any("auth" in h.path for h in hits)
    syms = index.find_symbol("AuthService")
    assert syms and syms[0].kind == "class"


@pytest.mark.asyncio
async def test_code_tools(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_mini_repo(repo)
    index = await CodeIndex.build(repo, persist_path=tmp_path / "i.json")
    tools = {t.name: t for t in CodeTools(index).tools()}
    mapped = await tools["repo_map"].invoke({"max_entries": 20})
    assert "auth.py" in str(mapped.content)
    searched = await tools["code_search"].invoke({"query": "verify_token", "limit": 3})
    assert "verify" in str(searched.content).lower() or "auth" in str(searched.content)
    found = await tools["find_symbol"].invoke({"name": "handle_login"})
    assert "handle_login" in str(found.content)


@pytest.mark.asyncio
async def test_pluggable_backend(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_mini_repo(repo)
    backend = ZvecVectorBackend()  # in-memory custom backend
    index = await CodeIndex.build(
        repo, backend=backend, embedder=HashingEmbedder(), persist_path=None
    )
    # Override default file path by passing backend= — store uses custom
    assert index.size >= 1
    hits = await index.search("login", k=3)
    assert hits


def test_coding_skill_bundled() -> None:
    assert "coding" in list_bundled_skills()


@pytest.mark.asyncio
async def test_create_deep_agent_profile_code(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_mini_repo(repo)
    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Noop()),
        workspace=tmp_path / "ws",
        profile="code",
        repo=repo,
        enable_task_tool=False,
        think_tool=True,
        use_llm_summarizer=False,
        modalities="text",
    )
    assert getattr(agent, "_name", None) == "code-agent"
    built = agent.build()
    names = set(built.tool_runtime._tools) | {
        t.name for t in built.discovery.catalog.tools
    }
    for required in ("repo_map", "code_search", "find_symbol", "run_python", "run_shell"):
        assert required in names, required
    assert "coding" in list_bundled_skills()
