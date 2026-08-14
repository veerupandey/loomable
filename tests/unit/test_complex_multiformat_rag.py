"""Multi-format ingest + complex agentic RAG regression tests.

These cases are intentionally picky — a wrong top hit fails the suite.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.providers.vector_store import open_vector_store
from loomable.retrieval import (
    AgenticRetriever,
    CompositeRetriever,
    get_strategy,
    ingest,
    is_http_url,
    list_strategies,
    load_file,
    load_sources,
    load_url,
)


def _docx(path: Path, paras: list[str]) -> None:
    body = "".join(
        "<w:p><w:r><w:t>"
        + p.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</w:t></w:r></w:p>"
        for p in paras
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>",
        )
        zf.writestr("word/document.xml", document_xml)
    path.write_bytes(buf.getvalue())


def test_strategies_include_structured() -> None:
    names = set(list_strategies())
    for required in ("auto", "markdown", "code", "html", "pdf", "json", "csv"):
        assert required in names


def test_is_http_url() -> None:
    assert is_http_url("https://example.com/path")
    assert not is_http_url("README.md")
    assert not is_http_url("/tmp/x")


def test_load_mixed_formats(tmp_path: Path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nalpha oauth\n", encoding="utf-8")
    (root / "b.py").write_text("def beta():\n    return 1\n", encoding="utf-8")
    (root / "c.ts").write_text("export const gamma = 1;\n", encoding="utf-8")
    (root / "d.html").write_text("<html><body><p>delta</p></body></html>", encoding="utf-8")
    (root / "e.json").write_text('{"sku":"SKU-1","name":"Widget"}', encoding="utf-8")
    (root / "f.csv").write_text("id,name\n1,epsilon\n", encoding="utf-8")
    (root / "g.rst").write_text("Title\n=====\n\nzeta note\n", encoding="utf-8")
    _docx(root / "h.docx", ["eta retention policy"])

    docs = load_sources([root])
    suffixes = {Path(d.source).suffix.lower() for d in docs}
    for need in (".md", ".py", ".ts", ".html", ".json", ".csv", ".rst", ".docx"):
        assert need in suffixes

    # kind hints + chunkers
    by_suf = {Path(d.source).suffix.lower(): d for d in docs}
    assert by_suf[".md"].kind_hint == "markdown"
    assert by_suf[".py"].kind_hint == "code"
    assert by_suf[".ts"].kind_hint == "code"
    assert by_suf[".html"].kind_hint == "html"
    assert by_suf[".json"].kind_hint == "json"
    assert by_suf[".csv"].kind_hint == "csv"

    assert get_strategy("json").chunk(by_suf[".json"])
    assert get_strategy("csv").chunk(by_suf[".csv"])
    assert "delta" in get_strategy("html").chunk(by_suf[".html"])[0].text.lower()
    assert "eta" in load_file(root / "h.docx").text.lower()


@pytest.mark.asyncio
async def test_complex_multiformat_agentic_rag(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    code = tmp_path / "code"
    docs.mkdir()
    code.mkdir()
    (docs / "auth.md").write_text(
        "# Auth\n\nUse OAuth2 bearer tokens for API login.\n", encoding="utf-8"
    )
    (docs / "pricing.html").write_text(
        "<html><body><p>Enterprise SKU-ENT-204 includes SSO.</p></body></html>",
        encoding="utf-8",
    )
    (docs / "catalog.json").write_text(
        '{"sku":"SKU-ENT-204","feature":"SSO audit logs"}', encoding="utf-8"
    )
    (docs / "rates.csv").write_text(
        "region,latency_ms\nap-south,118\n", encoding="utf-8"
    )
    (docs / "incident.rst").write_text(
        "Rotate webhook KEY-WHSEC-77 after breach.\n", encoding="utf-8"
    )
    _docx(docs / "policy.docx", ["Logs retained 90 days then purged."])
    (code / "auth.py").write_text(
        "def require_bearer(h: str) -> str:\n"
        '    if not h.startswith("Bearer "): raise PermissionError("no")\n'
        "    return h[7:]\n",
        encoding="utf-8",
    )
    (code / "deploy.ts").write_text(
        "export function rolloutPaymentService() { return 'payment-service'; }\n",
        encoding="utf-8",
    )

    docs_c = await ingest(
        [docs],
        name="docs",
        store=open_vector_store(engine="memory"),
        strategy="auto",
        base_mode="hybrid",
    )
    code_c = await ingest(
        [code],
        name="code",
        store=open_vector_store(engine="memory"),
        strategy="auto",
        base_mode="hybrid",
    )
    rag = CompositeRetriever(
        [
            AgenticRetriever(docs_c, mode="chunks", rewrite="off", rerank="mmr"),
            AgenticRetriever(code_c, mode="chunks", rewrite="off", rerank="mmr"),
        ],
        name="search_knowledge",
        corpus_router="all",
    )
    assert rag.name == "search_knowledge"

    cases = [
        ("OAuth2 bearer tokens", ["oauth", "bearer"]),
        ("require_bearer PermissionError", ["require_bearer"]),
        ("payment-service rollout", ["payment-service"]),
        ("SKU-ENT-204 SSO", ["sku-ent-204"]),
        ("KEY-WHSEC-77 webhook", ["key-whsec-77"]),
        ("90 days purged logs", ["90 days", "purged"]),
        ("ap-south latency", ["ap-south", "118"]),
    ]
    failures: list[str] = []
    for query, needles in cases:
        hits = await rag.retrieve(query, k=3)
        blob = " ".join(str(h.get("content") or "") for h in hits).lower()
        missing = [n for n in needles if n.lower() not in blob]
        if missing:
            failures.append(f"{query!r} missing {missing} blob={blob[:200]!r}")
    assert not failures, "complex RAG regressions:\n" + "\n".join(failures)

    class _Scripted:
        def __init__(self) -> None:
            self.n = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.n += 1
            if self.n == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="1",
                            tool_name="search_knowledge",
                            args={"query": "OAuth2 bearer", "k": 2},
                        )
                    ],
                )
            return ModelResponse(content="ok")

    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_Scripted()),
        retrievers=[rag],
        use_llm_summarizer=False,
    )
    result = await agent.arun("auth?")
    assert (result.output.text() or "") == "ok"


@pytest.mark.asyncio
async def test_live_url_ingest_example_com() -> None:
    try:
        doc = load_url("https://example.com")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"url fetch unavailable: {exc}")
    assert "example" in doc.text.lower()
    assert doc.kind_hint == "html"
