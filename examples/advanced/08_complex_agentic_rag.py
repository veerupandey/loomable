"""Complex multi-format agentic RAG — fails loudly when retrieval is wrong.

Seeds markdown, Python, TypeScript, HTML, JSON, CSV, RST, DOCX (+ optional URL),
builds hybrid+MMR agentic tools across two corpora, and asserts hard queries.

Run::

    python examples/advanced/08_complex_agentic_rag.py
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path
from typing import Any

from loomable.agent import Agent, ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.providers.vector_store import open_vector_store
from loomable.retrieval import AgenticRetriever, CompositeRetriever, ingest

ROOT = Path(__file__).resolve().parent / ".complex_rag_demo"
ROOT.mkdir(parents=True, exist_ok=True)


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    """Minimal DOCX writer (stdlib zip/XML)."""
    body = []
    for p in paragraphs:
        body.append(
            "<w:p><w:r><w:t>"
            + (
                p.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            + "</w:t></w:r></w:p>"
        )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body)
        + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)
    path.write_bytes(buf.getvalue())


def _seed() -> tuple[Path, Path]:
    docs = ROOT / "docs"
    code = ROOT / "code"
    docs.mkdir(exist_ok=True)
    code.mkdir(exist_ok=True)

    (docs / "auth.md").write_text(
        "# Authentication\n\nClients must use OAuth2 bearer tokens.\n"
        "Never put secrets in query strings.\n",
        encoding="utf-8",
    )
    (docs / "runbook.rst").write_text(
        "Incident response\n=================\n\n"
        "Rotate the webhook-signing secret KEY-WHSEC-77 after a breach.\n",
        encoding="utf-8",
    )
    (docs / "pricing.html").write_text(
        "<html><body><h1>Pricing</h1>"
        "<p>Enterprise plan SKU-ENT-204 includes SSO and audit logs.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    (docs / "catalog.json").write_text(
        '{"plans":[{"sku":"SKU-ENT-204","name":"Enterprise","sso":true}],'
        '"addons":[{"sku":"SKU-ADD-LOG","name":"Audit pack"}]}',
        encoding="utf-8",
    )
    (docs / "rates.csv").write_text(
        "region,latency_ms,tier\neus-east,42,gold\nap-south,118,silver\n",
        encoding="utf-8",
    )
    _write_docx(
        docs / "policy.docx",
        [
            "Data Retention Policy",
            "Customer logs are retained for 90 days then purged from cold storage.",
        ],
    )

    (code / "auth_middleware.py").write_text(
        '"""Auth middleware."""\n\n'
        "def require_bearer(header: str) -> str:\n"
        '    if not header.startswith("Bearer "):\n'
        '        raise PermissionError("missing bearer")\n'
        "    return header[7:]\n",
        encoding="utf-8",
    )
    (code / "deploy.ts").write_text(
        "export async function rolloutPaymentService(ns: string) {\n"
        '  // kubectl apply -f payment-service.yaml -n ${ns}\n'
        '  return `rolled out payment-service in ${ns}`;\n'
        "}\n",
        encoding="utf-8",
    )
    return docs, code


CASES: list[tuple[str, list[str], str]] = [
    # query, required substrings in top hit, label
    ("OAuth2 bearer authentication", ["oauth", "bearer"], "auth-md"),
    ("require_bearer PermissionError", ["require_bearer", "permission"], "auth-py"),
    ("rollout payment-service kubectl", ["payment-service", "rollout"], "deploy-ts"),
    ("SKU-ENT-204 enterprise SSO", ["sku-ent-204", "sso"], "pricing-html-or-json"),
    ("webhook-signing KEY-WHSEC-77", ["key-whsec-77", "webhook"], "runbook-rst"),
    ("customer logs retained purged", ["90 days", "purged"], "policy-docx"),
    ("ap-south latency tier", ["ap-south", "118"], "rates-csv"),
]


class _Scripted:
    """Calls search_knowledge then answers — proves agent tool wiring."""

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
                        args={"query": "OAuth2 bearer authentication", "k": 3},
                    )
                ],
            )
        # Inspect tool results in history for a hard check
        blob = str(request.messages).lower()
        if "oauth" not in blob and "bearer" not in blob:
            return ModelResponse(content="FAIL: tool result missing auth content")
        return ModelResponse(content="PASS: agentic RAG tool returned auth context")


def _hit_blob(hits: list[dict[str, Any]]) -> str:
    return " ".join(str(h.get("content") or "") for h in hits).lower()


async def main() -> None:
    docs_dir, code_dir = _seed()
    docs_corpus = await ingest(
        [docs_dir],
        name="docs",
        description="Product docs, policies, pricing HTML/JSON/CSV",
        store=open_vector_store(engine="memory"),
        strategy="auto",
        base_mode="hybrid",
    )
    code_corpus = await ingest(
        [code_dir],
        name="code",
        description="Application source: Python and TypeScript",
        store=open_vector_store(engine="memory"),
        strategy="auto",
        base_mode="hybrid",
    )

    # Optional live URL — skipped quietly if network/SSRF blocks it
    try:
        from loomable.retrieval import load_url

        url_doc = load_url("https://example.com")
        await docs_corpus.upsert(
            [{"id": "example-com", "text": url_doc.text, "source": url_doc.source, "media_type": "text/html"}]
        )
        print("url-ingest: ok (example.com)")
    except Exception as exc:  # noqa: BLE001
        print(f"url-ingest: skipped ({exc})")

    rag = CompositeRetriever(
        [
            AgenticRetriever(docs_corpus, name="search_docs", mode="chunks", rewrite="off", rerank="mmr"),
            AgenticRetriever(code_corpus, name="search_code", mode="chunks", rewrite="off", rerank="mmr"),
        ],
        name="search_knowledge",
        corpus_router="all",
        rerank="mmr",
    )
    assert rag.name == "search_knowledge"

    failures: list[str] = []
    for query, needles, label in CASES:
        hits = await rag.retrieve(query, k=3)
        blob = _hit_blob(hits)
        missing = [n for n in needles if n.lower() not in blob]
        if missing:
            top = (hits[0].get("content") if hits else "")[:120]
            failures.append(f"{label}: missing {missing} | top={top!r}")
            print(f"FAIL [{label}] q={query!r} missing={missing}")
        else:
            print(f"PASS [{label}] q={query!r}")

    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_Scripted()),
        retrievers=[rag],
        use_llm_summarizer=False,
    )
    result = await agent.arun("How do we authenticate?")
    text = result.output.text() or ""
    print("agent:", text)
    if not text.startswith("PASS"):
        failures.append(f"agent-tool: {text}")

    if failures:
        raise SystemExit("COMPLEX RAG FAILURES:\n- " + "\n- ".join(failures))
    print("ALL COMPLEX CASES PASSED")


if __name__ == "__main__":
    asyncio.run(main())
