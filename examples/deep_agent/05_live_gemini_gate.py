"""Live Gemini gate for loomable create_deep_agent (research + discovery).

Measures schema budget, runs a bounded research brief, and checks accept.

    DEEP_AGENT_LIVE=1 python examples/deep_agent/05_live_gemini_gate.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable.agent.deep import DEEP_DISCOVERY_CORE_TOOLS, create_deep_agent
from loomable.providers.gemini import GeminiProvider


ROOT = Path(__file__).resolve().parent / ".workspace_gemini_gate"


def _schema_count(built) -> int:
    return len(built.tool_runtime.names)


def _catalog_counts(built) -> dict:
    disc = built.discovery
    if disc is None:
        return {"tools_total": 0, "deferred": 0, "skills": 0}
    tools = disc.catalog.tools
    return {
        "tools_total": len(tools),
        "deferred": sum(1 for t in tools if not t.activated),
        "skills": len(disc.catalog.skills),
        "namespaces": len(disc.catalog.namespaces),
    }


async def main() -> None:
    if not (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ):
        raise SystemExit("GEMINI_API_KEY required")

    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    model = GeminiProvider(
        model=model_name,
        api_key=os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY"),
    )
    print(f"model={model_name}")
    print(f"workspace={ROOT}")

    agent = create_deep_agent(
        model,
        profile="research",
        workspace=ROOT,
        session_id="gemini-gate",
        web_search=True,
        url_fetch=True,
        citations=True,
        images=True,
        documents=True,
        think_tool=True,
        enable_task_tool=False,  # keep gate focused / cheaper
        use_llm_summarizer=False,
        max_tool_iterations=28,
        modalities="text+image",
        discovery=True,
        debug=os.environ.get("DEEP_DEBUG", "") in {"1", "true", "yes"},
        instructions=(
            "You are running a live gate. First call load_skill('research'). "
            "Then research the topic briefly (web_search at most twice). "
            "register_source at least once. Write reports/gate.md with a short "
            "brief and bibliography. Prefer text tools; only activate image/PDF "
            "tools if clearly needed via search_tools/activate_tool."
        ),
    )

    built = agent.build()
    advertised = _schema_count(built)
    catalog = _catalog_counts(built)
    print("--- schema budget ---")
    print(f"advertised_tools={advertised}")
    print(f"catalog_tools_total={catalog['tools_total']}")
    print(f"catalog_deferred={catalog['deferred']}")
    print(f"catalog_skills={catalog['skills']}")
    print(f"core_allowlist_size={len(DEEP_DISCOVERY_CORE_TOOLS)}")
    all_eager_preview = max(catalog["tools_total"], advertised + catalog["deferred"])
    reduction_preview = (
        1.0 - (advertised / all_eager_preview) if all_eager_preview else 0.0
    )
    print(f"schema_reduction_vs_all_eager={reduction_preview:.0%}")

    topic = os.environ.get(
        "DEEP_RESEARCH_TOPIC",
        "What is progressive disclosure for agent skills and tools?",
    )
    prompt = (
        f"Research topic: {topic}\n\n"
        "Constraints: load_skill('research') first; at most 2 web_search calls; "
        "fetch/extract one primary page if useful; register_source >= 1; "
        "write reports/gate.md; then stop with a short final answer."
    )

    t0 = time.monotonic()
    result = await agent.arun(prompt)
    elapsed = time.monotonic() - t0

    stop = None
    meta = getattr(result, "metadata", None) or {}
    if isinstance(meta, dict):
        stop = meta.get("stop_reason")
    stop_kind = getattr(stop, "kind", None) or stop
    text = (result.output.text() or "").strip()
    report = ROOT / "reports" / "gate.md"
    any_report = False
    reports_dir = ROOT / "reports"
    if reports_dir.is_dir():
        any_report = any(p.is_file() for p in reports_dir.rglob("*"))
    sources_path = ROOT / "sources.json"
    sources = []
    if sources_path.is_file():
        try:
            data = json.loads(sources_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                sources = data
            elif isinstance(data, dict):
                sources = data.get("sources") or []
        except (OSError, json.JSONDecodeError, TypeError):
            sources = []

    verification = getattr(result, "verification", None)
    accept_ok = None
    if verification is not None:
        accept_ok = bool(getattr(verification, "ok", verification))

    # Fallback accept check matching make_research_accept
    if accept_ok is None:
        accept_ok = any_report and len(sources) >= 1

    all_eager = max(catalog["tools_total"], advertised + catalog["deferred"])
    reduction = 1.0 - (advertised / all_eager) if all_eager else 0.0

    print("--- run result ---")
    print(f"elapsed_s={elapsed:.1f}")
    print(f"stop_reason={stop_kind}")
    print(f"accept_ok={accept_ok}")
    print(f"report_exists={report.is_file() or any_report}")
    print(f"sources_count={len(sources)}")
    print(f"final_text_chars={len(text)}")
    if text:
        print("--- final text (trim) ---")
        print(text[:800])
    if report.is_file():
        print("--- reports/gate.md (trim) ---")
        print(report.read_text(encoding="utf-8")[:1000])
    elif any_report:
        for p in sorted(reports_dir.rglob("*")):
            if p.is_file():
                print(f"--- {p.relative_to(ROOT)} (trim) ---")
                print(p.read_text(encoding="utf-8")[:1000])
                break

    sk = str(stop_kind).lower()
    gate = {
        "model": model_name,
        "advertised_tools": advertised,
        "catalog_tools_total": catalog["tools_total"],
        "catalog_deferred": catalog["deferred"],
        "schema_reduction_vs_all_eager": round(reduction, 3),
        "elapsed_s": round(elapsed, 1),
        "stop_reason": stop_kind,
        "accept_ok": accept_ok,
        "report_exists": bool(report.is_file() or any_report),
        "sources_count": len(sources),
        "pass": bool(
            accept_ok
            and (report.is_file() or any_report)
            and len(sources) >= 1
            and ("final" in sk or sk in {"", "none"})
        ),
    }
    # Prefer explicit final; allow None when deliverable+accept already prove completion
    if "final" not in sk and sk not in {"", "none"}:
        gate["pass"] = False

    out_json = ROOT / "gate_result.json"
    out_json.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print("--- gate_result ---")
    print(json.dumps(gate, indent=2))
    if not gate["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
