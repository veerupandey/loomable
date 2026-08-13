"""Phase 1c — Multimodal: image input + tool image output.

Agent receives a dashboard spike screenshot, may generate a severity badge
image via tool, and writes a multimodal-aware brief + structured finding.
"""

from __future__ import annotations

import asyncio
import json
import struct
import zlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from loomable.agent import Agent, Image, tool
from loomable.toolkits import FileTools

from _common import ESCALATION_EMAIL, FIXTURES, OUTPUT, make_provider
from build_fixtures import main as build_fixtures


def _badge_png(color: tuple[int, int, int] = (200, 40, 40)) -> bytes:
    """Tiny solid-color PNG used as a severity badge."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    w = h = 64
    r, g, b = color
    rows = []
    for _ in range(h):
        rows.append(bytes([0]) + bytes([r, g, b]) * w)
    raw = b"".join(rows)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


@tool
def render_severity_badge(severity: str) -> Image:
    """Render a simple severity badge image for the war-room channel."""
    colors = {
        "SEV-1": (220, 40, 40),
        "SEV-2": (230, 140, 20),
        "SEV-3": (230, 200, 40),
        "SEV-4": (40, 160, 80),
    }
    png = _badge_png(colors.get(severity.upper(), (120, 120, 120)))
    # Persist for humans; framework also tracks Image on RunResult
    out = OUTPUT / f"badge_{severity.upper().replace('-', '')}.png"
    out.write_bytes(png)
    return Image(content=png, format="png")


@tool
def annotate_dashboard_finding(summary: str) -> str:
    """Record a short finding about the dashboard image for the packet."""
    return json.dumps({"dashboard_finding": summary})


class VisualTriage(BaseModel):
    severity: Literal["SEV-1", "SEV-2", "SEV-3", "SEV-4"]
    what_the_chart_shows: str
    likely_impact: str
    badge_rendered: bool
    next_checks: list[str] = Field(default_factory=list)


async def main() -> None:
    build_fixtures()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dash = FIXTURES / "dashboard_spike.png"
    assert dash.exists(), "run build_fixtures.py first"

    agent = Agent(
        model=make_provider(),
        role="War-room Visual Triage Analyst",
        goal="Interpret ops dashboards and emit severity artifacts",
        instructions=(
            "You will receive a dashboard screenshot. Describe the anomaly concretely. "
            "Use render_severity_badge for the chosen severity. "
            "Use annotate_dashboard_finding once. "
            "write_file output/visual_triage.md with your analysis. "
            "Final answer MUST be VisualTriage JSON only with keys: "
            "severity, what_the_chart_shows, likely_impact, badge_rendered, next_checks."
        ),
        tools=[
            FileTools(base_dir=str(Path(__file__).resolve().parent)),
            render_severity_badge,
            annotate_dashboard_finding,
        ],
        multimodal=True,
        feedback_media=True,
        response_model=VisualTriage,
        max_tool_iterations=12,
    )

    result = await agent.arun(
        "Analyze this settlement error dashboard for the BharatNova escalation. "
        "Choose severity, render a badge, annotate the finding, write "
        "output/visual_triage.md, and return VisualTriage JSON.\n\n"
        f"Context email:\n{ESCALATION_EMAIL}",
        images=[str(dash)],
    )

    print("\n======== MULTIMODAL RESULT ========\n")
    print(result.output.text()[:2500])
    print(f"\nimages on result: {len(getattr(result, 'images', []) or [])}")
    print(f"tools used: {len(result.tool_activity or [])}")

    triage = result.structured
    assert isinstance(triage, VisualTriage)
    print("\n======== VisualTriage ========\n")
    print(triage.model_dump_json(indent=2))

    md = OUTPUT / "visual_triage.md"
    # Soft requirement — agent should write it; record issue if not
    if not md.exists():
        print("[warn] output/visual_triage.md was not written")
    else:
        print("\n======== visual_triage.md (head) ========\n")
        print(md.read_text(encoding="utf-8")[:1500])

    badges = list(OUTPUT.glob("badge_*.png"))
    print(f"badge files: {[b.name for b in badges]}")
    if not badges and not (getattr(result, "images", None) or []):
        print("[warn] no badge image produced via tool or result.images")
    print("[ok] multimodal phase completed (check warns)")


if __name__ == "__main__":
    asyncio.run(main())
