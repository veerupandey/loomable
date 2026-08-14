"""Deep code — index a repo, search it, fix a bug, verify in the sandbox.

USE WHEN: The agent should understand and change a codebase. ``profile="code"``
loads the coding skill, indexes ``repo=`` (zvec when installed, in-memory
otherwise), and turns on ``run_python`` / ``run_shell``.

``arun()`` builds the agent. You do not call ``agent.build()``.

Requires a live LLM key — see ``.env.example``.

Run::

    python examples/deep_agent/02_code.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import create_deep_agent

ROOT = Path(__file__).resolve().parent / ".workspace_code"
REPO = ROOT / "shop"


def _seed_repo() -> Path:
    pkg = REPO / "shop"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "pricing.py").write_text(
        '"""Checkout pricing helpers."""\n\n'
        "def apply_discount(amount_cents: int, pct: int) -> int:\n"
        '    """Discount ``amount_cents`` by ``pct`` percent.\n\n'
        "    Round the *final price* half away from zero (not truncate).\n"
        '    """\n'
        "    # Bug: integer truncation of the discount, not half-up of the price.\n"
        "    return amount_cents - (amount_cents * pct // 100)\n",
        encoding="utf-8",
    )
    (pkg / "cart.py").write_text(
        "from shop.pricing import apply_discount\n\n"
        "def checkout_total(item_cents: list[int], discount_pct: int = 0) -> int:\n"
        "    return apply_discount(sum(item_cents), discount_pct)\n",
        encoding="utf-8",
    )
    return REPO


def _tool_names(result) -> list[str]:
    names: list[str] = []
    for outcome in result.tool_activity or []:
        meta = (outcome.result.metadata or {}) if outcome.result else {}
        name = meta.get("tool_name")
        if name:
            names.append(str(name))
    return names


async def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    repo = _seed_repo()
    agent = create_deep_agent(
        require_provider(),
        profile="code",
        repo=repo,
        workspace=repo,
    )
    result = await agent.arun(
        "In shop/pricing.py, apply_discount(199, 15) must return 169 "
        "(199 * 0.85 = 169.15, round half away from zero). The current "
        "integer truncation returns 170. Fix the function, add a test under "
        "tests/, and run it with the sandbox until it passes."
    )
    print(result.output.text() or "(no final text)")
    print("tools:", ", ".join(_tool_names(result)) or "(none)")
    pricing = repo / "shop" / "pricing.py"
    print("\n--- shop/pricing.py ---")
    print(pricing.read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
