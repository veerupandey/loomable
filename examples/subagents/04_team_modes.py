"""Team convenience class — explicit orchestration modes.

USE WHEN: You want named Team modes without writing custom parent
agent instructions.

Team modes: ``coordinate`` (all + synthesize), ``route`` (LLM picks one),
``broadcast`` (all same input), ``sequential`` (chain in order).

``mode="route"`` is an **LLM specialist picker**, not Workflow control flow.
For Workflow forks see ``Workflow.branch`` (``advanced/02_workflow_branch.py``)
or ``Workflow.route`` (``patterns/08_route_command.py``). Focused route-only
demo: ``patterns/04_router.py``.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable.agent import Agent, Team

provider = require_provider()

# --- Define team members ---

optimist = Agent(
    model=provider,
    role="Optimist Analyst",
    goal="Find the positive aspects and opportunities",
)

pessimist = Agent(
    model=provider,
    role="Risk Analyst",
    goal="Identify risks, downsides, and potential failures",
)

pragmatist = Agent(
    model=provider,
    role="Pragmatic Advisor",
    goal="Give balanced, actionable recommendations",
)

# --- Coordinate: all members work, coordinator synthesizes ---

coord = Team(
    members=[optimist, pessimist, pragmatist],
    model=provider,
    mode="coordinate",
)

result = asyncio.run(
    coord.arun("Should a startup adopt microservices architecture from day one?")
)

from loomable.display import pp, delegation_outputs

print("=== Team Coordinate Mode ===")
pp(result)

outputs = delegation_outputs(result)
for member, text in outputs.items():
    print(f"\n--- {member} ---")
    print(text[:150])

# --- Route: pick the single best member ---

router = Team(
    members=[optimist, pessimist, pragmatist],
    model=provider,
    mode="route",
)
routed = asyncio.run(
    router.arun("List only the top risk of adopting microservices on day one.")
)
print("\n=== Team Route Mode ===")
pp(routed)
