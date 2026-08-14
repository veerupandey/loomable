"""Team Convenience Class — Explicit orchestration modes.

USE WHEN: You want named orchestration patterns without writing
custom parent agent instructions.

Team modes: coordinate (all + synthesize), route (pick best),
broadcast (all same input), sequential (chain in order).
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

# --- Coordinate mode: all members work, coordinator synthesizes ---

team = Team(
    members=[optimist, pessimist, pragmatist],
    model=provider,
    mode="coordinate",
)

result = asyncio.run(team.arun(
    "Should a startup adopt microservices architecture from day one?"
))

# Pretty-print with delegation breakdown
from loomable.display import pp, delegation_outputs

print("=== Team Coordinate Mode ===")
pp(result)

# Access what each member said
outputs = delegation_outputs(result)
for member, text in outputs.items():
    print(f"\n--- {member} ---")
    print(text[:150])
