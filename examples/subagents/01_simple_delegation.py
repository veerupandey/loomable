"""Subagents: An agent that delegates to specialists.

USE WHEN: Your task needs multiple perspectives or specializations
that a single agent's instructions can't cover.

The parent agent sees subagents as tools it can call.
It decides at runtime who to delegate to and in what order.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable.agent import Agent

provider = require_provider()

security_reviewer = Agent(
    model=provider,
    role="Security Reviewer",
    goal="Find security vulnerabilities in code",
    instructions="Focus on OWASP Top 10 issues. Be specific.",
)

perf_reviewer = Agent(
    model=provider,
    role="Performance Engineer",
    goal="Identify performance bottlenecks and suggest optimizations",
    instructions="Focus on algorithmic complexity and resource usage.",
)

lead = Agent(
    model=provider,
    role="Tech Lead",
    goal="Coordinate code reviews and prioritize issues",
    instructions="Get both reviews, then prioritize the top 3 issues to fix first.",
    subagents=[security_reviewer, perf_reviewer],
)

result = asyncio.run(lead.arun(
    "Review this code:\n"
    "def get_users(db, filter):\n"
    "    query = f'SELECT * FROM users WHERE {filter}'\n"
    "    results = db.execute(query)\n"
    "    return [serialize(r) for r in results]"
))

# Pretty-print with delegation breakdown
from loomable.display import pp, delegation_outputs

pp(result)

# Access individual subagent outputs by name
outputs = delegation_outputs(result)
for name, text in outputs.items():
    print(f"\n--- {name} said ---")
    print(text[:200])
