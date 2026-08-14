"""Router — intent-based routing via Team(mode="route").

USE WHEN: Different inputs should go to different specialist Agents.
The coordinator picks the single best member and delegates once.

For deterministic keyword branching, see ``advanced/02_workflow_branch.py``
(``Workflow.branch``).
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import Agent, Team

provider = require_provider()

code_agent = Agent(
    model=provider,
    role="Code Assistant",
    goal="Write and explain code",
    instructions="Help with programming questions. Include code examples.",
)

math_agent = Agent(
    model=provider,
    role="Math Tutor",
    goal="Solve math problems step by step",
    instructions="Solve math problems showing your work.",
)

general_agent = Agent(
    model=provider,
    role="General Assistant",
    goal="Answer general knowledge questions",
    instructions="Answer clearly and concisely.",
)

team = Team(
    members=[code_agent, math_agent, general_agent],
    model=provider,
    mode="route",
)

result = asyncio.run(team.arun("Solve the quadratic equation x^2 - 5x + 6 = 0"))
print(result.output.text())
