"""Fan-Out — same task, multiple perspectives via Workflow.parallel.

USE WHEN: You want diverse viewpoints on the same question
without any single agent dominating the analysis.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import Agent, Workflow
from loomable.display import pp, step_outputs

provider = require_provider()

frontend_expert = Agent(
    model=provider,
    role="Frontend Expert",
    goal="Evaluate from a frontend development perspective",
    instructions="Analyze the topic from a frontend/UX perspective. 2-3 sentences.",
)

backend_expert = Agent(
    model=provider,
    role="Backend Expert",
    goal="Evaluate from a backend/infrastructure perspective",
    instructions="Analyze the topic from a backend/scalability perspective. 2-3 sentences.",
)

security_expert = Agent(
    model=provider,
    role="Security Expert",
    goal="Evaluate from a security perspective",
    instructions="Analyze the topic from a security perspective. 2-3 sentences.",
)

fan_out = Workflow("fan-out").parallel(
    frontend=frontend_expert,
    backend=backend_expert,
    security=security_expert,
)

result = asyncio.run(fan_out.arun("Should we adopt GraphQL for our public API?"))

pp(result)

branches = step_outputs(result)
for branch_name, text in branches.items():
    print(f"\n--- {branch_name} ---")
    print(text)
