"""Loops and Iteration — All Loop patterns in one file

Shows every way to loop in loomable:

1. Loop with body + verifier (classic API)
   - Single agent retries until a verifier function passes
   - Good for: format validation, length checks, quality gates

2. Loop with steps + end_condition (new Workflow API)
   - Multi-step iteration with a declarative end condition
   - Good for: draft/edit cycles, iterative refinement

3. Loop with tool-using agents
   - Agent can use tools inside the loop to check its own work
   - Good for: self-correcting agents, constraint satisfaction
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool
from loomable.content import AgentOutput
from loomable.agent.context import RunContext
from loomable.flow import Step, Workflow, Loop
from loomable.providers.openai import AzureOpenAIProvider


provider = AzureOpenAIProvider()


# ============================================================================
# Example 1: Classic Loop — body + verifier
# ============================================================================

print("=" * 60)
print("EXAMPLE 1: Classic Loop — body agent + verifier function")
print("=" * 60)

haiku_agent = Agent(
    model=provider,
    instructions=(
        "Write a haiku (exactly 3 lines, 5-7-5 syllable pattern). "
        "Output ONLY the 3 lines of the haiku, nothing else."
    ),
)


def haiku_verifier(output: AgentOutput, ctx: RunContext) -> bool:
    """Verify the output has exactly 3 non-empty lines."""
    lines = [l for l in output.text().strip().split("\n") if l.strip()]
    return len(lines) == 3


loop = Loop(
    body=haiku_agent,
    verifier=haiku_verifier,
    max_iterations=3,
)

result = asyncio.run(loop.arun("Write a haiku about programming."))
print("Final haiku:")
print(result.output.text())
print(f"Iterations used: {result.metadata.get('loop_iterations', '?')}")
print(f"Verified: {result.metadata.get('loop_verified', '?')}\n")


# ============================================================================
# Example 2: Loop with steps + end_condition (Workflow API)
# ============================================================================

print("=" * 60)
print("EXAMPLE 2: Loop with steps — multi-step iteration")
print("=" * 60)

drafter = Agent(
    model=provider,
    instructions="Take research notes and write a single coherent paragraph (3-4 sentences).",
)

editor = Agent(
    model=provider,
    instructions="Polish the draft for clarity and impact. Keep it to 2-3 sentences.",
)

researcher = Agent(
    model=provider,
    instructions="You are a researcher. List 3-5 key facts about the topic. Be concise.",
)

refine_loop = Loop(
    steps=[
        Step("draft", drafter),
        Step("edit", editor),
    ],
    end_condition=lambda result: len(result.output.text()) < 200,
    max_iterations=3,
)

loop_workflow = Workflow(
    name="iterative_writing",
    steps=[
        Step("research", researcher),
        refine_loop,
    ],
)

result = asyncio.run(loop_workflow.arun("The rise of electric vehicles"))
print(f"Iterations: {result.metadata.get('loop_iterations', 'N/A')}")
print(f"Output:\n{result.output.text()}\n")


# ============================================================================
# Example 3: Loop with tool-using agent
# ============================================================================

print("=" * 60)
print("EXAMPLE 3: Loop with tool-using agent")
print("=" * 60)


@tool
def word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())


@tool
def char_count(text: str) -> int:
    """Count the number of characters in a text."""
    return len(text)


def length_check(output: AgentOutput, ctx: RunContext) -> bool:
    """Verify the answer is between 20 and 40 words."""
    words = len(output.text().split())
    return 20 <= words <= 40


summarizer = Agent(
    model=provider,
    instructions=(
        "Summarize topics. Your summary MUST be between 20 and 40 words. "
        "Use the word_count tool to check your answer length before finalizing. "
        "If too short or too long, adjust."
    ),
    tools=[word_count, char_count],
)

tool_loop = Loop(body=summarizer, verifier=length_check, max_iterations=3)

result = asyncio.run(tool_loop.arun("Explain the theory of relativity in 20-40 words."))
print("Summary:", result.output.text())
print(f"Word count: {len(result.output.text().split())}")
print(f"Iterations: {result.metadata.get('loop_iterations')}")
print(f"Verified: {result.metadata.get('loop_verified')}")
