"""17 — Flow with Shared TieredMemoryStore

Demonstrates the TieredMemoryStore shared across flow nodes:
- Node A writes research findings to EPISODIC memory
- Node B recalls them to build on previous work
- Memory persists across the flow's session
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.flow import sequential, TieredMemoryStore, Tier


# --- Nodes as plain async functions ---


async def research_node(input, *, memory=None, **kwargs):
    """Gather research findings and store in episodic memory."""
    findings = [
        "Python 3.12 introduced performance improvements of 5-10%",
        "Type hints are now used by 70% of Python projects",
        "asyncio adoption has grown 3x since 2020",
    ]
    if memory:
        for fact in findings:
            await memory.write(fact, tier=Tier.EPISODIC, source="research")

    return f"Researched: stored {len(findings)} facts about Python trends."


async def analysis_node(input, *, memory=None, **kwargs):
    """Recall research from memory and produce analysis."""
    recalled = []
    if memory:
        results = await memory.recall("Python", tiers=[Tier.EPISODIC], k=5)
        recalled = [r["record"] for r in results]

    if recalled:
        analysis = "Analysis based on research:\n"
        for fact in recalled:
            analysis += f"  • {fact}\n"
        analysis += "\nConclusion: Python continues to evolve rapidly with a strong community."
    else:
        analysis = "No prior research found to analyze."

    # Store the conclusion in working memory
    if memory:
        await memory.write("Python is evolving rapidly with strong community adoption.", tier=Tier.WORKING)

    return analysis


async def report_node(input, *, memory=None, **kwargs):
    """Produce final report using all accumulated memory."""
    all_facts = []
    if memory:
        episodic = await memory.recall("Python", tiers=[Tier.EPISODIC], k=10)
        working = await memory.recall("Python", tiers=[Tier.WORKING], k=5)
        all_facts = [r["record"] for r in episodic + working]

    return f"=== Report ===\nBased on {len(all_facts)} data points:\n{input}"


# --- Build flow with shared memory ---

shared_memory = TieredMemoryStore(session_id="flow-memory-demo")

pipeline = sequential(
    research_node,
    analysis_node,
    report_node,
    session_id="flow-memory-demo",
    memory=shared_memory,
)

result = asyncio.run(pipeline.arun("Analyze Python ecosystem trends"))
print(result.output.text())

# Show what's in memory after the flow
print("\n--- Memory contents ---")
episodic = asyncio.run(shared_memory.recall("Python", tiers=[Tier.EPISODIC], k=10))
print(f"Episodic tier ({len(episodic)} records):")
for r in episodic:
    print(f"  - {r['record']}")

working = asyncio.run(shared_memory.recall("evolving", tiers=[Tier.WORKING], k=5))
print(f"\nWorking tier ({len(working)} records):")
for r in working:
    print(f"  - {r['record']}")
