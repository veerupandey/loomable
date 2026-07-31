"""29 — FlowClass: Decorator-Driven Event Workflows

The FlowClass API provides a class-based, decorator-driven approach for
defining complex event-driven workflows — similar to CrewAI's Flow or
Agno's Workflow patterns. Methods decorated with @start, @listen, and
@router are compiled into an execution graph automatically.

Demonstrates:
- @start(): entry-point methods receiving initial input
- @listen(source): methods triggered by another method's output
- @router(source): conditional routing based on return value
- Fan-out: multiple listeners on the same source (parallel)
- agents attribute: pre-configured agents accessible in methods
- explain(): inspect the compiled topology
- Composability: FlowClass as a Runnable in other flows
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import FlowClass, start, listen, router, sequential, Loop
from loomable.providers.openai import AzureOpenAIProvider


# ============================================================================
# Setup
# ============================================================================

provider = AzureOpenAIProvider()


# ============================================================================
# Example 1: Simple Linear FlowClass
# ============================================================================

print("=" * 60)
print("EXAMPLE 1: Simple Linear FlowClass")
print("=" * 60)


class ArticleFlow(FlowClass):
    """A simple 3-step article pipeline using decorators."""

    agents = {
        "researcher": Agent(
            model=provider,
            instructions="List 3-5 key facts about the topic. Be concise.",
        ),
        "writer": Agent(
            model=provider,
            instructions="Write a coherent paragraph from the research notes.",
        ),
        "editor": Agent(
            model=provider,
            instructions="Polish for clarity. Keep to 2-3 sentences.",
        ),
    }

    @start()
    async def research(self, input):
        """Entry point: run the researcher agent."""
        result = await self.agents["researcher"].arun(input)
        return result.output.text()

    @listen("research")
    async def write(self, input):
        """Triggered after research completes — receives research output."""
        result = await self.agents["writer"].arun(input)
        return result.output.text()

    @listen("write")
    async def edit(self, input):
        """Triggered after writing — polishes the draft."""
        result = await self.agents["editor"].arun(input)
        return result.output.text()


# Inspect topology before running
flow = ArticleFlow()
plan = flow.explain()
print(f"\nTopology: {plan.original_nodes}")
print(f"Edges: {plan.original_edges}")

result = asyncio.run(flow.kickoff("The history of open-source software"))
print(f"\nFinal output:\n{result.output.text()}\n")


# ============================================================================
# Example 2: Fan-Out — Parallel Analysis from One Source
# ============================================================================

print("=" * 60)
print("EXAMPLE 2: Fan-Out — Parallel Analysis")
print("=" * 60)


class AnalysisFlow(FlowClass):
    """One start method fans out to multiple parallel analysts."""

    agents = {
        "researcher": Agent(model=provider, instructions="Summarize the topic in 2 sentences."),
        "pros_analyst": Agent(model=provider, instructions="List 3 pros. One sentence each."),
        "cons_analyst": Agent(model=provider, instructions="List 3 cons. One sentence each."),
        "opportunity_analyst": Agent(model=provider, instructions="List 3 opportunities. One sentence each."),
    }

    @start()
    async def research(self, input):
        result = await self.agents["researcher"].arun(input)
        return result.output.text()

    @listen("research")
    async def analyze_pros(self, input):
        """Runs in parallel with other listeners."""
        result = await self.agents["pros_analyst"].arun(input)
        return result.output.text()

    @listen("research")
    async def analyze_cons(self, input):
        """Runs in parallel with other listeners."""
        result = await self.agents["cons_analyst"].arun(input)
        return result.output.text()

    @listen("research")
    async def analyze_opportunities(self, input):
        """Runs in parallel with other listeners."""
        result = await self.agents["opportunity_analyst"].arun(input)
        return result.output.text()


analysis = AnalysisFlow()
plan = analysis.explain()
print(f"\nFan-out topology: {plan.original_nodes}")
print(f"Edges: {plan.original_edges}")
# research -> analyze_pros, research -> analyze_cons, research -> analyze_opportunities

result = asyncio.run(analysis.kickoff("Adopting Kubernetes in small startups"))
print(f"\nAnalysis complete. Output:\n{result.output.text()}\n")


# ============================================================================
# Example 3: Router — Conditional Branching
# ============================================================================

print("=" * 60)
print("EXAMPLE 3: Router — Conditional Branching")
print("=" * 60)


class ContentRouter(FlowClass):
    """Routes content to different processing pipelines based on type."""

    agents = {
        "classifier": Agent(
            model=provider,
            instructions=(
                "Classify the input as either 'technical' or 'creative'. "
                "Respond with exactly one word: technical or creative."
            ),
        ),
        "tech_writer": Agent(
            model=provider,
            instructions="Write a technical explanation with precise terminology. 2-3 sentences.",
        ),
        "creative_writer": Agent(
            model=provider,
            instructions="Write a creative, engaging narrative. 2-3 sentences.",
        ),
    }

    @start()
    async def classify(self, input):
        """Classify the input content type."""
        result = await self.agents["classifier"].arun(input)
        return result.output.text()

    @router("classify")
    async def route_content(self, input):
        """Route to appropriate handler based on classification."""
        text = input.lower().strip() if isinstance(input, str) else str(input).lower()
        if "technical" in text:
            return "handle_technical"
        return "handle_creative"

    @listen("route_content")
    async def handle_technical(self, input):
        result = await self.agents["tech_writer"].arun(input)
        return result.output.text()

    @listen("route_content")
    async def handle_creative(self, input):
        result = await self.agents["creative_writer"].arun(input)
        return result.output.text()


router_flow = ContentRouter()
plan = router_flow.explain()
print(f"\nRouter topology: {plan.original_nodes}")
print(f"Edges: {plan.original_edges}")

result = asyncio.run(router_flow.kickoff("Explain how neural networks learn"))
print(f"\nRouted output:\n{result.output.text()}\n")


# ============================================================================
# Example 4: FlowClass as a Runnable — Composing with Existing APIs
# ============================================================================

print("=" * 60)
print("EXAMPLE 4: FlowClass as Runnable — Composing with Loops")
print("=" * 60)


class DraftFlow(FlowClass):
    """A small FlowClass that can be used as a Runnable in other constructs."""

    agents = {
        "drafter": Agent(model=provider, instructions="Write a 3-sentence paragraph about the topic."),
        "critic": Agent(model=provider, instructions="Suggest one specific improvement. Be brief."),
    }

    @start()
    async def draft(self, input):
        result = await self.agents["drafter"].arun(input)
        return result.output.text()

    @listen("draft")
    async def critique(self, input):
        result = await self.agents["critic"].arun(f"Critique this: {input}")
        return result.output.text()


# Use the FlowClass inside a Loop for iterative refinement
draft_flow = DraftFlow()

# FlowClass satisfies Runnable, so it works as a Loop body
refinement_loop = Loop(body=draft_flow, max_iterations=2)

# Or compose with sequential()
editor_agent = Agent(model=provider, instructions="Final polish. Keep to 2 sentences.")
pipeline = sequential(draft_flow, editor_agent)

result = asyncio.run(pipeline.arun("The future of space exploration"))
print(f"\nComposed output:\n{result.output.text()}\n")


# ============================================================================
# Example 5: Complex Multi-Stage Pipeline
# ============================================================================

print("=" * 60)
print("EXAMPLE 5: Complex Multi-Stage Pipeline")
print("=" * 60)


class ResearchPipeline(FlowClass):
    """A complex research pipeline with multiple stages and fan-out."""

    agents = {
        "topic_expander": Agent(
            model=provider,
            instructions="Expand the topic into 3 specific sub-questions. Number them 1-3.",
        ),
        "deep_researcher": Agent(
            model=provider,
            instructions="Research this specific question. Provide 2-3 key findings.",
        ),
        "synthesizer": Agent(
            model=provider,
            instructions="Synthesize all research into a coherent 3-4 sentence summary.",
        ),
    }

    @start()
    async def expand_topic(self, input):
        """Break the topic into sub-questions."""
        result = await self.agents["topic_expander"].arun(input)
        return result.output.text()

    @listen("expand_topic")
    async def research_depth(self, input):
        """Deep-dive research on the expanded questions."""
        result = await self.agents["deep_researcher"].arun(input)
        return result.output.text()

    @listen("research_depth")
    async def synthesize(self, input):
        """Pull everything together into a final summary."""
        result = await self.agents["synthesizer"].arun(input)
        return result.output.text()


pipeline = ResearchPipeline()
plan = pipeline.explain()
print(f"\nPipeline topology: {plan.original_nodes}")
print(f"Edges: {plan.original_edges}")

result = asyncio.run(pipeline.kickoff("How will AI change education in the next decade?"))
print(f"\nSynthesized research:\n{result.output.text()}\n")
