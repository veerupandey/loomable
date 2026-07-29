"""Real-world stress tests — Simple → Tough → Tougher → Toughest.

Each scenario mimics a genuine production situation an agent would face.
Providers are scripted to behave like real LLMs (returning structured tool
calls, handling errors realistically, streaming multi-step workflows).

Run with: uv run pytest tests/integration/test_real_world_scenarios.py -v
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

import pytest

from loomable.agent import Agent, ModelSpec, tool, FunctionTool
from loomable.agent.context import RunContext, StopReason
from loomable.agent.events import Event, JSONTracer, NoOpEvents
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolOutcome,
    ToolResult,
)
from loomable.providers.errors import PermanentProviderError, TransientProviderError
from loomable.providers.resilient import ResilientModel, RetryPolicy

pytestmark = pytest.mark.asyncio


# ===========================================================================
# REALISTIC TOOLS — mimics what a production agent would have
# ===========================================================================


@tool
def query_database(sql: str) -> str:
    """Execute a SQL query against the customer database."""
    # Simulates real DB responses
    if "SELECT" in sql.upper() and "users" in sql.lower():
        return '{"rows": [{"id": 1, "name": "Alice", "plan": "pro"}, {"id": 2, "name": "Bob", "plan": "free"}]}'
    if "SELECT" in sql.upper() and "orders" in sql.lower():
        return '{"rows": [{"order_id": 101, "user_id": 1, "total": 299.99, "status": "shipped"}]}'
    if "UPDATE" in sql.upper():
        return '{"affected_rows": 1}'
    return '{"rows": [], "message": "No results"}'


@tool(idempotent=False)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer. THIS HAS REAL SIDE EFFECTS."""
    return f"Email sent to {to} with subject '{subject}'"


@tool
def search_knowledge_base(query: str, limit: int = 5) -> str:
    """Search the internal knowledge base for relevant articles."""
    articles = {
        "billing": '[{"title": "Billing FAQ", "content": "Refunds processed in 5-7 days..."}]',
        "api": '[{"title": "API Rate Limits", "content": "Free tier: 100 req/min, Pro: 10000 req/min"}]',
        "outage": '[{"title": "Incident Playbook", "content": "Step 1: Check status page..."}]',
    }
    for key, val in articles.items():
        if key in query.lower():
            return val
    return '[{"title": "General Help", "content": "Contact support@company.com"}]'


@tool
def get_system_metrics(service: str, timeframe: str) -> str:
    """Pull real-time metrics for a service (latency, error rate, throughput)."""
    return (
        f'{{"service": "{service}", "timeframe": "{timeframe}", '
        f'"p50_ms": 45, "p99_ms": 890, "error_rate": 0.02, "rps": 1250}}'
    )


@tool(idempotent=False)
def create_jira_ticket(title: str, description: str, priority: str) -> str:
    """Create a JIRA ticket for the engineering team."""
    return f'{{"ticket_id": "ENG-4521", "title": "{title}", "priority": "{priority}", "status": "created"}}'


@tool
def call_external_api(url: str, method: str = "GET") -> str:
    """Call an external REST API endpoint."""
    if "payments" in url:
        return '{"status": "active", "balance": 1523.40, "currency": "USD"}'
    if "inventory" in url:
        return '{"items": 42, "warehouse": "US-WEST", "last_sync": "2026-07-29T10:00:00Z"}'
    return '{"error": "endpoint_not_found"}'


@tool
async def run_deployment(service: str, version: str, environment: str) -> str:
    """Deploy a service version to an environment (takes time)."""
    await asyncio.sleep(0.02)  # Simulate deployment time
    return f'{{"deployed": true, "service": "{service}", "version": "{version}", "env": "{environment}"}}'


@tool
async def health_check(service: str) -> str:
    """Check if a service is healthy."""
    await asyncio.sleep(0.01)
    return f'{{"service": "{service}", "healthy": true, "uptime_hours": 720}}'


# Simulates a flaky external service (MCP-style tool)
class ExternalPaymentGateway:
    """Simulates an MCP tool connected to a payment gateway."""

    def __init__(self):
        self.name = "payment_gateway"
        self.description = "Process payments, check balances, issue refunds via Stripe."
        self.parameters = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["charge", "refund", "balance"]},
                "amount": {"type": "number"},
                "customer_id": {"type": "string"},
            },
            "required": ["action", "customer_id"],
        }
        self.idempotent = False
        self.call_log: list[dict] = []

    def schema(self):
        return {"type": "function", "function": {"name": self.name,
                "description": self.description, "parameters": self.parameters}}

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        self.call_log.append(args)
        action = args.get("action", "")
        cid = args.get("customer_id", "")
        if action == "balance":
            return ToolResult(content=f'{{"customer": "{cid}", "balance": 249.99}}')
        if action == "refund":
            return ToolResult(content=f'{{"refund_id": "ref_abc123", "amount": {args.get("amount", 0)}, "status": "processed"}}')
        if action == "charge":
            return ToolResult(content=f'{{"charge_id": "ch_xyz789", "amount": {args.get("amount", 0)}, "status": "succeeded"}}')
        return ToolResult(error=f"Unknown action: {action}")



# ===========================================================================
# SIMPLE — Customer support agent answers a basic question
# ===========================================================================


class TestSimple:
    """Level 1: Basic agent tasks that every production agent must handle."""

    async def test_customer_asks_billing_question(self):
        """
        SCENARIO: Customer asks "How long do refunds take?"
        EXPECTED: Agent searches KB, returns answer. No side effects.
        """
        provider = ScriptedProvider([
            # Agent decides to search the knowledge base
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="search_knowledge_base",
                                     args={"query": "billing refund timeline"})],
                usage={"input_tokens": 85, "output_tokens": 22},
            ),
            # Agent reads KB result and answers the customer
            ModelResponse(
                content="Refunds are typically processed within 5-7 business days. "
                        "You'll receive a confirmation email once it's complete.",
                usage={"input_tokens": 120, "output_tokens": 28},
            ),
        ])
        tracer = JSONTracer()
        agent = Agent(
            model=ModelSpec(provider="support-llm", provider_impl=provider),
            instructions=(
                "You are a customer support agent for a SaaS company. "
                "Always search the knowledge base before answering billing questions. "
                "Be concise and helpful."
            ),
            tools=[search_knowledge_base, query_database, send_email],
            events=tracer,
        )

        result = await agent.arun("How long do refunds take?")

        # Assertions
        assert "5-7" in result.output.text()
        assert result.metadata["stop_reason"] == "final"
        # Only 1 tool call (KB search), no email sent
        tool_events = [e for e in tracer.trace if e.kind == "tool_call"]
        assert len(tool_events) == 1
        # Event ordering: run_start first, run_end last
        assert tracer.trace[0].kind == "run_start"
        assert tracer.trace[-1].kind == "run_end"


    async def test_agent_with_think_tool_reasons_before_acting(self):
        """
        SCENARIO: Agent uses scratchpad to reason before choosing which tool.
        EXPECTED: Think tool has no side effects, reasoning improves answer.
        """
        provider = ScriptedProvider([
            # Agent thinks first
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="think", args={
                    "thought": "The user wants their order status. I should query "
                               "the database for orders by user_id. Let me check "
                               "if I need to look up the user first."
                })],
                usage={"input_tokens": 90, "output_tokens": 35},
            ),
            # Then queries DB
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c2", tool_name="query_database", args={
                    "sql": "SELECT * FROM orders WHERE user_id = 1 ORDER BY created_at DESC LIMIT 1"
                })],
                usage={"input_tokens": 140, "output_tokens": 28},
            ),
            # Final answer
            ModelResponse(
                content="Your most recent order (#101) for $299.99 has been shipped! "
                        "You should receive it within 3-5 business days.",
                usage={"input_tokens": 180, "output_tokens": 32},
            ),
        ])
        agent = Agent(
            model=ModelSpec(provider="support-llm", provider_impl=provider),
            tools=[query_database, search_knowledge_base],
            think_tool=True,
        )

        result = await agent.arun("Where's my order? I'm user #1")

        assert "shipped" in result.output.text()
        assert "$299.99" in result.output.text()
        assert result.metadata["stop_reason"] == "final"


# ===========================================================================
# TOUGH — Multi-step workflow with side effects and timeouts
# ===========================================================================


class TestTough:
    """Level 2: Multi-tool workflows, side effects, network issues."""

    async def test_customer_refund_workflow(self):
        """
        SCENARIO: Process a customer refund — look up order, verify, refund, email.
        EXPECTED: Agent calls tools in correct order, side-effecting tools called once.
        """
        payment_gw = ExternalPaymentGateway()

        provider = ScriptedProvider([
            # Step 1: Look up the customer's order
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="query_database", args={
                    "sql": "SELECT * FROM orders WHERE order_id = 101"
                })],
                usage={"input_tokens": 100, "output_tokens": 25},
            ),
            # Step 2: Issue refund via payment gateway
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c2", tool_name="payment_gateway", args={
                    "action": "refund", "amount": 299.99, "customer_id": "cust_alice"
                })],
                usage={"input_tokens": 150, "output_tokens": 30},
            ),
            # Step 3: Send confirmation email
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c3", tool_name="send_email", args={
                    "to": "alice@example.com",
                    "subject": "Your refund has been processed",
                    "body": "Hi Alice, your refund of $299.99 for order #101 is on its way."
                })],
                usage={"input_tokens": 200, "output_tokens": 35},
            ),
            # Final answer
            ModelResponse(
                content="Done! I've processed your refund of $299.99 for order #101. "
                        "You'll receive a confirmation email shortly. The refund "
                        "should appear in your account within 5-7 business days.",
                usage={"input_tokens": 250, "output_tokens": 40},
            ),
        ])
        tracer = JSONTracer()
        agent = Agent(
            model=ModelSpec(provider="support-llm", provider_impl=provider),
            instructions="You are a customer support agent who can process refunds.",
            tools=[query_database, send_email, search_knowledge_base, payment_gw],
            events=tracer,
            loop_repeat_threshold=5,
        )

        result = await agent.arun("Please refund order #101 for alice@example.com")

        assert "refund" in result.output.text().lower()
        assert "$299.99" in result.output.text()
        assert result.metadata["stop_reason"] == "final"
        # Payment gateway called exactly once (side-effecting)
        assert len(payment_gw.call_log) == 1
        assert payment_gw.call_log[0]["action"] == "refund"
        # 3 tool_call events
        tool_events = [e for e in tracer.trace if e.kind == "tool_call"]
        assert len(tool_events) == 3
        # 4 model_call events (4 model invocations)
        model_events = [e for e in tracer.trace if e.kind == "model_call"]
        assert len(model_events) == 4


    async def test_agent_handles_tool_timeout_gracefully(self):
        """
        SCENARIO: External API is slow, agent's tool times out.
        EXPECTED: Timeout error fed back, agent apologizes and offers alternative.
        """
        @tool
        async def slow_external_api(endpoint: str) -> str:
            """Call a slow third-party API."""
            await asyncio.sleep(10.0)  # Will be killed by timeout
            return "unreachable"

        provider = ScriptedProvider([
            # Agent tries the slow API
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="slow_external_api",
                                     args={"endpoint": "/v2/shipping/track"})],
                usage={"input_tokens": 80, "output_tokens": 20},
            ),
            # Sees timeout, offers alternative
            ModelResponse(
                content="I'm sorry, the shipping tracking service is currently "
                        "unavailable. I can check your order status in our database "
                        "instead, or you can try again in a few minutes.",
                usage={"input_tokens": 130, "output_tokens": 35},
            ),
        ])
        agent = Agent(
            model=ModelSpec(provider="support-llm", provider_impl=provider),
            tools=[slow_external_api, query_database],
        )
        built = agent.build()
        built.tool_timeout = 0.05  # 50ms — will timeout

        result = await built.arun("Track my package")

        assert "unavailable" in result.output.text()
        assert result.metadata["stop_reason"] == "final"
        # Verify the timeout error was fed to the model
        last_req = provider.calls[-1]
        tool_msgs = [m for m in last_req.messages if m.get("role") == "tool"]
        assert any("timed out" in str(m) for m in tool_msgs)

    async def test_resilient_agent_recovers_from_api_blip(self):
        """
        SCENARIO: LLM API has a 503 blip, agent retries and succeeds.
        EXPECTED: Transparent retry, user never sees the error.
        """
        success = ModelResponse(
            content="Here's your account summary: Pro plan, $49/month, next billing Aug 15.",
            usage={"input_tokens": 60, "output_tokens": 20},
        )
        flaky = FlakyProvider(fail_count=2, success_response=success)
        tracer = JSONTracer()

        agent = Agent(
            model=ModelSpec(provider="openai", provider_impl=flaky),
            instructions="You are a billing assistant.",
            resilience=RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.05),
            events=tracer,
        )

        result = await agent.arun("What's my billing summary?")

        assert "Pro plan" in result.output.text()
        assert "$49/month" in result.output.text()
        assert flaky.attempts == 3  # 2 failures + 1 success
        # User experience is seamless — they just get the answer


# ===========================================================================
# TOUGHER — Concurrent operations, budget exhaustion, loop detection
# ===========================================================================


class TestTougher:
    """Level 3: Concurrency stress, budget limits, runaway detection."""


    async def test_parallel_health_checks_with_concurrency_cap(self):
        """
        SCENARIO: DevOps agent checks health of 6 services simultaneously.
        EXPECTED: Concurrency capped at 3, all checks complete, no starvation.
        """
        peak = [0]
        current = [0]
        completed_services = []

        @tool
        async def check_service(service_name: str) -> str:
            """Check if a microservice is healthy."""
            current[0] += 1
            if current[0] > peak[0]:
                peak[0] = current[0]
            await asyncio.sleep(0.02)
            current[0] -= 1
            completed_services.append(service_name)
            return f'{{"service": "{service_name}", "status": "healthy", "latency_ms": {random.randint(5, 50)}}}'

        provider = ScriptedProvider([
            # Agent requests all 6 health checks at once
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id=f"c{i}", tool_name="check_service",
                             args={"service_name": svc})
                    for i, svc in enumerate(["auth", "payments", "inventory",
                                             "notifications", "search", "analytics"])
                ],
                usage={"input_tokens": 100, "output_tokens": 40},
            ),
            # Agent summarizes
            ModelResponse(
                content="All 6 services are healthy:\n"
                        "- auth: 12ms\n- payments: 23ms\n- inventory: 8ms\n"
                        "- notifications: 31ms\n- search: 15ms\n- analytics: 45ms\n"
                        "No issues detected.",
                usage={"input_tokens": 200, "output_tokens": 50},
            ),
        ])
        agent = Agent(
            model=ModelSpec(provider="devops-llm", provider_impl=provider),
            tools=[check_service],
        )
        built = agent.build()
        built.tool_concurrency = 3  # Max 3 concurrent

        result = await built.arun("Check health of all microservices")

        assert "healthy" in result.output.text()
        assert peak[0] <= 3  # Concurrency cap respected
        assert len(completed_services) == 6  # All completed

    async def test_agent_stuck_in_loop_gets_stopped(self):
        """
        SCENARIO: Agent keeps searching the same query — classic LLM loop bug.
        EXPECTED: Loop detection fires at threshold, produces a stop reason.
        """
        call_num = [0]

        class LoopingLLM:
            """Simulates an LLM stuck in a reasoning loop."""
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1
                # LLM keeps thinking it needs more info from the same source
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id=f"c{call_num[0]}",
                        tool_name="search_knowledge_base",
                        args={"query": "api rate limits"},  # Same args every time!
                    )],
                    usage={"input_tokens": 50 + call_num[0] * 10, "output_tokens": 15},
                )

        tracer = JSONTracer()
        agent = Agent(
            model=ModelSpec(provider="stuck-llm", provider_impl=LoopingLLM()),
            instructions="Help the user with API questions.",
            tools=[search_knowledge_base],
            events=tracer,
            loop_repeat_threshold=4,
        )

        result = await agent.arun("What are the API rate limits?")

        assert result.metadata["stop_reason"] == "loop_detected"
        # The 4th identical call triggered detection — it was NOT dispatched
        tool_events = [e for e in tracer.trace if e.kind == "tool_call"]
        assert len(tool_events) == 3  # 3 dispatched, 4th blocked
        # Total tokens tracked
        model_events = [e for e in tracer.trace if e.kind == "model_call"]
        assert all(e.tokens_in is not None and e.tokens_in >= 0 for e in model_events)


    async def test_token_budget_stops_expensive_research_agent(self):
        """
        SCENARIO: Research agent doing deep analysis burns through token budget.
        EXPECTED: Budget enforced, agent stops cleanly before overspending.
        """
        call_num = [0]

        class VerboseResearcher:
            """LLM that generates large tool calls and eats tokens fast."""
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1
                if not request.tools:
                    return ModelResponse(content="forced stop",
                                        usage={"input_tokens": 100, "output_tokens": 50})
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id=f"c{call_num[0]}",
                        tool_name="search_knowledge_base",
                        args={"query": f"deep research topic {call_num[0]}"},
                    )],
                    # Each call uses 2000 tokens — budget will be hit fast
                    usage={"input_tokens": 1500, "output_tokens": 500},
                )

        agent = Agent(
            model=ModelSpec(provider="research-llm", provider_impl=VerboseResearcher()),
            tools=[search_knowledge_base],
            token_budget=4000,  # Budget: 4000 tokens
            loop_repeat_threshold=50,
        )

        result = await agent.arun("Do comprehensive research on our API architecture")

        assert result.metadata["stop_reason"] == "token_budget"
        # Should have stopped after 2 calls (2000 tokens each = 4000 total)
        assert call_num[0] == 2

    async def test_non_idempotent_tool_never_double_fires(self):
        """
        SCENARIO: LLM tries to send the same JIRA ticket twice.
        EXPECTED: Second identical call is blocked, ticket created only once.
        """
        ticket_count = [0]

        @tool(idempotent=False)
        def create_ticket(title: str, priority: str) -> str:
            """Create a JIRA ticket."""
            ticket_count[0] += 1
            return f'{{"id": "ENG-{1000 + ticket_count[0]}", "title": "{title}"}}'

        provider = ScriptedProvider([
            # First: create ticket
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c1", tool_name="create_ticket",
                                     args={"title": "Fix login bug", "priority": "high"})],
                usage={"input_tokens": 60, "output_tokens": 20},
            ),
            # LLM tries to create the SAME ticket again (confused)
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="c2", tool_name="create_ticket",
                                     args={"title": "Fix login bug", "priority": "high"})],
                usage={"input_tokens": 80, "output_tokens": 20},
            ),
            # Final answer
            ModelResponse(
                content="I've created ticket ENG-1001 for the login bug fix.",
                usage={"input_tokens": 100, "output_tokens": 15},
            ),
        ])
        agent = Agent(
            model=ModelSpec(provider="devops-llm", provider_impl=provider),
            tools=[create_ticket],
            loop_repeat_threshold=10,
        )

        result = await agent.arun("Create a high-priority ticket for the login bug")

        assert ticket_count[0] == 1  # Created exactly once
        assert "ENG-1001" in result.output.text()


# ===========================================================================
# TOUGHEST — Full production chaos: cascading failures, cancellation,
#            pinned facts under compaction, multi-feature integration
# ===========================================================================


class TestToughest:
    """Level 4: Production nightmare scenarios — chaos engineering for agents."""


    async def test_incident_response_under_cascading_failures(self):
        """
        SCENARIO: Production incident. Agent must:
          1. Check metrics (succeeds)
          2. Call external API (times out — service is down!)
          3. See timeout, pivot to creating a ticket
          4. Send alert email
        Meanwhile the LLM API itself is flaky (503s).

        EXPECTED: Agent recovers from LLM flakiness via retry, handles tool
        timeout gracefully, completes the incident workflow.
        """
        @tool
        async def check_status_page(service: str) -> str:
            """Check the external status page."""
            await asyncio.sleep(10.0)  # Simulates the status page being down too
            return "unreachable"

        call_num = [0]
        flaky_count = [0]
        successful_calls = [0]

        class FlakyIncidentLLM:
            """LLM that 503s on the 2nd call, then recovers."""
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1

                # Simulate a 503 on raw call #2 (LLM API having issues too)
                if call_num[0] == 2:
                    flaky_count[0] += 1
                    raise TransientProviderError("openai", status_code=503, retry_after=0.01)

                successful_calls[0] += 1
                messages_text = str(request.messages)

                # First successful call: check metrics
                if successful_calls[0] == 1:
                    return ModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="c1", tool_name="get_system_metrics",
                                             args={"service": "payments", "timeframe": "5m"})],
                        usage={"input_tokens": 80, "output_tokens": 20},
                    )
                # Second successful call: try status page
                if successful_calls[0] == 2:
                    return ModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="c2", tool_name="check_status_page",
                                             args={"service": "payments"})],
                        usage={"input_tokens": 120, "output_tokens": 20},
                    )
                # Third successful call: sees timeout, creates ticket
                if successful_calls[0] == 3:
                    return ModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="c3", tool_name="create_jira_ticket", args={
                            "title": "INCIDENT: Payments service degraded",
                            "description": "P99 latency at 890ms, status page unreachable",
                            "priority": "critical"
                        })],
                        usage={"input_tokens": 160, "output_tokens": 30},
                    )
                # Fourth: final summary
                return ModelResponse(
                    content="🚨 Incident Response Summary:\n"
                            "- Payments service showing elevated latency (p99: 890ms)\n"
                            "- External status page is unreachable\n"
                            "- Created critical ticket ENG-4521\n"
                            "- Recommended: page on-call engineer",
                    usage={"input_tokens": 200, "output_tokens": 45},
                )

        tracer = JSONTracer()
        agent = Agent(
            model=ModelSpec(provider="incident-llm", provider_impl=FlakyIncidentLLM()),
            instructions="You are an SRE incident response agent. Act quickly and decisively.",
            tools=[get_system_metrics, check_status_page, create_jira_ticket, send_email],
            events=tracer,
            resilience=RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.05),
            loop_repeat_threshold=5,
        )
        built = agent.build()
        built.tool_timeout = 0.05  # 50ms timeout for tools

        result = await built.arun("Payments service alerts firing — investigate NOW")

        # Agent completed the workflow despite cascading failures
        assert "ENG-4521" in result.output.text() or "Incident" in result.output.text()
        assert result.metadata["stop_reason"] == "final"
        # LLM retry happened transparently
        assert flaky_count[0] >= 1
        # Events captured the full trace
        assert any(e.kind == "run_start" for e in tracer.trace)
        assert any(e.kind == "tool_call" for e in tracer.trace)
        assert any(e.kind == "run_end" for e in tracer.trace)


    async def test_long_session_with_pinned_facts_and_compaction(self):
        """
        SCENARIO: Agent handles 12 customer interactions in one session.
        A critical account note is pinned early. After compaction triggers,
        the pinned fact MUST survive.

        EXPECTED: Pinned fact persists through multiple compaction cycles.
        """
        responses = [
            ModelResponse(content=f"Response to message {i}.",
                         usage={"input_tokens": 50, "output_tokens": 10})
            for i in range(15)
        ]
        provider = ScriptedProvider(responses)

        agent = Agent(
            model=ModelSpec(provider="session-llm", provider_impl=provider),
            instructions="You are a helpful assistant.",
            session_id="long-session-test",
            memory_window=4,
            compaction_threshold=6,
        )
        built = agent.build()

        # Pin a critical fact after the first interaction
        await built.arun("Hello, I'm starting a new conversation")
        built.pin_fact("CRITICAL: Customer account #A-7291 has a $50k credit line override. NEVER expire this.")

        # Run 10 more interactions to force compaction
        for i in range(10):
            await built.arun(f"Question number {i + 2}")

        # Verify the pinned fact survived compaction
        memory = built._memory_prefix()
        memory_text = " ".join(str(m) for m in memory)
        assert "A-7291" in memory_text
        assert "$50k" in memory_text

    async def test_cancellation_during_expensive_operation(self):
        """
        SCENARIO: User starts a long research task, then cancels mid-way.
        EXPECTED: Agent stops cleanly at next boundary, no further calls made.
        """
        call_num = [0]
        cancel_event = asyncio.Event()

        class ExpensiveResearchLLM:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1
                if call_num[0] >= 3:
                    # Signal cancellation after 2 tool cycles
                    cancel_event.set()
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id=f"c{call_num[0]}",
                        tool_name="search_knowledge_base",
                        args={"query": f"research topic {call_num[0]}"},
                    )],
                    usage={"input_tokens": 100, "output_tokens": 20},
                )

        agent = Agent(
            model=ModelSpec(provider="research-llm", provider_impl=ExpensiveResearchLLM()),
            tools=[search_knowledge_base],
            loop_repeat_threshold=50,
        )
        built = agent.build()

        # Build a context and cancel it after a few iterations
        ctx = RunContext(max_steps=20, loop_repeat_threshold=50)

        async def cancel_after_signal():
            await cancel_event.wait()
            ctx.cancel()

        # Run concurrently: the agent loop and the cancellation trigger
        cancel_task = asyncio.create_task(cancel_after_signal())
        result = await built._run_tool_loop(built._coerce_input("Do deep research"), ctx=ctx)
        cancel_task.cancel()

        assert result.metadata["stop_reason"] == "cancelled"
        # Should have completed 3 model calls, then stopped at boundary
        assert call_num[0] == 3


    async def test_full_production_stack_integration(self):
        """
        SCENARIO: A DevOps agent with EVERYTHING enabled:
          - Resilience (LLM API flaky)
          - Tool timeout (one tool is slow)
          - Concurrency cap (don't overwhelm downstream)
          - Loop detection (prevent infinite loops)
          - Event tracing (full observability)
          - Think tool (better reasoning)
          - Multiple tool types (local + MCP-style)
          - Context bounding (stay within limits)

        The agent must: check health, handle a timeout, reason, and respond.

        EXPECTED: All features compose correctly under stress.
        """
        @tool
        async def fast_health(service: str) -> str:
            """Quick health check."""
            await asyncio.sleep(0.005)
            return f'{{"service": "{service}", "healthy": true}}'

        @tool
        async def slow_dependency(service: str) -> str:
            """Check an external dependency (slow)."""
            await asyncio.sleep(10.0)
            return "unreachable"

        call_num = [0]
        flaky_attempts = [0]

        class ProductionLLM:
            """Simulates a production LLM with occasional 503s."""
            async def complete(self, request: ModelRequest) -> ModelResponse:
                call_num[0] += 1
                # 503 on call #3
                if call_num[0] == 3:
                    flaky_attempts[0] += 1
                    raise TransientProviderError("azure-openai", status_code=503)

                messages_text = str(request.messages)

                if call_num[0] == 1:
                    # Think first
                    return ModelResponse(
                        content="",
                        tool_calls=[ToolCall(id="t1", tool_name="think", args={
                            "thought": "I need to check auth and payments health, "
                                       "then verify the external dependency."
                        })],
                        usage={"input_tokens": 80, "output_tokens": 25},
                    )
                if call_num[0] == 2:
                    # Parallel health checks
                    return ModelResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="h1", tool_name="fast_health", args={"service": "auth"}),
                            ToolCall(id="h2", tool_name="fast_health", args={"service": "payments"}),
                            ToolCall(id="h3", tool_name="slow_dependency", args={"service": "stripe"}),
                        ],
                        usage={"input_tokens": 120, "output_tokens": 30},
                    )
                # After retry succeeds, model sees timeout and answers
                return ModelResponse(
                    content="Infrastructure Status:\n"
                            "✅ auth — healthy\n"
                            "✅ payments — healthy\n"
                            "⚠️ stripe dependency — timeout (investigating)\n\n"
                            "Recommendation: Monitor stripe connectivity.",
                    usage={"input_tokens": 200, "output_tokens": 40},
                )

        tracer = JSONTracer()
        agent = Agent(
            model=ModelSpec(provider="azure-openai", provider_impl=ProductionLLM()),
            instructions="You are a DevOps monitoring agent.",
            tools=[fast_health, slow_dependency],
            events=tracer,
            think_tool=True,
            resilience=RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.05),
            loop_repeat_threshold=5,
            token_budget=50000,
        )
        built = agent.build()
        built.tool_timeout = 0.05   # 50ms
        built.tool_concurrency = 2  # Max 2 parallel tool calls

        result = await built.arun("Full infrastructure health check")

        # Agent completed despite: 503 retry + tool timeout + concurrency limits
        assert "auth" in result.output.text()
        assert "timeout" in result.output.text().lower() or "stripe" in result.output.text().lower()
        assert result.metadata["stop_reason"] == "final"
        # Retry happened
        assert flaky_attempts[0] >= 1
        # Full event trace captured
        kinds = {e.kind for e in tracer.trace}
        assert "run_start" in kinds
        assert "model_call" in kinds
        assert "tool_call" in kinds
        assert "loop_stop" in kinds
        assert "run_end" in kinds
        # All model_call events have non-negative durations
        for e in tracer.trace:
            if e.kind == "model_call" and e.duration_ms is not None:
                assert e.duration_ms >= 0


# ===========================================================================
# HELPERS (at bottom to not distract from scenarios)
# ===========================================================================


class ScriptedProvider:
    """Returns pre-scripted responses in sequence."""

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if self._idx >= len(self._responses):
            return ModelResponse(content="(exhausted)", usage={})
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class FlakyProvider:
    """Fails N times with transient errors, then succeeds."""

    def __init__(self, fail_count: int, success_response: ModelResponse):
        self._fail_count = fail_count
        self._success = success_response
        self.attempts = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.attempts += 1
        if self.attempts <= self._fail_count:
            raise TransientProviderError("flaky", status_code=503, retry_after=0.01)
        return self._success
