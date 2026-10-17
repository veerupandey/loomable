# Loomable API reference

Loomable's `arun()` method returns a `RunResult`. Start with the
[quickstart](../README.md#quickstart) or follow a runnable example in the
[examples guide](../examples/README.md).

| I want to… | Read |
|---|---|
| Run a model with tools and validated output | [Agent](#agent) · [Structured output](#structured-output) |
| Coordinate specialists or define a graph | [Team](#team) · [Workflow](#workflow) |
| Work toward a goal with a plan and acceptance check | [Case](#case) · [Deep agent](#deep-agent) |
| Recall conversation history or search documents | [Memory](#memory) · [Knowledge and retrieval](#knowledge-and-retrieval) |
| Select a provider or register an adapter | [Models](#models) |
| Persist a run or wait for approval | [Durable tool approval](#durable-tool-approval) · [Checkpointing](#checkpointing) |
| Stream output or expose an HTTP endpoint | [Streaming and serving](#streaming-and-serving) |
| Extend an agent with MCP and tune execution | [Retries, limits, and MCP](#retries-limits-and-mcp) |

Import the main building blocks from `loomable`. Use `loomable.agent`,
`loomable.memory`, `loomable.providers`, and `loomable.persist` for configuration
types and adapters. The lower-level [Flow API](#low-level-flow) is available
when you need to define nodes and edges directly.

See the [changelog](../CHANGELOG.md) for compatibility changes and the
[security policy](../SECURITY.md) for deployment boundaries.

## Agent

An `Agent` combines a model, instructions, and optional tools. The synchronous
`run()` method wraps `arun()`. When tools are registered, Loomable executes
requested tool calls and returns their results to the model.

```python
from loomable import Agent

agent = Agent(model="openai:gpt-5.4-mini", instructions="Explain concepts in one sentence.")
result = agent.run("What is a database index?")
print(result.text)
```

The [typed-tool walkthrough](../examples/agents/02_with_tools.py) shows how to
register a Python function with `@tool`.

```python
result.output.text()
result.session_id
result.usage            # {"input_tokens", "output_tokens"}
result.structured       # when response_model= is set
result.tool_activity
result.verification     # when verifier= is set
result.trace            # debug=True or events=JSONTracer()
```

Persona fields (`role`, `goal`, `instructions`, `description`) assemble into the system prompt. `role` is reused in delegation labels.

| Option | Behavior |
|---|---|
| `require_tools=["write_file:output/report.md"]` | Require a successful tool call, optionally to a matching path. `strict_require_tools=True` raises `RequireToolsError` when the requirement is unmet. |
| `require_confirmation=["publish"]` | Ask an `approver` before dispatching a listed tool. Without an approver, ordinary runs deny the call. Durable runs can pause for a decision. |
| `verifier=` | Check output with a callable `(output, context)` or a `Verifier`. Enable `retry_on_failure=True` and bound retries with `max_verify_retries=`. |
| `complexity_router=` | Choose single-call, tool-loop, or plan execution. Set `planning_model=` for a dedicated planner model or enable `plan_tool=True` for on-demand planning. |
| `agent.cancel()` | Request cooperative cancellation at execution boundaries. |

### Execution and streaming

`Agent.arun`, `Agent.astream`, and `Agent.astream_events` use the same model and tool lifecycle. Streamed deltas are provisional until validation and persistence succeed. The final `RunChunk` carries the `RunResult` in `result`. The AG-UI terminal event includes final text and usage. Model turns with advertised tools use buffered completions for provider compatibility.

Pass `session_id=` per call to keep histories separate on one Agent. Calls on the same runtime must run sequentially. Cancel a specific active session with `agent.cancel(session_id=...)`. Use a shared session store for conversation history across process restarts. For advanced limits, import `ExecutionOptions` from `loomable.agent` and pass it as `execution=`. Keep existing flat arguments or pass the options object for those settings.

Use `async with OpenAIProvider(...) as provider` to reuse a client pool within one event loop, or inject a caller-owned `httpx.AsyncClient`. The same pooling pattern works with `GeminiProvider`. Set `native_structured_output=True` on a compatible endpoint to send Pydantic JSON Schema in the request. Local validation still applies.

### Durable tool approval

```python
from loomable import Agent, SQLiteCheckpointer
from loomable.agent import AgentPaused, DurabilityOptions

agent = Agent(
    model=provider,
    tools=[publish],
    session_id="review-42",
    require_confirmation=["publish"],
    durability=DurabilityOptions(SQLiteCheckpointer("runs.sqlite", max_checkpoints=8)),
)
try:
    result = await agent.arun("Prepare and publish the draft.")
except AgentPaused as pause:
    # Save the run ID and proposed calls for a reviewer.
    run_id = pause.run_id
    call_id = pause.pending[0].id

# Later, reconstruct the same agent and pass explicit reviewer decisions.
result = await agent.acontinue_run(run_id, approvals={call_id: True})
```

Use static tools with JSON arguments and results, plain text inputs, and a stable configuration. The runtime acquires a checkpoint claim before it reads or changes the journal. File-backed SQLite and JSON coordinate workers using the same local storage directory. PostgreSQL uses a session advisory lock and the same connection for journal reads and writes. A competing worker receives `CheckpointBusy` from `loomable.persist`. Custom durable checkpointers must provide an async `claim(thread_id)` context manager. In-memory claims are instance-local. If a worker exits after recording the start of a tool batch but before recording its result, `UnknownToolOutcome` identifies the affected call IDs. Reconcile the effects externally and pass a `ToolResult` for each affected call via `outcomes=`. A known completed tool batch is replayed from its recorded outcomes. Keep the checkpointer separate from conversation history if history must survive later runs and process restarts.

Agent retains up to 128 per-call session runtimes by default. Set `max_cached_sessions=` to adjust that cap. The default in-memory conversation store retains the most recently saved sessions up to the same limit. An explicit session store controls its own history retention and lets an evicted runtime reload its history. Requests holding a runtime keep their reference until they finish.

### Idempotent durable runs

Use `await agent.arun_once(text, session_id=request_id)` with
`DurabilityOptions` and a dedicated session ID per logical request. Retrying
the same text under the same session ID starts, resumes, or retrieves the run
under one checkpoint claim. A different input for that session raises
`ValueError`. Include the identity of any request-specific tool data when
deriving the session ID. Keep the agent configuration stable while runs are
in flight. Use a new session ID for a new request or `arun` for multi-turn
conversations.

Completed retries return the saved text, structured output, usage, and run ID
with `metadata["replayed"] == True`. They make no model or tool calls. Trace
and tool activity are available from the live run. Approval and unknown tool
outcomes still require `acontinue_run` with explicit decisions. The replay
window lasts until the session checkpoint is deleted.

---

## Team

A Team gives specialist agents a shared task. Choose a mode based on how you
want work assigned and results combined.

```python
from loomable import Agent, Team

model = "openai:gpt-5.4-mini"
security = Agent(model, role="Security reviewer", instructions="Identify one access-control risk.")
performance = Agent(model, role="Performance reviewer", instructions="Identify one scaling risk.")
team = Team([security, performance], model=model, mode="broadcast")
result = await team.arun("Review a new endpoint that exports orders as CSV.")
print(result.text)
```

| Mode | Orchestration | Behavior |
|---|---|---|
| `coordinate` | Model-directed | Delegate across the roster and synthesize the results. |
| `route` | Model-directed | Choose one specialist for the request. |
| `broadcast` | Deterministic by default | Run every specialist on the same input and merge labeled outputs. |
| `sequential` | Deterministic by default | Pass each specialist's output to the next. |
| `tasks` | Model-directed | Maintain a checklist and delegate work within an iteration budget. |

`hard=True` applies to broadcast and sequential modes. Coordinate mode can
run members skipped by the coordinator and records them in
`metadata["team_coordinate_fallback"]`. Teams can also contain other Teams.

Route mode requires one successful specialist delegation per run. Its choice
can be any roster member. A trailing `*` in `require_tools` matches a tool-name
prefix. The final `Team.astream` chunk carries the run result. In sequential
mode, the final member's structured output is available on that result.

Or pass `subagents=[...]` on a parent `Agent` — each becomes `delegate_to_<role>`. Nested subagents are allowed; `max_depth` (default 4) is an absolute nest budget from the top parent (child rebuilds at `depth+1`). `max_delegations` caps successful `delegate_to_*` calls in one parent run.

Note: Agent `subagents=` (LLM delegation tools) is separate from kernel `SubagentManager` (used by Workflow parallel/map engines).

---

## Workflow

A workflow connects agents, teams, or Python functions. Each step can consume
the previous step's output or a named value from shared state.

```python
import asyncio

from loomable import FlowPaused, InMemoryCheckpointer, Workflow


def draft(change):
    return f"Release note: {change}"


def publish(note):
    return note.text()


async def main():
    workflow = (
        Workflow("release", session_id="release-42", checkpointer=InMemoryCheckpointer())
        .step("draft", draft)
        .step("publish", publish, confirm=True)
    )
    try:
        result = await workflow.arun("Checkout retries a failed payment once.")
    except FlowPaused as pause:
        # In an application, obtain a reviewer decision before approving.
        await workflow.approve(pause.node_id)
        result = await workflow.arun(resume=True)
    print(result.text)


asyncio.run(main())
```

This example runs offline. For persistence across worker restarts, use the
[SQLite approval walkthrough](../examples/advanced/03_checkpointing.py).

Declarative: `Workflow("pipe", steps=[Step("a", a), Step("b", b)])`.

`confirm=True` requires `checkpointer=` + `session_id=`. Not supported inside `.parallel()` / `.branch()` / `.loop()` / `.route()`. Catch `FlowPaused` and use `.node_id` (or `pending.tool_name`) before `approve` + `arun(resume=True)`.

Built-in checkpointers coordinate workflow execution and approval with a
claim per checkpoint thread. A competing worker receives `CheckpointBusy`.
See the [live release-review test](../tests/integration/test_release_review_live.py) for a workflow that combines parallel teams, routing, and approval.

`Workflow(memory=True)` is a callable-step blackboard on `RunContext.memory` — **not** Agent chat memory. Agent steps share via SharedState output chaining.

Top-level runs also write `_workflow_input` (the original `arun` argument) into SharedState. After a pipeline, `.route` choosers see the **previous step's output** as ambient input — read `_workflow_input` or your own keys when classifying the original ticket.

`Loop(body=agent, verifier=..., max_iterations=3)` is a Runnable on its own. Workflow uses `.loop(..., until=)` or `.verify(body, check=..., max_retries=)` (bounded generate → check → repair).

### Graph controls

The offline examples show [routing with `Command`](../examples/patterns/08_route_command.py)
and [failure policies, named inputs, and verification](../examples/patterns/07_graph_engineering.py).

| Knob | Where | Effect |
|------|--------|--------|
| `on_failure=` | `Step` / `.step` | `raise` / `retry` / `skip` / `fallback` / `stop` — failure stays local |
| `max_retries=` | `Step` / `.step` | Extra primary attempts before `on_failure` applies (default `2` for `retry`, else `0`) |
| `reads=` | `Step` / `.step` | Feeds `state[reads]` instead of ambient previous output (works for nested/root steps too) |
| `.verify` | `Workflow` | Verifier gate with hard repair budget (`max_retries + 1` attempts) |
| `.route` | `Workflow` | Choose among named branches. The chooser may return `Command(goto=…)` |
| `Command` | step / chooser return | `goto` selects route arms; `update` patches SharedState (`resume` reserved / unwired) |
| `Send` | chooser `Command.update` lists | Dynamic map payloads — `Send.node` is metadata only; use `Workflow.map_over(..., over="tasks")` |
| `get_state` / `update_state` / `list_states` / `fork_session` | `Workflow` | Checkpoint control plane for resume / time-travel |
| `reducers=` | `Workflow` | Per-key SharedState merge (`append` / `extend` / `merge`) for parallel joins |
| `complexity=` | `Step` / `.step` | `"low"` / `"high"` cost hint for model-tier optimization |
| `_route_decision` | SharedState | Inspectable `{selected, choices, reason, handoff}` after routers / `.branch` / `.route` |
| `_workflow_input` | SharedState | Original top-level `arun` input (for routers after a pipeline) |

`on_failure="stop"` raises `StepFailed` after successful parallel siblings are committed (and checkpointed when a checkpointer is configured). `Workflow.state` still reflects completed work after a hard stop. Cooperative `cancel()` interrupts further retries.

---

## Case

A `Case` turns a goal into work items, dispatches them, synthesizes the
results, and checks the handoff against your acceptance rule.

```python
from loomable import Case

case = Case(
    model="openai:gpt-5.4-mini",
    goal="Prepare a release handoff.",
    accept=lambda output, context: "rollback" in output.text().lower(),
    max_rounds=2,
)
result = await case.arun("Checkout now retries a failed payment once. Include a rollback check.")
print(result.text)
```

Use `dispatch="reuse"` for a shared worker or `dispatch="spawn"` for
independent specialists. `Case.as_workflow()` exposes the pipeline as a
nestable workflow. To observe board changes, consume `STATE_SNAPSHOT` and
`STATE_DELTA` from `astream_events()`.

The [Case example](../examples/advanced/09_case.py) shows a bounded release
review with an acceptance check.

## Deep agent

`create_deep_agent()` configures an Agent with a workspace, planning tools,
and specialist delegation. Install `loomable[toolkits]` for the research
example's web and document integrations.

```python
from loomable import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-5.4-mini",
    profile="research",
    workspace="./.deep_workspace",
)
result = await agent.arun("Research database indexes and write reports/brief.md with sources.")
print(result.text)
```

| Profile | Starting configuration |
|---|---|
| `general` | Planning, workspace tools, and delegation |
| `research` | Web research, source registration, and a report acceptance check |
| `code` | Codebase indexing, editing, and execution tools |
| `sandbox` | Python and shell tools for a configured sandbox |

Use `arun()` to build and run the agent. Call `build()` directly when
inspecting the runtime or attaching listeners. Progressive discovery exposes
`search_tools`, `activate_tool`, and `load_skill` when configured.
`discovery_core="research-slim"` is experimental.

See the [Deep Agent examples](../examples/deep_agent/README.md) and the
[execution sandbox policy](../SECURITY.md#execution-sandbox-beta).

## Memory

```python
from loomable import Agent, Memory, ConversationMemory, UserMemory, open_session_store, open_vector_store
from loomable.memory import NoteStore
from loomable.providers import OpenAIEmbedder

notes = NoteStore(long_term=open_vector_store(engine="memory"), embedder=OpenAIEmbedder())
memory = Memory.compose(
    conversation=ConversationMemory(store=open_session_store("sqlite", path="sessions.db"), window=8),
    user=UserMemory(note_store=notes, memory_tool=True, auto_extract=True),
)
agent = Agent(
    model="openai:gpt-5.4-mini",
    memory=memory,
    session_id="conv-1",
    user_id="alice",
    scopes={"claim_id": "CLM-4421"},
)
```

| Layer | Class | Stores |
|-------|--------|--------|
| Conversation | `ConversationMemory` | L1 turns + L2 summaries for `session_id` |
| User | `UserMemory` | Cross-session notes (`NoteStore`, scoped) |
| Knowledge | `KnowledgeMemory` | Passive RAG, or `knowledge_base` → `search_*` |
| Working | `WorkingMemory` | `Workflow(memory=True)` blackboard — not `Agent(memory=)` |

Do not pass both `memory=` and flat `session_store=` / `note_store=` / `memory_backend=`. `resume=True` means the session row must already exist. `memory_tool=True` / `UserMemory(auto_extract=True)` without a note store raises. `user_id` / `scopes` stamp UserMemory (`MemoryScope.of(...)`).

`ConversationMemory(enabled=False)` disables conversation replay while
`UserMemory` continues to recall and save notes, including on calls without
a session ID. Each turn recalls the current contents of the note store.
Updating or deleting a note affects subsequent recall. Conversation history
and summaries have their own lifecycle.

Scoped notes use the scope prefix in the stored note ID as their ownership
boundary. Scope tags are labels. Scope keys or values containing `%`, `|`,
`=`, or `:` use escaped prefixes. To migrate notes written under a legacy
unescaped prefix, verify their ownership before re-saving them through
`ScopedNoteStore`.

Default L3 vector store is Alibaba zvec (`pip install 'loomable[zvec]'`, `.loomable/memory_zvec`). Also: `engine="faiss"|"chroma"|"milvus"|"memory"` or `postgres_url=`. Embedders: `OpenAIEmbedder`, `GeminiEmbedder`, `AzureOpenAIEmbedder`, `HuggingFaceEmbedder`.

Same compose object on Team (coordinator) and Case (`from_agent` copies memory). Team has no `scopes=` — set on members or coordinator memory.

The live support console test exercises file-backed conversations and
persistent FAISS notes through the HTTP adapter. Install `loomable[dev,faiss]`,
set `LOOMABLE_LIVE_PROVIDER=openai` or `gemini` and its API key, then run
`python -m pytest --live tests/integration/test_support_memory_live.py`.

---

## Knowledge and retrieval

```python
agent = Agent(model=..., knowledge_base=["./handbook.pdf", "./runbooks"])
# named collections → search_personal, search_company
agent = Agent(model=..., knowledge_base={"personal": ["./notes"], "company": store})
agent = Agent(model=..., knowledge_base=store, retrievers=[custom_retriever])

# Passive snippets (no tool) — requires embedder=
agent = Agent(model=..., knowledge=["FAQ..."], embedder=OpenAIEmbedder(), knowledge_top_k=5)
```

`knowledge=` without `embedder=` raises. Prefer `knowledge_base=` for search tools. Team / Case / Workflow inherit `knowledge_base=` onto Agent members that do not already have one.

`loomable.retrieval` (`ingest`, `build_agentic_retriever`) is experimental — prefer `knowledge_base=` / `retrievers=` on Agent. See [building a retriever from documents](../examples/advanced/05_build_retriever.py).

---

## Models

```python
Agent(model="openai:gpt-5.4-mini")
Agent(model="gemini:gemini-flash-latest")
Agent(model="gpt-4o-mini")  # bare name → OpenAI
```

Keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_DEPLOYMENT_NAME`.

### Provider plugins

Pass an adapter instance directly, or register a factory once at application startup. The factory receives a model name and returns an adapter implementing async `complete(request)`. Implement `stream(request)` when the adapter supports streaming.

```python
from loomable import Agent
from loomable.providers import OpenAIProvider, register_provider

register_provider("company-openai", OpenAIProvider)
agent = Agent(model="company-openai:gpt-5.4-mini")
```

An installed Python distribution can declare an entry point under `[project.entry-points."loomable.providers"]`. Its name becomes the model prefix and its value points to the factory. Loomable loads that factory when an application selects the prefix. Built-in providers bypass plugin discovery.

Amazon Bedrock: `pip install "loomable[bedrock]"`. Uses the standard AWS credential chain (env vars, shared profile, or SSO). Select with `BedrockProvider(model, region_name=..., profile_name=...)` or the `bedrock:` / `aws:` model-string prefix. Requires `bedrock:InvokeModel` permission and model access granted in the region.

---

## Structured output

```python
from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str
    population: int

agent = Agent(model="openai:gpt-5.4-mini", response_model=CityInfo)
city = agent.run("Info about Tokyo").structured
# or per-call: await agent.arun("...", output_schema=CityInfo)
```

---

## Multimodal

The provider and model determine which attachments can be sent. Restrict an agent to text with `modalities="text"` or `text_only=True`. OpenAI Chat Completions accepts images on vision models. Its Loomable adapter rejects audio and video. For video analysis with an OpenAI vision model, extract frames and pass them as images.

```python
from loomable.media import Image, Audio, Video, File
from loomable import Agent, tool
from loomable.providers import GeminiMediaProvider, OpenAIProvider

vision = Agent(GeminiMediaProvider("gemini-flash-latest"))
result = await vision.arun("Summarize", videos=["./demo.mp4"], audio=["./clip.wav"])

artist = Agent(GeminiMediaProvider("gemini-3.1-flash-image", image_output=True))
result = await artist.arun("Generate a blue circle on a white background")
result.images[0].save("./native.png")

async with OpenAIProvider("gpt-5.4-mini", timeout=120) as provider:
    @tool(idempotent=False)
    async def render_image(prompt: str) -> Image:
        """Generate a PNG from a description."""
        return await provider.generate_image(prompt, model="gpt-image-2")

    designer = Agent(provider, tools=[render_image], feedback_media=False)
    result = await designer.arun("Use render_image to create a blue circle.")
    result.images[0].save("./tool.png")
```

`GeminiMediaProvider` uses native GenerateContent with inline attachments. Use a file path or inline bytes, and choose a model that supports the supplied modalities. This provider buffers the response and handles media turns without function calling. Use `GeminiProvider` for function calling. A function tool can call the native provider's `generate_image()` method when configured with an image model.

`Image` / `Audio` / `Video` / `File` take exactly one of `url=`, `filepath=`, `content=`. Tools can return these objects. Read results through `result.images`, `result.audio`, `result.videos`, and `result.files`. With `feedback_media=True` (default), supported tool media is also sent back to the model. Disable feedback to deliver tool media directly to the caller without another media input charge. Passing an `AgentOutput` to another agent preserves its model-generated media. Tool artifacts are available separately on the originating `RunResult`.

---

## Streaming and serving

```python
async for chunk in agent.astream("hello"):
    if chunk.delta.data:
        print(chunk.delta.data.decode(), end="")
```

`astream` streams token deltas for single-shot runs. With tools, model turns that advertise tool schemas use ``complete()`` (so broken ``stream()`` stubs cannot skip the tool loop); final text is still streamed as deltas. Without ``stream()`` on the provider, falls back to ``arun`` then chunks. `Team.astream` follows the same rules. Case / `mode="case"` do not support `astream`.

```python
async for ev in agent.astream_events(prompt):
    print(ev.type)  # RUN_* / TEXT_* / TOOL_*
async for ev in workflow.astream_events(prompt):
    ...  # + NODE_*
async for ev in case.astream_events(prompt):
    ...  # + STATE_*
```

| Family | Events |
|--------|--------|
| Lifecycle | `RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR`, `RUN_PAUSED` (Workflow HITL — not an error) |
| Text | `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END` |
| Tools | `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`, `TOOL_CALL_RESULT` |
| Graph | `NODE_STARTED`, `NODE_FINISHED` |
| State | `STATE_SNAPSHOT`, `STATE_DELTA` |

Install `loomable[serve]` to mount a runnable in FastAPI.

```python
import os

from fastapi import FastAPI

from loomable import Agent
from loomable.serve import mount_agent

app = FastAPI()
mount_agent(
    app,
    Agent(model="openai:gpt-5.4-mini"),
    prefix="/agent",
    api_key=os.environ["LOOMABLE_API_KEY"],
    max_concurrent_runs=16,
)
```

| Endpoint | Response |
|---|---|
| `POST /agent/run` | Completed run as JSON |
| `POST /agent/run/events` | AG-UI events over server-sent events (SSE) |
| `POST /agent/run/stream` | Newline-delimited JSON for targets with `astream()` |
| `GET /agent/health` | Health response |

Use `mount_team`, `mount_case`, or `mount_workflow` for the corresponding
runnable. Workflow mounts also expose `GET/PATCH /state` and
`POST /approve` under their prefix. Agent admission limits apply per mount
and process. Streaming requests hold a slot until the response closes. At
capacity, authenticated callers receive HTTP 429 with `Retry-After: 1`.

See the [local serving walkthrough](../examples/README.md#extend-and-serve)
and [serving security boundaries](../SECURITY.md#trust-boundaries-beta).


---

## Checkpointing

```python
from loomable.persist import JsonFileCheckpointer, SQLiteCheckpointer, CheckpointListener, CheckpointConfig
from loomable import PostgresCheckpointer

cp = JsonFileCheckpointer(".checkpoints", max_checkpoints=20)
# SQLiteCheckpointer("agent.db")
# PostgresCheckpointer("postgresql://loomable:loomable@127.0.0.1:5432/loomable")  # loomable[postgres]

built = agent.build()
built.events = CheckpointListener(
    checkpointer=cp,
    config=CheckpointConfig(on_events=["run_end", "tool_call"], max_checkpoints=10),
    thread_id="session-123",
)
```

`PostgresCheckpointer` supports Case and Workflow checkpoints and the opt-in
Agent execution journal configured through `DurabilityOptions`. For a
PostgreSQL-backed journal, call
`await cp.prune_completed_agent_sessions(older_than_seconds=604800, limit=100)`
from scheduled maintenance. It deletes whole expired sessions in bounded
batches, using the same claims as Agent runs. It skips owned sessions and
preserves sessions whose latest checkpoint is incomplete. It returns the number
of sessions deleted.

---

## Retries, limits, and MCP

Configure provider retries with `RetryPolicy`. Set `tool_timeout` and
`tool_concurrency` separately to bound tool execution. Use `token_budget` to
shape the context window and `max_run_tokens` to cap cumulative model usage
at execution boundaries.

### MCP tools

Install `loomable[mcp]` to connect to MCP servers. Use `command` and `args` for stdio, or `url` for Streamable HTTP. Select `transport="sse"` explicitly for a legacy SSE endpoint. HTTP specs accept `headers` and all specs accept `connect_timeout` in seconds.

MCP discovery closes its connection after listing tools. Each tool call opens and closes its own session, so MCP tools must work across independent sessions. A stdio command launches a fresh process for each invocation. Use `discovery=True` to defer the initial catalog connection and `tool_timeout=` to bound execution.

Unique remote tool names are exposed as-is when they fit the model's tool schema. Colliding or incompatible names get a server-qualified alias. Use the exposed name in `require_confirmation` to gate side effects. Remote tools default to non-idempotent. List known repeat-safe remote names in the server spec's `idempotent_tools`.

Structured results are available on the tool outcome. MCP images, audio, and embedded binary artifacts appear on the corresponding `RunResult` media properties. Resource links remain references and are never fetched automatically.

See the [local stdio example](../examples/advanced/01_mcp_servers.py) for a
working server and client in one file.

### Execution settings

```python
from loomable import Agent
from loomable.providers import RetryPolicy

agent = Agent(
    model="openai:gpt-5.4-mini",
    resilience=RetryPolicy(max_attempts=3, base_delay=0.5),
    tool_timeout=5.0,
    tool_concurrency=3,
    token_budget=8192,
    loop_repeat_threshold=3,
    think_tool=True,
    plan_tool=True,
    debug=True,
)
```

---

## Display

```python
from loomable.display import pp, delegation_outputs, step_outputs, show_graph, mermaid_graph

pp(result)  # RunResult only; anything else is print()
delegation_outputs(result)["researcher"]
step_outputs(result)["research"]          # Workflow step names, not node_0
show_graph(wf.flow)                       # prints Mermaid (paste into mermaid.live)
```

---

## Low-level Flow

Use `Flow` with `Node` and `Edge` for explicit graph construction and engine selection. The `loomable.flow.helpers` module provides `sequential`, `parallel`, `route`, and `coordinate` builders. `FlowClass` / `start` / `listen` / `router` are experimental.

```python
from loomable.flow import Flow, Loop, Workflow, Step, FlowPaused
from loomable.flow.helpers import sequential  # advanced only
```

---

## Compatibility

| Old | Use |
|-----|-----|
| `from loomable.agent import NoteStore` | `from loomable.memory import NoteStore` (compat re-export still works) |
| `loomable/case.py` | `loomable.case` package (imports unchanged) |

### Removed options

| Old | Use |
|-----|-----|
| `Agent(multimodal=)` | `modalities=` / `text_only=` |
| `Memory.compose(short=` / `long=`) | `conversation=` / `user=` |
| `Memory.with_user_id()` | `with_scopes(user_id=...)` |
| `Loop(end_condition=)` | `verifier=` (Workflow: `until=`) |
| `from loomable import sequential` | `Workflow` / `Team` |
| `Agent(memory=MemoryManager())` | `Memory.compose(...)` |
| `HITLPause` | `FlowPaused` |
