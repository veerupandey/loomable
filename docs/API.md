# Loomable API

Public beta **0.2.0b0**. One contract: `arun` → `RunResult`.

| Use | API |
|-----|-----|
| One model + tools | `Agent` |
| Named orchestration | `Team` |
| Multi-step process | `Workflow` |
| Goal + WorkItems board | `Case` |
| Long-horizon research/code | `create_deep_agent(profile=...)` |
| Memory | `Memory.compose` |
| Searchable RAG | `knowledge_base=` / `retrievers=` |

Prefer `Workflow` / `Team`. Low-level `Flow` / `loomable.flow.helpers` are an escape hatch. Breaking changes are in [CHANGELOG.md](../CHANGELOG.md).

Beta limits: local workspace FS, cooperative cancel (not hard-kill of provider HTTP), serve auth is a shared API key (not RBAC).

### Package tiers (no separate `core`)

| Tier | Packages | Who imports |
|------|----------|-------------|
| **Facade** | `loomable` | Everyone — progressive disclosure |
| **Product** | `agent`, `memory`, `flow`, `case`, `toolkits`, `serve`, … | App / example code |
| **Runtime** | `kernel`, `providers`, `persist` | Advanced / internals |

`kernel` **is** the runtime core — do not add a parallel `loomable.core`. Prefer `loomable.memory` for L1/L2/L3 product APIs (`Memory.compose`, `NoteStore`); use `loomable.kernel` only when you need stores/contracts directly.

---

## Agent

```python
from loomable import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

agent = Agent(
    model="openai:gpt-4o-mini",  # or "anthropic:...", "gemini:...", provider instance
    role="Researcher",
    goal="Find accurate information",
    instructions="Be concise.",
    tools=[search],
    knowledge_base=["./handbook.pdf"],  # vector DB → search_knowledge
)
result = await agent.arun("When was Python created?")
print(result.output.text())
```

`run()` is a sync wrapper. With tools, the tool loop runs automatically; without tools, a single model call. `complexity_router=` is opt-in (`SINGLE` / `TOOL_LOOP` / `PLAN`). `PLAN` uses `_run_plan` (`plan_and_execute`); when `planner=` is set, planning goes through the kernel `Planner`, and each step runs the tool loop when tools are registered. `planning_model=` sets the kernel planner's dedicated model/tier id. `plan_tool=True` exposes an on-demand plan tool with the same worker tool-loop behavior.

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

**HITL tools:** `require_confirmation=["deploy"]` gates tools; without a custom `approver`, the default is **deny** (headless-safe). Deep `code_exec` / `shell` auto-join that list.

**Require tools:** `require_tools=["write_file:output/x.md"]` nudges until those side effects happen. `strict_require_tools=True` raises `RequireToolsError`. Same knobs inherit onto Agent steps via `Workflow(require_tools=...)` or `.step(..., require_tools=...)`.

**Verifier:** `verifier=` is a callable `(output, context) -> bool` or a `Verifier`. `retry_on_failure=True` + `max_verify_retries=` re-asks with feedback.

**HITL (tools):** `require_confirmation=["send_email"]` + `approver=` (default deny-all, headless-safe). Workflow step HITL is separate (`confirm=True`).

**Cancel:** `agent.cancel()` / `built.cancel()` — cooperative at tool-loop boundaries.

### Execution and streaming

`Agent.arun`, `Agent.astream`, and `Agent.astream_events` use the same model and tool lifecycle. Streamed deltas are provisional until validation and persistence succeed. The final `RunChunk` carries the `RunResult` in `result`. The AG-UI terminal event includes final text and usage. Model turns with advertised tools use buffered completions for provider compatibility.

Pass `session_id=` per call to keep histories separate on one Agent. Calls on the same runtime must run sequentially. Cancel a specific active session with `agent.cancel(session_id=...)`. Use a shared session store for conversation history across process restarts. For advanced limits, import `ExecutionOptions` from `loomable.agent` and pass it as `execution=`. Keep existing flat arguments or pass the options object for those settings.

Use `async with OpenAIProvider(...) as provider` to reuse a client pool within one event loop, or inject a caller-owned `httpx.AsyncClient`. The same applies to `AzureOpenAIProvider`. Set `native_structured_output=True` on a compatible endpoint to send Pydantic JSON Schema in the request. Local validation still applies.

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

Set `mount_agent(..., max_concurrent_runs=16)` to cap admitted runs per mount and process. Streaming requests hold a slot until their response closes. At capacity, authenticated callers receive HTTP 429 with `Retry-After: 1`. Agent SSE buffers up to 128 pending frames per stream.

Install `loomable[serve]` for FastAPI and Uvicorn, `loomable[mcp]` for MCP, and `loomable[examples]` for example scripts that load dotenv files.

---

## Team

```python
from loomable import Agent, Team

team = Team(
    members=[researcher, writer, critic],
    model="openai:gpt-4o-mini",
    mode="coordinate",  # coordinate | route | broadcast | sequential | tasks
)
result = await team.arun("Review our API design")
```

| Mode | Default | Behavior |
|------|---------|----------|
| `coordinate` | soft | LLM delegates to all members, synthesizes |
| `route` | soft | LLM picks one member |
| `broadcast` | hard | Same input to all, merge labeled results |
| `sequential` | hard | Chain members in order |
| `tasks` | soft | Shared TodoTools checklist + delegate until done (`max_iterations=`) — Agno `TeamMode.tasks` parity |

`hard=True` is only valid with `broadcast` / `sequential`. Soft `coordinate` auto-requires `delegate_to_*` and runs skipped members (`metadata["team_coordinate_fallback"]`). `Team.astream` mirrors `Agent.astream` for soft modes; hard modes chunk the merged `arun` result. Members may be nested `Team` instances (wrapped via `Team.as_agent()`).

Or pass `subagents=[...]` on a parent `Agent` — each becomes `delegate_to_<role>`. Nested subagents are allowed; `max_depth` (default 4) is an absolute nest budget from the top parent (child rebuilds at `depth+1`). `max_delegations` caps successful `delegate_to_*` calls in one parent run.

Note: Agent `subagents=` (LLM delegation tools) is separate from kernel `SubagentManager` (used by Workflow parallel/map engines).

---

## Workflow

```python
from loomable import Agent, Workflow, FlowPaused, JsonFileCheckpointer

wf = (
    Workflow("article", session_id="job-1", checkpointer=JsonFileCheckpointer("./ckpts"))
    .step("research", researcher)
    .parallel(analyst=analyst, visual=visual)
    .branch(when=needs_human, then=approver, else_=auto)
    .loop(polisher, until=quality_ok, max_iterations=3)
    .step("publish", publisher, confirm=True)
)
try:
    result = await wf.arun("AI agents in 2025")
except FlowPaused:
    await wf.approve("publish")
    result = await wf.arun(resume=True)

print(wf.explain())
print(wf.state.get("research").text())
```

Declarative: `Workflow("pipe", steps=[Step("a", a), Step("b", b)])`.

`confirm=True` requires `checkpointer=` + `session_id=`. Not supported inside `.parallel()` / `.branch()` / `.loop()` / `.route()`. Catch `FlowPaused` and use `.node_id` (or `pending.tool_name`) before `approve` + `arun(resume=True)`.

`Workflow(memory=True)` is a callable-step blackboard on `RunContext.memory` — **not** Agent chat memory. Agent steps share via SharedState output chaining.

Top-level runs also write `_workflow_input` (the original `arun` argument) into SharedState. After a pipeline, `.route` choosers see the **previous step's output** as ambient input — read `_workflow_input` or your own keys when classifying the original ticket.

`Loop(body=agent, verifier=..., max_iterations=3)` is a Runnable on its own. Workflow uses `.loop(..., until=)` or `.verify(body, check=..., max_retries=)` (bounded generate → check → repair).

### Graph-engineering knobs

```python
from loomable import Command, Workflow

wf = (
    Workflow("job", session_id="t1", checkpointer=cp, reducers={"items": append})
    .parallel(
        Step("a", agent_a),
        Step("b", agent_b, on_failure="skip"),
    )
    .route(
        lambda change: Command(goto="full", update={"severity": "high"}),
        quick=quick_review,
        full=full_audit,
    )
    .step("draft", drafter, reads="evidence")
    .verify(polisher, check=quality_ok, max_retries=2)
)
state = await wf.get_state()
await wf.update_state({"note": "human edit"})
```

| Knob | Where | Effect |
|------|--------|--------|
| `on_failure=` | `Step` / `.step` | `raise` / `retry` / `skip` / `fallback` / `stop` — failure stays local |
| `max_retries=` | `Step` / `.step` | Extra primary attempts before `on_failure` applies (default `2` for `retry`, else `0`) |
| `reads=` | `Step` / `.step` | Feeds `state[reads]` instead of ambient previous output (works for nested/root steps too) |
| `.verify` | `Workflow` | Verifier gate with hard repair budget (`max_retries + 1` attempts) |
| `.route` | `Workflow` | N-way Router (Agno / LangGraph multi-edge); chooser may return `Command(goto=…)` |
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

```python
from loomable import Case

case = Case(
    model=provider,
    goal="Close INC-88421 with SEV packet",
    board=True,           # open → in_progress → blocked → done
    dispatch="spawn",     # or "reuse"
    accept=lambda out, ctx: "SEV-" in out.text(),
    max_rounds=3,
)
result = await case.arun(email)
print(result.metadata["board"])

# Same pipeline:
Agent(model=provider, mode="case", dispatch="reuse", accept=check)
```

`Case.as_workflow()` returns a nestable `Workflow`. Board mutations stream as `STATE_SNAPSHOT` / `STATE_DELTA`. Case has no `astream` / NDJSON — use `astream_events` / SSE. `checkpointer=` / `max_rounds=` / `dispatch=` on `Agent` require `mode="case"`.

---

## Deep agent

```python
from loomable import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-4o-mini",
    profile="research",          # skills=["research"] + report/citation gates
    workspace="./.deep_workspace",
    knowledge_base=store,        # same Agent kwarg
)
await agent.arun("Research the topic; write reports/brief.md")

agent = create_deep_agent(model, profile="code", repo="./my-app")
```

`arun()` builds the agent. Call `agent.build()` only when you need the `BuiltAgent` (inspect tools, attach listeners).

Planning (`TodoTools`), local workspace FS, `task` / `task_batch` specialists, skills (`load_skill`), discovery (`search_tools` / `activate_tool`). `discovery_core="research-slim"` is experimental. Profiles: `general` | `research` | `code` | `sandbox` (`sandbox` = general + `code_exec` / `shell` with exec tools in the discovery core). Case-only kwargs (`dispatch`, `max_rounds`, `checkpointer`) require `mode="case"`. `board=False` is allowed without case mode.

---

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
agent = Agent(model=..., memory=memory, session_id="conv-1", user_id="alice",
              scopes={"claim_id": "CLM-4421"})
```

| Layer | Class | Stores |
|-------|--------|--------|
| Conversation | `ConversationMemory` | L1 turns + L2 summaries for `session_id` |
| User | `UserMemory` | Cross-session notes (`NoteStore`, scoped) |
| Knowledge | `KnowledgeMemory` | Passive RAG, or `knowledge_base` → `search_*` |
| Working | `WorkingMemory` | `Workflow(memory=True)` blackboard — not `Agent(memory=)` |

Do not pass both `memory=` and flat `session_store=` / `note_store=` / `memory_backend=`. `resume=True` means the session row must already exist. `memory_tool=True` / `UserMemory(auto_extract=True)` without a note store raises. `user_id` / `scopes` stamp UserMemory (`MemoryScope.of(...)`).

Default L3 vector store is Alibaba zvec (`pip install 'loomable[zvec]'`, `.loomable/memory_zvec`). Also: `engine="faiss"|"chroma"|"milvus"|"memory"` or `postgres_url=`. Embedders: `OpenAIEmbedder`, `GeminiEmbedder`, `AzureOpenAIEmbedder`, `HuggingFaceEmbedder`.

Same compose object on Team (coordinator) and Case (`from_agent` copies memory). Team has no `scopes=` — set on members or coordinator memory.

---

## Knowledge / RAG

```python
agent = Agent(model=..., knowledge_base=["./handbook.pdf", "./runbooks"])
# named collections → search_personal, search_company
agent = Agent(model=..., knowledge_base={"personal": ["./notes"], "company": store})
agent = Agent(model=..., knowledge_base=store, retrievers=[custom_retriever])

# Passive snippets (no tool) — requires embedder=
agent = Agent(model=..., knowledge=["FAQ..."], embedder=OpenAIEmbedder(), knowledge_top_k=5)
```

`knowledge=` without `embedder=` raises. Prefer `knowledge_base=` for search tools. Team / Case / Workflow inherit `knowledge_base=` onto Agent members that do not already have one.

`loomable.retrieval` (`ingest`, `build_agentic_retriever`) is experimental — prefer `knowledge_base=` / `retrievers=` on Agent. See `examples/advanced/06_agentic_retriever.py`.

---

## Models

```python
Agent(model="openai:gpt-4o-mini")
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
agent = Agent(model="company-openai:gpt-4o-mini")
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

agent = Agent(model="openai:gpt-4o-mini", response_model=CityInfo)
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

## Streaming, SSE, serving

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

```python
from fastapi import FastAPI
from loomable.serve import mount_agent, mount_case, mount_team, mount_workflow

app = FastAPI()
mount_agent(app, agent, prefix="/agent", api_key="secret")
mount_team(app, team, prefix="/teams", api_key="secret")      # alias of mount_agent
mount_case(app, case, prefix="/cases", api_key="secret")
mount_workflow(app, wf, prefix="/workflows", api_key="secret")
# POST /agent/run          JSON  (resume= on body when supported)
# POST /agent/run/events   SSE  (disconnect → cancel; RUN_PAUSED on Workflow HITL)
# POST /agent/run/stream   NDJSON, Agent only (omitted for Case / mode=case)
# GET/PATCH /workflows/state — Workflow get_state / update_state
# POST /workflows/approve  — HITL approve (+ auto_run to resume; optional session_id on body)
```

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

## Production knobs

### MCP tools

Install `loomable[mcp]` to connect to MCP servers. Use `command` and `args` for stdio, or `url` for Streamable HTTP. Select `transport="sse"` explicitly for a legacy SSE endpoint. HTTP specs accept `headers` and all specs accept `connect_timeout` in seconds.

MCP discovery closes its connection after listing tools. Each tool call opens and closes its own session, so MCP tools must work across independent sessions. A stdio command launches a fresh process for each invocation. Use `discovery=True` to defer the initial catalog connection and `tool_timeout=` to bound execution.

Unique remote tool names are exposed as-is when they fit the model's tool schema. Colliding or incompatible names get a server-qualified alias. Use the exposed name in `require_confirmation` to gate side effects. Remote tools default to non-idempotent. List known repeat-safe remote names in the server spec's `idempotent_tools`.

Structured results are available on the tool outcome. MCP images, audio, and embedded binary artifacts appear on the corresponding `RunResult` media properties. Resource links remain references and are never fetched automatically.

All opt-in.

```python
from loomable.providers import RetryPolicy

agent = Agent(
    model="openai:gpt-4o-mini",
    tools=[search],
    resilience=RetryPolicy(max_attempts=3, base_delay=0.5),
    tool_timeout=5.0,
    tool_concurrency=3,
    token_budget=8192,
    loop_repeat_threshold=3,
    think_tool=True,
    plan_tool=True,
    debug=True,
    mcp_servers=[
        {"command": "uvx", "args": ["some-mcp-server"]},
        {"url": "http://localhost:8080/sse", "headers": {"Authorization": "Bearer ..."}},
    ],
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

## Flow (escape hatch)

Prefer `Workflow.step` / `.parallel` / `.branch` / `.map`. `Flow`, `Node`, `Edge`, engines, and `loomable.flow.helpers` (`sequential`, `parallel`, `route`, `coordinate`) remain for power users. `FlowClass` / `start` / `listen` / `router` are experimental.

```python
from loomable.flow import Flow, Loop, Workflow, Step, FlowPaused
from loomable.flow.helpers import sequential  # advanced only
```

---

## Moved / renamed

| Old | Use |
|-----|-----|
| `from loomable.agent import NoteStore` | `from loomable.memory import NoteStore` (compat re-export still works) |
| `loomable/case.py` | `loomable.case` package (imports unchanged) |

## Removed names (no shims)

| Old | Use |
|-----|-----|
| `Agent(multimodal=)` | `modalities=` / `text_only=` |
| `Memory.compose(short=` / `long=`) | `conversation=` / `user=` |
| `Memory.with_user_id()` | `with_scopes(user_id=...)` |
| `Loop(end_condition=)` | `verifier=` (Workflow: `until=`) |
| `from loomable import sequential` | `Workflow` / `Team` |
| `Agent(memory=MemoryManager())` | `Memory.compose(...)` |
| `HITLPause` | `FlowPaused` |
