"""Replay recorded model turns and tool batches without repeating side effects.

A persisted running batch has an unknown outcome after interruption. Resumption
requires application-supplied outcomes rather than guessing whether tools ran.
The runtime holds the checkpoint claim across reads, tool execution, and writes.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

from loomable.kernel.models import ModelResponse, ToolCall, ToolResult, ToolError, ToolOutcome
from loomable.kernel.errors import GuardrailViolation
from loomable.kernel.stores import serialize_session, deserialize_session
from loomable.persist.checkpoint import Checkpoint, PendingAction


class AgentPaused(Exception):
    def __init__(self, run_id, session_id, pending):
        self.run_id, self.session_id, self.pending = run_id, session_id, pending
        super().__init__(f"Agent run {run_id} is paused for tool approval")


class UnknownToolOutcome(RuntimeError):
    def __init__(self, run_id, call_ids):
        self.run_id, self.call_ids = run_id, call_ids
        super().__init__(f"Tool outcome is unknown for {call_ids}; supply reconciled outcomes to acontinue_run")


def _key(prefix, value):
    return prefix + hashlib.sha256(value.encode()).hexdigest()


def session_key(session_id: str) -> str:
    return _key("agent-session-", session_id)


def _digest(value):
    # Refuse lossy serialization before persisting executable proposals.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _response(data):
    return ModelResponse(**{**data, "tool_calls": [ToolCall(**c) for c in data["tool_calls"]]})


def _outcome(data):
    return ToolOutcome(call_id=data["call_id"],
                       result=ToolResult(**data["result"]) if data["result"] is not None else None,
                       error=ToolError(**data["error"]) if data["error"] is not None else None)


class ExecutionJournal:
    def __init__(self, store, state):
        self.store, self.state = store, state
        self.model_index = self.batch_index = 0

    @classmethod
    async def begin(cls, agent, input, ctx, index, *, once=False):
        if not isinstance(input, str):
            raise ValueError("Durable agents currently accept plain text input only")
        store = agent.durability.checkpointer
        if index is not None and not index.complete:
            raise RuntimeError("Session has an unfinished durable run; use acontinue_run")
        if index is not None and index.session_state.get("once"):
            raise ValueError("Use a new session_id to start another run")
        if index is not None and "final_session" in index.session_state:
            agent.session = deserialize_session(index.session_state["final_session"])
        state = {"version": 1, "run_id": ctx.run_id, "session_id": agent.session.session_id,
                 "input": input, "session": serialize_session(agent.session),
                 "instructions": agent.instructions,
                 "confirmation_tools": sorted(agent.require_confirmation),
                 "models": [], "batches": [], "approvals": {},
                 "status": "running"}
        if once:
            state["once"] = True
        journal = cls(store, state)
        await journal.save()
        return journal

    @classmethod
    async def load(cls, agent, cp, run_id, approvals=None, outcomes=None):
        if cp is None or cp.session_state.get("run_id") != run_id:
            raise ValueError("Durable run not found")
        state = deepcopy(cp.session_state)
        if state.get("version") != 1 or state["session_id"] != agent.session.session_id:
            raise ValueError("Durable run version or session identity does not match")
        if cp.complete:
            raise ValueError("Durable run is already completed")
        if (state["instructions"] != agent.instructions
                or state["confirmation_tools"] != sorted(agent.require_confirmation)):
            raise ValueError("Agent instructions or approval policy changed since the durable run began")
        journal = cls(agent.durability.checkpointer, state)
        known = {c["id"] for b in state["batches"] if b["status"] == "pending"
                 for c in b["calls"] if c["tool_name"] in agent.require_confirmation}
        if set(approvals or {}) - known or any(type(v) is not bool for v in (approvals or {}).values()):
            raise ValueError("Approvals must map recorded tool call IDs to booleans")
        state["approvals"].update(approvals or {})
        unresolved = [b for b in state["batches"] if b["status"] == "running"]
        required = {c["id"] for b in unresolved for c in b["calls"]}
        if outcomes is not None:
            if set(outcomes) != required or not all(isinstance(v, ToolResult) for v in outcomes.values()):
                raise ValueError("Supply a ToolResult for every unknown call ID and no other IDs")
            for batch in unresolved:
                batch["outcomes"] = [asdict(ToolOutcome(call_id=c["id"], result=outcomes[c["id"]]))
                                     for c in batch["calls"]]
                batch["blocked"] = []
                batch["status"] = "done"
        elif required:
            raise UnknownToolOutcome(run_id, sorted(required))
        agent.session = deserialize_session(state["session"])
        await journal.save()
        return journal

    async def save(self):
        _digest(self.state)
        pending = [PendingAction(c["tool_name"], c["id"], c["args"])
                   for b in self.state["batches"] if b["status"] == "pending"
                   for c in b["calls"] if c["id"] in b.get("approval_ids", [])
                   and c["id"] not in self.state["approvals"]]
        complete = self.state["status"] == "completed"
        cp = Checkpoint(thread_id=session_key(self.state["session_id"]),
                        step=len(self.state["models"]), session_state=deepcopy(self.state),
                        complete=complete, pending=pending, event_kind="agent_execution")
        await self.store.put(cp)

    async def model(self, request, invoke):
        digest = _digest(asdict(request))
        index = self.model_index
        self.model_index += 1
        records = self.state["models"]
        if index < len(records):
            if records[index]["request"] != digest:
                raise ValueError("Durable replay diverged: model request changed")
            return _response(records[index]["response"])
        response = await invoke()
        records.append({"request": digest, "response": asdict(response)})
        await self.save()
        return response

    async def dispatch(self, agent, calls, execute):
        from ._support import GatedDispatchResult
        index = self.batch_index
        self.batch_index += 1
        batches = self.state["batches"]
        raw_calls = [asdict(c) for c in calls]
        if index < len(batches):
            batch = batches[index]
            if batch["calls"] != raw_calls:
                raise ValueError("Durable replay diverged: tool batch changed")
        else:
            batch = {"calls": raw_calls, "status": "pending"}
            batches.append(batch)
        if batch["status"] == "done":
            # The tool loop annotates outcomes in place. Preserve the journal record.
            return GatedDispatchResult([_outcome(o) for o in deepcopy(batch["outcomes"])],
                                       [GuardrailViolation(**v) for v in batch["blocked"]])
        gated_ids = {c.id for c in calls if c.tool_name in agent.require_confirmation}
        if len({c.id for c in calls}) != len(calls) or any(
            c["id"] in gated_ids
            for previous in batches[:index] for c in previous["calls"]
        ):
            raise ValueError("Durable execution requires unique approval call IDs")
        pending = [c for c in calls if c.id in gated_ids and c.id not in self.state["approvals"]]
        if pending:
            batch["approval_ids"] = [c.id for c in pending]
            await self.save()
            raise AgentPaused(self.state["run_id"], self.state["session_id"], pending)
        batch["status"] = "running"
        await self.save()  # acknowledge the uncertainty boundary before side effects
        result = await execute(calls)
        batch["outcomes"] = [asdict(o) for o in result.outcomes]
        batch["blocked"] = [{"rule_id": v.rule_id, "action": v.action} for v in result.blocked]
        batch["status"] = "done"
        await self.save()
        return result

    async def finish(self, session, *, response=None):
        self.state["final_session"] = serialize_session(session)
        if response is not None:
            self.state["response"] = response
        self.state["status"] = "completed"
        await self.save()
