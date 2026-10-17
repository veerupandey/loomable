"""Two-worker HTTP deployment for the read-only incident triage example."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from hmac import compare_digest
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from loomable.agent.durable import UnknownToolOutcome
from loomable.agent.errors import StructuredOutputError
from loomable.agent.options import DurabilityOptions
from loomable.kernel.errors import ModelProviderError
from loomable.persist.checkpoint import CheckpointBusy
from loomable.persist.postgres import PostgresCheckpointer
from loomable.providers import OpenAIProvider, RetryPolicy, TransientProviderError

from .__main__ import Handoff, build_triage_agent, validate_handoff

logger = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)
RUNS_PER_WORKER = 2
RUN_TIMEOUT = 25
RETRY_HEADERS = {"Retry-After": "1"}
MAX_BODY_BYTES = 128 * 1024


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=2048)


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    incident: str = Field(min_length=1, max_length=128)
    signals: list[Evidence] = Field(min_length=1, max_length=32)
    policy: Evidence


async def sweep(checkpoints, retention_seconds, interval):
    while True:
        try:
            async with asyncio.timeout(15):
                await checkpoints.prune_completed_agent_sessions(retention_seconds)
        except Exception:
            logger.exception("Completed-session retention sweep failed")
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app):
    token = os.environ["TRIAGE_ACCESS_TOKEN"].encode()
    retention = int(os.environ.get("TRIAGE_RETENTION_SECONDS", 7 * 86400))
    interval = int(os.environ.get("TRIAGE_RETENTION_INTERVAL_SECONDS", 60))
    if len(token) < 32 or retention < 1 or interval < 1:
        raise ValueError("Configure a 32-byte access token and positive retention settings")
    checkpoints = PostgresCheckpointer(
        os.environ["TRIAGE_DATABASE_URL"],
        table="triage_checkpoints",
        max_checkpoints=8,
        max_size=RUNS_PER_WORKER + 1,
    )
    provider = OpenAIProvider(
        "gpt-5.4-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("TRIAGE_OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    try:
        await checkpoints.setup()
        async with provider:
            app.state.token = token
            app.state.active = 0
            app.state.checkpoints = checkpoints
            app.state.provider = provider
            maintenance = asyncio.create_task(sweep(checkpoints, retention, interval))
            try:
                yield
            finally:
                maintenance.cancel()
                with suppress(asyncio.CancelledError):
                    await maintenance
    finally:
        await checkpoints.aclose()


app = FastAPI(title="Loomable incident triage", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ready", "worker": os.getpid()}


@app.post("/triage")
async def triage(request: Request, credentials: HTTPAuthorizationCredentials | None = Security(bearer)) -> Handoff:
    state = request.app.state
    if credentials is None or not compare_digest(credentials.credentials.encode(), state.token):
        raise HTTPException(401, "Invalid access token", headers={"WWW-Authenticate": "Bearer"})
    if state.active >= RUNS_PER_WORKER:
        raise HTTPException(429, "Worker at capacity", headers=RETRY_HEADERS)
    state.active += 1
    try:
        async with asyncio.timeout(RUN_TIMEOUT):
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_BODY_BYTES:
                    raise HTTPException(413, "Incident packet exceeds 128 KiB")
                body.extend(chunk)
            try:
                submission = TriageRequest.model_validate_json(body)
            except ValueError:
                raise HTTPException(422, "Invalid incident packet") from None
            packet = submission.model_dump(exclude={"request_id"})
            digest = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
            session_id = f"triage:{submission.request_id}:{digest}"
            agent = build_triage_agent(
                state.provider, packet, session_id=session_id,
                durability=DurabilityOptions(state.checkpoints),
                resilience=RetryPolicy(max_attempts=3, max_delay=1, per_call_timeout=7),
            )
            result = await agent.arun_once(f"Triage incident {packet['incident']}.")
            return validate_handoff(result.structured, packet)
    except CheckpointBusy:
        raise HTTPException(409, "Request already running", headers=RETRY_HEADERS) from None
    except (TransientProviderError, TimeoutError):
        raise HTTPException(503, "Triage temporarily unavailable", headers=RETRY_HEADERS) from None
    except UnknownToolOutcome:
        raise HTTPException(409, "Run requires tool outcome reconciliation") from None
    except (ModelProviderError, StructuredOutputError, ValueError):
        raise HTTPException(502, "Triage could not produce a valid handoff") from None
    finally:
        state.active -= 1
