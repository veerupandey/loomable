"""Local subprocess for the live PostgreSQL pilot."""
import asyncio
import os
import socket
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from loomable import Agent, tool
from loomable.agent import DurabilityOptions, UnknownToolOutcome
from loomable.kernel.models import ToolResult
from loomable.persist import PostgresCheckpointer
from loomable.serve import mount_agent
from tests.integration.test_model_live import live_provider


def app_for_worker(directory: Path):
    provider = live_provider()
    store = PostgresCheckpointer(os.environ["LOOMABLE_TEST_POSTGRES_URL"], max_size=2, max_checkpoints=8)

    @tool
    async def record_receipt(incident: str) -> str:
        """Record the incident receipt and return its receipt code."""
        if incident != "INC-PILOT":
            raise ValueError("Unexpected incident")
        with (directory / "receipts").open("a") as stream:
            stream.write("RECEIPT-726\n")
            stream.flush()
            os.fsync(stream.fileno())
        await asyncio.Event().wait()
        return "RECEIPT-726"

    agent = Agent(
        provider, session_id="pilot-default",
        max_cached_sessions=4, max_tool_iterations=4,
        durability=DurabilityOptions(store),
    )

    receipt_agent = Agent(
        provider, session_id="crash", tools=[record_receipt],
        require_tools=["record_receipt"], strict_require_tools=True, max_tool_iterations=4,
        durability=DurabilityOptions(store),
    )

    @asynccontextmanager
    async def lifespan(app):
        await store.setup()
        try:
            async with provider:
                yield
        finally:
            await store.close()

    app = FastAPI(lifespan=lifespan)
    mount_agent(app, agent, max_concurrent_runs=2)
    mount_agent(app, receipt_agent, prefix="/receipt", max_concurrent_runs=1)

    @app.middleware("http")
    async def capture_pilot_failure(request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            frames = traceback.extract_tb(exc.__traceback__)[-6:]
            return JSONResponse({
                "error_type": type(exc).__name__,
                "provider_status": getattr(exc, "status_code", None),
                "frames": [f"{Path(frame.filename).name}:{frame.lineno}" for frame in frames],
            }, status_code=500)

    @app.get("/pilot/metrics")
    async def metrics():
        pool = store._pool
        return {"pool_size": pool.get_size(), "pool_busy": pool.get_size() - pool.get_idle_size(),
                "cached_sessions": len(agent._session_agents)}

    @app.post("/pilot/recover")
    async def recover(run_id: str, call_id: str | None = None):
        outcomes = {call_id: ToolResult(content="RECEIPT-726")} if call_id else None
        try:
            result = await receipt_agent.acontinue_run(run_id, outcomes=outcomes)
        except UnknownToolOutcome as exc:
            return JSONResponse({"call_ids": exc.call_ids}, status_code=409)
        return {"text": result.text}

    return app


async def main():
    directory = Path(sys.argv[1])
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app_for_worker(directory), log_level="critical"))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("Pilot worker did not start")
            await asyncio.sleep(0.02)
        Path(sys.argv[2]).write_text(str(port))
        await task


if __name__ == "__main__":
    asyncio.run(main())
