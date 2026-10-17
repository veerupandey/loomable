"""Expose an Agent over HTTP and AG-UI with a pooled provider."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from examples._provider import make_provider
from loomable import Agent
from loomable.serve import mount_agent

provider = make_provider()


@asynccontextmanager
async def lifespan(_app):
    async with provider:
        yield


app = FastAPI(lifespan=lifespan)
mount_agent(app, Agent(provider, instructions="Answer briefly."), max_concurrent_runs=2)
