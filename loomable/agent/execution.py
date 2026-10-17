"""Stream projections over the ordinary agent execution lifecycle.

The execution task owns validation, verification, persistence, and cleanup.
Consumers can see provisional deltas; only the terminal chunk is a committed result.
"""
from __future__ import annotations

import asyncio
from contextlib import aclosing, suppress

from loomable.content import Text
from .context import RunContext
from .run import RunChunk


def run_context(agent, context=None):
    return context if context is not None else RunContext(
        events=agent.events, max_steps=agent.max_tool_iterations,
        token_budget=agent._token_budget, max_run_tokens=agent._max_run_tokens,
        loop_repeat_threshold=agent.loop_repeat_threshold,
    )


async def stream_chunks(agent, input, *, context=None, **kwargs):
    # Fork control/accounting, but never attach transport state to a caller's context.
    ctx = run_context(agent, context).fork()
    queue = asyncio.Queue(maxsize=128)
    ctx._delta_queue = queue
    task = asyncio.create_task(agent.arun(input, context=ctx, **kwargs))

    def wake_consumer(_):
        # A full queue already wakes the consumer. Never block task cleanup.
        with suppress(asyncio.QueueFull):
            queue.put_nowait(None)

    task.add_done_callback(wake_consumer)
    emitted = False
    try:
        while not (task.done() and queue.empty()):
            delta = await queue.get()
            if delta is None:
                break
            emitted = True
            yield RunChunk(delta=Text(delta))
        result = await task  # raises before exposing a success terminal
        if not emitted and result.output.parts:
            *parts, last = result.output.parts
            for part in parts:
                yield RunChunk(delta=part)
            yield RunChunk(delta=last, done=True, result=result)
        else:
            yield RunChunk(delta=Text(""), done=True, result=result)
    finally:
        if not task.done():
            ctx.cancel()
            task.cancel()
        with suppress(asyncio.CancelledError):
            # Retrieve errors on all paths, including explicit iterator closure.
            if task.cancelled() or not task.done():
                await task
        if task.done() and not task.cancelled():
            task.exception()


async def stream_events(agent, input, *, context=None, **kwargs):
    from loomable.stream import (
        AsyncStreamBus, StreamBridge, RUN_ERROR, RUN_FINISHED,
        TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END,
    )
    ctx = run_context(agent, context).fork()
    bus = AsyncStreamBus(max_pending=128)
    bridge = StreamBridge(bus, run_id=ctx.run_id,
                          session_id=agent.session.session_id, inner=ctx.events)
    ctx.events = bridge

    async def produce():
        text_started = False
        try:
            async with aclosing(stream_chunks(agent, input, context=ctx, **kwargs)) as chunks:
                async for chunk in chunks:
                    if chunk.delta.data and chunk.delta.modality.value == "text":
                        if not text_started:
                            await bridge.apublish(TEXT_MESSAGE_START, {"role": "assistant"})
                            text_started = True
                        delta = chunk.delta.data.decode("utf-8")
                        await bridge.apublish(TEXT_MESSAGE_CONTENT, {"delta": delta})
                    if chunk.done:
                        result = chunk.result
                        if text_started:
                            await bridge.apublish(TEXT_MESSAGE_END, {"text": result.text})
                        await bridge.apublish(RUN_FINISHED, {
                            "text": result.text, "usage": result.usage,
                            "metadata": result.metadata,
                        })
        except Exception as exc:
            from .durable import AgentPaused
            if isinstance(exc, AgentPaused):
                await bridge.apublish("RUN_PAUSED", {"run_id": exc.run_id,
                               "pending": [{"call_id": c.id, "tool_name": c.tool_name, "args": c.args}
                                           for c in exc.pending]})
                return
            await bridge.apublish(RUN_ERROR, {"message": str(exc), "error_type": type(exc).__name__})
        finally:
            if not asyncio.current_task().cancelling():
                await bus.close()

    task = asyncio.create_task(produce())
    try:
        async for event in bus.events():
            yield event
        await task
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
