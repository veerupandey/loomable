"""loomable.agent.pipeline - Sequential and iterative multi-agent pipeline.

A Pipeline runs agents in sequence, optionally looping for iterative refinement.
It supports session-based memory so follow-up questions work across runs.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from loomable.content import AgentOutput, Text, to_agent_input
from loomable.kernel.models import Session, Turn

from .builder import Agent, BuiltAgent
from .channels import Channel, ChannelMessage, InMemoryChannel
from .run import RunResult


class Pipeline:
    """Runs agents sequentially with optional iterative refinement.

    Each step's output becomes the next step's input. When a feedback_channel
    is provided, the last step can send feedback that loops back to the first
    step for refinement, up to max_iterations.

    Memory: when session_id is set, the pipeline maintains conversation history
    across runs, enabling multi-turn follow-up questions.

    Parameters
    ----------
    steps:
        Ordered list of agents to run sequentially.
    feedback_channel:
        Optional channel for iterative refinement (last step -> first step).
    max_iterations:
        Maximum refinement loops (only applies when feedback_channel is set).
    stop_condition:
        Callable that receives the last step's output text and returns True
        to stop iterating. Default: stops when "APPROVED" is in the output.
    session_id:
        Optional session identifier for multi-turn memory.
    name:
        Optional pipeline name for tracing/logging.
    """

    def __init__(
        self,
        steps: list[Agent | BuiltAgent],
        *,
        feedback_channel: Channel | None = None,
        max_iterations: int = 3,
        stop_condition: Callable[[str], bool] | None = None,
        session_id: str | None = None,
        name: str = "pipeline",
    ) -> None:
        self.steps = steps
        self.feedback_channel = feedback_channel
        self.max_iterations = max_iterations
        self.stop_condition = stop_condition or (lambda text: "APPROVED" in text)
        self.session_id = session_id
        self.name = name

        # Session for pipeline-level memory
        self._session: Session | None = None
        if session_id:
            self._session = Session(
                session_id=session_id,
                agent_config_ref=f"pipeline:{name}",
            )

    async def run(
        self,
        input: str | Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run the pipeline, returning the final step's result.

        When session_id is set, conversation history is maintained so
        follow-up calls can reference earlier runs.
        """
        # Resolve input to text
        if isinstance(input, str):
            input_text = input
        else:
            # Extract text from AgentInput
            pieces = []
            for msg in input.messages:
                for part in msg.parts:
                    if part.data is not None:
                        pieces.append(part.data.decode("utf-8"))
            input_text = "".join(pieces)

        # Prepend conversation history if we have a session
        effective_input = input_text
        if self._session and self._session.l1:
            history_context = self._build_history_context()
            effective_input = f"{history_context}\n\nCurrent request: {input_text}"

        # Record user turn
        if self._session:
            self._session.l1.append(Turn(
                role="user", content=input_text, tokens=0, step=self._session.step,
            ))

        # Run the pipeline
        result = await self._execute_steps(effective_input)

        # Record assistant turn
        if self._session:
            self._session.l1.append(Turn(
                role="assistant", content=result.output.text(),
                tokens=0, step=self._session.step,
            ))
            self._session.step += 1

        return result

    async def _execute_steps(self, input_text: str) -> RunResult:
        """Execute the pipeline steps with optional iterative refinement."""
        current_input = input_text
        last_result: RunResult | None = None

        for iteration in range(self.max_iterations):
            # Run each step sequentially
            for i, step in enumerate(self.steps):
                built = step.build() if isinstance(step, Agent) else step
                last_result = await built.arun(current_input)

                # Next step gets this step's output as input
                # (except for the last step whose output is the pipeline output)
                if i < len(self.steps) - 1:
                    current_input = last_result.output.text()

            # Check stop condition
            output_text = last_result.output.text() if last_result else ""
            if self.stop_condition(output_text):
                break

            # If feedback channel is not set, no feedback loop configured
            if self.feedback_channel is None:
                break

            # Send the output to the feedback channel
            await self.feedback_channel.send(ChannelMessage(
                sender=self.name,
                content=output_text,
            ))

            # The output becomes the next iteration's input (refinement)
            current_input = f"Previous output:\n{output_text}\n\nPlease refine based on feedback."

        if last_result is None:
            # Should not happen with at least one step, but be defensive
            last_result = RunResult(
                output=AgentOutput(parts=[Text("")]),
                session_id=self.session_id or str(uuid.uuid4()),
            )

        return last_result

    def _build_history_context(self) -> str:
        """Build a conversation history string from the session."""
        if not self._session:
            return ""
        parts = []
        for turn in self._session.l1[-8:]:  # last 8 turns
            parts.append(f"{turn.role}: {turn.content}")
        return "Conversation history:\n" + "\n".join(parts)
