"""Unit tests for loomable.agent.routing - ComplexityRouter."""

from __future__ import annotations

from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.content.message import AgentInput


class TestRunStrategy:
    """Tests for the RunStrategy enum."""

    def test_enum_values(self) -> None:
        assert RunStrategy.SINGLE.value == "single"
        assert RunStrategy.TOOL_LOOP.value == "tool_loop"
        assert RunStrategy.PLAN.value == "plan"

    def test_enum_has_exactly_three_members(self) -> None:
        assert len(RunStrategy) == 3


class TestComplexityRouterDefaults:
    """Tests for the default heuristic behavior of ComplexityRouter."""

    def test_simple_input_no_tools_returns_single(self) -> None:
        router = ComplexityRouter()
        agent_input = AgentInput.from_text("Hello, how are you?")
        result = router.classify(agent_input, has_tools=False)
        assert result == RunStrategy.SINGLE

    def test_simple_input_with_tools_returns_tool_loop(self) -> None:
        router = ComplexityRouter()
        agent_input = AgentInput.from_text("What's the weather?")
        result = router.classify(agent_input, has_tools=True)
        assert result == RunStrategy.TOOL_LOOP

    def test_complex_multi_question_returns_plan(self) -> None:
        router = ComplexityRouter()
        text = (
            "Compare the performance of these three algorithms step by step. "
            "What are the time complexities? What are the space complexities? "
            "Which one is best for each use case?"
        )
        agent_input = AgentInput.from_text(text)
        result = router.classify(agent_input, has_tools=True)
        assert result == RunStrategy.PLAN

    def test_step_cues_trigger_plan(self) -> None:
        router = ComplexityRouter()
        text = (
            "First analyze the logs and then compare the error rates. "
            "For each service, break down the response times. "
            "Decompose into multiple steps. What failed? What recovered?"
        )
        agent_input = AgentInput.from_text(text)
        result = router.classify(agent_input, has_tools=True)
        assert result == RunStrategy.PLAN

    def test_short_paragraph_ask_stays_single(self) -> None:
        router = ComplexityRouter()
        text = (
            "Compare Python and Rust step by step, but answer in one short paragraph."
        )
        result = router.classify(AgentInput.from_text(text), has_tools=False)
        assert result == RunStrategy.SINGLE

    def test_cue_rich_without_extra_signal_can_stay_single(self) -> None:
        """After Z.AI experiments, cue-only score=3 no longer forces PLAN."""
        router = ComplexityRouter()
        text = (
            "Compare and analyze how to launch software. Break down the work "
            "step by step. For each area cover pricing. Decompose into multiple steps."
        )
        result = router.classify(AgentInput.from_text(text), has_tools=False)
        assert result == RunStrategy.SINGLE

    def test_long_input_with_cues_returns_plan(self) -> None:
        router = ComplexityRouter()
        # Generate a long input (~500+ tokens) with step cues
        text = "Please analyze this data step by step. " + "word " * 500 + " Compare the results."
        agent_input = AgentInput.from_text(text)
        result = router.classify(agent_input, has_tools=False)
        assert result == RunStrategy.PLAN

    def test_moderate_input_with_tools_returns_tool_loop(self) -> None:
        router = ComplexityRouter()
        text = "Search for the latest news about Python."
        agent_input = AgentInput.from_text(text)
        result = router.classify(agent_input, has_tools=True)
        assert result == RunStrategy.TOOL_LOOP

    def test_classify_always_returns_run_strategy(self) -> None:
        """classify always returns a valid RunStrategy member."""
        router = ComplexityRouter()
        inputs = [
            ("Hi", False),
            ("Hi", True),
            ("Do A and then B and then C? What about D? How about E?", True),
            ("x " * 1000, False),
        ]
        for text, has_tools in inputs:
            result = router.classify(AgentInput.from_text(text), has_tools=has_tools)
            assert isinstance(result, RunStrategy)


class TestComplexityRouterModelClassifier:
    """Tests for injected model-based classifier override."""

    def test_model_classifier_overrides_heuristic(self) -> None:
        """When a model classifier is injected, it overrides the heuristic."""

        class AlwaysPlan:
            def classify(self, agent_input: "AgentInput", *, has_tools: bool) -> RunStrategy:
                return RunStrategy.PLAN

        router = ComplexityRouter(model_classifier=AlwaysPlan())
        # Simple input that would normally be SINGLE
        agent_input = AgentInput.from_text("Hi")
        result = router.classify(agent_input, has_tools=False)
        assert result == RunStrategy.PLAN

    def test_model_classifier_receives_arguments(self) -> None:
        """The model classifier receives the agent_input and has_tools."""
        captured: dict = {}

        class CapturingClassifier:
            def classify(self, agent_input: "AgentInput", *, has_tools: bool) -> RunStrategy:
                captured["agent_input"] = agent_input
                captured["has_tools"] = has_tools
                return RunStrategy.TOOL_LOOP

        router = ComplexityRouter(model_classifier=CapturingClassifier())
        inp = AgentInput.from_text("test")
        router.classify(inp, has_tools=True)

        assert captured["agent_input"] is inp
        assert captured["has_tools"] is True


class TestDefaultBehaviorPreservation:
    """Tests that the default (no router) behavior is preserved:
    tools → TOOL_LOOP, else SINGLE."""

    def test_no_tools_simple_is_single(self) -> None:
        router = ComplexityRouter()
        result = router.classify(AgentInput.from_text("Hello"), has_tools=False)
        assert result == RunStrategy.SINGLE

    def test_with_tools_simple_is_tool_loop(self) -> None:
        router = ComplexityRouter()
        result = router.classify(AgentInput.from_text("Hello"), has_tools=True)
        assert result == RunStrategy.TOOL_LOOP
