"""Unit tests for loomable.kernel.errors - Error taxonomy."""

import pytest

from loomable.kernel.errors import (
    APIToolError,
    APIToolTimeoutError,
    GuardrailViolation,
    LoomableError,
    MCPConnectionError,
    MCPToolError,
    MemoryBackendError,
    ModelProviderError,
    PlanningModelError,
    ScriptToolError,
    SessionNotFoundError,
    SkillLoadError,
    SubagentError,
    UnsupportedExtensionError,
)


class TestLoomableErrorHierarchy:
    """All errors extend LoomableError."""

    @pytest.mark.parametrize(
        "error",
        [
            UnsupportedExtensionError(["skill", "mcp_server", "api_tool"]),
            ModelProviderError("openai"),
            MCPConnectionError("server-1"),
            MCPToolError("search_tool"),
            SkillLoadError("my_skill"),
            ScriptToolError("run_script"),
            APIToolError(404),
            APIToolTimeoutError("fetch_data", 30.0),
            MemoryBackendError("postgres"),
            SessionNotFoundError("sess-abc"),
            PlanningModelError("gpt-4"),
            SubagentError("agent-worker-1"),
            GuardrailViolation("no-delete", "delete_file"),
        ],
    )
    def test_all_errors_extend_base(self, error: LoomableError) -> None:
        assert isinstance(error, LoomableError)
        assert isinstance(error, Exception)


class TestUnsupportedExtensionError:
    def test_carries_supported_mechanisms(self) -> None:
        mechanisms = ["skill", "mcp_server", "api_tool"]
        err = UnsupportedExtensionError(mechanisms)
        assert err.supported_mechanisms == mechanisms

    def test_str_names_supported_mechanisms(self) -> None:
        err = UnsupportedExtensionError(["skill", "mcp_server", "api_tool"])
        msg = str(err)
        assert "skill" in msg
        assert "mcp_server" in msg
        assert "api_tool" in msg


class TestModelProviderError:
    def test_carries_provider_id(self) -> None:
        err = ModelProviderError("anthropic-claude")
        assert err.provider_id == "anthropic-claude"
        assert "anthropic-claude" in str(err)


class TestMCPConnectionError:
    def test_carries_server_id(self) -> None:
        err = MCPConnectionError("mcp-tools-server")
        assert err.server_id == "mcp-tools-server"
        assert "mcp-tools-server" in str(err)


class TestMCPToolError:
    def test_carries_tool_name(self) -> None:
        err = MCPToolError("web_search")
        assert err.tool_name == "web_search"
        assert "web_search" in str(err)


class TestSkillLoadError:
    def test_carries_skill_name(self) -> None:
        err = SkillLoadError("code_review")
        assert err.skill_name == "code_review"
        assert "code_review" in str(err)


class TestScriptToolError:
    def test_carries_tool_name(self) -> None:
        err = ScriptToolError("lint_code")
        assert err.tool_name == "lint_code"
        assert "lint_code" in str(err)


class TestAPIToolError:
    def test_carries_status_code(self) -> None:
        err = APIToolError(503)
        assert err.status_code == 503
        assert "503" in str(err)


class TestAPIToolTimeoutError:
    def test_carries_tool_name_and_timeout(self) -> None:
        err = APIToolTimeoutError("slow_api", 15.0)
        assert err.tool_name == "slow_api"
        assert err.timeout == 15.0
        assert "slow_api" in str(err)
        assert "15.0" in str(err)


class TestMemoryBackendError:
    def test_carries_backend_id(self) -> None:
        err = MemoryBackendError("redis-store")
        assert err.backend_id == "redis-store"
        assert "redis-store" in str(err)


class TestSessionNotFoundError:
    def test_carries_session_id(self) -> None:
        err = SessionNotFoundError("sess-xyz-123")
        assert err.session_id == "sess-xyz-123"
        assert "sess-xyz-123" in str(err)


class TestPlanningModelError:
    def test_carries_model_id(self) -> None:
        err = PlanningModelError("o1-preview")
        assert err.model_id == "o1-preview"
        assert "o1-preview" in str(err)


class TestSubagentError:
    def test_carries_subagent_id(self) -> None:
        err = SubagentError("worker-3")
        assert err.subagent_id == "worker-3"
        assert "worker-3" in str(err)


class TestGuardrailViolation:
    def test_carries_rule_id_and_action(self) -> None:
        err = GuardrailViolation("no-rm-rf", "rm -rf /")
        assert err.rule_id == "no-rm-rf"
        assert err.action == "rm -rf /"
        assert "no-rm-rf" in str(err)
        assert "rm -rf /" in str(err)
