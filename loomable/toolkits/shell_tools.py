"""Shell command toolkit backed by a :class:`~loomable.sandbox.types.Sandbox`."""

from __future__ import annotations

from loomable.agent.tools import FunctionTool
from loomable.sandbox import SubprocessSandbox
from loomable.sandbox.types import Sandbox
from loomable.toolkits._base import Toolkit


class ShellTools(Toolkit):
    """Run shell commands inside an agent sandbox.

    Soft policy: timeout + working-directory root + deny-list for destructive
    patterns. Not a hard multi-tenant jail.

    Usage::

        from loomable.toolkits import ShellTools
        from loomable.sandbox import SubprocessSandbox

        tools = ShellTools(sandbox=SubprocessSandbox(root="./.sandbox"))
        agent = Agent(model=..., tools=[tools])
    """

    def __init__(
        self,
        *,
        sandbox: Sandbox | None = None,
        working_dir: str | None = None,
        timeout: float = 30.0,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        super().__init__(include_tools=include_tools, exclude_tools=exclude_tools)
        self._sandbox: Sandbox = sandbox or SubprocessSandbox(
            root=working_dir, timeout=timeout
        )

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    def _register_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(self._run_shell, name="run_shell", idempotent=False),
        ]

    async def _run_shell(self, command: str, timeout: float | None = None) -> str:
        """Run a shell command in the agent sandbox and return stdout/stderr.

        Prefer workspace-relative paths. Destructive or privilege-escalating
        commands are blocked by sandbox policy.
        """
        result = await self._sandbox.run_shell(command, timeout=timeout)
        return result.format()
