"""MCP servers — external tools via Model Context Protocol.

USE WHEN: You want filesystem / DB / browser tools from an MCP server.
MCP tools register as normal Agent tools (discoverable under ``discovery=True``).

This demo starts the official filesystem MCP server (requires ``npx``).
For browser MCP via a bundled skill, see ``deep_agent/06_sandbox_browser.py``.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import Agent


async def main() -> None:
    if not shutil.which("npx"):
        raise SystemExit(
            "npx not found — install Node.js to run the filesystem MCP demo,\n"
            "or see examples/deep_agent/06_sandbox_browser.py for MCP via skills."
        )

    model = require_provider()
    root = Path(tempfile.mkdtemp(prefix="loomable-mcp-"))
    (root / "hello.txt").write_text("Hello from Loomable MCP demo.\n", encoding="utf-8")

    agent = Agent(
        model=model,
        role="File Assistant",
        goal="Read and summarize files using MCP filesystem tools",
        instructions="Use the MCP filesystem tools. Be concise.",
        mcp_servers=[
            {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(root)],
            }
        ],
        modalities="text",
    )
    result = await agent.arun(f"List files in {root} and summarize hello.txt")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
