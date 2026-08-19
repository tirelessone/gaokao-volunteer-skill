"""Factory for the local MCP server used by the general consultation Agent."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from agents.mcp import MCPServerStdio


PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_SCRIPT = PROJECT_ROOT / "mcp_servers" / "gaokao_consulting_server.py"


def create_consulting_mcp_server() -> MCPServerStdio:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    return MCPServerStdio(
        params={
            "command": sys.executable,
            "args": ["-X", "utf8", str(SERVER_SCRIPT)],
            "cwd": str(PROJECT_ROOT),
            "env": env,
        },
        cache_tools_list=True,
        name="gaokao-consulting-mcp",
        client_session_timeout_seconds=60,
    )
