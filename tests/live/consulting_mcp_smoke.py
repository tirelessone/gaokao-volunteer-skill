"""Connect to the local consulting MCP server and list its runtime tools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

from agents.mcp import MCPServerManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from consulting_mcp import create_consulting_mcp_server


EXPECTED_TOOLS = {
    "database_search",
    "web_search",
    "query_admission_probability_by_score",
}


async def main() -> None:
    async with MCPServerManager(
        [create_consulting_mcp_server()],
        strict=True,
        connect_timeout_seconds=15,
    ) as manager:
        server = manager.active_servers[0]
        tools = await server.list_tools()
        names = {tool.name for tool in tools}
        if names != EXPECTED_TOOLS:
            raise RuntimeError(f"Unexpected MCP tools: {sorted(names)}")
        print(
            json.dumps(
                {"server": server.name, "tools": sorted(names)},
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
