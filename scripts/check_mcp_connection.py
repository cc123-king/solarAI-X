"""连接正在运行的MCP Server并列出领域工具。"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from main import MCPClient  # noqa: E402


async def check() -> int:
    server_url = os.getenv(
        "MCP_SERVER_URL",
        "http://127.0.0.1:12346/sse",
    )
    client = MCPClient()
    connected = await client.connect(server_url)
    print(f"connected={connected}")

    if not connected:
        return 1

    try:
        tools = await client.list_tools()
        print(f"tool_count={len(tools)}")
        print("tools=" + ",".join(tool.name for tool in tools))
    finally:
        await client.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(check()))
