"""MCP 协议客户端 — 使用 mcp SDK 与 MCP Server 通信"""

from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import config
from app.utils.logger import log_error, logger


class MCPClient:
    """MCP 协议客户端，通过 stdio 子进程调用 MCP Server"""

    def __init__(self):
        self.servers: dict[str, dict[str, Any]] = {}
        self._init_servers()

    def _init_servers(self) -> None:
        for server_config in config.mcp.servers:
            self.servers[server_config.name] = {
                "name": server_config.name,
                "command": server_config.command,
                "args": server_config.args or [],
                "env": server_config.env or None,
            }
            transport = "stdio"
            if server_config.url:
                transport = "sse"
            self.servers[server_config.name]["transport"] = transport
            logger.info(
                f"MCP Server registered: {server_config.name} (transport={transport})"
            )

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        server = self.servers.get(server_name)
        if not server:
            raise ValueError(f"Server {server_name} not found")

        if server["transport"] != "stdio":
            raise NotImplementedError("Only stdio transport is currently supported")

        params = StdioServerParameters(
            command=server["command"],
            args=server["args"],
            env=server.get("env"),
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result.content[0].text

    async def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        server = self.servers.get(server_name)
        if not server:
            raise ValueError(f"Server {server_name} not found")

        params = StdioServerParameters(
            command=server["command"],
            args=server["args"],
            env=server.get("env"),
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [
                    {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                    for t in tools.tools
                ]

    async def close(self) -> None:
        pass


mcp_client = MCPClient()
