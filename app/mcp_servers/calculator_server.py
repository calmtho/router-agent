"""Calculator MCP Server — 提供加减乘除工具"""

import asyncio
import json
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("calculator")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="add",
            description="加法：计算 a + b",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"},
                },
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="subtract",
            description="减法：计算 a - b",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "被减数"},
                    "b": {"type": "number", "description": "减数"},
                },
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="multiply",
            description="乘法：计算 a * b",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个因数"},
                    "b": {"type": "number", "description": "第二个因数"},
                },
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="divide",
            description="除法：计算 a / b",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "被除数"},
                    "b": {"type": "number", "description": "除数"},
                },
                "required": ["a", "b"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    a = float(arguments["a"])
    b = float(arguments["b"])

    if name == "add":
        result = a + b
        expression = f"{a} + {b}"
    elif name == "subtract":
        result = a - b
        expression = f"{a} - {b}"
    elif name == "multiply":
        result = a * b
        expression = f"{a} * {b}"
    elif name == "divide":
        if b == 0:
            return [types.TextContent(type="text", text="错误：除数不能为零")]
        result = a / b
        expression = f"{a} / {b}"
    else:
        return [types.TextContent(type="text", text=f"未知工具：{name}")]

    # 返回 JSON 格式，包含表达式和结果
    return [types.TextContent(type="text", text=json.dumps({
        "expression": expression,
        "result": result
    }, ensure_ascii=False))]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
