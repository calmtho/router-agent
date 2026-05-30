from typing import Any, AsyncGenerator

import asyncio

from app.agents.sub_agent_base import SubAgentBase
from app.services.history_service import append_history, get_summary
from app.services.llm_client import get_llm_client
from app.services.mcp_client import mcp_client
from app.utils.logger import logger


def _extract_json(text: str) -> dict | None:
    import json
    import re

    # Remove markdown code blocks
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Brace-matching extraction for nested JSON
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None

    return None


class MCPAgent(SubAgentBase):
    """MCP 工具调用子代理"""

    def __init__(self):
        super().__init__("mcp")

    async def can_handle(self, query: str, context: dict[str, Any]) -> bool:
        # 检查是否需要工具调用（简化版：检查关键词）
        tool_keywords = ["计算", "算", "天气", "汇率", "转换", "convert", "calculator", "weather", "fetch", "抓取"]
        return any(keyword in query.lower() for keyword in tool_keywords)

    async def handle(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        from app.main import get_langfuse_client

        session_id = context.get("session_id", "default")
        chat_history = context.get("chat_history", [])
        summary = get_summary(session_id) if session_id else ""
        langfuse = get_langfuse_client()
        span = langfuse.span(
            name="mcp_agent",
            input={"query": query, "session_id": session_id},
            session_id=session_id,
        ) if langfuse else None

        # 1. 获取可用的 MCP 工具列表（只需执行一次）
        all_tools: list[dict] = []
        for server_name in mcp_client.servers:
            try:
                tools = await mcp_client.list_tools(server_name)
                for t in tools:
                    t["_server"] = server_name
                    all_tools.append(t)
            except Exception as e:
                logger.warning(f"Failed to list tools for {server_name}: {e}")

        if not all_tools:
            result = {
                "answer": "当前没有可用的工具，请检查 MCP 服务配置。",
                "agent": self.name,
            }
            # 保存对话历史
            append_history(session_id, query, result["answer"])
            if span:
                span.update(output=result)
            return result

        tools_desc = "\n".join(
            f"  - {t['_server']}/{t['name']}: {t.get('description', '无描述')}"
            for t in all_tools
        )

        # 构建消息列表，包含摘要和历史对话
        messages = []
        # 添加摘要（如果存在）
        if summary:
            messages.append({
                "role": "system",
                "content": f"【会话摘要】\n{summary}"
            })
        # 添加历史消息
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        # 添加工具描述和当前问题
        messages.append({
            "role": "user",
            "content": f"""用户问题是：{query}

可用的工具：
{tools_desc}

请分析用户的问题，判断是否需要调用工具。
如果需要，请以 JSON 格式输出：{{"server": "服务器名", "tool": "工具名", "arguments": {{"参数名": 参数值}}}}
如果不需要工具调用，输出：{{"need_tool": false, "reason": "原因"}}
只输出 JSON，不要附带其他文字。"""
        })

        # 2. 带重试的决策 + 工具调用
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                decision = await get_llm_client().chat(messages)

                # 解析决策 JSON
                decision_data = _extract_json(decision)
                if decision_data is None:
                    raise ValueError(f"LLM 返回格式无法解析: {decision[:200]}")

                # 如果不需要工具
                if decision_data.get("need_tool") is False:
                    result = {
                        "answer": decision_data.get("reason", "抱歉，无法处理此请求"),
                        "agent": self.name,
                    }
                    # 保存对话历史
                    append_history(session_id, query, result["answer"])
                    if span:
                        span.update(output=result)
                    return result

                # 调用 MCP 工具
                server_name = decision_data.get("server")
                tool_name = decision_data.get("tool")
                arguments = decision_data.get("arguments", {})

                if not server_name or not tool_name:
                    raise ValueError(f"缺少 server 或 tool 字段: {decision_data}")

                tool_result_raw = await mcp_client.call_tool(server_name, tool_name, arguments)
                tool_result_text = tool_result_raw[0].text if isinstance(tool_result_raw, list) else str(tool_result_raw)

                # 尝试解析 JSON 格式的结果（计算器等工具返回）
                import json as json_module
                try:
                    tool_result_data = json_module.loads(tool_result_text)
                    expression = tool_result_data.get("expression", tool_result_text)
                    tool_result_display = f"{expression} = {tool_result_data.get('result', '')}"
                except json_module.JSONDecodeError:
                    tool_result_display = tool_result_text

                # 让 LLM 根据工具结果生成回答
                # 重新构建消息列表，包含摘要和历史
                answer_messages = []
                if summary:
                    answer_messages.append({
                        "role": "system",
                        "content": f"【会话摘要】\n{summary}"
                    })
                for msg in chat_history:
                    answer_messages.append({"role": msg["role"], "content": msg["content"]})
                answer_messages.append({
                    "role": "user",
                    "content": f"""用户问题：{query}
工具调用结果：{tool_result_display}

请根据工具调用结果，用自然语言回答用户的问题。"""
                })

                answer = await get_llm_client().chat(answer_messages)

                result = {
                    "answer": answer,
                    "agent": self.name,
                    "tool_result": tool_result_display,
                }
                # 保存对话历史
                append_history(session_id, query, answer)
                if span:
                    span.update(output=result)
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"MCP attempt {attempt}/{max_retries} failed: {e}")
                if attempt >= max_retries:
                    if span:
                        span.update(status_code=500, error=str(e))
                    raise

        raise RuntimeError(f"MCP failed after {max_retries} attempts: {last_error}")

    async def handle_stream(
        self, query: str, context: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """流式 MCP 响应（仅流式输出最终回答）"""
        from app.main import get_langfuse_client

        session_id = context.get("session_id", "default")
        chat_history = context.get("chat_history", [])
        summary = get_summary(session_id) if session_id else ""
        langfuse = get_langfuse_client()
        span = None
        if langfuse:
            span = langfuse.span(
                name="mcp_agent_stream",
                input={"query": query, "session_id": session_id},
                session_id=session_id,
            )

        # 1. 获取可用的 MCP 工具列表
        all_tools: list[dict] = []
        for server_name in mcp_client.servers:
            try:
                tools = await mcp_client.list_tools(server_name)
                for t in tools:
                    t["_server"] = server_name
                    all_tools.append(t)
            except Exception as e:
                logger.warning(f"Failed to list tools for {server_name}: {e}")

        if not all_tools:
            yield "当前没有可用的工具，请检查 MCP 服务配置。"
            if span:
                span.update(output={"answer": "当前没有可用的工具，请检查 MCP 服务配置。"})
            return

        tools_desc = "\n".join(
            f"  - {t['_server']}/{t['name']}: {t.get('description', '无描述')}"
            for t in all_tools
        )

        # 构建消息列表，包含摘要和历史对话
        messages = []
        if summary:
            messages.append({
                "role": "system",
                "content": f"【会话摘要】\n{summary}"
            })
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({
            "role": "user",
            "content": f"""用户问题是：{query}

可用的工具：
{tools_desc}

请分析用户的问题，判断是否需要调用工具。
如果需要，请以 JSON 格式输出：{{"server": "服务器名", "tool": "工具名", "arguments": {{"参数名": 参数值}}}}
如果不需要工具调用，输出：{{"need_tool": false, "reason": "原因"}}
只输出 JSON，不要附带其他文字。"""
        })

        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                decision = await get_llm_client().chat(messages)

                decision_data = _extract_json(decision)
                if decision_data is None:
                    raise ValueError(f"LLM 返回格式无法解析: {decision[:200]}")

                if decision_data.get("need_tool") is False:
                    reason = decision_data.get("reason", "抱歉，无法处理此请求")
                    yield reason
                    # 保存对话历史
                    append_history(session_id, query, reason)
                    if span:
                        span.update(output={"answer": reason})
                    return

                server_name = decision_data.get("server")
                tool_name = decision_data.get("tool")
                arguments = decision_data.get("arguments", {})

                if not server_name or not tool_name:
                    raise ValueError(f"缺少 server 或 tool 字段: {decision_data}")

                tool_result_raw = await mcp_client.call_tool(server_name, tool_name, arguments)
                tool_result_text = tool_result_raw[0].text if isinstance(tool_result_raw, list) else str(tool_result_raw)

                # 尝试解析 JSON 格式的结果（计算器等工具返回）
                import json as json_module
                try:
                    tool_result_data = json_module.loads(tool_result_text)
                    expression = tool_result_data.get("expression", tool_result_text)
                    tool_result_display = f"{expression} = {tool_result_data.get('result', '')}"
                except json_module.JSONDecodeError:
                    tool_result_display = tool_result_text

                # 流式获取回答
                answer_messages = []
                if summary:
                    answer_messages.append({
                        "role": "system",
                        "content": f"【会话摘要】\n{summary}"
                    })
                for msg in chat_history:
                    answer_messages.append({"role": msg["role"], "content": msg["content"]})
                answer_messages.append({
                    "role": "user",
                    "content": f"""用户问题：{query}
工具调用结果：{tool_result_display}

请根据工具调用结果，用自然语言回答用户的问题。"""
                })

                full_answer = ""
                async for chunk in get_llm_client().chat_stream(answer_messages):
                    full_answer += chunk
                    yield chunk
                    # 短暂休眠，让事件循环有机会处理其他任务
                    await asyncio.sleep(0.005)  # 5ms

                # 保存对话历史
                append_history(session_id, query, full_answer)

                result = {
                    "answer": full_answer,
                    "agent": self.name,
                    "tool_result": tool_result_display,
                }
                if span:
                    span.update(output=result)
                return

            except Exception as e:
                last_error = e
                logger.warning(f"MCP attempt {attempt}/{max_retries} failed: {e}")
                if attempt >= max_retries:
                    if span:
                        span.update(status_code=500, error=str(e))
                    yield "抱歉，工具调用失败，请稍后重试。"
                    raise

        raise RuntimeError(f"MCP failed after {max_retries} attempts: {last_error}")


mcp_agent = MCPAgent()
