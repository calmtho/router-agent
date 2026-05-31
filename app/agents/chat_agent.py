from typing import Any, AsyncGenerator

import asyncio

from app.agents.sub_agent_base import SubAgentBase
from app.services.history_service import append_history, get_summary, get_title
from app.services.llm_client import get_llm_client
from app.utils.logger import logger


class ChatAgent(SubAgentBase):
    """闲聊子代理，处理一般对话"""

    def __init__(self):
        super().__init__("chat")

    async def can_handle(self, query: str, context: dict[str, Any]) -> bool:
        # 闲聊代理可以处理所有请求，作为默认/回退代理
        return True

    async def handle(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        from app.main import get_langfuse_client

        langfuse = get_langfuse_client()
        session_id = context.get("session_id", "default")
        span = langfuse.span(
            name="chat_agent",
            input={
                "query": query,
                "original_message": context.get("original_message"),
                "session_id": session_id,
            },
            session_id=session_id,
        ) if langfuse else None

        try:
            # 构建消息列表，包含历史对话和摘要
            messages = [
                {
                    "role": "system",
                    "content": "你是一个友好、专业的 AI 助手。请用自然、流畅的语言与用户交流。",
                },
            ]

            # 添加标题（如果存在）
            title = get_title(session_id)
            if title:
                messages.append({
                    "role": "system",
                    "content": f"【会话主题】{title}"
                })

            # 添加摘要（如果存在）
            summary = get_summary(session_id)
            if summary:
                messages.append({
                    "role": "system",
                    "content": f"【历史摘要】\n{summary}"
                })

            # 添加历史对话（如果存在）
            history = context.get("chat_history", [])
            messages.extend(history)

            # 添加当前用户消息（使用纠正后的文本，让 LLM 自然回答）
            messages.append({"role": "user", "content": query})

            response = await get_llm_client().chat(messages)

            # 保存对话历史
            append_history(session_id, query, response)

            result = {
                "answer": response,
                "agent": self.name,
            }
            if span:
                span.update(output=result)
            return result

        except Exception as e:
            logger.error(f"Chat agent failed: {e}")
            error_result = {
                "answer": "抱歉，我遇到了一些问题，请稍后重试。",
                "agent": self.name,
                "error": str(e),
            }
            if span:
                span.update(output=error_result, status_code=500)
            return error_result

    async def handle_stream(
        self, query: str, context: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """流式聊天响应"""
        from app.main import get_langfuse_client

        langfuse = get_langfuse_client()
        session_id = context.get("session_id", "default")
        span = None
        if langfuse:
            span = langfuse.span(
                name="chat_agent_stream",
                input={
                    "query": query,
                    "original_message": context.get("original_message"),
                    "session_id": session_id,
                },
                session_id=session_id,
            )

        try:
            # 构建消息列表，包含历史对话和摘要
            messages = [
                {
                    "role": "system",
                    "content": "你是一个友好、专业的 AI 助手。请用自然、流畅的语言与用户交流。",
                },
            ]

            # 添加标题（如果存在）
            title = get_title(session_id)
            if title:
                messages.append({
                    "role": "system",
                    "content": f"【会话主题】{title}"
                })

            # 添加摘要（如果存在）
            summary = get_summary(session_id)
            if summary:
                messages.append({
                    "role": "system",
                    "content": f"【历史摘要】\n{summary}"
                })

            # 添加历史对话（如果存在）
            history = context.get("chat_history", [])
            messages.extend(history)

            # 添加当前用户消息（使用纠正后的文本，让 LLM 自然回答）
            messages.append({"role": "user", "content": query})

            full_answer = ""
            async for chunk in get_llm_client().chat_stream(messages):
                full_answer += chunk
                yield chunk  # 实时流式输出
                # 短暂休眠，让事件循环有机会处理其他任务
                await asyncio.sleep(0.005)  # 5ms

            # 保存对话历史
            append_history(session_id, query, full_answer)

            result = {
                "answer": full_answer,
                "agent": self.name,
            }
            if span:
                span.update(output=result)

        except Exception as e:
            logger.error(f"Chat agent stream failed: {e}")
            if span:
                span.update(output={"error": str(e)}, status_code=500)
            yield "抱歉，我遇到了一些问题，请稍后重试。"


chat_agent = ChatAgent()
