from typing import Any, AsyncGenerator

import asyncio

from app.agents.sub_agent_base import SubAgentBase
from app.chains.rag_chain import rag_chain
from app.utils.logger import logger


class RAGAgent(SubAgentBase):
    """RAG 子代理，处理基于文档的问答"""

    def __init__(self):
        super().__init__("rag")

    async def can_handle(self, query: str, context: dict[str, Any]) -> bool:
        return bool(context.get("file_ids") or context.get("session_id"))

    async def handle(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        from app.main import get_langfuse_client

        session_id = context.get("session_id", "default")
        file_ids = context.get("file_ids")
        chat_history = context.get("chat_history", [])

        langfuse = get_langfuse_client()
        span = langfuse.span(
            name="rag_agent",
            input={"query": query, "session_id": session_id, "file_ids": file_ids},
            session_id=session_id,
        ) if langfuse else None

        try:
            result = await rag_chain.run(query, session_id, file_ids=file_ids, chat_history=chat_history)
            result["agent"] = self.name

            if span:
                span.update(output=result)
            return result

        except Exception as e:
            logger.error(f"RAG agent failed: {e}")
            error_result = {
                "answer": "抱歉，检索文档或生成回答时遇到了问题，请稍后重试。",
                "agent": self.name,
                "error": str(e),
            }
            if span:
                span.update(output=error_result, status_code=500)
            return error_result

    async def handle_stream(
        self, query: str, context: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """流式 RAG 响应（仅流式输出生成的回答，不包括检索过程）"""
        from app.main import get_langfuse_client

        session_id = context.get("session_id", "default")
        file_ids = context.get("file_ids")
        chat_history = context.get("chat_history", [])

        langfuse = get_langfuse_client()
        span = None
        if langfuse:
            span = langfuse.span(
                name="rag_agent_stream",
                input={"query": query, "session_id": session_id, "file_ids": file_ids},
                session_id=session_id,
            )

        try:
            # 先执行 RAG 检索（非流式）
            result = await rag_chain.run(query, session_id, file_ids=file_ids, chat_history=chat_history)
            result["agent"] = self.name

            # 流式输出回答内容 - 按字节流式输出，每块约50字符
            answer = result.get("answer", "")
            chunk_size = 50  # 每次流式输出的字符数

            # 使用 async_generator 来避免阻塞
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i : i + chunk_size]
                yield chunk
                # 短暂休眠，让事件循环有机会处理其他任务
                await asyncio.sleep(0.01)  # 10ms

            if span:
                span.update(output=result)

        except Exception as e:
            logger.error(f"RAG agent stream failed: {e}")
            if span:
                span.update(output={"error": str(e)}, status_code=500)
            yield "抱歉，检索文档或生成回答时遇到了问题，请稍后重试。"


rag_agent = RAGAgent()
