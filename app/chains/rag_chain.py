from typing import Any

from app.config import config
from app.services.history_service import append_history, get_summary
from app.services.llm_client import get_llm_client
from app.services.milvus_service import milvus_service
from app.utils.logger import logger


class RAGChain:
    """LangChain 风格的 RAG 链"""

    def __init__(self):
        self.top_k = config.rag.top_k

    async def run(self, query: str, session_id: str | None = None, file_ids: list[str] | None = None, chat_history: list[dict] | None = None) -> dict[str, Any]:
        """执行 RAG 检索和生成"""

        # 从 Milvus 检索相关文档
        retrieved_docs = await milvus_service.search(query, session_id, top_k=self.top_k, file_ids=file_ids)

        if not retrieved_docs:
            # 没有检索到文档，返回空结果
            result = {
                "answer": "抱歉，我没有找到相关的文档内容来回答您的问题。",
                "sources": [],
                "answer_from": "no_docs",
            }
            # 仍然保存对话历史（虽然没有用到文档）
            if session_id and chat_history:
                append_history(session_id, query, result["answer"])
            return result

        # 获取摘要
        summary = get_summary(session_id) if session_id else ""

        # 构建上下文
        context = "\n\n".join([f"[来源{i+1}] {doc['text']}" for i, doc in enumerate(retrieved_docs)])

        # 构建 RAG Prompt，包含摘要
        prompt = "请根据以下文档内容回答用户的问题，如果文档中没有相关信息，请如实告知。"

        if summary:
            prompt += f"\n\n【会话摘要】\n{summary}"

        prompt += f"""

文档内容：
{context}

用户问题：{query}

请直接回答问题，不要编造信息。如果需要引用文档，使用 [来源1]、[来源2] 等方式标注。"""

        try:
            messages = [{"role": "user", "content": prompt}]
            answer = await get_llm_client().chat(messages)

            result = {
                "answer": answer,
                "sources": [{"text": doc["text"], "score": doc["score"]} for doc in retrieved_docs],
                "answer_from": "rag",
            }

            # 保存对话历史
            if session_id and chat_history:
                append_history(session_id, query, answer)

            return result

        except Exception as e:
            logger.error(f"RAG chain failed: {e}")
            error_result = {
                "answer": "抱歉，在生成回答时遇到了问题，请稍后重试。",
                "sources": retrieved_docs,
                "answer_from": "generation_failed",
            }
            # 仍然保存对话历史
            if session_id and chat_history:
                append_history(session_id, query, error_result["answer"])
            return error_result


rag_chain = RAGChain()
