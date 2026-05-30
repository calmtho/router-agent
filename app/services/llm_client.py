from typing import Any, AsyncGenerator

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatGenerationChunk

from app.config import config
from app.utils.logger import log_error


class LLMClient:
    """LLM 客户端，基于 LangChain ChatOpenAI + OpenAIEmbeddings"""

    def __init__(self, callback_handler=None):
        self._callback_handler = callback_handler
        self.chat_model = ChatOpenAI(
            base_url=config.llm.openai_base_url,
            api_key=config.llm.api_key,
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            callbacks=[callback_handler] if callback_handler else [],
        )
        self.embed_model = HuggingFaceEmbeddings(
            model_name=config.milvus.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _to_lc_messages(self, messages: list[dict[str, str]]) -> list:
        role_map = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }
        return [role_map[m["role"]](content=m["content"]) for m in messages]

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """同步聊天接口，返回完整响应"""
        lc_messages = self._to_lc_messages(messages)

        callbacks = [self._callback_handler] if self._callback_handler else []

        model = self.chat_model
        if temperature is not None or max_tokens is not None:
            model = ChatOpenAI(
                base_url=config.llm.openai_base_url,
                api_key=config.llm.api_key,
                model=config.llm.model_name,
                temperature=temperature if temperature is not None else config.llm.temperature,
                max_tokens=max_tokens if max_tokens is not None else config.llm.max_tokens,
                callbacks=callbacks,
            )

        try:
            response = await model.ainvoke(lc_messages)
            return response.content
        except Exception as e:
            log_error(e, "LLM chat request failed")
            raise

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """流式聊天接口，逐块返回响应内容"""
        lc_messages = self._to_lc_messages(messages)

        callbacks = [self._callback_handler] if self._callback_handler else []

        model = self.chat_model
        if temperature is not None or max_tokens is not None:
            model = ChatOpenAI(
                base_url=config.llm.openai_base_url,
                api_key=config.llm.api_key,
                model=config.llm.model_name,
                temperature=temperature if temperature is not None else config.llm.temperature,
                max_tokens=max_tokens if max_tokens is not None else config.llm.max_tokens,
                callbacks=callbacks,
                stream=True,  # 启用流式
            )

        try:
            async for chunk in model.astream(lc_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            log_error(e, "LLM chat stream request failed")
            raise

    async def embed(self, text: str) -> list[float]:
        try:
            return await self.embed_model.aembed_query(text)
        except Exception as e:
            log_error(e, "Embedding request failed")
            raise


llm_client: LLMClient | None = None


def get_llm_client(handler=None) -> LLMClient:
    """获取全局 LLMClient 单例（延迟初始化，可传入 callback handler）"""
    global llm_client
    if llm_client is None:
        llm_client = LLMClient(callback_handler=handler)
    return llm_client
