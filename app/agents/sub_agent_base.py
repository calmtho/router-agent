from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


class SubAgentBase(ABC):
    """子代理抽象基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def can_handle(self, query: str, context: dict[str, Any]) -> bool:
        """判断是否能处理此请求"""
        pass

    @abstractmethod
    async def handle(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """处理请求并返回结果"""
        pass

    async def handle_stream(
        self, query: str, context: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """流式处理请求，默认委托给非流式方法"""
        result = await self.handle(query, context)
        yield result.get("answer", "")
