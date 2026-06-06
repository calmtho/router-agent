import json
import re
import uuid
from typing import Any, AsyncGenerator

from app.config import config
from app.services.llm_client import get_llm_client
from app.utils.logger import logger


class CoTChain:
    """CoT 推理链，用于主代理的路由决策"""

    def __init__(self):
        self.prompt_template = config.main_agent.cot_prompt_template

    async def route(self, query: str, has_file: bool = False, has_image: bool = False,
                    original_query: str | None = None) -> dict[str, str]:
        """根据用户查询决定路由到哪个子代理"""
        from app.main import get_langfuse_client

        prompt = self.prompt_template.replace("{query}", query)

        if has_image:
            prompt += "\n注意：用户已上传图片，优先考虑 vision 子代理。"
        if has_file:
            prompt += "\n注意：用户已上传文件，优先考虑 RAG 子代理。"

        messages = [
            {"role": "user", "content": prompt},
        ]

        langfuse = get_langfuse_client()

        max_retries = 3
        span_id = f"cot-route-{uuid.uuid4().hex[:8]}"
        span = langfuse.span(
            name="cot_routing", id=span_id, input={"query": query, "original_query": original_query, "has_file": has_file, "has_image": has_image}
        ) if langfuse else None

        for attempt in range(1, max_retries + 1):
            try:
                response = await get_llm_client().chat(messages, temperature=0.3)

                # 清洗推理模型的 <think>...</think> 标签，避免 JSON 解析失败
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

                # 尝试解析 JSON 响应
                try:
                    result = json.loads(response)
                    if "target" in result:
                        if span:
                            span.update(output=result)
                        return result
                except json.JSONDecodeError:
                    # 如果解析失败，尝试提取 JSON
                    json_match = re.search(r"\{[^}]+\}", response)
                    if json_match:
                        result = json.loads(json_match.group())
                        if "target" in result:
                            if span:
                                span.update(output=result)
                            return result

                # 解析失败，重试
                raise ValueError(f"Failed to parse CoT response: {response[:200]}")

            except Exception as e:
                logger.warning(f"CoT routing attempt {attempt}/{max_retries} failed: {e}")
                if attempt >= max_retries:
                    logger.error(f"CoT routing failed after {max_retries} attempts")
                    fallback_result = {"target": config.main_agent.fallback_agent, "reasoning": "路由请求失败"}
                    if span:
                        span.update(output=fallback_result, status_code=500)
                    return fallback_result

    async def route_stream(self, query: str, has_file: bool = False,
                           has_image: bool = False) -> AsyncGenerator[str, None]:
        """流式 CoT 推理，先返回固定提示，再返回完整 JSON"""
        # 先返回固定提示
        yield "正在分析你的问题...\n\n"

        # 然后调用 route() 获取完整结果
        result = await self.route(query, has_file=has_file, has_image=has_image)

        # 返回完整的推理过程（JSON 格式）
        yield json.dumps(result, ensure_ascii=False)


cot_chain = CoTChain()
