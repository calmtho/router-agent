"""Vision 子代理 - 图片理解问答

架构（两模型职责分离）：
  Phase 1 - VL 提取：根据用户问题从图片中提取结构化特征（JSON）
  Phase 2 - LLM 自检：文本大模型检查特征是否足以回答问题
  Phase 3 - 补偿轮（可选）：线索不足时 VL 定向补充提取
  Phase 4 - LLM 回答：文本大模型基于特征生成自然语言回答

模型选择是配置问题，架构保持职责分离：
  - VL 模型：负责视觉感知 → 结构化特征（JSON）
  - 文本 LLM：负责推理检查 + 最终回答
"""

import json
from typing import Any, AsyncGenerator

from app.agents.sub_agent_base import SubAgentBase
from app.config import config
from app.services.llm_client import get_llm_client
from app.services.vl_client import get_vl_client
from app.utils.logger import logger

# ---- 提示词模板 ----

EXTRACT_PROMPT_TEMPLATE = """你是一个精准的图片特征提取器。
请仔细分析图片，提取与用户问题相关的所有视觉特征。

用户问题：{query}

请以 JSON 格式输出提取结果，包含以下字段：
- "features": 提取到的视觉特征列表（每个特征是一个字符串描述）
- "scene_description": 场景描述（一句话概括画面内容）
- "text_content": 图片中出现的任何文本内容（如果没有则为空字符串）
- "objects": 图片中识别到的主要对象列表

注意：
1. 只提取与用户问题相关的特征
2. 如果图片不包含用户询问的信息，请如实说明"未发现相关特征"
3. 输出必须是合法的 JSON，不要包含其他文字"""

COMPENSATE_PROMPT_TEMPLATE = """你是一个精准的图片特征提取器。
之前已提取到以下特征：
{previous_features}

但用户的问题还需要更多线索才能回答。用户问题：{query}

请仔细补充分析图片，只提取【之前未提取到】且【与问题相关】的额外特征。

以 JSON 格式输出补充结果：
- "additional_features": 新提取的特征列表
- "note": 如果确实没有更多可提取的特征，请说明原因"""

CHECK_PROMPT_TEMPLATE = """你是一个严谨的信息审核者。
请判断已提取的视觉特征是否足够回答用户的问题。

用户问题：{query}
已提取的视觉特征：
{features}

请以 JSON 格式输出判断结果：
- "sufficient": true 或 false （特征是否足以回答问题）
- "missing_info": 如果 insufficient，列出缺少的关键信息（数组）
- "reason": 判断理由"""

ANSWER_PROMPT_TEMPLATE = """你是一个专业的 AI 问答助手。
请基于以下从图片中提取的视觉特征，回答用户的问题。

用户问题：{query}
视觉特征：{features}

要求：
1. 回答必须基于提取的特征，不要臆测图片中不存在的内容
2. 如果特征信息不足以回答，请诚实告知用户
3. 回答要自然流畅、条理清晰"""


class VisionAgent(SubAgentBase):
    """图片理解子代理"""

    def __init__(self):
        super().__init__("vision")
        self.vl = get_vl_client()
        self.llm = get_llm_client()

    async def can_handle(self, query: str, context: dict[str, Any]) -> bool:
        # 由 CoT 路由决定，此处返回 True
        return True

    async def handle(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """处理图片问答请求

        Args:
            query: 用户问题
            context: 上下文，需包含 image_paths: list[str]

        Returns:
            {"answer": str, "agent": "vision", ...}
        """
        image_paths: list[str] = context.get("image_paths", [])
        if not image_paths:
            return {
                "answer": "没有找到关联的图片，请先上传图片。",
                "agent": self.name,
            }

        logger.info(f"[VisionAgent] 开始处理，图片数: {len(image_paths)}, 查询: {query[:50]}...")

        try:
            # === Phase 1: VL 结构化特征提取 ===
            extract_prompt = EXTRACT_PROMPT_TEMPLATE.format(query=query)
            vl_response = await self.vl.analyze(extract_prompt, image_paths)
            features_json = self._try_parse_json(vl_response)

            if features_json is None:
                # 解析失败，直接使用原始 VL 输出
                features_str = vl_response
                structured_features = vl_response
            else:
                features_str = json.dumps(features_json, ensure_ascii=False, indent=2)
                structured_features = features_json

            logger.info(f"[VisionAgent] Phase1 特征提取完成")

            check_result = {"sufficient": True}  # 默认值（simple 模式无需自检）

            # === Phase 2 + Phase 3: 自检 + 补偿（仅 full 模式） ===
            if config.vision.phases != "simple":
                # Phase 2: LLM 自检特征是否充足
                check_prompt = CHECK_PROMPT_TEMPLATE.format(
                    query=query, features=features_str
                )
                check_response = await self.llm.chat([
                    {"role": "user", "content": check_prompt}
                ], temperature=0.1)
                check_result = self._try_parse_json(check_response) or {"sufficient": True}

                logger.info(f"[VisionAgent] Phase2 自检: sufficient={check_result.get('sufficient')}")

                # Phase 3: 补偿轮（可选）
                if not check_result.get("sufficient", True):
                    missing = check_result.get("missing_info", [])
                    logger.info(f"[VisionAgent] Phase3 补偿轮启动, 缺少信息: {missing}")

                    compensate_prompt = COMPENSATE_PROMPT_TEMPLATE.format(
                        query=query,
                        previous_features=features_str,
                    )
                    compensate_response = await self.vl.analyze(compensate_prompt, image_paths)
                    compensate_json = self._try_parse_json(compensate_response)

                    if compensate_json:
                        additional = compensate_json.get("additional_features", [])
                        if additional:
                            # 补充到现有特征中
                            if isinstance(structured_features, dict):
                                existing = structured_features.get("features", [])
                                if isinstance(existing, list):
                                    structured_features["features"] = existing + additional
                                features_str = json.dumps(structured_features, ensure_ascii=False, indent=2)
                            else:
                                features_str += "\n\n【补充特征】\n" + "\n".join(additional)

                        logger.info(f"[VisionAgent] Phase3 补偿完成, 新增 {len(additional)} 条特征")
                    else:
                        logger.info("[VisionAgent] Phase3 补偿无新增特征")
            else:
                logger.info("[VisionAgent] simple 模式，跳过 Phase2/3 自检与补偿")

            # === Phase 4: LLM 基于特征回答 ===
            answer_prompt = ANSWER_PROMPT_TEMPLATE.format(
                query=query, features=features_str
            )
            final_answer = await self.llm.chat([
                {"role": "user", "content": answer_prompt}
            ])

            logger.info(f"[VisionAgent] Phase4 回答完成")

            return {
                "answer": final_answer,
                "agent": self.name,
                "features": features_str,
                "check_result": check_result,
            }

        except Exception as e:
            logger.error(f"[VisionAgent] 处理失败: {e}")
            return {
                "answer": "抱歉，图片分析时遇到问题，请稍后重试。",
                "agent": self.name,
                "error": str(e),
            }

    async def handle_stream(
        self, query: str, context: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """流式处理（暂不支持流式 VL，回退到非流式）"""
        result = await self.handle(query, context)
        yield result.get("answer", "")

    def _try_parse_json(self, text: str) -> dict | None:
        """尝试解析 JSON，兼容含标记代码块的情况"""
        # 尝试直接解析
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 {...} 块
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        return None


# 全局实例
vision_agent = VisionAgent()
