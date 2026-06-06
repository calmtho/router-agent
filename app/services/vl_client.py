"""VL 多模态客户端 - 基于 LangChain ChatOpenAI 的图片理解接口

支持任何兼容 OpenAI API 格式的 VL 模型（Qwen2.5-VL、InternVL2、gpt-4o-mini 等）。
通过多模态 content 数组传递图片（Base64 data URI）和文本提示。
"""

import base64
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.config import config
from app.utils.logger import logger


class VLClient:
    """VL 多模态客户端

    使用 ChatOpenAI 构造包含图片的 HumanMessage，
    兼容 OpenAI、DashScope、OneAPI 等标准接口。
    """

    def __init__(self):
        self.vl_model = ChatOpenAI(
            base_url=config.vision.openai_base_url,
            api_key=config.vision.api_key,
            model=config.vision.model_name,
            temperature=config.vision.temperature,
            max_tokens=config.vision.max_tokens,
        )

    def _encode_image(self, image_path: str) -> str:
        """将图片文件编码为 Base64 data URI

        Args:
            image_path: 图片文件路径

        Returns:
            data URI 字符串，如 data:image/jpeg;base64,...
        """
        path = Path(image_path)
        suffix = path.suffix.lower()

        # 映射文件后缀到 MIME 类型
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }
        mime = mime_map.get(suffix, "image/jpeg")

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime};base64,{b64}"

    async def analyze(self, prompt: str, image_paths: list[str]) -> str:
        """使用 VL 模型分析图片

        Args:
            prompt: 分析提示词，要求输出特定格式
            image_paths: 图片文件路径列表

        Returns:
            VL 模型的文本响应
        """
        # 构建多模态 content 数组
        content: list[dict] = [{"type": "text", "text": prompt}]

        for img_path in image_paths:
            try:
                data_uri = self._encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                })
            except Exception as e:
                logger.warning(f"[VLClient] 编码图片失败 {img_path}: {e}")
                content.append({"type": "text", "text": f"[图片加载失败: {img_path}]"})

        message = HumanMessage(content=content)

        try:
            response = await self.vl_model.ainvoke([message])
            return response.content
        except Exception as e:
            logger.error(f"[VLClient] VL 分析失败: {e}")
            raise


# 全局单例
_vl_client: VLClient | None = None


def get_vl_client() -> VLClient:
    """获取 VL 客户端单例（延迟初始化）"""
    global _vl_client
    if _vl_client is None:
        _vl_client = VLClient()
    return _vl_client
