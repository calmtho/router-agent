"""中文错别字纠正服务 —— 基于 macbert4csc-base-chinese 直接推理"""

import asyncio
import re

import torch

from app.utils.logger import logger

# 检测文本中是否包含"真英文"的正则
# 全大写缩写：LLM, API; 首字母大写：Agent, Python; 驼峰：macOS, iPhone
_REAL_ENGLISH_PATTERN = re.compile(r'\b[A-Z]{2,}\b|\b[A-Z][a-z]+\b|[a-z]+[A-Z]\w*')


class TypoCorrector:
    """基于 shibing624/macbert4csc-base-chinese 的轻量纠错"""

    MODEL_NAME = "shibing624/macbert4csc-base-chinese"

    def __init__(self):
        self._model = None
        self._tokenizer = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load_model(self, model_name: str | None = None) -> bool:
        try:
            from transformers import BertForMaskedLM, BertTokenizer
        except ImportError:
            logger.warning("[Typo] transformers 未安装，纠错功能不可用")
            return False

        name = model_name or self.MODEL_NAME
        logger.info(f"[Typo] 正在加载纠错模型: {name} ...")
        try:
            self._tokenizer = BertTokenizer.from_pretrained(name)
            self._model = BertForMaskedLM.from_pretrained(name)
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            logger.info("[Typo] 纠错模型加载完成")
            return True
        except Exception as e:
            logger.error(f"[Typo] 纠错模型加载失败: {e}")
            return False

    @staticmethod
    def _has_real_english(text: str) -> bool:
        """检测是否包含真正的英文（非拼音/无意义小写）"""
        return bool(_REAL_ENGLISH_PATTERN.search(text))

    async def correct(self, text: str) -> tuple[str, list]:
        """纠正中文错别字，返回 (纠正后文本, [(错字, 正字, 位置), ...])"""
        if not self.is_ready or not text.strip():
            return text, []

        # 如果包含大写缩写/驼峰等真英文，跳过纠错
        if self._has_real_english(text):
            logger.debug(f"[Typo] 检测到英文，跳过纠错: {text!r}")
            return text, []

        def _infer():
            with torch.no_grad():
                inputs = self._tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = self._model(**inputs)
            logits = outputs.logits
            corrected_ids = torch.argmax(logits, dim=-1)
            corrected_text = self._tokenizer.decode(
                corrected_ids[0], skip_special_tokens=True
            ).replace(" ", "")
            # 截取到原文长度，防止模型输出多字（一般不会，但保险起见）
            corrected_text = corrected_text[: len(text)]
            errors = self._get_errors(text, corrected_text)
            return corrected_text, errors

        try:
            corrected_text, errors = await asyncio.to_thread(_infer)
            if errors:
                logger.info(f"[Typo] 纠正: {text!r} -> {corrected_text!r}, {errors}")
            return corrected_text, errors
        except Exception as e:
            logger.warning(f"[Typo] 纠正失败: {e}")
            return text, []

    @staticmethod
    def _get_errors(original: str, corrected: str) -> list[tuple[str, str, int]]:
        """对比纠正后文本与原文，返回 [(错字, 正字, 位置), ...]"""
        errors = []
        for i, orig_char in enumerate(original):
            if i >= len(corrected):
                break
            if orig_char == corrected[i]:
                continue
            errors.append((orig_char, corrected[i], i))
        return errors


# 全局单例
_typo_corrector: TypoCorrector | None = None


def get_typo_corrector() -> TypoCorrector:
    global _typo_corrector
    if _typo_corrector is None:
        _typo_corrector = TypoCorrector()
    return _typo_corrector
