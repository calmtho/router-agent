"""
stt_service.py — FunASR 语音转文字服务（封装 Paraformer-zh 模型）
"""

import asyncio
import re
import tempfile
from pathlib import Path

from app.utils.logger import logger


class STTService:
    """FunASR 语音转文字服务（封装 Paraformer-zh 模型）"""

    def __init__(self):
        self._model = None
        self._model_type = "paraformer-zh"

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load_model(self, model_type: str = "paraformer-zh"):
        """加载 FunASR Paraformer-zh 离线模型（应在应用启动时调用一次）"""
        try:
            from funasr import AutoModel
        except ImportError:
            logger.warning("[STT] funasr 未安装，STT 功能不可用。请运行: pip install funasr modelscope")
            return False

        logger.info(f"[STT] 正在加载模型: {model_type} ...")
        try:
            self._model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",    # 语音活动检测（长音频自动切分）
                punc_model="ct-punc",    # 标点恢复（同时消除多余空格）
                disable_update=True,
                hub="ms",
            )
            self._model_type = model_type
            logger.info("[STT] 模型加载完成")
            return True
        except Exception as e:
            logger.error(f"[STT] 模型加载失败: {e}")
            self._model = None
            return False

    @staticmethod
    def _ensure_wav(audio_path: str | Path) -> Path:
        """
        确保音频文件为 WAV 格式。
        若输入已是 wav 则直接返回原路径；否则用 soundfile 转换为 16kHz mono WAV 临时文件。
        使用 soundfile 库读取音频并转写为 16kHz mono WAV。
        """
        audio_path = Path(audio_path)
        if audio_path.suffix.lower() == ".wav":
            return audio_path

        try:
            import soundfile as sf
            import numpy as np
        except ImportError:
            raise RuntimeError(
                "需要 soundfile 库来转换音频格式，请运行: pip install soundfile numpy"
            )

        data, sample_rate = sf.read(str(audio_path), dtype="float32")

        # 转 mono
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # 重采样到 16kHz（线性插值，对语音足够）
        if sample_rate != 16000:
            duration = len(data) / sample_rate
            target_length = int(duration * 16000)
            x_old = np.linspace(0, duration, len(data), dtype=np.float32)
            x_new = np.linspace(0, duration, target_length, dtype=np.float32)
            data = np.interp(x_new, x_old, data).astype(np.float32)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, data, 16000)
        tmp.close()
        return Path(tmp.name)

    async def transcribe(self, audio_path: str | Path) -> dict:
        """
        异步转写音频文件
        返回: {"text": "转写结果", "model": "paraformer-zh"}
        """
        if not self.is_ready:
            raise RuntimeError("STT 模型未初始化")

        # 确保音频是 WAV 格式（FunASR 对 wav 支持最好）
        wav_path = self._ensure_wav(audio_path)
        converted = wav_path != Path(audio_path)

        # 在独立线程中执行同步的 FunASR inference
        result = await asyncio.to_thread(
            self._model.generate, input=str(wav_path)
        )

        # 清理转换产生的临时文件
        if converted:
            wav_path.unlink(missing_ok=True)

        text = ""
        if result and len(result) > 0:
            text = result[0].get("text", "").strip()
            # 去除中文字符之间的多余空格（英文单词间保留空格）
            text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)

        return {
            "text": text,
            "model": self._model_type,
        }


# ── 全局单例 ──────────────────────────────────────────
_stt_service: STTService | None = None


def get_stt_service() -> STTService:
    """获取全局 STT 服务实例"""
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service
