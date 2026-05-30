"""
stt.py — STT 语音转写路由
POST /stt/transcribe — 上传音频文件，返回 FunASR 转写结果
"""

import time
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.stt_service import get_stt_service
from app.utils.logger import logger, log_request, log_response

router = APIRouter(prefix="/stt", tags=["STT 语音转写"])


class TranscribeResponse(BaseModel):
    text: str
    model: str
    duration_ms: int


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = "paraformer-zh",
):
    """
    上传音频文件，返回 FunASR 转写结果

    支持格式: wav, mp3, webm, ogg, flac, m4a
    前端可通过 Web Audio API 录制后直接上传
    """
    log_request("POST", "/stt/transcribe", filename=file.filename)

    # 校验文件类型（application/octet-stream 为 curl 等工具默认值，放行）
    # 使用前缀匹配，忽略 ";codecs=xxx" 等参数（如 audio/webm;codecs=opus）
    allowed_prefixes = (
        "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
        "audio/webm", "audio/ogg", "audio/flac", "audio/mp4",
        "audio/x-m4a", "application/octet-stream",
    )
    if file.content_type and not any(file.content_type.startswith(p) for p in allowed_prefixes):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {file.content_type}，请使用 wav/mp3/webm/ogg",
        )

    # 写入临时文件
    # 清理文件名中的 MIME 参数（如 "recording.webm;codecs=opus" → ".webm"）
    raw_suffix = Path(file.filename or "audio.wav").suffix
    suffix = raw_suffix.split(";")[0] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="上传的音频文件为空")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        stt = get_stt_service()
        if not stt.is_ready:
            raise HTTPException(status_code=503, detail="STT 模型尚未就绪，请稍后重试")

        t0 = time.time()
        result = await stt.transcribe(tmp_path)
        elapsed_ms = int((time.time() - t0) * 1000)

        response = TranscribeResponse(
            text=result["text"],
            model=result["model"],
            duration_ms=elapsed_ms,
        )
        log_response("success", {"text_len": len(result["text"]), "ms": elapsed_ms})
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[STT] 转写失败: {e}")
        raise HTTPException(status_code=500, detail=f"转写失败: {str(e)}")
    finally:
        # 清理临时文件
        Path(tmp_path).unlink(missing_ok=True)
