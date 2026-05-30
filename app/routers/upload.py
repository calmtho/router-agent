import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.milvus_service import milvus_service
from app.utils.file_processor import FileProcessor
from app.utils.logger import logger

router = APIRouter(prefix="/upload", tags=["upload"])

UPLOAD_DIR = Path("uploads")
_file_registry: dict[str, dict[str, Any]] = {}

file_processor = FileProcessor()


async def _process_file_background(
    file_id: str, filename: str, content: bytes, session_id: str, save_path: Path
):
    """后台异步切片 + 入库"""
    try:
        text = file_processor.process(filename, content)
        chunks = file_processor.split_text(text, chunk_size=500, overlap=50)
        await milvus_service.insert_documents(chunks, session_id, file_id)

        _file_registry[file_id]["status"] = "ready"
        _file_registry[file_id]["chunks"] = len(chunks)
        logger.info(f"File processed: {filename} -> {file_id} ({len(chunks)} chunks)")
    except Exception as e:
        _file_registry[file_id]["status"] = "error"
        _file_registry[file_id]["error"] = str(e)
        save_path.unlink(missing_ok=True)
        logger.error(f"File processing failed: {filename} -> {file_id}, error: {e}")


@router.post("")
async def upload(
    file: UploadFile = File(..., description="要上传的文档"),
    session_id: str = Form("", description="会话标识"),
) -> dict[str, Any]:
    """上传文件，立即返回，后台异步切片存入向量库"""

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    file_id = uuid.uuid4().hex[:12]
    safe_name = f"{file_id}_{file.filename}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / safe_name

    content = await file.read()
    save_path.write_bytes(content)

    # 先校验文件格式（仅扩展名，快速失败）
    ext = Path(file.filename).suffix.lower()
    supported_exts = {".txt", ".pdf", ".md", ".docx"}
    if ext not in supported_exts:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    # 注册为 processing 状态
    _file_registry[file_id] = {
        "file_id": file_id,
        "filename": file.filename,
        "session_id": session_id or "default",
        "status": "processing",
        "chunks": 0,
    }

    # 提交后台切片任务
    asyncio.create_task(
        _process_file_background(file_id, file.filename, content, session_id or "default", save_path)
    )

    logger.info(f"File uploaded (processing): {file.filename} -> {file_id}")

    return {
        "file_id": file_id,
        "filename": file.filename,
        "status": "processing",
    }


@router.get("/status/{file_id}")
async def get_upload_status(file_id: str) -> dict[str, Any]:
    """查询文件处理状态"""
    if file_id not in _file_registry:
        raise HTTPException(status_code=404, detail="文件不存在")
    return _file_registry[file_id]
