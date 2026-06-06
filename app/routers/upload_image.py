"""图片上传路由

支持上传图片文件，返回 image_id 用于后续 Chat 请求。
图片保存在 uploads/images/ 目录下。
注册表持久化到 .registry.json，防止 uvicorn reload 丢失。
"""

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.utils.logger import logger

router = APIRouter(prefix="/upload", tags=["upload"])

IMAGE_DIR = Path("uploads/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = IMAGE_DIR / ".registry.json"

# 支持上传的图片格式
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


def _load_registry() -> dict[str, dict[str, Any]]:
    """从磁盘加载图片注册表，自动清理已不存在的文件记录"""
    if not REGISTRY_FILE.exists():
        return {}
    try:
        data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        valid = {}
        for img_id, info in data.items():
            path = info.get("path", "")
            if path and Path(path).exists():
                valid[img_id] = info
            else:
                logger.warning(f"[ImageRegistry] 清理失效图片: {img_id} -> {path}")
        return valid
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[ImageRegistry] 加载注册表失败: {e}")
        return {}


def _save_registry(registry: dict[str, dict[str, Any]]) -> None:
    """保存图片注册表到磁盘"""
    try:
        REGISTRY_FILE.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.error(f"[ImageRegistry] 保存注册表失败: {e}")


# 启动时从磁盘加载注册表
_image_registry: dict[str, dict[str, Any]] = _load_registry()
logger.info(f"[ImageRegistry] 启动时加载了 {len(_image_registry)} 张图片注册信息")


@router.post("/image")
async def upload_image(
    file: UploadFile = File(..., description="要上传的图片"),
) -> dict[str, Any]:
    """上传图片，返回 image_id

    支持格式：jpg, jpeg, png, webp, gif, bmp, tiff
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片格式: {ext}，支持: {', '.join(SUPPORTED_IMAGE_EXTS)}",
        )

    image_id = uuid.uuid4().hex[:12]
    safe_name = f"{image_id}{ext}"
    save_path = IMAGE_DIR / safe_name

    content = await file.read()

    # 简单校验：至少是有效图片文件头
    if len(content) < 20:
        raise HTTPException(status_code=400, detail="文件内容过短，不是有效图片")

    save_path.write_bytes(content)

    _image_registry[image_id] = {
        "image_id": image_id,
        "filename": file.filename,
        "path": str(save_path.resolve()),
    }
    # 持久化到磁盘
    _save_registry(_image_registry)

    logger.info(f"Image uploaded: {file.filename} -> {image_id} ({len(content)} bytes)")

    return {
        "image_id": image_id,
        "filename": file.filename,
        "size_bytes": len(content),
    }


def resolve_image_paths(image_ids: list[str]) -> list[str]:
    """根据 image_ids 解析出图片文件路径列表"""
    paths = []
    for img_id in image_ids:
        info = _image_registry.get(img_id)
        if info:
            path = info.get("path")
            if path and Path(path).exists():
                paths.append(path)
            else:
                logger.warning(f"[ImageRegistry] 图片不存在或路径无效: {img_id} -> {path}")
        else:
            logger.warning(f"[ImageRegistry] 未找到图片: {img_id}")
    return paths
