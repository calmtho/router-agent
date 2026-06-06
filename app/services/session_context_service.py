"""Session Context Bag - 为每个 session 维护上下文包

职责：
  1. 自动继承上一轮用到的 image_ids
  2. TTL 过期清理
  3. 显式清除（用户要求"不管那张图"时）
"""

import time
from typing import Dict, List

from app.utils.logger import logger

# 内存存储：session_id -> context dict
_session_context: Dict[str, dict] = {}

# 默认 TTL：30 分钟
DEFAULT_TTL = 30 * 60


def get_context(session_id: str) -> dict:
    """获取会话的上下文包"""
    return _session_context.get(session_id, {})


def set_context(session_id: str, context: dict):
    """设置会话的上下文包"""
    _session_context[session_id] = context


def update_image_ids(session_id: str, image_ids: List[str]):
    """更新会话的图片 ID 列表（去重 + 去空）

    Args:
        session_id: 会话 ID
        image_ids: 本轮用到的图片 ID 列表
    """
    if not image_ids:
        return

    ctx = _session_context.get(session_id, {})
    existing: list[str] = ctx.get("image_ids", [])
    # 去重合并，保留顺序
    merged = list(dict.fromkeys(existing + image_ids))
    ctx["image_ids"] = merged
    ctx["last_updated"] = time.time()
    _session_context[session_id] = ctx
    logger.debug(f"[SessionContext] session={session_id} image_ids updated: {merged}")


def get_recent_image_ids(session_id: str, max_age: float = DEFAULT_TTL) -> List[str]:
    """获取会话最近使用的图片 ID（未超时的）

    Args:
        session_id: 会话 ID
        max_age: TTL 秒数，默认 30 分钟

    Returns:
        未过期的图片 ID 列表
    """
    ctx = _session_context.get(session_id, {})
    last_updated = ctx.get("last_updated", 0)
    if time.time() - last_updated > max_age:
        logger.debug(f"[SessionContext] session={session_id} image_ids expired")
        return []
    return ctx.get("image_ids", [])


def clear_image_ids(session_id: str):
    """清除会话的图片 ID"""
    ctx = _session_context.get(session_id, {})
    ctx["image_ids"] = []
    ctx["last_updated"] = time.time()
    _session_context[session_id] = ctx
    logger.info(f"[SessionContext] session={session_id} image_ids cleared")


def clear_context(session_id: str):
    """清除整个会话上下文包"""
    _session_context.pop(session_id, None)
    logger.info(f"[SessionContext] session={session_id} context cleared")


def get_context_stats() -> Dict[str, int]:
    """获取上下文统计（调试用）"""
    return {sid: len(ctx.get("image_ids", [])) for sid, ctx in _session_context.items()}
