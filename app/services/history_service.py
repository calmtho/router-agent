"""对话历史服务 - 内存版实现"""

import asyncio
import re
from typing import Dict, List

from app.services.llm_client import get_llm_client
from app.utils.logger import logger

# 内存存储：session_id -> 对话历史列表
_session_history: Dict[str, List[dict]] = {}

# 摘要存储：session_id -> 摘要文本（用于上下文管理）
_session_summaries: Dict[str, str] = {}

# 标题存储：session_id -> 会话标题（用于 UI 展示）
_session_titles: Dict[str, str] = {}

# 轮次计数器：session_id -> 当前轮次号（每会话递增，取代 %2 推算）
_session_turn_counter: Dict[str, int] = {}


def get_history(session_id: str, window: int = 10) -> List[dict]:
    """获取最近 N 轮对话历史

    Args:
        session_id: 会话 ID
        window: 窗口大小，返回最近多少轮

    Returns:
        对话历史列表，格式：[{"role": "user", "content": "...", "turn": N}, ...]
    """
    if session_id not in _session_history:
        return []

    history = _session_history[session_id]
    if not history:
        return []

    # 按 turn 号取最近 window 轮
    all_turns = sorted(set(m.get("turn", 0) for m in history))
    if not all_turns:
        return []

    recent_turns = set(all_turns[-window:])
    return [m for m in history if m.get("turn", 0) in recent_turns]


def append_history(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """追加对话到历史记录

    Args:
        session_id: 会话 ID
        user_msg: 用户消息
        assistant_msg: 助手消息
    """
    if session_id not in _session_history:
        _session_history[session_id] = []
        _session_turn_counter[session_id] = 0

    turn = _session_turn_counter[session_id] + 1
    _session_turn_counter[session_id] = turn

    _session_history[session_id].extend([
        {"role": "user", "content": user_msg, "turn": turn},
        {"role": "assistant", "content": assistant_msg, "turn": turn},
    ])


def get_summary(session_id: str) -> str:
    """获取会话摘要"""
    return _session_summaries.get(session_id, "")


def set_summary(session_id: str, summary: str) -> None:
    """设置会话摘要"""
    _session_summaries[session_id] = summary


def get_title(session_id: str) -> str:
    """获取会话标题"""
    return _session_titles.get(session_id, "")


def set_title(session_id: str, title: str) -> None:
    """设置会话标题"""
    _session_titles[session_id] = title


def clear_history(session_id: str) -> bool:
    """清空指定会话的历史记录"""
    if session_id in _session_history:
        del _session_history[session_id]
    if session_id in _session_summaries:
        del _session_summaries[session_id]
    if session_id in _session_titles:
        del _session_titles[session_id]
    if session_id in _session_turn_counter:
        del _session_turn_counter[session_id]
    return True


def get_session_count() -> int:
    """获取当前内存中存储的会话数量（用于调试）"""
    return len(_session_history)


def get_all_sessions() -> Dict[str, int]:
    """获取所有会话的轮次号（用于调试）"""
    return {sid: _session_turn_counter.get(sid, 0) for sid in _session_history}


# 摘要生成锁，避免并发更新
_summary_locks: Dict[str, asyncio.Lock] = {}


def get_summary_lock(session_id: str) -> asyncio.Lock:
    """获取会话的摘要生成锁"""
    if session_id not in _summary_locks:
        _summary_locks[session_id] = asyncio.Lock()
    return _summary_locks[session_id]


def needs_summary(session_id: str, threshold: int = 10) -> bool:
    """判断是否需要生成摘要

    Args:
        session_id: 会话 ID
        threshold: 阈值，超过多少轮需要摘要

    Returns:
        True 需要摘要，False 不需要
    """
    return _session_turn_counter.get(session_id, 0) > threshold


async def generate_summary(session_id: str, conversation_history: List[dict] = None, keep_rounds: int = 10) -> str:
    """生成会话摘要（异步），并裁剪历史只保留最近 keep_rounds 轮

    Args:
        session_id: 会话 ID
        conversation_history: 对话历史（可选，默认使用 _session_history）
        keep_rounds: 保留最近多少轮对话，默认 10

    Returns:
        生成的摘要文本
    """
    lock = get_summary_lock(session_id)

    async with lock:
        # 使用传入的历史或内存中的历史
        history = conversation_history if conversation_history is not None else _session_history.get(session_id, [])
        current_turn = _session_turn_counter.get(session_id, 0)

        # 未超过保留轮数，不需要摘要
        if current_turn <= keep_rounds:
            return _session_summaries.get(session_id, "")

        try:
            # 按 turn 号分界：需要摘要的旧对话（turn ≤ cutoff）
            cutoff_turn = current_turn - keep_rounds
            old_history = [m for m in history if 0 < m.get("turn", 0) <= cutoff_turn]

            # 获取已有的旧摘要，追加到摘要 prompt 中
            existing_summary = _session_summaries.get(session_id, "")

            # 构建摘要 Prompt
            if existing_summary:
                summary_context = f"【已有摘要】\n{existing_summary}\n\n【新增对话】\n"
            else:
                summary_context = ""

            summary_prompt = f"""请根据以下内容，生成一个简短的摘要（50字以内）。

{summary_context}对话历史：
{chr(10).join(f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}" for m in old_history[-20:])}

请用中文生成摘要，格式：这是关于XXX的对话"""

            messages = [
                {"role": "system", "content": "你是一个会话摘要生成器，请用简短的中文总结对话内容。"},
                {"role": "user", "content": summary_prompt}
            ]

            summary = await get_llm_client().chat(messages)

            # 移除 <think>...</think> 推理标签（DeepSeek-R1 等模型可能输出）
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()

            # 清理摘要（移除可能的引号、前缀等）
            summary = summary.strip().strip('"').strip('\u201c').strip('\u201d').strip()

            # 限制摘要长度
            if len(summary) > 200:
                summary = summary[:200]

            # 覆盖旧摘要
            set_summary(session_id, summary)

            # 裁剪历史，只保留 turn > cutoff_turn 的消息
            if session_id in _session_history:
                _session_history[session_id] = [
                    m for m in _session_history[session_id]
                    if m.get("turn", 0) > cutoff_turn
                ]

            logger.info(f"Summary generated for session {session_id}: {summary[:50]}..., history trimmed to keep turns > {cutoff_turn}")

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary for session {session_id}: {e}")
            return _session_summaries.get(session_id, "")


async def generate_title(session_id: str) -> str:
    """生成会话标题（异步），根据前 1~2 轮对话生成精炼标题

    Args:
        session_id: 会话 ID

    Returns:
        生成的标题文本（5~15 字）
    """
    # 已有标题则不重复生成
    if _session_titles.get(session_id):
        return _session_titles[session_id]

    history = _session_history.get(session_id, [])
    if not history:
        return ""

    try:
        # 只取前 2 轮（4 条消息）作为标题生成的素材
        first_msgs = history[:4]

        title_prompt = f"""请为以下对话生成一个5~15字的标题。

对话：
{chr(10).join(f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'][:100]}" for m in first_msgs)}

直接输出标题，不要输出任何其他内容。"""

        messages = [
            {"role": "system", "content": "你只输出标题本身，5~15个字，不加引号、不加前缀、不解释。"},
            {"role": "user", "content": title_prompt}
        ]

        title = await get_llm_client().chat(messages)

        # 移除 <think>...</think> 推理标签
        title = re.sub(r'<think>.*?</think>', '', title, flags=re.DOTALL).strip()

        # 清理推理泄漏：LLM 可能输出 "用户需要...标题：关于AI" 之类的推理过程
        # 尝试从分隔符后提取真正标题
        title = title.strip()
        for sep in ["：", ":", "——", "—", ">>"]:
            if sep in title:
                parts = title.split(sep)
                # 取最后一个分隔符后的内容（最可能是真正的标题）
                candidate = parts[-1].strip().strip('"').strip('\u201c').strip('\u201d')
                if 2 <= len(candidate) <= 20:
                    title = candidate
                    break

        # 移除常见前缀
        for prefix in ["标题：", "标题:", "会话标题：", "会话标题:"]:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()

        # 清理引号
        title = title.strip('"').strip('\u201c').strip('\u201d').strip()

        # 限制标题长度
        if len(title) > 20:
            title = title[:20]

        set_title(session_id, title)

        logger.info(f"Title generated for session {session_id}: {title}")

        return title

    except Exception as e:
        logger.error(f"Failed to generate title for session {session_id}: {e}")
        return _session_titles.get(session_id, "")
