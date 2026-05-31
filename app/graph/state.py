"""LangGraph State 定义"""
from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """图状态定义

    total=False 表示所有键都是可选的，后续节点可以动态添加
    """
    # 输入
    message: str                    # 用户消息（可能经过错字纠正）
    original_message: Optional[str]  # 用户原始输入（未经处理）
    session_id: str                # 会话 ID
    file_ids: Optional[list[str]]  # 引用的文件 IDs
    chat_history: list[dict]       # 聊天历史（最近 N 轮）

    # 路由结果
    target_agent: str              # 目标代理名称
    cot_reasoning: str             # CoT 推理过程

    # 输出
    answer: str                    # 最终回答
    agent_used: str                # 实际使用的代理
    sources: list                  # 引用来源
    error: Optional[str]           # 错误信息
