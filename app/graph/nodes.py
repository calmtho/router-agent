"""LangGraph 节点定义 - 复用现有 Agent 实现"""

import json

from app.agents.chat_agent import chat_agent
from app.agents.mcp_agent import mcp_agent
from app.agents.rag_agent import rag_agent
from app.config import config
from app.data.sensitive_words import contains_sensitive_word
from app.services.llm_client import get_llm_client
from app.services.typo_service import get_typo_corrector
from app.utils.logger import logger

from .state import AgentState


async def preprocess_node(state: AgentState) -> AgentState:
    """
    预处理节点 - 错别字纠正 → 敏感词过滤
    """
    message = state.get("message", "")
    original_message = message

    # Step 1: 错别字纠正
    if config.preprocess.enable_typo_correction:
        try:
            corrector = get_typo_corrector()
            corrected, _ = await corrector.correct(message)
            if corrected != message:
                logger.info(f"[Preprocess] Typo: {message!r} -> {corrected!r}")
                message = corrected
        except Exception as e:
            logger.warning(f"[Preprocess] Typo correction failed: {e}")

    # Step 2: 敏感词过滤
    is_sensitive, matched_word = contains_sensitive_word(message)
    if is_sensitive:
        logger.warning(f"Sensitive content detected: {matched_word}")
        return {
            "answer": "抱歉，您的请求包含不合适的内容，我无法处理。",
            "agent_used": "preprocess",
            "cot_reasoning": f"检测到敏感词: {matched_word}",
            "error": "sensitive_content",
        }

    result: AgentState = {}
    if message != original_message:
        result["message"] = message
        result["original_message"] = original_message
    return result


async def router_node(state: AgentState) -> AgentState:
    """
    CoT 路由节点 - 使用现有的 cot_chain 进行路由决策

    从 LLM 获取智能路由决策，决定由哪个子代理处理请求。
    """
    # 延迟导入，避免循环依赖
    from app.chains.cot_chain import cot_chain

    has_file = bool(state.get("file_ids"))

    try:
        # 调用 CoT 链进行路由决策
        # cot_chain.route 内部已包含 Langfuse 追踪
        routing_result = await cot_chain.route(
            state["message"],
            has_file=has_file,
            original_query=state.get("original_message"),
        )

        return {
            "target_agent": routing_result.get("target", config.main_agent.fallback_agent),
            "cot_reasoning": routing_result.get("reasoning", ""),
        }

    except Exception as e:
        logger.error(f"Router node failed: {e}")
        return {
            "target_agent": config.main_agent.fallback_agent,
            "cot_reasoning": "路由失败，使用降级代理",
            "error": str(e),
        }


async def chat_node(state: AgentState) -> AgentState:
    """
    Chat 代理节点 - 通用对话处理

    直接调用 ChatAgent 处理用户消息。
    """
    from app.main import get_langfuse_client
    langfuse = get_langfuse_client()

    span = None
    if langfuse:
        span = langfuse.span(
            name="chat_agent",
            input={
                "query": state["message"],
                "original_message": state.get("original_message"),
                "session_id": state["session_id"],
            },
            session_id=state["session_id"],
        )

    try:
        result = await chat_agent.handle(state["message"], state)

        return {
            "answer": result.get("answer", ""),
            "agent_used": state.get("target_agent", "chat"),
            "sources": result.get("sources", []),
        }

    except Exception as e:
        logger.error(f"Chat node failed: {e}")
        error_result = {
            "answer": "抱歉，我遇到了一些问题，请稍后重试。",
            "agent_used": state.get("target_agent", "chat"),
            "error": str(e),
        }
        if span:
            span.update(output=error_result, status_code=500)
        return error_result


async def rag_node(state: AgentState) -> AgentState:
    """
    RAG 代理节点 - 基于文档的问答处理

    从 Milvus 检索相关文档并生成回答。
    """
    from app.main import get_langfuse_client
    langfuse = get_langfuse_client()

    span = None
    if langfuse:
        span = langfuse.span(
            name="rag_agent",
            input={
                "query": state["message"],
                "original_message": state.get("original_message"),
                "session_id": state["session_id"],
                "file_ids": state.get("file_ids"),
            },
            session_id=state["session_id"],
        )

    try:
        result = await rag_agent.handle(state["message"], state)

        return {
            "answer": result.get("answer", ""),
            "agent_used": state.get("target_agent", "rag"),
            "sources": result.get("sources", []),
        }

    except Exception as e:
        logger.error(f"RAG node failed: {e}")
        error_result = {
            "answer": "抱歉，检索文档或生成回答时遇到了问题，请稍后重试。",
            "agent_used": state.get("target_agent", "rag"),
            "error": str(e),
        }
        if span:
            span.update(output=error_result, status_code=500)
        return error_result


async def mcp_node(state: AgentState) -> AgentState:
    """
    MCP 代理节点 - 工具调用处理

    调用 MCP 工具执行具体操作并生成自然语言回答。
    """
    from app.main import get_langfuse_client
    langfuse = get_langfuse_client()

    span = None
    if langfuse:
        span = langfuse.span(
            name="mcp_agent",
            input={
                "query": state["message"],
                "original_message": state.get("original_message"),
                "session_id": state["session_id"],
            },
            session_id=state["session_id"],
        )

    try:
        result = await mcp_agent.handle(state["message"], state)

        return {
            "answer": result.get("answer", ""),
            "agent_used": state.get("target_agent", "mcp"),
            "sources": result.get("sources", []),
        }

    except Exception as e:
        logger.error(f"MCP node failed: {e}")
        error_result = {
            "answer": "抱歉，工具调用失败，请稍后重试。",
            "agent_used": state.get("target_agent", "mcp"),
            "error": str(e),
        }
        if span:
            span.update(output=error_result, status_code=500)
        return error_result


def fallback_router(state: AgentState) -> str:
    """
    降级路由函数 - 当目标代理失败时的降级策略

    返回降级后的目标代理名称。
    """
    return config.main_agent.fallback_agent
