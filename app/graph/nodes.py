"""LangGraph 节点定义 - 复用现有 Agent 实现"""

import json

from app.agents.chat_agent import chat_agent
from app.agents.mcp_agent import mcp_agent
from app.agents.rag_agent import rag_agent
from app.agents.vision_agent import vision_agent
from app.config import config
from app.data.sensitive_words import contains_sensitive_word
from app.services.llm_client import get_llm_client
from app.services.typo_service import get_typo_corrector
from app.utils.logger import logger

from .state import AgentState


async def preprocess_node(state: AgentState) -> AgentState:
    """
    预处理节点 - 错别字纠正 → 敏感词过滤 → 图片路径解析
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

    # Step 3: 图片 ID → 文件路径 解析
    image_ids = state.get("image_ids", [])
    image_paths = []
    if image_ids:
        from app.routers.upload_image import resolve_image_paths
        image_paths = resolve_image_paths(image_ids)
        logger.info(f"[Preprocess] Resolved {len(image_paths)}/{len(image_ids)} image paths")

    result: AgentState = {}
    if message != original_message:
        result["message"] = message
        result["original_message"] = original_message
    if image_paths:
        result["image_paths"] = image_paths
    return result


async def router_node(state: AgentState) -> AgentState:
    """
    CoT 路由节点 - 优先使用 Reranker 快速分类，置信度不够则走 LLM CoT 兜底
    """
    from app.chains.cot_chain import cot_chain

    has_file = bool(state.get("file_ids"))
    has_image = bool(state.get("image_ids"))

    # ── 阶段 1：Reranker 快速路由 ──
    try:
        from app.services.reranker_service import get_reranker_service

        reranker = get_reranker_service()
        if reranker.is_ready:
            target, confidence = await reranker.classify_route(state["message"])
            logger.info(f"[Router] Reranker: target={target}, confidence={confidence:.3f}")

            if confidence >= config.router.confidence_threshold:
                return {
                    "target_agent": target,
                    "cot_reasoning": f"Reranker 快速路由 (置信度={confidence:.2f})",
                }
            else:
                logger.info(
                    f"[Router] 置信度不足 ({confidence:.3f} < {config.router.confidence_threshold})，"
                    f"回退 CoT"
                )
    except Exception as e:
        logger.warning(f"[Router] Reranker 路由失败，回退 CoT: {e}")

    # ── 阶段 2：LLM CoT 兜底路由 ──
    try:
        routing_result = await cot_chain.route(
            state["message"],
            has_file=has_file,
            has_image=has_image,
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


async def vision_node(state: AgentState) -> AgentState:
    """
    Vision 代理节点 - 图片理解问答

    处理流程：
      Phase 1: VL 模型提取结构化特征
      Phase 2: 文本 LLM 自检特征充足性
      Phase 3: 补偿轮（可选）
      Phase 4: 文本 LLM 基于特征回答
    """
    from app.main import get_langfuse_client
    langfuse = get_langfuse_client()

    span = None
    if langfuse:
        span = langfuse.span(
            name="vision_agent",
            input={
                "query": state["message"],
                "image_count": len(state.get("image_paths", [])),
                "session_id": state["session_id"],
            },
            session_id=state["session_id"],
        )

    try:
        result = await vision_agent.handle(state["message"], state)

        return {
            "answer": result.get("answer", ""),
            "agent_used": state.get("target_agent", "vision"),
            "features": result.get("features", ""),
        }

    except Exception as e:
        logger.error(f"Vision node failed: {e}")
        error_result = {
            "answer": "抱歉，图片分析失败，请稍后重试。",
            "agent_used": state.get("target_agent", "vision"),
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
