import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.graph.graph import app_graph
from app.services.history_service import generate_summary, generate_title, get_history, get_title, needs_summary, update_turn_metadata
from app.services.session_context_service import get_recent_image_ids, update_image_ids
from app.utils.logger import log_request, log_response, logger


def _sse_reasoning(content: str) -> str:
    """构建 SSE reasoning 消息"""
    return f"data: {json.dumps({'type': 'reasoning', 'content': content}, ensure_ascii=False)}\n\n"


def _is_pure_greeting_or_intro(query: str) -> bool:
    """
    纯问候/纯自我介绍 → True，跳过 pipeline 直接路由 chat
    含实质请求 → False，交给两阶段 Reranker
    """
    q = query.strip()

    # 含实质动作词 → 否决
    action_words = [
        "帮", "算", "查", "搜", "找", "写", "画", "翻译",
        "告诉", "解释", "说明", "怎么", "如何", "什么", "为什么",
        "图片", "照片", "文件", "文档", "上传"
    ]
    for w in action_words:
        if w in q:
            return False

    # 纯问候/自我介绍模式
    patterns = [
        r'^(你好|您好|嗨|哈喽|hello|hi|hey|早上好|下午好|晚上好|晚安|再见|拜拜|bye|在吗|在不在)[呀啊哦噢哟]*[\s!！。.,，~～]*$',
        r'^(你好|您好|嗨|哈喽|hello|hi|hey)[呀啊哦噢哟]*[\s,，]+我(是|叫|叫作).+$',
        r'^我(是|叫|叫作).+$',
        r'^(好久不见|最近怎么样|吃了吗|干啥呢|在干嘛)[\s!！。.,，~～]*$',
        r'^(谢谢|多谢|感谢|辛苦了|ok|好的|嗯嗯|哦哦|哈哈)[你您]?[\s!！。.,，~～]*$',
    ]
    for p in patterns:
        if re.match(p, q, re.IGNORECASE):
            return True

    return False


def _strip_greeting_prefix(query: str) -> str:
    """去掉开头的问候前缀，避免 '你好，帮我xxx' 中的问候词干扰 Reranker"""
    q = query.strip()
    m = re.match(r'^(你好|您好|嗨|哈喽|hello|hi|hey)[呀啊哦噢哟]?[，,。.]+\s*', q)
    if m:
        stripped = q[m.end():].strip()
        if stripped:
            return stripped
    return q


async def _classify_via_reranker(query: str) -> tuple[str | None, dict | None]:
    """Reranker 两阶段路由，返回 (target, result_dict) 或 (None, None)"""
    try:
        from app.services.reranker_service import get_reranker_service
        from app.config import config as app_config

        reranker = get_reranker_service()
        if not reranker.is_ready:
            return None, None

        fast_target, fast_conf = await reranker.classify_route(query)
        logger.info(f"[Stream Route] Reranker: target={fast_target}, confidence={fast_conf:.3f}")

        if fast_conf >= app_config.router.confidence_threshold:
            return fast_target, {
                "target": fast_target,
                "reasoning": f"Reranker 快速路由 (置信度={fast_conf:.2f})"
            }
    except Exception as e:
        logger.warning(f"[Stream Route] Reranker 失败，回退 CoT: {e}")

    return None, None


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    file_ids: list[str] = []
    image_ids: list[str] = []
    chat_history: list[dict] = []


# 导入子代理（用于 stream 函数）
from app.agents.chat_agent import chat_agent
from app.agents.rag_agent import rag_agent
from app.agents.mcp_agent import mcp_agent
from app.agents.vision_agent import vision_agent


@router.post("")
async def chat(body: ChatRequest) -> dict[str, Any]:
    """聊天接口，使用 LangGraph 执行主子代理路由"""

    log_request("POST", "/chat", session_id=body.session_id, file_ids=body.file_ids)

    # 获取历史对话（如果 session_id 存在）
    history = get_history(body.session_id) if body.session_id else []

    # 继承 Session Context 中的 image_ids（非必传 scenario）
    image_ids = body.image_ids or []
    if not image_ids and body.session_id:
        inherited = get_recent_image_ids(body.session_id)
        if inherited:
            image_ids = inherited

    # file_ids 空值安全处理
    file_ids = body.file_ids or []

    # 构建初始状态
    initial_state = {
        "message": body.message,
        "session_id": body.session_id or "default",
        "file_ids": file_ids,
        "image_ids": image_ids,
        "chat_history": history,  # 注入历史
        "target_agent": "",
        "cot_reasoning": "",
        "answer": "",
        "agent_used": "",
        "sources": [],
    }

    try:
        # 执行 LangGraph
        result = await app_graph.ainvoke(initial_state)

        # ---- 保存 Session Context + 历史 metadata ----
        if body.session_id:
            # 将本轮用到的 image_ids 存入 Session Context
            used_ids = image_ids or result.get("image_ids", [])
            if used_ids:
                update_image_ids(body.session_id, used_ids)

            # 补写历史记录的 metadata
            user_meta = {}
            if used_ids:
                user_meta["image_ids"] = used_ids
            assistant_meta = {}
            if result.get("features"):
                assistant_meta["features"] = result["features"]
            if result.get("agent_used"):
                assistant_meta["agent_used"] = result["agent_used"]
            if user_meta or assistant_meta:
                update_turn_metadata(
                    body.session_id,
                    user_metadata=user_meta or None,
                    assistant_metadata=assistant_meta or None,
                )

        response = {
            "reply": result.get("answer", ""),
            "agent_used": result.get("agent_used", "chat"),
            "cot_reasoning": result.get("cot_reasoning", ""),
        }
        if "sources" in result:
            response["sources"] = result["sources"]
        log_response("success", response)
        return response
    except Exception as e:
        log_response("error", str(e))
        raise HTTPException(status_code=500, detail=f"处理请求时出错：{str(e)}")


@router.post("/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    """流式聊天接口，实时返回推理过程和回答"""

    log_request("POST", "/chat/stream", session_id=body.session_id, file_ids=body.file_ids)

    async def stream_generator():
        """流式生成器，逐块返回数据（SSE 格式）"""

        import json
        import asyncio

        # 检查 file_ids 是否全部就绪
        from app.routers.upload import _file_registry

        file_ids = body.file_ids or []
        image_ids = body.image_ids or []

        processing_files = []
        for fid in file_ids:
            info = _file_registry.get(fid)
            if info and info.get("status") != "ready":
                processing_files.append(info.get("filename", fid))

        if processing_files:
            # 等待文件处理完成，每 0.5 秒检查一次
            waiting_msg = "正在分析文件内容，请耐心等待: " + ", ".join(processing_files)
            yield f"data: {json.dumps({'type': 'waiting', 'message': waiting_msg}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

            while True:
                all_ready = True
                for fid in file_ids:
                    info = _file_registry.get(fid)
                    if not info or info.get("status") == "processing":
                        all_ready = False
                        break
                    if info.get("status") == "error":
                        err_name = info.get("filename", fid)
                        err_detail = info.get("error", "未知错误")
                        err_msg = f"文件处理失败: {err_name} - {err_detail}"
                        yield f"data: {json.dumps({'type': 'error', 'message': err_msg}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.01)
                        return

                if all_ready:
                    break
                await asyncio.sleep(0.5)

            yield f"data: {json.dumps({'type': 'waiting_done', 'message': '文件分析完成，开始回答...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

        # 获取历史对话（如果 session_id 存在）
        history = get_history(body.session_id) if body.session_id else []

        # 继承 Session Context 中的 image_ids（非必传 scenario）
        if not image_ids and body.session_id:
            inherited = get_recent_image_ids(body.session_id)
            if inherited:
                image_ids = inherited

        # 构建初始状态
        initial_state = {
            "message": body.message,
            "session_id": body.session_id or "default",
            "file_ids": file_ids,
            "image_ids": image_ids,
            "chat_history": history,  # 注入历史
            "target_agent": "",
            "cot_reasoning": "",
            "answer": "",
            "agent_used": "",
            "sources": [],
        }

        import json
        import asyncio

        try:
            # 1. 先执行预处理节点（敏感词过滤）
            from app.graph.nodes import preprocess_node

            preprocessed = await preprocess_node(initial_state)
            if preprocessed.get("error") == "sensitive_content":
                # 敏感内容，直接返回
                yield f"data: {json.dumps({'type': 'error', 'message': '检测到敏感内容'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)  # 确保数据发送
                return

            # 合并预处理结果（错别字纠正 + 图片路径解析）
            initial_state.update(preprocessed)

            # 2. 执行 CoT 路由节点（流式推理）
            from app.chains.cot_chain import cot_chain
            from app.main import get_langfuse_client

            langfuse = get_langfuse_client()
            span = None
            if langfuse:
                span = langfuse.span(
                    name="cot_routing_stream",
                    input={
                        "query": body.message,
                        "original_message": initial_state.get("original_message"),
                        "has_file": bool(file_ids),
                        "has_image": bool(image_ids),
                    },
                    session_id=initial_state["session_id"],
                )

            # ── 路由决策：明确场景跳过 CoT LLM 调用 ──
            has_file = bool(file_ids)
            has_image = bool(image_ids)  # image_ids 已经继承了 session 上下文中的图片
            routing_result = {"target": "chat", "reasoning": ""}

            if has_image and not has_file:
                # 仅图片，直接路由到 vision
                target_agent = "vision"
                routing_result = {"target": "vision", "reasoning": "用户上传了图片，直接路由"}
                reasoning_msg = json.dumps({'type': 'reasoning', 'content': '检测到图片，正在分析…\n'}, ensure_ascii=False)
                yield f"data: {reasoning_msg}\n\n"
                await asyncio.sleep(0.01)
                logger.info(f"Fast Route (vision) - Query: {body.message[:50]}")

            elif has_file:
                # 有文件（无论是否同时有图片），优先路由到 rag 检索文档
                target_agent = "rag"
                routing_result = {"target": "rag", "reasoning": "用户上传了文件，直接路由"}
                reasoning_msg = json.dumps({'type': 'reasoning', 'content': '检测到文件，正在检索…\n'}, ensure_ascii=False)
                yield f"data: {reasoning_msg}\n\n"
                await asyncio.sleep(0.01)
                logger.info(f"Fast Route (rag) - Query: {body.message[:50]}")

            else:
                # 场景不明（无附件）
                target_agent = None
                routing_result = {"target": "chat", "reasoning": ""}

                if _is_pure_greeting_or_intro(body.message):
                    # 纯问候/自我介绍 → 快速通道
                    target_agent = "chat"
                    routing_result = {"target": "chat", "reasoning": "问候语快速路由"}
                    yield _sse_reasoning('分析完成 → chat 代理处理中…\n')
                    await asyncio.sleep(0.01)
                    logger.info(f"Fast Route (chat) - Query: {body.message[:50]}")
                else:
                    clean_query = _strip_greeting_prefix(body.message)
                    target_agent, route_result = await _classify_via_reranker(clean_query)
                    if route_result is not None:
                        routing_result = route_result
                    if target_agent is not None:
                        yield _sse_reasoning(f'分析完成 → {target_agent} 代理处理中…\n')
                        await asyncio.sleep(0.01)
                        logger.info(f"Fast Route ({target_agent}) - Query: {body.message[:50]}")

                if target_agent is None:
                    # Reranker 没命中，走原来的 CoT LLM 路由
                    cot_full = ""
                    async for chunk in cot_chain.route_stream(body.message, has_file=has_file, has_image=has_image):
                        cot_full += chunk
                        # 实时流式输出推理过程（JSON）
                        yield f"data: {json.dumps({'type': 'reasoning', 'content': chunk}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.01)

                    # 解析路由目标
                    try:
                        result = json.loads(cot_full)
                        if "target" in result:
                            routing_result = result
                    except json.JSONDecodeError:
                        json_match = re.search(r"\{[^}]+\}", cot_full)
                        if json_match:
                            try:
                                result = json.loads(json_match.group())
                                if "target" in result:
                                    routing_result = result
                            except json.JSONDecodeError:
                                pass
                    target_agent = routing_result.get("target", "chat")

                    # 打印 CoT 推理过程到控制台
                    logger.info(f"CoT Routing - Query: {body.message}")
                    logger.info(f"CoT Response: {cot_full.strip()}")
                    logger.info(f"CoT Target Agent: {target_agent}")

            if span:
                span.update(output=routing_result)

            # 3. 根据路由结果，执行对应代理（流式响应）
            agent_used = target_agent
            answer_full = ""

            query = initial_state.get("message", body.message)

            if target_agent == "chat":
                async for chunk in chat_agent.handle_stream(query, initial_state):
                    answer_full += chunk
                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)  # 确保数据发送
            elif target_agent == "rag":
                async for chunk in rag_agent.handle_stream(query, initial_state):
                    answer_full += chunk
                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)  # 确保数据发送
            elif target_agent == "mcp":
                async for chunk in mcp_agent.handle_stream(query, initial_state):
                    answer_full += chunk
                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)  # 确保数据发送
            elif target_agent == "vision":
                async for chunk in vision_agent.handle_stream(query, initial_state):
                    answer_full += chunk
                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)  # 确保数据发送
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': f'未知的代理类型: {target_agent}'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

            # 4. 流式结束
            final_response = {
                "type": "done",
                "agent_used": agent_used,
                "cot_reasoning": routing_result.get("reasoning", ""),
                "answer": answer_full,
            }
            yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

            # 5. 保存 Session Context + 历史 metadata
            if body.session_id:
                if image_ids:
                    update_image_ids(body.session_id, image_ids)

                user_meta = {}
                if image_ids:
                    user_meta["image_ids"] = image_ids
                assistant_meta = {"agent_used": agent_used}
                if user_meta or assistant_meta:
                    update_turn_metadata(
                        body.session_id,
                        user_metadata=user_meta or None,
                        assistant_metadata=assistant_meta or None,
                    )

            # 6. 异步生成摘要并裁剪历史（仅在超过阈值时）
            if body.session_id and needs_summary(body.session_id, threshold=10):
                # 异步生成摘要，不阻塞响应
                async def generate_summary_task():
                    try:
                        await generate_summary(body.session_id, keep_rounds=10)
                        logger.info(f"Summary generated for session {body.session_id}")
                    except Exception as e:
                        logger.error(f"Failed to generate summary: {e}")

                # 启动后台任务
                asyncio.create_task(generate_summary_task())

            # 7. 异步生成标题（首次对话时）
            if body.session_id and not get_title(body.session_id):
                async def generate_title_task():
                    try:
                        await generate_title(body.session_id)
                        logger.info(f"Title generated for session {body.session_id}")
                    except Exception as e:
                        logger.error(f"Failed to generate title: {e}")

                asyncio.create_task(generate_title_task())

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓存
        },
    )
