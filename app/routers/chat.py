from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.graph.graph import app_graph
from app.services.history_service import generate_summary, generate_title, get_history, get_title, needs_summary
from app.utils.logger import log_request, log_response, logger

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    file_ids: list[str] = []
    chat_history: list[dict] = []


# 导入子代理（用于 stream 函数）
from app.agents.chat_agent import chat_agent
from app.agents.rag_agent import rag_agent
from app.agents.mcp_agent import mcp_agent


@router.post("")
async def chat(body: ChatRequest) -> dict[str, Any]:
    """聊天接口，使用 LangGraph 执行主子代理路由"""

    log_request("POST", "/chat", session_id=body.session_id, file_ids=body.file_ids)

    # 获取历史对话（如果 session_id 存在）
    history = get_history(body.session_id) if body.session_id else []

    # 构建初始状态
    initial_state = {
        "message": body.message,
        "session_id": body.session_id or "default",
        "file_ids": body.file_ids or [],
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

        processing_files = []
        for fid in (body.file_ids or []):
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
                for fid in (body.file_ids or []):
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

        # 构建初始状态
        initial_state = {
            "message": body.message,
            "session_id": body.session_id or "default",
            "file_ids": body.file_ids or [],
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

            # 2. 执行 CoT 路由节点（流式推理）
            from app.chains.cot_chain import cot_chain
            from app.main import get_langfuse_client

            langfuse = get_langfuse_client()
            span = None
            if langfuse:
                span = langfuse.span(
                    name="cot_routing_stream",
                    input={"query": body.message, "has_file": bool(body.file_ids)},
                    session_id=initial_state["session_id"],
                )

            # 流式获取 CoT 推理（直接输出 JSON）
            cot_full = ""
            async for chunk in cot_chain.route_stream(body.message, has_file=bool(body.file_ids)):
                cot_full += chunk
                # 实时流式输出推理过程（JSON）
                yield f"data: {json.dumps({'type': 'reasoning', 'content': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)  # 确保数据发送

            # 解析路由目标
            routing_result = {"target": "chat", "reasoning": ""}
            try:
                result = json.loads(cot_full)
                if "target" in result:
                    routing_result = result
            except json.JSONDecodeError:
                import re
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

            if target_agent == "chat":
                async for chunk in chat_agent.handle_stream(body.message, initial_state):
                    answer_full += chunk
                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)  # 确保数据发送
            elif target_agent == "rag":
                async for chunk in rag_agent.handle_stream(body.message, initial_state):
                    answer_full += chunk
                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)  # 确保数据发送
            elif target_agent == "mcp":
                async for chunk in mcp_agent.handle_stream(body.message, initial_state):
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

            # 5. 异步生成摘要并裁剪历史（仅在超过阈值时）
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

            # 6. 异步生成标题（首次对话时）
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
