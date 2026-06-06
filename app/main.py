import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langfuse import Langfuse

from app.config import config
from app.routers.chat import router as chat_router
from app.routers.stt import router as stt_router
from app.routers.upload import router as upload_router
from app.routers.upload_image import router as upload_image_router
from app.utils.logger import logger, setup_logger


# 全局 Langfuse 实例（通过 callback handler 自动追踪 LLM 调用）
_langfuse_client: Langfuse | None = None


def init_langfuse() -> None:
    """初始化 Langfuse 客户端和 callback handler"""
    global _langfuse_client

    if not config.langfuse.enabled:
        logger.info("Langfuse is disabled, skipping initialization")
        return

    if not config.langfuse.secret_key or not config.langfuse.public_key:
        logger.warning("Langfuse credentials not configured, skipping initialization")
        return

    _langfuse_client = Langfuse(
        host=config.langfuse.host,
        public_key=config.langfuse.public_key,
        secret_key=config.langfuse.secret_key,
    )
    logger.info(f"Langfuse initialized successfully (host: {config.langfuse.host})")


def get_langfuse_client() -> Langfuse | None:
    """获取全局 Langfuse 客户端实例"""
    return _langfuse_client


# 自定义 uvicorn 日志配置：关键点 disable_existing_loggers=False，
# 防止 uvicorn 启动时把我们自定义的 "app" logger 禁用掉
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "access": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
        },
        "access": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "access",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 初始化 Langfuse（如果启用）
    init_langfuse()

    # 初始化 LLM client（callback handler 在 langfuse 4.x + langchain 1.x 已废弃）
    from app.services.llm_client import get_llm_client

    get_llm_client()

    # ── 后台异步加载模型（不阻塞启动） ──
    async def _load_models():
        # STT
        try:
            from app.services.stt_service import get_stt_service

            stt = get_stt_service()
            logger.info("[STT] 开始加载模型...")
            await asyncio.to_thread(stt.load_model, "paraformer-zh")
            logger.info("[STT] 模型加载完成")
        except Exception as e:
            logger.warning(f"[STT] 模型加载失败: {e}")

        # 纠错
        try:
            from app.services.typo_service import get_typo_corrector

            corrector = get_typo_corrector()
            logger.info("[Typo] 开始加载模型...")
            await asyncio.to_thread(corrector.load_model, config.preprocess.typo_model)
            logger.info("[Typo] 模型加载完成")
        except Exception as e:
            logger.warning(f"[Typo] 模型加载失败: {e}")

        # Reranker（首次 RAG 请求不用再等 11s）
        try:
            from app.services.reranker_service import get_reranker_service

            reranker = get_reranker_service()
            logger.info("[Reranker] 开始预加载模型...")
            await asyncio.to_thread(reranker.load_model)
            logger.info("[Reranker] 模型预加载完成")
        except Exception as e:
            logger.warning(f"[Reranker] 模型预加载失败: {e}")

    background_task = asyncio.create_task(_load_models())

    # 兜底：如果用 uvicorn CLI 启动（不走 __main__），
    # 此处重新启用被 uvicorn dictConfig 禁用的 logger
    setup_logger("app")
    logging.getLogger("app").disabled = False
    logger.info("Starting up... (models loading in background)")
    os.makedirs("uploads", exist_ok=True)
    yield
    background_task.cancel()
    logger.info("Shutting down...")

    # 关闭 Langfuse 客户端（刷新缓冲区，确保数据发送）
    if _langfuse_client is not None:
        _langfuse_client.flush()
        logger.info("Langfuse client flushed and shut down")


app = FastAPI(
    title="Router Agent API",
    description="基于 CoT 的 Main-Sub Agent 架构智能代理系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件（支持前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境可指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)
app.include_router(stt_router)
app.include_router(upload_router)
app.include_router(upload_image_router)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    status = {"status": "ok", "service": "router-agent"}
    if _langfuse_client is not None:
        status["langfuse"] = "connected"
    else:
        status["langfuse"] = "disabled"
    return status


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Router Agent API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat (POST)",
            "upload": "/upload (POST)",
            "stt": "/stt/transcribe (POST)",
            "health": "/health (GET)",
            "static": "/static (静态文件)",
        },
    }


# 挂载静态文件目录（开发测试页面等）
# 注意：mount 必须放在所有路由定义之后，否则会截获该路径前缀下的所有请求
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=True,
        log_config=UVICORN_LOG_CONFIG,
    )
