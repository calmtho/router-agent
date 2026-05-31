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
from app.utils.logger import logger, setup_logger


# 全局 Langfuse 实例（通过 callback handler 自动追踪 LLM 调用）
_langfuse_client: Langfuse | None = None
_langfuse_callback_handler = None


def init_langfuse() -> None:
    """初始化 Langfuse 客户端和 callback handler"""
    global _langfuse_client, _langfuse_callback_handler

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


def get_langfuse_callback_handler():
    """
    获取 Langfuse callback handler（延迟初始化）。

    LangChain callback handler 需要在 LLM 模型初始化之后才能创建，
    因此这里延迟到第一次被调用时才构建。
    """
    global _langfuse_callback_handler

    if _langfuse_callback_handler is not None:
        return _langfuse_callback_handler

    if _langfuse_client is None:
        return None

    # 延迟导入，避免 langfuse 未安装时导入失败
    from langfuse.callback import CallbackHandler

    _langfuse_callback_handler = CallbackHandler(
        public_key=config.langfuse.public_key,
        secret_key=config.langfuse.secret_key,
        host=config.langfuse.host,
    )
    logger.info("Langfuse callback handler registered")

    return _langfuse_callback_handler


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

    # 初始化 LLM client，传入 Langfuse callback handler
    from app.services.llm_client import get_llm_client

    handler = get_langfuse_callback_handler()
    get_llm_client(handler=handler)

    # 预加载 FunASR STT 模型（可选，未安装 funasr 则跳过）
    from app.services.stt_service import get_stt_service
    stt = get_stt_service()
    stt.load_model("paraformer-zh")

    # 预加载 macbert4csc 纠错模型（可选，未安装 transformers 或模型下载失败则跳过）
    try:
        from app.services.typo_service import get_typo_corrector
        corrector = get_typo_corrector()
        corrector.load_model(config.preprocess.typo_model)
    except Exception as e:
        logger.warning(f"Typo correction model preload skipped: {e}")

    # 兜底：如果用 uvicorn CLI 启动（不走 __main__），
    # 此处重新启用被 uvicorn dictConfig 禁用的 logger
    setup_logger("app")
    logging.getLogger("app").disabled = False
    logger.info("Starting up...")
    os.makedirs("uploads", exist_ok=True)
    yield
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
