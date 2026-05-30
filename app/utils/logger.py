import logging
import sys
from typing import Any

from app.config import config


def setup_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)

    # 清除已有的 handler（uvicorn reload 或 dictConfig 可能干扰）
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止 root logger 干扰
    logger.disabled = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


logger = setup_logger()


def log_request(method: str, path: str, **kwargs: Any) -> None:
    logger.info(f"Request: {method} {path} - {kwargs}")


def log_response(status: str, data: Any) -> None:
    logger.info(f"Response: {status} - {data}")


def log_error(error: Exception, context: str = "") -> None:
    logger.error(f"Error: {context} - {str(error)}", exc_info=error)
