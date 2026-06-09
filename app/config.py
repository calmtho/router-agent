import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    openai_base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024


class RouterLLMConfig(BaseModel):
    """路由专用 LLM 配置（小模型，仅用于 reranker 置信度不足时的 LLM 兜底）"""
    openai_base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 256


class RouterConfig(BaseModel):
    """路由配置（Embedding 初筛 → Reranker 精排 → Margin 置信度）"""
    confidence_threshold: float = 0.6
    margin_temperature: float = 2.0
    embedding_top_k: int = 2
    category_descriptions: dict[str, str] = {}


class MilvusConfig(BaseModel):
    host: str
    port: int
    collection_name: str
    embedding_model: str
    embedding_dim: int
    index_params: dict[str, str]


class MCPServer(BaseModel):
    name: str
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None


class MCPConfig(BaseModel):
    servers: list[MCPServer]


class RAGConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 4
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L12-v2"
    rerank_batch_size: int = 4
    rerank_output_k: int = 4


class PaddleOCRConfig(BaseModel):
    endpoint: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    token: str = ""
    model: str = "PaddleOCR-VL-1.6"


class VisionConfig(BaseModel):
    openai_base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 2048
    phases: str = "simple"  # "simple" | "full"


class MainAgentConfig(BaseModel):
    cot_prompt_template: str
    fallback_agent: str = "chat"


class PreprocessConfig(BaseModel):
    enable_typo_correction: bool = True
    typo_model: str = "shibing624/macbert4csc-base-chinese"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    max_file_size_mb: int = 10


class LangfuseConfig(BaseModel):
    enabled: bool = False
    host: str = "https://cloud.langfuse.com"
    public_key: str = ""
    secret_key: str = ""


class Config(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_prefix="APP_",
        extra="ignore",
    )

    llm: LLMConfig
    router_llm: RouterLLMConfig
    router: RouterConfig = Field(default_factory=RouterConfig)
    milvus: MilvusConfig
    mcp: MCPConfig
    rag: RAGConfig
    vision: VisionConfig
    paddle_ocr: PaddleOCRConfig = Field(default_factory=PaddleOCRConfig)
    main_agent: MainAgentConfig
    server: ServerConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)


def load_config(config_path: str | None = None) -> Config:
    root_dir = Path(__file__).parent.parent
    config_path = config_path or (root_dir / "configs" / "config.yaml")

    # 先加载 .env 文件，确保环境变量在替换前就绪
    env_file = root_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 环境变量替换
    def replace_env_vars(obj: Any) -> Any:
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                var_name = obj[2:-1]
                return os.getenv(var_name, obj)
            return obj
        elif isinstance(obj, dict):
            return {k: replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_env_vars(item) for item in obj]
        return obj

    data = replace_env_vars(data)
    return Config(**data)


config = load_config()
