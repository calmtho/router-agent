import pytest
import yaml
from pathlib import Path
from typing import Any


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """创建测试用配置文件"""
    config_data = {
        "llm": {
            "openai_base_url": "https://api.openai.com/v1",
            "api_key": "test-key",
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_docs",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "index_params": {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
            },
        },
        "mcp": {
            "servers": [
                {
                    "name": "calculator",
                    "command": "python",
                    "args": ["-m", "mcp_server_calc"],
                    "env": {},
                }
            ]
        },
        "rag": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "top_k": 4,
        },
        "main_agent": {
            "cot_prompt_template": "Test template: {query}",
            "fallback_agent": "chat",
        },
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "max_file_size_mb": 10,
        },
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    return config_file


@pytest.fixture
def sample_text_chunks():
    """创建测试用文本块"""
    return [
        "这是第一段文本内容。",
        "这是第二段文本内容。",
        "这是第三段文本内容。",
    ]


@pytest.fixture
def mock_query():
    """测试查询"""
    return "测试问题"
