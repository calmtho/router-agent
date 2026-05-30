import os
import pytest
from pathlib import Path

from app.config import load_config


class TestConfig:
    """配置加载测试"""

    def test_load_config_basic(self, sample_config):
        """测试基础配置加载"""
        config = load_config(str(sample_config))

        assert config.llm.model_name == "gpt-4o-mini"
        assert config.llm.api_key == "test-key"
        assert config.milvus.port == 19530
        assert config.rag.top_k == 4

    def test_env_var_replacement(self, tmp_path, monkeypatch):
        """测试环境变量替换"""
        # 设置环境变量
        monkeypatch.setenv("OPENAI_API_KEY", "real-api-key")

        config_data = {
            "llm": {
                "openai_base_url": "https://api.openai.com/v1",
                "api_key": "${OPENAI_API_KEY}",
                "model_name": "gpt-4o-mini",
            },
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test",
                "embedding_model": "test",
                "embedding_dim": 1536,
                "index_params": {"metric_type": "IP", "index_type": "IVF_FLAT"},
            },
            "mcp": {"servers": []},
            "rag": {"chunk_size": 500, "chunk_overlap": 50, "top_k": 4},
            "main_agent": {"cot_prompt_template": "test", "fallback_agent": "chat"},
            "server": {"host": "0.0.0.0", "port": 8000, "max_file_size_mb": 10},
        }

        import yaml

        config_file = tmp_path / "test_env.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        assert config.llm.api_key == "real-api-key"

    def test_config_validation_defaults(self, sample_config):
        """测试配置默认值"""
        config = load_config(str(sample_config))

        assert config.llm.temperature == 0.7
        assert config.llm.max_tokens == 1024
        assert config.rag.chunk_size == 500
        assert config.rag.chunk_overlap == 50
