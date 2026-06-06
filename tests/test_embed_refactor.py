"""验证 SentenceTransformer 替换 HuggingFaceEmbeddings 后类型一致性"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from app.services.llm_client import LLMClient, get_llm_client


def test_embed_model_loads():
    """1. SentenceTransformer 能正常从本地加载模型"""
    client = LLMClient()
    assert client.embed_model is not None, "embed_model 应为 SentenceTransformer 实例"
    print("[PASS] SentenceTransformer 加载成功")


def test_embed_documents_returns_pure_list():
    """2. embed_documents() 返回纯 Python list[list[float]]，不含 numpy"""
    client = LLMClient()
    texts = ["你好世界", "向量检索测试"]
    result = client.embed_documents(texts)

    assert isinstance(result, list), f"外层应为 list，实际 {type(result)}"
    assert len(result) == 2, f"应有 2 条向量，实际 {len(result)}"
    for i, emb in enumerate(result):
        assert isinstance(emb, list), f"result[{i}] 应为 list，实际 {type(emb)}"
        assert not isinstance(emb, np.ndarray), f"result[{i}] 不应是 numpy array"
        for j, val in enumerate(emb):
            assert isinstance(val, float), f"result[{i}][{j}] 应为 float，实际 {type(val)}"
    print(f"[PASS] embed_documents 返回纯 list[list[float]]，每条维度={len(result[0])}")


def test_embed_query_returns_pure_list():
    """3. embed_query() 返回纯 Python list[float]"""
    client = LLMClient()
    result = client.embed_query("测试查询")

    assert isinstance(result, list), f"应为 list，实际 {type(result)}"
    assert not isinstance(result, np.ndarray), "不应是 numpy array"
    for val in result:
        assert isinstance(val, float), f"元素应为 float，实际 {type(val)}"
    print(f"[PASS] embed_query 返回纯 list[float]，维度={len(result)}")


def test_dimension_matches_config():
    """4. 向量维度与 config.yaml 中配置一致"""
    from app.config import config
    client = LLMClient()

    doc_embs = client.embed_documents(["测试"])
    query_emb = client.embed_query("测试")

    expected_dim = config.milvus.embedding_dim
    assert len(doc_embs[0]) == expected_dim, f"doc dim={len(doc_embs[0])} ≠ {expected_dim}"
    assert len(query_emb) == expected_dim, f"query dim={len(query_emb)} ≠ {expected_dim}"
    print(f"[PASS] 向量维度={expected_dim}，与配置一致")


def test_embed_documents_batch_consistency():
    """5. 多次调用返回类型始终一致"""
    client = LLMClient()
    for batch_size in [1, 3, 5]:
        texts = [f"test chunk {i}" for i in range(batch_size)]
        result = client.embed_documents(texts)
        assert isinstance(result, list)
        assert len(result) == batch_size
        for emb in result:
            assert isinstance(emb, list)
            assert not isinstance(emb, np.ndarray)
    print("[PASS] 多批次 embed_documents 类型始终一致")


if __name__ == "__main__":
    test_embed_model_loads()
    test_embed_documents_returns_pure_list()
    test_embed_query_returns_pure_list()
    test_dimension_matches_config()
    test_embed_documents_batch_consistency()
    print("\n✅ 全部测试通过！SentenceTransformer 重构成功。")
