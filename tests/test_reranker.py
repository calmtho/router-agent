"""
test_reranker.py — 测试重排序功能
"""

import asyncio
import sys
from pathlib import Path
from warnings import filterwarnings

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 忽略某些警告
filterwarnings("ignore")

from app.services.reranker_service import get_reranker_service
from app.utils.logger import logger


async def test_reranker():
    """测试重排序服务"""

    print("=" * 60)
    print("测试重排序功能")
    print("=" * 60)

    # 创建重排序服务实例
    reranker = get_reranker_service()

    # 测试 1: 模型加载
    print("\n[测试 1] 模型加载测试")
    print("-" * 60)

    loaded = reranker.load_model()

    if loaded is False:
        print("[FAIL] 模型加载失败")
        return

    if loaded is None:
        print("[WARN]  重排序功能被禁用")
        return

    print("[OK] 模型加载成功")

    if not reranker.is_ready:
        print("[FAIL] 模型未就绪")
        return

    print("[OK] 模型已就绪")

    # 测试 2: 简单重排序测试
    print("\n[测试 2] 重排序功能测试")
    print("-" * 60)

    query = "什么是向量数据库？"

    # 创建一些模拟文档（相关性和不相关）
    documents = [
        {"text": "向量数据库是一种专门存储和检索向量数据的数据库", "score": 0.9},
        {"text": "今天天气真好，适合出去玩", "score": 0.3},
        {"text": "向量数据库支持高维向量的存储和相似度搜索", "score": 0.85},
        {"text": "Python 是一门流行编程语言", "score": 0.1},
        {"text": "向量数据库使用 IP 或 Cosine 相似度进行检索", "score": 0.88},
    ]

    print(f"Query: {query}")
    print(f"原始文档数量: {len(documents)}")
    print("\n原始文档:")
    for i, doc in enumerate(documents):
        print(f"  {i+1}. score={doc['score']:.3f}: {doc['text'][:40]}...")

    # 执行重排序
    import time
    start = time.time()
    reranked_docs, rerank_used = await reranker.rerank(query, documents)
    elapsed = time.time() - start

    print(f"\n重排序耗时: {elapsed:.3f}s")
    print(f"重排序是否启用: {rerank_used}")
    print("\n重排序后文档:")
    for i, doc in enumerate(reranked_docs):
        rerank_score = doc.get("rerank_score", "N/A")
        print(f"  {i+1}. score={doc['score']:.3f} -> rerank={rerank_score:.3f}: {doc['text'][:40]}...")

    # 测试 3: 大批量测试（16个文档）
    print("\n[测试 3] 批量重排序测试（16个文档）")
    print("-" * 60)

    large_documents = [
        {"text": f"这是第 {i} 个测试文档", "score": 0.5 - i * 0.02}
        for i in range(16)
    ]

    print(f"文档数量: {len(large_documents)}")

    start = time.time()
    large_reranked, large_rerank_used = await reranker.rerank(query, large_documents)
    elapsed = time.time() - start

    print(f"重排序耗时: {elapsed:.3f}s")
    print(f"输出口径: {len(large_reranked)} 个文档")
    print(f"批处理次数: {len(large_documents)} / 4 = 4 次")

    print("\n重排序后前 4 个文档:")
    for i, doc in enumerate(large_reranked):
        rerank_score = doc.get("rerank_score", "N/A")
        print(f"  {i+1}. rerank={rerank_score:.3f}: {doc['text'][:50]}...")

    # 测试 4: 配置验证
    print("\n[测试 4] 配置验证")
    print("-" * 60)

    from app.config import config

    print(f"top_k (检索数量): {config.rag.top_k}")
    print(f"rerank_enabled: {config.rag.rerank_enabled}")
    print(f"rerank_model: {config.rag.rerank_model}")
    print(f"rerank_batch_size: {config.rag.rerank_batch_size}")
    print(f"rerank_output_k: {config.rag.rerank_output_k}")

    # 测试完成
    print("\n" + "=" * 60)
    print("[OK] All tests completed")
    print("=" * 60)

    print("\n[INFO] Next steps:")
    print("1. Check log file for detailed execution")
    print("2. Test complete RAG pipeline (with reranking)")
    print("3. Set rerank_enabled=false in config.yaml to disable")


if __name__ == "__main__":
    asyncio.run(test_reranker())
