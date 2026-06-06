"""
reranker_service.py — 轻量级重排序服务（使用 cross-encoder/ms-marco-MiniLM-L12-v2）
"""

import asyncio
import threading
from typing import Any

from app.config import config
from app.utils.logger import logger


class RerankerService:
    """轻量级重排序服务，基于 cross-encoder 模型对检索结果进行二次排序"""

    def __init__(self):
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        """检查模型是否已加载"""
        return self._model is not None

    @property
    def is_enabled(self) -> bool:
        """检查重排序功能是否启用"""
        return config.rag.rerank_enabled

    def load_model(self) -> bool | None:
        """
        加载重排序模型（应在应用启动时调用一次）

        Returns:
            True  — 加载成功
            None  — 重排序已禁用，或加载失败（已降级）
        """
        with self._load_lock:
            if self.is_ready:
                logger.info("[Reranker] 模型已加载")
                return True

            if not config.rag.rerank_enabled:
                logger.info("[Reranker] 重排序已禁用（配置中 rerank_enabled=false）")
                return None

            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"[Reranker] 正在加载模型: {config.rag.rerank_model} ...")

                self._model = CrossEncoder(
                    config.rag.rerank_model,
                    device="cpu",
                )

                # 简单模型测试（cross-encoder 期望 [query, doc] 二元组）
                import time
                test_pairs = [["这是一个测试", "测试"]]
                start = time.time()
                self._model.predict(test_pairs)
                elapsed = time.time() - start
                logger.info(f"[Reranker] 模型加载完成，测试延迟: {elapsed:.3f}s")

                return True

            except ImportError:
                logger.error(
                    "[Reranker] sentence-transformers 未安装，"
                    "请运行: pip install sentence-transformers"
                )
                return None

            except Exception as e:
                logger.error(f"[Reranker] 模型加载失败: {e}")
                self._model = None
                return None

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        对检索到的文档进行重排序

        Args:
            query: 用户问题
            documents: 检索到的文档列表，每个文档包含 {text, score, ...}

        Returns:
            (重排序后的文档列表, 是否实际使用了重排序)
        """
        # 未启用重排序，直接返回原始结果
        if not config.rag.rerank_enabled:
            return documents, False

        # 模型未就绪，尝试懒加载
        if not self.is_ready:
            loaded = self.load_model()
            if not self.is_ready:
                logger.warning("[Reranker] 模型不可用，返回原始检索结果 (reason=%s)",
                               "禁用" if loaded is None else "加载失败")
                return documents, False

        if not documents:
            return documents, False

        try:
            import time
            start = time.time()

            # 提取文档文本列表
            texts = [doc["text"] for doc in documents]

            # 准备查询-文档对
            pairs = [[query, text] for text in texts]

            # 在线程池中执行同步的模型推理，避免阻塞事件循环
            batch_size = config.rag.rerank_batch_size
            rerank_output_k = config.rag.rerank_output_k

            def _predict() -> list[float]:
                all_scores: list[float] = []
                for i in range(0, len(pairs), batch_size):
                    batch = pairs[i : i + batch_size]
                    scores = self._model.predict(batch)
                    all_scores.extend(scores)
                return all_scores

            all_scores = await asyncio.to_thread(_predict)

            # 计算重排序分数
            reranked_results = []
            for doc, score in zip(documents, all_scores):
                reranked_results.append(
                    {
                        **doc,
                        "rerank_score": float(score),
                    }
                )

            # 按 rerank_score 排序（降序）
            reranked_results.sort(
                key=lambda x: x["rerank_score"],
                reverse=True
            )

            # 返回重排序后的前 N 个结果
            results = reranked_results[:rerank_output_k]

            elapsed = time.time() - start
            logger.info(
                f"[Reranker] 重排序完成: {len(documents)}→{len(results)} 个结果, "
                f"耗时 {elapsed:.3f}s"
            )

            return results, True

        except Exception as e:
            import traceback
            logger.error(f"[Reranker] 重排序失败: {e}\n{traceback.format_exc()}，返回原始检索结果")
            if "timeout" in str(e).lower():
                logger.warning("[Reranker] 可能是超时，考虑减小 batch_size")
            return documents, False


    async def classify_route(
        self, query: str
    ) -> tuple[str, float]:
        """
        两阶段路由分类:
          Stage 1: Embedding 模型 → cosine 相似度初筛，取 Top-K
          Stage 2: Reranker 模型 → 精准打分 Top-K，margin 分析

        Args:
            query: 用户输入

        Returns:
            (target_agent, confidence)  e.g. ("mcp", 0.85) 或 ("fallback", 0.35)

        Raises:
            RuntimeError: 模型未就绪
        """
        import numpy as np
        from app.services.llm_client import get_llm_client

        if not self.is_ready:
            loaded = self.load_model()
            if not self.is_ready:
                raise RuntimeError("[Reranker] 模型未就绪，无法执行路由分类")

        categories = list(config.router.category_descriptions.keys())
        descriptions = list(config.router.category_descriptions.values())

        if not categories:
            raise RuntimeError("[Reranker] 未配置 category_descriptions，无法路由")

        try:
            # ═══════════════════════════════════════════
            # Stage 1: Embedding 模型初筛
            # ═══════════════════════════════════════════
            llm = get_llm_client()

            query_emb = await asyncio.to_thread(llm.embed_query, query)
            desc_embs = await asyncio.to_thread(llm.embed_documents, descriptions)

            query_vec = np.array(query_emb)
            desc_vecs = np.array(desc_embs)

            # 已 normalize_embeddings=True，dot product 即 cosine similarity
            similarities = np.dot(desc_vecs, query_vec)
            sorted_indices = np.argsort(similarities)[::-1]

            # 打印 Embedding 阶段所有分数
            for idx in sorted_indices:
                logger.info(
                    f"[Router] embedding: {categories[idx]:<8} = {similarities[idx]:.4f}"
                )

            # 取 Top-K 候选
            top_k = config.router.embedding_top_k
            candidate_indices = sorted_indices[:top_k]
            candidates = [categories[i] for i in candidate_indices]

            gap_first_two = (
                similarities[candidate_indices[0]] - similarities[candidate_indices[1]]
                if len(candidate_indices) > 1
                else 0.0
            )
            logger.info(
                f"[Router] Embedding Top-{top_k}: {candidates}, "
                f"gap(top1-top2)={gap_first_two:.4f}"
            )

            # ═══════════════════════════════════════════
            # Stage 2: Reranker 对 Top-K 精排
            # ═══════════════════════════════════════════
            pairs = [[query, descriptions[i]] for i in candidate_indices]
            scores = await asyncio.to_thread(self._model.predict, pairs)
            scores = np.array(scores, dtype=np.float64)

            # 在候选内排序
            local_sorted = np.argsort(scores)[::-1]
            best_local_idx = candidate_indices[local_sorted[0]]
            second_local_idx = (
                candidate_indices[local_sorted[1]]
                if len(candidate_indices) > 1
                else best_local_idx
            )

            best_score = float(scores[local_sorted[0]])
            second_score = (
                float(scores[local_sorted[1]])
                if len(candidate_indices) > 1
                else best_score
            )
            margin = best_score - second_score

            # Margin → Confidence (sigmoid 映射，margin 越大越接近 1.0)
            temperature = config.router.margin_temperature
            confidence = 1.0 / (1.0 + np.exp(-margin * temperature))

            logger.info(
                f"[Router] reranker: best={categories[best_local_idx]}({best_score:.4f}), "
                f"2nd={categories[second_local_idx]}({second_score:.4f}), "
                f"margin={margin:.4f} → confidence={confidence:.4f}"
            )

            threshold = config.router.confidence_threshold
            if confidence >= threshold:
                return categories[best_local_idx], confidence
            else:
                logger.info(
                    f"[Router] 置信度不足 ({confidence:.3f} < {threshold})，"
                    f"回退 CoT"
                )
                return "fallback", confidence

        except Exception:
            logger.warning("[Router] 路由分类失败，回退 CoT")
            raise


# 全局单例
_reranker_service = RerankerService()


def get_reranker_service() -> RerankerService:
    """获取重排序服务实例"""
    return _reranker_service
