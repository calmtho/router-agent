import asyncio
import uuid
from typing import Any

from pymilvus import DataType, MilvusClient

from app.config import config
from app.services.llm_client import get_llm_client
from app.utils.logger import logger


class MilvusService:
    """Milvus 向量库管理，直接使用 pymilvus"""

    def __init__(self):
        self.collection_name = config.milvus.collection_name
        self._client: MilvusClient | None = None
        self._collection_ready = False

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(
                uri=f"http://{config.milvus.host}:{config.milvus.port}"
            )
        return self._client

    def _ensure_collection(self) -> None:
        """确保集合存在，不存在则创建；若 schema 不匹配则自动重建"""
        if self._collection_ready:
            return

        if self.client.has_collection(self.collection_name):
            # 检查 schema 是否匹配：旧集合可能缺少 id VARCHAR 等字段
            existing_schema = self.client.describe_collection(self.collection_name)
            field_names = {f["name"] for f in existing_schema["fields"]}
            required_fields = {"id", "text", "embedding", "session_id", "file_id"}
            if not required_fields.issubset(field_names):
                logger.warning(
                    f"Collection {self.collection_name} schema mismatch "
                    f"(missing: {required_fields - field_names}), dropping and recreating..."
                )
                self.client.drop_collection(self.collection_name)
            else:
                self._collection_ready = True
                return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=config.milvus.embedding_dim)
        schema.add_field("session_id", DataType.VARCHAR, max_length=256)
        schema.add_field("file_id", DataType.VARCHAR, max_length=256)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=config.milvus.index_params["index_type"],
            metric_type=config.milvus.index_params["metric_type"],
            params={"nlist": 128},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self._collection_ready = True
        logger.info(f"Created Milvus collection: {self.collection_name}")

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """调用 embedding 模型获取向量（同步，用于 to_thread 包装）"""
        return get_llm_client().embed_documents(texts)

    def _embed_query(self, text: str) -> list[float]:
        """调用 embedding 模型获取查询向量（同步）"""
        return get_llm_client().embed_query(text)

    async def insert_documents(self, chunks: list[str], session_id: str, file_id: str = "") -> list[str]:
        self._ensure_collection()

        # 1. 过滤空白 chunk，避免 embedding 模型对空文本返回异常结果
        valid_chunks = [c for c in chunks if c.strip()]
        if not valid_chunks:
            logger.warning(f"All chunks are empty for session {session_id}, skipping insert")
            return []

        # 2. 分批 embedding
        batch_size = 4
        all_embeddings = []
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i : i + batch_size]
            batch_emb = await asyncio.to_thread(self._embed_texts, batch)
            all_embeddings.extend(batch_emb)

        # 3. 逐条校验维度（SentenceTransformer 保证返回纯 Python list）
        expected_dim = config.milvus.embedding_dim
        safe_chunks = []
        safe_embeddings = []
        for idx, (chunk, emb) in enumerate(zip(valid_chunks, all_embeddings)):
            if len(emb) != expected_dim:
                logger.warning(f"[Milvus] chunk[{idx}] dim={len(emb)} ≠ {expected_dim}, skip")
                continue
            safe_chunks.append(chunk)
            safe_embeddings.append(emb)

        if not safe_embeddings:
            logger.error(f"[Milvus] no valid embeddings for session {session_id}")
            return []

        # 4. 构建并插入
        ids = [uuid.uuid4().hex for _ in safe_chunks]
        data = [
            {
                "id": ids[i],
                "text": safe_chunks[i],
                "embedding": safe_embeddings[i],
                "session_id": session_id,
                "file_id": file_id,
            }
            for i in range(len(safe_chunks))
        ]

        await asyncio.to_thread(
            self.client.insert,
            collection_name=self.collection_name,
            data=data,
        )
        logger.info(f"Inserted {len(safe_chunks)} chunks for session {session_id}")
        return ids

    async def search(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 4,
        file_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_collection()

        # 构建过滤表达式
        parts: list[str] = []
        if session_id:
            parts.append(f'session_id == "{session_id}"')
        if file_ids:
            ids = ", ".join(f'"{fid}"' for fid in file_ids)
            parts.append(f"file_id in [{ids}]")
        filter_expr = " and ".join(parts) if parts else ""

        # 获取查询向量
        query_embedding = await asyncio.to_thread(self._embed_query, query)

        results = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            filter=filter_expr,
            output_fields=["text", "session_id", "file_id"],
        )

        if not results or not results[0]:
            return []

        return [
            {
                "text": hit["entity"]["text"],
                "session_id": hit["entity"].get("session_id"),
                "file_id": hit["entity"].get("file_id"),
                "score": hit["distance"],
            }
            for hit in results[0]
        ]

    def clear_session(self, session_id: str) -> None:
        self._ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            filter=f'session_id == "{session_id}"',
        )
        logger.info(f"Cleared session {session_id} from Milvus")


milvus_service = MilvusService()
