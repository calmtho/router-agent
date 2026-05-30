import asyncio
import uuid
from typing import Any

from langchain_community.vectorstores import Milvus

from app.config import config
from app.services.llm_client import get_llm_client
from app.utils.logger import logger


class MilvusService:
    """Milvus 向量库管理，使用 LangChain Milvus vectorstore"""

    def __init__(self):
        self.collection_name = config.milvus.collection_name
        self._vectorstore: Milvus | None = None

    @property
    def vectorstore(self) -> Milvus:
        if self._vectorstore is None:
            index_params = {
                "field_name": "embedding",
                "index_type": config.milvus.index_params["index_type"],
                "metric_type": config.milvus.index_params["metric_type"],
                "params": {"nlist": 128},
            }
            self._vectorstore = Milvus(
                embedding_function=get_llm_client().embed_model,
                collection_name=self.collection_name,
                connection_args={
                    "host": config.milvus.host,
                    "port": config.milvus.port,
                },
                index_params=index_params,
            )
        return self._vectorstore

    async def insert_documents(self, chunks: list[str], session_id: str, file_id: str = "") -> list[str]:
        metadatas = [{"session_id": session_id, "file_id": file_id} for _ in chunks]
        ids = [uuid.uuid4().hex for _ in chunks]
        await asyncio.to_thread(
            self.vectorstore.add_texts,
            texts=chunks,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Inserted {len(chunks)} chunks for session {session_id}")
        return ids

    async def search(
        self, query: str, session_id: str | None = None, top_k: int = 4, file_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        parts: list[str] = []
        if session_id:
            parts.append(f"session_id == '{session_id}'")
        if file_ids:
            ids = ", ".join(f"'{fid}'" for fid in file_ids)
            parts.append(f"file_id in [{ids}]")
        expr = " and ".join(parts) if parts else None

        docs_with_scores = await asyncio.to_thread(
            self.vectorstore.similarity_search_with_score,
            query,
            k=top_k,
            expr=expr,
        )

        return [
            {
                "text": doc.page_content,
                "session_id": doc.metadata.get("session_id"),
                "file_id": doc.metadata.get("file_id"),
                "score": score,
            }
            for doc, score in docs_with_scores
        ]

    def clear_session(self, session_id: str) -> None:
        self.vectorstore.delete(expr=f"session_id == '{session_id}'")
        logger.info(f"Cleared session {session_id} from Milvus")


milvus_service = MilvusService()
