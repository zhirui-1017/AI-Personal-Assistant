"""向量库抽象：默认 SQLite（零部署），可选 Chroma。

两个后端实现同一套接口，业务层不感知差异；切换只需改 ``VECTOR_BACKEND``。
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sqlalchemy import select, update

from app.config import Settings, get_settings
from app.db.base import session_scope
from app.db.models import Chunk
from app.rag.embeddings import bytes_to_vector, vector_to_bytes

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorHit:
    chunk_id: str
    score: float
    document_id: str | None = None


class VectorIndex(ABC):
    """向量索引接口。"""

    backend: str = "base"

    @abstractmethod
    def upsert(self, ids: Sequence[str], vectors: np.ndarray, document_ids: Sequence[str]) -> None:
        ...

    @abstractmethod
    def delete(self, *, ids: Sequence[str] | None = None, document_id: str | None = None) -> None:
        ...

    @abstractmethod
    def query(
        self, vector: np.ndarray, top_k: int = 10, document_ids: Sequence[str] | None = None
    ) -> list[VectorHit]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    def invalidate(self) -> None:  # pragma: no cover - 默认无缓存
        return None

    def health(self) -> dict[str, object]:
        return {"backend": self.backend, "vectors": self.count()}


class SQLiteVectorIndex(VectorIndex):
    """把向量直接存在 chunks 表里，检索时用 numpy 暴力内积。

    千级片段下矩阵乘法只需毫秒级，完全满足「千级片段不卡顿」的要求，
    而且与文档、片段数据同库，天然持久化、无需额外服务。
    """

    backend = "sqlite"

    def __init__(self, settings: Settings | None = None, cache_vectors: bool = True) -> None:
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []
        self._document_ids: list[str] = []
        self._cache_vectors = cache_vectors

    # ------------------------------------------------------------------ #
    def invalidate(self) -> None:
        with self._lock:
            self._matrix = None
            self._ids = []
            self._document_ids = []

    def upsert(self, ids: Sequence[str], vectors: np.ndarray, document_ids: Sequence[str]) -> None:
        if len(ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        model_name = self.settings.embedding_model
        with session_scope() as db:
            for row, chunk_id in enumerate(ids):
                db.execute(
                    update(Chunk)
                    .where(Chunk.id == chunk_id)
                    .values(
                        embedding=vector_to_bytes(vectors[row]),
                        embedding_dim=int(vectors.shape[1]),
                        embedding_model=model_name,
                    )
                )
        self.invalidate()

    def delete(self, *, ids: Sequence[str] | None = None, document_id: str | None = None) -> None:
        with session_scope() as db:
            stmt = update(Chunk).values(embedding=None, embedding_dim=None)
            if ids:
                db.execute(stmt.where(Chunk.id.in_(list(ids))))
            elif document_id:
                db.execute(stmt.where(Chunk.document_id == document_id))
        self.invalidate()

    def query(
        self, vector: np.ndarray, top_k: int = 10, document_ids: Sequence[str] | None = None
    ) -> list[VectorHit]:
        matrix, ids, doc_ids = self._ensure_loaded()
        if matrix is None or not ids:
            return []

        query_vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query_vector.shape[0] != matrix.shape[1]:
            logger.warning(
                "查询向量维度 %s 与索引维度 %s 不一致，已跳过向量检索（通常是更换了 Embedding 模型，需重新解析文档）",
                query_vector.shape[0],
                matrix.shape[1],
            )
            return []
        norm = float(np.linalg.norm(query_vector)) or 1.0
        scores = matrix @ (query_vector / norm)

        if document_ids:
            allowed = set(document_ids)
            mask = np.array([doc in allowed for doc in doc_ids], dtype=bool)
            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                return []
        else:
            candidates = np.arange(scores.shape[0])

        k = min(top_k, candidates.size)
        if k <= 0:
            return []
        local_scores = scores[candidates]
        top_local = np.argpartition(-local_scores, k - 1)[:k]
        top_local = top_local[np.argsort(-local_scores[top_local])]
        top_indexes = candidates[top_local]
        return [
            VectorHit(chunk_id=ids[index], score=float(scores[index]), document_id=doc_ids[index])
            for index in top_indexes
        ]

    def count(self) -> int:
        with session_scope() as db:
            rows = db.execute(
                select(Chunk.id).where(Chunk.embedding.is_not(None)).where(Chunk.embedding_dim.is_not(None))
            ).all()
        return len(rows)

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> tuple[np.ndarray | None, list[str], list[str]]:
        if not self._cache_vectors:
            return self._load()
        with self._lock:
            if self._matrix is None:
                matrix, ids, doc_ids = self._load()
                self._matrix, self._ids, self._document_ids = matrix, ids, doc_ids
            return self._matrix, self._ids, self._document_ids

    def _load(self) -> tuple[np.ndarray | None, list[str], list[str]]:
        with session_scope() as db:
            rows = db.execute(
                select(Chunk.id, Chunk.document_id, Chunk.embedding, Chunk.embedding_dim)
                .where(Chunk.embedding.is_not(None))
                .order_by(Chunk.created_at)
            ).all()

        ids: list[str] = []
        doc_ids: list[str] = []
        vectors: list[np.ndarray] = []
        dimension: int | None = None
        for chunk_id, document_id, blob, dim in rows:
            vector = bytes_to_vector(blob, dim)
            if vector is None or vector.size == 0:
                continue
            if dimension is None:
                dimension = vector.size
            if vector.size != dimension:
                continue
            ids.append(chunk_id)
            doc_ids.append(document_id)
            vectors.append(vector)
        if not vectors:
            return None, [], []
        return np.vstack(vectors).astype(np.float32), ids, doc_ids


class ChromaVectorIndex(VectorIndex):
    """Chroma 后端（可选）：适合片段数上万、需要 HNSW 加速的场景。"""

    backend = "chroma"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("未安装 chromadb，请执行 pip install chromadb 或改用 VECTOR_BACKEND=sqlite") from exc
        self.settings.ensure_dirs()
        self._client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.settings.chroma_collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids: Sequence[str], vectors: np.ndarray, document_ids: Sequence[str]) -> None:
        if len(ids) == 0:
            return
        self._collection.upsert(
            ids=list(ids),
            embeddings=[np.asarray(v, dtype=np.float32).tolist() for v in vectors],
            metadatas=[{"document_id": doc} for doc in document_ids],
        )

    def delete(self, *, ids: Sequence[str] | None = None, document_id: str | None = None) -> None:
        if ids:
            self._collection.delete(ids=list(ids))
        elif document_id:
            self._collection.delete(where={"document_id": document_id})

    def query(
        self, vector: np.ndarray, top_k: int = 10, document_ids: Sequence[str] | None = None
    ) -> list[VectorHit]:
        if self._collection.count() == 0:
            return []
        where = {"document_id": {"$in": list(document_ids)}} if document_ids else None
        result = self._collection.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            n_results=max(1, top_k),
            where=where,
            include=["metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        hits = []
        for index, chunk_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            metadata = metadatas[index] if index < len(metadatas) else {}
            hits.append(
                VectorHit(chunk_id=chunk_id, score=1.0 - distance, document_id=(metadata or {}).get("document_id"))
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())


def build_vector_index(settings: Settings | None = None) -> VectorIndex:
    settings = settings or get_settings()
    backend = settings.vector_backend
    if backend == "chroma":
        try:
            index = ChromaVectorIndex(settings)
            logger.info("向量库后端：Chroma（%s）", settings.chroma_dir)
            return index
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma 初始化失败（%s），自动退回 SQLite 向量库", exc)
    elif backend == "faiss":
        logger.warning("FAISS 后端尚未实现，本次使用 SQLite 向量库")
    return SQLiteVectorIndex(settings)
