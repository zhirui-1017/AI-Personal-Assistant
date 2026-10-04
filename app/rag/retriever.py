"""混合检索：BM25 关键词召回 + 向量语义召回 → 归一化融合 → MMR 重排。"""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db.base import session_scope
from app.db.models import Chunk, Document
from app.rag.embeddings import BaseEmbedder, bytes_to_vector
from app.rag.rerank import BaseReranker, build_reranker
from app.rag.vector_store import VectorIndex
from app.text import tokenize

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Evidence:
    """一条检索到的证据片段（可直接用于溯源展示）。"""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    score: float
    heading: str | None = None
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    relevance: float = 0.0
    ordinal: int = 0
    start: int = 0
    end: int = 0
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self, index: int | None = None) -> dict:
        payload = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "heading": self.heading,
            "content": self.content,
            "score": round(self.score, 4),
            "relevance": round(self.relevance, 4),
            "semantic_score": round(self.semantic_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "ordinal": self.ordinal,
            "start_offset": self.start,
            "end_offset": self.end,
            "matched_terms": self.matched_terms,
        }
        if index is not None:
            payload["index"] = index
        return payload


# --------------------------------------------------------------------------- #
class BM25Index:
    """轻量 BM25（Okapi）实现，进程内缓存，随写操作失效。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len: list[int] = []
        self._chunk_ids: list[str] = []
        self._document_of: list[str] = []
        self._avgdl: float = 1.0
        self._n_docs: int = 0

    def build(self, rows: Sequence[tuple[str, str, str]]) -> "BM25Index":
        self._postings.clear()
        self._doc_len.clear()
        self._chunk_ids.clear()
        self._document_of.clear()

        for chunk_id, document_id, content in rows:
            tokens = tokenize(content)
            index = len(self._chunk_ids)
            self._chunk_ids.append(chunk_id)
            self._document_of.append(document_id)
            self._doc_len.append(len(tokens) or 1)
            for term, tf in Counter(tokens).items():
                self._postings[term].append((index, tf))
        self._n_docs = len(self._chunk_ids)
        self._avgdl = (sum(self._doc_len) / self._n_docs) if self._n_docs else 1.0
        return self

    def search(
        self, query: str, top_k: int = 10, document_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float, list[str]]]:
        if not self._n_docs:
            return []
        allowed = set(document_ids) if document_ids else None
        terms = tokenize(query)
        if not terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, set[str]] = defaultdict(set)
        for term in set(terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (self._n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc_index, tf in postings:
                if allowed is not None and self._document_of[doc_index] not in allowed:
                    continue
                length = self._doc_len[doc_index]
                denominator = tf + self.k1 * (1 - self.b + self.b * length / self._avgdl)
                scores[doc_index] += idf * (tf * (self.k1 + 1) / denominator)
                matched[doc_index].add(term)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self._chunk_ids[index], score, sorted(matched[index])) for index, score in ranked]

    @property
    def size(self) -> int:
        return self._n_docs

    @property
    def corpus_size(self) -> int:
        return self._n_docs

    @property
    def max_idf(self) -> float:
        """语料中完全不存在的词所对应的（最大）IDF。"""
        if not self._n_docs:
            return 1.0
        return math.log(1 + (self._n_docs + 0.5) / 0.5)

    def idf(self, term: str) -> float | None:
        postings = self._postings.get(term)
        if not postings:
            return None
        return math.log(1 + (self._n_docs - len(postings) + 0.5) / (len(postings) + 0.5))


# --------------------------------------------------------------------------- #
class Retriever:
    """检索入口，供 Agent 工具层调用。"""

    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        vector_index: VectorIndex,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_index = vector_index
        self.reranker = reranker or build_reranker(settings)
        self._bm25 = BM25Index()
        self._bm25_ready = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    def invalidate(self) -> None:
        with self._lock:
            self._bm25_ready = False
        self.vector_index.invalidate()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
        min_score: float | None = None,
    ) -> list[Evidence]:
        query = (query or "").strip()
        if not query:
            return []
        top_k = top_k or self.settings.retrieval_top_k
        min_score = self.settings.min_relevance_score if min_score is None else min_score
        candidate_k = max(top_k * 4, self.settings.retrieval_candidate_k)

        semantic: dict[str, float] = {}
        try:
            query_vector = self.embedder.embed_query(query)
            for hit in self.vector_index.query(query_vector, candidate_k, document_ids):
                semantic[hit.chunk_id] = max(0.0, float(hit.score))
        except Exception as exc:  # noqa: BLE001 - 向量检索失败时退化为纯关键词
            logger.warning("向量检索失败，退回关键词检索：%s", exc)

        bm25 = self._bm25_index()
        keyword: dict[str, float] = {}
        matched_terms: dict[str, list[str]] = {}
        for chunk_id, score, terms in bm25.search(query, candidate_k, document_ids):
            keyword[chunk_id] = score
            matched_terms[chunk_id] = terms

        if not semantic and not keyword:
            return []

        max_keyword = max(keyword.values()) if keyword else 0.0
        max_semantic = max(semantic.values()) if semantic else 0.0
        candidates = list(dict.fromkeys([*semantic.keys(), *keyword.keys()]))

        phrase = " ".join(query.split())
        query_tokens = set(tokenize(query))
        scored: list[tuple[str, float, float, float]] = []
        for chunk_id in candidates:
            sem = semantic.get(chunk_id, 0.0) / max_semantic if max_semantic > 0 else 0.0
            kw = keyword.get(chunk_id, 0.0) / max_keyword if max_keyword > 0 else 0.0
            score = self.settings.semantic_weight * sem + self.settings.keyword_weight * kw
            scored.append((chunk_id, score, sem, kw))

        records = self._load_chunks([chunk_id for chunk_id, *_ in scored])
        if not records:
            return []

        evidences: list[Evidence] = []
        for chunk_id, score, sem, kw in scored:
            record = records.get(chunk_id)
            if record is None:
                continue
            content = record["content"]
            if phrase and len(phrase) >= 2 and phrase in content:
                score = min(1.0, score + 0.12)

            # 「绝对相关度」= 语义相似度 与「问题关键信息覆盖率」的加权和。
            # 覆盖率按 IDF 加权，避免「实现」「系统」这类高频词制造虚假命中；
            # 语料中完全不存在的词按最大 IDF 计入分母（问的知识库里没有 → 覆盖率被拉低）。
            # 注意：score 是归一化后的「排序分」，relevance 才是「知识库里到底有没有」的判据。
            raw_cosine = semantic.get(chunk_id, 0.0)
            coverage = self._idf_coverage(query_tokens, set(tokenize(content)), bm25)
            relevance = (
                self.settings.semantic_weight * raw_cosine + self.settings.keyword_weight * coverage
            )

            evidences.append(
                Evidence(
                    chunk_id=chunk_id,
                    document_id=record["document_id"],
                    document_name=record["document_name"],
                    content=content,
                    score=score,
                    heading=record["heading"],
                    semantic_score=sem,
                    keyword_score=kw,
                    relevance=relevance,
                    ordinal=record["ordinal"],
                    start=record["start_offset"],
                    end=record["end_offset"],
                    matched_terms=matched_terms.get(chunk_id, []),
                )
            )

        evidences.sort(key=lambda item: item.score, reverse=True)
        evidences = [item for item in evidences if item.relevance >= min_score]
        if not evidences:
            return []
        # 先精排再 MMR：精排决定「谁更相关」，MMR 决定「选哪些不重复」
        shortlist = evidences[: max(top_k, self.settings.rerank_top_n)]
        return self._mmr(query, self.reranker.rerank(query, shortlist), top_k)

    def search_many(
        self, queries: Sequence[str], *, top_k: int | None = None, document_ids: Sequence[str] | None = None
    ) -> list[Evidence]:
        """多子问题并行检索，合并去重后保留最高分。"""
        unique = list(dict.fromkeys(query for query in queries if query and query.strip()))
        if not unique:
            return []
        per_query = max(2, (top_k or self.settings.retrieval_top_k))

        def run(query: str) -> list[Evidence]:
            try:
                return self.search(query, top_k=per_query, document_ids=document_ids)
            except Exception as exc:  # noqa: BLE001 - 单个子问题失败不应拖垮整体
                logger.warning("子问题检索失败（%s）：%s", query, exc)
                return []

        if len(unique) == 1:
            batches = [run(unique[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(len(unique), 4)) as pool:
                batches = list(pool.map(run, unique))

        merged: dict[str, Evidence] = {}
        for batch in batches:
            for evidence in batch:
                existing = merged.get(evidence.chunk_id)
                if existing is None or evidence.score > existing.score:
                    merged[evidence.chunk_id] = evidence
        results = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        limit = top_k or self.settings.retrieval_top_k
        return results[: max(limit, self.settings.retrieval_top_k)]

    # ------------------------------------------------------------------ #
    def _bm25_index(self) -> BM25Index:
        with self._lock:
            if self._bm25_ready:
                return self._bm25
            with session_scope() as db:
                rows = db.execute(
                    select(Chunk.id, Chunk.document_id, Chunk.content).order_by(Chunk.created_at)
                ).all()
            self._bm25.build([(row[0], row[1], row[2]) for row in rows])
            self._bm25_ready = True
            logger.debug("BM25 索引已构建，共 %s 个片段", self._bm25.size)
            return self._bm25

    @staticmethod
    def _idf_coverage(query_tokens: set[str], chunk_tokens: set[str], bm25: BM25Index) -> float:
        """查询词的 IDF 加权覆盖比例：多少「关键信息」在这个片段里出现过。"""
        if not query_tokens:
            return 0.0
        weights = {term: bm25.idf(term) for term in query_tokens}
        known = [weight for weight in weights.values() if weight is not None]
        if not known:
            # 问题里的词在语料中一个都不存在：视为完全不覆盖（触发「暂无相关资料」兜底）
            return 0.0

        # 语料里不存在的词也要计入分母（说明知识库没覆盖到），但惩罚力度取「已命中词的平均 IDF」
        # 而不是理论最大 IDF —— 否则随着知识库变大，一个生僻词就能把整句的覆盖率打穿。
        penalty = min(bm25.max_idf, sum(known) / len(known))
        total = 0.0
        hit = 0.0
        for term, weight in weights.items():
            effective = penalty if weight is None else weight
            total += effective
            if weight is not None and term in chunk_tokens:
                hit += effective
        return hit / total if total else 0.0

    def _load_chunks(self, chunk_ids: Sequence[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        with session_scope() as db:
            rows = db.execute(
                select(
                    Chunk.id,
                    Chunk.document_id,
                    Chunk.content,
                    Chunk.heading,
                    Chunk.ordinal,
                    Chunk.start_offset,
                    Chunk.end_offset,
                    Document.name,
                )
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id.in_(list(chunk_ids)))
            ).all()
        return {
            row[0]: {
                "document_id": row[1],
                "content": row[2],
                "heading": row[3],
                "ordinal": row[4],
                "start_offset": row[5],
                "end_offset": row[6],
                "document_name": row[7],
            }
            for row in rows
        }

    def _load_vectors(self, chunk_ids: Sequence[str]) -> dict[str, np.ndarray]:
        if not chunk_ids:
            return {}
        with session_scope() as db:
            rows = db.execute(
                select(Chunk.id, Chunk.embedding, Chunk.embedding_dim).where(Chunk.id.in_(list(chunk_ids)))
            ).all()
        vectors: dict[str, np.ndarray] = {}
        for chunk_id, blob, dim in rows:
            vector = bytes_to_vector(blob, dim)
            if vector is not None and vector.size:
                vectors[chunk_id] = vector
        return vectors

    def _mmr(self, query: str, evidences: list[Evidence], top_k: int) -> list[Evidence]:
        """最大边际相关：既保证相关性，又避免内容高度重复。"""
        if len(evidences) <= 1:
            return evidences[:top_k]

        try:
            query_vector = self.embedder.embed_query(query)
        except Exception:  # noqa: BLE001
            return evidences[:top_k]

        vectors = self._load_vectors([item.chunk_id for item in evidences])
        if not vectors:
            return evidences[:top_k]

        query_vector = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query_vector)) or 1.0
        query_vector = query_vector / norm

        lambda_ = self.settings.mmr_lambda
        selected: list[Evidence] = []
        remaining = list(evidences)
        while remaining and len(selected) < top_k:
            best_index = -1
            best_value = -1e9
            for index, candidate in enumerate(remaining):
                vector = vectors.get(candidate.chunk_id)
                if vector is None:
                    similarity = 0.0
                else:
                    norm = float(np.linalg.norm(vector)) or 1.0
                    similarity = float(np.dot(vector / norm, query_vector))
                redundancy = 0.0
                for chosen in selected:
                    chosen_vector = vectors.get(chosen.chunk_id)
                    if chosen_vector is None:
                        continue
                    denominator = (float(np.linalg.norm(vector)) or 1.0) * (
                        float(np.linalg.norm(chosen_vector)) or 1.0
                    )
                    if denominator == 0:
                        continue
                    redundancy = max(redundancy, float(np.dot(vector, chosen_vector) / denominator))
                if selected and redundancy >= self.settings.chunk_dedup_threshold:
                    # 与已选片段近似重复，直接跳过（知识库自动去重的一环）
                    continue
                value = lambda_ * candidate.score + (1 - lambda_) * similarity - (1 - lambda_) * redundancy
                if value > best_value:
                    best_value = value
                    best_index = index
            if best_index < 0:
                break
            selected.append(remaining.pop(best_index))
        return selected[:top_k]


def build_retriever(settings: Settings | None = None, **kwargs) -> Retriever:
    from app.rag.embeddings import get_embedder
    from app.rag.vector_store import build_vector_index

    settings = settings or get_settings()
    return Retriever(
        settings=settings,
        embedder=kwargs.get("embedder") or get_embedder(settings),
        vector_index=kwargs.get("vector_index") or build_vector_index(settings),
        reranker=kwargs.get("reranker"),
    )
