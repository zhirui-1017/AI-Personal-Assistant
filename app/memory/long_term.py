"""用户长期记忆：抽取 → 合并去重 → 召回 → 轻量化清理。"""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np

from app.config import Settings
from app.db import repo
from app.db.base import session_scope
from app.llm.client import BaseLLM, extract_json
from app.rag.embeddings import BaseEmbedder, bytes_to_vector, vector_to_bytes
from app.text import truncate

logger = logging.getLogger(__name__)

_CATEGORIES = {"preference", "interest", "fact", "scenario"}

_RULE_PATTERNS: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"(?:我|本人)(?:是|是一名|是一个|做|从事|就读于|就职于)([^。！？；\n]{2,30})"), "fact", 0.7),
    (re.compile(r"我(?:喜欢|偏好|习惯用?|常用|更倾向|倾向于)([^。！？；\n]{2,40})"), "preference", 0.75),
    (re.compile(r"我(?:关注|重点研究|正在研究|正在学|想学|需要学)([^。！？；\n]{2,40})"), "interest", 0.7),
    (re.compile(r"(?:以后|下次|今后|之后)([^。！？；\n]{2,40})"), "scenario", 0.6),
    (re.compile(r"记住([^。！？；\n]{2,40})"), "fact", 0.8),
)

# 只有「像在介绍自己」的问题才值得额外花一次 LLM 调用来抽取长期记忆，
# 否则每个知识库问答都要多打一次模型，纯属浪费。
_MEMORY_HINTS = re.compile(
    r"我|咱|本人|记住|记一下|记下|以后|下次|今后|之后|偏好|习惯|常用|喜欢|研究|关注|正在学|需要学"
)


def memory_worthy(question: str) -> bool:
    """问题里是否包含「值得写进长期记忆」的自我介绍类信号。"""

    return bool(question and _MEMORY_HINTS.search(question))


class LongTermMemory:
    """跨会话记住用户偏好、身份与关注点，并在回答前注入上下文。"""

    def __init__(self, settings: Settings, llm: BaseLLM, embedder: BaseEmbedder) -> None:
        self.settings = settings
        self.llm = llm
        self.embedder = embedder

    # ------------------------------------------------------------------ #
    def recall(self, user_id: str, *, query: str | None = None, limit: int = 5) -> list[dict]:
        """召回与当前问题最相关的长期记忆。"""
        with session_scope() as db:
            memories = repo.list_memories(db, user_id=user_id, limit=self.settings.long_term_max_items)
            payload = [
                {
                    "id": item.id,
                    "category": item.category,
                    "content": item.content,
                    "importance": item.importance,
                    "hits": item.hits,
                    "embedding": bytes_to_vector(item.embedding, item.embedding_dim),
                }
                for item in memories
            ]
        if not payload:
            return []

        scores: list[tuple[float, dict]] = []
        query_vector = None
        if query:
            try:
                query_vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
                query_vector = query_vector / (float(np.linalg.norm(query_vector)) or 1.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("记忆召回向量化失败：%s", exc)
                query_vector = None

        for item in payload:
            base = float(item["importance"])
            similarity = 0.0
            vector = item["embedding"]
            if query_vector is not None and vector is not None and vector.size == query_vector.size:
                norm = float(np.linalg.norm(vector)) or 1.0
                similarity = float(np.dot(vector / norm, query_vector))
            score = 0.55 * base + 0.45 * max(0.0, similarity) + min(item["hits"], 5) * 0.01
            scores.append((score, item))

        scores.sort(key=lambda row: row[0], reverse=True)
        selected = [item for _, item in scores[:limit]]
        if selected:
            with session_scope() as db:
                for item in selected:
                    repo.update_memory(db, item["id"], hits=item["hits"] + 1)
        return [
            {"id": item["id"], "category": item["category"], "content": item["content"],
             "importance": item["importance"]}
            for item in selected
        ]

    def remember(self, user_id: str, *, question: str, answer: str, session_id: str | None = None) -> list[str]:
        """从一轮对话里抽取长期信息并写入记忆库。"""
        if not self.settings.memory_extract_enabled:
            return []
        if not memory_worthy(question):
            return []
        candidates = self._extract(question, answer)
        if not candidates:
            return []

        stored: list[str] = []
        with session_scope() as db:
            existing = repo.list_memories(db, user_id=user_id, limit=self.settings.long_term_max_items)
            existing_payload = [
                {
                    "id": item.id,
                    "content": item.content,
                    "importance": item.importance,
                    "vector": bytes_to_vector(item.embedding, item.embedding_dim),
                }
                for item in existing
            ]

            vectors: dict[str, np.ndarray] = {}
            try:
                encodings = self.embedder.embed_documents([item["content"] for item in candidates])
                for item, vector in zip(candidates, encodings, strict=False):
                    vectors[item["content"]] = np.asarray(vector, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                logger.warning("记忆向量化失败，仅按文本相似度合并：%s", exc)

            for item in candidates:
                content = item["content"]
                vector = vectors.get(content)
                merged = self._find_similar(content, vector, existing_payload)
                if merged is not None:
                    new_importance = max(float(merged["importance"]), float(item["importance"]))
                    repo.update_memory(db, merged["id"], importance=new_importance)
                    merged["importance"] = new_importance
                    continue
                memory = repo.add_memory(
                    db,
                    user_id=user_id,
                    content=content,
                    category=item["category"],
                    importance=float(item["importance"]),
                    source_session_id=session_id,
                    embedding=vector_to_bytes(vector) if vector is not None else None,
                    embedding_dim=int(vector.size) if vector is not None else None,
                )
                existing_payload.append({"id": memory.id, "content": content, "importance": float(memory.importance), "vector": vector})
                stored.append(content)
        if stored:
            self.cleanup(user_id)
        return stored

    def cleanup(self, user_id: str) -> int:
        """记忆轻量化：按「重要度 × 时效」保留高价值记忆，淘汰低价值记忆。"""
        with session_scope() as db:
            memories = repo.list_memories(db, user_id=user_id, limit=1000)
            if len(memories) <= self.settings.long_term_max_items:
                return 0
            scored = []
            for item in memories:
                scored.append((item, float(item.importance) + min(item.hits, 10) * 0.02))
            scored.sort(key=lambda row: (row[1], row[0].updated_at or row[0].created_at), reverse=True)
            removed = 0
            for item, _score in scored[self.settings.long_term_max_items :]:
                repo.deactivate_memory(db, item.id)
                removed += 1
        if removed:
            logger.info("长期记忆清理：停用 %s 条低价值记忆", removed)
        return removed

    def forget(self, memory_id: str) -> None:
        with session_scope() as db:
            repo.deactivate_memory(db, memory_id)

    def list(self, user_id: str) -> list[dict]:
        with session_scope() as db:
            memories = repo.list_memories(db, user_id=user_id, limit=self.settings.long_term_max_items)
        return [
            {
                "id": item.id,
                "category": item.category,
                "content": item.content,
                "importance": round(float(item.importance), 3),
                "hits": item.hits,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            }
            for item in memories
        ]

    # ------------------------------------------------------------------ #
    def _extract(self, question: str, answer: str) -> list[dict]:
        if self.llm.is_mock:
            return self._rule_extract(question)
        from app.agent import prompts

        try:
            response = self.llm.chat(
                prompts.build_memory_messages(question, answer),
                temperature=0.0,
                max_tokens=400,
                json_mode=True,
            )
            data = extract_json(response.content)
            if isinstance(data, dict):
                data = data.get("memories") or data.get("items")
            if isinstance(data, list):
                items = [_normalize(item) for item in data]
                return [item for item in items if item]
        except Exception as exc:  # noqa: BLE001
            logger.warning("长期记忆抽取失败，改用规则抽取：%s", exc)
        return self._rule_extract(question)

    @staticmethod
    def _rule_extract(question: str) -> list[dict]:
        text = (question or "").strip()
        if not text:
            return []
        results: list[dict] = []
        seen: set[str] = set()
        for pattern, category, importance in _RULE_PATTERNS:
            for match in pattern.finditer(text):
                content = match.group(1).strip(" ，,。.、")
                if len(content) < 2:
                    continue
                statement = f"{_category_prefix(category)}{content}"
                if statement in seen:
                    continue
                seen.add(statement)
                results.append({"category": category, "content": truncate(statement, 120), "importance": importance})
        return results[:3]

    def _find_similar(
        self, content: str, vector: np.ndarray | None, existing: Sequence[dict]
    ) -> dict | None:
        """语义近似或文本高度重合时，视为同一条记忆（自动去重）。"""
        for item in existing:
            if item["content"] == content:
                return item
            other = item.get("vector")
            if vector is not None and other is not None and other.size == vector.size:
                denominator = (float(np.linalg.norm(vector)) or 1.0) * (float(np.linalg.norm(other)) or 1.0)
                if denominator and float(np.dot(vector, other) / denominator) >= self.settings.long_term_merge_threshold:
                    return item
                continue
            if _text_overlap(content, item["content"]) >= 0.85:
                return item
        return None



def _normalize(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    content = str(item.get("content", "")).strip()
    if len(content) < 2:
        return None
    category = str(item.get("category", "fact")).strip().lower()
    if category not in _CATEGORIES:
        category = "fact"
    try:
        importance = float(item.get("importance", 0.5))
    except (TypeError, ValueError):
        importance = 0.5
    return {"category": category, "content": truncate(content, 200), "importance": min(max(importance, 0.0), 1.0)}


def _category_prefix(category: str) -> str:
    return {
        "preference": "用户偏好：",
        "interest": "用户关注：",
        "fact": "用户信息：",
        "scenario": "常用场景：",
    }.get(category, "")


def _text_overlap(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    a, b = set(left), set(right)
    return len(a & b) / max(len(a), 1)
