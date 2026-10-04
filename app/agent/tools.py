"""Agent 可调用的工具集。

Agent 的「工具调用决策」就落在这一层：Planner 决定意图，Executor 选择工具，
工具统一返回 ``ToolResult``（结构化数据 + 证据片段）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from sqlalchemy import func, select

from app.config import Settings
from app.db.base import session_scope
from app.db.models import ChatSession, Chunk, Document, Memory, Message, QaLog
from app.rag.retriever import Evidence, Retriever
from app.text import truncate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolResult:
    name: str
    ok: bool = True
    summary: str = ""
    data: dict = field(default_factory=dict)
    evidences: list[Evidence] = field(default_factory=list)


class AgentTools:
    """Agent 工具注册表。"""

    def __init__(self, settings: Settings, retriever: Retriever) -> None:
        self.settings = settings
        self.retriever = retriever

    # ------------------------------------------------------------------ #
    def search_knowledge(
        self, query: str, *, top_k: int | None = None, document_ids: Sequence[str] | None = None
    ) -> ToolResult:
        """语义 + 关键词混合检索知识库。"""
        evidences = self.retriever.search(query, top_k=top_k, document_ids=document_ids)
        return ToolResult(
            name="search_knowledge",
            ok=bool(evidences),
            summary=f"检索「{truncate(query, 24)}」命中 {len(evidences)} 个片段",
            data={
                "query": query,
                "count": len(evidences),
                "top_relevance": round(evidences[0].relevance, 4) if evidences else 0.0,
            },
            evidences=evidences,
        )

    def search_many(
        self,
        queries: Sequence[str],
        *,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> ToolResult:
        """多子问题并行检索并合并去重。"""
        evidences = self.retriever.search_many(queries, top_k=top_k, document_ids=document_ids)
        return ToolResult(
            name="search_many",
            ok=bool(evidences),
            summary=f"{len(queries)} 个子问题共命中 {len(evidences)} 个去重片段",
            data={"queries": list(queries), "count": len(evidences)},
            evidences=evidences,
        )

    def expand_neighbors(self, evidences: Sequence[Evidence], *, span: int = 1) -> list[Evidence]:
        """补全相邻片段：把命中片段前后各 span 个片段一并取回，避免上下文被切断。"""
        if not evidences:
            return []
        pairs: list[tuple[str, int]] = []
        for evidence in evidences:
            for offset in range(-span, span + 1):
                ordinal = evidence.ordinal + offset
                if ordinal >= 0:
                    pairs.append((evidence.document_id, ordinal))

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
                .where(Document.id.in_({document_id for document_id, _ in pairs}))
            ).all()

        lookup: dict[tuple[str, int], dict] = {(row[1], row[4]): {
            "chunk_id": row[0],
            "document_id": row[1],
            "content": row[2],
            "heading": row[3],
            "ordinal": row[4],
            "start": row[5],
            "end": row[6],
            "document_name": row[7],
        } for row in rows}

        base_scores = {evidence.chunk_id: evidence.score for evidence in evidences}
        merged: dict[str, Evidence] = {evidence.chunk_id: evidence for evidence in evidences}
        for document_id, ordinal in pairs:
            record = lookup.get((document_id, ordinal))
            if record is None or record["chunk_id"] in merged:
                continue
            parent_score = base_scores.get(record["chunk_id"], 0.0)
            merged[record["chunk_id"]] = Evidence(
                chunk_id=record["chunk_id"],
                document_id=record["document_id"],
                document_name=record["document_name"],
                content=record["content"],
                score=max(0.05, parent_score * 0.85),
                heading=record["heading"],
                ordinal=record["ordinal"],
                start=record["start"],
                end=record["end"],
            )
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)

    def list_documents(self, *, limit: int = 50) -> ToolResult:
        """列出知识库中的文档。"""
        with session_scope() as db:
            rows = db.execute(
                select(Document.id, Document.name, Document.chunk_count, Document.status,
                       Document.file_size, Document.created_at)
                .order_by(Document.created_at.desc())
                .limit(limit)
            ).all()
        documents = [
            {
                "id": row[0],
                "name": row[1],
                "chunk_count": row[2],
                "status": row[3],
                "file_size": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
            }
            for row in rows
        ]
        return ToolResult(
            name="list_documents",
            ok=bool(documents),
            summary=f"知识库共有 {len(documents)} 篇文档",
            data={"documents": documents},
        )

    def read_document(self, document_id: str, *, max_chars: int = 12000) -> ToolResult:
        """读取整篇文档的片段（用于总结 / 对比类问题）。"""
        with session_scope() as db:
            document = db.get(Document, document_id)
            if document is None:
                return ToolResult(name="read_document", ok=False, summary="文档不存在", data={})
            rows = db.execute(
                select(
                    Chunk.id, Chunk.content, Chunk.heading, Chunk.ordinal, Chunk.start_offset, Chunk.end_offset
                )
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.ordinal)
            ).all()
            name = document.name

        evidences: list[Evidence] = []
        used = 0
        for chunk_id, content, heading, ordinal, start, end in rows:
            if used + len(content) > max_chars and evidences:
                break
            used += len(content)
            evidences.append(
                Evidence(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=name,
                    content=content,
                    score=1.0,
                    heading=heading,
                    ordinal=ordinal,
                    start=start,
                    end=end,
                )
            )
        return ToolResult(
            name="read_document",
            ok=bool(evidences),
            summary=f"读取《{name}》共 {len(evidences)} 个片段（{used} 字）",
            data={"document_id": document_id, "document_name": name, "chars": used},
            evidences=evidences,
        )

    def kb_stats(self, *, user_id: str | None = None) -> ToolResult:
        """知识库统计：文档数、片段数、问答次数、会话数、记忆条数。"""
        with session_scope() as db:
            documents = int(db.scalar(select(func.count()).select_from(Document)) or 0)
            indexed = int(
                db.scalar(select(func.count()).select_from(Document).where(Document.status == "indexed")) or 0
            )
            failed = int(
                db.scalar(select(func.count()).select_from(Document).where(Document.status == "failed")) or 0
            )
            chunks = int(db.scalar(select(func.count()).select_from(Chunk)) or 0)
            sessions = int(db.scalar(select(func.count()).select_from(ChatSession)) or 0)
            messages = int(db.scalar(select(func.count()).select_from(Message)) or 0)
            qa_count = int(db.scalar(select(func.count()).select_from(QaLog)) or 0)
            memories = int(
                db.scalar(select(func.count()).select_from(Memory).where(Memory.active.is_(True))) or 0
            )
            top_rows = db.execute(
                select(Document.name, Document.chunk_count)
                .where(Document.chunk_count > 0)
                .order_by(Document.chunk_count.desc())
                .limit(5)
            ).all()

        data = {
            "documents": documents,
            "indexed_documents": indexed,
            "failed_documents": failed,
            "chunks": chunks,
            "sessions": sessions,
            "messages": messages,
            "qa_count": qa_count,
            "memories": memories,
            "top_documents": [{"name": row[0], "chunk_count": row[1]} for row in top_rows],
        }
        summary = (
            f"知识库现有 {documents} 篇文档（已索引 {indexed} 篇，失败 {failed} 篇），"
            f"共 {chunks} 个知识片段，累计问答 {qa_count} 次，会话 {sessions} 个，长期记忆 {memories} 条。"
        )
        return ToolResult(name="kb_stats", ok=True, summary=summary, data=data)

    def health(self) -> dict:
        return {
            "tools": ["search_knowledge", "search_many", "expand_neighbors", "list_documents", "read_document", "kb_stats"],
            "retriever": {
                "backend": self.retriever.vector_index.backend,
                "bm25_ready": self.retriever._bm25_ready,
                "rerank": self.retriever.reranker.health(),
            },
        }
