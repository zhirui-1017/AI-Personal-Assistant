"""系统接口：健康检查与知识库统计。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from app import __version__
from app.api.auth import require_api_token
from app.api.deps import services
from app.db import repo
from app.db.base import session_scope
from app.db.models import ChatSession, Chunk, Document, Memory, Message
from app.schemas import HealthResponse
from app.services import Services

router = APIRouter(prefix="/api", tags=["系统"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
def health(svc: Services = Depends(services)) -> HealthResponse:
    return HealthResponse(
        status="ok", version=__version__, config=svc.settings.describe(), components=svc.health()
    )


@router.get(
    "/stats",
    summary="知识库统计（文档数 / 片段数 / 问答次数）",
    dependencies=[Depends(require_api_token)],
)
def stats(
    user_id: str | None = None, days: int = 7, svc: Services = Depends(services)
) -> dict:
    user_id = user_id or svc.settings.default_user_id
    with session_scope() as db:
        documents = int(db.scalar(select(func.count()).select_from(Document)) or 0)
        indexed = int(db.scalar(select(func.count()).select_from(Document).where(Document.status == "indexed")) or 0)
        failed = int(db.scalar(select(func.count()).select_from(Document).where(Document.status == "failed")) or 0)
        chunks = int(db.scalar(select(func.count()).select_from(Chunk)) or 0)
        sessions = int(db.scalar(select(func.count()).select_from(ChatSession)) or 0)
        messages = int(db.scalar(select(func.count()).select_from(Message)) or 0)
        memories = int(db.scalar(select(func.count()).select_from(Memory).where(Memory.active.is_(True))) or 0)
        payload = {
            "documents": documents,
            "indexed_documents": indexed,
            "failed_documents": failed,
            "chunks": chunks,
            "sessions": sessions,
            "messages": messages,
            "memories": memories,
            "qa_count": repo.count_qa(db, user_id=None),
            "average_latency_ms": repo.average_latency(db),
            "qa_recent_days": repo.qa_recent_days(db, days=max(1, min(days, 30))),
            "intent_distribution": repo.qa_intent_distribution(db),
            "vector_vectors": svc.vector_index.count(),
        }
    return payload
