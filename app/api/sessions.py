"""会话管理接口：新建、列表、清空、删除、历史消息。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import services
from app.db import repo
from app.db.base import session_scope
from app.schemas import SessionCreateRequest
from app.services import Services

router = APIRouter(prefix="/api/sessions", tags=["会话"])


def _session_to_dict(session) -> dict:
    return {
        "id": session.id,
        "title": session.title,
        "user_id": session.user_id,
        "message_count": session.message_count,
        "summary": session.summary,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }


@router.get("", summary="会话列表")
def list_sessions(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    svc: Services = Depends(services),
) -> dict:
    user_id = user_id or svc.settings.default_user_id
    with session_scope() as db:
        sessions = repo.list_sessions(db, user_id=user_id, limit=limit)
        return {"items": [_session_to_dict(item) for item in sessions]}


@router.post("", summary="新建会话")
def create_session(payload: SessionCreateRequest, svc: Services = Depends(services)) -> dict:
    user_id = payload.user_id or svc.settings.default_user_id
    with session_scope() as db:
        session = repo.create_session(db, user_id=user_id, title=payload.title or "新会话")
        return _session_to_dict(session)


@router.delete("/{session_id}", summary="删除会话")
def delete_session(session_id: str, svc: Services = Depends(services)) -> dict:
    with session_scope() as db:
        if not repo.delete_session(db, session_id):
            raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True, "message": "会话已删除"}


@router.post("/{session_id}/clear", summary="清空会话消息（保留会话本身）")
def clear_session(session_id: str, svc: Services = Depends(services)) -> dict:
    with session_scope() as db:
        if repo.get_session(db, session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        removed = repo.clear_session_messages(db, session_id)
    return {"ok": True, "removed": removed, "message": f"已清空 {removed} 条消息"}


@router.get("/{session_id}/messages", summary="会话历史消息")
def list_messages(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    svc: Services = Depends(services),
) -> dict:
    with session_scope() as db:
        if repo.get_session(db, session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        messages = repo.list_messages(db, session_id, limit=limit, ascending=True)
    return {
        "items": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "citations": message.citations or [],
                "plan": message.plan or {},
                "grounded": message.grounded,
                "latency_ms": message.latency_ms,
                "created_at": message.created_at.isoformat() if message.created_at else None,
            }
            for message in messages
        ]
    }
