"""问答接口：普通问答、SSE 流式问答、长期记忆管理。"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.api.deps import services
from app.db import repo
from app.db.base import session_scope
from app.schemas import ChatRequest, FeedbackRequest
from app.services import Services
from app.text import truncate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["问答"])


def _ensure_session(svc: Services, session_id: str | None, user_id: str | None, question: str) -> str:
    user_id = user_id or svc.settings.default_user_id
    with session_scope() as db:
        if session_id:
            session = repo.get_session(db, session_id)
            if session is None:
                session = repo.create_session(
                    db, user_id=user_id, session_id=session_id, title=truncate(question, 30) or "新会话"
                )
            return session.id
        session = repo.create_session(db, user_id=user_id, title=truncate(question, 30) or "新会话")
        return session.id


@router.post("/chat", summary="知识库问答（Agent 完整链路）")
def chat(payload: ChatRequest, svc: Services = Depends(services)) -> dict:
    session_id = _ensure_session(svc, payload.session_id, payload.user_id, payload.question)
    result = svc.executor.run(
        payload.question,
        session_id=session_id,
        user_id=payload.user_id,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )
    data = result.to_dict()
    data["session_id"] = session_id
    return data


@router.post("/chat/stream", summary="知识库问答（SSE 流式输出）")
def chat_stream(payload: ChatRequest, svc: Services = Depends(services)):
    session_id = _ensure_session(svc, payload.session_id, payload.user_id, payload.question)

    def event_stream():
        def pack(event: dict) -> str:
            return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        yield pack({"type": "session", "data": {"session_id": session_id}})
        try:
            for event in svc.executor.stream(
                payload.question,
                session_id=session_id,
                user_id=payload.user_id,
                top_k=payload.top_k,
                document_ids=payload.document_ids,
            ):
                yield pack(event)
        except Exception as exc:  # noqa: BLE001 - 流式过程出错也要让前端知道
            logger.exception("流式问答失败")
            yield pack({"type": "error", "data": f"问答失败：{exc}"})
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/memories", summary="查看用户长期记忆")
def list_memories(
    user_id: str | None = Query(default=None), svc: Services = Depends(services)
) -> dict:
    user_id = user_id or svc.settings.default_user_id
    items = svc.long_term.list(user_id)
    return {"user_id": user_id, "total": len(items), "items": items}


@router.delete("/memories/{memory_id}", summary="删除一条长期记忆")
def delete_memory(memory_id: str, svc: Services = Depends(services)) -> dict:
    svc.long_term.forget(memory_id)
    return {"ok": True, "message": "已遗忘该条记忆"}


@router.post("/chat/feedback", summary="对最近一次回答给出评价（👍/👎）")
def chat_feedback(payload: FeedbackRequest, svc: Services = Depends(services)) -> dict:
    """把反馈记到该会话最近一条问答日志上，供后续分析回答质量。"""

    with session_scope() as db:
        log = repo.set_latest_qa_feedback(db, payload.session_id, payload.rating)
    if log is None:
        raise HTTPException(status_code=404, detail="该会话还没有问答记录")
    return {"ok": True, "rating": payload.rating, "qa_log_id": log.id}
