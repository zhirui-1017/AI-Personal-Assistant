"""数据访问层：把 SQLAlchemy 的样板代码集中在这里，业务层只调用函数。"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.db.models import ChatSession, Chunk, Document, IngestTask, Memory, Message, QaLog, utcnow


def new_id() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------------------- #
# 文档
# --------------------------------------------------------------------------- #
def create_document(
    db: Session,
    *,
    name: str,
    source_type: str = "file",
    source_uri: str | None = None,
    stored_path: str | None = None,
    file_size: int = 0,
    content_hash: str | None = None,
    simhash: str | None = None,
    meta: dict[str, Any] | None = None,
    doc_id: str | None = None,
) -> Document:
    document = Document(
        id=doc_id or new_id(),
        name=name,
        source_type=source_type,
        source_uri=source_uri,
        stored_path=stored_path,
        file_size=file_size,
        content_hash=content_hash,
        simhash=simhash,
        meta=meta or {},
        status="pending",
    )
    db.add(document)
    db.flush()
    return document


def get_document(db: Session, document_id: str) -> Document | None:
    return db.get(Document, document_id)


def list_documents(
    db: Session, *, limit: int | None = None, offset: int = 0, include_failed: bool = True
) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.desc()).offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    if not include_failed:
        stmt = stmt.where(Document.status != "failed")
    return list(db.scalars(stmt))


def find_document_by_content_hash(db: Session, content_hash: str) -> Document | None:
    stmt = select(Document).where(Document.content_hash == content_hash).order_by(Document.created_at.desc())
    return db.scalars(stmt).first()


def find_documents_by_simhash(db: Session, simhash: str | None) -> list[Document]:
    if not simhash:
        return []
    stmt = select(Document).where(Document.simhash == simhash)
    return list(db.scalars(stmt))


def update_document(db: Session, document_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = utcnow()
    db.execute(update(Document).where(Document.id == document_id).values(**fields))


def delete_document(db: Session, document_id: str) -> bool:
    document = db.get(Document, document_id)
    if document is None:
        return False
    db.delete(document)
    db.flush()
    return True


def count_documents(db: Session) -> int:
    return int(db.scalar(select(func.count()).select_from(Document)) or 0)


# --------------------------------------------------------------------------- #
# 片段
# --------------------------------------------------------------------------- #
def add_chunk(
    db: Session,
    *,
    document_id: str,
    ordinal: int,
    content: str,
    content_hash: str,
    heading: str | None = None,
    start_offset: int = 0,
    end_offset: int = 0,
    simhash: str | None = None,
    chunk_id: str | None = None,
) -> Chunk:
    chunk = Chunk(
        id=chunk_id or new_id(),
        document_id=document_id,
        ordinal=ordinal,
        content=content,
        heading=heading,
        char_count=len(content),
        start_offset=start_offset,
        end_offset=end_offset,
        content_hash=content_hash,
        simhash=simhash,
    )
    db.add(chunk)
    return chunk


def add_chunks(db: Session, chunks: Sequence[Chunk]) -> None:
    db.add_all(list(chunks))
    db.flush()


def get_chunk(db: Session, chunk_id: str) -> Chunk | None:
    return db.get(Chunk, chunk_id)


def get_chunks_by_ids(db: Session, chunk_ids: Sequence[str]) -> list[Chunk]:
    if not chunk_ids:
        return []
    stmt = select(Chunk).where(Chunk.id.in_(list(chunk_ids)))
    return list(db.scalars(stmt))


def list_chunks(db: Session, document_id: str, *, limit: int | None = None, offset: int = 0) -> list[Chunk]:
    stmt = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.ordinal)
        .offset(offset)
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt))


def list_chunk_window(db: Session, document_id: str, ordinal: int, *, span: int = 2) -> list[Chunk]:
    """取某个片段前后各 span 个片段（溯源弹窗用）。

    直接在 SQL 里按 ordinal 取区间，避免把整篇文档的片段都读出来再排序。
    """

    stmt = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .where(Chunk.ordinal >= ordinal - span)
        .where(Chunk.ordinal <= ordinal + span)
        .order_by(Chunk.ordinal)
    )
    return list(db.scalars(stmt))


def delete_chunks_of_document(db: Session, document_id: str) -> None:
    db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    db.flush()


def iter_all_chunks(db: Session, batch_size: int = 500):
    """流式遍历全部片段，避免一次性把千级片段读进内存。"""
    offset = 0
    while True:
        batch = list(db.scalars(select(Chunk).order_by(Chunk.ordinal).offset(offset).limit(batch_size)))
        if not batch:
            return
        yield from batch
        offset += batch_size


def count_chunks(db: Session, document_id: str | None = None) -> int:
    stmt = select(func.count()).select_from(Chunk)
    if document_id:
        stmt = stmt.where(Chunk.document_id == document_id)
    return int(db.scalar(stmt) or 0)


# --------------------------------------------------------------------------- #
# 会话与消息
# --------------------------------------------------------------------------- #
def create_session(db: Session, *, user_id: str, title: str = "新会话", session_id: str | None = None) -> ChatSession:
    session = ChatSession(id=session_id or new_id(), user_id=user_id, title=title[:200])
    db.add(session)
    db.flush()
    return session


def get_session(db: Session, session_id: str) -> ChatSession | None:
    return db.get(ChatSession, session_id)


def list_sessions(db: Session, *, user_id: str | None = None, limit: int = 100) -> list[ChatSession]:
    stmt = select(ChatSession).where(ChatSession.archived.is_(False))
    if user_id:
        stmt = stmt.where(ChatSession.user_id == user_id)
    stmt = stmt.order_by(ChatSession.updated_at.desc()).limit(limit)
    return list(db.scalars(stmt))


def delete_session(db: Session, session_id: str) -> bool:
    session = db.get(ChatSession, session_id)
    if session is None:
        return False
    db.delete(session)
    db.flush()
    return True


def clear_session_messages(db: Session, session_id: str) -> int:
    removed = db.execute(delete(Message).where(Message.session_id == session_id)).rowcount or 0
    db.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(summary=None, summarized_upto=0, message_count=0, updated_at=utcnow())
    )
    db.flush()
    return int(removed)


def touch_session(
    db: Session, session_id: str, *, title: str | None = None, summary: str | None = None,
    summarized_upto: int | None = None,
) -> None:
    fields: dict[str, Any] = {"updated_at": utcnow()}
    if title is not None:
        fields["title"] = title[:200]
    if summary is not None:
        fields["summary"] = summary
    if summarized_upto is not None:
        fields["summarized_upto"] = summarized_upto
    db.execute(update(ChatSession).where(ChatSession.id == session_id).values(**fields))


def add_message(
    db: Session,
    *,
    session_id: str,
    role: str,
    content: str,
    citations: list | None = None,
    evidence: list | None = None,
    plan: dict | None = None,
    grounded: bool = True,
    latency_ms: int = 0,
    message_id: str | None = None,
) -> Message:
    message = Message(
        id=message_id or new_id(),
        session_id=session_id,
        role=role,
        content=content,
        citations=citations or [],
        evidence=evidence or [],
        plan=plan or {},
        grounded=grounded,
        latency_ms=latency_ms,
    )
    db.add(message)
    db.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(message_count=ChatSession.message_count + 1, updated_at=utcnow())
    )
    db.flush()
    return message


def list_messages(db: Session, session_id: str, *, limit: int | None = None, ascending: bool = True) -> list[Message]:
    stmt = select(Message).where(Message.session_id == session_id)
    stmt = stmt.order_by(Message.created_at.asc() if ascending else Message.created_at.desc())
    if limit is not None:
        stmt = stmt.limit(limit)
    messages = list(db.scalars(stmt))
    return messages if ascending else list(reversed(messages))


def count_messages(db: Session, session_id: str | None = None) -> int:
    stmt = select(func.count()).select_from(Message)
    if session_id:
        stmt = stmt.where(Message.session_id == session_id)
    return int(db.scalar(stmt) or 0)


# --------------------------------------------------------------------------- #
# 长期记忆
# --------------------------------------------------------------------------- #
def list_memories(db: Session, *, user_id: str, active_only: bool = True, limit: int | None = None) -> list[Memory]:
    stmt = select(Memory).where(Memory.user_id == user_id)
    if active_only:
        stmt = stmt.where(Memory.active.is_(True))
    stmt = stmt.order_by(Memory.importance.desc(), Memory.updated_at.desc())
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt))


def add_memory(
    db: Session,
    *,
    user_id: str,
    content: str,
    category: str = "fact",
    importance: float = 0.5,
    source_session_id: str | None = None,
    embedding: bytes | None = None,
    embedding_dim: int | None = None,
) -> Memory:
    memory = Memory(
        id=new_id(),
        user_id=user_id,
        category=category,
        content=content,
        importance=importance,
        source_session_id=source_session_id,
        embedding=embedding,
        embedding_dim=embedding_dim,
    )
    db.add(memory)
    db.flush()
    return memory


def update_memory(db: Session, memory_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = utcnow()
    db.execute(update(Memory).where(Memory.id == memory_id).values(**fields))


def deactivate_memory(db: Session, memory_id: str) -> None:
    db.execute(update(Memory).where(Memory.id == memory_id).values(active=False, updated_at=utcnow()))


def count_memories(db: Session, user_id: str | None = None) -> int:
    stmt = select(func.count()).select_from(Memory).where(Memory.active.is_(True))
    if user_id:
        stmt = stmt.where(Memory.user_id == user_id)
    return int(db.scalar(stmt) or 0)


# --------------------------------------------------------------------------- #
# 问答日志与统计
# --------------------------------------------------------------------------- #
def add_qa_log(
    db: Session,
    *,
    user_id: str,
    question: str,
    session_id: str | None = None,
    intent: str = "simple_qa",
    complexity: str = "simple",
    sub_question_count: int = 0,
    retrieved_count: int = 0,
    grounded: bool = True,
    latency_ms: int = 0,
    reflection_rounds: int = 0,
) -> QaLog:
    log = QaLog(
        id=new_id(),
        session_id=session_id,
        user_id=user_id,
        question=question[:2000],
        intent=intent,
        complexity=complexity,
        sub_question_count=sub_question_count,
        retrieved_count=retrieved_count,
        grounded=grounded,
        latency_ms=latency_ms,
        reflection_rounds=reflection_rounds,
    )
    db.add(log)
    db.flush()
    return log


def set_latest_qa_feedback(db: Session, session_id: str, feedback: str) -> QaLog | None:
    """把反馈写到该会话最近一次的问答日志上（前端的 👍/👎 就落在这里）。"""

    stmt = (
        select(QaLog)
        .where(QaLog.session_id == session_id)
        .order_by(QaLog.created_at.desc())
        .limit(1)
    )
    log = db.scalars(stmt).first()
    if log is None:
        return None
    log.feedback = feedback[:16]
    db.flush()
    return log


def count_qa(db: Session, *, user_id: str | None = None) -> int:
    stmt = select(func.count()).select_from(QaLog)
    if user_id:
        stmt = stmt.where(QaLog.user_id == user_id)
    return int(db.scalar(stmt) or 0)


def qa_recent_days(db: Session, days: int = 7) -> list[dict[str, Any]]:
    since = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=days - 1)
    day = func.date(QaLog.created_at)
    rows = db.execute(
        select(day.label("day"), func.count().label("total")).where(QaLog.created_at >= since).group_by(day).order_by(day)
    ).all()
    return [{"date": str(row.day), "total": int(row.total)} for row in rows]


def qa_intent_distribution(db: Session) -> list[dict[str, Any]]:
    rows = db.execute(
        select(QaLog.intent, func.count().label("total")).group_by(QaLog.intent).order_by(func.count().desc())
    ).all()
    return [{"intent": str(row.intent), "total": int(row.total)} for row in rows]


def average_latency(db: Session) -> float:
    value = db.scalar(select(func.avg(QaLog.latency_ms)))
    return round(float(value), 1) if value is not None else 0.0


# --------------------------------------------------------------------------- #
# 入库任务
# --------------------------------------------------------------------------- #
def create_task(db: Session, *, name: str, document_id: str | None = None, task_id: str | None = None) -> IngestTask:
    task = IngestTask(id=task_id or new_id(), name=name[:512], document_id=document_id, status="queued", stage="queued")
    db.add(task)
    db.flush()
    return task


def update_task(db: Session, task_id: str, **fields: Any) -> None:
    if fields.get("status") in {"succeeded", "failed", "skipped"}:
        fields.setdefault("finished_at", utcnow())
    fields["updated_at"] = utcnow()
    db.execute(update(IngestTask).where(IngestTask.id == task_id).values(**fields))


def get_task(db: Session, task_id: str) -> IngestTask | None:
    return db.get(IngestTask, task_id)


def list_tasks(db: Session, *, limit: int = 50, active_only: bool = False) -> list[IngestTask]:
    stmt = select(IngestTask).order_by(IngestTask.created_at.desc()).limit(limit)
    if active_only:
        stmt = stmt.where(IngestTask.status.in_(["queued", "running"]))
    return list(db.scalars(stmt))


def stale_tasks(db: Session) -> list[IngestTask]:
    """重启后仍处于 queued/running 的任务（用于标记为中断）。"""
    stmt = select(IngestTask).where(IngestTask.status.in_(["queued", "running"]))
    return list(db.scalars(stmt))
