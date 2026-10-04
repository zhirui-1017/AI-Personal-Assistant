"""知识库管理接口：上传、列表、删除、重新解析、溯源定位。"""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select

from app.api.deps import services
from app.db import repo
from app.db.base import session_scope
from app.db.models import Chunk, Document
from app.schemas import TextIngestRequest, UrlIngestRequest
from app.services import Services
from app.text import truncate

router = APIRouter(prefix="/api/documents", tags=["知识库"])


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _document_to_dict(document: Document, *, with_meta: bool = False) -> dict:
    payload = {
        "id": document.id,
        "name": document.name,
        "source_type": document.source_type,
        "source_uri": document.source_uri,
        "file_size": document.file_size,
        "status": document.status,
        "error": document.error,
        "char_count": document.char_count,
        "chunk_count": document.chunk_count,
        "duplicate_of": document.duplicate_of,
        "created_at": _iso(document.created_at),
        "updated_at": _iso(document.updated_at),
    }
    if with_meta:
        payload["meta"] = document.meta or {}
    return payload


# --------------------------------------------------------------------------- #
# 任务进度（放在 /{document_id} 之前，避免被路径参数吞掉）
# --------------------------------------------------------------------------- #
@router.get("/tasks", summary="入库任务列表")
def list_tasks(limit: int = Query(default=30, ge=1, le=200), svc: Services = Depends(services)) -> dict:
    return {"items": svc.ingest.tasks(limit=limit)}


@router.get("/tasks/{task_id}", summary="查询入库任务进度")
def get_task(task_id: str, svc: Services = Depends(services)) -> dict:
    status = svc.ingest.status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return status


# --------------------------------------------------------------------------- #
@router.post("/upload", summary="上传文件（支持多选批量导入）")
async def upload_documents(
    files: list[UploadFile] = File(...), svc: Services = Depends(services)
) -> dict:
    settings = svc.settings
    results: list[dict] = []
    max_bytes = settings.max_upload_mb * 1024 * 1024

    for file in files:
        filename = file.filename or "未命名文件"
        suffix = Path(filename).suffix.lower()
        if suffix not in settings.supported_extensions:
            results.append(
                {
                    "name": filename,
                    "status": "rejected",
                    "message": f"不支持的文件类型 {suffix or '未知'}，支持：{'/'.join(settings.supported_extensions)}",
                }
            )
            continue

        content = await file.read()
        if len(content) > max_bytes:
            results.append(
                {"name": filename, "status": "rejected", "message": f"文件超过 {settings.max_upload_mb}MB 限制"}
            )
            continue
        if not content:
            results.append({"name": filename, "status": "rejected", "message": "文件内容为空"})
            continue

        stored = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
        stored.write_bytes(content)
        task_id = svc.ingest.submit_path(stored, name=filename)
        results.append(
            {
                "name": filename,
                "status": "queued",
                "task_id": task_id,
                "size": len(content),
                "message": "已加入解析队列",
            }
        )
    return {"items": results}


@router.post("/url", summary="导入网页链接")
def ingest_url(payload: UrlIngestRequest, svc: Services = Depends(services)) -> dict:
    task_id = svc.ingest.submit_url(payload.url, name=payload.name)
    return {"task_id": task_id, "status": "queued", "name": payload.name or payload.url}


@router.post("/text", summary="导入一段文本")
def ingest_text(payload: TextIngestRequest, svc: Services = Depends(services)) -> dict:
    task_id = svc.ingest.submit_text(payload.text, name=payload.name)
    return {"task_id": task_id, "status": "queued", "name": payload.name}


@router.get("", summary="知识库文档列表")
def list_documents(
    limit: int | None = Query(default=None, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    svc: Services = Depends(services),
) -> dict:
    with session_scope() as db:
        documents = repo.list_documents(db, limit=limit, offset=offset)
        total = repo.count_documents(db)
        items = [_document_to_dict(document) for document in documents]
    return {"total": total, "items": items}


@router.get("/{document_id}", summary="文档详情")
def get_document(document_id: str, svc: Services = Depends(services)) -> dict:
    with session_scope() as db:
        document = repo.get_document(db, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        payload = _document_to_dict(document, with_meta=True)
        payload["headings"] = list(
            db.execute(
                select(Chunk.heading)
                .where(Chunk.document_id == document_id)
                .where(Chunk.heading.is_not(None))
                .distinct()
            ).scalars()
        )[:50]
    return payload


@router.delete("/{document_id}", summary="删除文档（同时清理向量与片段）")
def delete_document(document_id: str, svc: Services = Depends(services)) -> dict:
    with session_scope() as db:
        document = repo.get_document(db, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        name = document.name
        stored_path = document.stored_path
        repo.delete_document(db, document_id)

    with contextlib.suppress(Exception):
        # 片段已随文档级联删除，向量清理失败不影响主流程
        svc.vector_index.delete(document_id=document_id)
    svc.retriever.invalidate()

    if stored_path:
        with contextlib.suppress(OSError):
            Path(stored_path).unlink(missing_ok=True)
    return {"ok": True, "message": f"已删除《{name}》"}


@router.post("/{document_id}/reindex", summary="重新解析并刷新向量索引")
def reindex_document(document_id: str, svc: Services = Depends(services)) -> dict:
    with session_scope() as db:
        document = repo.get_document(db, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        name = document.name
    task_id = svc.ingest.submit_reindex(document_id, name=name)
    return {"task_id": task_id, "status": "queued", "document_id": document_id}


@router.get("/{document_id}/chunks", summary="文档片段列表（分页）")
def list_chunks(
    document_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    preview: int = Query(default=200, ge=40, le=2000),
    svc: Services = Depends(services),
) -> dict:
    with session_scope() as db:
        chunks = repo.list_chunks(db, document_id, limit=limit, offset=offset)
        total = repo.count_chunks(db, document_id)
    return {
        "total": total,
        "items": [
            {
                "id": chunk.id,
                "ordinal": chunk.ordinal,
                "heading": chunk.heading,
                "char_count": chunk.char_count,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "preview": truncate(chunk.content, preview),
            }
            for chunk in chunks
        ],
    }


@router.get("/{document_id}/chunks/{chunk_id}/context", summary="溯源：定位片段及其上下文")
def chunk_context(
    document_id: str,
    chunk_id: str,
    span: int = Query(default=2, ge=0, le=5),
    svc: Services = Depends(services),
) -> dict:
    with session_scope() as db:
        chunk = repo.get_chunk(db, chunk_id)
        if chunk is None or chunk.document_id != document_id:
            raise HTTPException(status_code=404, detail="片段不存在")
        document = repo.get_document(db, document_id)
        window = repo.list_chunk_window(db, document_id, chunk.ordinal, span=span)

    return {
        "document": {"id": document_id, "name": document.name if document else ""},
        "target": {"id": chunk.id, "ordinal": chunk.ordinal, "heading": chunk.heading,
                   "start_offset": chunk.start_offset, "end_offset": chunk.end_offset,
                   "content": chunk.content},
        "window": [
            {
                "id": item.id,
                "ordinal": item.ordinal,
                "heading": item.heading,
                "content": item.content,
                "is_target": item.id == chunk_id,
            }
            for item in window
        ],
    }
