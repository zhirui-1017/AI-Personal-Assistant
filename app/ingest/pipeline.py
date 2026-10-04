"""文档入库流水线 + 后台任务管理。

一次入库 = 解析 → 清洗 → 智能切片 → 自动去重 → 向量化 → 建索引，
全过程写入 ``ingest_tasks`` 表，前端可轮询进度；重启后仍能查到历史任务。
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from app.config import Settings
from app.db import repo
from app.db.base import session_scope
from app.ingest.cleaner import CleanReport, clean_text
from app.ingest.dedup import build_chunk_index, deduplicate_chunks, find_duplicate_document
from app.ingest.loader import LoadedDocument, load_from_path, load_from_text, load_from_url
from app.ingest.splitter import TextChunk, split_document
from app.rag.embeddings import BaseEmbedder
from app.rag.vector_store import VectorIndex
from app.text import sha1_hex, simhash64

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestOutcome:
    task_id: str
    document_id: str | None
    name: str
    status: str
    message: str
    chunk_count: int = 0
    char_count: int = 0
    dropped_chunks: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "document_id": self.document_id,
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "chunk_count": self.chunk_count,
            "char_count": self.char_count,
            "dropped_chunks": self.dropped_chunks,
        }


class IngestPipeline:
    """同步执行入库流程；异步调度交给 ``IngestManager``。"""

    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        vector_index: VectorIndex,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_index = vector_index
        self.on_change = on_change
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    def ingest_path(self, path: str | Path, *, name: str | None = None, task_id: str | None = None) -> IngestOutcome:
        task_id = task_id or str(uuid.uuid4())
        file_path = Path(path)
        self._update_task(
            task_id, name=name or file_path.name, status="running", stage="parsing", progress=0.05, message="正在解析文件…"
        )
        try:
            loaded = load_from_path(file_path, name=name)
        except Exception as exc:  # noqa: BLE001
            return self._fail(task_id, name or file_path.name, exc)
        return self._process(
            loaded,
            task_id,
            stored_path=str(file_path),
            file_size=file_path.stat().st_size if file_path.exists() else 0,
            source_type="file",
        )

    def _load_url(self, url: str, name: str | None) -> LoadedDocument:
        """统一的安全抓取参数：协议校验、内网拦截、体积与重定向限制。"""

        return load_from_url(
            url,
            timeout=self.settings.url_fetch_timeout,
            name=name,
            max_bytes=self.settings.url_fetch_max_bytes,
            max_redirects=self.settings.url_fetch_max_redirects,
            allow_private=self.settings.url_allow_private_hosts,
        )

    def ingest_url(self, url: str, *, name: str | None = None, task_id: str | None = None) -> IngestOutcome:
        task_id = task_id or str(uuid.uuid4())
        self._update_task(
            task_id, name=name or url, status="running", stage="parsing", progress=0.05, message="正在抓取网页…"
        )
        try:
            loaded = self._load_url(url, name)
        except Exception as exc:  # noqa: BLE001
            return self._fail(task_id, name or url, exc)
        return self._process(loaded, task_id, source_type="url", source_uri=url)

    def ingest_text(self, text: str, *, name: str = "手动输入文本", task_id: str | None = None) -> IngestOutcome:
        task_id = task_id or str(uuid.uuid4())
        self._update_task(
            task_id, name=name, status="running", stage="parsing", progress=0.05, message="正在处理文本…"
        )
        loaded = load_from_text(text, name)
        return self._process(loaded, task_id, source_type="text")

    def reindex(self, document_id: str, *, task_id: str | None = None) -> IngestOutcome:
        """重新解析已有文档（含重新计算向量）。"""
        task_id = task_id or str(uuid.uuid4())
        with session_scope() as db:
            document = repo.get_document(db, document_id)
            if document is None:
                return IngestOutcome(task_id, document_id, "", "failed", "文档不存在")
            source_type = document.source_type
            stored_path = document.stored_path
            source_uri = document.source_uri
            name = document.name
            file_size = document.file_size

        self._update_task(
            task_id,
            name=name,
            status="running",
            stage="parsing",
            progress=0.05,
            message="正在重新解析…",
            document_id=document_id,
        )
        try:
            if source_type == "file" and stored_path and Path(stored_path).exists():
                loaded = load_from_path(stored_path, name=name)
            elif source_type == "url" and source_uri:
                loaded = self._load_url(source_uri, name)
            else:
                # 手动输入 / 原始文件被删除：没有源文本可重新解析，改为只刷新向量索引
                return self._reembed(document_id, task_id, name)
        except Exception as exc:  # noqa: BLE001
            return self._fail(task_id, name, exc)

        with session_scope() as db:
            repo.delete_chunks_of_document(db, document_id)
        return self._process(
            loaded,
            task_id,
            stored_path=stored_path,
            file_size=file_size,
            source_type=source_type,
            source_uri=source_uri,
            existing_document_id=document_id,
        )

    def _reembed(self, document_id: str, task_id: str, name: str) -> IngestOutcome:
        """原始文本不可用时的降级方案：用当前 Embedding 模型重新计算已有片段的向量。"""
        with session_scope() as db:
            chunks = repo.list_chunks(db, document_id)
        if not chunks:
            return self._fail(task_id, name, RuntimeError("该文档没有可用的知识片段"))

        contents = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.id for chunk in chunks]
        batch = max(1, self.settings.embedding_batch_size)
        try:
            import numpy as np

            vectors = []
            for start in range(0, len(contents), batch):
                vectors.extend(self.embedder.embed_documents(contents[start : start + batch]))
                ratio = min(1.0, (start + batch) / max(len(contents), 1))
                self._update_task(
                    task_id,
                    stage="embedding",
                    progress=round(0.3 + 0.6 * ratio, 4),
                    message=f"正在刷新向量 {min(start + batch, len(contents))}/{len(contents)}…",
                    document_id=document_id,
                )
            self.vector_index.upsert(chunk_ids, np.asarray(vectors, dtype="float32"), [document_id] * len(chunk_ids))
        except Exception as exc:  # noqa: BLE001
            return self._fail(task_id, name, exc)

        self._update_document(document_id, status="indexed", error=None, chunk_count=len(chunk_ids))
        self._notify_change()
        message = f"已刷新向量索引：{len(chunk_ids)} 个片段（原文不可用，未重新解析）"
        self._update_task(
            task_id, status="succeeded", stage="finished", progress=1.0, message=message, document_id=document_id
        )
        return IngestOutcome(task_id, document_id, name, "succeeded", message, chunk_count=len(chunk_ids))

    # ------------------------------------------------------------------ #
    def _process(
        self,
        loaded: LoadedDocument,
        task_id: str,
        *,
        stored_path: str | None = None,
        file_size: int = 0,
        source_type: str = "file",
        source_uri: str | None = None,
        existing_document_id: str | None = None,
    ) -> IngestOutcome:
        name = loaded.name
        try:
            self._update_task(task_id, stage="cleaning", progress=0.18, message="正在清洗文本…")
            report: CleanReport = clean_text(loaded.text)
            if not report.text.strip():
                raise RuntimeError("清洗后没有有效文本内容")

            document_simhash = simhash64(report.text)
            document_id = existing_document_id
            if document_id is None:
                duplicate, similarity = self._find_duplicate(
                    loaded.content_hash, document_simhash, len(report.text)
                )
                if duplicate is not None and self.settings.duplicate_policy != "keep":
                    message = f"检测到重复文档，与《{duplicate.name}》相似度 {similarity:.0%}，已跳过入库"
                    self._update_task(
                        task_id,
                        status="skipped",
                        stage="finished",
                        progress=1.0,
                        message=message,
                        document_id=duplicate.id,
                    )
                    return IngestOutcome(task_id, duplicate.id, name, "skipped", message)
                document_id = self._create_document(
                    loaded, report, document_simhash, stored_path, file_size, source_type, source_uri
                )
            else:
                self._reset_document(existing_document_id, report, document_simhash, file_size)

            self._update_task(task_id, stage="splitting", progress=0.35, message="正在智能切片…", document_id=document_id)
            chunks = split_document(
                report.text,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
                min_chars=self.settings.min_chunk_chars,
                max_chars=self.settings.max_chunk_chars,
            )
            if not chunks:
                raise RuntimeError("切片结果为空，文档可能没有可索引的正文")

            self._update_task(task_id, stage="deduplicating", progress=0.45, message="正在去重…", document_id=document_id)
            kept, dropped = self._deduplicate(chunks)

            self._update_task(task_id, stage="embedding", progress=0.55, message="正在向量化…", document_id=document_id)
            chunk_ids = self._store_chunks(document_id, kept, task_id)

            self._update_task(task_id, stage="indexing", progress=0.92, message="正在建立索引…", document_id=document_id)
            self._update_document(document_id, chunk_count=len(chunk_ids), char_count=len(report.text), status="indexed")
            self._notify_change()

            message = (
                f"入库成功：{len(chunk_ids)} 个片段（自动去除 {len(dropped)} 个重复片段），"
                f"共 {len(report.text)} 字"
            )
            self._update_task(
                task_id, status="succeeded", stage="finished", progress=1.0, message=message, document_id=document_id
            )
            return IngestOutcome(
                task_id, document_id, name, "succeeded", message,
                chunk_count=len(chunk_ids), char_count=len(report.text), dropped_chunks=len(dropped),
            )
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响批量导入
            logger.exception("文档入库失败：%s", name)
            return self._fail(task_id, name, exc)

    def _find_duplicate(self, content_hash: str, simhash: str, text_length: int):
        with session_scope() as db:
            return find_duplicate_document(
                db,
                content_hash=content_hash,
                simhash=simhash,
                text_length=text_length,
                max_distance=self.settings.doc_dedup_distance,
                length_tolerance=self.settings.doc_dedup_length_tolerance,
            )

    def _create_document(
        self,
        loaded: LoadedDocument,
        report: CleanReport,
        document_simhash: str,
        stored_path: str | None,
        file_size: int,
        source_type: str,
        source_uri: str | None,
    ) -> str:
        with session_scope() as db:
            document = repo.create_document(
                db,
                name=loaded.name,
                source_type=source_type,
                source_uri=source_uri or loaded.source_uri,
                stored_path=stored_path,
                file_size=file_size,
                content_hash=loaded.content_hash,
                simhash=document_simhash,
                meta={
                    **loaded.meta,
                    "original_chars": report.original_chars,
                    "cleaned_chars": report.cleaned_chars,
                    "dropped_lines": report.dropped_lines,
                    "merged_lines": report.merged_lines,
                },
            )
            repo.update_document(db, document.id, status="parsing")
            return document.id

    def _reset_document(
        self, document_id: str, report: CleanReport, document_simhash: str, file_size: int
    ) -> None:
        with session_scope() as db:
            repo.update_document(
                db,
                document_id,
                status="parsing",
                error=None,
                content_hash=None,
                simhash=document_simhash,
                file_size=file_size or 0,
                char_count=0,
                chunk_count=0,
                duplicate_of=None,
            )

    def _deduplicate(self, chunks: Sequence[TextChunk]) -> tuple[list, list]:
        with session_scope() as db:
            index = build_chunk_index(db, max_distance=self.settings.doc_dedup_distance)
        return deduplicate_chunks(chunks, index, max_distance=self.settings.doc_dedup_distance)

    def _store_chunks(self, document_id: str, kept, task_id: str) -> list[str]:
        chunk_ids: list[str] = []
        with session_scope() as db:
            for ordinal, (chunk, digest) in enumerate(kept):
                record = repo.add_chunk(
                    db,
                    document_id=document_id,
                    ordinal=ordinal,
                    content=chunk.content,
                    content_hash=sha1_hex(chunk.content),
                    heading=chunk.heading,
                    start_offset=chunk.start,
                    end_offset=chunk.end,
                    simhash=digest,
                )
                chunk_ids.append(record.id)
            db.flush()

        if not chunk_ids:
            return []

        batch = max(1, self.settings.embedding_batch_size)
        contents = [chunk.content for chunk, _ in kept]
        vectors = []
        try:
            for start in range(0, len(contents), batch):
                vectors.extend(self.embedder.embed_documents(contents[start : start + batch]))
                ratio = min(1.0, (start + batch) / max(len(contents), 1))
                self._update_task(
                    task_id,
                    stage="embedding",
                    progress=round(0.55 + 0.35 * ratio, 4),
                    message=f"正在向量化 {min(start + batch, len(contents))}/{len(contents)} 个片段…",
                    document_id=document_id,
                )
            import numpy as np

            matrix = np.asarray(vectors, dtype="float32")
            self.vector_index.upsert(chunk_ids, matrix, [document_id] * len(chunk_ids))
        except Exception as exc:  # noqa: BLE001 - 向量化失败时保留片段，仅提示
            logger.exception("向量化失败：%s", exc)
            self._update_task(
                task_id,
                stage="embedding",
                progress=0.9,
                message=f"向量化失败，片段已入库但暂不可语义检索：{exc}",
                document_id=document_id,
            )
        return chunk_ids

    def _update_document(self, document_id: str, **fields) -> None:
        with session_scope() as db:
            repo.update_document(db, document_id, **fields)

    def _update_task(self, task_id: str, **fields) -> None:
        with session_scope() as db:
            if repo.get_task(db, task_id) is None:
                repo.create_task(db, name=fields.get("name", ""), document_id=fields.get("document_id"), task_id=task_id)
            repo.update_task(db, task_id, **fields)

    def _fail(self, task_id: str, name: str, error: Exception) -> IngestOutcome:
        message = f"处理失败：{error}"
        self._update_task(task_id, status="failed", stage="failed", progress=1.0, message=message)
        document_id = self._task_document(task_id)
        if document_id:
            self._update_document(document_id, status="failed", error=str(error))
        logger.warning("入库失败 [%s]：%s", name, error)
        return IngestOutcome(task_id, document_id, name, "failed", message)

    def _task_document(self, task_id: str) -> str | None:
        with session_scope() as db:
            task = repo.get_task(db, task_id)
            return task.document_id if task else None

    def _notify_change(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception as exc:  # noqa: BLE001
                logger.warning("索引失效通知失败：%s", exc)


class IngestManager:
    """后台线程池：上传接口立即返回 task_id，解析在后台进行。"""

    def __init__(self, settings: Settings, pipeline: IngestPipeline) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self._executor = ThreadPoolExecutor(max_workers=settings.ingest_workers, thread_name_prefix="ingest")
        self._futures: dict[str, object] = {}

    # ------------------------------------------------------------------ #
    def submit_path(self, path: str | Path, *, name: str | None = None) -> str:
        return self._submit(lambda task_id: self.pipeline.ingest_path(path, name=name, task_id=task_id), name or Path(path).name)

    def submit_url(self, url: str, *, name: str | None = None) -> str:
        return self._submit(lambda task_id: self.pipeline.ingest_url(url, name=name, task_id=task_id), name or url)

    def submit_text(self, text: str, *, name: str = "手动输入文本") -> str:
        return self._submit(lambda task_id: self.pipeline.ingest_text(text, name=name, task_id=task_id), name)

    def submit_reindex(self, document_id: str, *, name: str = "") -> str:
        return self._submit(lambda task_id: self.pipeline.reindex(document_id, task_id=task_id), name)

    def _submit(self, job: Callable[[str], IngestOutcome], name: str) -> str:
        task_id = str(uuid.uuid4())
        with session_scope() as db:
            repo.create_task(db, name=name, task_id=task_id)
        future = self._executor.submit(self._run, job, task_id, name)
        self._futures[task_id] = future
        return task_id

    def _run(self, job: Callable[[str], IngestOutcome], task_id: str, name: str) -> None:
        try:
            job(task_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("后台入库任务异常：%s", name)
            self.pipeline._fail(task_id, name, exc)
        finally:
            self._futures.pop(task_id, None)

    # ------------------------------------------------------------------ #
    def status(self, task_id: str) -> dict | None:
        with session_scope() as db:
            task = repo.get_task(db, task_id)
            if task is None:
                return None
            return {
                "task_id": task.id,
                "document_id": task.document_id,
                "name": task.name,
                "status": task.status,
                "stage": task.stage,
                "progress": round(task.progress, 4),
                "message": task.message,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "finished_at": task.finished_at.isoformat() if task.finished_at else None,
            }

    def tasks(self, limit: int = 30) -> list[dict]:
        with session_scope() as db:
            items = repo.list_tasks(db, limit=limit)
            return [
                {
                    "task_id": item.id,
                    "document_id": item.document_id,
                    "name": item.name,
                    "status": item.status,
                    "stage": item.stage,
                    "progress": round(item.progress, 4),
                    "message": item.message,
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                }
                for item in items
            ]

    def mark_interrupted(self) -> int:
        """服务启动时把上次未完成的任务标记为中断（重启不丢数据，但状态要如实）。"""
        with session_scope() as db:
            stale = repo.stale_tasks(db)
            for task in stale:
                repo.update_task(
                    db, task.id, status="failed", stage="interrupted", progress=1.0,
                    message="服务重启导致任务中断，请重新上传或重新解析",
                )
            return len(stale)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
