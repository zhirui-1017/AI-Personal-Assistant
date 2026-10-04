from app.db import repo
from app.db.base import session_scope


def _document(title: str, sections: list[tuple[str, str]]) -> str:
    body = "\n\n".join(f"## {index}. {heading}\n\n{content}" for index, (heading, content) in enumerate(sections, 1))
    return f"# {title}\n\n{body}\n"


PIPELINE_DOC = _document(
    "入库流水线测试文档",
    [
        ("解析阶段", "解析阶段负责把 PDF、Word、网页等格式统一转换成纯文本，并记录来源信息。"),
        ("清洗阶段", "清洗阶段会删除页眉页脚、页码、零宽字符，并修复被硬换行截断的句子。"),
        ("切片阶段", "切片阶段按照标题层级与段落边界切块，同时保留字符偏移量以便溯源定位。"),
    ],
)

REINDEX_DOC = _document(
    "重新解析功能说明",
    [
        ("功能目标", "重新解析用于在清洗规则或切片参数调整后，重新生成文档的知识片段。"),
        ("实现方式", "重新解析会先删除旧的片段与向量，再按当前配置重新走一遍完整流水线。"),
    ],
)

PROGRESS_DOC = _document(
    "任务进度可视化说明",
    [
        ("进度来源", "每个入库任务都会在数据库中记录阶段、百分比与提示信息，供前端轮询展示。"),
        ("异常处理", "单个文件解析失败不会中断批量导入，失败原因会写入任务记录并提示用户。"),
    ],
)


def test_ingest_text_creates_document_and_chunks(services):
    outcome = services.pipeline.ingest_text(PIPELINE_DOC, name="流水线测试文档")
    assert outcome.status == "succeeded"
    assert outcome.chunk_count > 0

    with session_scope() as db:
        document = repo.get_document(db, outcome.document_id)
        assert document is not None
        assert document.status == "indexed"
        assert document.chunk_count == outcome.chunk_count
        assert document.char_count > 0
        chunks = repo.list_chunks(db, document.id)
        assert len(chunks) == document.chunk_count
        for chunk in chunks:
            assert chunk.embedding is not None
            assert chunk.char_count == len(chunk.content)
            assert chunk.heading


def test_ingest_duplicate_is_skipped(services):
    first = services.pipeline.ingest_text(REINDEX_DOC, name="重复检测 A")
    second = services.pipeline.ingest_text(REINDEX_DOC, name="重复检测 B")
    assert first.status == "succeeded"
    assert second.status == "skipped"
    assert "重复" in second.message


def test_ingest_unsupported_file_fails(services, tmp_path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04 not a document")
    outcome = services.pipeline.ingest_path(path)
    assert outcome.status == "failed"
    assert "失败" in outcome.message


def test_reindex_document(services):
    outcome = services.pipeline.ingest_text(PROGRESS_DOC, name="重新解析测试")
    again = services.pipeline.reindex(outcome.document_id)
    assert again.status == "succeeded"
    assert again.chunk_count > 0

    with session_scope() as db:
        document = repo.get_document(db, outcome.document_id)
        assert document.status == "indexed"
        assert repo.count_chunks(db, outcome.document_id) == document.chunk_count


def test_duplicate_policy_keep(services, settings, monkeypatch):
    monkeypatch.setattr(settings, "duplicate_policy", "keep")
    first = services.pipeline.ingest_text(PIPELINE_DOC, name="保留策略 A")
    assert first.status == "succeeded"
    monkeypatch.setattr(settings, "duplicate_policy", "skip")


def test_task_progress_recorded(services):
    outcome = services.pipeline.ingest_text(PROGRESS_DOC, name="进度测试")
    status = services.ingest.status(outcome.task_id)
    assert status is not None
    assert status["status"] in {"succeeded", "skipped"}
    assert status["progress"] == 1.0
    assert status["name"] == "进度测试"


def test_chunk_offsets_match_cleaned_text(services):
    from app.ingest.cleaner import clean_text
    from app.ingest.splitter import split_document

    raw = "# 标题\n\n第一段内容，用于验证偏移量是否正确。\n\n第二段内容，同样用于验证。\n"
    cleaned = clean_text(raw).text
    chunks = split_document(cleaned, chunk_size=100, overlap=10, min_chars=5, max_chars=400)
    assert chunks
    for chunk in chunks:
        assert cleaned[chunk.start : chunk.end] == chunk.content
