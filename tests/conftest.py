"""测试配置。

注意：环境变量必须在导入 app 包之前设置（load_dotenv 使用 setdefault，不会覆盖已有变量），
因此这里在模块顶层就把数据目录指向临时目录，保证测试不会污染真实知识库。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="ragagent-test-"))
os.environ["DATA_DIR"] = str(_TMP)
os.environ["DB_PATH"] = str(_TMP / "knowledge.db")
os.environ["CHROMA_DIR"] = str(_TMP / "chroma")
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["AGENT_LLM_PLANNER"] = "false"
os.environ["VECTOR_BACKEND"] = "sqlite"
os.environ["RETRIEVAL_TOP_K"] = "5"
os.environ["MIN_RELEVANCE_SCORE"] = "0.2"


@pytest.fixture(scope="session")
def settings():
    from app.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def services(settings):
    from app.services import init_services, shutdown_services

    service = init_services(settings, force=True)
    yield service
    shutdown_services()


@pytest.fixture(scope="session")
def sample_document() -> str:
    return """# RAG 系统设计说明

## 1. 系统目标
本系统面向个人知识库场景，目标是在完全私有化的前提下提供可溯源的问答能力。
系统不上传任何原始文档到公网，所有检索与向量计算都在本地完成。

## 2. 核心模块
- 文档导入模块：支持 TXT、Markdown、PDF、Word 与网页链接。
- 切片模块：按标题层级与段落语义自适应切块，长段落按句子边界继续切分。
- 检索模块：BM25 关键词召回与向量语义召回加权融合，再做 MMR 重排。
- Agent 模块：先判断意图，再决定是否需要拆解为多个子问题。

## 3. 性能指标
单次问答响应时间要求控制在 2 秒以内，千级知识片段检索不能出现卡顿。
文档解析过程需要提供可视化进度，异常文件自动跳过并提示用户。

## 4. 存储设计
知识片段与向量统一存放在 SQLite 中，默认使用暴力内积检索；
当片段规模超过一万时，可以切换到 Chroma 使用 HNSW 索引加速。
"""


@pytest.fixture(scope="session")
def ingest_sample(services, sample_document):
    """把示例文档灌进测试知识库，返回 document_id。"""
    outcome = services.pipeline.ingest_text(sample_document, name="RAG系统设计说明")
    assert outcome.status == "succeeded", outcome.message
    return outcome.document_id


def pytest_sessionfinish(session, exitstatus):  # pragma: no cover - 清理临时目录
    shutil.rmtree(_TMP, ignore_errors=True)
