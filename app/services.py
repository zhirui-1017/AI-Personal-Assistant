"""服务容器：把各层组件装配成单一入口，供 API / CLI / Gradio 复用。"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.agent.executor import AgentExecutor
from app.agent.planner import Planner
from app.agent.tools import AgentTools
from app.config import Settings, get_settings
from app.db.base import init_db
from app.ingest.pipeline import IngestManager, IngestPipeline
from app.llm.client import BaseLLM, build_llm
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.rag.embeddings import BaseEmbedder, get_embedder
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorIndex, build_vector_index

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Services:
    settings: Settings
    llm: BaseLLM
    embedder: BaseEmbedder
    vector_index: VectorIndex
    retriever: Retriever
    tools: AgentTools
    planner: Planner
    short_term: ShortTermMemory
    long_term: LongTermMemory
    executor: AgentExecutor
    pipeline: IngestPipeline
    ingest: IngestManager

    def health(self) -> dict:
        return {
            "llm": self.llm.health(),
            "embedding": {"provider": self.embedder.name, "dim": self.embedder.dim},
            "vector_store": self.vector_index.health(),
            "tools": self.tools.health(),
        }


_services: Services | None = None
_lock = threading.Lock()


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    init_db()

    llm = build_llm(settings)
    embedder = get_embedder(settings)
    vector_index = build_vector_index(settings)
    retriever = Retriever(settings=settings, embedder=embedder, vector_index=vector_index)
    tools = AgentTools(settings, retriever)
    planner = Planner(settings, llm)
    short_term = ShortTermMemory(settings, llm)
    long_term = LongTermMemory(settings, llm, embedder)
    executor = AgentExecutor(settings, llm, tools, planner, short_term, long_term)
    pipeline = IngestPipeline(settings, embedder, vector_index, on_change=retriever.invalidate)
    ingest = IngestManager(settings, pipeline)

    interrupted = ingest.mark_interrupted()
    if interrupted:
        logger.warning("检测到 %s 个上次未完成的入库任务，已标记为中断", interrupted)

    return Services(
        settings=settings,
        llm=llm,
        embedder=embedder,
        vector_index=vector_index,
        retriever=retriever,
        tools=tools,
        planner=planner,
        short_term=short_term,
        long_term=long_term,
        executor=executor,
        pipeline=pipeline,
        ingest=ingest,
    )


def init_services(settings: Settings | None = None, *, force: bool = False) -> Services:
    global _services
    with _lock:
        if _services is None or force:
            if _services is not None:
                _services.ingest.shutdown()
            _services = build_services(settings)
        return _services


def get_services() -> Services:
    return init_services()


def shutdown_services() -> None:
    global _services
    with _lock:
        if _services is not None:
            _services.ingest.shutdown()
            try:
                _services.llm.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响退出
                logger.warning("关闭 LLM 客户端失败", exc_info=True)
            _services = None


def reset_services() -> None:
    """测试用：重建服务容器。"""
    shutdown_services()
