"""召回之后的精排（rerank）。

混合检索解决「别漏」，精排解决「排准」：BM25 + 向量融合只负责把候选捞回来，
最终的顺序交给这里。实现上刻意做成可插拔，并且**任何情况下都不引入硬依赖**：

- ``none``：不精排，保持召回顺序；
- ``lexical``：零依赖词面精排（短语命中 / 查询词覆盖 / 标题命中）；
- ``cross-encoder``：可选的句对重排模型（需 ``pip install sentence-transformers``），
  未安装或加载失败时自动降级为 ``lexical``，不会让服务起不来；
- ``auto``（默认）：能用重模型就用，否则用词面精排。
"""

from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

from app.config import Settings
from app.text import tokenize

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注，避免循环导入
    from app.rag.retriever import Evidence

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """精排器接口。"""

    name = "base"

    @abstractmethod
    def rerank(self, query: str, evidences: Sequence["Evidence"]) -> list["Evidence"]:
        ...

    def health(self) -> dict:
        return {"provider": self.name}


class NoopReranker(BaseReranker):
    """不做任何精排，用于对比 / 排障。"""

    name = "none"

    def rerank(self, query: str, evidences: Sequence["Evidence"]) -> list["Evidence"]:
        return list(evidences)


class LexicalReranker(BaseReranker):
    """零依赖词面精排。

    打分由三部分组成（都归一化到 0~1 后再与原召回分加权融合）：

    - **查询词覆盖**：按词长加权，因为中文里「响应时间」显然比「的」更有信息量；
    - **整串短语命中**：召回阶段是逐词匹配的，命中原句说明相关性很高；
    - **标题命中**：小标题通常就是这段内容的主题，命中应显著加分。
    """

    name = "lexical"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rerank(self, query: str, evidences: Sequence["Evidence"]) -> list["Evidence"]:
        if not evidences:
            return []
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return list(evidences)

        weight = self.settings.rerank_weight
        rescored = [
            replace(
                item,
                score=(1 - weight) * item.score
                + weight * self._lexical_score(query, item.content, item.heading, query_tokens),
            )
            for item in evidences
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored

    def _lexical_score(self, query: str, content: str, heading: str | None, query_tokens: set[str]) -> float:
        content_tokens = set(tokenize(content))
        total = sum(self._weight(term) for term in query_tokens) or 1.0
        coverage = sum(self._weight(term) for term in query_tokens if term in content_tokens) / total

        phrase = " ".join(query.split())
        phrase_bonus = 0.15 if len(phrase) >= 4 and phrase in content else 0.0

        heading_bonus = 0.0
        if heading:
            heading_tokens = set(tokenize(heading))
            if heading_tokens:
                hit = sum(self._weight(term) for term in query_tokens if term in heading_tokens)
                heading_bonus = 0.2 * (hit / total)

        return min(1.0, coverage + phrase_bonus + heading_bonus)

    @staticmethod
    def _weight(term: str) -> float:
        """词越长信息量越大；单个汉字权重下限为 1。"""

        return max(1.0, float(len(term)))


class CrossEncoderReranker(BaseReranker):
    """句对重排模型（可选依赖）。"""

    name = "cross-encoder"

    def __init__(self, settings: Settings, model=None) -> None:
        self.settings = settings
        self.model = model if model is not None else self._load_model(settings.rerank_model)

    @staticmethod
    def _load_model(name: str):
        from sentence_transformers import CrossEncoder  # 延迟导入，避免影响启动速度

        return CrossEncoder(name)

    def rerank(self, query: str, evidences: Sequence["Evidence"]) -> list["Evidence"]:
        if not evidences:
            return []
        if not query.strip():
            return list(evidences)

        pairs = [(query, item.content) for item in evidences]
        raw = self.model.predict(pairs)
        weight = self.settings.rerank_weight
        rescored = [
            replace(item, score=(1 - weight) * item.score + weight * _sigmoid(float(value)))
            for item, value in zip(evidences, raw, strict=False)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + pow(2.718281828459045, -value))
    factor = pow(2.718281828459045, value)
    return factor / (1.0 + factor)


@lru_cache(maxsize=1)
def cross_encoder_available() -> bool:
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):  # pragma: no cover - 环境异常时按不可用处理
        return False


def build_reranker(settings: Settings) -> BaseReranker:
    """按配置构建精排器，任何异常都退化成可用的实现。"""

    provider = (settings.rerank_provider or "auto").strip().lower()
    if provider in {"none", "off", "false", "0"}:
        return NoopReranker()

    if provider in {"cross-encoder", "cross_encoder", "auto"}:
        if cross_encoder_available():
            try:
                return CrossEncoderReranker(settings)
            except Exception as exc:  # noqa: BLE001 - 模型下载失败不能拖垮检索
                logger.warning("精排模型加载失败，降级为词面精排：%s", exc)
        elif provider != "auto":
            logger.warning("未安装 sentence-transformers，已降级为词面精排")

    return LexicalReranker(settings)
