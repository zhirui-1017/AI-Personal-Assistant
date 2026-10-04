"""向量化层：可插拔的 Embedding 提供方。

- ``hash``  ：内置离线向量（哈希技巧 + 字符 n-gram），零依赖零下载，开箱即用
- ``api``   ：OpenAI 兼容的 /embeddings 接口（通义、智谱、OpenAI 等）
- ``sentence-transformers``：本地开源模型（如 BAAI/bge-small-zh-v1.5）
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from typing import Sequence

import numpy as np

from app.config import Settings, get_settings
from app.text import char_ngrams, tokenize

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
def vector_to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def bytes_to_vector(blob: bytes | None, dim: int | None = None) -> np.ndarray | None:
    if not blob:
        return None
    vector = np.frombuffer(blob, dtype=np.float32)
    if dim and vector.size != dim:
        return None
    return vector


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class BaseEmbedder(ABC):
    """Embedding 抽象基类。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.asarray(self._encode(list(texts)), dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        self.dim = matrix.shape[1]
        return _normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class CachedEmbedder(BaseEmbedder):
    """给任意 Embedder 加一层查询缓存（同一问题不重复计算）。"""

    def __init__(self, inner: BaseEmbedder, max_size: int = 2048) -> None:
        self._inner = inner
        self._max_size = max(0, max_size)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.name = inner.name
        self.dim = inner.dim

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._inner._encode(texts)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self._inner.embed_documents(texts)
        self.dim = self._inner.dim
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        if not self._max_size:
            return self._inner.embed_query(text)
        key = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        vector = self._inner.embed_query(text)
        with self._lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        return vector


class HashEmbedder(BaseEmbedder):
    """离线哈希向量：词 + 字符 n-gram 双通道哈希，无需任何模型下载。

    说明：它属于「词法向量」，语义泛化能力弱于神经网络模型，
    但配合混合检索（BM25 + 向量）在个人知识库场景下完全可用，
    且保证项目在没有网络 / 没有 API Key 的机器上也能跑通全流程。
    """

    name = "hash"

    def __init__(self, dim: int = 512, word_weight: float = 1.0, ngram_weight: float = 0.6) -> None:
        self.dim = dim
        self.word_weight = word_weight
        self.ngram_weight = ngram_weight

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counters: Counter[str] = Counter()
            for token in tokenize(text, keep_stopwords=False):
                counters[f"w:{token}"] += self.word_weight
            for ngram in char_ngrams(text, 2, 3):
                if ngram.strip():
                    counters[f"c:{ngram}"] += self.ngram_weight
            for feature, weight in counters.items():
                digest = hashlib.md5(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign * (1.0 + math.log(weight))
        return matrix


class ApiEmbedder(BaseEmbedder):
    """OpenAI 兼容接口向量化。"""

    name = "api"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        dim: int = 0,
        batch_size: int = 32,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.max_retries = max_retries

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                payload = {"model": self.model, "input": batch, "encoding_format": "float"}
                last_error: Exception | None = None
                for attempt in range(self.max_retries + 1):
                    try:
                        response = client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
                        response.raise_for_status()
                        data = response.json()["data"]
                        vectors.extend(item["embedding"] for item in sorted(data, key=lambda x: x.get("index", 0)))
                        last_error = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        logger.warning("Embedding 接口调用失败（第 %s 次）：%s", attempt + 1, exc)
                if last_error is not None:
                    raise RuntimeError(f"Embedding 接口调用失败：{last_error}") from last_error
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.size == 0:
            raise RuntimeError("Embedding 接口未返回向量")
        return matrix


class SentenceTransformerEmbedder(BaseEmbedder):
    """本地开源向量模型（首次使用会下载权重）。"""

    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


# --------------------------------------------------------------------------- #
_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_CACHE: dict[str, BaseEmbedder] = {}


def build_embedder(settings: Settings) -> BaseEmbedder:
    provider = settings.resolved_embedding_provider
    if provider == "api":
        api_key = settings.embedding_api_key or settings.llm_api_key
        base_url = settings.embedding_base_url or settings.llm_base_url
        if not api_key:
            logger.warning("未配置 Embedding API Key，自动退回内置 hash 向量")
            return HashEmbedder(dim=settings.embedding_dim)
        return ApiEmbedder(
            model=settings.embedding_model,
            api_key=api_key,
            base_url=base_url,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
    if provider in {"sentence-transformers", "st", "local"}:
        try:
            return SentenceTransformerEmbedder(settings.st_model)
        except Exception as exc:  # noqa: BLE001 - 缺少依赖或下载失败时兜底
            logger.warning("本地向量模型加载失败（%s），自动退回内置 hash 向量", exc)
            return HashEmbedder(dim=settings.embedding_dim)
    return HashEmbedder(dim=settings.embedding_dim)


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    settings = settings or get_settings()
    key = f"{settings.resolved_embedding_provider}:{settings.embedding_model}:{settings.embedding_dim}:{settings.st_model}"
    with _EMBEDDER_LOCK:
        embedder = _EMBEDDER_CACHE.get(key)
        if embedder is None:
            embedder = CachedEmbedder(build_embedder(settings), settings.embedding_cache_size)
            _EMBEDDER_CACHE[key] = embedder
        return embedder


def reset_embedder_cache() -> None:
    with _EMBEDDER_LOCK:
        _EMBEDDER_CACHE.clear()
