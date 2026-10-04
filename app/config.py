"""全局配置。

所有配置项都可以通过环境变量或项目根目录下的 ``.env`` 覆盖，
这里不依赖 pydantic-settings 等第三方库，保证最小环境也能启动。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """把 .env 里的键值对写入 os.environ（已存在的变量优先，不被覆盖）。"""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(key.strip(), value.strip())


def _str(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or not value.strip() else value.strip()


def _int(key: str, default: int, low: int | None = None, high: int | None = None) -> int:
    try:
        value = int(_str(key, str(default)))
    except ValueError:
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _float(key: str, default: float, low: float | None = None, high: float | None = None) -> float:
    try:
        value = float(_str(key, str(default)))
    except ValueError:
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _bool(key: str, default: bool) -> bool:
    return _str(key, "true" if default else "false").lower() in {"1", "true", "yes", "y", "on"}


def _path(key: str, default: Path) -> Path:
    raw = _str(key, str(default))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(slots=True)
class Settings:
    """运行期配置。"""

    # ---------- 路径 ----------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", PROJECT_ROOT / "data"))
    upload_dir: Path = field(default_factory=lambda: _path("UPLOAD_DIR", _path("DATA_DIR", PROJECT_ROOT / "data") / "uploads"))
    db_path: Path = field(default_factory=lambda: _path("DB_PATH", _path("DATA_DIR", PROJECT_ROOT / "data") / "knowledge.db"))
    chroma_dir: Path = field(default_factory=lambda: _path("CHROMA_DIR", _path("DATA_DIR", PROJECT_ROOT / "data") / "chroma"))

    # ---------- LLM ----------
    llm_provider: str = field(default_factory=lambda: _str("LLM_PROVIDER", "auto").lower())
    llm_base_url: str = field(default_factory=lambda: _str("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_api_key: str = field(default_factory=lambda: _str("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _str("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.2, 0.0, 2.0))
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 1536, 128, 32768))
    llm_timeout: float = field(default_factory=lambda: _float("LLM_TIMEOUT", 60.0, 5.0, 600.0))
    llm_max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 2, 0, 5))

    # ---------- Embedding ----------
    embedding_provider: str = field(default_factory=lambda: _str("EMBEDDING_PROVIDER", "auto").lower())
    embedding_model: str = field(default_factory=lambda: _str("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_api_key: str = field(default_factory=lambda: _str("EMBEDDING_API_KEY", ""))
    embedding_base_url: str = field(default_factory=lambda: _str("EMBEDDING_BASE_URL", ""))
    embedding_dim: int = field(default_factory=lambda: _int("EMBEDDING_DIM", 512, 64, 8192))
    embedding_batch_size: int = field(default_factory=lambda: _int("EMBEDDING_BATCH_SIZE", 32, 1, 256))
    st_model: str = field(default_factory=lambda: _str("ST_MODEL", "BAAI/bge-small-zh-v1.5"))
    embedding_cache_size: int = field(default_factory=lambda: _int("EMBEDDING_CACHE_SIZE", 4096, 0, 100000))

    # ---------- 向量库 ----------
    vector_backend: str = field(default_factory=lambda: _str("VECTOR_BACKEND", "sqlite").lower())
    chroma_collection: str = field(default_factory=lambda: _str("CHROMA_COLLECTION", "personal_kb"))

    # ---------- 切片 ----------
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 600, 120, 4000))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 80, 0, 1000))
    min_chunk_chars: int = field(default_factory=lambda: _int("MIN_CHUNK_CHARS", 40, 1, 1000))
    max_chunk_chars: int = field(default_factory=lambda: _int("MAX_CHUNK_CHARS", 1600, 200, 8000))

    # ---------- 检索 ----------
    retrieval_top_k: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K", 6, 1, 50))
    retrieval_candidate_k: int = field(default_factory=lambda: _int("RETRIEVAL_CANDIDATE_K", 30, 2, 200))
    semantic_weight: float = field(default_factory=lambda: _float("SEMANTIC_WEIGHT", 0.6, 0.0, 1.0))
    keyword_weight: float = field(default_factory=lambda: _float("KEYWORD_WEIGHT", 0.4, 0.0, 1.0))
    min_relevance_score: float = field(default_factory=lambda: _float("MIN_RELEVANCE_SCORE", 0.18, 0.0, 1.0))
    mmr_lambda: float = field(default_factory=lambda: _float("MMR_LAMBDA", 0.7, 0.0, 1.0))
    # 精排：召回后的二次排序。none=关闭，lexical=内置词面精排，cross-encoder=可选重模型，auto=可用则用重模型
    rerank_provider: str = field(default_factory=lambda: _str("RERANK_PROVIDER", "auto").lower())
    rerank_top_n: int = field(default_factory=lambda: _int("RERANK_TOP_N", 20, 1, 200))
    rerank_model: str = field(default_factory=lambda: _str("RERANK_MODEL", "BAAI/bge-reranker-base"))
    rerank_weight: float = field(default_factory=lambda: _float("RERANK_WEIGHT", 0.4, 0.0, 1.0))
    chunk_dedup_threshold: float = field(default_factory=lambda: _float("CHUNK_DEDUP_THRESHOLD", 0.96, 0.5, 1.0))
    doc_dedup_distance: int = field(default_factory=lambda: _int("DOC_DEDUP_DISTANCE", 3, 0, 32))
    # 文档级去重还要求长度接近，避免「同主题但内容有更新」的文档被误判为重复
    doc_dedup_length_tolerance: float = field(
        default_factory=lambda: _float("DOC_DEDUP_LENGTH_TOLERANCE", 0.02, 0.0, 1.0)
    )

    # ---------- Agent ----------
    agent_max_sub_questions: int = field(default_factory=lambda: _int("AGENT_MAX_SUB_QUESTIONS", 4, 1, 10))
    agent_max_rounds: int = field(default_factory=lambda: _int("AGENT_MAX_ROUNDS", 2, 1, 6))
    agent_llm_planner: bool = field(default_factory=lambda: _bool("AGENT_LLM_PLANNER", True))
    agent_strict_citation: bool = field(default_factory=lambda: _bool("AGENT_STRICT_CITATION", True))

    # ---------- 自我反思闭环 ----------
    # 生成答案后再自检一次：发现幻觉风险或信息缺口就改写 Query 补充检索并重新生成
    reflection_enabled: bool = field(default_factory=lambda: _bool("REFLECTION_ENABLED", True))
    reflection_max_rounds: int = field(default_factory=lambda: _int("REFLECTION_MAX_ROUNDS", 2, 0, 5))
    # 引用依据度低于该值即认为存在幻觉风险
    reflection_min_grounded: float = field(
        default_factory=lambda: _float("REFLECTION_MIN_GROUNDED", 0.8, 0.0, 1.0)
    )
    # 问题信息点被检索覆盖的最低比例，低于该值说明检索没打中要点
    reflection_min_coverage: float = field(
        default_factory=lambda: _float("REFLECTION_MIN_COVERAGE", 0.6, 0.0, 1.0)
    )
    reflection_max_queries: int = field(default_factory=lambda: _int("REFLECTION_MAX_QUERIES", 3, 1, 6))
    # 反思环节的时间预算，超出后立即停止，防止"无限检索"
    reflection_time_budget_ms: int = field(
        default_factory=lambda: _int("REFLECTION_TIME_BUDGET_MS", 6000, 1000, 60000)
    )

    # ---------- 记忆 ----------
    short_term_turns: int = field(default_factory=lambda: _int("SHORT_TERM_TURNS", 6, 1, 50))
    summary_trigger_messages: int = field(default_factory=lambda: _int("SUMMARY_TRIGGER_MESSAGES", 12, 4, 200))
    memory_extract_enabled: bool = field(default_factory=lambda: _bool("MEMORY_EXTRACT_ENABLED", True))
    long_term_max_items: int = field(default_factory=lambda: _int("LONG_TERM_MAX_ITEMS", 60, 5, 1000))
    long_term_min_importance: float = field(default_factory=lambda: _float("LONG_TERM_MIN_IMPORTANCE", 0.35, 0.0, 1.0))
    long_term_merge_threshold: float = field(default_factory=lambda: _float("LONG_TERM_MERGE_THRESHOLD", 0.9, 0.5, 1.0))

    # ---------- 入库 ----------
    ingest_workers: int = field(default_factory=lambda: _int("INGEST_WORKERS", 2, 1, 8))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 50, 1, 2048))
    url_fetch_timeout: float = field(default_factory=lambda: _float("URL_FETCH_TIMEOUT", 30.0, 3.0, 120.0))
    url_fetch_max_bytes: int = field(default_factory=lambda: _int("URL_FETCH_MAX_BYTES", 10485760, 65536, 536870912))
    url_fetch_max_redirects: int = field(default_factory=lambda: _int("URL_FETCH_MAX_REDIRECTS", 3, 0, 10))
    # 默认拒绝抓取内网 / 回环地址，避免 SSRF；确有需要（如内网 wiki）再显式打开
    url_allow_private_hosts: bool = field(default_factory=lambda: _bool("URL_ALLOW_PRIVATE_HOSTS", False))
    duplicate_policy: str = field(default_factory=lambda: _str("DUPLICATE_POLICY", "skip").lower())

    # ---------- 服务 ----------
    api_host: str = field(default_factory=lambda: _str("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _int("API_PORT", 8000, 1, 65535))
    api_cors_origins: str = field(default_factory=lambda: _str("API_CORS_ORIGINS", "*"))
    api_token: str = field(default_factory=lambda: _str("API_TOKEN", ""))
    default_user_id: str = field(default_factory=lambda: _str("DEFAULT_USER_ID", "local-user"))
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO").upper())

    # ------------------------------------------------------------------
    @property
    def auth_enabled(self) -> bool:
        """配置了 API_TOKEN 才开启接口鉴权（本地自用默认关闭）。"""
        return bool(self.api_token.strip())

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".txt", ".md", ".markdown", ".pdf", ".docx", ".doc", ".html", ".htm")

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider == "auto":
            return "openai" if self.llm_api_key else "mock"
        return self.llm_provider

    @property
    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider == "auto":
            return "api" if (self.embedding_api_key or self.embedding_base_url) else "hash"
        return self.embedding_provider

    @property
    def database_url(self) -> str:
        return f"sqlite+pysqlite:///{self.db_path.as_posix()}"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.db_path.parent, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, object]:
        """用于 /api/health 展示当前生效配置（隐藏密钥）。"""
        return {
            "llm_provider": self.resolved_llm_provider,
            "llm_model": self.llm_model if self.resolved_llm_provider != "mock" else "mock-extractive",
            "embedding_provider": self.resolved_embedding_provider,
            "embedding_dim": self.embedding_dim,
            "vector_backend": self.vector_backend,
            "chunk_size": self.chunk_size,
            "retrieval_top_k": self.retrieval_top_k,
            "rerank_provider": self.rerank_provider,
            "reflection_enabled": self.reflection_enabled,
            "auth_enabled": self.auth_enabled,
            "data_dir": str(self.data_dir),
            "db_path": str(self.db_path),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    settings = Settings()
    settings.ensure_dirs()
    return settings


def reset_settings_cache() -> None:
    """测试用：清空配置缓存。"""
    get_settings.cache_clear()
