"""数据库引擎与会话管理。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None

# create_all 只会建新表，不会给已存在的表加列；这里维护一份最小的「加列」清单，
# 让老版本升级上来时不用手工改库。
_LIGHT_MIGRATIONS: dict[str, dict[str, str]] = {
    "qa_logs": {"feedback": "VARCHAR(16)", "reflection_rounds": "INTEGER"},
}


def _apply_light_migrations(engine: Engine) -> None:
    from sqlalchemy import inspect

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, columns in _LIGHT_MIGRATIONS.items():
            if not inspector.has_table(table):
                continue
            existing = {column["name"] for column in inspector.get_columns(table)}
            for name, ddl in columns.items():
                if name in existing:
                    continue
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                logger.info("数据库轻量迁移：%s 增加列 %s", table, name)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.ensure_dirs()
        _engine = create_engine(
            settings.database_url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        _install_sqlite_pragmas(_engine)
    return _engine


def _install_sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _record):  # pragma: no cover - 驱动回调
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False, future=True
        )
    return _SessionLocal


def init_db() -> None:
    """创建全部表（幂等）。"""
    settings = get_settings()
    settings.ensure_dirs()
    Base.metadata.create_all(bind=get_engine())
    _apply_light_migrations(get_engine())
    # 触发一次写入，确保 WAL 文件就绪
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.debug("数据库初始化完成：%s", settings.db_path)


@contextmanager
def session_scope() -> Iterator[Session]:
    """事务性会话上下文：正常提交，异常回滚。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """测试用：释放引擎与连接池。"""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
