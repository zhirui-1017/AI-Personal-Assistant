"""回答反馈接口与数据库轻量迁移的测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from app.db.base import _apply_light_migrations
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_feedback_records_rating(client):
    session = client.post("/api/sessions", json={"title": "反馈会话"}).json()
    session_id = session["id"]

    client.post("/api/chat", json={"question": "这个知识库支持哪些格式？", "session_id": session_id})

    response = client.post("/api/chat/feedback", json={"session_id": session_id, "rating": "up"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["rating"] == "up"
    assert isinstance(payload["qa_log_id"], str)

    assert client.delete(f"/api/sessions/{session_id}").json()["ok"] is True


def test_feedback_unknown_session_returns_404(client):
    response = client.post("/api/chat/feedback", json={"session_id": "不存在的会话", "rating": "down"})
    assert response.status_code == 404


def test_feedback_rejects_invalid_rating(client):
    response = client.post("/api/chat/feedback", json={"session_id": "any", "rating": "maybe"})
    assert response.status_code == 422


def test_light_migration_adds_missing_column(tmp_path):
    """老库（qa_logs 没有 feedback 列）升级后应自动补列，且重复执行不报错。"""

    db_path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    try:
        with engine.begin() as conn:
            conn.execute(
                text("CREATE TABLE qa_logs (id INTEGER PRIMARY KEY, session_id VARCHAR(64), question TEXT)")
            )
            conn.execute(text("INSERT INTO qa_logs (session_id, question) VALUES ('s1', 'q1')"))

        _apply_light_migrations(engine)

        columns = {column["name"] for column in inspect(engine).get_columns("qa_logs")}
        assert "feedback" in columns
        assert "reflection_rounds" in columns

        _apply_light_migrations(engine)  # 幂等

        with engine.connect() as conn:
            row = conn.execute(text("SELECT session_id, question, feedback FROM qa_logs")).one()
        assert row.feedback is None
    finally:
        engine.dispose()


def test_light_migration_skips_absent_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}", future=True)
    try:
        _apply_light_migrations(engine)
        assert not inspect(engine).has_table("qa_logs")
    finally:
        engine.dispose()
