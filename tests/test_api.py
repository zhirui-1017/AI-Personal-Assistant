import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _wait_task(client: TestClient, task_id: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        status = client.get(f"/api/documents/tasks/{task_id}").json()
        if status.get("status") in {"succeeded", "failed", "skipped"}:
            return status
        time.sleep(0.15)
    return status


def test_health(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["config"]["llm_provider"] == "mock"
    assert payload["components"]["vector_store"]["backend"] == "sqlite"


def test_cors_wildcard_does_not_send_credentials(client):
    """通配来源不能与携带凭据同时生效，否则任意站点都能带着 Cookie 调接口。"""

    response = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert response.headers.get("access-control-allow-origin") == "*"
    assert response.headers.get("access-control-allow-credentials") is None


def test_ingest_text_and_list_documents(client):
    response = client.post(
        "/api/documents/text",
        json={"text": "# 接口测试文档\n\nFastAPI 接口层负责接收请求并调度 Agent。\n\n## 接口列表\n- /api/chat\n- /api/documents\n", "name": "接口测试文档"},
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    status = _wait_task(client, task_id)
    assert status["status"] == "succeeded", status

    documents = client.get("/api/documents").json()
    assert documents["total"] >= 1
    assert any(item["name"] == "接口测试文档" for item in documents["items"])


def test_upload_rejects_unsupported_file(client):
    response = client.post(
        "/api/documents/upload", files={"files": ("bad.exe", b"MZ binary", "application/octet-stream")}
    )
    assert response.status_code == 200
    assert response.json()["items"][0]["status"] == "rejected"


def test_chat_endpoint(client):
    response = client.post("/api/chat", json={"question": "接口层负责什么？"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["session_id"]
    assert payload["plan"]["intent"]


def test_chat_stream_endpoint(client):
    with client.stream("POST", "/api/chat/stream", json={"question": "接口层负责什么？"}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert '"type": "plan"' in body
    assert '"type": "final"' in body


def test_chat_with_document_scope(client):
    """document_ids 要能一路透传到检索层：引用只能来自指定文档。"""

    created = client.post(
        "/api/documents/text",
        json={"text": "# 范围限定测试\n\n本项目采用本地私有化部署，不依赖任何外部服务。", "name": "范围限定测试"},
    ).json()
    status = _wait_task(client, created["task_id"])
    assert status["status"] == "succeeded"
    document_id = status["document_id"]
    try:
        payload = client.post(
            "/api/chat", json={"question": "本项目采用什么部署方式？", "document_ids": [document_id]}
        ).json()
        assert payload["citations"]
        assert {item["document_id"] for item in payload["citations"]} == {document_id}
    finally:
        assert client.delete(f"/api/documents/{document_id}").json()["ok"] is True


def test_session_lifecycle(client):
    session = client.post("/api/sessions", json={"title": "接口会话"}).json()
    session_id = session["id"]

    client.post("/api/chat", json={"question": "接口层负责什么？", "session_id": session_id})
    messages = client.get(f"/api/sessions/{session_id}/messages").json()
    assert len(messages["items"]) == 2

    cleared = client.post(f"/api/sessions/{session_id}/clear").json()
    assert cleared["removed"] == 2

    sessions = client.get("/api/sessions").json()
    assert any(item["id"] == session_id for item in sessions["items"])

    assert client.delete(f"/api/sessions/{session_id}").json()["ok"] is True
    assert client.get(f"/api/sessions/{session_id}/messages").status_code == 404


def test_stats_endpoint(client):
    payload = client.get("/api/stats").json()
    for key in ("documents", "chunks", "qa_count", "sessions", "memories", "intent_distribution"):
        assert key in payload
    assert payload["documents"] >= 1


def test_memories_endpoint(client):
    payload = client.get("/api/memories").json()
    assert payload["user_id"] == "local-user"
    assert isinstance(payload["items"], list)


def test_citation_context_endpoint(client):
    documents = client.get("/api/documents").json()["items"]
    document = next(item for item in documents if item["chunk_count"] > 0)
    chunks = client.get(f"/api/documents/{document['id']}/chunks?limit=1").json()
    chunk_id = chunks["items"][0]["id"]
    context = client.get(f"/api/documents/{document['id']}/chunks/{chunk_id}/context").json()
    assert context["target"]["id"] == chunk_id
    assert any(item["is_target"] for item in context["window"])


def test_document_reindex_and_delete(client):
    created = client.post(
        "/api/documents/text", json={"text": "临时文档内容：用于验证删除与重新解析接口的行为。", "name": "临时文档"}
    ).json()
    status = _wait_task(client, created["task_id"])
    assert status["status"] == "succeeded"
    document_id = status["document_id"]

    reindex = client.post(f"/api/documents/{document_id}/reindex").json()
    assert _wait_task(client, reindex["task_id"])["status"] == "succeeded"

    assert client.delete(f"/api/documents/{document_id}").json()["ok"] is True
    assert client.get(f"/api/documents/{document_id}").status_code == 404
