"""API_TOKEN 鉴权开关测试。"""

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings, reset_settings_cache
from app.main import app


@pytest.fixture
def token_client(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret-token")
    reset_settings_cache()
    assert get_settings().auth_enabled is True
    with TestClient(app) as client:
        yield client
    reset_settings_cache()


@pytest.fixture
def plain_client():
    with TestClient(app) as client:
        yield client


def test_health_stays_public(token_client):
    assert token_client.get("/api/health").status_code == 200


@pytest.mark.parametrize(
    "path",
    ["/api/sessions", "/api/documents", "/api/memories", "/api/stats"],
)
def test_protected_endpoints_require_token(token_client, path):
    assert token_client.get(path).status_code == 401


def test_bearer_token_accepted(token_client):
    response = token_client.get("/api/sessions", headers={"Authorization": "Bearer s3cret-token"})
    assert response.status_code == 200


def test_x_api_token_header_accepted(token_client):
    response = token_client.get("/api/sessions", headers={"X-API-Token": "s3cret-token"})
    assert response.status_code == 200


def test_wrong_token_rejected(token_client):
    response = token_client.get("/api/sessions", headers={"X-API-Token": "nope"})
    assert response.status_code == 401
    assert "鉴权失败" in response.json()["detail"]


def test_write_endpoints_require_token(token_client):
    assert token_client.post("/api/chat", json={"question": "你好"}).status_code == 401


def test_auth_disabled_without_token(plain_client):
    """未配置 API_TOKEN 时保持零配置可用。"""
    assert plain_client.get("/api/sessions").status_code == 200
