"""API 请求 / 响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(default=None, description="会话 ID，留空则自动新建")
    user_id: str | None = Field(default=None, description="用户标识，默认 local-user")
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索片段数量")
    document_ids: list[str] | None = Field(
        default=None,
        max_length=50,
        description="限定检索范围：只在指定的文档内检索，留空则全库检索",
    )


class SessionCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=100)
    user_id: str | None = None


class UrlIngestRequest(BaseModel):
    url: str = Field(..., min_length=5, description="网页链接")
    name: str | None = Field(default=None, max_length=200)


class TextIngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    name: str = Field(default="手动输入文本", max_length=200)


class HealthResponse(BaseModel):
    status: str
    version: str
    config: dict[str, Any]
    components: dict[str, Any]


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, description="会话 ID")
    rating: str = Field(..., pattern="^(up|down)$", description="up=有帮助，down=没帮助")
